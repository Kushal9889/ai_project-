"""
Module 3: Human-like summarizer for genuine reviews.
Uses 'Split -> Bucket -> Summarize' workflow for strict aspect separation.
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from .utils_text import split_compound_sentences
from .api_clients import HFClient, MockHFClient
from .config import DEFAULT_CONFIG
from .adapters.phone_adapter import find_aspect_for_sentence, ASPECT_KEYWORDS

try:
    from .ollama_client import OllamaClient
except ImportError:
    OllamaClient = None

CFG = DEFAULT_CONFIG["summarizer"]
EMB_MODEL = DEFAULT_CONFIG["embedding_model"]

def _extract_and_split_sentences(df_or_texts, text_col: str = "reviewText") -> List[str]:
    """Extract texts and aggressively split them into atomic aspect-specific chunks."""
    texts = []
    if isinstance(df_or_texts, pd.DataFrame):
        if text_col in df_or_texts.columns:
            texts = df_or_texts[text_col].fillna("").astype(str).tolist()
    else:
        try:
            texts = list(df_or_texts)
        except:
            texts = []

    all_segments = []
    for text in texts:
        # Use the smart splitter that breaks on 'but', 'however', etc.
        segments = split_compound_sentences(str(text))
        all_segments.extend(segments)
    return all_segments

def _embed_sentences(sentences: List[str], model_name: str = EMB_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        import hashlib
        dim = 384
        vecs = []
        for t in sentences:
            h = hashlib.sha1((t or "").encode("utf-8")).digest()
            base = np.frombuffer(h, dtype=np.uint8).astype(float)
            v = np.tile(base, int(np.ceil(dim / base.size)))[:dim]
            v = v / (v.max() + 1e-9)
            vecs.append(v)
        return np.vstack(vecs)

def _cluster_sentences(embeddings: np.ndarray, threshold: float = 0.65):
    if len(embeddings) < 2: return [0] * len(embeddings)
    kwargs = {"n_clusters": None, "distance_threshold": 1 - threshold, "linkage": "average", "metric": "cosine"}
    try:
        clustering = AgglomerativeClustering(**kwargs)
    except TypeError:
        kwargs.pop("metric"); kwargs["affinity"] = "cosine"
        clustering = AgglomerativeClustering(**kwargs)
    return clustering.fit_predict(embeddings)

def _top_k_representatives(sentences: List[str], emb: np.ndarray, labels: List[int], k: int = 3) -> List[str]:
    all_reps = []
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        if not idxs: continue
        cent = np.mean(emb[idxs], axis=0)
        sims = np.dot(emb[idxs], cent) / (np.linalg.norm(emb[idxs], axis=1) * np.linalg.norm(cent) + 1e-8)
        order = np.argsort(-sims)
        for i in order[:k]:
            all_reps.append(sentences[idxs[i]])
    return all_reps

def summarize_reviews(
    df: pd.DataFrame, 
    text_col: str = "reviewText", 
    hf_client: Optional[HFClient] = None, 
    mock_mode: bool = True,
    provider: str = "mock",
    ollama_model: str = "gemma3:1b"
) -> Dict[str, Any]:
    
    # 1. SPLIT: Break reviews into atomic thought segments
    segments = _extract_and_split_sentences(df, text_col)
    
    if not segments:
        return {"positive_summary": "", "negative_summary": "", "balanced_summary": "", "aspect_breakdown": {}, "insights": {}}

    # 2. BUCKET: Sort segments into aspects based on keyword density
    aspect_buckets = defaultdict(list)
    for seg in segments:
        asp = find_aspect_for_sentence(seg)
        if asp != "misc":
            aspect_buckets[asp].append(seg)

    # Init Ollama
    ollama_client = None
    if provider == "ollama" and OllamaClient:
        try:
            ollama_client = OllamaClient()
        except:
            pass

    # 3. SUMMARIZE: Process each bucket
    aspect_breakdown = {}
    
    for asp, sents in aspect_buckets.items():
        if len(sents) < 2: continue
        
        if ollama_client:
            # Smart Mode: Generate clean paragraph
            summary = ollama_client.generate_summary(ollama_model, sents, topic=asp)
            if summary:
                aspect_breakdown[asp] = summary
            else:
                aspect_breakdown[asp] = " ".join(sents[:3])
        else:
            # Mock Mode: Cluster and pick representatives
            # Only cluster if enough data, else just join unique ones
            unique_sents = sorted(list(set(sents)), key=len, reverse=True) # Prefer longer sentences
            if len(unique_sents) > 5:
                emb = _embed_sentences(unique_sents)
                labels = _cluster_sentences(emb)
                reps = _top_k_representatives(unique_sents, emb, labels, k=1)
                aspect_breakdown[asp] = " ".join(reps[:3])
            else:
                aspect_breakdown[asp] = " ".join(unique_sents[:3])

    # 4. GLOBAL SUMMARY
    pos_pts = []
    neg_pts = []
    
    # Simple check on the final summaries
    for txt in aspect_breakdown.values():
        t_low = txt.lower()
        if any(w in t_low for w in ["good", "great", "excellent", "love", "best"]):
            pos_pts.append(txt)
        if any(w in t_low for w in ["bad", "poor", "terrible", "issue", "problem"]):
            neg_pts.append(txt)
            
    pos_sum = " ".join(pos_pts[:2]) or "No major strengths detected."
    neg_sum = " ".join(neg_pts[:2]) or "No major weaknesses detected."
    balanced = f"**Pros:** {pos_sum}\n\n**Cons:** {neg_sum}"

    insights = {"hidden_issues": "", "emotional_signals": "", "underlying_patterns": ""}
    if any("overheat" in s.lower() for s in segments):
        insights["hidden_issues"] += "Overheating reported by users. "

    return {
        "positive_summary": pos_sum,
        "negative_summary": neg_sum,
        "balanced_summary": balanced,
        "aspect_breakdown": aspect_breakdown,
        "insights": insights
    }