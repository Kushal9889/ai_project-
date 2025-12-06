"""
Module 1: Fake Review Detection (mock-friendly).
Updated to support Ollama fake detection.
"""
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .api_clients import HFClient, MockHFClient
# Try to import OllamaClient
try:
    from .ollama_client import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None

from .utils_text import shannon_entropy, flesch_kincaid, tokenize_words
from .config import DEFAULT_CONFIG

DEFAULTS = DEFAULT_CONFIG["fake_detection"]

def heuristics_score(text: str, rating: Optional[float] = None, defaults: Dict[str, Any] = DEFAULTS) -> float:
    score = 0.0
    words = tokenize_words(text)
    wc = len(words)
    if wc < defaults.get("min_words_short", 12):
        score += 0.25
    lower = (text or "").lower()
    if any(p in lower for p in ["amazing product", "best product ever", "highly recommend", "five stars"]):
        score += 0.2
    vocab_div = len(set(words)) / max(1, wc) if wc > 0 else 0.0
    if vocab_div < defaults.get("vocab_diversity_min", 0.2):
        score += 0.15
    return min(1.0, score)

def _hf_fake_score(text: str, hf_client: HFClient, model: str) -> float:
    try:
        out = hf_client.classify(model, [text])
    except Exception:
        return 0.2
    if isinstance(out, list) and len(out) > 0:
        top = out[0][0]
        label = (top.get("label") or "").lower()
        score = float(top.get("score", 0.0))
        if "fake" in label or "spam" in label:
            return score
        if "real" in label or "genuine" in label:
            return 1.0 - score
        return 1.0 - score
    return 0.2

# Cache for sentence transformer model to avoid reloading
_SENTENCE_TRANSFORMER_MODEL = None

def _compute_embeddings(texts: List[str], model_name: str = DEFAULT_CONFIG["embedding_model"], use_transformer: bool = True) -> np.ndarray:
    if not use_transformer:
        import hashlib
        dim = 384
        vecs = []
        for t in texts:
            h = hashlib.md5((t or "").encode("utf-8")).digest()
            base = np.frombuffer(h, dtype=np.uint8).astype(float)
            v = np.tile(base, int(np.ceil(dim / base.size)))[:dim]
            v = v / (v.max() + 1e-9)
            vecs.append(v)
        return np.vstack(vecs)
    
    try:
        global _SENTENCE_TRANSFORMER_MODEL
        from sentence_transformers import SentenceTransformer  # type: ignore
        if _SENTENCE_TRANSFORMER_MODEL is None:
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(model_name)
        return _SENTENCE_TRANSFORMER_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        import hashlib
        dim = 384
        vecs = []
        for t in texts:
            h = hashlib.md5((t or "").encode("utf-8")).digest()
            base = np.frombuffer(h, dtype=np.uint8).astype(float)
            v = np.tile(base, int(np.ceil(dim / base.size)))[:dim]
            v = v / (v.max() + 1e-9)
            vecs.append(v)
        return np.vstack(vecs)

def _find_similarity_clusters(embeddings: np.ndarray, sim_threshold: float, min_cluster_size: int):
    n = embeddings.shape[0]
    if n == 0:
        return [False] * n, []
    sim = cosine_similarity(embeddings)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        queue = [i]
        comp = []
        visited[i] = True
        while queue:
            u = queue.pop()
            comp.append(u)
            neighbors = np.where(sim[u] > sim_threshold)[0].tolist()
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        clusters.append(comp)
    suspicious_mask = [False] * n
    suspicious_clusters = []
    for comp in clusters:
        if len(comp) >= min_cluster_size:
            suspicious_clusters.append(comp)
            for idx in comp:
                suspicious_mask[idx] = True
    return suspicious_mask, suspicious_clusters

def detect_fake_reviews(
    df: pd.DataFrame,
    text_col: str = "reviewText",
    rating_col: Optional[str] = "rating",
    hf_client: Optional[HFClient] = None,
    mock_mode: bool = True,
    sim_threshold: float = DEFAULTS["sim_threshold"],
    min_cluster_size: int = DEFAULTS["cluster_size_min"],
    entropy_min: float = DEFAULTS["min_entropy"],
    heuristics_weight: float = DEFAULTS["heuristics_weight"],
    hf_model: str = DEFAULTS["hf_model"],  # <--- THIS IS THE MISSING ARGUMENT
    provider: str = "mock",
    ollama_model: str = "gemma3:1b"
) -> pd.DataFrame:
    
    # Init Ollama if needed
    ollama_client = None
    if provider == "ollama":
        if OLLAMA_AVAILABLE:
            try:
                ollama_client = OllamaClient()
            except Exception:
                pass
    
    hf_client = hf_client or (MockHFClient() if mock_mode else HFClient())
    df = df.copy().reset_index(drop=True)
    texts = df[text_col].fillna("").astype(str).tolist()

    # Probability scores (0.0 = Real, 1.0 = Fake)
    fake_probs = []
    
    if provider == "ollama" and ollama_client:
        # Use Ollama
        for t in texts:
            try:
                out = ollama_client.classify_fake(ollama_model, t).lower()
                if "fake" in out:
                    fake_probs.append(0.9)
                elif "real" in out:
                    fake_probs.append(0.1)
                else:
                    fake_probs.append(0.3) # Unsure
            except Exception:
                fake_probs.append(0.2)
    else:
        # Use HF or Mock
        for t in texts:
            # This line was failing because hf_model wasn't defined in the args
            hf_p = _hf_fake_score(t, hf_client, hf_model)
            fake_probs.append(hf_p)

    heur_scores = []
    entropies = []
    fks = []
    for i, t in enumerate(texts):
        heur = heuristics_score(t, rating=df[rating_col].iloc[i] if rating_col in df.columns else None)
        heur_scores.append(heur)
        entropies.append(shannon_entropy(t))
        fks.append(flesch_kincaid(t))

    # Embeddings & clusters
    try:
        # Don't use heavy transformers in mock/ollama mode for speed, unless specified
        use_trans = (not mock_mode) and (provider == "huggingface")
        embeddings = _compute_embeddings(texts, use_transformer=use_trans)
        sim_mask, clusters = _find_similarity_clusters(embeddings, sim_threshold, min_cluster_size)
    except Exception:
        sim_mask = [False] * len(texts)
        clusters = []

    combined = []
    for i in range(len(texts)):
        p_fake = float(fake_probs[i])
        heur = float(heur_scores[i])
        ent = float(entropies[i])
        sim_flag = 1.0 if sim_mask[i] else 0.0
        ent_pen = 0.0 if ent >= entropy_min else (1.0 - (ent / (entropy_min + 1e-6)))
        
        # Weighted combination
        p = (1 - heuristics_weight) * p_fake + heuristics_weight * heur + 0.25 * sim_flag + 0.5 * ent_pen
        p = max(0.0, min(1.0, p))
        combined.append(round(p, 4))

    df["hf_fake_score"] = combined
    df["heuristics_score"] = heur_scores
    df["entropy"] = entropies
    df["flesch_kincaid"] = fks
    df["similarity_cluster"] = sim_mask
    df["is_fake"] = [(p > 0.55) or (sim and (hs > 0.35)) for p, sim, hs in zip(df["hf_fake_score"], df["similarity_cluster"], df["heuristics_score"])]
    return df