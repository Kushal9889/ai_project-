"""
Module 2: Sentiment + Emotion engine.
Updated to handle 7-point granular sentiment scale from Ollama.
Produces per-review sentiment_pos, sentiment_neg, compound (pos-neg), emotion_scores, strength, sentiment_label.
"""
from typing import Optional, Dict, Any, Literal
import numpy as np
import pandas as pd
from .api_clients import HFClient, MockHFClient
import requests
import logging

# Try to import OllamaClient; if it fails the user hasn't added it yet
try:
    from .ollama_client import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None

# Track models that have been reported unavailable (HTTP 410) to avoid repeated noisy logs
HF_MODELS_UNAVAILABLE = set()
# Last used (resolved) sentiment/emotion model names when fallback occurs
LAST_USED_SENTIMENT_MODEL = None
LAST_USED_EMOTION_MODEL = None
from .utils_text import shannon_entropy, has_linguistic_intensity
from .config import DEFAULT_CONFIG

CFG = DEFAULT_CONFIG["sentiment"]


def _call_sentiment_model(texts, hf_client: HFClient, model: str):
    global LAST_USED_SENTIMENT_MODEL
    try:
        resp = hf_client.classify(model, texts)
        LAST_USED_SENTIMENT_MODEL = model
    except requests.exceptions.HTTPError as e:
        # If model endpoint removed (HTTP 410) try fallbacks listed in config
        try:
            status = e.response.status_code if getattr(e, 'response', None) is not None else None
        except Exception:
            status = None
        if status == 410:
            if model not in HF_MODELS_UNAVAILABLE:
                logging.warning("HF classify HTTPError for model %s: %s — model unavailable (410). Attempting fallbacks.", model, e)
                HF_MODELS_UNAVAILABLE.add(model)
            resp = None
            for alt in CFG.get("fallback_models", []):
                if alt in HF_MODELS_UNAVAILABLE:
                    continue
                try:
                    resp = hf_client.classify(alt, texts)
                    LAST_USED_SENTIMENT_MODEL = alt
                    logging.info("HF classify: using fallback model %s", alt)
                    break
                except requests.exceptions.HTTPError as ee:
                    try:
                        status2 = ee.response.status_code if getattr(ee, 'response', None) is not None else None
                    except Exception:
                        status2 = None
                    if status2 == 410:
                        HF_MODELS_UNAVAILABLE.add(alt)
                        continue
                    else:
                        logging.warning("HF classify HTTPError for fallback model %s: %s", alt, ee)
                        continue
                except Exception as ee:
                    logging.warning("Error trying fallback model %s: %s", alt, ee)
                    continue
            if resp is None:
                logging.warning("No HF fallback models succeeded; using MockHFClient for sentiment.")
                resp = MockHFClient().classify(model, texts)
        else:
            logging.warning("HF classify HTTPError for model %s: %s — falling back to MockHFClient", model, e)
            resp = MockHFClient().classify(model, texts)
    except requests.exceptions.RequestException as e:
        logging.warning("HF classify RequestException for model %s: %s — falling back to MockHFClient", model, e)
        resp = MockHFClient().classify(model, texts)
    except Exception as e:
        logging.warning("Unexpected error calling HF classify for model %s: %s — falling back to MockHFClient", model, e)
        resp = MockHFClient().classify(model, texts)
    probs = []
    for out in resp:
        # out should be list of dicts
        if isinstance(out, list) and len(out) > 0:
            top = out[0]
            label = (top.get("label") or "").lower()
            score = float(top.get("score", 0.0))
            if "neg" in label:
                probs.append((0.0, score))
            elif "pos" in label:
                probs.append((score, 0.0))
            else:
                probs.append((0.5, 0.5))
        else:
            probs.append((0.5, 0.5))
    return probs


def _call_emotion_model(texts, hf_client: HFClient, model: str):
    global LAST_USED_EMOTION_MODEL
    try:
        resp = hf_client.classify(model, texts)
        LAST_USED_EMOTION_MODEL = model
    except requests.exceptions.HTTPError as e:
        try:
            status = e.response.status_code if getattr(e, 'response', None) is not None else None
        except Exception:
            status = None
        if status == 410:
            if model not in HF_MODELS_UNAVAILABLE:
                logging.warning("HF emotion classify HTTPError for model %s: %s — model unavailable (410). Attempting emotion fallbacks.", model, e)
                HF_MODELS_UNAVAILABLE.add(model)
            resp = None
            for alt in CFG.get("fallback_emotion_models", []):
                if alt in HF_MODELS_UNAVAILABLE:
                    continue
                try:
                    resp = hf_client.classify(alt, texts)
                    LAST_USED_EMOTION_MODEL = alt
                    logging.info("HF emotion classify: using fallback model %s", alt)
                    break
                except requests.exceptions.HTTPError as ee:
                    try:
                        status2 = ee.response.status_code if getattr(ee, 'response', None) is not None else None
                    except Exception:
                        status2 = None
                    if status2 == 410:
                        HF_MODELS_UNAVAILABLE.add(alt)
                        continue
                    else:
                        logging.warning("HF emotion classify HTTPError for fallback model %s: %s", alt, ee)
                        continue
                except Exception as ee:
                    logging.warning("Error trying emotion fallback model %s: %s", alt, ee)
                    continue
            if resp is None:
                logging.warning("No HF emotion fallback models succeeded; using MockHFClient for emotion.")
                resp = MockHFClient().classify(model, texts)
        else:
            logging.warning("HF emotion classify HTTPError for model %s: %s — falling back to MockHFClient", model, e)
            resp = MockHFClient().classify(model, texts)
    except requests.exceptions.RequestException as e:
        logging.warning("HF emotion classify RequestException for model %s: %s — falling back to MockHFClient", model, e)
        resp = MockHFClient().classify(model, texts)
    except Exception as e:
        logging.warning("Unexpected error calling HF emotion classify for model %s: %s — falling back to MockHFClient", model, e)
        resp = MockHFClient().classify(model, texts)
    outlist = []
    for r in resp:
        if isinstance(r, list):
            outlist.append({entry.get("label"): float(entry.get("score")) for entry in r})
        else:
            outlist.append({})
    return outlist


def analyze_sentiments(
    df: pd.DataFrame,
    text_col: str = "reviewText",
    hf_client: Optional[HFClient] = None,
    mock_mode: bool = True,
    config: Optional[Dict[str, Any]] = None,
    provider: Literal["mock", "huggingface", "ollama"] = "mock",
    ollama_model: str = "gemma3:1b"
) -> Dict[str, Any]:
    """
    Analyze sentiments with configurable provider.
    
    Args:
        provider: "mock" (fast/deterministic), "huggingface" (remote API), or "ollama" (local LLM)
        ollama_model: model name for Ollama (e.g., 'gemma3:1b', 'llama2')
    """
    config = config or CFG
    
    # Provider selection logic
    if provider == "ollama":
        if not OLLAMA_AVAILABLE:
            logging.warning("Ollama provider requested but ollama_client not available; falling back to mock.")
            provider = "mock"
        else:
            try:
                ollama_client = OllamaClient()
                logging.info(f"Using Ollama provider with model: {ollama_model}")
            except Exception as e:
                logging.warning(f"Failed to initialize OllamaClient: {e}. Falling back to mock.")
                provider = "mock"
    
    hf_client = hf_client or (MockHFClient() if provider != "huggingface" else HFClient())
    df = df.copy().reset_index(drop=True)
    texts = df[text_col].fillna("").astype(str).tolist()
    
    if len(texts) == 0:
        return {"df": df, "summary": {}}

    # 1. GET SENTIMENT SCORES
    if provider == "ollama":
        # === NEW 7-POINT SCALE LOGIC ===
        sent_preds = []
        for text in texts:
            label = ollama_client.classify_sentiment(ollama_model, text).strip().lower()
            
            # Map labels to (Pos_Prob, Neg_Prob) pairs for compatibility
            # Compound = Pos - Neg
            
            if "extreme positive" in label:
                sent_preds.append((0.975, 0.025)) # Compound ~ +0.95
            elif "extreme negative" in label:
                sent_preds.append((0.025, 0.975)) # Compound ~ -0.95
            
            elif "slightly positive" in label:
                sent_preds.append((0.675, 0.325)) # Compound ~ +0.35
            elif "slightly negative" in label:
                sent_preds.append((0.325, 0.675)) # Compound ~ -0.35
            
            elif "positive" in label:
                sent_preds.append((0.825, 0.175)) # Compound ~ +0.65
            elif "negative" in label:
                sent_preds.append((0.175, 0.825)) # Compound ~ -0.65
            
            else: # Neutral
                sent_preds.append((0.50, 0.50))   # Compound 0.00
                
        # Mock emotion as fallback for Ollama
        emo_client = MockHFClient()
        chosen_emo = config.get("emotion_model")
        
    elif provider == "huggingface":
        # Use candidate_models if configured, otherwise fall back to primary+fallbacks
        candidates = config.get("candidate_models") or ([config.get("primary_model")] + config.get("fallback_models", []))
        try:
            chosen_sent = hf_client.find_working_model([c for c in candidates if c])
        except Exception:
            chosen_sent = None
        if not chosen_sent:
            logging.warning("No working HF sentiment model found from candidates; using MockHFClient for sentiment.")
            sent_client = MockHFClient()
            chosen_sent = config.get("primary_model")
        else:
            sent_client = hf_client

        emo_candidates = [config.get("emotion_model")] + config.get("fallback_emotion_models", [])
        try:
            chosen_emo = hf_client.find_working_model([c for c in emo_candidates if c])
        except Exception:
            chosen_emo = None
        if not chosen_emo:
            logging.warning("No working HF emotion model found from candidates; using MockHFClient for emotion.")
            emo_client = MockHFClient()
            chosen_emo = config.get("emotion_model")
        else:
            emo_client = hf_client
            
        sent_preds = _call_sentiment_model(texts, sent_client, chosen_sent)

    else: # Mock
        sent_client = MockHFClient()
        emo_client = MockHFClient()
        chosen_sent = config.get("primary_model")
        chosen_emo = config.get("emotion_model")
        sent_preds = _call_sentiment_model(texts, sent_client, chosen_sent)

    emotion_preds = _call_emotion_model(texts, emo_client, config.get("emotion_model"))

    # 2. CALCULATE COMPOUND & STRENGTH
    compounds = []
    strengths = []
    labels = []
    emotion_scores_list = []

    for i, t in enumerate(texts):
        pos_prob, neg_prob = sent_preds[i]
        compound = pos_prob - neg_prob # Range -1.0 to 1.0
        
        emotion_dist = emotion_preds[i] if i < len(emotion_preds) else {}
        emotion_scores_list.append(emotion_dist)
        
        # Calculate Strength
        prob_gap = abs(pos_prob - neg_prob)
        lingu = has_linguistic_intensity(t)
        
        # Formula to calculate how "strong" the opinion is
        # Using 0.6*prob_gap to account for the new granular levels
        raw_strength = 0.6 * prob_gap + 0.1 * lingu 
        strength = 1.0 / (1.0 + np.exp(-5 * (raw_strength - 0.2)))
        
        # Assign Label
        if compound >= 0.8: label = "positive_extreme"
        elif compound >= 0.5: label = "positive"
        elif compound >= 0.1: label = "positive_slight"
        elif compound <= -0.8: label = "negative_extreme"
        elif compound <= -0.5: label = "negative"
        elif compound <= -0.1: label = "negative_slight"
        else: label = "neutral"
            
        compounds.append(round(float(compound), 4))
        strengths.append(round(float(strength), 4))
        labels.append(label)

    df["sentiment_pos"], df["sentiment_neg"] = zip(*sent_preds)
    df["compound"] = compounds
    df["strength"] = strengths
    df["sentiment_label"] = labels
    df["emotion_scores"] = emotion_scores_list

    # Summary stats
    genuine_mask = ~df.get("is_fake", pd.Series([False] * len(df)))
    genuine_df = df[genuine_mask] if not genuine_mask.empty else df
    
    summary = {
        "avg_compound": float(genuine_df["compound"].mean()) if len(genuine_df) > 0 else 0.0,
        "compound_variance": float(genuine_df["compound"].var(ddof=0)) if len(genuine_df) > 1 else 0.0,
        "emotion_histogram": {},
        "polarity_spread": genuine_df["sentiment_label"].value_counts().to_dict() if len(genuine_df) > 0 else {}
    }
    
    agg_em = {}
    for d in genuine_df["emotion_scores"].tolist():
        for k, v in (d or {}).items():
            agg_em[k] = agg_em.get(k, 0.0) + v
    if agg_em:
        tot = sum(agg_em.values())
        summary["emotion_histogram"] = {k: v / tot for k, v in agg_em.items()}
    
    return {"df": df, "summary": summary}