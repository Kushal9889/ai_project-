"""
Phone-specific scoring wrapper that builds on core signals (fake, sentiment, summarizer)
and the phone adapter to compute per-aspect scores and final Trust Score (0-100).
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .adapters.phone_adapter import get_adapter, detect_overheat_with_charging, detect_conditional_praise
from math import log1p

def _normalize_compound_to_0_100(compound: float) -> float:
    """compound in [-1,1] => map to [0,100]"""
    return float(max(0.0, min(100.0, ((compound + 1.0) / 2.0) * 100.0)))

def _aspect_base_score(compound_values: List[float], strengths: List[float]) -> float:
    """Compute base aspect score using mean compound and strength bonus."""
    if not compound_values:
        return 50.0  # neutral default
    mean_compound = float(np.mean(compound_values))
    base = _normalize_compound_to_0_100(mean_compound)
    # intensity bonus (strength 0..1) scaled to 0..10
    avg_strength = float(np.mean(strengths)) if strengths else 0.0
    bonus = avg_strength * 10.0
    return float(min(100.0, base * 0.9 + bonus * 0.1))

def compute_aspect_scores(
    df: pd.DataFrame,
    adapter_cfg: Dict[str, Any],
    text_col: str = "reviewText",
    sentiment_col: str = "compound",
    strength_col: str = "strength",
    min_mentions_map: Dict[str, int] = None
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Return tuple (aspect_scores, aspect_counts).
    Operates only on genuine reviews (df['is_fake']==False) if column exists.
    """
    if min_mentions_map is None:
        min_mentions_map = adapter_cfg.get("min_mentions", {})
    
    # filter genuine
    if "is_fake" in df.columns:
        genuine_df = df[~df["is_fake"]].reset_index(drop=True)
    else:
        genuine_df = df.copy().reset_index(drop=True)

    # extract sentences mapped to reviews
    from .aspect_extractor import map_sentences_to_reviews
    review_sentence_pairs = map_sentences_to_reviews(genuine_df, text_col=text_col)  # list of (review_idx, sentence)
    aspects = []
    sentences = []
    review_indices = []
    from .adapters.phone_adapter import find_aspect_for_sentence
    for ridx, sent in review_sentence_pairs:
        asp = find_aspect_for_sentence(sent)
        aspects.append(asp)
        sentences.append(sent)
        review_indices.append(ridx)

    # prepare per-aspect lists
    aspect_compounds = {k: [] for k in adapter_cfg["aspect_weights"].keys()}
    aspect_strengths = {k: [] for k in adapter_cfg["aspect_weights"].keys()}
    aspect_counts = {k: 0 for k in adapter_cfg["aspect_weights"].keys()}

    # when sentence-level sentiment isn't available, fall back to review-level
    for (ridx, sent), asp in zip(review_sentence_pairs, aspects):
        if asp not in aspect_compounds:
            continue
        # fetch review-level sentiment metrics
        try:
            compound = float(genuine_df.loc[ridx, sentiment_col]) if sentiment_col in genuine_df.columns else 0.0
            strength = float(genuine_df.loc[ridx, strength_col]) if strength_col in genuine_df.columns else 0.0
        except Exception:
            compound = 0.0
            strength = 0.0
        aspect_compounds[asp].append(compound)
        aspect_strengths[asp].append(strength)
        aspect_counts[asp] += 1

    # compute aspect scores
    aspect_scores = {}
    for asp, weight in adapter_cfg["aspect_weights"].items():
        compounds = aspect_compounds.get(asp, [])
        strengths = aspect_strengths.get(asp, [])
        count = aspect_counts.get(asp, 0)
        base = _aspect_base_score(compounds, strengths)
        min_req = min_mentions_map.get(asp, 3)
        
        # If mentions are fewer than the adapter's minimum required, treat as insufficient data
        # This allows the final score calculation to ignore this aspect entirely
        if count < min_req:
            aspect_scores[asp] = None
            continue
            
        # otherwise compute score normally (without artificial penalty for low count)
        score = (base * 0.9 + (np.mean(strengths) * 10.0 if strengths else 0.0) * 0.1)
        aspect_scores[asp] = float(round(score, 2))

    return aspect_scores, aspect_counts

def apply_overheat_penalty(aspect_scores: Dict[str, float], overheating_count: int, overheat_threshold: int = 2, multiplier: float = 0.7) -> Dict[str, float]:
    """If overheating_count >= threshold, penalize battery and performance."""
    if overheating_count >= overheat_threshold:
        for a in ["battery", "performance"]:
            if a in aspect_scores and aspect_scores[a] is not None:
                aspect_scores[a] = round(aspect_scores[a] * multiplier, 2)
    return aspect_scores

def compute_trust_score(
    df: pd.DataFrame,
    adapter_cfg: Dict[str, Any],
    rating_col: str = "rating",
    sentiment_col: str = "compound",
    strength_col: str = "strength"
) -> Dict[str, Any]:
    """
    End-to-end phone trust score computation based purely on weighted Aspect Quality.
    Missing aspects are ignored and weights are re-normalized.
    """
    n_reviews = len(df)
    
    # Calculate stats for reporting (fake percent, genuine count)
    genuine_mask = ~df.get("is_fake", pd.Series([False]*len(df)))
    genuine_count = int(genuine_mask.sum())
    fake_percent = float(round(100.0 * (1.0 - (genuine_count / max(1, n_reviews))), 2)) if n_reviews > 0 else 0.0

    # 1. Compute Aspect Scores (using only genuine reviews inside the function)
    adapter_bundle = {
        "aspect_weights": adapter_cfg["aspect_weights"],
        "min_mentions": adapter_cfg.get("min_mentions", {})
    }
    aspect_scores, aspect_counts = compute_aspect_scores(df, adapter_bundle, sentiment_col=sentiment_col, strength_col=strength_col)

    # 2. Apply Overheat Penalty
    text_col = "reviewText"
    if text_col in df.columns:
        texts = df[text_col].fillna("").astype(str).tolist()
    else:
        fallback = df.get("reviewText", [])
        texts = list(fallback) if fallback is not None else []
    
    # Only detect overheating in genuine texts to avoid penalizing based on spam
    genuine_texts = [t for t, is_fake in zip(texts, ~genuine_mask) if not is_fake]
    overheat_count = detect_overheat_with_charging(genuine_texts)
    aspect_scores = apply_overheat_penalty(aspect_scores, overheat_count, overheat_threshold=2, multiplier=adapter_cfg.get("overheat_multiplier", 0.7))

    # 3. Calculate Final Score (Aspect Composite)
    # This logic ignores missing attributes and re-normalizes weights as requested
    weights = adapter_cfg["aspect_weights"]
    available_aspects = [k for k, v in aspect_scores.items() if v is not None]
    
    if not available_aspects:
        # Fallback if no aspects were found at all
        final_score = 50.0
        norm_weights = {}
    else:
        # Filter weights to only available aspects
        norm_weights = {k: float(weights.get(k, 0.0)) for k in available_aspects}
        total_w = sum(norm_weights.values())
        
        if total_w == 0:
            final_score = 50.0 
        else:
            # Re-normalize so they sum to 1.0
            norm_weights = {k: v/total_w for k, v in norm_weights.items()}
            
            final_score = 0.0
            for k, w in norm_weights.items():
                final_score += aspect_scores.get(k, 50.0) * w
    
    final_score = float(max(0.0, min(100.0, round(final_score, 2))))

    # 4. Legacy Components (for UI compatibility only, set to 0 or purely informational)
    if rating_col in df.columns and df[rating_col].dropna().shape[0] > 0:
        avg_rating = float(df[rating_col].dropna().mean())
    else:
        avg_rating = 3.0
    rating_component = (avg_rating / 5.0) * 50.0 # Informational only
    genuineness_component = (genuine_count / max(1, n_reviews)) * 20.0 # Informational only
    
    # Confidence estimate: based on number of genuine reviews and aspect coverage
    avg_strength = float(df.get(strength_col, pd.Series([0.0]*len(df))).mean()) if len(df)>0 else 0.0
    aspect_coverage_ratio = len(available_aspects) / len(weights) if weights else 0.0
    confidence = min(1.0, (genuine_count / max(1.0, max(10, n_reviews))) * 0.5 + aspect_coverage_ratio * 0.3 + avg_strength * 0.2)

    # assemble JSON output
    product_summary = {
        "n_reviews": n_reviews,
        "n_genuine": genuine_count,
        "fake_percent": fake_percent,
        
        # Legacy components kept for UI compatibility, but effectively removed from final calculation
        "rating_component": 0.0, 
        "genuineness_component": 0.0,
        "aspect_composite_raw": final_score,
        "scale_aspect_component": final_score, # The component IS the score now
        "volume_recency_component": 0.0,
        
        "final_score": final_score,
        "confidence": float(round(confidence, 3)),
        
        "aspect_scores": aspect_scores,
        "aspect_counts": aspect_counts,
        "aspect_weights_effective": norm_weights,
        "aspect_contributions": {k: round(aspect_scores.get(k, 50.0) * w, 2) for k, w in norm_weights.items()},
        
        "overheat_events_detected": overheat_count,
        "overheat_penalty_applied": overheat_count >= 2
    }
    return product_summary
"""
Phone-specific scoring wrapper that builds on core signals (fake, sentiment, summarizer)
and the phone adapter to compute per-aspect scores and final Trust Score (0-100).
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .adapters.phone_adapter import get_adapter, detect_overheat_with_charging, detect_conditional_praise
from math import log1p

def _normalize_compound_to_0_100(compound: float) -> float:
    """compound in [-1,1] => map to [0,100]"""
    return float(max(0.0, min(100.0, ((compound + 1.0) / 2.0) * 100.0)))

def _aspect_base_score(compound_values: List[float], strengths: List[float]) -> float:
    """Compute base aspect score using mean compound and strength bonus."""
    if not compound_values:
        return 50.0  # neutral default
    mean_compound = float(np.mean(compound_values))
    base = _normalize_compound_to_0_100(mean_compound)
    # intensity bonus (strength 0..1) scaled to 0..10
    avg_strength = float(np.mean(strengths)) if strengths else 0.0
    bonus = avg_strength * 10.0
    return float(min(100.0, base * 0.9 + bonus * 0.1))

def compute_aspect_scores(
    df: pd.DataFrame,
    adapter_cfg: Dict[str, Any],
    text_col: str = "reviewText",
    sentiment_col: str = "compound",
    strength_col: str = "strength",
    min_mentions_map: Dict[str, int] = None
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Return tuple (aspect_scores, aspect_counts).
    Operates only on genuine reviews (df['is_fake']==False) if column exists.
    """
    if min_mentions_map is None:
        min_mentions_map = adapter_cfg.get("min_mentions", {})
    
    # filter genuine
    if "is_fake" in df.columns:
        genuine_df = df[~df["is_fake"]].reset_index(drop=True)
    else:
        genuine_df = df.copy().reset_index(drop=True)

    # extract sentences mapped to reviews
    from .aspect_extractor import map_sentences_to_reviews
    review_sentence_pairs = map_sentences_to_reviews(genuine_df, text_col=text_col)  # list of (review_idx, sentence)
    aspects = []
    sentences = []
    review_indices = []
    from .adapters.phone_adapter import find_aspect_for_sentence
    for ridx, sent in review_sentence_pairs:
        asp = find_aspect_for_sentence(sent)
        aspects.append(asp)
        sentences.append(sent)
        review_indices.append(ridx)

    # prepare per-aspect lists
    aspect_compounds = {k: [] for k in adapter_cfg["aspect_weights"].keys()}
    aspect_strengths = {k: [] for k in adapter_cfg["aspect_weights"].keys()}
    aspect_counts = {k: 0 for k in adapter_cfg["aspect_weights"].keys()}

    # when sentence-level sentiment isn't available, fall back to review-level
    for (ridx, sent), asp in zip(review_sentence_pairs, aspects):
        if asp not in aspect_compounds:
            continue
        # fetch review-level sentiment metrics
        try:
            compound = float(genuine_df.loc[ridx, sentiment_col]) if sentiment_col in genuine_df.columns else 0.0
            strength = float(genuine_df.loc[ridx, strength_col]) if strength_col in genuine_df.columns else 0.0
        except Exception:
            compound = 0.0
            strength = 0.0
        aspect_compounds[asp].append(compound)
        aspect_strengths[asp].append(strength)
        aspect_counts[asp] += 1

    # compute aspect scores
    aspect_scores = {}
    for asp, weight in adapter_cfg["aspect_weights"].items():
        compounds = aspect_compounds.get(asp, [])
        strengths = aspect_strengths.get(asp, [])
        count = aspect_counts.get(asp, 0)
        base = _aspect_base_score(compounds, strengths)
        min_req = min_mentions_map.get(asp, 3)
        
        # If mentions are fewer than the adapter's minimum required, treat as insufficient data
        # This allows the final score calculation to ignore this aspect entirely
        if count < min_req:
            aspect_scores[asp] = None
            continue
            
        # otherwise compute score normally (without artificial penalty for low count)
        score = (base * 0.9 + (np.mean(strengths) * 10.0 if strengths else 0.0) * 0.1)
        aspect_scores[asp] = float(round(score, 2))

    return aspect_scores, aspect_counts

def apply_overheat_penalty(aspect_scores: Dict[str, float], overheating_count: int, overheat_threshold: int = 2, multiplier: float = 0.7) -> Dict[str, float]:
    """If overheating_count >= threshold, penalize battery and performance."""
    if overheating_count >= overheat_threshold:
        for a in ["battery", "performance"]:
            if a in aspect_scores and aspect_scores[a] is not None:
                aspect_scores[a] = round(aspect_scores[a] * multiplier, 2)
    return aspect_scores

def compute_trust_score(
    df: pd.DataFrame,
    adapter_cfg: Dict[str, Any],
    rating_col: str = "rating",
    sentiment_col: str = "compound",
    strength_col: str = "strength"
) -> Dict[str, Any]:
    """
    End-to-end phone trust score computation based purely on weighted Aspect Quality.
    Missing aspects are ignored and weights are re-normalized.
    """
    n_reviews = len(df)
    
    # Calculate stats for reporting (fake percent, genuine count)
    genuine_mask = ~df.get("is_fake", pd.Series([False]*len(df)))
    genuine_count = int(genuine_mask.sum())
    fake_percent = float(round(100.0 * (1.0 - (genuine_count / max(1, n_reviews))), 2)) if n_reviews > 0 else 0.0

    # 1. Compute Aspect Scores (using only genuine reviews inside the function)
    adapter_bundle = {
        "aspect_weights": adapter_cfg["aspect_weights"],
        "min_mentions": adapter_cfg.get("min_mentions", {})
    }
    aspect_scores, aspect_counts = compute_aspect_scores(df, adapter_bundle, sentiment_col=sentiment_col, strength_col=strength_col)

    # 2. Apply Overheat Penalty
    text_col = "reviewText"
    if text_col in df.columns:
        texts = df[text_col].fillna("").astype(str).tolist()
    else:
        fallback = df.get("reviewText", [])
        texts = list(fallback) if fallback is not None else []
    
    # Only detect overheating in genuine texts to avoid penalizing based on spam
    genuine_texts = [t for t, is_fake in zip(texts, ~genuine_mask) if not is_fake]
    overheat_count = detect_overheat_with_charging(genuine_texts)
    aspect_scores = apply_overheat_penalty(aspect_scores, overheat_count, overheat_threshold=2, multiplier=adapter_cfg.get("overheat_multiplier", 0.7))

    # 3. Calculate Final Score (Aspect Composite)
    # This logic ignores missing attributes and re-normalizes weights as requested
    weights = adapter_cfg["aspect_weights"]
    available_aspects = [k for k, v in aspect_scores.items() if v is not None]
    
    if not available_aspects:
        # Fallback if no aspects were found at all
        final_score = 50.0
        norm_weights = {}
    else:
        # Filter weights to only available aspects
        norm_weights = {k: float(weights.get(k, 0.0)) for k in available_aspects}
        total_w = sum(norm_weights.values())
        
        if total_w == 0:
            final_score = 50.0 
        else:
            # Re-normalize so they sum to 1.0
            norm_weights = {k: v/total_w for k, v in norm_weights.items()}
            
            final_score = 0.0
            for k, w in norm_weights.items():
                final_score += aspect_scores.get(k, 50.0) * w
    
    final_score = float(max(0.0, min(100.0, round(final_score, 2))))

    # 4. Legacy Components (for UI compatibility only, set to 0 or purely informational)
    if rating_col in df.columns and df[rating_col].dropna().shape[0] > 0:
        avg_rating = float(df[rating_col].dropna().mean())
    else:
        avg_rating = 3.0
    rating_component = (avg_rating / 5.0) * 50.0 # Informational only
    genuineness_component = (genuine_count / max(1, n_reviews)) * 20.0 # Informational only
    
    # Confidence estimate: based on number of genuine reviews and aspect coverage
    avg_strength = float(df.get(strength_col, pd.Series([0.0]*len(df))).mean()) if len(df)>0 else 0.0
    aspect_coverage_ratio = len(available_aspects) / len(weights) if weights else 0.0
    confidence = min(1.0, (genuine_count / max(1.0, max(10, n_reviews))) * 0.5 + aspect_coverage_ratio * 0.3 + avg_strength * 0.2)

    # assemble JSON output
    product_summary = {
        "n_reviews": n_reviews,
        "n_genuine": genuine_count,
        "fake_percent": fake_percent,
        
        # Legacy components kept for UI compatibility, but effectively removed from final calculation
        "rating_component": 0.0, 
        "genuineness_component": 0.0,
        "aspect_composite_raw": final_score,
        "scale_aspect_component": final_score, # The component IS the score now
        "volume_recency_component": 0.0,
        
        "final_score": final_score,
        "confidence": float(round(confidence, 3)),
        
        "aspect_scores": aspect_scores,
        "aspect_counts": aspect_counts,
        "aspect_weights_effective": norm_weights,
        "aspect_contributions": {k: round(aspect_scores.get(k, 50.0) * w, 2) for k, w in norm_weights.items()},
        
        "overheat_events_detected": overheat_count,
        "overheat_penalty_applied": overheat_count >= 2
    }
    return product_summary