"""
Phone adapter: aspect keywords, weights, heuristics and helper utilities for Smartphone domain.
Updated: Refined keywords to prevent cross-contamination (e.g., 'video' removed from camera to avoid matching 'video playback').
"""
from typing import Dict, List, Tuple, Optional
import re

# Refined keywords to be more specific and avoid overlap
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "battery":[
        "battery", "battery life", "charge", "charging", "fast charge", "drain", 
        "sot", "screen on time", "mah", "endurance", "playback", "standby"
    ], 
    "camera": [
        "camera", "photo", "low light", "night mode", "zoom", "selfie", "autofocus", 
        "image", "picture", "video recording", "4k video", "1080p", "lens", "portrait", 
        "shutter", "exposure", "ultrawide", "telephoto", "cinematic"
    ],
    "performance": [
        "performance", "lag", "stutter", "slow", "fast", "fps", "benchmark", 
        "throttle", "overheat", "heating", "thermal", "processor", "chip", "ram", 
        "gaming", "snapdragon", "bionic", "dimensity", "multitasking"
    ], 
    "display": [
        "screen", "display", "resolution", "oled", "lcd", "brightness", "touch", 
        "ghost touch", "flicker", "dead pixel", "hz", "refresh rate", "amoled", "nits"
    ],
    "build":[
        "build", "durable", "durability", "scratch", "crack", "drop", "case", 
        "back glass", "feel", "design", "plastic", "metal", "glass", "weight", "grip"
    ],
    "software": [
        "update", "os", "ui", "bug", "bloatware", "security patch", "firmware", 
        "restart", "ios", "android", "glitch", "crash", "interface", "software"
    ],
    "connectivity": [
        "5g", "lte", "wifi", "bluetooth", "nfc", "signal", "dropped call", 
        "hotspot", "reception", "network", "sim", "gps"
    ], 
    "value": [
        "price", "cost", "value", "expensive", "cheap", "warranty", "support", 
        "customer service", "worth", "deal", "budget"
    ]
}

# Default weights (sum to 1.0)
ASPECT_WEIGHTS: Dict[str, float] = {
    "battery": 0.18,
    "camera": 0.16,
    "performance": 0.14,
    "display": 0.12,
    "build": 0.10,
    "software": 0.12,
    "connectivity": 0.08,
    "value": 0.10
}

MIN_MENTIONS_FOR_CONFIDENCE: Dict[str, int] = {
    "battery": 5, "camera": 4, "performance": 4, "display": 3, 
    "build": 3, "software": 3, "connectivity": 3, "value": 3
}

OVERHEAT_COMBINATION_KEYS = ["overheat", "overheats", "overheating", "heating", "hot", "thermal", "temperature"]
CHARGING_KEYS = ["charging", "charge", "charger"]

SAMPLE_DATASETS = [
    {"name": "Kaggle - Amazon Electronics (Phones subset)", "url": "https://www.kaggle.com/datasets?search=amazon+electronics+phone"},
    {"name": "Public AWS Amazon Review dataset - Electronics", "url": "https://registry.opendata.aws/amazon-reviews/"}
]

def find_aspect_for_sentence(sentence: str) -> str:
    """
    Return aspect name or 'misc' based on keyword density.
    Counts occurrences of keywords for each aspect and returns the winner.
    """
    s = sentence.lower()
    counts = {asp: 0 for asp in ASPECT_KEYWORDS}
    
    total_matches = 0
    for asp, kws in ASPECT_KEYWORDS.items():
        for k in kws:
            if k in s:
                # Give bonus points for exact word matches vs substrings
                # e.g. "os" in "cost" is bad, but " os " is good
                if re.search(rf"\b{re.escape(k)}\b", s):
                    counts[asp] += 3  # Higher weight for exact match
                else:
                    counts[asp] += 1
                total_matches += 1
    
    if total_matches == 0:
        return "misc"
        
    best_aspect = max(counts, key=counts.get)
    
    # Threshold: Must have at least a strong match to override 'misc'
    if counts[best_aspect] == 0:
        return "misc"
        
    return best_aspect

def detect_overheat_in_reviews(sentences: List[str]) -> Tuple[int, List[int]]:
    indices = []
    for i, s in enumerate(sentences):
        sl = s.lower()
        if any(k in sl for k in OVERHEAT_COMBINATION_KEYS):
            indices.append(i)
    return len(indices), indices

def _contains_negation(text: str) -> bool:
    if not text: return False
    t = text.lower()
    neg_tokens = [" not ", "n't", " no ", " without ", " never ", "not related", "doesn't", "didn't", "cannot", "can't"]
    return any(tok in t for tok in neg_tokens)

def detect_overheat_with_charging(reviews_texts: List[str]) -> int:
    c = 0
    for t in reviews_texts:
        tl = (str(t) if t is not None else "").lower()
        has_overheat = any(k in tl for k in OVERHEAT_COMBINATION_KEYS)
        has_charging = any(k in tl for k in CHARGING_KEYS)
        if has_overheat and has_charging:
            if _contains_negation(tl):
                continue
            c += 1
    return c

def detect_conditional_praise(review_text: str) -> bool:
    t = review_text.lower()
    positive = ["good", "great", "excellent", "amazing", "love", "satisfied"]
    negative = ["bad", "poor", "terrible", "awful", "hate", "disappoint", "broken", "overheat", "drain"]
    sentences = re.split(r'[.!?]+', t)
    for s in sentences:
        if any(p in s for p in positive) and any(n in s for n in negative):
            return True
    return False

def temporal_burst_detector(df_reviews, date_col: str = "reviewTime", threshold_count: int = 10, threshold_days: int = 3) -> bool:
    try:
        import pandas as pd
        if date_col not in df_reviews.columns:
            return False
        ts = pd.to_datetime(df_reviews[date_col], errors="coerce").dropna().sort_values()
        if len(ts) < threshold_count:
            return False
        for i in range(len(ts)-threshold_count+1):
            start = ts.iloc[i]
            end = ts.iloc[i+threshold_count-1]
            if (end - start).days <= threshold_days:
                return True
        return False
    except Exception:
        return False

def get_adapter():
    return {
        "aspect_keywords": ASPECT_KEYWORDS,
        "aspect_weights": ASPECT_WEIGHTS,
        "min_mentions": MIN_MENTIONS_FOR_CONFIDENCE,
        "heuristics": {
            "overheat_keys": OVERHEAT_COMBINATION_KEYS,
            "charging_keys": CHARGING_KEYS
        },
        "samples": SAMPLE_DATASETS
    }