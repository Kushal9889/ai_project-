"""
Utilities: tokenization, sentence splitting, entropy, Flesch-Kincaid, intensity markers.
Lightweight, no heavy external deps.
"""
import re
import math
from collections import Counter
from typing import List

_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    # Standard splitting on punctuation
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_compound_sentences(text: str) -> List[str]:
    """
    Aggressively split text into smaller atomic thoughts.
    Splits on punctuation AND conjunctions like 'but', 'however', 'whereas' to separate topics.
    """
    if not text:
        return []
    
    # 1. Standard punctuation split
    initial_parts = re.split(r'(?<=[.!?])\s+', text.strip())
    
    final_parts = []
    # 2. Split on strong separators to isolate topics
    # e.g., "Camera is great BUT battery is bad" -> ["Camera is great", "battery is bad"]
    separators = r', but |, however |, whereas |, although |, while |; '
    
    for part in initial_parts:
        sub_parts = re.split(separators, part, flags=re.IGNORECASE)
        for sp in sub_parts:
            sp = sp.strip()
            # Filter out tiny fragments that don't mean anything
            if sp and len(sp.split()) > 2:
                final_parts.append(sp)
    
    return final_parts


def shannon_entropy(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return round(entropy, 4)


def flesch_kincaid(text: str) -> float:
    words = tokenize_words(text)
    sentences = [s for s in sentence_split(text) if s.strip()]
    if len(words) == 0 or len(sentences) == 0:
        return 0.0
    def syllables(word):
        w = word.lower()
        if len(w) <= 3:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_v = False
        for ch in w:
            v = ch in vowels
            if v and not prev_v:
                count += 1
            prev_v = v
        if w.endswith("e") and count > 1:
            count -= 1
        return max(1, count)
    syll = sum(syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syll / len(words)
    score = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    return round(score, 2)


def has_linguistic_intensity(text: str) -> float:
    t = text or ""
    exclamations = t.count("!")
    caps = sum(1 for c in t if c.isupper())
    caps_ratio = caps / max(1, len(t))
    emphatics = exclamations + (2 if "very" in t.lower() or "extremely" in t.lower() else 0)
    score = min(1.0, (exclamations / 3.0) + caps_ratio + (emphatics * 0.05))
    return round(score, 4)


def analyze_sentence_sentiment(text: str) -> tuple[float, float]:
    """Lightweight sentence-level sentiment scorer.

    Returns (compound, strength) where compound is in [-1.0, 1.0]
    and strength is a 0..1 intensity estimate.
    This is intentionally fast and deterministic (lexicon-based) to be
    used in `fast_local_mode` or as a fallback when external models are
    unavailable.
    """
    if not text or not text.strip():
        return 0.0, 0.0
    words = tokenize_words(text)
    if not words:
        return 0.0, 0.0

    # compact lexicon tuned for phones domain (subset)
    lex = {
        "excellent": 0.9, "amazing": 0.9, "great": 0.8, "good": 0.6, "love": 0.9,
        "fast": 0.5, "smooth": 0.5, "reliable": 0.6, "clear": 0.4, "bright": 0.4,
        "poor": -0.7, "bad": -0.7, "terrible": -0.9, "awful": -0.9, "hate": -0.9,
        "slow": -0.6, "overheat": -0.8, "drain": -0.6, "broken": -0.8, "noise": -0.4,
        "disappoint": -0.6, "disappointing": -0.6, "problem": -0.5
    }

    # intensifiers and negations
    intensifiers = {"very": 1.3, "extremely": 1.5, "really": 1.2}
    negations = {"not", "no", "never", "n't"}

    score_sum = 0.0
    weight_sum = 0.0
    neg_window = 0
    for i, w in enumerate(words):
        val = lex.get(w, 0.0)
        if w in negations:
            neg_window = 3  # next up to 3 words may be negated
            continue
        if w in intensifiers:
            # apply to next token
            continue
        mult = 1.0
        # look back for intensifier
        if i > 0 and words[i-1] in intensifiers:
            mult *= intensifiers[words[i-1]]
        if neg_window > 0:
            val = -val * 0.8
        if val != 0.0:
            score_sum += val * mult
            weight_sum += abs(val) * mult
        if neg_window > 0:
            neg_window -= 1

    if weight_sum == 0.0:
        compound = 0.0
        strength = 0.0
    else:
        compound = max(-1.0, min(1.0, score_sum / (weight_sum)))
        # strength is how confident the sentence is (scaled by weight density)
        strength = min(1.0, weight_sum / (len(words) + 1e-6))

    return round(compound, 4), round(strength, 4)