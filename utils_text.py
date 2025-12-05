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