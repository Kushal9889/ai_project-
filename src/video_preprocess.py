"""
Transcript preprocessing & chunking utilities.
"""
import re
from typing import List


def clean_transcript(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r'\b(\w+)( \1\b)+', r'\1', t, flags=re.IGNORECASE)
    return t


def chunk_text(text: str, max_words: int = 120) -> List[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks