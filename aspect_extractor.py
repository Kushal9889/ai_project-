"""
Aspect extraction utilities used by the phone adapter pipeline.
- extract_sentences_from_reviews: split reviews into sentences
- assign_aspects_to_sentences: map sentences -> aspects using adapter keywords
"""
from typing import List, Tuple, Dict
import pandas as pd
from .utils_text import sentence_split
from .adapters.phone_adapter import find_aspect_for_sentence


def extract_sentences_from_reviews(df: pd.DataFrame, text_col: str = "reviewText") -> List[str]:
    """Return list of sentences (non-empty) extracted from all reviews (in order)."""
    sents = []
    texts = []
    # defensive handling: accept DataFrame without column, single-column DataFrame, or Series/list
    if isinstance(df, pd.DataFrame):
        if text_col in df.columns:
            texts = df[text_col].fillna("").astype(str).tolist()
        elif len(df.columns) == 1:
            texts = df.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            texts = []
    elif isinstance(df, (pd.Series, list, tuple)):
        texts = list(df)
    else:
        try:
            texts = list(df)
        except Exception:
            texts = []

    for text in texts:
        for s in sentence_split(str(text or "")):
            s = s.strip()
            if s:
                sents.append(s)
    return sents


def assign_aspects_to_sentences(sentences: List[str]) -> List[str]:
    """Map each sentence to an aspect using phone adapter keywords."""
    return [find_aspect_for_sentence(s) for s in sentences]


def map_sentences_to_reviews(df: pd.DataFrame, text_col: str = "reviewText") -> List[Tuple[int, str]]:
    """
    Return list of tuples (review_index, sentence) to preserve which review each sentence belongs to.
    Useful when computing aspect counts per-review.
    """
    result = []
    texts = []
    if isinstance(df, pd.DataFrame):
        if text_col in df.columns:
            texts = df[text_col].fillna("").astype(str).tolist()
        elif len(df.columns) == 1:
            texts = df.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            texts = []
    elif isinstance(df, (pd.Series, list, tuple)):
        texts = list(df)
    else:
        try:
            texts = list(df)
        except Exception:
            texts = []

    for idx, text in enumerate(texts):
        sents = [s.strip() for s in sentence_split(str(text or "")) if s.strip()]
        for s in sents:
            result.append((idx, s))
    return result