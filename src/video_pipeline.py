"""
Simple video transcript -> sentiment/emotion -> video_score -> fusion pipeline.
Mock-first; uses sentiment_engine for per-block sentiment analysis.
"""
from typing import Optional, Dict, Any
from .video_transcript import get_youtube_transcript
from .video_preprocess import clean_transcript, chunk_text
from .api_clients import MockHFClient, HFClient
from .sentiment_engine import analyze_sentiments
from .config import DEFAULT_CONFIG
import numpy as np

CFG = DEFAULT_CONFIG["video"]


def analyze_video(video_url_or_id: str, hf_client: Optional[HFClient] = None, mock_mode: bool = True) -> Dict[str, Any]:
    text = get_youtube_transcript(video_url_or_id)
    if not text:
        return {"error": "no_transcript"}
    text = clean_transcript(text)
    chunks = chunk_text(text, max_words=CFG["chunk_max_words"])
    client = hf_client or (MockHFClient() if mock_mode else HFClient())
    import pandas as pd
    blocks_df = pd.DataFrame({"reviewText": chunks})
    sent_out = analyze_sentiments(blocks_df, hf_client=client, mock_mode=mock_mode)
    df_blocks = sent_out["df"]
    pos = sum(1 for p in df_blocks["sentiment_pos"] if p > 0.5)
    neg = sum(1 for p in df_blocks["sentiment_neg"] if p > 0.5)
    tot = len(df_blocks)
    pos_ratio = pos / tot if tot else 0.0
    neg_ratio = neg / tot if tot else 0.0
    emotion_hist = sent_out["summary"].get("emotion_histogram", {})
    if emotion_hist:
        vals = np.array(list(emotion_hist.values()))
        p = vals / (vals.sum() + 1e-9)
        emo_depth = float(-(p * np.log2(p + 1e-9)).sum())
    else:
        emo_depth = 0.0
    aspect_consistency = 0.5
    weights = CFG["video_score_weights"]
    video_score = (weights["pos_ratio"] * (pos_ratio * 100) + weights["emotion_depth"] * (emo_depth * 100) + weights["aspect_consistency"] * (aspect_consistency * 100)) - (weights["neg_penalty"] * (neg_ratio * 100))
    video_score = max(0.0, min(100.0, video_score))
    return {"video_score": video_score, "pos_ratio": pos_ratio, "neg_ratio": neg_ratio, "emo_depth": emo_depth, "blocks": len(df_blocks)}


def fuse_scores(text_score: float, video_score: float, fusion_weights: Optional[Dict[str, float]] = None) -> float:
    if fusion_weights is None:
        fusion_weights = CFG["fusion_weights"]
    final = fusion_weights["text"] * text_score + fusion_weights["video"] * video_score
    return float(max(0.0, min(100.0, final)))