"""
Central configuration for True Score Analyser project, updated with phone adapter defaults.
"""
import os

DEFAULT_CONFIG = {
    "mock_mode": os.getenv("ELITEK_MOCK", "true").lower() in ("1", "true", "yes"),
    "embedding_model": "all-MiniLM-L6-v2",
    "fake_detection": {
        "hf_model": "mrm8488/bert-tiny-fake-review-detection",
        "heuristics_weight": 0.5,
        "sim_threshold": 0.90,
        "min_entropy": 2.0,
        "min_words_short": 12,
        "cluster_size_min": 10,
        "vocab_diversity_min": 0.2
    },
    "sentiment": {
        "primary_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
        # Fallback lists: if a configured model endpoint is removed from HF inference API
        # (HTTP 410) we'll try these alternatives in order before using the mock client.
        "fallback_models": [
            "cardiffnlp/twitter-roberta-base-sentiment",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "distilbert-base-uncased-finetuned-sst-2"
        ],
        # Expanded candidate list used by the automatic model picker. The
        # sentiment engine will probe these in order and pick the first that
        # responds via the router endpoint. You can add/remove models here.
        "candidate_models": [
            "distilbert-base-uncased-finetuned-sst-2",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "cardiffnlp/twitter-roberta-base-sentiment",
            "siebert/sentiment-roberta-large-english",
            "mrm8488/distilbert-base-uncased-finetuned-sst-2-english",
            "bhadresh-savani/bert-base-uncased-emotion"
        ],
        "fallback_emotion_models": [
            "j-hartmann/emotion-english-distilroberta-base",
            "bhadresh-savani/bert-base-uncased-emotion"
        ],
        "batch_size": 16,
        "pos_threshold": 0.2,
        "neg_threshold": -0.2,
        "small_gap": 0.12,
        "strong_threshold": 0.65,
        "alpha_prob_gap": 0.6,
        "alpha_emotion_div": 0.3,
        "alpha_linguistic": 0.1
    },
    "summarizer": {
        "min_reviews_for_clustering": 8,
        "top_k_representatives": 6,
        "dedupe_similarity_threshold": 0.92,
        "min_cluster_size_to_summarize": 3,
        "summarizer_model": "sshleifer/distilbart-cnn-12-6",
        "mock_mode": True
    },
    "video": {
        "chunk_max_words": 120,
        "sentiment_model_choice": "distilbert",
        "batch_size": 16,
        "video_score_weights": {"pos_ratio": 0.4, "emotion_depth": 0.2, "aspect_consistency": 0.3, "neg_penalty": 0.1},
        "fusion_weights": {"text": 0.7, "video": 0.3}
    },
    # Phone adapter defaults
    "phone_adapter": {
        "aspects": {
            "battery": 0.18,
            "camera": 0.16,
            "performance": 0.14,
            "display": 0.12,
            "build": 0.10,
            "software": 0.12,
            "connectivity": 0.08,
            "value": 0.10
        },
        "min_mentions_for_confidence": {
            "battery": 5, "camera": 4, "performance": 4, "display": 3, "build": 3, "software": 3, "connectivity": 3, "value": 3
        },
        "overheat_escalation_multiplier": 0.7,
        "conditional_praise_window": 80,  # chars threshold to consider same review conditional mention
        "temporal_burst_threshold": {"count": 10, "days": 3}
    }
}