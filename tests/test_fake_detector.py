import pandas as pd

from src.fake_detector import heuristics_score, detect_fake_reviews


def test_heuristics_score_short_and_spam():
    s1 = "Great phone"
    s2 = "Amazing product, best product ever!"
    # short text should get at least the short-word penalty
    assert heuristics_score(s1) >= 0.25
    # spammy phrase should increase score
    assert heuristics_score(s2) > heuristics_score(s1)


def test_detect_fake_reviews_similarity_clusters():
    # Create a DF with duplicate/similar reviews to trigger clustering
    rows = [
        {"reviewText": "This product is great and works perfectly."},
        {"reviewText": "This product is great and works perfectly."},
        {"reviewText": "Battery life is excellent and reliable."},
    ]
    df = pd.DataFrame(rows)

    # Use tight similarity threshold and min_cluster_size=2 to detect duplicates
    out = detect_fake_reviews(df, sim_threshold=0.9, min_cluster_size=2, mock_mode=True)

    # similarity_cluster should mark the two identical entries
    sim_flags = out["similarity_cluster"].tolist()
    assert sum(sim_flags) >= 2

    # hf_fake_score column exists and probabilities are between 0 and 1
    probs = out["hf_fake_score"].tolist()
    assert all(0.0 <= p <= 1.0 for p in probs)
