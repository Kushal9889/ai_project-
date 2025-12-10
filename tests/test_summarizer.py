import pandas as pd

from src.summarizer import summarize_reviews


def test_summarize_reviews_basic_mock():
    df = pd.DataFrame({
        "reviewText": [
            "Battery life is excellent and charges fast.",
            "Battery overheats while charging and drains quickly.",
            "Camera takes great photos in low light.",
            "Camera is poor in low light and often blurry.",
            "Battery shows inconsistent behavior when charging.",
        ]
    })

    out = summarize_reviews(df, text_col="reviewText", provider="mock")
    # Should return a dict with keys
    assert isinstance(out, dict)
    assert "aspect_breakdown" in out
    assert "insights" in out
    # Overheat should be detected in insights because one review contains 'overheats'
    assert "Overheating" in out["insights"]["hidden_issues"] or "overheat" in out["insights"]["hidden_issues"].lower()


def test_summarize_reviews_empty():
    df = pd.DataFrame({"reviewText": []})
    out = summarize_reviews(df, text_col="reviewText", provider="mock")
    assert out["positive_summary"] == ""
    assert out["negative_summary"] == ""
