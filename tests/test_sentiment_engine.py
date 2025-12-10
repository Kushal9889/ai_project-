import pandas as pd

from src.sentiment_engine import analyze_sentiments


def test_analyze_sentiments_mock_basic():
    df = pd.DataFrame({"reviewText": ["This is excellent and amazing", "This is terrible and awful"]})
    out = analyze_sentiments(df, provider="mock")
    assert isinstance(out, dict)
    d = out["df"]
    assert "compound" in d.columns
    assert "strength" in d.columns
    # first should be positive compound, second negative
    assert d["compound"].iloc[0] > 0
    assert d["compound"].iloc[1] < 0
    # summary contains avg_compound
    assert "avg_compound" in out["summary"]
