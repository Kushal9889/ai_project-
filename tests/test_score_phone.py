import pandas as pd
import pytest

from src import score_phone


def test_normalize_compound_to_0_100():
    assert score_phone._normalize_compound_to_0_100(-1.0) == 0.0
    assert score_phone._normalize_compound_to_0_100(0.0) == 50.0
    assert score_phone._normalize_compound_to_0_100(1.0) == 100.0
    assert score_phone._normalize_compound_to_0_100(0.5) == 75.0


def test_aspect_base_score_empty_and_simple():
    # empty compounds -> neutral 50
    assert score_phone._aspect_base_score([], []) == 50.0

    # simple case: compound 0.0 and strength 0.5
    val = score_phone._aspect_base_score([0.0], [0.5])
    # base = 50, avg_strength=0.5 -> bonus=5, result = base*0.9 + bonus*0.1 = 45 + 0.5 = 45.5
    assert pytest.approx(val, rel=1e-6) == 45.5


def test_apply_overheat_penalty():
    asp = {"battery": 80.0, "performance": 70.0, "camera": 90.0}
    out = score_phone.apply_overheat_penalty(dict(asp), overheating_count=2)
    # battery and performance should be multiplied by 0.7 and rounded to 2 decimals
    assert out["battery"] == round(80.0 * 0.7, 2)
    assert out["performance"] == round(70.0 * 0.7, 2)
    # unrelated aspect remains unchanged
    assert out["camera"] == 90.0


def test_compute_trust_score_with_monkeypatch(monkeypatch):
    # Prepare a small DataFrame - details won't matter because we'll monkeypatch
    df = pd.DataFrame({
        "reviewText": ["Good battery", "Overheats sometimes"],
        "compound": [0.5, -0.2],
        "strength": [0.6, 0.4],
        "rating": [4, 3],
        "is_fake": [False, False],
    })

    # Provide an adapter config with equal weights so final is simple average
    adapter_cfg = {"aspect_weights": {"battery": 1.0, "performance": 1.0}}

    # Monkeypatch compute_aspect_scores to return deterministic aspect scores and counts
    def fake_compute_aspect_scores(_df, _cfg, **kwargs):
        return ({"battery": 80.0, "performance": 70.0}, {"battery": 3, "performance": 2})

    monkeypatch.setattr(score_phone, "compute_aspect_scores", fake_compute_aspect_scores)

    # Monkeypatch detect_overheat_with_charging so it returns 0 (no penalty)
    monkeypatch.setattr(score_phone, "detect_overheat_with_charging", lambda texts: 0)

    out = score_phone.compute_trust_score(df, adapter_cfg)
    # With equal weights 1 and 1, normalized weights are 0.5 each -> final score = (80*0.5 + 70*0.5) = 75.0
    assert out["final_score"] == 75.0
    # Check structure keys exist
    assert "aspect_scores" in out and "aspect_counts" in out
