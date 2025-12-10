import pandas as pd

from src.aspect_extractor import extract_sentences_from_reviews, assign_aspects_to_sentences, map_sentences_to_reviews


def test_extract_sentences_and_assign_aspects():
    df = pd.DataFrame({
        "reviewText": [
            "Battery lasts long. Camera is good in low light.",
            "Phone lags when gaming, but display is bright and pleasant."
        ]
    })

    sents = extract_sentences_from_reviews(df)
    # Should extract multiple sentences
    assert any("Battery lasts long" in s for s in sents)
    assert any("Camera is good" in s for s in sents)

    aspects = assign_aspects_to_sentences(sents)
    # aspects list length must match sentences
    assert len(aspects) == len(sents)
    # Some known mappings should be present (battery, camera, display, performance)
    assert any(a == "battery" for a in aspects)
    assert any(a in ("camera", "display", "performance") for a in aspects)


def test_map_sentences_to_reviews_indices():
    df = pd.DataFrame({"reviewText": ["Good battery. Bad camera.", "Excellent display."]})
    mapped = map_sentences_to_reviews(df)
    # Should return tuples (index, sentence)
    assert isinstance(mapped, list)
    assert all(isinstance(t, tuple) and isinstance(t[0], int) for t in mapped)
    # There should be at least 3 sentences mapped
    assert len(mapped) >= 3
