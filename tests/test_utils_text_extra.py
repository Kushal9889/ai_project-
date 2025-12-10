from src import utils_text


def test_sentence_split_and_compound_split():
    text = "Camera is great, but battery is bad. However, screen is excellent!"
    sents = utils_text.sentence_split(text)
    assert any("Camera is great" in s for s in sents)
    parts = utils_text.split_compound_sentences(text)
    # should split into at least two topical parts
    assert any("camera" in p.lower() for p in parts)
    assert any("battery" in p.lower() for p in parts)


def test_analyze_sentence_sentiment_basic():
    pos, strength = utils_text.analyze_sentence_sentiment("This is excellent and amazing")
    assert pos > 0
    assert strength >= 0

    neg, nstrength = utils_text.analyze_sentence_sentiment("This is terrible and awful")
    assert neg < 0
    assert nstrength >= 0
