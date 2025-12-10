from src.adapters import phone_adapter


def test_find_aspect_for_sentence_basic():
    assert phone_adapter.find_aspect_for_sentence("The battery life is excellent and charging is fast") == "battery"
    assert phone_adapter.find_aspect_for_sentence("Amazing camera in low light, great photos") == "camera"
    assert phone_adapter.find_aspect_for_sentence("Phone lags when gaming and shows overheat warnings") == "performance"
    assert phone_adapter.find_aspect_for_sentence("Comes with good case and accessories") in ("build", "value", "misc")


def test_detect_overheat_with_charging():
    texts = [
        "My phone overheats while charging and becomes hot",
        "It overheats but I don't charge it while playing",
        "Charging makes it hot and overheat occurs",
    ]
    # Two entries contain both overheat and charging keywords
    assert phone_adapter.detect_overheat_with_charging(texts) == 2


def test_detect_conditional_praise():
    t = "I love the camera but the battery is terrible"
    assert phone_adapter.detect_conditional_praise(t) is True
    t2 = "Great phone overall, no issues"
    assert phone_adapter.detect_conditional_praise(t2) is False
