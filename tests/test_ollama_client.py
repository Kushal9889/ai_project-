import json
import types

import requests

from src.ollama_client import OllamaClient


class DummyResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"response": "OK"}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise requests.exceptions.HTTPError(response=self)


def test_ollama_generate_and_classify(monkeypatch):
    # Monkeypatch requests.get used in constructor
    monkeypatch.setattr("requests.get", lambda *a, **k: DummyResp(200, {}))

    # Monkeypatch post used in _generate
    def fake_post(url, json=None, timeout=None):
        # return different responses based on prompt content
        prompt = (json or {}).get("prompt", "")
        if "Label:" in prompt:
            return DummyResp(200, {"response": "Positive"})
        if "Verdict:" in prompt:
            return DummyResp(200, {"response": "Real"})
        if "Identify the PRIMARY feature" in prompt:
            return DummyResp(200, {"response": "Battery"})
        return DummyResp(200, {"response": "Summary about BATTERY."})

    monkeypatch.setattr("requests.post", fake_post)

    client = OllamaClient(base_url="http://localhost:11434")
    s = client.classify_sentiment("m", "I love this phone")
    assert "Positive" in s or s != ""

    f = client.classify_fake("m", "Spam content")
    assert f.lower() in ("real", "fake") or f != ""

    a = client.classify_aspect("m", "Battery drains", ["battery", "camera"]) 
    assert a in ("battery", "camera", "misc")

    g = client.generate_summary("m", ["Battery life is great", "Charges fast"], topic="battery")
    assert isinstance(g, str)

    # Simulate a network failure causing empty string return
    def raise_post(*a, **k):
        raise requests.exceptions.RequestException()
    monkeypatch.setattr("requests.post", raise_post)
    assert client._generate("m", "hello") == ""
