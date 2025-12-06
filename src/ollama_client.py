"""
Simple Ollama API client wrapper.
Updated to support 7-point granular sentiment scale, Aspect Classification, and Targeted Summarization.
"""
import requests
import json
from typing import Optional, List

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # Verify connection
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                print(f"Warning: Ollama server connected but returned {resp.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama at {self.base_url}. Ensure 'ollama serve' is running.")

    def classify_sentiment(self, model: str, text: str) -> str:
        prompt = (
            "Analyze the sentiment of this review text and assign exactly one label from this list:\n"
            "1. Extreme Positive\n"
            "2. Positive\n"
            "3. Slightly Positive\n"
            "4. Neutral\n"
            "5. Slightly Negative\n"
            "6. Negative\n"
            "7. Extreme Negative\n\n"
            "Reply with ONLY the label text. Do not explain.\n"
            f"Text: \"{text}\"\nLabel:"
        )
        return self._generate(model, prompt, num_predict=8)

    def classify_fake(self, model: str, text: str) -> str:
        prompt = (
            "Is this product review Real or Fake/Spam?\n"
            "Reply with exactly one word: 'Real' or 'Fake'.\n"
            f"Review: \"{text}\"\nVerdict:"
        )
        return self._generate(model, prompt, num_predict=6)

    def classify_aspect(self, model: str, text: str, valid_aspects: List[str]) -> str:
        aspects_str = ", ".join(valid_aspects)
        prompt = (
            f"Identify the PRIMARY feature discussed in this text.\n"
            f"Valid categories: {aspects_str}, misc.\n"
            f"Text: \"{text}\"\n"
            "Reply with ONLY the category name."
        )
        response = self._generate(model, prompt, num_predict=10).lower().strip()
        for aspect in valid_aspects:
            if aspect in response:
                return aspect
        return "misc"

    def generate_summary(self, model: str, reviews: List[str], topic: str = "general") -> str:
        """Generate a concise, topic-focused summary."""
        # Clean inputs
        clean_reviews = [r.replace("\n", " ").strip() for r in reviews if len(r) > 10]
        text = "\n- ".join(clean_reviews[:20]) # Limit input size
        
        prompt = (
            f"You are a tech reviewer. Write a concise 2-3 sentence summary specifically about the **{topic.upper()}** of this product, based ONLY on the excerpts below.\n"
            f"IMPORTANT: Strictly IGNORE any information about other features (e.g. if topic is Battery, do not mention Camera).\n"
            "Synthesize the points into a coherent paragraph. Do NOT list bullet points.\n\n"
            f"User Feedback on {topic}:\n- {text}\n\n"
            "Summary:"
        )
        return self._generate(model, prompt, num_predict=100)

    def _generate(self, model: str, prompt: str, num_predict: int = 20) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict}
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException:
            return ""