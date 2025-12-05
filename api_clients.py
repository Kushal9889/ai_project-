"""HF + Mock clients for classification and embeddings.
Used by multiple modules. Mock client runs offline and deterministically.

Secret resolution order (preferred):
 - Streamlit secrets (if running inside Streamlit and secret set)
 - .env file (loaded via python-dotenv if available)
 - environment variable HF_API_KEY
 - explicit api_key passed to HFClient()
"""
import os
import requests
from typing import Any, Dict, List, Optional

# Use the new router endpoint by default. Allow overriding via HF_API_URL env var
# Old endpoint (api-inference.huggingface.co) is deprecated and returns HTTP 410.
HF_API_URL = os.getenv("HF_API_URL", "https://router.huggingface.co/models/{}")

# Try to load a .env file if python-dotenv is installed (no hard failure)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _get_streamlit_secret():
    try:
        import streamlit as _st

        return _st.secrets.get("HF_API_KEY") if hasattr(_st, "secrets") else None
    except Exception:
        return None


# Resolve HF_API_KEY from streamlit secrets first, then env var
HF_API_KEY = _get_streamlit_secret() or os.getenv("HF_API_KEY", None)


class HFClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or HF_API_KEY
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def classify(self, model: str, inputs: List[str], params: Dict[str, Any] = None, timeout: int = 30):
        """
        Send a batch of inputs to HF inference endpoint.
        Returns the raw JSON response (list per input).
        """
        url = HF_API_URL.format(model)
        body = {"inputs": inputs}
        if params:
            body["parameters"] = params
        resp = requests.post(url, headers=self.headers, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def embeddings(self, model: str, inputs: List[str], timeout: int = 30):
        """
        Request embeddings via HF inference API (if model supports).
        """
        url = HF_API_URL.format(model)
        body = {"inputs": inputs}
        resp = requests.post(url, headers=self.headers, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def find_working_model(self, candidates: List[str], timeout: int = 8) -> Optional[str]:
        """
        Probe a list of Hugging Face model IDs and return the first model that
        responds with HTTP 200 and returns JSON (indicating a usable inference API).

        This helps automatically pick a public/free model that the token (or
        unauthenticated requests) can call via the router endpoint.
        """
        for m in candidates:
            try:
                url = HF_API_URL.format(m)
                body = {"inputs": "test"}
                # allow missing auth header (public models may be callable without a token)
                headers = self.headers or {}
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
                # Accept 200 with valid JSON only
                if resp.status_code == 200:
                    try:
                        _ = resp.json()
                        return m
                    except ValueError:
                        # Not JSON, skip
                        continue
                else:
                    # non-200 (404, 401, 410, 403, etc.) -> skip
                    continue
            except requests.exceptions.RequestException:
                # network/timeouts/other -> try next candidate
                continue
        return None


class MockHFClient(HFClient):
    """
    Deterministic mock client for local development and unit tests.
    Returns handcrafted outputs suitable for the rest of the pipeline.
    Uses comprehensive sentiment lexicon similar to VADER for accurate analysis.
    """

    # Comprehensive sentiment lexicon with intensity scores (-1.0 to +1.0)
    SENTIMENT_LEXICON = {
        # Very Positive (0.8-1.0)
        "amazing": 0.95, "excellent": 0.95, "outstanding": 0.95, "superb": 0.95,
        "fantastic": 0.95, "incredible": 0.95, "phenomenal": 0.95, "exceptional": 0.95,
        "love": 0.9, "perfect": 0.9, "wonderful": 0.9, "brilliant": 0.9,
        "awesome": 0.9, "magnificent": 0.9, "terrific": 0.9, "fabulous": 0.9,
        "best": 0.85, "great": 0.85, "impressive": 0.85, "remarkable": 0.85,
        
        # Positive (0.4-0.79)
        "good": 0.7, "nice": 0.7, "solid": 0.7, "happy": 0.7,
        "satisfied": 0.7, "pleased": 0.7, "enjoy": 0.7, "like": 0.65,
        "works": 0.6, "fast": 0.6, "smooth": 0.6, "clear": 0.6,
        "bright": 0.55, "reliable": 0.65, "quality": 0.6, "decent": 0.5,
        "acceptable": 0.4, "okay": 0.4, "fine": 0.5, "adequate": 0.45,
        
        # Very Negative (-1.0 to -0.8)
        "terrible": -0.95, "awful": -0.95, "horrible": -0.95, "worst": -0.95,
        "pathetic": -0.95, "disgusting": -0.95, "atrocious": -0.95, "abysmal": -0.95,
        "hate": -0.9, "useless": -0.9, "garbage": -0.9, "trash": -0.9,
        "junk": -0.85, "scam": -0.9, "fraud": -0.9, "waste": -0.85,
        
        # Negative (-0.79 to -0.4)
        "bad": -0.75, "poor": -0.75, "disappointing": -0.75, "disappointed": -0.75,
        "fail": -0.7, "failed": -0.7, "broken": -0.75, "defective": -0.75,
        "problem": -0.65, "issue": -0.6, "slow": -0.65, "lag": -0.65,
        "crash": -0.75, "freeze": -0.7, "overheat": -0.7, "hot": -0.5,
        "expensive": -0.5, "overpriced": -0.6, "cheap": -0.45, "flimsy": -0.6,
        "difficult": -0.5, "confusing": -0.5, "complicated": -0.45, "annoying": -0.6,
        "frustrating": -0.7, "frustrate": -0.7, "concern": -0.4, "worried": -0.5,
        
        # Negations and modifiers
        "not": 0.0, "no": 0.0, "never": 0.0, "neither": 0.0, "nobody": 0.0,
        "very": 0.0, "really": 0.0, "extremely": 0.0, "absolutely": 0.0,
        "quite": 0.0, "highly": 0.0, "too": 0.0, "so": 0.0,
    }
    
    # Negation words that flip sentiment
    NEGATIONS = {"not", "no", "never", "neither", "nobody", "none", "nothing", "nowhere"}
    
    # Intensifiers that boost sentiment
    INTENSIFIERS = {
        "very": 1.3, "really": 1.3, "extremely": 1.5, "absolutely": 1.5,
        "quite": 1.2, "highly": 1.3, "too": 1.3, "so": 1.3, "incredibly": 1.5,
        "exceptionally": 1.5, "remarkably": 1.4, "particularly": 1.2,
    }

    def __init__(self):
        super().__init__(api_key=None)

    def _analyze_sentiment_score(self, text: str) -> float:
        """
        Advanced sentiment analysis using lexicon + negation handling + intensifiers.
        Returns score from -1.0 (very negative) to +1.0 (very positive).
        """
        if not text:
            return 0.0
        
        # Tokenize and clean
        words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
        
        scores = []
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check if this word is in lexicon
            if word in self.SENTIMENT_LEXICON:
                base_score = self.SENTIMENT_LEXICON[word]
                
                # Look for intensifier before this word
                intensifier = 1.0
                if i > 0 and words[i-1] in self.INTENSIFIERS:
                    intensifier = self.INTENSIFIERS[words[i-1]]
                
                # Look for negation within 3 words before
                negated = False
                for j in range(max(0, i-3), i):
                    if words[j] in self.NEGATIONS:
                        negated = True
                        break
                
                # Apply intensifier and negation
                score = base_score * intensifier
                if negated:
                    score = -score * 0.8  # Flip and slightly reduce intensity
                
                scores.append(score)
            
            i += 1
        
        # Calculate final sentiment
        if not scores:
            return 0.0
        
        # Use sum for aggregate sentiment (allows multiple words to compound)
        total = sum(scores)
        # Normalize to [-1, 1] range with dampening for very long texts
        sentiment = total / (1.0 + abs(total) * 0.1)
        
        return max(-1.0, min(1.0, sentiment))

    def classify(self, model: str, inputs: List[str], params: Dict[str, Any] = None, timeout: int = 30):
        results = []
        for text in inputs:
            sentiment_score = self._analyze_sentiment_score(text)
            
            # Convert sentiment score to positive/negative probabilities
            # Score ranges from -1 to +1
            # Positive score -> higher positive probability
            # Negative score -> higher negative probability
            
            if sentiment_score > 0.2:
                # Positive sentiment
                pos_prob = 0.5 + (sentiment_score * 0.45)  # Range: 0.59 to 0.95
                neg_prob = 1.0 - pos_prob
                label = "POSITIVE"
                score = pos_prob
            elif sentiment_score < -0.2:
                # Negative sentiment
                neg_prob = 0.5 + (abs(sentiment_score) * 0.45)  # Range: 0.59 to 0.95
                pos_prob = 1.0 - neg_prob
                label = "NEGATIVE"
                score = neg_prob
            else:
                # Neutral sentiment
                pos_prob = 0.5 + (sentiment_score * 0.3)  # Range: 0.44 to 0.56
                neg_prob = 1.0 - pos_prob
                label = "NEUTRAL"
                score = 0.5 + abs(sentiment_score) * 0.2
            
            results.append([{"label": label, "score": float(score)}])
        
        return results

    def embeddings(self, model: str, inputs: List[str], timeout: int = 30):
        import numpy as np
        # return deterministic pseudo-embeddings based on text length
        vecs = []
        dim = 384
        for t in inputs:
            l = len(t or "")
            v = (np.arange(dim) + (l % 7)) % 8
            v = v.astype(float) / (v.max() + 1e-6)
            vecs.append(v.tolist())
        return vecs


