"""
YouTube transcript fetcher with cache and normalization.
"""
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import os
import json
import re
from urllib.parse import urlparse, parse_qs


def _extract_video_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    if len(url_or_id) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', url_or_id):
        return url_or_id
    try:
        u = urlparse(url_or_id)
        if u.hostname in ('www.youtube.com', 'youtube.com'):
            qs = parse_qs(u.query)
            return qs.get('v', [None])[0]
        if u.hostname == 'youtu.be':
            return u.path.lstrip('/')
    except Exception:
        pass
    return url_or_id


def normalize_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text


def get_youtube_transcript(video_url_or_id: str, cache_dir: str = "data/cache", use_cache: bool = True):
    video_id = _extract_video_id(video_url_or_id)
    if not video_id:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{video_id}.json")
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f).get("text", None)
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t["text"] for t in transcript_list])
        text = normalize_transcript(text)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"video_id": video_id, "text": text}, f, ensure_ascii=False)
        return text
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None