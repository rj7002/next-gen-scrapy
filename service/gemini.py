"""Gemini fallback for questions the rule parser couldn't fully understand.

Uses Google's Interactions API (POST /v1beta/interactions) - the older generateContent endpoint
isn't available to new API keys. The key comes from GEMINI_API_KEY (loaded from .env by app.py).
Gemini only proposes filters; service/nl.py validates every one against the curated filter list.
"""

from __future__ import annotations

import json
import os
import re

import requests

URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
_cache: dict[str, dict] = {}


class GeminiUnavailable(Exception):
    pass


def _extract_text(resp: dict) -> str:
    for step in resp.get("steps", []):
        if step.get("type") == "model_output":
            return "".join(c.get("text", "") for c in step.get("content", []) if c.get("type") == "text")
    raise GeminiUnavailable("no model output in response")


def ask(prompt: str) -> dict:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiUnavailable("GEMINI_API_KEY not set")
    if prompt in _cache:
        return _cache[prompt]
    try:
        r = requests.post(URL, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                          json={"model": MODEL, "input": prompt}, timeout=25)
    except requests.RequestException as e:
        raise GeminiUnavailable(f"network error: {e.__class__.__name__}") from e
    if r.status_code == 429:
        raise GeminiUnavailable("rate limit reached")
    if not r.ok:
        raise GeminiUnavailable(f"HTTP {r.status_code}")
    text = _extract_text(r.json()).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise GeminiUnavailable("reply wasn't JSON") from e
        out = json.loads(m.group(0))
    _cache[prompt] = out
    return out
