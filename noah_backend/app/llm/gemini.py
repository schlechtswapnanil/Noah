"""Gemini adapter: the last step in the provider chain (app/llm/provider.py).

REST through httpx rather than the google-genai SDK: one endpoint, no new
dependency. Two key formats reach two endpoints:

* ``AIza...`` keys from Google AI Studio -> generativelanguage.googleapis.com
* ``AQ....`` keys from Vertex AI express mode -> aiplatform.googleapis.com

``GEMINI_API_URL`` (with ``{model}`` in it) overrides that choice. Thinking is
switched off for the 2.5 models, where it is a token cost with nothing to
show for it in a two-sentence reply; the parameter is not sent to other
generations, which name it differently.

Free-tier prompts may be used by Google to improve its models. The prompts
carry the user's shopping message and any typed place, never their identity.
"""

from __future__ import annotations

import os

import httpx

AI_STUDIO_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
                 "{model}:generateContent")
VERTEX_EXPRESS_URL = ("https://aiplatform.googleapis.com/v1/publishers/google/models/"
                      "{model}:generateContent")
DEFAULT_MODEL = "gemini-2.5-flash-lite"
TIMEOUT_SECONDS = 20.0


def endpoint_for(api_key: str, model: str) -> str:
    override = os.getenv("GEMINI_API_URL", "").strip()
    if override:
        return override.format(model=model)
    template = VERTEX_EXPRESS_URL if api_key.startswith("AQ.") else AI_STUDIO_URL
    return template.format(model=model)


def request_body(system_prompt: str, user_prompt: str, temperature: float,
                 max_tokens: int, model: str) -> dict:
    generation: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if model.startswith("gemini-2.5"):
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    return {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": generation,
    }


def _post(url: str, body: dict, api_key: str) -> httpx.Response:
    return httpx.post(url, json=body, timeout=TIMEOUT_SECONDS,
                      headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})


def text_of(payload: dict) -> str:
    """The reply text, or "" when the response carries none (a safety block
    shows up as a candidate without parts, or no candidate at all)."""
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def generate_text(system_prompt: str, user_prompt: str,
                  temperature: float = 0.5, max_tokens: int = 220,
                  model: str | None = None) -> str:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    response = _post(endpoint_for(api_key, model),
                     request_body(system_prompt, user_prompt, temperature, max_tokens, model),
                     api_key)
    response.raise_for_status()
    return text_of(response.json())
