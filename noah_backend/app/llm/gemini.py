"""Hosted open-weight LLM adapter (module name kept for import compatibility)."""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """Generate text only; callers retain ownership of all structured data."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise RuntimeError("HUGGINGFACE_API_KEY is missing.")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 250,
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
