"""Provider boundary for Noah's optional natural-language response layer."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# Action replies get room to sound natural; knowledge answers are quoted from
# retrieved text and run cooler, because sampling freedom there turns into
# invented policy details.
DEFAULT_TEMPERATURE = 0.5
GROUNDED_TEMPERATURE = 0.2


def generate_text(system_prompt: str, user_prompt: str,
                  temperature: float = DEFAULT_TEMPERATURE,
                  max_tokens: int = 220) -> str:
    if os.getenv("LLM_PROVIDER", "groq").lower() != "groq":
        raise RuntimeError("Only the configured Groq provider is supported.")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is missing for Groq.")
    from groq import Groq
    completion = Groq(api_key=api_key).chat.completions.create(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.6-27b"),
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature, max_completion_tokens=max_tokens, reasoning_effort="none",
    )
    return completion.choices[0].message.content.strip()
