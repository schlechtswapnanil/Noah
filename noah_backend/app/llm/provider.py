"""Provider boundary for Noah's optional natural-language response layer.

A chain, not a single model. Every free tier caps per day and per model, and
the 429 that ended the day used to drop every reply to the deterministic
template. Each call now tries, in order:

1. the primary Groq model (``LLM_MODEL``);
2. a second Groq model (``LLM_FALLBACK_MODEL``). Groq publishes its limits per
   model and a 429 names the model, so this one has a daily budget of its own;
3. Gemini (``GEMINI_MODEL``) when ``GEMINI_API_KEY`` is set: a different
   provider on a different clock (its daily quota resets at midnight Pacific).

A step that answers 429 is skipped for as long as its ``retry-after`` header
says; any other failure sidelines it briefly, so a dead key is not retried on
every request. The callers' contract is unchanged: a string, or an exception
when every step failed, which response_generator turns into the deterministic
reply. ``LLM_PROVIDER=gemini`` puts Gemini first; any value other than
``groq`` or ``gemini`` is refused.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# Action replies get room to sound natural; knowledge answers are quoted from
# retrieved text and run cooler, because sampling freedom there turns into
# invented policy details.
DEFAULT_TEMPERATURE = 0.5
GROUNDED_TEMPERATURE = 0.2

# Groq retires models without notice: qwen/qwen3.6-27b answered on
# 2026-09-14 and was a 404 on 2026-09-21. When the primary disappears the
# chain carries on with the fallback, but check `Groq().models.list()` and
# move the defaults on.
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
# A different family with a 200K-token daily budget of its own. It reasons
# before it answers, which costs completion tokens; the Groq step allows for
# that below.
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

# How long a failed step is skipped. A 429 with a retry-after header uses that
# value instead, capped so a "come back tomorrow" does not silence a model
# whose window may in fact roll over sooner.
RATE_LIMIT_COOLDOWN = 60.0
FAILURE_COOLDOWN = 30.0
MAX_COOLDOWN = 3600.0

Caller = Callable[[str, str, float, int], str]


class Step(NamedTuple):
    name: str
    call: Caller


_cooldown_until: Dict[str, float] = {}
_lock = threading.Lock()


def reset_cooldowns() -> None:
    with _lock:
        _cooldown_until.clear()


# Completion tokens a reasoning model may spend thinking before it answers.
# Measured on gpt-oss-20b at "low": a few dozen for a one-line reply; the
# allowance keeps a five-item recommendation list from being cut off.
REASONING_ALLOWANCE = 120


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def _groq_reasoning_effort(model: str) -> str:
    """Groq's qwen3 models accept ``"none"``; its gpt-oss models accept only
    low/medium/high and reject ``"none"``."""
    return "low" if _is_reasoning_model(model) else "none"


def _groq_request(model: str, system_prompt: str, user_prompt: str,
                  temperature: float, max_tokens: int) -> dict:
    """The keyword arguments for one chat completion on Groq."""
    return {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_completion_tokens": max_tokens + (REASONING_ALLOWANCE if _is_reasoning_model(model) else 0),
        "reasoning_effort": _groq_reasoning_effort(model),
    }


def _groq(model: str) -> Step:
    def call(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is missing for Groq.")
        from groq import Groq
        completion = Groq(api_key=api_key).chat.completions.create(
            **_groq_request(model, system_prompt, user_prompt, temperature, max_tokens))
        return (completion.choices[0].message.content or "").strip()
    return Step(f"groq:{model}", call)


def _gemini(model: str) -> Step:
    def call(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        from .gemini import generate_text as gemini_generate_text
        return gemini_generate_text(system_prompt, user_prompt, temperature=temperature,
                                    max_tokens=max_tokens, model=model)
    return Step(f"gemini:{model}", call)


def provider_chain() -> List[Step]:
    """The steps to try, in order. Read from the environment on every call so
    a change at deploy time, or a test's monkeypatch, takes effect at once."""
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in ("groq", "gemini"):
        raise RuntimeError(f"Unsupported LLM_PROVIDER {provider!r}: use groq or gemini.")

    primary = os.getenv("LLM_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    fallback = os.getenv("LLM_FALLBACK_MODEL", DEFAULT_GROQ_FALLBACK_MODEL).strip()
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    groq_steps = [_groq(primary)]
    if fallback and fallback != primary:
        groq_steps.append(_groq(fallback))
    # Without a key, Gemini is left out of the chain rather than failing on
    # every request - unless it was asked for as the primary, in which case
    # the missing key should be visible in the log.
    gemini_steps = ([_gemini(gemini_model)]
                    if os.getenv("GEMINI_API_KEY") or provider == "gemini" else [])

    return gemini_steps + groq_steps if provider == "gemini" else groq_steps + gemini_steps


def _status_and_headers(error: BaseException):
    """HTTP status and headers of a provider error, if it carries a response.
    Covers the Groq SDK's APIStatusError and httpx's HTTPStatusError alike."""
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None), getattr(response, "headers", None) or {}


def _cooldown_for(error: BaseException) -> float:
    status, headers = _status_and_headers(error)
    if status != 429:
        return FAILURE_COOLDOWN
    try:
        wait = float(headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return RATE_LIMIT_COOLDOWN
    return min(max(wait, 1.0), MAX_COOLDOWN)


def _describe(error: BaseException) -> str:
    status, _ = _status_and_headers(error)
    text = " ".join(str(error).split())
    return f"{type(error).__name__}{f' {status}' if status else ''}: {text[:160]}"


def generate_text(system_prompt: str, user_prompt: str,
                  temperature: float = DEFAULT_TEMPERATURE,
                  max_tokens: int = 220) -> str:
    """One reply from the first step in the chain that can give one."""
    skipped: List[str] = []
    for step in provider_chain():
        now = time.monotonic()
        with _lock:
            until = _cooldown_until.get(step.name, 0.0)
        if until > now:
            skipped.append(f"{step.name}: skipped for another {until - now:.0f}s")
            continue
        try:
            text = step.call(system_prompt, user_prompt, temperature, max_tokens)
        except Exception as error:
            wait = _cooldown_for(error)
            with _lock:
                _cooldown_until[step.name] = time.monotonic() + wait
            skipped.append(f"{step.name}: {_describe(error)}")
            logger.warning("LLM step %s failed, skipping it for %.0fs: %s",
                           step.name, wait, _describe(error))
            continue
        if text:
            if skipped:
                logger.info("LLM reply from %s after: %s", step.name, "; ".join(skipped))
            return text
        skipped.append(f"{step.name}: empty reply")
    raise RuntimeError("No LLM step produced a reply: " + "; ".join(skipped))
