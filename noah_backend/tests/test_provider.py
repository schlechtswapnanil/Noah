"""The LLM provider chain: primary Groq -> second Groq model -> Gemini.

Nothing here touches the network. Steps are stubbed at the chain level, the
Gemini adapter's HTTP call is stubbed at the module level, and the Groq SDK's
own exception type is used for the 429 so the cooldown logic is tested
against what the SDK really raises.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest
from fastapi.testclient import TestClient

import app.llm.gemini as gemini
import app.llm.provider as provider
import app.llm.response_generator as generator_module
from app.llm.provider import Step, generate_text, provider_chain


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    provider.reset_cooldowns()
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK_MODEL",
                 "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_API_URL"):
        monkeypatch.delenv(name, raising=False)
    yield
    provider.reset_cooldowns()


def _response(status, headers=None):
    request = httpx.Request("POST", "https://example.test/v1")
    return httpx.Response(status, headers=headers or {}, request=request, json={})


def _groq_429(retry_after=None):
    from groq import RateLimitError
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return RateLimitError("Rate limit reached for model `x` on tokens per day (TPD)",
                          response=_response(429, headers), body=None)


def _step(name, result=None, error=None, calls=None):
    def call(system_prompt, user_prompt, temperature, max_tokens):
        if calls is not None:
            calls.append(name)
        if error is not None:
            raise error
        return result
    return Step(name, call)


# --------------------------------------------------------------------------
# Chain composition
# --------------------------------------------------------------------------

def _names():
    return [s.name for s in provider_chain()]


def test_default_chain_is_two_groq_models_and_no_gemini_without_a_key():
    assert _names() == ["groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-20b"]


def test_gemini_joins_the_chain_last_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    assert _names() == ["groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-20b",
                        "gemini:gemini-2.5-flash-lite"]


def test_chain_follows_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "qwen/qwen3.6-27b")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    assert _names() == ["groq:openai/gpt-oss-20b", "groq:qwen/qwen3.6-27b",
                        "gemini:gemini-2.5-flash"]


def test_empty_fallback_model_disables_the_second_groq_step(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "")
    assert _names() == ["groq:qwen/qwen3.8-27b"]


def test_fallback_equal_to_primary_is_not_tried_twice(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "qwen/qwen3.8-27b")
    assert _names() == ["groq:qwen/qwen3.8-27b"]


def test_gemini_first_when_it_is_the_configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    assert _names()[0] == "gemini:gemini-2.5-flash-lite"
    assert _names()[1:] == ["groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-20b"]


def test_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        provider_chain()


def test_groq_request_matches_the_model_family():
    qwen = provider._groq_request("qwen/qwen3.8-27b", "S", "U", 0.5, 220)
    assert qwen["reasoning_effort"] == "none"
    assert qwen["max_completion_tokens"] == 220
    assert qwen["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

    oss = provider._groq_request("openai/gpt-oss-20b", "S", "U", 0.2, 220)
    assert oss["reasoning_effort"] == "low"                 # "none" is rejected by gpt-oss
    assert oss["max_completion_tokens"] == 220 + provider.REASONING_ALLOWANCE
    assert oss["temperature"] == 0.2


# --------------------------------------------------------------------------
# Falling through the chain
# --------------------------------------------------------------------------

def test_rate_limited_primary_falls_to_the_second_model(monkeypatch):
    calls = []
    chain = [_step("groq:primary", error=_groq_429(300), calls=calls),
             _step("groq:second", result="from the second model", calls=calls)]
    monkeypatch.setattr(provider, "provider_chain", lambda: chain)

    assert generate_text("sys", "user") == "from the second model"
    assert calls == ["groq:primary", "groq:second"]

    # The primary is not asked again while its retry-after runs.
    assert generate_text("sys", "user") == "from the second model"
    assert calls == ["groq:primary", "groq:second", "groq:second"]
    remaining = provider._cooldown_until["groq:primary"] - time.monotonic()
    assert 290 < remaining <= 300


def test_cooldown_expires(monkeypatch):
    calls = []
    chain = [_step("groq:primary", error=_groq_429(1), calls=calls),
             _step("groq:second", result="second", calls=calls)]
    monkeypatch.setattr(provider, "provider_chain", lambda: chain)
    generate_text("sys", "user")
    provider._cooldown_until["groq:primary"] = time.monotonic() - 1   # window over
    generate_text("sys", "user")
    assert calls == ["groq:primary", "groq:second", "groq:primary", "groq:second"]


def test_retry_after_is_bounded():
    assert provider._cooldown_for(_groq_429(99999)) == provider.MAX_COOLDOWN
    assert provider._cooldown_for(_groq_429(0)) == 1.0
    assert provider._cooldown_for(_groq_429()) == provider.RATE_LIMIT_COOLDOWN
    assert provider._cooldown_for(_groq_429("soon")) == provider.RATE_LIMIT_COOLDOWN


def test_other_failures_get_the_short_cooldown(monkeypatch):
    chain = [_step("gemini:x", error=httpx.HTTPStatusError(
                 "401", request=httpx.Request("POST", "https://x"), response=_response(401))),
             _step("groq:second", result="ok")]
    monkeypatch.setattr(provider, "provider_chain", lambda: chain)
    assert generate_text("sys", "user") == "ok"
    remaining = provider._cooldown_until["gemini:x"] - time.monotonic()
    assert 25 < remaining <= provider.FAILURE_COOLDOWN
    assert provider._cooldown_for(RuntimeError("key missing")) == provider.FAILURE_COOLDOWN


def test_empty_reply_moves_to_the_next_step(monkeypatch):
    chain = [_step("groq:primary", result=""), _step("groq:second", result="filled")]
    monkeypatch.setattr(provider, "provider_chain", lambda: chain)
    assert generate_text("sys", "user") == "filled"
    assert "groq:primary" not in provider._cooldown_until   # empty is not a failure


def test_every_step_failing_raises_with_the_reasons(monkeypatch):
    chain = [_step("groq:primary", error=_groq_429(60)),
             _step("groq:second", error=RuntimeError("LLM_API_KEY is missing for Groq."))]
    monkeypatch.setattr(provider, "provider_chain", lambda: chain)
    with pytest.raises(RuntimeError) as excinfo:
        generate_text("sys", "user")
    message = str(excinfo.value)
    assert "groq:primary: RateLimitError 429" in message
    assert "groq:second: RuntimeError: LLM_API_KEY is missing" in message


def test_rate_limited_primary_is_invisible_to_the_api(monkeypatch):
    """Through POST /api/chat: the primary is at its daily cap, the second
    model answers, the reply is conversational rather than the template."""
    from app.main import app
    chain = [_step("groq:primary", error=_groq_429(300)),
             _step("groq:second", result="Here's your Payback barcode - hold it up to the scanner.")]
    monkeypatch.setattr(provider, "provider_chain", lambda: chain)
    # conftest stubs the generator's LLM out; put the real chain back for this one
    monkeypatch.setattr(generator_module, "generate_text", provider.generate_text)

    payload = TestClient(app).post("/api/chat", json={"instruction": "Show me my Payback barcode."}).json()
    assert "DISPLAY_BARCODE" in payload["planner_actions"]
    assert payload["response"] == "Here's your Payback barcode - hold it up to the scanner."


# --------------------------------------------------------------------------
# Gemini adapter
# --------------------------------------------------------------------------

def _gemini_ok(text):
    return httpx.Response(200, request=httpx.Request("POST", "https://x"), json={
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
    })


def test_gemini_request_shape_and_ai_studio_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    seen = {}

    def fake_post(url, body, api_key):
        seen.update(url=url, body=body, api_key=api_key)
        return _gemini_ok("  Hallo!  ")

    monkeypatch.setattr(gemini, "_post", fake_post)
    assert gemini.generate_text("SYS", "USER", temperature=0.2, max_tokens=99) == "Hallo!"
    assert seen["url"] == ("https://generativelanguage.googleapis.com/v1beta/models/"
                           "gemini-2.5-flash-lite:generateContent")
    assert seen["api_key"] == "AIza-test-key"
    assert seen["body"]["system_instruction"] == {"parts": [{"text": "SYS"}]}
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    assert seen["body"]["generationConfig"] == {
        "temperature": 0.2, "maxOutputTokens": 99, "thinkingConfig": {"thinkingBudget": 0}}


def test_gemini_vertex_express_key_uses_the_aiplatform_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test-key")
    seen = {}
    monkeypatch.setattr(gemini, "_post", lambda url, body, key: seen.update(url=url) or _gemini_ok("ok"))
    gemini.generate_text("s", "u", model="gemini-2.5-flash")
    assert seen["url"] == ("https://aiplatform.googleapis.com/v1/publishers/google/models/"
                           "gemini-2.5-flash:generateContent")


def test_gemini_url_override_and_thinking_config_only_for_2_5(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setenv("GEMINI_API_URL", "https://proxy.test/{model}")
    seen = {}
    monkeypatch.setattr(gemini, "_post", lambda url, body, key: seen.update(url=url, body=body) or _gemini_ok("ok"))
    gemini.generate_text("s", "u", model="gemini-3.6-flash")
    assert seen["url"] == "https://proxy.test/gemini-3.6-flash"
    assert "thinkingConfig" not in seen["body"]["generationConfig"]


def test_gemini_http_error_propagates_with_the_status(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setattr(gemini, "_post", lambda url, body, key: _response(401))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        gemini.generate_text("s", "u")
    assert excinfo.value.response.status_code == 401
    assert provider._cooldown_for(excinfo.value) == provider.FAILURE_COOLDOWN


def test_gemini_429_retry_after_is_honoured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setattr(gemini, "_post", lambda url, body, key: _response(429, {"retry-after": "42"}))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        gemini.generate_text("s", "u")
    assert provider._cooldown_for(excinfo.value) == 42.0


def test_gemini_blocked_or_empty_reply_is_empty_string():
    assert gemini.text_of({}) == ""
    assert gemini.text_of({"candidates": [{"finishReason": "SAFETY"}]}) == ""
    assert gemini.text_of({"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}) == "ab"


def test_gemini_without_a_key_is_an_error():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini.generate_text("s", "u")
