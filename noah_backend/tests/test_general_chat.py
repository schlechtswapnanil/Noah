"""An unrecognised request is answered in conversation, not with a template.

The router still declines it - no action, no OfferHopper call, the empty-plan
wire shape - but the reply comes from the LLM under the constrained general
prompt, with any documentation that matched as context. The fixed clarifying
question is the floor: LLM switched off, LLM failing, or nothing better to say.
Greetings, thanks, the RAG routes and unresolved follow-ups are untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app.api.chat as chat_module
import app.llm.response_generator as generator_module
from app.llm.provider import DEFAULT_TEMPERATURE, GROUNDED_TEMPERATURE
from app.llm.response_generator import (CLARIFY, CONVERSATIONAL, FOLLOW_UP_CLARIFY,
                                        GENERAL_PROMPT, SYSTEM_PROMPT)
from app.main import app

client = TestClient(app)

UNRECOGNISED_EN = "Can you get my live location?"
UNRECOGNISED_DE = "Wo bin ich gerade?"
OFF_TOPIC = "What is the capital of France?"
FAQ_CHUNK = ("[PayTo FAQ Web]: PayTO uses your location only to show nearby "
             "verified merchants and relevant QR codes.")


@pytest.fixture
def llm(monkeypatch):
    """Capture every LLM call and answer with a fixed sentence."""
    calls = []

    def fake(system_prompt, user_prompt, **kwargs):
        calls.append({"system": system_prompt, "user": user_prompt, **kwargs})
        return "Sure - here is a reply."

    monkeypatch.setattr(generator_module, "generate_text", fake)
    return calls


@pytest.fixture
def retrieval(monkeypatch):
    """Pin retrieval so the test does not depend on the live FAQ cache."""
    calls = []

    def fake(query, **kwargs):
        calls.append({"query": query, **kwargs})
        return [FAQ_CHUNK] if "location" in query.lower() else []

    monkeypatch.setattr(chat_module, "retrieve", fake)
    return calls


def _ask(instruction, **extra):
    response = client.post("/api/chat", json={"instruction": instruction, **extra})
    assert response.status_code == 200
    return response.json()


def _assert_empty_plan(payload):
    assert payload["planner_actions"] == [] and payload["tool_sequence"] == []
    assert payload["planner_action_count"] == 0
    assert payload["domain"] == "CHAT" and payload["sub_intent"] == "UNKNOWN"
    assert payload["response_mode"] == "text" and payload["offerhopperData"] is None


def test_unrecognised_request_is_answered_by_the_llm(llm, retrieval, offline):
    payload = _ask(OFF_TOPIC)
    _assert_empty_plan(payload)
    assert payload["response"] == "Sure - here is a reply."
    assert offline == []                                   # no OfferHopper call

    assert len(llm) == 1
    call = llm[0]
    assert call["system"] == GENERAL_PROMPT                # not the action prompt
    assert f"<user_request>\n{OFF_TOPIC}\n</user_request>" in call["user"]
    assert call["user"].rstrip().endswith("Reply in English.")
    assert "DOCUMENT CONTEXT" not in call["user"]
    assert call["temperature"] == DEFAULT_TEMPERATURE


def test_unrecognised_request_with_matching_docs_is_grounded(llm, retrieval):
    payload = _ask(UNRECOGNISED_EN)
    _assert_empty_plan(payload)
    assert payload["response"] == "Sure - here is a reply."

    # retrieval ran unscoped - both shelves - for the declined request
    assert retrieval == [{"query": UNRECOGNISED_EN}]
    call = llm[0]
    assert call["system"] == GENERAL_PROMPT
    assert "DOCUMENT CONTEXT:\n" + FAQ_CHUNK in call["user"]
    assert call["temperature"] == GROUNDED_TEMPERATURE


def test_german_unrecognised_request_asks_for_a_german_reply(llm, retrieval):
    payload = _ask(UNRECOGNISED_DE)
    _assert_empty_plan(payload)
    assert llm[0]["user"].rstrip().endswith("Antworte auf Deutsch, mit du.")


def test_clarifying_question_when_the_llm_is_unavailable(retrieval):
    """The default offline fixture makes every LLM call raise."""
    assert _ask(OFF_TOPIC)["response"] == CLARIFY["en"]
    assert _ask(UNRECOGNISED_EN)["response"] == CLARIFY["en"]
    assert _ask(UNRECOGNISED_DE)["response"] == CLARIFY["de"]


def test_clarifying_question_when_the_llm_returns_nothing(monkeypatch, retrieval):
    monkeypatch.setattr(generator_module, "generate_text", lambda *a, **k: "")
    assert _ask(OFF_TOPIC)["response"] == CLARIFY["en"]


def test_general_chat_can_be_switched_off(llm, retrieval, monkeypatch):
    monkeypatch.setenv("NOAH_GENERAL_CHAT", "0")
    payload = _ask(OFF_TOPIC)
    _assert_empty_plan(payload)
    assert payload["response"] == CLARIFY["en"]
    assert llm == []                                       # no quota spent


def test_greetings_and_small_talk_keep_their_templates(llm):
    """Only UNKNOWN falls through. The other zero-action routes are recognised
    conversation and keep their fixed replies, so they cost no quota."""
    for instruction in ("Hello!", "Hello! Who are you?", "Thanks a lot", "Bye!"):
        payload = _ask(instruction)
        assert payload["planner_actions"] == [], instruction
        assert payload["sub_intent"] in CONVERSATIONAL and payload["sub_intent"] != "UNKNOWN", instruction
        assert payload["response"] == CONVERSATIONAL[payload["sub_intent"]]["en"], instruction
    assert llm == []


def test_unresolved_follow_up_still_asks_which_store(llm):
    payload = _ask("Take me there")
    assert payload["planner_actions"] == []
    assert payload["response"] == FOLLOW_UP_CLARIFY["navigation"]["en"]
    assert llm == []


def test_rag_routes_keep_the_grounded_action_prompt(llm):
    """A recognised PayTo question goes through the RAG route as before: the
    action prompt, a scoped document context, the grounded temperature."""
    payload = _ask("What is PayTo and how do loyalty points work?")
    assert payload["requires_rag"] is True
    assert "rag_engine" in payload["tool_sequence"]
    assert payload["response"] == "Sure - here is a reply."

    assert len(llm) == 1
    assert llm[0]["system"] == SYSTEM_PROMPT
    assert llm[0]["system"] != GENERAL_PROMPT
    assert "DOCUMENT CONTEXT:" in llm[0]["user"]
    assert llm[0]["temperature"] == GROUNDED_TEMPERATURE


def test_rag_route_without_documents_still_declines_to_improvise(llm, monkeypatch):
    """No matching passage: the RAG route says so and never reaches the LLM.
    That rule predates the general fallthrough and is unchanged by it."""
    from app.llm.response_generator import NO_DOCS
    monkeypatch.setattr(chat_module, "retrieve", lambda query, **kwargs: [])
    payload = _ask("What is PayTo and how do loyalty points work?")
    assert payload["requires_rag"] is True
    assert payload["response"] == NO_DOCS["en"]
    assert llm == []
