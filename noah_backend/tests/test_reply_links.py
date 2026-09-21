"""A reply may only carry links the model was given.

Observed live on 2026-09-21: a navigation reply ended with an invented
"https://maps.app.link/REWE_Hamburg_route". The OfferHopper share URL and a
previous turn's share URL are the only links a reply should ever contain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app.llm.response_generator as generator_module
from app.llm.response_generator import _only_given_links
from app.main import app

client = TestClient(app)
SHARE = "https://offerhopper.ai/s/VcEZiHHo"


def test_invented_link_is_removed_and_given_link_is_kept():
    prompt = f"LIVE RESULTS ... Map link: {SHARE}\n"
    reply = f"Cheapest at REWE.\nhttps://maps.app.link/REWE_route\n\n{SHARE}"
    assert _only_given_links(reply, prompt) == f"Cheapest at REWE.\n\n{SHARE}"


def test_link_with_trailing_punctuation_is_still_recognised():
    prompt = f"Map link: {SHARE}"
    assert _only_given_links(f"Here you go: {SHARE}.", prompt) == f"Here you go: {SHARE}."
    # an invented link goes together with its trailing punctuation
    assert _only_given_links("See https://example.com/x.", "no links here") == "See"


def test_reply_without_links_is_untouched():
    text = "Got it - heading to REWE in Hamburg.\nStarting navigation now."
    assert _only_given_links(text, "prompt without links") == text


@pytest.fixture
def llm(monkeypatch):
    def fake(system_prompt, user_prompt, **kwargs):
        return ("Got it - heading to REWE in Hamburg.\n"
                "https://maps.app.link/REWE_Hamburg_route")
    monkeypatch.setattr(generator_module, "generate_text", fake)


def test_navigation_reply_loses_the_invented_map_link(llm):
    history = [{"instruction": "Find the cheapest basket for milk in Hamburg",
                "planner_actions": ["FIND_CHEAPEST_BASKET"],
                "entities": {"product": "Milk", "location": "Hamburg"},
                "results": [{"name": "Milch", "store": "REWE", "price": 0.99}],
                "share_url": SHARE, "stores": [{"name": "REWE"}]}]
    payload = client.post("/api/chat", json={"instruction": "Take me there", "history": history}).json()
    assert "PLAN_ROUTE" in payload["planner_actions"]
    assert payload["response"] == "Got it - heading to REWE in Hamburg."


def test_result_question_keeps_the_previous_share_url(monkeypatch):
    monkeypatch.setattr(generator_module, "generate_text",
                        lambda *a, **k: f"Milch is cheapest at 0.99.\n{SHARE}\nhttps://evil.example/x")
    history = [{"instruction": "Find the cheapest basket for milk in Hamburg",
                "planner_actions": ["FIND_CHEAPEST_BASKET"],
                "entities": {"product": "Milk", "location": "Hamburg"},
                "results": [{"name": "Milch", "store": "REWE", "price": 0.99}],
                "share_url": SHARE, "stores": [{"name": "REWE"}]}]
    payload = client.post("/api/chat", json={"instruction": "Which one is cheapest?", "history": history}).json()
    assert payload["response"] == f"Milch is cheapest at 0.99.\n{SHARE}"
