"""Frozen wire contract for POST /api/chat.

The Flutter client (see noah_flutter_implementation_plan.md) dispatches on
``planner_actions``, ``entities.*``, ``response_mode``, ``offerhopperData`` and
``response``.  These tests pin the request shape, the exact response key set,
the value types and the label vocabularies so that retraining or refactoring
the NLP layer can never silently change the wire format.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app.tools.offerhopper as offerhopper_module
from app.planner.action_registry import AVAILABLE_ACTIONS


# --------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------

RESPONSE_KEYS = {
    "instruction",
    "domain", "intent", "sub_intent", "tool", "response_mode",
    "requires_memory", "requires_rag", "requires_recommendation",
    "workflow_type",
    "entity_product", "entity_merchant", "entity_brand", "entity_category",
    "entity_price_min", "entity_price_max", "entity_location", "entity_radius",
    "entity_loyalty_card",
    "planner_actions", "planner_action_count", "tool_sequence",
    "offerhopperData", "response",
}

STRING_OR_NULL = {
    "domain", "intent", "sub_intent", "tool", "response_mode", "workflow_type",
    "entity_product", "entity_merchant", "entity_brand", "entity_category",
    "entity_location", "entity_loyalty_card",
}
NUMBER_OR_NULL = {"entity_price_min", "entity_price_max", "entity_radius"}
BOOLEANS = {"requires_memory", "requires_rag", "requires_recommendation"}

DOMAINS = {"ACCOUNT", "CHAT", "KNOWLEDGE", "MERCHANT", "NAVIGATION", "OFFER",
           "PRODUCT", "RECOMMENDATION", "SHOPPING", "SYSTEM", "WALLET"}
RESPONSE_MODES = {"functional", "hybrid", "navigation", "text", "informational"}
WORKFLOW_TYPES = {"single", "single_step", "multi_step", "sequential"}
TOOLS = {"none", "wallet_tool", "offers_tool", "maps_tool", "merchant_api",
         "offerhopper_mcp", "rag_engine", "recommendation_engine", "firestore",
         "planner"}

PROBES = [
    "Show me my Payback barcode.",
    "Where can I buy frozen pizzas for less than €2 near me?",
    "Take me to the nearest Lidl.",
    "Find the cheapest basket for milk, eggs, and bread in Berlin.",
    "What is PayTo and how do loyalty points work?",
    "Hello! Who are you?",
    "asdkjhasd",
]


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    return TestClient(_app())


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _app():
    from app.main import app
    return app


@pytest.fixture(scope="module", autouse=True)
def _no_network(monkeypatch_module):
    """Keep the contract deterministic: never call OfferHopper or the LLM."""
    monkeypatch_module.setattr(
        offerhopper_module, "call_offerhopper_mcp", lambda **kwargs: {}
    )


@pytest.mark.parametrize("instruction", PROBES)
def test_response_key_set_is_frozen(client, instruction):
    payload = client.post("/api/chat", json={"instruction": instruction}).json()
    assert set(payload.keys()) == RESPONSE_KEYS, (
        f"wire contract changed for {instruction!r}: "
        f"added={set(payload) - RESPONSE_KEYS} removed={RESPONSE_KEYS - set(payload)}"
    )


@pytest.mark.parametrize("instruction", PROBES)
def test_response_types_are_frozen(client, instruction):
    payload = client.post("/api/chat", json={"instruction": instruction}).json()

    assert isinstance(payload["instruction"], str)
    assert isinstance(payload["response"], str) and payload["response"].strip()
    assert isinstance(payload["planner_actions"], list)
    assert all(isinstance(a, str) for a in payload["planner_actions"])
    assert isinstance(payload["tool_sequence"], list)
    assert all(isinstance(t, str) for t in payload["tool_sequence"])
    assert isinstance(payload["planner_action_count"], int)
    assert payload["offerhopperData"] is None or isinstance(payload["offerhopperData"], dict)

    for key in STRING_OR_NULL:
        assert payload[key] is None or isinstance(payload[key], str), key
    for key in NUMBER_OR_NULL:
        assert payload[key] is None or isinstance(payload[key], (int, float)), key
    for key in BOOLEANS:
        assert isinstance(payload[key], bool), key


@pytest.mark.parametrize("instruction", PROBES)
def test_label_vocabularies_are_frozen(client, instruction):
    payload = client.post("/api/chat", json={"instruction": instruction}).json()
    assert payload["domain"] in DOMAINS
    assert payload["response_mode"] in RESPONSE_MODES
    assert payload["workflow_type"] in WORKFLOW_TYPES
    assert payload["tool"] in TOOLS
    for action in payload["planner_actions"]:
        assert action in AVAILABLE_ACTIONS, f"unknown planner action {action}"
    for tool in payload["tool_sequence"]:
        assert tool in TOOLS, f"unknown tool {tool}"


@pytest.mark.parametrize("instruction", PROBES)
def test_plan_is_internally_consistent(client, instruction):
    """action_count and tool_sequence must always agree with planner_actions."""
    payload = client.post("/api/chat", json={"instruction": instruction}).json()
    assert payload["planner_action_count"] == len(payload["planner_actions"])
    assert len(payload["tool_sequence"]) == len(payload["planner_actions"])
    for action, tool in zip(payload["planner_actions"], payload["tool_sequence"]):
        assert AVAILABLE_ACTIONS[action]["tool"] == tool


def test_request_shape_is_frozen(client):
    """Only `instruction` is accepted; blank input keeps its legacy error body."""
    assert client.post("/api/chat", json={}).status_code == 422
    assert client.post("/api/chat", json={"instruction": "   "}).json() == {
        "error": "Instruction cannot be empty."
    }
    assert client.post("/api/chat", json={"instruction": "hi"}).status_code == 200
