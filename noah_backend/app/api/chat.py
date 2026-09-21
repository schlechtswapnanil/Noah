# pyrefly: ignore [missing-import]
import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..nlp.model_loader import UNKNOWN_ROUTE, load_models
from ..nlp.classifier import RESPONSE_FIELDS, classify, decline
from ..nlp.follow_up import (ResultQuestion, Rewritten, Unresolved, build_context,
                             resolve_follow_up)
from ..planner.planner import create_plan
from ..llm.response_generator import generate_response
from ..rag.retriever import retrieve, SCOPE_CAPABILITIES, SCOPE_PAYTO

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Noah"]
)


# Load models once when API starts
MODELS = load_models()

# A question about the results just shown ("which one is cheapest?") is
# answered from the carried context: no classifier, no OfferHopper. It goes
# out in the empty-plan shape the client already renders as plain text.
RESULT_QUESTION_ROUTE = {
    **UNKNOWN_ROUTE, "intent": "FOLLOW_UP", "sub_intent": "RESULT_QUESTION",
    "requires_memory": True,
}


class DeviceLocation(BaseModel):
    """Where the user actually is, from the app's location services.

    Used only when the instruction itself does not name a place - "near me",
    "nearby", or no location at all. A city or postcode typed by the user
    always wins over the device position.
    """

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    postal_code: Optional[str] = None


class HistoryEntities(BaseModel):
    """The previous response's ``entity_*`` fields, without the prefix."""

    product: Optional[str] = None
    merchant: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    location: Optional[str] = None
    radius: Optional[float] = None
    loyalty_card: Optional[str] = None


class HistoryResult(BaseModel):
    """One product from a previous turn's ``offerhopperData``."""

    name: Optional[str] = None
    store: Optional[str] = None
    price: Optional[float] = None


class HistoryStore(BaseModel):
    """One store from a previous turn's ``offerhopperData``."""

    name: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class HistoryTurn(BaseModel):
    """One earlier exchange, as the client stored it from the response.

    Only `instruction` is required. `results`, `stores` and `share_url` are a
    compact summary of that turn's ``offerhopperData`` (name, store, price;
    name, address, position) - never the payload itself. Unknown keys are
    ignored, oversized lists and texts are truncated, wrong types are a 422.
    """

    instruction: str
    response: Optional[str] = None
    planner_actions: Optional[List[str]] = None
    entities: Optional[HistoryEntities] = None
    results: Optional[List[HistoryResult]] = None
    share_url: Optional[str] = None
    stores: Optional[List[HistoryStore]] = None


class ChatRequest(BaseModel):

    instruction: str

    # Optional and additive: a request with only `instruction` behaves exactly
    # as before. Without it, "near me" resolves to a fixed default (Munich
    # city centre), which is silently wrong for everyone not in Munich.
    location: Optional[DeviceLocation] = None

    # Optional and additive: the previous turns, oldest first, newest last.
    # With it, "take me there" after a basket search becomes a route to the
    # store that search found. Without it every call is independent, as before.
    history: Optional[List[HistoryTurn]] = None


def _resolve_location(text_location, device: Optional[DeviceLocation]):
    """Pick the place OfferHopper should search from, and say where it came from.

    Returns (location string, basis). A place named in the text ("in Berlin",
    "10178") is authoritative. Otherwise the device position is used if the
    app sent one; failing that, the fixed default applies and the basis says
    so, so the reply can tell the user the area was assumed.
    """
    if text_location and text_location != "CURRENT_LOCATION":
        return text_location, "named in request"
    if device is not None:
        if device.latitude is not None and device.longitude is not None:
            return f"{device.latitude:.5f},{device.longitude:.5f}", "device GPS"
        if device.postal_code:
            return device.postal_code.strip(), "device postal code"
    return "CURRENT_LOCATION", "assumed default area (Munich city centre) - no device location supplied"


def _payload(instruction: str, prediction: dict, plan: dict, offerhopper_data, response_text: str) -> dict:
    """The frozen 24-key wire shape (tests/test_response_contract.py).

    `instruction` echoes what the user typed, never the resolved sentence.
    """
    entities = prediction.pop("entities")
    return {
        "instruction": instruction,
        **prediction,
        **{f"entity_{key}": value for key, value in entities.items()},

        "planner_actions":
            plan["planner_actions"],

        "planner_action_count":
            plan["planner_action_count"],

        "tool_sequence":
            plan["tool_sequence"],

        "offerhopperData":
            offerhopper_data,

        "response":
            response_text,
    }


@router.post("")
def chat(
    request: ChatRequest
):

    instruction = request.instruction.strip()

    if not instruction:

        return {
            "error": "Instruction cannot be empty."
        }

    # --------------------------------------------------------
    # Conversation context: a follow-up becomes the sentence it stands for
    # --------------------------------------------------------

    context = build_context(request.history)
    resolved = resolve_follow_up(instruction, context)

    if isinstance(resolved, ResultQuestion):
        logger.info("follow-up %r answered from the previous results", instruction)
        prediction = {field: RESULT_QUESTION_ROUTE[field] for field in RESPONSE_FIELDS}
        prediction["planner_actions"] = []
        prediction["tool_sequence"] = []
        prediction["entities"] = context.entities()
        plan = create_plan(prediction)
        plan["follow_up"] = {"kind": "result_question", "language": resolved.language}
        response_text = generate_response(instruction, plan, context=context)
        return _payload(instruction, prediction, plan, None, response_text)

    if isinstance(resolved, Unresolved):
        # A follow-up with nothing to resolve against ("take me there" and no
        # history): ask, rather than launch maps with no destination.
        logger.info("follow-up %r (%s) has no context to resolve against; asking",
                    instruction, resolved.family)
        prediction = decline(instruction)
        plan = create_plan(prediction)
        plan["follow_up"] = {"kind": "unresolved", "family": resolved.family,
                             "language": resolved.language}
        response_text = generate_response(instruction, plan, context=context)
        return _payload(instruction, prediction, plan, None, response_text)

    text = resolved.text
    if isinstance(resolved, Rewritten):
        logger.info("follow-up %r resolved as %r (%s)", instruction, text, resolved.family)

    # --------------------------------------------------------
    # NLP
    # --------------------------------------------------------

    prediction = classify(
        text,
        MODELS
    )

    # --------------------------------------------------------
    # PLANNER
    # --------------------------------------------------------

    plan = create_plan(
        prediction
    )
    if isinstance(resolved, Rewritten):
        plan["follow_up"] = {"kind": "rewritten", "resolved": text,
                             "language": resolved.language}

    # --------------------------------------------------------
    # OfferHopper MCP Integration
    # --------------------------------------------------------
    offerhopper_data = None
    if "offerhopper_mcp" in plan.get("tool_sequence", []):
        from ..tools.offerhopper import call_offerhopper_mcp
        entities_dict = prediction.get("entities", {})
        items = entities_dict.get("product")
        location, plan["location_basis"] = _resolve_location(
            entities_dict.get("location"), request.location)

        if not items:
            # "Recommend me something" with no kind of thing named: there is
            # nothing to search for, and sending the whole sentence produces
            # error_invalid_list. Ask instead of guessing.
            offerhopper_data = None
            plan["offerhopper_failure"] = "no_terms"
        else:
            result = call_offerhopper_mcp(items=items, location=location)
            if result and result.get("success") is not False:
                offerhopper_data = result
            else:
                # offerhopperData stays null for every failure so the client
                # never opens a route card on an error object; the *kind* of
                # failure rides on the plan so the reply can be honest about it:
                # nothing matched versus the service being unreachable.
                offerhopper_data = None
                plan["offerhopper_failure"] = "no_match" if result else "unavailable"
                if result:
                    plan["offerhopper_error"] = result.get("error")
        plan["offerhopperData"] = offerhopper_data

    # --------------------------------------------------------
    # RAG: only document-grounded context from backend/rag is passed on.
    # --------------------------------------------------------

    # Questions about Noah itself are answered from the capability document;
    # questions about PayTo from PayTo's own documentation. Searching one pool
    # for both lets an unrelated FAQ fragment outrank the real answer.
    rag_context = []
    if prediction.get("requires_rag"):
        actions = plan.get("planner_actions", [])
        scope = SCOPE_CAPABILITIES if "PROVIDE_APP_HELP" in actions else SCOPE_PAYTO
        rag_context = retrieve(text, scope=scope)
    elif not plan["planner_actions"] and prediction.get("sub_intent") == "UNKNOWN":
        # The router declined. Before the reply is left to the LLM, check
        # whether the documentation answers it: "does PayTo track my
        # location?" is a PayTo question whether or not the classifier
        # recognised it as one. Both shelves, at the stricter unscoped floor,
        # so a loose match does not masquerade as an answer.
        rag_context = retrieve(text)

    response_text = generate_response(
        instruction,
        plan,
        rag_context=rag_context,
        requires_rag=prediction.get("requires_rag", False),
        context=context,
    )

    # --------------------------------------------------------
    # FINAL JSON
    # --------------------------------------------------------

    return _payload(instruction, prediction, plan, offerhopper_data, response_text)
