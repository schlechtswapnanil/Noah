# pyrefly: ignore [missing-import]
from fastapi import APIRouter
# pyrefly: ignore [missing-import]
from typing import Optional

from pydantic import BaseModel

from ..nlp.model_loader import load_models
from ..nlp.classifier import classify
from ..planner.planner import create_plan
from ..llm.response_generator import generate_response
from ..rag.retriever import retrieve, SCOPE_CAPABILITIES, SCOPE_PAYTO

router = APIRouter(
    prefix="/chat",
    tags=["Noah"]
)


# Load models once when API starts
MODELS = load_models()


class DeviceLocation(BaseModel):
    """Where the user actually is, from the app's location services.

    Used only when the instruction itself does not name a place - "near me",
    "nearby", or no location at all. A city or postcode typed by the user
    always wins over the device position.
    """

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    postal_code: Optional[str] = None


class ChatRequest(BaseModel):

    instruction: str

    # Optional and additive: a request with only `instruction` behaves exactly
    # as before. Without it, "near me" resolves to a fixed default (Munich
    # city centre), which is silently wrong for everyone not in Munich.
    location: Optional[DeviceLocation] = None


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
    # NLP
    # --------------------------------------------------------

    prediction = classify(
        instruction,
        MODELS
    )

    # --------------------------------------------------------
    # PLANNER
    # --------------------------------------------------------

    plan = create_plan(
        prediction
    )

    # --------------------------------------------------------
    # OfferHopper MCP Integration
    # --------------------------------------------------------
    offerhopper_data = None
    if "offerhopper_mcp" in plan.get("tool_sequence", []):
        from ..tools.offerhopper import call_offerhopper_mcp
        entities_dict = prediction.get("entities", {})
        items = entities_dict.get("product") or instruction
        location, plan["location_basis"] = _resolve_location(
            entities_dict.get("location"), request.location)
        # An empty dict means the MCP call failed.  Emit null instead so the
        # Flutter dispatcher does not open an empty route card, and so the
        # response layer says the price service was unreachable rather than
        # promising results that never arrived.
        offerhopper_data = call_offerhopper_mcp(items=items, location=location) or None
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
        rag_context = retrieve(instruction, scope=scope)

    response_text = generate_response(
        instruction,
        plan,
        rag_context=rag_context,
        requires_rag=prediction.get("requires_rag", False),
    )

    # --------------------------------------------------------
    # FINAL JSON
    # --------------------------------------------------------

    entities = prediction.pop("entities")
    result = {
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

    return result
