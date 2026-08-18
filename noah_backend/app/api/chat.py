# pyrefly: ignore [missing-import]
from fastapi import APIRouter
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

from ..nlp.model_loader import load_models
from ..nlp.classifier import classify
from ..planner.planner import create_plan
from ..llm.response_generator import generate_response
from ..rag.retriever import retrieve

router = APIRouter(
    prefix="/chat",
    tags=["Noah"]
)


# Load models once when API starts
MODELS = load_models()


class ChatRequest(BaseModel):

    instruction: str


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
        location = entities_dict.get("location") or "CURRENT_LOCATION"
        offerhopper_data = call_offerhopper_mcp(items=items, location=location)
        plan["offerhopperData"] = offerhopper_data

    # --------------------------------------------------------
    # RAG: only document-grounded context from backend/rag is passed on.
    # --------------------------------------------------------

    rag_context = []
    if prediction.get("requires_rag"):
        rag_context = retrieve(instruction)

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
