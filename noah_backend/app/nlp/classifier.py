"""Route a user instruction to a PayTo action plan.

One model predicts ``(sub_intent, planner_actions)``; the rest of the response
is looked up from that route.  Two properties follow:

* the label tuple is always self-consistent - ``domain``, ``intent``, ``tool``,
  ``response_mode`` and ``tool_sequence`` cannot contradict the plan;
* when the winning route is below the calibrated confidence floor, Noah routes
  to UNKNOWN and asks instead of acting.  Firing the wrong action costs the user
  an opened card, a launched map or a paid MCP call; a clarifying question costs
  a tap.
"""

from __future__ import annotations

import logging
import re

from .entity_extractor import extract_entities
from .model_loader import UNKNOWN_ROUTE
from .routing import score_routes

logger = logging.getLogger(__name__)

# "and then", "danach" - an explicit ordering marker is what separates a
# sequential plan from a merely multi-step one.  It is a property of the
# sentence, not of the route, so it is derived per request rather than read
# from the registry (which stores whichever value the route's first row had).
_SEQUENCE_MARKER = re.compile(
    r"\b(?:and\s+then|then|after\s+that|afterwards|"
    r"und\s+dann|dann|danach|anschließend|anschliessend)\b",
    re.IGNORECASE,
)

RESPONSE_FIELDS = [
    "domain", "intent", "sub_intent", "tool", "response_mode",
    "requires_memory", "requires_rag", "requires_recommendation",
    "workflow_type", "planner_actions", "planner_action_count", "tool_sequence",
]


def _workflow_type(instruction: str, actions: list) -> str:
    if len(actions) > 1:
        return "sequential" if _SEQUENCE_MARKER.search(instruction or "") else "multi_step"
    return "single_step"


def _unknown() -> dict:
    result = {field: UNKNOWN_ROUTE[field] for field in RESPONSE_FIELDS}
    result["planner_actions"] = []
    result["tool_sequence"] = []
    return result


def classify(instruction: str, models: dict) -> dict:
    """Predict the route for `instruction` and expand it to response fields."""
    model = models.get("model")
    registry = models.get("registry") or {}
    threshold = float(models.get("threshold", 0.4))

    if model is None or not registry:
        logger.warning("Route model unavailable; falling back to a clarifying question.")
        result = _unknown()
        result["entities"] = extract_entities(instruction)
        return result

    routes, confidences = score_routes(model, registry, [instruction])
    route, confidence = str(routes[0]), float(confidences[0])

    entry = registry.get(route)
    if entry is None or confidence < threshold:
        # Confidence stays out of the response body: the wire contract in
        # tests/test_response_contract.py is frozen.  Log it instead.
        logger.info("declining route %s (confidence %.3f < %.3f)", route, confidence, threshold)
        result = _unknown()
    else:
        result = {field: entry[field] for field in RESPONSE_FIELDS}
        result["planner_actions"] = list(entry["planner_actions"])
        result["tool_sequence"] = list(entry["tool_sequence"])
        result["planner_action_count"] = len(result["planner_actions"])
        result["workflow_type"] = _workflow_type(instruction, result["planner_actions"])

    # The plan tells the extractor what kind of request this is: a wallet or
    # navigation action has no product to read out of the sentence.
    result["entities"] = extract_entities(instruction, result["planner_actions"])
    return result
