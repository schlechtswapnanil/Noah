"""Validate a predicted route into an executable plan.

The plan carries the routing labels alongside the actions so the response layer
can phrase a reply without re-deriving them (a greeting and an unrecognised
request both have zero actions, but they need different answers).
"""

import logging

from .action_registry import AVAILABLE_ACTIONS, is_valid_action

logger = logging.getLogger(__name__)

CARRIED_FIELDS = ("domain", "intent", "sub_intent", "response_mode",
                  "workflow_type", "requires_rag")


def create_plan(prediction: dict) -> dict:
    valid_actions = []
    for action in prediction.get("planner_actions", []):
        if is_valid_action(action):
            valid_actions.append(action)
        else:
            logger.warning("Unknown action ignored: %s", action)

    plan = {field: prediction.get(field) for field in CARRIED_FIELDS}
    plan["workflow_type"] = prediction.get("workflow_type", "single_step")
    plan["planner_actions"] = valid_actions
    plan["planner_action_count"] = len(valid_actions)
    plan["tool_sequence"] = [AVAILABLE_ACTIONS[a]["tool"] for a in valid_actions]
    plan["entities"] = prediction.get("entities", {})
    return plan
