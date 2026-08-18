from .action_registry import (
    AVAILABLE_ACTIONS,
    is_valid_action,
)


def create_plan(prediction: dict):

    actions = prediction.get(
        "planner_actions",
        []
    )

    valid_actions = []

    for action in actions:

        if is_valid_action(action):

            valid_actions.append(action)

        else:

            print(
                f"[WARNING] Unknown action ignored: "
                f"{action}"
            )

    plan = {
        "workflow_type": prediction.get(
            "workflow_type",
            "single"
        ),

        "planner_actions": valid_actions,

        "planner_action_count": len(
            valid_actions
        ),

        "tool_sequence": [
            AVAILABLE_ACTIONS[action]["tool"]
            for action in valid_actions
        ],

        "entities": prediction.get(
            "entities",
            {}
        ),
    }

    return plan
