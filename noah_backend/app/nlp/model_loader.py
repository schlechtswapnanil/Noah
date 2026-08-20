from pathlib import Path
# pyrefly: ignore [missing-import]
import joblib


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "trained_models"


SVM_FIELDS = [
    "domain",
    "intent",
    "sub_intent",
    "tool",
    "response_mode",
    "entity_category",
    "requires_memory",
    "requires_rag",
    "requires_recommendation",
    "workflow_type",
    "planner_actions",
    "planner_action_count",
    "tool_sequence",
]


def load_models():

    models = {}

    for field in SVM_FIELDS:

        path = MODEL_DIR / f"{field}.joblib"

        if path.exists():

            models[field] = joblib.load(path)

        else:

            print(
                f"[WARNING] Model not found: {field}"
            )

    return models
