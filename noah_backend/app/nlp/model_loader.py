"""Load Noah's trained router and the route registry.

The pipeline used to load thirteen independent classifiers - one per output
field - which could and did disagree with one another.  There is now a single
route model; every other field is looked up from the route it predicts, so an
inconsistent response is not representable.

``load_models()`` keeps its name and dict return type because
``app/api/chat.py`` calls it at import time.
"""

from __future__ import annotations

import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "trained_models"
ROUTE_MODEL_PATH = MODEL_DIR / "route.joblib"
REGISTRY_PATH = MODEL_DIR / "route_registry.json"

# Used when the model is unsure, and whenever the artifacts are missing.
UNKNOWN_ROUTE = {
    "domain": "CHAT",
    "intent": "UNKNOWN",
    "sub_intent": "UNKNOWN",
    "tool": "none",
    "response_mode": "text",
    "requires_memory": False,
    "requires_rag": False,
    "requires_recommendation": False,
    "workflow_type": "single_step",
    "planner_actions": [],
    "planner_action_count": 0,
    "tool_sequence": [],
}


def load_models() -> dict:
    """Return ``{"model", "registry", "threshold"}``; empty model if untrained."""
    if not ROUTE_MODEL_PATH.exists() or not REGISTRY_PATH.exists():
        print("[WARNING] Route model not found - run `python -m app.nlp.train`. "
              "Every request will fall back to the clarifying-question route.")
        return {"model": None, "registry": {}, "threshold": 1.0}

    bundle = joblib.load(ROUTE_MODEL_PATH)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {
        "model": bundle["model"],
        "registry": registry,
        "threshold": float(bundle.get("threshold", 0.4)),
    }
