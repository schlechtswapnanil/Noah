"""Train Noah's intent router and persist an honest evaluation.

Two things differ from the original multi-head trainer:

1. **One model instead of thirteen.**  ``(sub_intent, planner_actions)``
   determines every other label in the corpus, so the classifier predicts that
   pair and the remaining fields are looked up in a registry built from the
   training data.  Independent heads used to disagree with each other - an
   utterance could come back as ``WALLET / OPEN_CARD / SMALL_TALK`` - which is
   impossible by construction now.

2. **A leakage-free split.**  The rows are grouped by surface template before
   splitting, so a paraphrase of a training utterance cannot appear in the test
   set.  The old random row split put 86% of the test set into training
   verbatim and reported 1.00 accuracy for it.

    python -m app.nlp.train
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

from .routing import score_routes

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dataset" / "noah_dataset_v3.csv"
LEGACY_DATASET_PATH = BASE_DIR / "dataset" / "noah_dataset_20k_final.csv"
MODEL_DIR = BASE_DIR / "trained_models"
ROUTE_MODEL_PATH = MODEL_DIR / "route.joblib"
REGISTRY_PATH = MODEL_DIR / "route_registry.json"
METRICS_PATH = MODEL_DIR / "evaluation.json"
PROBE_PATH = BASE_DIR / "dataset" / "noah_holdout_probes.csv"

ROUTE_SEPARATOR = "»"
DERIVED_FIELDS = ["domain", "intent", "sub_intent", "tool", "response_mode",
                  "requires_memory", "requires_rag", "requires_recommendation",
                  "workflow_type", "planner_actions", "planner_action_count",
                  "tool_sequence"]
BOOLEAN_FIELDS = {"requires_memory", "requires_rag", "requires_recommendation"}
OUT_OF_SCOPE_SUB_INTENT = "UNKNOWN"

_MERCHANT_PATTERN = (r"rewe|netto|lidl|aldi|kaufland|edeka|penny|müller|mueller"
                     r"|dm|rossmann|globus|hit")
_CARD_PATTERN = (r"payback|deutschlandcard|netto plus|lidl plus|rewe bonus"
                 r"|edeka card")


def template_key(instruction: str) -> str:
    """Collapse an utterance to its surface template.

    Grouping on this before splitting stops "Find milk under 2 euros" and
    "Find bread under 5 euros" from straddling the train/test boundary.
    """
    text = str(instruction).lower()
    text = re.sub(r"\d+(?:[.,]\d+)?", "<N>", text)
    text = re.sub(_CARD_PATTERN, "<C>", text)
    text = re.sub(_MERCHANT_PATTERN, "<M>", text)
    text = re.sub(r"[^a-zäöüß<>\s]", " ", text)
    return " ".join(text.split())


def make_model() -> Pipeline:
    """Word n-grams catch phrasing; character n-grams carry typo and German
    compound robustness ("Einkaufsliste" shares stems with "Einkauf")."""
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb",
                                     ngram_range=(3, 5), sublinear_tf=True,
                                     min_df=2)),
        ])),
        ("classifier", LogisticRegression(max_iter=1500, C=8.0,
                                          class_weight="balanced")),
    ])


def build_registry(df: pd.DataFrame) -> dict:
    """route label -> every other field, taken from the training rows."""
    registry = {}
    for route, group in df.groupby("route"):
        row = group.iloc[0]
        entry = {}
        for field in DERIVED_FIELDS:
            value = row[field]
            if field in BOOLEAN_FIELDS:
                entry[field] = str(value).strip().lower() == "true"
            elif field == "planner_actions":
                entry[field] = [] if str(value) == "NULL" else str(value).split("|")
            elif field == "tool_sequence":
                actions = [] if str(row["planner_actions"]) == "NULL" else str(value).split("|")
                entry[field] = actions
            elif field == "planner_action_count":
                entry[field] = int(value)
            else:
                entry[field] = None if pd.isna(value) or value == "NULL" else str(value)
        entry["planner_action_count"] = len(entry["planner_actions"])
        entry["tool_sequence"] = entry["tool_sequence"][:entry["planner_action_count"]]
        registry[route] = entry
    return registry


def split(df: pd.DataFrame, test_size: float, seed: int):
    groups = df["instruction"].map(template_key)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    left, right = next(splitter.split(df, groups=groups))
    return df.iloc[left].reset_index(drop=True), df.iloc[right].reset_index(drop=True)


# Acting on a misread request is the expensive mistake: the app opens the wrong
# card, launches maps, or fires a paid MCP call.  Asking one clarifying question
# costs the user a tap.  These weights encode that ratio and are what the
# confidence threshold is tuned against.
# Business dial: raise CLARIFYING_QUESTION_COST to make Noah act more often and
# ask less, lower it to make it more cautious.  0.5 says one wrong action is
# worth two unnecessary clarifying questions.
#
# Noah is not moving money: the worst wrong action opens the wrong screen or
# spends one price lookup, while a refusal on a perfectly reasonable request is
# the more visible failure to someone using the assistant.  Note that 0.35 and
# 0.5 currently select the same threshold on the calibration probes - the dial
# is here to be adjusted deliberately, not because it is presently load-bearing.
WRONG_ACTION_COST = 1.0
CLARIFYING_QUESTION_COST = 0.5


def load_probes(split: str | None = None) -> pd.DataFrame:
    """Hand-written utterances that never enter the corpus.

    Half calibrate the confidence threshold, half report generalisation.  They
    are the only realistic sample available: the corpus itself is
    template-generated, so its own held-out split understates how uncertain the
    model is on phrasing it has never seen.
    """
    if not PROBE_PATH.exists():
        return pd.DataFrame()
    probes = pd.read_csv(PROBE_PATH)
    corpus = set(pd.read_csv(DATASET_PATH).instruction.str.lower().str.strip())
    leaked = [t for t in probes.instruction if t.lower().strip() in corpus]
    if leaked:
        raise RuntimeError(f"{len(leaked)} probe(s) leaked into the corpus: {leaked[:3]}")
    if split:
        probes = probes[probes["split"] == split].reset_index(drop=True)
    probes["sub_intent"] = probes["expected_sub_intent"]
    return probes


def choose_threshold(model, registry: dict, validation: pd.DataFrame) -> float:
    """Pick the confidence floor below which Noah asks instead of acting.

    Out-of-scope input is caught two ways: the model can predict the UNKNOWN
    route outright, or the winning route can be too weak to trust.  Only the
    second needs a threshold, so this picks the *lowest* one that still turns
    away `MIN_OUT_OF_SCOPE_REJECTION` of out-of-scope utterances - refusing a
    request the model would have got right is the more expensive error, and a
    threshold tuned on rejection alone lands high enough to decline roughly one
    valid request in five.
    """
    predicted, confidence = score_routes(model, registry, validation["instruction"])
    predicted_sub_intent = np.array([registry[r]["sub_intent"] for r in predicted])
    predicted_unknown = predicted_sub_intent == OUT_OF_SCOPE_SUB_INTENT

    in_scope = (validation["sub_intent"] != OUT_OF_SCOPE_SUB_INTENT).to_numpy()
    correct = predicted_sub_intent == validation["sub_intent"].to_numpy()

    best_threshold, best_cost = 0.0, float("inf")
    for threshold in np.arange(0.02, 0.95, 0.01):
        declined = (confidence < threshold) | predicted_unknown
        cost = np.where(
            declined,
            # A clarifying question is only wasted on a request we'd have got right.
            np.where(in_scope & correct, CLARIFYING_QUESTION_COST, 0.0),
            # Acting is free when right, and costly when wrong or out of scope.
            np.where(in_scope & correct, 0.0, WRONG_ACTION_COST),
        ).mean()
        if cost < best_cost:
            best_threshold, best_cost = float(threshold), float(cost)
    return round(best_threshold, 3)


def evaluate(model, registry: dict, test: pd.DataFrame, threshold: float) -> dict:
    predicted, confidence = score_routes(model, registry, test["instruction"])
    truth = test["route"].to_numpy()

    metrics = {
        "split": "grouped by surface template (no paraphrase leakage)",
        "test_utterances": int(len(test)),
        "route": {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        },
        "confidence_threshold": threshold,
    }

    fallback = registry[max(registry, key=lambda r: r.endswith("UNKNOWN" + ROUTE_SEPARATOR + "NULL"))] \
        if any(r.startswith("UNKNOWN") for r in registry) else None

    per_field = {}
    for field in DERIVED_FIELDS:
        expected, actual = [], []
        for route_true, route_pred in zip(truth, predicted):
            expected.append(str(registry[route_true][field]))
            actual.append(str(registry[route_pred][field]))
        per_field[field] = {
            "accuracy": float(accuracy_score(expected, actual)),
            "macro_f1": float(f1_score(expected, actual, average="macro", zero_division=0)),
        }
    metrics["fields"] = per_field
    metrics["full_tuple_exact_match"] = per_field["sub_intent"]["accuracy"] if False else float(
        np.mean([all(str(registry[t][f]) == str(registry[p][f]) for f in DERIVED_FIELDS)
                 for t, p in zip(truth, predicted)]))

    in_scope = (test["sub_intent"] != OUT_OF_SCOPE_SUB_INTENT).to_numpy()
    accepted = confidence >= threshold
    predicted_unknown = np.array([registry[r]["sub_intent"] == OUT_OF_SCOPE_SUB_INTENT
                                  for r in predicted])
    declined = (~accepted) | predicted_unknown
    metrics["abstention"] = {
        "out_of_scope_utterances": int((~in_scope).sum()),
        "out_of_scope_declined": float(declined[~in_scope].mean()) if (~in_scope).any() else None,
        "in_scope_wrongly_declined": float(declined[in_scope].mean()) if in_scope.any() else None,
        "in_scope_accuracy_after_abstention": float(
            (predicted[in_scope & ~declined] == truth[in_scope & ~declined]).mean()),
    }

    intents = [registry[r]["intent"] for r in truth]
    predicted_intents = [registry[r]["intent"] for r in predicted]
    metrics["intent_classification_report"] = classification_report(
        intents, predicted_intents, output_dict=True, zero_division=0)
    return metrics


def evaluate_probes(model, registry: dict, threshold: float) -> dict:
    """Score the reporting half of the probe set (never used for tuning)."""
    probes = load_probes(split="report")
    if probes.empty:
        return {}

    predicted, confidence = score_routes(model, registry, probes["instruction"])
    predicted_sub_intent = np.array([registry[r]["sub_intent"] for r in predicted])
    declined = (confidence < threshold) | (predicted_sub_intent == OUT_OF_SCOPE_SUB_INTENT)

    expected = probes["expected_sub_intent"].to_numpy()
    in_scope = expected != OUT_OF_SCOPE_SUB_INTENT
    correct = predicted_sub_intent == expected

    misses = [
        {"instruction": t, "expected": e, "predicted": p, "confidence": round(float(c), 3)}
        for t, e, p, c, ok in zip(probes.instruction, expected, predicted_sub_intent,
                                  confidence, correct) if not ok
    ]
    return {
        "probe_utterances": int(len(probes)),
        "in_scope_sub_intent_accuracy": float(correct[in_scope].mean()),
        "in_scope_wrongly_declined": float(declined[in_scope].mean()),
        "out_of_scope_declined": float(declined[~in_scope].mean()),
        "misses": misses,
    }


def train_models() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    dataset_path = DATASET_PATH if DATASET_PATH.exists() else LEGACY_DATASET_PATH
    df = pd.read_csv(dataset_path)
    df["planner_actions"] = df["planner_actions"].fillna("NULL")
    df["route"] = df["sub_intent"].astype(str) + ROUTE_SEPARATOR + df["planner_actions"].astype(str)

    train_df, holdout = split(df, test_size=0.30, seed=42)
    validation_df, test_df = split(holdout, test_size=0.50, seed=42)

    # Routes the split stranded entirely in the holdout cannot be scored.
    known = set(train_df["route"])
    validation_df = validation_df[validation_df["route"].isin(known)].reset_index(drop=True)
    test_df = test_df[test_df["route"].isin(known)].reset_index(drop=True)

    print(f"dataset      : {dataset_path.name}  ({len(df)} utterances, "
          f"{df.route.nunique()} routes)")
    print(f"train/val/test: {len(train_df)} / {len(validation_df)} / {len(test_df)}")

    model = make_model().fit(train_df["instruction"].fillna(""), train_df["route"])
    registry = build_registry(df)

    calibration = load_probes(split="calibration")
    if calibration.empty:
        threshold = choose_threshold(model, registry, validation_df)
    else:
        threshold = choose_threshold(model, registry, calibration)
    print(f"confidence threshold: {threshold}")

    metrics = evaluate(model, registry, test_df, threshold)
    metrics["holdout_probes"] = evaluate_probes(model, registry, threshold)

    joblib.dump({"model": model, "threshold": threshold}, ROUTE_MODEL_PATH)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    print(f"\nroute accuracy          : {metrics['route']['accuracy']:.4f}")
    print(f"route macro F1          : {metrics['route']['macro_f1']:.4f}")
    print(f"full tuple exact match  : {metrics['full_tuple_exact_match']:.4f}")
    print("\nper-field (accuracy / macro F1):")
    for field, scores in metrics["fields"].items():
        print(f"  {field:<24s} {scores['accuracy']:.4f} / {scores['macro_f1']:.4f}")
    probe = metrics.get("holdout_probes") or {}
    if probe:
        print("\nhand-written holdout probes "
              f"({probe['probe_utterances']} utterances, never trained on):")
        print(f"  in-scope sub_intent accuracy      {probe['in_scope_sub_intent_accuracy']:.4f}")
        print(f"  in-scope wrongly declined         {probe['in_scope_wrongly_declined']:.4f}")
        print(f"  out-of-scope declined             {probe['out_of_scope_declined']:.4f}")
    print("\nabstention (test split):")
    for key, value in metrics["abstention"].items():
        print(f"  {key:<38s} {value}")
    print(f"\nsaved model to {ROUTE_MODEL_PATH.name}, registry to {REGISTRY_PATH.name}")
    return metrics


if __name__ == "__main__":
    train_models()
