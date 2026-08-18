from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dataset" / "noah_dataset_20k_final.csv"
MODEL_DIR = BASE_DIR / "trained_models"
METRICS_PATH = MODEL_DIR / "evaluation.json"
SVM_FIELDS = ["domain", "intent", "sub_intent", "tool", "response_mode", "entity_category",
              "requires_memory", "requires_rag", "requires_recommendation", "workflow_type",
              "planner_actions", "planner_action_count", "tool_sequence"]
REPORT_FIELDS = {"domain", "intent", "sub_intent", "workflow_type"}


def make_model():
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
                     ("svm", LinearSVC(C=1.5, class_weight="balanced"))])


def train_models():
    """Fit SVMs on the authoritative CSV and persist held-out evaluation."""
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    train_df, holdout_df = train_test_split(df, test_size=0.20, random_state=42)
    validation_df, test_df = train_test_split(holdout_df, test_size=0.50, random_state=42)
    metrics = {}
    for field in SVM_FIELDS:
        y_train = train_df[field].fillna("NULL").astype(str)
        if y_train.nunique() < 2:
            continue
        model = make_model().fit(train_df["instruction"].fillna(""), y_train)
        joblib.dump(model, MODEL_DIR / f"{field}.joblib")
        y_test = test_df[field].fillna("NULL").astype(str)
        predicted = model.predict(test_df["instruction"].fillna(""))
        metrics[field] = {"accuracy": accuracy_score(y_test, predicted),
                          "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
                          "weighted_f1": f1_score(y_test, predicted, average="weighted", zero_division=0)}
        if field in REPORT_FIELDS:
            metrics[field]["classification_report"] = classification_report(y_test, predicted, output_dict=True, zero_division=0)
        print(f"{field}: accuracy={metrics[field]['accuracy']:.4f} macro_f1={metrics[field]['macro_f1']:.4f} weighted_f1={metrics[field]['weighted_f1']:.4f}")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved models to {MODEL_DIR}; evaluation to {METRICS_PATH}")


if __name__ == "__main__":
    train_models()
