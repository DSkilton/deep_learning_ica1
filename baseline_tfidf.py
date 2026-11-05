"""Classical text baseline for comparison with neural emotion classifiers.

By default only report validation metrics. Evaluate the held-out test split
once, after model selection, by adding --evaluate-test. No GPU is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from data_splits import make_splits


def scores(model, text, labels):
    prediction = model.predict(text)
    class_names = sorted(set(labels) | set(prediction))
    return {
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "classification_report": classification_report(
            labels, prediction, labels=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=class_names).tolist(),
        "label_order": [int(x) for x in class_names],
        "sample_count": int(len(labels)),
    }


def run_baseline(frame: pd.DataFrame, evaluate_test: bool = False) -> dict:
    x_train, x_val, x_test, y_train, y_val, y_test = make_splits(frame)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=30000, sublinear_tf=True
        )),
        ("logistic", LogisticRegression(
            max_iter=600, class_weight="balanced", random_state=42
        )),
    ])
    # Fit all vectoriser statistics and classifier weights on train ONLY.
    model.fit(x_train, y_train)
    result = {
        "model": "TF-IDF (1-2 grams) + balanced Logistic Regression",
        "training_examples": int(len(y_train)),
        "validation": scores(model, x_val, y_val),
        "test_evaluated": evaluate_test,
        "note": "Comparison requires identical splits; do not tune on test results.",
    }
    if evaluate_test:
        result["test"] = scores(model, x_test, y_test)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tfidf_baseline.json"))
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    result = run_baseline(pd.read_parquet(args.data), evaluate_test=args.evaluate_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Validation macro-F1: {result['validation']['macro_f1']:.4f}")
    if args.evaluate_test:
        print(f"Held-out test macro-F1: {result['test']['macro_f1']:.4f}")
    print(f"Full class report and confusion matrix: {args.output}")


if __name__ == "__main__":
    main()
