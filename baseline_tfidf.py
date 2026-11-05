"""Traditional non-neural reference for the same de-duplicated emotion split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from data_splits import split_emotion_data

def evaluate(path: Path) -> dict:
    frame = pd.read_parquet(path)
    (X_train, X_val, X_test, y_train, y_val, y_test), split_stats = split_emotion_data(frame)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_features=50000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    val_prediction = model.predict(X_val)
    test_prediction = model.predict(X_test)
    return {
        "model": "TF-IDF (1-2 grams) + Logistic Regression",
        "dataset_cleanup": split_stats,
        "train_rows": len(X_train), "validation_rows": len(X_val), "test_rows": len(X_test),
        "validation_macro_f1": float(f1_score(y_val, val_prediction, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, test_prediction)),
        "test_macro_f1": float(f1_score(y_test, test_prediction, average="macro")),
        "test_per_class": classification_report(
            y_test, test_prediction, output_dict=True, zero_division=0
        ),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data.parquet"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data), indent=2))
