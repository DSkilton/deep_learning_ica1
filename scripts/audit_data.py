"""Audit source text duplication and the split actually used by the notebooks."""
import argparse
import json
from pathlib import Path

import pandas as pd

from data_splits import canonical_text, make_splits


def audit(path: Path) -> dict:
    frame = pd.read_parquet(path)
    source_rows = len(frame)
    clean = frame[["text", "label"]].dropna().copy()
    clean["_key"] = canonical_text(clean["text"])
    clean = clean.loc[clean["_key"] != ""]
    label_counts = clean.groupby("_key")["label"].nunique()
    conflicted = int((label_counts > 1).sum())
    unique_texts = int(label_counts.size)
    train_x, val_x, test_x, train_y, val_y, test_y = make_splits(frame)
    splits = {
        "train": (train_x, train_y),
        "validation": (val_x, val_y),
        "test": (test_x, test_y),
    }
    keys = {name: set(canonical_text(x)) for name, (x, _) in splits.items()}
    overlap = {
        "train_validation": len(keys["train"] & keys["validation"]),
        "train_test": len(keys["train"] & keys["test"]),
        "validation_test": len(keys["validation"] & keys["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"Text overlap is not zero: {overlap}")
    return {
        "source_rows": source_rows,
        "unique_nonempty_normalised_texts": unique_texts,
        "conflicting_label_texts_excluded": conflicted,
        "deduplicated_usable_texts": unique_texts - conflicted,
        "sizes": {name: int(len(x)) for name, (x, _) in splits.items()},
        "class_counts": {
            name: {str(label): int(count) for label, count in y.value_counts().items()}
            for name, (_, y) in splits.items()
        },
        "overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/split_audit.json"))
    args = parser.parse_args()
    report = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
