"""A single deterministic, duplicate-safe train/validation/test split.

This function is used by both the training and feature-visualisation notebooks.
Fitted tokenizers and saved models from earlier splits are NOT compatible.
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split


def canonical_text(text: pd.Series) -> pd.Series:
    """Normalise case and whitespace for cross-split overlap checks."""
    return text.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()


def make_splits(frame: pd.DataFrame, seed: int = 42):
    """Return X_train, X_val, X_test, y_train, y_val, y_test (80/10/10).

    Exact/normalised duplicate text cannot cross partitions. Text with conflicting
    labels is excluded rather than arbitrarily choosing an emotion. Only the
    cleaned text is used for training (never the private grouping key).
    """
    needed = {"text", "label"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {needed - set(frame.columns)}")

    clean = frame[["text", "label"]].dropna().copy()
    clean["text"] = clean["text"].astype(str).str.strip()
    clean["_key"] = canonical_text(clean["text"])
    clean = clean.loc[clean["_key"] != ""].copy()
    conflicts = clean.groupby("_key")["label"].nunique()
    conflicting_texts = set(conflicts[conflicts > 1].index)
    if conflicting_texts:
        warnings.warn(
            f"Dropping {len(conflicting_texts)} texts with conflicting emotion labels",
            UserWarning, stacklevel=2,
        )
        clean = clean.loc[~clean["_key"].isin(conflicting_texts)].copy()
    original_rows = len(clean)
    clean = clean.drop_duplicates(subset=["_key"]).reset_index(drop=True)
    if len(clean) < original_rows:
        print(f"Removed {original_rows - len(clean)} duplicate text rows before splitting")

    x = clean["text"]
    y = clean["label"].astype(int)

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )
    keys = [set(canonical_text(s)) for s in (x_train, x_val, x_test)]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise AssertionError("Duplicate text crosses a data split")
    return x_train, x_val, x_test, y_train, y_val, y_test
