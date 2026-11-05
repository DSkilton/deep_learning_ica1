"""Stable leak-resistant train/validation/test split for tweet emotion data."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def clean_emotion_data(frame: pd.DataFrame):
    """Remove exact/normalised duplicates and ambiguous conflicting labels.

    Do this before splitting so duplicate texts cannot leak across partitions.
    Conflicting labels for the same normalised text are *excluded*, not
    arbitrarily resolved.
    """
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
    data = frame.loc[frame["text"].notna() & frame["label"].notna()].copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data.loc[data["text"].ne("")].copy()
    data["_text_key"] = data["text"].str.casefold()
    label_count = data.groupby("_text_key")["label"].nunique()
    conflicts = set(label_count[label_count > 1].index)
    conflicting_rows = int(data["_text_key"].isin(conflicts).sum())
    data = data.loc[~data["_text_key"].isin(conflicts)].copy()
    before_dedup = len(data)
    data = data.drop_duplicates(subset=["_text_key"], keep="first")
    stats = {
        "input_rows": len(frame),
        "removed_empty_or_missing": len(frame) - before_dedup - conflicting_rows,
        "removed_conflicting_rows": conflicting_rows,
        "removed_duplicate_rows": before_dedup - len(data),
        "remaining_rows": len(data),
    }
    return data.drop(columns=["_text_key"]).reset_index(drop=True), stats


def split_emotion_data(frame: pd.DataFrame, random_state: int = SEED):
    """Return the same 80/10/10 stratified split for all model families."""
    clean, stats = clean_emotion_data(frame)
    X, y = clean["text"], clean["label"].astype(int)
    Xt, Xhold, yt, yhold = train_test_split(
        X, y, test_size=.20, random_state=random_state, stratify=y
    )
    Xval, Xtest, yval, ytest = train_test_split(
        Xhold, yhold, test_size=.5, random_state=random_state, stratify=yhold
    )
    keys = [set(s.str.strip().str.casefold()) for s in (Xt, Xval, Xtest)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    return (Xt, Xval, Xtest, yt, yval, ytest), stats
