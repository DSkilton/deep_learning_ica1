import os
import shutil
from pathlib import Path
import pandas as pd

SCORE_THRESHOLD = 0.90
TOP_N_PER_MODEL = 10

project_root = Path("/workspace/deep_learning_ica1")
log_dir = project_root / "logs" / "fit"

results_csv = log_dir / "results_so_far.csv"

top_dir = project_root / "logs" / "top10"

results_files = [
    log_dir / "att_results.csv",
    log_dir / "bigru_results.csv",
    log_dir / "bigru_att_results.csv",
]

# JUST FOR SAFE RUNS WHILE CHECKING IT WORKS AS EXPECTED
APPLY_DELETIONS = True

dataframes = []

for results_csv in results_files:
    if not results_csv.exists():
        print(f"WARNING: Missing results file: {results_csv}")
        continue

    temp_df = pd.read_csv(results_csv)

    required_cols = {"model_tag", "run_name", "macro_f1"}
    missing = required_cols - set(temp_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {results_csv.name}: {missing}")

    temp_df["source_file"] = results_csv.name
    dataframes.append(temp_df)

if not dataframes:
    raise FileNotFoundError("No results CSV files were found.")

df = pd.concat(dataframes, ignore_index=True)

# Keep best score per run_name in case duplicates exist
df = df.sort_values("macro_f1", ascending=False).drop_duplicates(subset=["run_name"])

print(f"Loaded {len(df)} unique runs from {len(dataframes)} CSV files")

low_df = df[df["macro_f1"] < SCORE_THRESHOLD].copy()
print(f"\nRuns below {SCORE_THRESHOLD:.2f}: {len(low_df)}")

deleted_count = 0
missing_count = 0

for _, row in low_df.iterrows():
    run_name = row["run_name"]
    run_path = log_dir / run_name

    if run_path.exists():
        print(f"DELETE: {run_path}  (macro_f1={row['macro_f1']:.4f})")
        if APPLY_DELETIONS:
            shutil.rmtree(run_path)
        deleted_count += 1
    else:
        print(f"MISSING: {run_path}  (macro_f1={row['macro_f1']:.4f})")
        missing_count += 1

print(f"\nLow-score cleanup summary")
print(f"  Candidate folders: {len(low_df)}")
print(f"  Existing folders found: {deleted_count}")
print(f"  Missing folders: {missing_count}")
print(f"  APPLY_DELETIONS = {APPLY_DELETIONS}")

top_dir.mkdir(parents=True, exist_ok=True)

top_summary_rows = []

for model_tag, model_group in df.groupby("model_tag"):
    model_top_dir = top_dir / model_tag
    model_top_dir.mkdir(parents=True, exist_ok=True)

    # Clear old contents first
    for child in model_top_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    top_n = model_group.sort_values("macro_f1", ascending=False).head(TOP_N_PER_MODEL)

    print(f"\nTop {TOP_N_PER_MODEL} for model: {model_tag}")

    for rank, (_, row) in enumerate(top_n.iterrows(), start=1):
        run_name = row["run_name"]
        src = log_dir / run_name
        dst = model_top_dir / run_name

        print(f"  #{rank}: {run_name}  (macro_f1={row['macro_f1']:.4f})")

        top_summary_rows.append({
            "model_tag": model_tag,
            "rank": rank,
            "run_name": run_name,
            "macro_f1": row["macro_f1"],
            "source_file": row["source_file"]
        })

        if src.exists():
            if APPLY_DELETIONS:
                shutil.copytree(src, dst)
        else:
            print(f"     WARNING: source folder not found: {src}")

top_summary_df = pd.DataFrame(top_summary_rows)
top_summary_csv = top_dir / "top10_summary.csv"
top_summary_df.to_csv(top_summary_csv, index=False)

print(f"\nTop-10 summary written to: {top_summary_csv}")
print("Done.")