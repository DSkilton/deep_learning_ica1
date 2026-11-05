"""Safely list/copy top experiment runs, without deleting anything by default.

Examples:
  python scripts/top10.py
  python scripts/top10.py --apply
  python scripts/top10.py --apply --delete-low-scores

Only the final command removes low-scoring source runs. All paths are checked.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def safe_run_dir(log_dir: Path, name: str) -> Path:
    if not isinstance(name, str) or name in {"", ".", ".."}:
        raise ValueError(f"Invalid run name: {name!r}")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe run name: {name!r}")
    target = log_dir / name
    if target.is_symlink():
        raise ValueError(f"Symlink run directory not allowed: {target}")
    if target.resolve().parent != log_dir.resolve():
        raise ValueError(f"Run directory escapes log root: {name!r}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs/fit/group_split_v2")
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Copy selected runs and write summary")
    parser.add_argument(
        "--delete-low-scores", action="store_true",
        help="Also delete source runs below threshold (requires --apply)",
    )
    args = parser.parse_args()
    if args.delete_low_scores and not args.apply:
        parser.error("--delete-low-scores requires --apply")
    if args.top_n < 1:
        parser.error("--top-n must be positive")

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")
    result_files = sorted(log_dir.glob("*_results.csv"))
    frames = []
    for results_path in result_files:
        frame = pd.read_csv(results_path)
        needed = {"model_tag", "run_name", "macro_f1"}
        if not needed.issubset(frame):
            raise ValueError(f"Missing columns in {results_path}: {needed - set(frame)}")
        frame["source_file"] = results_path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No experiment result CSVs under {log_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("macro_f1", ascending=False).drop_duplicates("run_name")

    # Validate ALL names before performing ANY writes or deletions.
    for name in df["run_name"]:
        safe_run_dir(log_dir, name)

    low_df = df.loc[df["macro_f1"] < args.threshold]
    top_dir = log_dir.parent / "top10"
    rows = []
    for model_tag, group in df.groupby("model_tag"):
        if Path(str(model_tag)).name != str(model_tag) or str(model_tag) in {".", ".."}:
            raise ValueError(f"Unsafe model tag: {model_tag!r}")
        for rank, (_, row) in enumerate(
            group.sort_values("macro_f1", ascending=False).head(args.top_n).iterrows(),
            start=1,
        ):
            src = safe_run_dir(log_dir, row["run_name"])
            dest = top_dir / str(model_tag) / row["run_name"]
            rows.append({
                "model_tag": model_tag, "rank": rank,
                "run_name": row["run_name"], "macro_f1": float(row["macro_f1"]),
                "source_file": row["source_file"],
            })
            print(f"{'[copy]' if args.apply else '[would copy]'} {src} -> {dest}")
            if args.apply and src.is_dir():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copytree(src, dest)

    for _, row in low_df.iterrows():
        src = safe_run_dir(log_dir, row["run_name"])
        destructive = args.apply and args.delete_low_scores
        print(f"{'[delete]' if destructive else '[would delete only with --apply --delete-low-scores]'} {src}")
        if destructive and src.is_dir():
            shutil.rmtree(src)

    if args.apply:
        top_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(top_dir / "top10_summary.csv", index=False)
    else:
        print("Dry run: nothing was copied, overwritten, or deleted.")


if __name__ == "__main__":
    main()
