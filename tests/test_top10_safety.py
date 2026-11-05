"""A regression test for the destructive experiment tidy-up helper."""
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_top10_dry_run_never_modifies_directories(tmp_path):
    log_dir = tmp_path / "logs" / "fit" / "group_split_v2"
    run_dir = log_dir / "run_one"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "do-not-delete.txt"
    sentinel.write_text("important")
    pd.DataFrame([{
        "model_tag": "att_bpe", "run_name": "run_one", "macro_f1": 0.2,
    }]).to_csv(log_dir / "att_bpe_results.csv", index=False)

    process = subprocess.run(
        [sys.executable, "scripts/top10.py", "--log-dir", str(log_dir)],
        capture_output=True, text=True, check=True,
    )
    assert "Dry run" in process.stdout
    assert sentinel.read_text() == "important"
    assert not (log_dir.parent / "top10").exists()


def test_top10_rejects_traversal_before_cleanup(tmp_path):
    log_dir = tmp_path / "logs" / "fit" / "group_split_v2"
    log_dir.mkdir(parents=True)
    pd.DataFrame([{
        "model_tag": "att_bpe", "run_name": "../outside", "macro_f1": 0.2,
    }]).to_csv(log_dir / "att_bpe_results.csv", index=False)
    process = subprocess.run(
        [sys.executable, "scripts/top10.py", "--log-dir", str(log_dir), "--apply",
         "--delete-low-scores"],
        capture_output=True, text=True,
    )
    assert process.returncode != 0
    assert "Unsafe run name" in process.stderr
