import pandas as pd

from baseline_tfidf import run_baseline


def test_baseline_never_reports_test_by_default():
    frame = pd.DataFrame({
        "text": [f"sample {i} topic {i % 6} vocabulary {i // 6}" for i in range(120)],
        "label": [i % 6 for i in range(120)],
    })
    validation_only = run_baseline(frame)
    assert 0 <= validation_only["validation"]["macro_f1"] <= 1
    assert validation_only["test_evaluated"] is False
    assert "test" not in validation_only
