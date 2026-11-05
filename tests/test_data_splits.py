import pandas as pd
import pytest

from data_splits import canonical_text, make_splits


def test_splits_are_disjoint_after_deduplicating_and_conflict_removal():
    frame = pd.DataFrame({
        "text": [f"sample emotion {i}" for i in range(120)],
        "label": [i % 6 for i in range(120)],
    })
    frame = pd.concat(
        [frame,
         pd.DataFrame({"text": [" SAMPLE emotion 0  ", "sample emotion 1"],
                       "label": [0, 2]})],
        ignore_index=True,
    )
    with pytest.warns(UserWarning, match="conflicting"):
        x_train, x_val, x_test, y_train, y_val, y_test = make_splits(frame)
    assert len(x_train) + len(x_val) + len(x_test) == 119
    sets = [set(canonical_text(x)) for x in (x_train, x_val, x_test)]
    assert all(not sets[i].intersection(sets[j])
               for i, j in [(0, 1), (0, 2), (1, 2)])
    assert len(y_train) == len(x_train)
    assert len(y_val) == len(x_val)
    assert len(y_test) == len(x_test)
