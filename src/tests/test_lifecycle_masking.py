import numpy as np
import pandas as pd
import pytest

from src.segmentation import (
    classify_runs,
    find_zero_runs,
    lifecycle_bounds,
    mask_breaks,
    mask_zero_runs,
    trim_lifecycle,
)

NAN = np.nan


def _frame(rows, start_d=1):
    n = len(rows[0])
    return pd.DataFrame(np.array(rows, dtype=float), index=pd.Index(["A", "B"][: len(rows)], name="item_id"),
                        columns=[f"d_{d}" for d in range(start_d, start_d + n)])


def test_lifecycle_and_trim_both_ends():
    u = _frame([[NAN, 0, 2, 0, 0, 3, 0, 0]])
    b = lifecycle_bounds(u)
    assert b.loc["A", "first_sale_d"] == 3 and b.loc["A", "last_sale_d"] == 6
    out = trim_lifecycle(u, b)
    assert out.iloc[0].isna().tolist() == [True, True, False, False, False, False, True, True]


def test_classify_and_mask_only_internal_breaks():
    raw = _frame([[5, 0, 0, 0, 1, 0, 2, 0, 0, 0]])
    u = trim_lifecycle(raw, lifecycle_bounds(raw))
    runs = classify_runs(find_zero_runs(u), pd.Series({"A": 3}))
    assert runs["kind"].tolist() == ["break", "demand"]
    out = mask_breaks(u, runs)
    assert out.iloc[0].isna().tolist() == [False, True, True, True, False, False, False, True, True, True]
    with pytest.raises(ValueError, match="LONG_RUN"):
        classify_runs(find_zero_runs(u), pd.Series({"Z": 3}))


def test_matches_old_mask_zero_runs():
    rng = np.random.default_rng(0)
    raw = rng.choice([0.0, 0.0, 0.0, 1.0, 2.0], size=(2, 60))
    raw[1, :5] = NAN  # prelanzamiento
    u = _frame(raw.tolist())
    lr = pd.Series({"A": 3, "B": 4})
    lc = trim_lifecycle(u, lifecycle_bounds(u))
    new = mask_breaks(lc, classify_runs(find_zero_runs(lc), lr)).to_numpy()
    old = mask_zero_runs(u.to_numpy(), u.index, lr, 1, 60)
    np.testing.assert_array_equal(np.isnan(new), np.isnan(old))
    np.testing.assert_array_equal(np.nan_to_num(new, nan=-1), np.nan_to_num(old, nan=-1))

