import pandas as pd
import pytest

from src.modeling import assign_folds, fold_masks


def _data():
    # A: 3 bloques antes de la ventana, 2 de ventana, 1 de test; B: empieza más tarde.
    return pd.DataFrame({
        "item_id": ["A"] * 6 + ["B"] * 4,
        "block": [1, 2, 3, 4, 5, 6, 3, 4, 5, 6],
        "split": ["train"] * 5 + ["test"] + ["train"] * 3 + ["test"],
        "backtest": [False, False, False, True, True, False, False, True, True, False],
    })


def test_assign_folds_by_position_in_window():
    out = assign_folds(_data())
    a, b = out[out["item_id"] == "A"], out[out["item_id"] == "B"]
    assert a["fold"].tolist()[:5] == [0, 0, 0, 1, 2] and pd.isna(a["fold"].iloc[5])
    assert b["fold"].tolist()[:3] == [0, 1, 2] and pd.isna(b["fold"].iloc[3])


def test_fold_masks_expanding_without_test():
    out = assign_folds(_data())
    train, valid = fold_masks(out, 2)
    assert out.loc[valid, "block"].tolist() == [5, 5]
    assert out.loc[train, "block"].tolist() == [1, 2, 3, 4, 3, 4]
    assert not (out.loc[train | valid, "split"] == "test").any()
    with pytest.raises(ValueError):
        fold_masks(out, 3)


def test_seasonal_naive_shifts_units_by_m():
    from src.modeling import seasonal_naive

    blocks = pd.DataFrame({"item_id": ["A"] * 4 + ["B"] * 3, "block": [1, 2, 3, 4, 5, 6, 7],
                           "units": [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0]})
    out = seasonal_naive(blocks, 2)
    assert out.iloc[:4].isna().tolist() == [True, True, False, False]
    assert out.iloc[2:4].tolist() == [10.0, 20.0] and out.iloc[6] == 1.0

