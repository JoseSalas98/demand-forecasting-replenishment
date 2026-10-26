import numpy as np
import pandas as pd

from src.segmentation import (
    build_matrix,
    classify_abc,
    classify_xyz,
    compute_cv,
    compute_usage_spend,
)

SEG = {
    "abc_cuts": [0.80, 0.95],
    "xyz_thresholds": [10.0, 25.0],
    "cv_aggregation": "block28",
    "cv_min_blocks": 2,
}


def test_item_crossing_cut_is_a():
    # Gastos 70, 20, 6, 4: el segundo cruza el 80 % (antes de sumarse está en 70 %).
    out = classify_abc(["a", "b", "c", "d"], np.array([70.0, 20.0, 6.0, 4.0]), [0.80, 0.95])
    assert list(out.abc_class) == ["A", "A", "B", "C"]
    np.testing.assert_allclose(out.cum_pct_before, [0.0, 0.70, 0.90, 0.96])


def test_ties_are_broken_by_item_id():
    out = classify_abc(["z", "b", "m"], np.array([5.0, 5.0, 5.0]), [0.80, 0.95])
    assert list(out.item_id) == ["b", "m", "z"]


def test_xyz_bounds():
    out = classify_xyz(pd.Series([9.99, 10.0, 24.9, 25.0, np.nan]), [10.0, 25.0])
    assert list(out[:4]) == ["X", "Y", "Y", "Z"]
    assert pd.isna(out.iloc[4])


def test_cv_block28_population_std_and_incomplete_blocks():
    days = 84
    units = np.zeros((3, days))
    units[0, :28] = 1.0
    units[0, 28:56] = 2.0
    units[0, 56:84] = 3.0
    units[1, :] = 5.0
    units[2, :] = 1.0
    units[2, :40] = np.nan  # Lanzamiento tardío: solo el tercer bloque está completo.
    out = compute_cv(units, 1, days, "block28", 2)
    sums = np.array([28.0, 56.0, 84.0])
    assert np.isclose(out.cv[0], sums.std(ddof=0) / sums.mean() * 100)
    assert out.cv[1] == 0.0
    assert out.n_blocks[2] == 1
    assert np.isnan(out.cv[2])


def test_cv_undefined_when_mean_zero():
    out = compute_cv(np.zeros((1, 56)), 1, 56, "block28", 2)
    assert np.isnan(out.cv[0])


def test_cv_daily_and_weekly():
    units = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 4).reshape(1, 28)
    daily = compute_cv(units, 1, 28, "daily", 2)
    assert np.isclose(daily.cv[0], np.std(units[0]) / np.mean(units[0]) * 100)
    wod = np.repeat(np.arange(4), 7)
    weekly = compute_cv(units, 1, 28, "weekly", 2, week_of_day=wod)
    assert weekly.n_blocks[0] == 4
    assert weekly.cv[0] == 0.0
    # Semana cortada por el borde del rango: se descarta.
    partial = compute_cv(units, 4, 28, "weekly", 2, week_of_day=wod)
    assert partial.n_blocks[0] == 3


def test_usage_spend_ignores_nan():
    u = np.array([[1.0, np.nan, 2.0], [np.nan, np.nan, np.nan]])
    s = np.array([[2.0, np.nan, 4.0], [np.nan, np.nan, np.nan]])
    out = compute_usage_spend(u, s, 1, 3)
    assert list(out.usage) == [3.0, 0.0]
    assert list(out.spend) == [6.0, 0.0]
    assert list(out.active) == [True, False]


def _agg():
    days = 56
    units = np.ones((3, days))
    units[1, ::2] = 3.0
    units[2, :] = np.nan
    return {
        "item_ids": np.array(["i1", "i2", "i3"]),
        "item_units": units,
        "item_spend": units * 2.0,
        "item_launch_d": np.array([1, 1, np.nan]),
    }


def test_matrix_columns_cells_and_exclusions():
    m = build_matrix("train_global", _agg(), 1, 56, SEG)
    cols = [
        "set_id",
        "item_id",
        "launch_d",
        "usage",
        "spend",
        "cum_spend",
        "cum_pct_before",
        "abc_class",
        "cv",
        "xyz_class",
        "cell",
        "exclusion_reason",
    ]
    assert list(m.columns) == cols
    row3 = m.set_index("item_id").loc["i3"]
    assert row3.exclusion_reason == "inactive_in_set"
    assert pd.isna(row3.abc_class)
    seg = dict(SEG, cv_min_blocks=3)
    m2 = build_matrix("train_global", _agg(), 1, 56, seg).set_index("item_id")
    assert m2.loc["i1", "exclusion_reason"] == "cv_undefined"
    assert m2.loc["i1", "cell"].endswith("-")


def test_sensitivity_table_counts():
    from src.segmentation import sensitivity_table

    # Gastos diarios constantes: 76, 14, 6, 4 → con cortes base A=2, B=1, C=1.
    spend = np.repeat(np.array([[76.0], [14.0], [6.0], [4.0]]), 56, axis=1)
    agg = {
        "item_ids": np.array(["a", "b", "c", "d"], dtype=object),
        "item_units": np.ones((4, 56)),
        "item_spend": spend,
    }
    out = sensitivity_table(agg, 1, 56, SEG).set_index(["variant", "dimension", "class"])["count"]
    assert out[("base", "abc", "A")] == 2
    # Corte A en 85 %: b (76 % antes de sumarse) sigue en A.
    assert out[("abc_cut_A+5pp", "abc", "A")] == 2
    # Corte A en 75 %: b pasa a B.
    assert out[("abc_cut_A-5pp", "abc", "A")] == 1
    # Corte B en 90 %: c (90 % antes de sumarse) pasa a C.
    assert out[("abc_cut_B-5pp", "abc", "C")] == 2
    assert out[("base", "xyz", "X")] == 4
