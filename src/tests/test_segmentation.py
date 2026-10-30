import numpy as np
import pandas as pd
import pytest

from src.segmentation import (
    build_matrix,
    classify_abc,
    classify_xyz,
    compute_cv,
    compute_usage_spend,
    find_cv_cuts,
    longest_run_threshold,
    mask_zero_runs,
    zero_run_thresholds,
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


def _cv_spend_three_levels(n=300):
    cv = np.linspace(5, 100, n)
    spend = np.where(cv < 30, 300.0, np.where(cv < 60, 200.0, 100.0))
    return cv, spend


def test_find_cv_cuts_recovers_monotone_levels():
    cv, spend = _cv_spend_three_levels()
    out = find_cv_cuts(cv, spend, n_cuts=2, min_share=0.10)
    assert out["segments"]["class"].tolist() == ["X", "Y", "Z"]
    assert out["segments"]["mean_spend"].is_monotonic_decreasing
    np.testing.assert_allclose(out["thresholds"], [30, 60], atol=1.5)


def test_find_cv_cuts_respects_min_share():
    cv = np.linspace(5, 100, 200)
    # El salto está en el 3 % inicial, por debajo del mínimo por segmento.
    spend = np.where(cv < 8, 1000.0, 10.0) + np.linspace(1, 0, 200)
    out = find_cv_cuts(cv, spend, n_cuts=2, min_share=0.10)
    assert (out["segments"]["share_items"] >= 0.10).all()
    assert out["segments"]["share_items"].sum() == pytest.approx(1.0)


def test_find_cv_cuts_one_cut_and_custom_labels():
    cv, spend = _cv_spend_three_levels()
    out = find_cv_cuts(cv, spend, n_cuts=1, labels=["bajo", "alto"])
    assert len(out["thresholds"]) == 1
    assert out["segments"]["class"].tolist() == ["bajo", "alto"]


def test_find_cv_cuts_ignores_nan_and_matches_classify_xyz():
    cv, spend = _cv_spend_three_levels()
    cv_nan = np.r_[cv, np.nan, np.nan]
    out = find_cv_cuts(cv_nan, np.r_[spend, 1.0, 1.0])
    classes = classify_xyz(pd.Series(cv), out["thresholds"])
    counts = classes.value_counts().reindex(["X", "Y", "Z"]).tolist()
    assert counts == out["segments"]["n_items"].tolist()


def test_find_cv_cuts_fails_without_descending_partition():
    cv = np.linspace(5, 100, 300)
    with pytest.raises(ValueError, match="descendente"):
        find_cv_cuts(cv, cv * 10, n_cuts=2)  # el gasto sube con el CV


def test_find_cv_cuts_fails_when_segments_do_not_fit():
    with pytest.raises(ValueError, match="No caben"):
        find_cv_cuts(np.arange(10.0), np.arange(10.0)[::-1], n_cuts=2, min_share=0.5)


def _longest_zero_run(x: np.ndarray) -> np.ndarray:
    """Racha de ceros más larga de cada fila de una matriz booleana (True = cero)."""
    best = np.zeros(x.shape[0], dtype=int)
    cur = np.zeros(x.shape[0], dtype=int)
    for col in x.T:
        cur = np.where(col, cur + 1, 0)
        best = np.maximum(best, cur)
    return best


@pytest.mark.parametrize("p0, n", [(0.2, 200), (0.5, 120), (0.05, 400)])
def test_longest_run_threshold_matches_monte_carlo(p0, n):
    alpha = 0.05
    k = longest_run_threshold(p0, n, alpha)
    sims = _longest_zero_run(np.random.default_rng(0).random((20_000, n)) < p0)
    assert (sims >= k).mean() <= alpha + 0.01
    assert (sims >= k - 1).mean() > alpha - 0.01


def test_longest_run_threshold_low_p0_and_undefined_cases():
    assert longest_run_threshold(0.01, 1450, 0.01) == 3
    assert longest_run_threshold(0.0, 100, 0.01) is None
    assert longest_run_threshold(1.0, 100, 0.01) is None
    assert longest_run_threshold(0.9, 5, 0.01) is None  # ningún k ≤ n cumple


def _zero_agg():
    units = np.array(
        [
            [np.nan, np.nan, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 4, 1, 0, 0],
            [5, 0, 1] * 5 + [0, 0],
            [np.nan] * 17,
            [0] * 17,
        ],
        dtype=float,
    )
    return {"item_ids": np.array(["a", "b", "c", "d"]), "item_units": units}


def test_zero_run_thresholds_fallbacks_and_reasons():
    out = zero_run_thresholds(_zero_agg(), 1, 17, 0.01, 7).set_index("item_id")
    assert out.loc["a", "n_days"] == 11  # de la primera venta (d_5) a la última (d_15)
    assert out.loc["a", "p0"] == pytest.approx(6 / 11)
    assert out.loc["c", "reason"] == "no_sales" and out.loc["d", "reason"] == "no_sales"
    assert (out.loc[["c", "d"], "threshold"] == 7).all()
    assert (out.loc[["c", "d"], "source"] == "fallback").all()
    no_zero = {"item_ids": np.array(["e"]), "item_units": np.ones((1, 20))}
    row = zero_run_thresholds(no_zero, 1, 20, 0.01, 7).iloc[0]
    assert row["reason"] == "no_zeros" and row["threshold"] == 7


def test_mask_zero_runs_internal_and_edges():
    agg = _zero_agg()
    thr = pd.Series({"a": 3, "b": 3})
    out = mask_zero_runs(agg["item_units"], agg["item_ids"], thr, 1, 17)
    a = out[0]
    assert np.isnan(a[:4]).all()  # prelanzamiento (ya NaN) y arranque d_3–d_4
    assert a[6] == 0  # cero suelto interno: se conserva
    assert np.isnan(a[8:13]).all()  # racha interna de 5 ≥ 3
    assert np.isnan(a[15:]).all()  # ceros finales
    assert np.array_equal(out[1][:15], agg["item_units"][1][:15])  # ceros sueltos de b
    only_internal = mask_zero_runs(
        agg["item_units"], agg["item_ids"], thr, 1, 17, kinds=("leading", "internal")
    )
    assert (only_internal[0, 15:] == 0).all()
    assert np.isnan(only_internal[0, 8:13]).all()
    untouched = mask_zero_runs(agg["item_units"], agg["item_ids"], pd.Series({"a": 6}), 1, 17)
    assert (untouched[0, 8:13] == 0).all()  # 5 < 6


def test_run_train_matrix_declares_before_freezing(staged):
    from src.freeze import load_manifest
    from src.segmentation import TRAIN_DECLARATION_KEYS

    manifest = load_manifest()
    declared = {d["key"]: d["declared_at"] for d in manifest["declarations"]}
    assert set(TRAIN_DECLARATION_KEYS) <= set(declared)
    frozen_at = manifest["artifacts"]["zero_thresholds"]["frozen_at"]
    assert all(declared[k] < frozen_at for k in TRAIN_DECLARATION_KEYS)
    assert {"zero_thresholds", "train_cuts", "matrix_train"} <= set(manifest["artifacts"])
