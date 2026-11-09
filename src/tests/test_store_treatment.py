import numpy as np
import pandas as pd
import pytest

from src.store_treatment import (
    add_store_fill,
    cut_item_bounds,
    exclude_long_runs,
    extreme_blocks,
    fill_store_breaks,
    flag_store_breaks,
    long_break_runs,
    store_block_summary,
    store_units,
    store_zero_thresholds,
)


def _frame(rows, start_d=1, stores=None):
    arr = np.asarray(rows, dtype=float)
    stores = stores or [("I1", f"S{i}") for i in range(len(arr))]
    index = pd.MultiIndex.from_tuples(stores, names=["item_id", "store_id"])
    return pd.DataFrame(arr, index=index, columns=[f"d_{d}" for d in range(start_d, start_d + arr.shape[1])])


def _thr(units, k, p0=np.nan):
    return pd.DataFrame({"n": 0, "p0": p0, "threshold": k, "source": "estimated", "reason": ""}, index=units.index)


def test_store_units_selects_items_and_days():
    idx = pd.DataFrame({"store_id": ["S1", "S2", "S1"], "item_id": ["I1", "I1", "I2"]})
    interim = {"series_index": idx, "units": np.arange(15, dtype=float).reshape(3, 5)}
    out = store_units(interim, ["I1"], 2, 4)
    assert list(out.index) == [("I1", "S1"), ("I1", "S2")]
    assert list(out.columns) == ["d_2", "d_3", "d_4"]
    assert out.iloc[1].tolist() == [6, 7, 8]


def test_store_zero_thresholds_uses_train_only_and_fallback():
    rng = np.random.default_rng(0)
    x = (rng.random(400) > 0.3).astype(float)
    x[0] = x[-1] = 1
    u = _frame([np.r_[x, np.zeros(50)], np.r_[np.zeros(399), 1.0, np.ones(50)]])
    out = store_zero_thresholds(u, train_end=400, alpha=0.01, fallback=7)
    assert out.loc[("I1", "S0"), "source"] == "estimated" and out.loc[("I1", "S0"), "n"] == 400
    assert out.loc[("I1", "S1"), "source"] == "fallback" and out.loc[("I1", "S1"), "reason"] == "few_sales"
    assert out.loc[("I1", "S1"), "threshold"] == 7


def test_flag_store_breaks_history_vs_causal():
    days = 112  # 4 bloques; la ventana empieza en el día 57 (bloque 3)
    x = np.ones(days)
    x[10:20] = 0          # racha de 10 en la historia que vuelve antes de la ventana → toda marcada
    x[30:33] = 0          # racha corta → intermitencia
    x[60:90] = 0          # racha en la ventana: se marca solo desde que, al cierre del bloque, ya mide ≥ 5
    x[100:] = 0           # racha final del bloque 4 sin regreso: se marca porque al cierre mide 12 ≥ 5
    u = _frame([x])
    out = flag_store_breaks(u, _thr(u, 5), pd.Series({"I1": 57}), alpha=0.01).to_numpy()[0]
    assert (out[10:20] == 1).all() and (out[30:33] == 0).all()
    assert (out[60:90] == 2).all()          # días 61–84 (bloque 3, mide 24 al cierre) y 85–90 (bloque 4)
    assert (out[100:] == 2).all()


def test_flag_store_breaks_causal_short_run_at_block_end_is_not_flagged():
    x = np.ones(56)
    x[25:28] = 0  # 3 días al cierre del bloque 1 y la tienda vuelve en el bloque 2 (ventana)
    u = _frame([x])
    out = flag_store_breaks(u, _thr(u, 5), pd.Series({"I1": 29}), alpha=0.01).to_numpy()[0]
    assert (out == 0).all()


def test_flag_store_breaks_excess_zeros_and_leading_zeros():
    x = np.ones(84)
    x[:5] = 0                 # antes de la primera venta: nunca es quiebre
    x[[30, 33, 36, 39, 42, 45, 48, 51]] = 0  # bloque 2: 8 ceros sueltos, p0 = 0.05 → exceso
    u = _frame([x])
    out = flag_store_breaks(u, _thr(u, 30, p0=0.05), pd.Series({"I1": 85}), alpha=0.01).to_numpy()[0]
    assert (out[:5] == 0).all()
    assert (out[[30, 33, 36, 39, 42, 45, 48, 51]] == 3).all() and out.sum() == 8 * 3


def test_fill_store_breaks_uses_past_valid_days_only():
    x = np.r_[np.full(28, 2.0), np.zeros(5), np.full(5, 9.0)]
    u = _frame([x, np.r_[np.zeros(3), np.ones(35)]])
    brk = pd.DataFrame(0, index=u.index, columns=u.columns, dtype=np.int8)
    brk.iloc[0, 28:33] = 1
    brk.iloc[1, 0:3] = 1   # sin 28 días válidos previos → no se completa
    filled, added = fill_store_breaks(u, brk, rate_days=28)
    assert np.allclose(filled.iloc[0, 28:33], 2.0) and np.allclose(added.iloc[0, 28:33], 2.0)
    assert added.iloc[1, 0:3].isna().all() and (filled.iloc[1, 0:3] == 0).all()
    assert (added.iloc[0, :28] == 0).all()
    s = store_block_summary(u.iloc[:, :28], brk.iloc[:, :28], added.iloc[:, :28])
    assert s.loc[1, "unfilled_days"] == 3 and s.loc[1, "break_days"] == 3


def test_add_store_fill_only_on_observed_item_days():
    item = pd.DataFrame([[5.0, np.nan, 4.0]], index=pd.Index(["I1"], name="item_id"), columns=["d_1", "d_2", "d_3"])
    added = _frame([[1.0, 2.0, np.nan], [0.5, 0.0, 0.0]])
    out = add_store_fill(item, added)
    assert out.iloc[0, 0] == 6.5 and np.isnan(out.iloc[0, 1]) and out.iloc[0, 2] == 4.0


def test_extreme_blocks_kinds():
    b = pd.DataFrame({"item_id": "I1", "block": [1, 2, 3, 4], "units": [10.0, 30.0, 30.0, 10.0],
                      "split": ["train", "train", "train", "test"], "backtest": [False, False, True, False]})
    cases = pd.DataFrame({"item_id": ["I1"], "block": [2], "group": ["estructural"]})
    out = extreme_blocks(b, cases, pd.Series({"I1": "AZ"}))
    assert out["block"].tolist() == [2, 4]
    assert out["kind"].tolist() == ["estructural", "atipico"] and out["section"].tolist() == ["historia", "test"]


def test_flag_store_breaks_requires_window_start():
    u = _frame([np.ones(28)])
    with pytest.raises(ValueError, match="ventana"):
        flag_store_breaks(u, _thr(u, 5), pd.Series({"I9": 1}), alpha=0.01)


def test_add_store_fill_completes_total_breaks_only_when_masked():
    item = pd.DataFrame([[5.0, np.nan, np.nan, np.nan]], index=pd.Index(["I1"], name="item_id"),
                        columns=["d_1", "d_2", "d_3", "d_4"])
    added = _frame([[0.0, 2.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0]])
    masked = pd.DataFrame([[False, True, True, False]], index=item.index, columns=item.columns)
    out = add_store_fill(item, added, masked)
    assert out.iloc[0, 1] == 3.0                 # quiebre total enmascarado con tasa → suma de las tiendas
    assert np.isnan(out.iloc[0, 2])              # enmascarado sin tasa → sigue NaN
    assert np.isnan(out.iloc[0, 3])              # NaN que no es quiebre (fuera del ciclo) → no cambia
    assert np.isnan(add_store_fill(item, added).iloc[0, 1])   # sin masked: comportamiento anterior


def test_share_deficit_test_flags_low_store_and_uses_others():
    from src.store_treatment import share_deficit_test, spread_block_extra
    idx = pd.MultiIndex.from_tuples([("I1", "S1"), ("I1", "S2"), ("I1", "S3")], names=["item_id", "store_id"])
    u = pd.DataFrame([[100, 100, 100, 100, 10.0], [100, 100, 100, 100, 100.0], [200, 200, 200, 200, 200.0]],
                     index=idx, columns=[1, 2, 3, 4, 5])
    days = pd.DataFrame(28, index=idx, columns=u.columns)
    fixed, table = share_deficit_test(u, days, window=6, min_ref=3, alpha=0.01, max_level=0.5)
    assert table[["store_id", "block"]].values.tolist() == [["S1", 5]]
    assert np.isclose(fixed.loc[("I1", "S1"), 5], 100.0)            # 0.25 × (300 / 0.75)
    assert (fixed.drop(columns=5) == u.drop(columns=5)).all().all()
    # Todas las tiendas bajan juntas: el reparto no cambia y no se toca nada.
    u2 = u.copy(); u2[5] = [50.0, 50.0, 100.0]
    _, t2 = share_deficit_test(u2, days, window=6, min_ref=3, alpha=0.01, max_level=0.5)
    assert t2.empty
    daily = spread_block_extra(_frame(np.zeros((3, 140)), stores=list(idx)), fixed - u)
    assert np.isclose(daily.iloc[0, 112:].sum(), 90.0) and daily.iloc[0, :112].sum() == 0


def _breaks(units, spans):
    b = pd.DataFrame(0, index=units.index, columns=units.columns, dtype=np.int8)
    for r, s, e in spans:
        b.iloc[r, s:e] = 1
    return b


def test_long_break_runs_generalized_cut_and_isolated_exclusion():
    # 3 tiendas, 10 bloques (280 días); la ventana empieza en el día 197 (bloque 8).
    u = _frame(np.ones((3, 280)))
    # Tiendas 0 y 1 a la vez en los días 20–89 (70 días): caída generalizada con 2 tiendas y racha > 56.
    # Tienda 2: racha aislada en la historia (días 100–169) y otra que cruza a la ventana (días 180–279).
    brk = _breaks(u, [(0, 20, 90), (1, 20, 90), (2, 100, 170), (2, 180, 280)])
    runs, cuts = long_break_runs(brk, pd.Series({"I1": 197}), min_days=56, chain_stores=2)
    assert len(runs) == 4 and (runs["kind"] == ["generalizada", "generalizada", "aislada", "aislada"]).all()
    assert cuts["cut_d"].tolist() == [113] and not cuts["no_return"].any()  # día 91 → primer bloque completo: día 113
    iso = runs[runs["kind"] == "aislada"].set_index("start_d")
    assert iso.loc[101, "excl_start_d"] == 101          # termina antes de la ventana: toda la racha
    # Cruza a la ventana: al cierre del bloque 7 (día 196) mide 16; del 8 (día 224), 44; del 9 (día 252), 72 > 56.
    assert iso.loc[181, "excl_start_d"] == 225
    units_x, breaks_x = exclude_long_runs(u, brk, runs)
    assert units_x.iloc[2, 100:170].isna().all() and units_x.iloc[2, 224:].isna().all()
    assert (units_x.iloc[2, 180:224] == 1).all() and (breaks_x.iloc[2, 180:224] == 1).all()  # se completan
    assert (breaks_x.iloc[2, 224:] == 0).all() and units_x.iloc[0].notna().all()            # generalizada: intacta
    bounds = pd.DataFrame({"first_sale_d": [1.0], "last_sale_d": [280.0]}, index=pd.Index(["I1"], name="item_id"))
    assert cut_item_bounds(bounds, cuts).loc["I1", "first_sale_d"] == 113


def test_long_break_runs_no_return_drops_item():
    u = _frame(np.ones((2, 112)))
    brk = _breaks(u, [(0, 40, 112), (1, 40, 112)])
    runs, cuts = long_break_runs(brk, pd.Series({"I1": 85}), min_days=28, chain_stores=2)
    assert cuts["no_return"].all() and cuts["cut_d"].isna().all()
    bounds = pd.DataFrame({"first_sale_d": [1.0], "last_sale_d": [112.0]}, index=pd.Index(["I1"], name="item_id"))
    assert cut_item_bounds(bounds, cuts).empty
