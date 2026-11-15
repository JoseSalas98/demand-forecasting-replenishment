import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data import active_stores_by_item
from src.segmentation import (
    classical_seasonal_index,
    classify_abc,
    classify_abc_share,
    describe_items,
    drop_zero_runs,
    find_cv_cuts,
    friedman_seasonality,
    monthly_seasonal_index,
    seasonal_strength,
    select_abc_cuts,
    zero_breakdown,
    zero_runs,
)
from src.stability import (
    CELL_LABELS,
    base_target_classes,
    interleaved_halves,
    period_pairs,
    psi,
    transition_stats,
)


def test_zero_share_uses_active_days_only():
    units = np.array([[np.nan, np.nan, 0.0, 2.0, 0.0, 4.0], [1.0, 1.0, 1.0, 1.0, 1.0, 9.0]])
    agg = {"item_ids": np.array(["a", "b"], dtype=object), "item_units": units}
    out = describe_items(agg, 1, 6).set_index("item_id")
    assert out.loc["a", "active_days"] == 4
    assert out.loc["a", "zero_share"] == 0.5
    assert out.loc["a", "mean_daily_units"] == 1.5
    assert out.loc["a", "median_daily_units"] == 1.0
    assert out.loc["b", "median_daily_units"] == 1.0
    assert out.loc["b", "mean_daily_units"] == 14.0 / 6
    assert out.loc["b", "total_units"] == 14.0


# Gasto de 10 artículos (total 100). % acumulado antes de cada uno: 0, 40, 60, 70, 78, 84, ...
BASE_SPEND = np.array([40.0, 20.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 2.0])
BASE_IDS = np.array([f"i{k:02d}" for k in range(1, 11)], dtype=object)


def test_select_abc_cuts_picks_feasible_rule():
    # A = 2 artículos (20 %, 60 % del gasto) y B = 3 artículos (30 %, 24 %) es la única
    # partición sin violación; gana el par de cortes más bajo que la produce.
    out = select_abc_cuts(BASE_IDS, BASE_SPEND)
    assert out["cuts"] == [0.41, 0.79]
    best = out["scenarios"].iloc[0]
    assert best["violation"] == 0
    assert best["A_items_pct"] == pytest.approx(20) and best["A_spend_pct"] == pytest.approx(60)
    assert best["B_items_pct"] == pytest.approx(30) and best["B_spend_pct"] == pytest.approx(24)
    classes = out["abc"].set_index("item_id")["abc_class"]
    assert list(classes.loc[BASE_IDS]) == list("AABBBCCCCC")


def test_select_abc_cuts_minimizes_violation_without_exact_solution():
    # Con A de 0–5 % de artículos no hay solución: 2 artículos A (20 %, 60 %) violan 15 puntos,
    # menos que 1 (5 + 20) o 3 (25 + 0).
    out = select_abc_cuts(BASE_IDS, BASE_SPEND, item_bounds=((0.0, 0.05), (0.20, 0.30)))
    assert out["scenarios"].iloc[0]["violation"] == pytest.approx(15)
    assert 0.40 < out["cuts"][0] <= 0.60


def test_share_bounds_reproduce_base_classes():
    out = select_abc_cuts(BASE_IDS, BASE_SPEND)
    assert out["share_bounds"] == pytest.approx([0.15, 0.055])
    by_share = classify_abc_share(BASE_SPEND, out["share_bounds"])
    classes = out["abc"].set_index("item_id")["abc_class"].loc[BASE_IDS]
    assert list(by_share) == list(classes)


def _base_target_fixture():
    """Base Y1 (días 1–56) y objetivo Y2 (días 57–112) con 11 artículos.

    `i10` deja de tener dato en el objetivo y `i11` no existe en la base.
    """
    ids = np.array([f"i{k:02d}" for k in range(1, 12)], dtype=object)
    units = np.ones((11, 112))
    units[10, :56] = np.nan
    units[9, 59] = np.nan
    target_spend = [20, 20, 20, 20, 5, 5, 4, 3, 2, np.nan, 1]
    base_cv = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, np.nan]
    target_cv = [50, 10, 15, 20, 25, 30, 35, 40, 45, np.nan, 12]
    mats = pd.concat(
        [
            pd.DataFrame(
                {"set_id": "Y1", "item_id": ids, "spend": [*BASE_SPEND, np.nan], "cv": base_cv}
            ),
            pd.DataFrame({"set_id": "Y2", "item_id": ids, "spend": target_spend, "cv": target_cv}),
        ],
        ignore_index=True,
    )
    agg = {"item_ids": ids, "item_units": units}
    ranges = {"Y1": (1, 56), "Y2": (57, 112)}
    return mats, agg, ranges


PSI_CFG = {"psi_epsilon": 1e-4, "psi_bands": [0.10, 0.25]}
CV_KW = {"n_cuts": 2, "min_share": 0.2, "resolution": 100}


def test_base_target_uses_base_bounds_and_base_cv_cuts():
    mats, agg, ranges = _base_target_fixture()
    out = base_target_classes(mats, ranges, ["Y1"], "Y2", agg, {}, PSI_CFG, CV_KW)
    items = out["items"].set_index("item_id")
    assert items.loc["i11", "status"] == "new"
    assert items.loc["i10", "status"] == "no_target"
    assert out["summary"]["n_compared"] == 9
    # i04 tiene 20 % del gasto del objetivo: con la frontera de la base (15 %) es A, aunque
    # reclasificar el objetivo por % acumulado con los cortes de la base lo dejaría en B.
    assert items.loc["i04", "tgt_abc"] == "A"
    tgt = items[items["tgt_spend"].notna()]
    reranked = classify_abc(tgt.index, tgt["tgt_spend"].to_numpy(), [0.41, 0.79])
    assert reranked.set_index("item_id").loc["i04", "abc_class"] == "B"
    # Los umbrales XYZ salen solo de la base.
    base = items.iloc[:10]
    expected = find_cv_cuts(base["ref_cv_mean"], base["ref_spend_mean"], **CV_KW)["thresholds"]
    assert [out["summary"]["cv_X"], out["summary"]["cv_Z"]] == pytest.approx(expected)


def test_base_target_psi_uses_item_share_of_cells():
    mats, agg, ranges = _base_target_fixture()
    out = base_target_classes(mats, ranges, ["Y1"], "Y2", agg, {}, PSI_CFG, CV_KW)
    items = out["items"]
    both = items[(items["status"] == "both") & items["ref_cell"].notna()]
    p_ref = both["ref_cell"].value_counts(normalize=True).reindex(CELL_LABELS, fill_value=0)
    p_tgt = both["tgt_cell"].value_counts(normalize=True).reindex(CELL_LABELS, fill_value=0)
    eps = PSI_CFG["psi_epsilon"]
    r, c = np.maximum(p_ref.to_numpy(), eps), np.maximum(p_tgt.to_numpy(), eps)
    assert out["summary"]["psi"] == pytest.approx(float(((c - r) * np.log(c / r)).sum()))
    assert (p_ref == 0).any()  # hay celdas vacías: se usa ε
    assert out["summary"]["psi"] == pytest.approx(psi(p_ref.to_dict(), p_tgt.to_dict(), eps))
    props = out["proportions"].set_index("cell")
    assert props["base_pct"].sum() == pytest.approx(100)


def test_base_target_without_xyz_has_no_psi():
    mats, agg, ranges = _base_target_fixture()
    out = base_target_classes(mats, ranges, ["Y1"], "Y2", agg, {}, PSI_CFG, None)
    assert np.isnan(out["summary"]["psi"])
    assert out["items"].set_index("item_id").loc["i01", "ref_cell"] == "A"


def _zero_agg():
    nan = np.nan
    units = np.array(
        [
            [nan, 0, 0, 3, 0, 1, 0, 0, 0, nan, 0, 2, 0, 0],
            [1] * 14,
            [0] * 14,
        ],
        dtype=float,
    )
    return {"item_ids": np.array(["a", "b", "c"], dtype=object), "item_units": units}


def test_zero_breakdown_splits_runs_by_position():
    agg = _zero_agg()
    runs = zero_runs(agg, 1, 14)
    # El NaN del día 10 no corta la racha 7–11.
    long = runs[(runs["item_id"] == "a") & (runs["length"] == 4)].iloc[0]
    assert (long["start_d"], long["end_d"], long["position"]) == (7, 11, "internal")
    out = zero_breakdown(agg, 1, 14, long_run=3).set_index("item_id")
    a = out.loc["a"]
    assert (a["active_days"], a["zero_days"]) == (12, 9)
    assert (a["leading"], a["short"], a["long_run"], a["trailing"]) == (2, 1, 4, 2)
    assert (a["n_long_runs"], a["max_internal_run"]) == (1, 4)
    assert out.loc["b", "zero_days"] == 0
    assert out.loc["c", "leading"] == 14
    parts = out[["leading", "short", "long_run", "trailing"]].sum(axis=1)
    assert (parts == out["zero_days"]).all()


def test_drop_zero_runs_masks_selected_kinds():
    agg = _zero_agg()
    runs = zero_runs(agg, 1, 14)
    out = drop_zero_runs(agg["item_units"], agg["item_ids"], runs, {"long_run"}, long_run=3)
    assert np.isnan(out[0, 6:11]).all()
    assert out[0, 1] == 0 and out[0, 12] == 0
    assert np.array_equal(out[1], agg["item_units"][1])


def test_active_stores_by_item_counts_valid_cells():
    units = np.array([[1.0, np.nan], [0.0, 2.0], [np.nan, np.nan]])
    series_index = pd.DataFrame({"store_id": ["S1", "S1", "S2"], "item_id": ["b", "a", "a"]})
    out = active_stores_by_item(units, series_index)
    assert out.tolist() == [[1, 1], [1, 0]]


def test_seasonal_strength_separates_seasonal_from_noise():
    days = np.arange(364)
    rng = np.random.default_rng(0)
    seasonal = 10 + 5 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 0.1, len(days))
    assert seasonal_strength(seasonal, 7)["fs"] > 0.9
    # El ruido blanco no da 0: STL absorbe parte del ruido en la estacionalidad (~0,4 aquí).
    noise = rng.normal(10, 1, len(days))
    assert seasonal_strength(noise, 7)["fs"] < 0.5
    assert np.isnan(seasonal_strength(np.ones(10), 7)["fs"])


def test_seasonal_strength_trims_edges_and_interpolates():
    x = np.r_[np.zeros(5), 10 + np.sin(np.arange(28)), np.zeros(3)]
    x[10] = np.nan
    out = seasonal_strength(x, 7)
    assert out["n_obs"] == 28
    assert out["n_interpolated"] == 1


def test_monthly_seasonal_index_uses_daily_means():
    agg = {
        "item_ids": np.array(["a", "b"], dtype=object),
        "item_units": np.array([[1.0, 1.0, 3.0, 3.0], [np.nan, 1.0, 1.0, 1.0]]),
    }
    groups = pd.Series({"a": "AX", "b": "AX"})
    out = monthly_seasonal_index(agg, groups, np.array([1, 1, 2, 2]), {"Y1": (1, 4)})
    by_month = out.set_index("month")
    assert by_month.loc[1, "index"] == pytest.approx(0.5)
    assert by_month.loc[2, "index"] == pytest.approx(1.5)
    assert (by_month["n_items"] == 1).all()  # b no está activo todo el año


def _classical_case(daily_by_item: dict, classes: dict, dates):
    agg = {
        "item_ids": np.array(list(daily_by_item), dtype=object),
        "item_units": np.vstack(list(daily_by_item.values())),
    }
    return classical_seasonal_index(agg, pd.Series(classes), dates, 1, len(dates))


def test_classical_seasonal_index_recovers_pattern_under_trend():
    # Cinco años que empiezan a mitad de mes; tendencia lineal × patrón mensual conocido.
    dates = pd.date_range("2011-01-15", "2016-01-20", freq="D")
    pattern = np.array([0.8, 0.85, 0.9, 1.0, 1.05, 1.1, 1.2, 1.15, 1.05, 1.0, 0.95, 0.95])
    pattern = pattern * 12 / pattern.sum()
    trend = 100 + 0.02 * np.arange(len(dates))
    daily = trend * pattern[dates.month - 1]
    out = _classical_case({"a": daily, "b": daily / 2}, {"a": "A", "b": "A"}, dates)
    assert out["index"].loc["A"].sum() == pytest.approx(12)
    assert out["index"].loc["A"].to_numpy() == pytest.approx(pattern, abs=0.01)
    # Los meses incompletos (ene-2011 y ene-2016) no entran: primer ratio = feb-2011 + 6 meses.
    assert str(out["ratios"]["period"].min()) == "2011-08"
    assert str(out["ratios"]["period"].max()) == "2015-06"


def test_classical_seasonal_index_uses_daily_means_and_full_items():
    # Venta diaria constante: sin estacionalidad, aunque los meses tengan 28–31 días.
    dates = pd.date_range("2012-01-01", "2014-12-31", freq="D")
    flat = np.full(len(dates), 5.0)
    launched = flat.copy()
    launched[:100] = np.nan
    out = _classical_case(
        {"a": flat, "b": launched, "c": flat, "d": flat * 3},
        {"a": "X", "b": "X", "c": "-", "d": "Y"},
        dates,
    )
    assert out["index"].to_numpy() == pytest.approx(np.ones((2, 12)))
    assert out["n_items"].to_dict() == {"X": 1, "Y": 1}  # b (lanzado) y c (CV indefinido) fuera


def _ratios_frame(values: np.ndarray, first: str, cls: str = "A") -> pd.DataFrame:
    period = pd.period_range(first, periods=len(values), freq="M")
    return pd.DataFrame({"class": cls, "period": period, "month": period.month, "ratio": values})


def test_friedman_seasonality_matches_scipy_and_drops_incomplete_years():
    rng = np.random.default_rng(1)
    values = rng.normal(1, 0.05, 47)  # dic-2011 a oct-2015: solo 2012–2014 completos
    out = friedman_seasonality(_ratios_frame(values, "2011-12"), n_perm=200).loc["A"]
    full = values[1:37].reshape(3, 12)
    expected = stats.friedmanchisquare(*full.T)
    assert out["n_years"] == 3
    assert out["chi2"] == pytest.approx(expected.statistic)
    assert out["p_asymptotic"] == pytest.approx(expected.pvalue)


def test_friedman_seasonality_detects_a_repeated_pattern():
    pattern = np.linspace(0.8, 1.2, 12)
    out = friedman_seasonality(_ratios_frame(np.tile(pattern, 4), "2012-01"), n_perm=2000).loc["A"]
    assert out["kendall_w"] == pytest.approx(1)
    assert out["p_permutation"] < 0.001


def test_interleaved_halves_splits_odd_and_even_weeks():
    # Periodo de 4 semanas (días 1–28); el gasto diario es el número de semana.
    spend = np.repeat([1.0, 2.0, 3.0, 4.0], 7)[None, :].repeat(2, axis=0)
    spend[1, 0] = np.nan
    agg = {"item_ids": np.array(["a", "b"], dtype=object), "item_spend": spend}
    wod = np.repeat([11101, 11102, 11103, 11104], 7)
    out = interleaved_halves(agg, "Y1-Q1", 1, 28, wod).set_index(["set_id", "item_id"])
    assert out.loc[("Y1-Q1-odd", "a"), "spend"] == 7 * (1 + 3)
    assert out.loc[("Y1-Q1-even", "a"), "spend"] == 7 * (2 + 4)
    assert out.loc[("Y1-Q1-odd", "b"), "spend"] == 7 * (1 + 3) - 1
    assert out["cv"].isna().all()
    # Sin `week_of_day` se usan bloques de 7 días y el resultado es el mismo.
    plain = interleaved_halves(agg, "Y1-Q1", 1, 28).set_index(["set_id", "item_id"])
    assert plain["spend"].equals(out["spend"])


def test_period_pairs_adjacent_crosses_year_and_same_last_year():
    pairs = period_pairs(["Y1", "Y2"], 4, "Q")
    adj = pairs[pairs["comparison"] == "adjacent"]
    same = pairs[pairs["comparison"] == "same_last_year"]
    assert len(adj) == 7
    assert ("Y1-Q4", "Y2-Q1") in set(zip(adj["base_id"], adj["target_id"]))
    assert list(zip(same["base_id"], same["target_id"])) == [
        (f"Y1-Q{k}", f"Y2-Q{k}") for k in range(1, 5)
    ]


def _items(ref, tgt, status=None):
    n = len(ref)
    return pd.DataFrame(
        {
            "status": status or ["both"] * n,
            "ref_abc": ref,
            "tgt_abc": tgt,
            "ref_spend_mean": np.arange(n, 0, -1, dtype=float),
            "tgt_spend": np.arange(n, 0, -1, dtype=float),
        }
    )


def test_transition_stats_identity_and_full_swap():
    same = transition_stats(_items(list("AABBCC"), list("AABBCC")))
    assert same["keep_pct"] == 100 and same["mobility"] == 0
    assert same["kappa"] == pytest.approx(1) and same["spearman"] == pytest.approx(1)
    # A → B, B → C, C → A: nadie se queda; Shorrocks = (3 − 0) / 2.
    swap = transition_stats(_items(list("ABC"), list("BCA")))
    assert swap["keep_pct"] == 0 and swap["mobility"] == pytest.approx(1.5)
    assert swap["down_pct"] == pytest.approx(200 / 3) and swap["up_pct"] == pytest.approx(100 / 3)


def test_transition_stats_stuart_maxwell_and_shift():
    same = transition_stats(_items(list("AABBCC"), list("AABBCC")))
    assert same["stuart_maxwell"] == 0 and same["sm_pvalue"] == 1 and same["max_shift_pp"] == 0
    # 10 de 30 pasan de A a B: A pierde 33,3 pp; los cambios en un solo sentido dan p < 0,01.
    ref, tgt = list("A" * 20 + "B" * 5 + "C" * 5), list("B" * 10 + "A" * 10 + "B" * 5 + "C" * 5)
    out = transition_stats(_items(ref, tgt))
    assert out["max_shift_pp"] == pytest.approx(100 / 3) and out["max_shift_class"] in {"A", "B"}
    assert out["sm_pvalue"] < 0.01 and out["sm_df"] == 2


def test_transition_stats_levels_xyz_and_cell():
    items = _items(list("AAB"), list("AAB")).assign(
        ref_xyz=list("XYZ"), tgt_xyz=list("XZZ"), ref_cv_mean=[5.0, 15.0, 40.0],
        tgt_cv=[6.0, 30.0, 45.0], ref_cell=["AX", "AY", "BZ"], tgt_cell=["AX", "AZ", "BZ"],
    )
    xyz = transition_stats(items, level="xyz")
    assert xyz["keep_pct"] == pytest.approx(200 / 3) and xyz["down_pct"] == pytest.approx(100 / 3)
    assert xyz["spearman"] == pytest.approx(1)
    assert np.isnan(xyz["mixed_pct"])
    cell = transition_stats(items, level="cell")
    assert cell["keep_pct"] == pytest.approx(200 / 3) and np.isnan(cell["spearman"])


def test_transition_stats_cell_moves_by_dimension():
    # AX → AX mantiene; AZ → AY mejora; BX → AY mejora en ABC y baja en XYZ (mixto);
    # AY → BY empeora; AZ → BX es mixto, aunque en el orden AX…CZ quedaría "debajo".
    ref, tgt = ["AX", "AZ", "BX", "AY", "AZ"], ["AX", "AY", "AY", "BY", "BX"]
    items = pd.DataFrame({"status": ["both"] * 5, "ref_cell": ref, "tgt_cell": tgt})
    out = transition_stats(items, level="cell")
    assert out["keep_pct"] == pytest.approx(20) and out["up_pct"] == pytest.approx(20)
    assert out["down_pct"] == pytest.approx(20) and out["mixed_pct"] == pytest.approx(40)


def test_transition_stats_uses_only_both():
    items = _items(list("AABC"), list("ABBC"), status=["both", "both", "new", "both"])
    out = transition_stats(items)
    assert out["n"] == 3
    assert out["keep_pct"] == pytest.approx(200 / 3)
    assert out["down_pct"] == pytest.approx(100 / 3)
