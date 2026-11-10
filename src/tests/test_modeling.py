import numpy as np
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



# --- SARIMAX: diferencias, búsqueda de órdenes y contraste -------------------------------------

from src.modeling import (  # noqa: E402
    fit_sarimax,
    kpss_d,
    ocsb_test,
    sarimax_differencing,
    sarimax_one_step,
    seasonal_acf,
    select_sarimax_item,
)


def test_ocsb_detects_seasonal_unit_root():
    rng = np.random.default_rng(0)
    e = rng.normal(size=240)
    walk = np.zeros(240)
    for t in range(240):
        walk[t] = (walk[t - 4] if t >= 4 else 0) + e[t]   # y_t = y_{t−4} + e: raíz unitaria estacional
    assert ocsb_test(walk, 4)["D"] == 1
    assert ocsb_test(e, 4)["D"] == 0
    assert ocsb_test(e[:10], 4) is None   # muestra insuficiente


def test_kpss_d_and_nan_handling():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=200))
    noise = rng.normal(size=200)
    noise[[5, 50]] = np.nan
    assert kpss_d(walk)["d"] == 1 and kpss_d(noise)["d"] == 0


def test_sarimax_differencing_feasibility():
    y = np.random.default_rng(2).normal(1, 0.1, size=20)
    out = sarimax_differencing(y, [3, 13]).set_index("s")
    assert out.loc[0, "feasible"] and out.loc[3, "feasible"]
    assert not out.loc[13, "feasible"]    # 20 < 2·13 + 1


def test_fit_sarimax_drops_constant_exog_and_small_samples():
    rng = np.random.default_rng(3)
    y = 1 + rng.normal(0, 0.1, size=40)
    X = np.column_stack([rng.integers(0, 2, 40), np.zeros(40)])
    fit = fit_sarimax(y, X, (1, 0, 0), (0, 0, 0, 0))
    assert fit["status"] == "ok" and fit["exog_cols"] == [0] and fit["k"] == 4   # AR, constante, exógena, varianza
    assert np.isclose(fit["aicc"], fit["aic"] + 2 * 4 * 5 / (40 - 4 - 1))
    assert fit_sarimax(y[:4], X[:4], (2, 0, 2), (0, 0, 0, 0))["status"] == "muestra_insuficiente"


def test_sarimax_one_step_uses_only_the_past():
    rng = np.random.default_rng(4)
    y = 1 + rng.normal(0, 0.1, size=30)
    X = rng.integers(0, 2, size=(30, 1)).astype(float)
    prev = np.full(30, 100.0)
    f1, _ = sarimax_one_step(y, X, prev, (1, 0, 0), (0, 0, 0, 0), start=24)
    y2 = y.copy()
    y2[24] = 9.0                                   # cambiar el bloque que se predice no cambia su pronóstico
    f2, _ = sarimax_one_step(y2, X, prev, (1, 0, 0), (0, 0, 0, 0), start=24)
    assert len(f1) == 6 and np.isclose(f1[0], f2[0]) and not np.isclose(f1[1], f2[1])


def _item_frame(n_pre=30, n_win=13, seed=5):
    rng = np.random.default_rng(seed)
    n = n_pre + n_win
    ratio = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n) / 3) + rng.normal(0, 0.05, size=n)
    return pd.DataFrame({"item_id": "I1", "block": np.arange(1, n + 1), "ratio": ratio,
                         "units": 100 * ratio, "units_prev": np.full(n, 100.0),
                         "n_religious": rng.integers(0, 2, n).astype(float),
                         "fold": [0] * n_pre + list(range(1, n_win + 1))})


def test_select_sarimax_item_uses_only_fold_zero():
    kw = dict(exog_cols=["n_religious"], s_list=[3], p_list=[0, 1], q_list=[0], seasonal_pq=[(1, 0)], n_valid=6)
    item = _item_frame()
    out = select_sarimax_item(item, **kw)
    assert out["choice"]["status"] == "ok" and out["choice"]["n_sel"] == 24
    assert set(out["winners"]["s"]) <= {0, 3} and out["winners"]["rmse_valid"].notna().all()
    changed = item.copy()
    changed.loc[changed["fold"] > 0, "ratio"] = 50.0   # la ventana no influye en la elección
    again = select_sarimax_item(changed, **kw)
    assert {k: again["choice"][k] for k in ("s", "p", "q", "P", "Q")} == {k: out["choice"][k] for k in ("s", "p", "q", "P", "Q")}


def test_seasonal_acf_flags_period():
    out = seasonal_acf(_item_frame(n_pre=60), lags=[3, 13], n_valid=6)
    assert out.loc[0, "sig_3"] and out.loc[0, "n"] == 54


def test_sarimax_backtest_item_refits_per_fold_without_test():
    from src.modeling import sarimax_backtest_item
    item = _item_frame()
    item = pd.concat([item, item.tail(2).assign(block=[44, 45], fold=pd.NA)], ignore_index=True)   # 2 bloques de test
    item["fold"] = item["fold"].astype("Int64")
    choice = {"status": "ok", "s": 0, "p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "params": None}
    bt, _ = sarimax_backtest_item(item, choice, ["n_religious"])
    assert bt["fold"].tolist() == list(range(1, 14)) and bt["sarimax"].notna().all()
    changed = item.copy()
    changed.loc[changed["fold"] == 13, "ratio"] = 50.0     # el bloque que predice el fold 13 no entra a su ajuste
    changed.loc[changed["fold"].isna(), "ratio"] = 50.0    # el test tampoco
    bt2, _ = sarimax_backtest_item(changed, choice, ["n_religious"])
    assert np.allclose(bt["sarimax"], bt2["sarimax"])
    none, _ = sarimax_backtest_item(item, {"status": "sin_modelo"}, ["n_religious"])
    assert none["sarimax"].isna().all() and len(none) == 13


# --- SARIMAX con puntos de control (guardado por lotes y reanudación) ---------------------------

_CKPT_KW = dict(exog_cols=["n_religious"], s_list=[3], p_list=[0, 1], q_list=[0], seasonal_pq=[(1, 0)], n_valid=6)


def _items_frame(n_items=3):
    return pd.concat([_item_frame(seed=s).assign(item_id=f"I{s}") for s in range(1, n_items + 1)], ignore_index=True)


def _core(choice):
    return choice.drop(columns=["seconds", "params"]).sort_values("item_id").reset_index(drop=True)


def test_checkpointed_matches_plain_search_and_resumes(tmp_path):
    from src.modeling import search_sarimax_checkpointed, search_sarimax_orders
    data = _items_frame()
    items = ["I1", "I2", "I3"]
    res = search_sarimax_checkpointed(data, items, tmp_path, batch_size=2, progress=False, **_CKPT_KW)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["batch_0000.pkl", "batch_0001.pkl"]   # 2 + 1
    assert res["meta"]["n_items"].tolist() == [2, 1]
    plain = search_sarimax_orders(data, items, backtest=True, **_CKPT_KW)
    pd.testing.assert_frame_equal(_core(res["choice"]), _core(plain["choice"]))
    pd.testing.assert_frame_equal(res["backtest"].sort_values(["item_id", "block"]).reset_index(drop=True),
                                  plain["backtest"].sort_values(["item_id", "block"]).reset_index(drop=True))
    # Reanuda: borrado el último lote, solo se recalculan sus artículos y el resultado es el mismo.
    lost = pd.read_pickle(tmp_path / "batch_0001.pkl")["meta"]["items"]
    (tmp_path / "batch_0001.pkl").unlink()
    again = search_sarimax_checkpointed(data, items, tmp_path, batch_size=2, progress=False, **_CKPT_KW)
    assert pd.read_pickle(tmp_path / "batch_0001.pkl")["meta"]["items"] == lost
    pd.testing.assert_frame_equal(_core(again["choice"]), _core(res["choice"]))
    # Con todo guardado no corre nada nuevo.
    search_sarimax_checkpointed(data, items, tmp_path, batch_size=2, progress=False, **_CKPT_KW)
    assert len(list(tmp_path.glob("batch_*"))) == 2


def test_checkpointed_rejects_other_config(tmp_path):
    from src.modeling import search_sarimax_checkpointed
    data = _items_frame(1)
    search_sarimax_checkpointed(data, ["I1"], tmp_path, progress=False, **_CKPT_KW)
    with pytest.raises(ValueError, match="otra configuración"):
        search_sarimax_checkpointed(data, ["I1"], tmp_path, progress=False, **{**_CKPT_KW, "maxiter": 50})


def test_checkpointed_rejects_other_number_of_items(tmp_path):
    from src.modeling import search_sarimax_checkpointed
    data = _items_frame(2)
    search_sarimax_checkpointed(data, ["I1"], tmp_path, progress=False, **_CKPT_KW)
    with pytest.raises(ValueError, match="otra configuración"):
        search_sarimax_checkpointed(data, ["I1", "I2"], tmp_path, progress=False, **_CKPT_KW)


def test_checkpointed_saves_finished_items_when_one_fails(tmp_path, monkeypatch):
    import src.modeling as m
    original = m._search_item

    def failing(df, backtest, kwargs):
        if df["item_id"].iloc[0] == "I3":
            raise RuntimeError("falla simulada")
        return original(df, backtest, kwargs)

    monkeypatch.setattr(m, "_search_item", failing)   # n_jobs=1: corre en este proceso
    with pytest.raises(RuntimeError, match="falla simulada"):
        m.search_sarimax_checkpointed(_items_frame(), ["I1", "I2", "I3"], tmp_path, batch_size=10,
                                      progress=False, **_CKPT_KW)
    saved = m.load_sarimax_checkpoints(tmp_path)
    assert sorted(saved["choice"]["item_id"]) == ["I1", "I2"]
    assert not list(tmp_path.glob(".tmp_*"))


def test_checkpointed_extension_sets_compression(tmp_path):
    from src.modeling import load_sarimax_checkpoints, search_sarimax_checkpointed
    search_sarimax_checkpointed(_items_frame(1), ["I1"], tmp_path, ext=".pkl.gz", progress=False, **_CKPT_KW)
    path = tmp_path / "batch_0000.pkl.gz"
    assert path.read_bytes()[:2] == b"\x1f\x8b"   # gzip
    assert load_sarimax_checkpoints(tmp_path, ext=".pkl.gz")["choice"]["item_id"].tolist() == ["I1"]
    assert load_sarimax_checkpoints(tmp_path, ext=".pkl") == {}


# --- Boosting global: backtesting y búsqueda ----------------------------------------------------

_XGB_FIXED = {"tree_method": "hist", "enable_categorical": True, "max_bin": 16}
_XGB_SPACE = {"max_depth": ("int", 1, 2), "n_estimators": ("int", 5, 15, 5), "learning_rate": ("float_log", 0.05, 0.3)}


def _boost_frame(n_items=3, n_pre=20, n_win=13, n_test=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_items):
        n = n_pre + n_win + n_test
        ratio = 1 + 0.1 * np.sin(np.arange(n)) + rng.normal(0, 0.02, n)
        fold = [0] * n_pre + list(range(1, n_win + 1)) + [pd.NA] * n_test
        rows.append(pd.DataFrame({"item_id": f"I{j}", "block": np.arange(1, n + 1), "ratio": ratio,
                                  "ratio_lag1": np.r_[np.nan, ratio[:-1]], "units": 100 * ratio,
                                  "units_prev": np.full(n, 100.0), "fold": pd.array(fold, dtype="Int64")}))
    d = pd.concat(rows, ignore_index=True)
    d["item_id"] = d["item_id"].astype("category")
    d.loc[d.index[5], "ratio"] = np.nan   # una fila tratada: no entrena
    return d


_LGB_FIXED = {"max_depth": 2, "max_bin": 15, "subsample_freq": 1, "deterministic": True, "force_row_wise": True,
              "verbose": -1, "min_child_samples": 3}


@pytest.mark.parametrize("algorithm,params", [
    ("xgboost", {**_XGB_FIXED, "max_depth": 2, "n_estimators": 10}),
    ("lightgbm", {**_LGB_FIXED, "num_leaves": 3, "n_estimators": 10, "subsample": 0.8}),
    ("catboost", {"depth": 2, "n_estimators": 10, "border_count": 16, "boosting_type": "Plain",
                  "bootstrap_type": "Bernoulli", "subsample": 0.8, "cat_features": ["item_id"]}),
])
def test_boosting_backtest_uses_only_the_past(algorithm, params):
    from src.modeling import boosting_backtest
    data = _boost_frame()
    bt = boosting_backtest(data, ["ratio_lag1", "item_id"], algorithm, params, seed=1, n_jobs=1)
    assert len(bt) == 3 * 13 and sorted(bt["fold"].unique()) == list(range(1, 14))
    assert np.allclose(bt["pred"], bt["units_prev"] * bt["ratio_hat"])
    changed = data.copy()
    changed.loc[changed["fold"].eq(13).fillna(False), "ratio"] = 50.0   # el objetivo del bloque que se predice
    changed.loc[changed["fold"].isna(), "ratio"] = 50.0                 # y el test
    bt2 = boosting_backtest(changed, ["ratio_lag1", "item_id"], algorithm, params, seed=1, n_jobs=1)
    assert np.allclose(bt["ratio_hat"], bt2["ratio_hat"])


def test_boosting_search_resumes_and_rejects_other_config(tmp_path):
    import optuna
    from src.modeling import boosting_search
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = _boost_frame()
    scales = pd.Series(1.0, index=[f"I{j}" for j in range(3)])
    weights = pd.Series([1.0, 2.0, 3.0], index=scales.index)
    kw = dict(data=data, features=["ratio_lag1", "item_id"], algorithm="xgboost", space=_XGB_SPACE, fixed=_XGB_FIXED,
              scales=scales, weights=weights, seed=7, n_jobs=1, storage=f"sqlite:///{tmp_path}/s.db",
              study_name="xgb", progress=False)
    study = boosting_search(n_trials=2, **kw)
    assert len(study.trials) == 2 and study.best_value > 0
    assert len(study.best_trial.user_attrs["wrmsse_fold"]) == 13
    study = boosting_search(n_trials=3, **kw)   # retoma: corre solo 1 más
    assert len(study.trials) == 3
    first = [t.params for t in study.trials]
    assert first[2] not in first[:2]
    with pytest.raises(ValueError, match="otra configuración"):
        boosting_search(n_trials=3, **{**kw, "fixed": {**_XGB_FIXED, "max_bin": 32}})


def test_suggest_params_kinds():
    import optuna
    from src.modeling import suggest_params
    space = {"a": ("int", 1, 3), "b": ("int", 10, 50, 10), "c": ("int_log", 5, 100), "d": ("float", 0.1, 0.9),
             "e": ("float_log", 0.01, 1.0)}
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    p = suggest_params(trial, space)
    assert p["a"] in (1, 2, 3) and p["b"] % 10 == 0 and 5 <= p["c"] <= 100 and isinstance(p["c"], int)
    assert 0.1 <= p["d"] <= 0.9 and 0.01 <= p["e"] <= 1.0
    with pytest.raises(ValueError, match="tipo desconocido"):
        suggest_params(trial, {"x": ("cat", 0, 1)})


def _window_base():
    return pd.DataFrame({"item_id": ["A", "A", "B", "B"], "block": [1, 2, 1, 2], "fold": [1, 2, 1, 2],
                         "units": [10.0, 12.0, 5.0, 7.0], "snaive": [9.0, 11.0, 6.0, 6.0]})


def test_window_forecasts_aligns_candidates():
    from src.modeling import window_forecasts
    base = _window_base()
    cand = pd.DataFrame({"item_id": ["B", "B", "A", "A"], "block": [2, 1, 2, 1], "units": [7.0, 5.0, 12.0, 10.0],
                         "m1": [7.5, np.nan, 12.5, 10.5]})
    out = window_forecasts(base, {"m1": cand})
    np.testing.assert_array_equal(out["m1"].to_numpy(), [10.5, 12.5, np.nan, 7.5])   # orden de la base; NaN se conserva
    assert list(out.columns) == list(base.columns) + ["m1"]


@pytest.mark.parametrize("change", ["missing", "units", "duplicate"])
def test_window_forecasts_rejects_mismatch(change):
    from src.modeling import window_forecasts
    cand = _window_base().drop(columns=["snaive", "fold"]).assign(m1=1.0)
    if change == "missing":
        cand = cand.iloc[:3]
    elif change == "units":
        cand.loc[0, "units"] = 99.0
    else:
        cand = pd.concat([cand.iloc[:3], cand.iloc[[0]]])
    with pytest.raises(ValueError):
        window_forecasts(_window_base(), {"m1": cand})


def test_select_by_item_filter_tie_and_missing():
    from src.modeling import select_by_item
    errors = pd.DataFrame({
        "item_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 2,
        "candidate": ["base", "m1", "m2"] * 3 + ["base", "m2"],
        "rmsse": [1.0, 0.8, 0.6,    # A: gana m2 (el menor)
                  0.5, 0.7, 0.9,    # B: nadie le gana a la base
                  1.0, 0.7, 0.7,    # C: empate m1 = m2 → el más simple
                  1.0, 1.2],        # D: sin m1; m2 no pasa el filtro
    })
    out = select_by_item(errors, ["base", "m1", "m2"]).set_index("item_id")
    assert out["winner"].to_dict() == {"A": "m2", "B": "base", "C": "m1", "D": "base"}
    assert out["winner_rmsse"].to_dict() == {"A": 0.6, "B": 0.5, "C": 0.7, "D": 1.0}
    assert np.isnan(out.loc["D", "m1"])
    with pytest.raises(ValueError):
        select_by_item(errors[errors["candidate"] != "base"], ["base", "m1", "m2"])


def test_selected_forecast_takes_winner_column():
    from src.modeling import selected_forecast
    window = _window_base().assign(m1=[1.0, 2.0, np.nan, np.nan])
    np.testing.assert_array_equal(selected_forecast(window, pd.Series({"A": "m1", "B": "snaive"})), [1, 2, 6, 6])
    with pytest.raises(ValueError):
        selected_forecast(window, pd.Series({"A": "m1", "B": "m1"}))   # B no tiene m1


class _LagModel:
    """Modelo de prueba: la tasa pronosticada es 1 + 0,1 × ratio_lag1."""
    def predict(self, X):
        return 1 + 0.1 * X["ratio_lag1"].to_numpy()


def _ratio_data():
    rows = []
    for item, base in [("A", 1.0), ("B", 2.0)]:
        for b in range(1, 7):
            fold = 0 if b <= 3 else b - 3
            rows.append({"item_id": item, "block": b, "fold": fold, "ratio": base + b / 10,
                         "ratio_treated": False, "n_x": b})
    return pd.DataFrame(rows)


def test_recursive_ratio_forecast_chains_predictions():
    from src.modeling import recursive_ratio_forecast
    data = _ratio_data()
    out = recursive_ratio_forecast(data, _LagModel(), ["ratio_lag1", "n_x"], lags=[1], windows=[], stats=[],
                                   first_fold=1, horizon=3).set_index(["item_id", "step"])["ratio_hat"]
    # Paso 1 usa la tasa real del origen (bloque 3); los pasos 2 y 3, la tasa pronosticada del paso anterior.
    a1 = 1 + 0.1 * 1.3
    a2 = 1 + 0.1 * a1
    assert np.isclose(out[("A", 1)], a1) and np.isclose(out[("A", 2)], a2) and np.isclose(out[("A", 3)], 1 + 0.1 * a2)
    assert np.isclose(out[("B", 1)], 1 + 0.1 * 2.3)
    # Nunca usa las tasas reales posteriores al origen.
    changed = data.assign(ratio=np.where(data["fold"] >= 1, 99.0, data["ratio"]))
    out2 = recursive_ratio_forecast(changed, _LagModel(), ["ratio_lag1", "n_x"], lags=[1], windows=[], stats=[],
                                    first_fold=1, horizon=3).set_index(["item_id", "step"])["ratio_hat"]
    pd.testing.assert_series_equal(out, out2)


def test_chain_units_compounds_from_origin():
    from src.modeling import chain_units
    path = pd.DataFrame({"item_id": ["A", "B", "A", "A", "B"], "step": [2, 1, 1, 3, 2],
                         "ratio_hat": [0.5, 2.0, 1.1, 2.0, 1.0]})
    out = chain_units(path, pd.Series({"A": 100.0, "B": 10.0}))
    # A: 100 × 1,1 = 110 → 110 × 0,5 = 55 → 55 × 2 = 110; B: 10 × 2 = 20 → 20 × 1 = 20 (orden de `path`).
    np.testing.assert_allclose(out, [55, 20, 110, 110, 20])
    with pytest.raises(ValueError):
        chain_units(path, pd.Series({"A": 100.0}))


def test_plot_forecast_item_rates_and_units():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.modeling import plot_forecast_item
    data = pd.DataFrame({"item_id": "A", "block": range(1, 6), "start_date": pd.date_range("2012-01-01", periods=5, freq="28D"),
                         "split": ["train"] * 4 + ["test"], "units": [10.0, 12, 12, 6, 99], "ratio_raw": [1.0, 1.2, 1.0, 0.5, 9.0],
                         "imputed": [False, True, False, False, False], "cell": "AX"})
    table = pd.DataFrame({"item_id": "A", "block": [3, 4], "step": [1, 2], "ratio_1step": [1.0, 0.5], "ratio_chain": [1.0, 1.0],
                          "units_1step": [12.0, 6.0], "units_chain": [12.0, 12.0]})
    fig = plot_forecast_item(data, table, "A", rate_scale=1.0, unit_scale=1.0)
    ax_rate, ax_rate_err, ax_units, ax_units_err = fig.axes
    assert 9.0 not in ax_rate.lines[0].get_ydata() and 99.0 not in ax_units.lines[0].get_ydata()   # sin test
    labels = [t.get_text() for t in ax_units.get_legend().get_texts()]
    assert any("a un paso (WAPE 0.0%" in s for s in labels) and any("encadenado (WAPE 33.3%" in s for s in labels)   # |12−6| / 18
    assert len(ax_rate_err.patches) == len(ax_units_err.patches) == 4
    plt.close(fig)



def _small_catboost():
    from catboost import CatBoostRegressor
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60), "g": pd.Categorical(rng.choice(["a", "b"], 60))})
    y = 2 * X["x"] + (X["g"] == "a") + 0.1 * rng.normal(size=60)
    model = CatBoostRegressor(iterations=30, depth=3, cat_features=["g"], random_seed=0, verbose=0, allow_writing_files=False)
    return model.fit(X, y), X


def test_catboost_shap_adds_up_to_prediction():
    from src.modeling import catboost_shap
    model, X = _small_catboost()
    shap, base = catboost_shap(model, X)
    assert list(shap.columns) == list(X.columns) and shap.index.equals(X.index)
    np.testing.assert_allclose(base + shap.sum(axis=1), model.predict(X), atol=1e-9)


def test_shap_importance_total_and_groups():
    from src.modeling import shap_importance
    shap = pd.DataFrame({"x": [1.0, -3.0, 2.0], "z": [0.5, 0.5, -0.5]}, index=[10, 11, 12])
    out = shap_importance(shap, pd.Series(["AX", "AY", "AY"], index=[10, 11, 12]))
    assert list(out.index) == ["x", "z"] and list(out.columns) == ["total", "AX", "AY"]
    np.testing.assert_allclose(out.loc["x"], [2.0, 1.0, 2.5])
    np.testing.assert_allclose(out.loc["z"], [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        shap_importance(shap, pd.Series(["AX"] * 3))


def test_plot_shap_importance_panels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.modeling import plot_shap_importance
    imp = pd.DataFrame({"total": [2.0, 0.5], "AX": [1.0, 1.0], "AY": [3.0, 1.0]}, index=["x", "z"])
    fig = plot_shap_importance(imp)
    ax_bar, ax_map = fig.axes[:2]
    assert len(ax_bar.patches) == 2
    texts = sorted(t.get_text() for t in ax_map.texts)
    assert texts == ["25.0", "50.0", "50.0", "75.0"]   # AX: 50/50; AY: 75/25
    plt.close(fig)


def test_save_catboost_model_roundtrip_and_lock(tmp_path):
    from catboost import CatBoostRegressor
    from src.modeling import load_catboost_model, save_catboost_model
    model, X = _small_catboost()
    out = tmp_path / "m"
    save_catboost_model(model, {"seed": 0, "features": list(X.columns)}, out)
    loaded, meta = load_catboost_model(out)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    assert meta["features"] == ["x", "z", "g"] and "saved_at" in meta
    stamp = meta["saved_at"]
    save_catboost_model(model, {"seed": 0, "features": list(X.columns)}, out)   # igual: no escribe
    with pytest.raises(FileExistsError):
        save_catboost_model(model, {"seed": 1, "features": list(X.columns)}, out)   # otro meta
    other = CatBoostRegressor(iterations=5, depth=2, cat_features=["g"], random_seed=1, verbose=0,
                              allow_writing_files=False).fit(X, X["x"])
    with pytest.raises(FileExistsError):
        save_catboost_model(other, {"seed": 0, "features": list(X.columns)}, out)   # otro modelo
    save_catboost_model(other, {"seed": 1}, out, overwrite=True)
    loaded, meta = load_catboost_model(out)
    assert loaded == other and meta == {"saved_at": meta["saved_at"], "seed": 1}
    assert stamp  # la primera escritura dejó fecha


def test_assign_test_folds_numbers_test_by_position():
    from src.modeling import assign_test_folds, fold_masks
    data = pd.DataFrame({"item_id": ["B"] * 4 + ["A"] * 5, "block": [3, 4, 5, 6, 1, 2, 3, 4, 5],
                         "split": ["train", "train", "test", "test", "train", "train", "train", "test", "test"],
                         "fold": pd.array([0, 1, None, None, 0, 1, 2, None, None], dtype="Int64")})
    out = assign_test_folds(data)
    assert out["fold"].tolist() == [0, 0, 0, 1, 2, 0, 0, 1, 2]   # A: 1–3 train, 4–5 test; B: 3–4 train, 5–6 test
    train, valid = fold_masks(out, 2)
    assert out.loc[valid, "block"].tolist() == [5, 6] and train.sum() == 7   # todo el train + el test 1
    with pytest.raises(ValueError):
        assign_test_folds(data[~((data["item_id"] == "A") & (data["block"] == 4))])   # hueco antes del test


def test_plot_forecast_item_draws_test_when_asked():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.modeling import plot_forecast_item
    data = pd.DataFrame({"item_id": "A", "block": range(1, 6), "start_date": pd.date_range("2012-01-01", periods=5, freq="28D"),
                         "split": ["train"] * 3 + ["test"] * 2, "units": [10.0, 12, 12, 6, 99], "ratio_raw": [1.0, 1.2, 1.0, 0.5, 16.5],
                         "imputed": False, "cell": "AX"})
    table = pd.DataFrame({"item_id": "A", "block": [4, 5], "step": [1, 2], "ratio_1step": [0.5, 16.5], "ratio_chain": [0.5, 1.0],
                          "units_1step": [6.0, 99.0], "units_chain": [6.0, 6.0]})
    fig = plot_forecast_item(data, table, "A", 1.0, 1.0, splits=("train", "test"), window_label="test")
    ax_rate, _, ax_units, _ = fig.axes
    assert 99.0 in ax_units.lines[0].get_ydata() and 16.5 in ax_rate.lines[0].get_ydata()   # el test se dibuja
    assert ax_rate.get_title(loc="left").startswith("A · AX · test:")
    plt.close(fig)


def test_plot_tradeoff_curves_panels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.modeling import plot_tradeoff_curves
    curves = pd.DataFrame({"segment": "total", "forecast": ["m"] * 3 + ["seasonal naive"] * 3, "z": [-1.0, 0, 1] * 2,
                           "lost": [30.0, 10, 2, 40, 20, 8], "excess": [1.0, 5, 20, 3, 12, 40]})
    marks = pd.DataFrame({"segment": ["total"], "forecast": ["m"], "lost": [5.0], "excess": [10.0]})
    fig = plot_tradeoff_curves(curves, ["m", "seasonal naive"], ["total"], marks, "x")
    ax = fig.axes[0]
    assert len(ax.lines) == 2 and list(ax.lines[0].get_xdata()) == [1.0, 5, 20]
    assert np.isclose(ax.get_ylim()[1], 1.15 * 20)   # recorte: mayor perdida entre z = 0 (20) y la estrella (5)
    plt.close(fig)
