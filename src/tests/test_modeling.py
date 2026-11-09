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
