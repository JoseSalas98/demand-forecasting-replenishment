import copy

import numpy as np
import pytest

from src import forecasting
from src.data import load_interim
from src.forecasting import (
    FORECASTERS,
    candidate_grid,
    fold_starts,
    forecast_frozen,
    run_backtest,
    share_series,
)
from src.freeze import load_frozen
from src.stability import run_replication


def test_share_series_excludes_zero_chain_total_days():
    units = np.array([[1.0, 2.0, np.nan, 0.0], [3.0, 0.0, 1.0, 0.0]])
    chain = np.array([4.0, 2.0, 1.0, 0.0])
    shares, log = share_series(units, chain, np.array(["a", "b"]), 1, 4)
    np.testing.assert_allclose(shares[0, :2], [0.25, 1.0])
    assert np.isnan(shares[0, 2])
    assert np.isnan(shares[:, 3]).all()
    assert list(log["reason"]) == ["zero_chain_total"]
    assert list(log["column"]) == ["d_4"]


def test_seasonal_naive_repeats_last_week():
    pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / 100
    rng = np.random.default_rng(0)
    series = np.tile(pattern, 12) + rng.normal(0, 1e-4, 84)
    frozen = FORECASTERS["snaive"]["fit"](series, {"m": 7})
    fc = forecast_frozen(series, frozen, 14)
    np.testing.assert_allclose(fc, np.tile(series[-7:], 2), rtol=1e-6, atol=1e-9)


def test_moving_average_uses_last_window():
    series = np.array([1.0, 2.0, np.nan, 4.0, 6.0])
    frozen = FORECASTERS["moving_average"]["fit"](series, {"window": 2})
    np.testing.assert_allclose(forecast_frozen(series, frozen, 3), [5.0, 5.0, 5.0])


def test_forecast_frozen_does_not_change_params():
    rng = np.random.default_rng(1)
    series = 0.05 + 0.01 * rng.standard_normal(200)
    frozen = FORECASTERS["arima"]["fit"](series, {"order": [1, 0, 0]})
    before = copy.deepcopy(frozen)
    forecast_frozen(np.r_[series, 0.06, 0.04], frozen, 28)
    assert frozen == before


def test_candidate_grid_and_folds(tiny_cfg):
    grid = candidate_grid(tiny_cfg)
    families = [c["family"] for c in grid]
    assert families == ["snaive", "moving_average", "arima", "sarima"]
    assert fold_starts(tiny_cfg) == [176, 204]
    cfg = copy.deepcopy(tiny_cfg)
    cfg["split"]["years"] = {
        "Y1": [122, 485],
        "Y2": [486, 849],
        "Y3": [850, 1213],
        "Y4": [1214, 1577],
    }
    cfg["forecast"]["backtest_folds"] = 4
    starts = fold_starts(cfg)
    assert starts == [1466, 1494, 1522, 1550]
    assert starts[0] >= 1466 and starts[-1] + 27 == 1577


@pytest.fixture
def replicated(staged):
    run_replication(staged["cfg"])
    return staged


def _inputs(cfg):
    from src.config import train_end

    agg = load_interim(cfg, train_end(cfg))
    shares, _ = share_series(
        agg["item_units"], agg["chain_total"], agg["item_ids"], 1, train_end(cfg)
    )
    return shares, agg, load_frozen("targets")


def test_backtest_logs_failures_and_continues(replicated, monkeypatch):
    cfg = replicated["cfg"]

    def boom(series, hyperparams):
        raise RuntimeError("fallo simulado")

    monkeypatch.setitem(FORECASTERS, "arima", {**FORECASTERS["arima"], "fit": boom})
    shares, agg, targets = _inputs(cfg)
    scores, failures = run_backtest(shares, agg, targets, cfg)
    assert "fallo simulado" in " ".join(failures["message"])
    assert set(failures.columns) == {"item_id", "candidate_id", "stage", "message"}
    assert {"snaive_7", "moving_average_7"} <= set(scores["candidate_id"])
    assert set(scores["fold"]) == {1, 2}


def test_backtest_stops_if_forecast_changed_after_replication(replicated):
    cfg = copy.deepcopy(replicated["cfg"])
    cfg["forecast"]["ma_windows"] = [14]
    shares, agg, targets = _inputs(cfg)
    with pytest.raises(RuntimeError, match="forecast.ma_windows"):
        run_backtest(shares, agg, targets, cfg)


def test_backtest_stops_without_replication(staged):
    cfg = staged["cfg"]
    shares, agg, targets = _inputs(cfg)
    with pytest.raises(RuntimeError):
        run_backtest(shares, agg, targets, cfg)


def test_full_forecast_flow_and_single_test_evaluation(replicated):
    from src.forecasting import run_backtesting_stage, run_test_evaluation_stage

    cfg = replicated["cfg"]
    out = run_backtesting_stage(cfg)
    models = load_frozen("frozen_models")
    assert {m["item_ids"][0] for m in models} == set(load_frozen("targets")["item_id"])
    assert out["scores"]["fold"].max() == 2
    evaluation = run_test_evaluation_stage(cfg)
    assert len(evaluation) == 2
    assert {f"wrmsse_o{k}" for k in (1, 2)} <= set(evaluation.columns)
    assert "beats_snaive" in evaluation.columns
    assert load_frozen("frozen_models") == models
    with pytest.raises(RuntimeError, match="SC-007"):
        run_test_evaluation_stage(cfg)


def test_selection_level_labels():
    assert forecasting._selection_unit("FOODS_A", "item") == "FOODS_A"
    assert forecasting._selection_unit("FOODS_A", "global") == "global"
    with pytest.raises(ValueError):
        forecasting._selection_unit("FOODS_A", "cluster_x")
