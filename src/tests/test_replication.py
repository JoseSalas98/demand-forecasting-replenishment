"""Pruebas de la réplica en el año 5."""

import copy

import pandas as pd
import pytest

from src.config import forecast_keys
from src.data import load_interim, run_data_validation
from src.freeze import load_frozen, load_manifest, require_declared_before
from src.segmentation import build_matrix
from src.stability import replicate_year5, run_replication


def test_stops_without_frozen_targets(workdir, tiny_cfg):
    run_data_validation(tiny_cfg, persist=True)
    with pytest.raises(RuntimeError, match="targets"):
        run_replication(tiny_cfg)


def test_replication_declares_forecast_and_keeps_targets(staged):
    cfg = staged["cfg"]
    before = load_frozen("targets")
    sha = load_manifest()["artifacts"]["targets"]["sha256"]
    out = run_replication(cfg)
    pd.testing.assert_frame_equal(load_frozen("targets"), before)
    assert load_manifest()["artifacts"]["targets"]["sha256"] == sha
    require_declared_before(forecast_keys(cfg), "matrix_y5")
    rep = out["year5_replication"]
    assert list(rep["item_id"]) == list(before["item_id"])
    y5_cell = out["matrix_y5"].set_index("item_id")["cell"].reindex(rep["item_id"])
    assert (rep["same_as_frozen"].to_numpy() == (y5_cell == rep["frozen_cell"].to_numpy())).all()
    st = out["stability_y5"]
    assert set(st["target"]) == {"Y5"}
    assert set(zip(st["base_kind"], st["base"])) == {("prev", "Y4"), ("y1", "Y1")}
    assert set(st["level"]) == {"abc", "xyz", "cell"}


def test_replication_uses_frozen_params_not_config(staged):
    cfg = staged["cfg"]
    seg_params = copy.deepcopy(load_frozen("segmentation_params"))
    seg_params["abc_cuts"] = [0.5, 0.9]
    agg = load_interim(cfg, cfg["split"]["test_days"][1])
    matrix_y5, _, _ = replicate_year5(
        load_frozen("targets"),
        seg_params,
        load_frozen("zero_thresholds"),
        agg,
        cfg,
        agg["week_of_day"],
    )
    seg = {
        k: seg_params[k] for k in ("abc_cuts", "xyz_thresholds", "cv_aggregation", "cv_min_blocks")
    }
    expected = build_matrix("Y5", agg, *cfg["split"]["test_days"], seg, agg["week_of_day"])
    pd.testing.assert_series_equal(matrix_y5["abc_class"], expected["abc_class"])


def test_year5_reference_is_year4(staged):
    cfg = staged["cfg"]
    agg = load_interim(cfg, cfg["split"]["test_days"][1])
    seg_params = load_frozen("segmentation_params")
    _, _, rep = replicate_year5(
        load_frozen("targets"),
        seg_params,
        load_frozen("zero_thresholds"),
        agg,
        cfg,
        agg["week_of_day"],
    )
    assert rep["ref_abc"].notna().all()
