import copy
import json

import pytest

from src.config import check_history, flatten, validate_config


def test_default_config_is_valid(tiny_cfg):
    validate_config(tiny_cfg)


@pytest.mark.parametrize(
    "path, value",
    [
        (("segmentation", "abc_cuts"), [0.95, 0.80]),
        (("segmentation", "abc_cuts"), [0.0, 0.95]),
        (("segmentation", "abc_cuts"), [0.80, 1.0]),
        (("segmentation", "xyz_thresholds"), [25.0, 10.0]),
        (("segmentation", "xyz_thresholds"), [-1.0, 25.0]),
        (("stability", "psi_bands"), [0.25, 0.10]),
        (("stability", "psi_bands"), [-0.1, 0.25]),
        (("stability", "psi_epsilon"), 0.0),
        (("stability", "agreement_min"), 0.0),
        (("stability", "agreement_min"), 1.5),
        (("segmentation", "cv_aggregation"), "monthly"),
        (("stability", "stability_rule"), "anything"),
        (("stability", "segment_candidates"), ["quarter", "semester"]),
        (("forecast", "backtest_folds"), 0),
        (("forecast", "ma_windows"), [7, -1]),
        (("forecast", "arima_p"), [0.5]),
        (("split", "test_origins"), 3),
        (("split", "test_days"), [200, 255]),
    ],
)
def test_range_rules_stop(tiny_cfg, path, value):
    cfg = copy.deepcopy(tiny_cfg)
    cfg[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_split_must_be_contiguous(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["split"]["years"]["Y2"] = [65, 119]
    with pytest.raises(ValueError, match="split"):
        validate_config(cfg)


def test_empty_treatment_stops(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["treatment_map"]["AX"] = ""
    with pytest.raises(ValueError, match="treatment_map"):
        validate_config(cfg)


def test_missing_key_stops(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    del cfg["stability"]["psi_epsilon"]
    with pytest.raises(ValueError, match="psi_epsilon"):
        validate_config(cfg)


def _write_history(path, rows):
    lines = ["date,parameter,old_value,new_value,reason"] + rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_history_requires_initial_row(tmp_path, tiny_cfg):
    hist = tmp_path / "h.csv"
    _write_history(hist, [])
    with pytest.raises(RuntimeError):
        check_history(tiny_cfg, None, hist)
    _write_history(hist, ['2026-10-24,*,,initial,"inicial"'])
    check_history(tiny_cfg, None, hist)


def test_history_detects_unlogged_change(tmp_path, tiny_cfg):
    hist = tmp_path / "h.csv"
    _write_history(hist, ['2026-10-24,*,,initial,"inicial"'])
    snapshot = copy.deepcopy(tiny_cfg)
    changed = copy.deepcopy(tiny_cfg)
    changed["segmentation"]["abc_cuts"] = [0.75, 0.95]
    with pytest.raises(RuntimeError, match="segmentation.abc_cuts"):
        check_history(changed, snapshot, hist)
    new_value = json.dumps([0.75, 0.95]).replace('"', '""')
    _write_history(
        hist,
        [
            '2026-10-24,*,,initial,"inicial"',
            f'2026-10-25,segmentation.abc_cuts,"[0.8, 0.95]","{new_value}","prueba"',
        ],
    )
    check_history(changed, snapshot, hist)


def test_flatten_uses_dotted_keys(tiny_cfg):
    flat = flatten(tiny_cfg)
    assert flat["segmentation.abc_cuts"] == [0.80, 0.95]
    assert flat["split.years.Y1"] == [8, 63]
