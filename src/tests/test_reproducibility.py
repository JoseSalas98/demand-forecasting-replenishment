"""Repetir las etapas 1, 3 y 4 con la misma config produce los mismos artefactos (SC-004)."""

from src.data import run_data_validation
from src.freeze import load_manifest
from src.segmentation import run_train_matrix
from src.stability import run_stability_stage
from src.tests.conftest import write_raw


def _run(root, cfg, monkeypatch):
    root.mkdir(parents=True)
    monkeypatch.chdir(root)
    write_raw(root, cfg)
    run_data_validation(cfg)
    run_train_matrix(cfg)
    run_stability_stage(cfg)
    arts = load_manifest()["artifacts"]
    return {
        name: arts[name]["sha256"] for name in ("matrix_train", "targets", "segmentation_params")
    }


def test_same_config_same_hashes(tmp_path, tiny_cfg, monkeypatch):
    first = _run(tmp_path / "run1", tiny_cfg, monkeypatch)
    second = _run(tmp_path / "run2", tiny_cfg, monkeypatch)
    assert first == second
