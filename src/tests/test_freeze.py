from datetime import UTC, datetime

import pandas as pd
import pytest

from src.freeze import (
    TestLedger,
    freeze,
    load_frozen,
    load_manifest,
    record_declarations,
    require_declarations,
    require_declared_before,
    verify_manifest,
)


def _now():
    return datetime.now(UTC)


def test_freeze_records_hashes_and_refuses_second_write(workdir, tiny_cfg):
    df = pd.DataFrame({"item_id": ["A", "B"], "spend": [2.0, 1.0]})
    digest = freeze(df, "matrix_train", "data/processed/matrix_train.parquet", tiny_cfg)
    entry = load_manifest()["artifacts"]["matrix_train"]
    assert entry["sha256"] == digest
    assert entry["config_sha256"]
    assert entry["frozen_at"]
    with pytest.raises(RuntimeError):
        freeze(df, "matrix_train", "data/processed/matrix_train.parquet", tiny_cfg)
    pd.testing.assert_frame_equal(load_frozen("matrix_train"), df)


def test_load_frozen_detects_tampering(workdir, tiny_cfg):
    freeze({"a": 1}, "params", "data/processed/params.json", tiny_cfg)
    (workdir / "data/processed/params.json").write_text('{"a": 2}', encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_frozen("params")
    with pytest.raises(RuntimeError):
        verify_manifest()


def test_declarations(workdir, tiny_cfg):
    keys = ["stability.stability_rule", "segmentation.abc_cuts"]
    with pytest.raises(RuntimeError):
        require_declarations(keys, _now(), tiny_cfg)
    record_declarations(keys, tiny_cfg)
    require_declarations(keys, _now(), tiny_cfg)
    # Declaración posterior al inicio de la etapa.
    with pytest.raises(RuntimeError):
        require_declarations(keys, datetime(2000, 1, 1, tzinfo=UTC), tiny_cfg)
    # Valor distinto al declarado.
    tiny_cfg["segmentation"]["abc_cuts"] = [0.7, 0.9]
    with pytest.raises(RuntimeError):
        require_declarations(keys, _now(), tiny_cfg)
    with pytest.raises(RuntimeError):
        record_declarations(keys, tiny_cfg)


def test_declared_before_artifact(workdir, tiny_cfg):
    keys = ["forecast.ma_windows"]
    freeze({"x": 1}, "matrix_y5", "data/processed/matrix_y5.json", tiny_cfg)
    record_declarations(keys, tiny_cfg)
    with pytest.raises(RuntimeError):
        require_declared_before(keys, "matrix_y5")


def test_declared_before_artifact_ok(workdir, tiny_cfg):
    keys = ["forecast.ma_windows"]
    record_declarations(keys, tiny_cfg)
    freeze({"x": 1}, "matrix_y5", "data/processed/matrix_y5.json", tiny_cfg)
    require_declared_before(keys, "matrix_y5")


def test_ledger_refuses_repeated_model(workdir):
    ledger = TestLedger()
    assert not ledger.has("m1")
    ledger.record("m1", "a", "b")
    assert ledger.has("m1")
    with pytest.raises(RuntimeError):
        TestLedger().record("m1", "a", "b")
