import numpy as np
import pandas as pd
import pytest

from src.data import (
    INTERIM_DIR,
    MASK_REASONS,
    aggregate_items,
    apply_masks,
    assert_train_only,
    audit_series,
    build_daily_prices,
    compute_launch,
    load_calendar,
    load_prices,
    load_sales,
    mask_summary,
    run_data_validation,
    save_data_validation,
    validate_calendar,
    week_of_day,
)


@pytest.fixture
def pipeline(workdir, tiny_cfg):
    calendar = load_calendar(tiny_cfg)
    last_day = tiny_cfg["data"]["last_day"]
    validate_calendar(calendar, last_day)
    units, series_index = load_sales(tiny_cfg)
    prices = load_prices(tiny_cfg)
    wod = week_of_day(calendar, last_day)
    prices_daily = build_daily_prices(prices, wod, series_index)
    launch = compute_launch(prices_daily, series_index, wod)
    masked_units, masked_prices, log, mask_reason = apply_masks(
        units, prices_daily, launch, series_index, wod, prices, tiny_cfg["data"]["sales_file"]
    )
    agg = aggregate_items(masked_units, masked_prices, series_index, launch)
    return {
        "units": units,
        "series_index": series_index,
        "prices_daily": prices_daily,
        "launch": launch,
        "masked_units": masked_units,
        "masked_prices": masked_prices,
        "log": log,
        "mask_reason": mask_reason,
        "agg": agg,
        "wod": wod,
    }


def _row(series_index, store, item):
    hit = series_index.index[(series_index.store_id == store) & (series_index.item_id == item)]
    return int(hit[0])


@pytest.mark.parametrize("problem", ["duplicate", "gap", "two_weeks"])
def test_bad_calendar_stops(raw_frames, problem):
    cal = raw_frames["calendar"].copy()
    if problem == "duplicate":
        cal = pd.concat([cal.iloc[:10], cal.iloc[[9]], cal.iloc[10:]], ignore_index=True)
    elif problem == "gap":
        cal = cal.drop(index=15).reset_index(drop=True)
    else:
        extra = cal.iloc[[5]].copy()
        extra["wm_yr_wk"] = extra["wm_yr_wk"] + 1
        cal = pd.concat([cal, extra], ignore_index=True)
    with pytest.raises(ValueError):
        validate_calendar(cal, 287)


def test_prelaunch_is_masked_not_zero(pipeline):
    si = pipeline["series_index"]
    r = _row(si, "TX_1", "FOODS_C")
    launch_d = int(pipeline["launch"].loc[r, "launch_d"])
    assert launch_d == 71
    assert np.isnan(pipeline["masked_units"][r, : launch_d - 1]).all()
    assert not np.isnan(pipeline["masked_units"][r, launch_d - 1])
    # CA_1 nunca tiene precio: toda la serie queda enmascarada y no se registra.
    r_ca = _row(si, "CA_1", "FOODS_C")
    assert np.isnan(pipeline["masked_units"][r_ca]).all()
    log = pipeline["log"]
    assert not ((log.detail.str.contains("FOODS_C")) & (log.reason == "invalid_units")).any()


def test_rejections_are_logged_with_reason_source_row(pipeline, tiny_cfg):
    log = pipeline["log"]
    assert set(log.columns) == {"reason", "source", "row_id", "column", "detail"}
    reasons = set(log.reason)
    assert {"invalid_units", "invalid_price_with_sales", "price_gap_no_sales"} <= reasons
    bad_units = log[log.reason == "invalid_units"]
    assert set(bad_units.column) == {"d_20", "d_21"}
    assert (bad_units.source == tiny_cfg["data"]["sales_file"]).all()
    assert bad_units.row_id.notna().all()
    si = pipeline["series_index"]
    assert np.isnan(pipeline["masked_units"][_row(si, "CA_1", "FOODS_B"), 19])
    # Semana sin precio con ventas: se enmascara la semana completa (d_36–d_42).
    r = _row(si, "CA_1", "FOODS_D")
    assert np.isnan(pipeline["masked_units"][r, 35:42]).all()
    # Semana sin precio y sin ventas: se enmascara y se registra como informativa.
    r = _row(si, "TX_1", "FOODS_D")
    assert np.isnan(pipeline["masked_units"][r, 42:49]).all()


def test_sales_before_first_price_logged(workdir, tiny_cfg, raw_frames):
    from src.tests.conftest import write_raw

    frames = {k: v.copy() for k, v in raw_frames.items()}
    sales = frames["sales"]
    mask = (sales.item_id == "FOODS_C") & (sales.store_id == "TX_1")
    sales.loc[mask, "d_3"] = 4.0
    write_raw(workdir, tiny_cfg, frames)
    calendar = load_calendar(tiny_cfg)
    units, si = load_sales(tiny_cfg)
    prices = load_prices(tiny_cfg)
    wod = week_of_day(calendar, 287)
    pdaily = build_daily_prices(prices, wod, si)
    launch = compute_launch(pdaily, si, wod)
    _, _, log, _ = apply_masks(
        units, pdaily, launch, si, wod, prices, tiny_cfg["data"]["sales_file"]
    )
    hit = log[log.reason == "sales_before_first_price"]
    assert list(hit.column) == ["d_3"]


def test_spend_uses_store_day_prices(pipeline):
    agg = pipeline["agg"]
    si = pipeline["series_index"]
    k = list(agg["item_ids"]).index("FOODS_A")
    rows = [_row(si, "CA_1", "FOODS_A"), _row(si, "TX_1", "FOODS_A")]
    u = pipeline["masked_units"][rows].astype(float)
    p = pipeline["masked_prices"][rows].astype(float)
    expected = np.nansum(u * p)
    assert np.isclose(np.nansum(agg["item_spend"][k]), expected)
    naive = np.nanmean(p) * np.nansum(u)
    assert not np.isclose(expected, naive)


def test_chain_total_excludes_masked_cells(pipeline):
    agg = pipeline["agg"]
    expected = np.nansum(pipeline["masked_units"].astype(float), axis=0)
    np.testing.assert_allclose(agg["chain_total"], expected)
    assert agg["item_launch_d"][list(agg["item_ids"]).index("FOODS_C")] == 71


def test_assert_train_only(tiny_cfg):
    assert_train_only(231, tiny_cfg)
    with pytest.raises(RuntimeError):
        assert_train_only(232, tiny_cfg)


def test_mask_reason_matches_masks(pipeline):
    """Cada celda en NaN tiene motivo y cada celda válida conserva el valor crudo."""
    codes = pipeline["mask_reason"]
    units, masked = pipeline["units"], pipeline["masked_units"]
    np.testing.assert_array_equal(np.isnan(masked), codes > 0)
    valid = codes == 0
    np.testing.assert_array_equal(masked[valid], units[valid])
    si = pipeline["series_index"]
    name = {v: k for k, v in MASK_REASONS.items()}
    # Prelanzamiento de TX_1/FOODS_C (lanzado el d_71); unidades inválidas en d_20 y d_21.
    assert (codes[_row(si, "TX_1", "FOODS_C"), :70] == name["pre_launch"]).all()
    assert codes[_row(si, "CA_1", "FOODS_B"), 19] == name["invalid_units"]
    assert (codes[_row(si, "CA_1", "FOODS_D"), 35:42] == name["invalid_price_with_sales"]).all()
    assert (codes[_row(si, "TX_1", "FOODS_D"), 42:49] == name["price_gap_no_sales"]).all()


def test_mask_summary_counts_cells(pipeline):
    summary = mask_summary(pipeline["mask_reason"], pipeline["units"], pipeline["series_index"])
    assert summary["cells"].sum() == pipeline["units"].size
    assert set(summary["reason"]) == set(MASK_REASONS.values())
    row = summary.set_index("reason").loc["invalid_units"]
    assert row["is_rejection"] and row["cells"] == 2


def test_run_without_persist_writes_nothing(workdir, tiny_cfg):
    result = run_data_validation(tiny_cfg)
    assert result["persisted"] == []
    assert not any(INTERIM_DIR.iterdir())
    assert result["units_raw"].shape == result["units"].shape == result["mask_reason"].shape
    written = save_data_validation(result)
    assert all(p.exists() for p in written)
    assert (INTERIM_DIR / "mask_reason.npy").exists()


def test_audit_series_shows_raw_and_masked(workdir, tiny_cfg):
    result = run_data_validation(tiny_cfg)
    view = audit_series(result, "TX_1", "FOODS_C", 68, 73)
    assert list(view["d"]) == [68, 69, 70, 71, 72, 73]
    assert view.loc[view.d < 71, "units"].isna().all()
    assert (view.loc[view.d < 71, "mask_reason"] == "pre_launch").all()
    assert view.loc[view.d >= 71, "units"].notna().all()
    with pytest.raises(KeyError):
        audit_series(result, "XX_9", "FOODS_C")


def test_profile_frame_counts_nulls_and_unique():
    from src.data import profile_frame

    df = pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "z", "w"]})
    prof = profile_frame(df)
    assert list(prof.columns) == ["dtype", "nulls", "null_pct", "unique"]
    assert prof.loc["a", "nulls"] == 1 and prof.loc["a", "null_pct"] == 25.0
    assert prof.loc["a", "unique"] == 2 and prof.loc["b", "unique"] == 4
