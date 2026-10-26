import numpy as np

from src.segmentation import describe_items


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
