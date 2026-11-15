import numpy as np
import pandas as pd

from src.stability import agreement_pct, psi, psi_band, year_matrices

SEG = {
    "abc_cuts": [0.80, 0.95],
    "xyz_thresholds": [10.0, 25.0],
    "cv_aggregation": "block28",
    "cv_min_blocks": 1,
}


def test_agreement_ignores_missing_classes():
    a = pd.Series({"i1": "A", "i2": "B", "i3": None, "i4": "C"})
    b = pd.Series({"i1": "A", "i2": "A", "i3": "A"})
    assert agreement_pct(a, b) == 50.0


def test_psi_uses_epsilon_for_empty_class():
    p_ref = {"A": 0.5, "B": 0.5, "C": 0.0}
    p_cur = {"A": 0.5, "B": 0.4, "C": 0.1}
    eps = 1e-4
    expected = (0.4 - 0.5) * np.log(0.4 / 0.5) + (0.1 - eps) * np.log(0.1 / eps)
    assert np.isclose(psi(p_ref, p_cur, eps), expected)
    assert psi(p_ref, p_ref, eps) == 0.0


def test_psi_band_limits():
    bands = [0.10, 0.25]
    assert psi_band(0.0999, bands) == "stable"
    assert psi_band(0.10, bands) == "moderate"
    assert psi_band(0.2499, bands) == "moderate"
    assert psi_band(0.25, bands) == "shift"


def test_year_matrices_one_set_per_range():
    agg = {
        "item_ids": np.array(["i1", "i2"]),
        "item_units": np.ones((2, 56)),
        "item_spend": np.ones((2, 56)),
        "item_launch_d": np.array([1, 1]),
    }
    out = year_matrices(agg, {"Y1": (1, 28), "Y2": (29, 56)}, SEG)
    assert sorted(out.set_id.unique()) == ["Y1", "Y2"]
    assert len(out) == 4
