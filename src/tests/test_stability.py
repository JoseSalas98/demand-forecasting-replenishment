import numpy as np
import pandas as pd
import pytest

from src.stability import (
    STABILITY_RULES,
    build_targets,
    choose_segment_period,
    reference_classes,
    segment_bounds,
    year_matrices,
)

SEG = {
    "abc_cuts": [0.80, 0.95],
    "xyz_thresholds": [10.0, 25.0],
    "cv_aggregation": "block28",
    "cv_min_blocks": 1,
}


def _stab(rows):
    cols = ["item_id", "comparison", "same_abc", "same_xyz", "same_cell", "a_every_year"]
    return pd.DataFrame(rows, columns=cols)


def test_rules_on_hand_built_inputs():
    st = _stab(
        [
            # i1: misma ABC en las 3, XYZ cambia en una.
            ("i1", "Y2", True, True, True, True),
            ("i1", "Y3", True, False, False, True),
            ("i1", "Y4", True, True, True, True),
            # i2: misma celda en 2 de 3; ABC cambia en una; un año en B.
            ("i2", "Y2", True, True, True, False),
            ("i2", "Y3", True, True, True, False),
            ("i2", "Y4", False, True, False, False),
            # i3: todo igual.
            ("i3", "Y2", True, True, True, True),
            ("i3", "Y3", True, True, True, True),
            ("i3", "Y4", True, True, True, True),
        ]
    )
    assert STABILITY_RULES["abc_all"](st).to_dict() == {"i1": True, "i2": False, "i3": True}
    assert STABILITY_RULES["abc_xyz_all"](st).to_dict() == {"i1": False, "i2": False, "i3": True}
    assert STABILITY_RULES["cell_majority"](st).to_dict() == {"i1": True, "i2": True, "i3": True}
    assert STABILITY_RULES["a_every_year"](st).to_dict() == {"i1": True, "i2": False, "i3": True}


def _agg_three_years():
    days = 84
    units = np.ones((4, days))
    spend = np.ones((4, days))
    spend[0] *= 10.0
    spend[1] *= 3.0
    spend[2, :] *= 6.0
    units[2, :28] = np.nan  # Lanzado en Y2.
    spend[2, :28] = np.nan
    units[3, :56] = np.nan  # Lanzado en Y3: sin años previos activos.
    spend[3, :56] = np.nan
    return {
        "item_ids": np.array(["a", "b", "c", "d"], dtype=object),
        "item_units": units,
        "item_spend": spend,
        "item_launch_d": np.array([1, 1, 29, 57]),
    }


def test_reference_uses_only_active_prior_years():
    agg = _agg_three_years()
    years = {"Y1": (1, 28), "Y2": (29, 56), "Y3": (57, 84)}
    ym = year_matrices(agg, years, SEG)
    ref = reference_classes(ym, agg, SEG, "Y3", ["Y1", "Y2"], 28, years).set_index("item_id")
    assert ref.loc["a", "n_ref_years"] == 2
    assert ref.loc["a", "ref_spend_mean"] == 280.0
    assert ref.loc["c", "n_ref_years"] == 1
    assert ref.loc["c", "ref_spend_mean"] == 168.0
    assert ref.loc["d", "n_ref_years"] == 0
    assert pd.isna(ref.loc["d", "ref_abc"])
    assert ref.loc["a", "ref_abc"] == "A"


def test_choose_segment_period_ties_go_to_first():
    ym = pd.DataFrame({"set_id": ["Y1", "Y1"], "item_id": ["a", "b"], "abc_class": ["A", "B"]})
    same = pd.DataFrame(
        {"set_id": ["Y1-P1", "Y1-P1"], "item_id": ["a", "b"], "abc_class": ["A", "B"]}
    )
    chosen, table = choose_segment_period(
        ym, {"four_month": same, "quarter": same.copy()}, ["four_month", "quarter"]
    )
    assert chosen == "four_month"
    assert set(table["kind"]) == {"four_month", "quarter"}
    worse = same.assign(abc_class=["B", "B"])
    chosen, _ = choose_segment_period(
        ym, {"quarter": worse, "four_month": same}, ["quarter", "four_month"]
    )
    assert chosen == "four_month"


def test_segment_bounds_cover_year():
    q = segment_bounds("quarter", 122, 485)
    assert q == [(122, 212), (213, 303), (304, 394), (395, 485)]
    t = segment_bounds("four_month", 122, 485)
    assert [b - a + 1 for a, b in t] == [119, 119, 126]


def _matrix_train():
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "spend": [80.0, 15.0, 5.0],
            "abc_class": ["A", "A", "C"],
            "xyz_class": ["X", "W", "Z"],
            "cell": ["AX", "AW", "CZ"],
        }
    )


def _item_stab(stable):
    return pd.DataFrame({"item_id": ["a", "b", "c"], "stable": stable})


def test_targets_are_stable_a_items_with_treatment(tiny_cfg):
    t = build_targets(_matrix_train(), _item_stab([True, False, True]), tiny_cfg)
    assert list(t.item_id) == ["a"]
    assert t.loc[0, "treatment"] == tiny_cfg["treatment_map"]["AX"]
    assert t.loc[0, "spend_share"] == 0.8
    assert list(t.columns) == [
        "item_id",
        "cell",
        "xyz_class",
        "spend_train",
        "spend_share",
        "treatment",
    ]


def test_targets_stop_on_unmapped_cell(tiny_cfg):
    with pytest.raises(ValueError, match="AW"):
        build_targets(_matrix_train(), _item_stab([True, True, True]), tiny_cfg)


def test_targets_stop_on_empty_list(tiny_cfg):
    with pytest.raises(ValueError, match="vacía"):
        build_targets(_matrix_train(), _item_stab([False, False, True]), tiny_cfg)
