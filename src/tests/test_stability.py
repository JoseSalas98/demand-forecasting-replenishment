import numpy as np
import pandas as pd
import pytest

from src.stability import (
    STABILITY_COLUMNS,
    STABILITY_DECLARATION_KEYS,
    STABILITY_RULES,
    SEG_KEYS,
    alternating_block_cv,
    build_item_stability,
    build_targets,
    comparison_plan,
    noise_baseline,
    paired_comparisons,
    segment_bounds,
    year_matrices,
)

SEG = {
    "abc_cuts": [0.80, 0.95],
    "xyz_thresholds": [10.0, 25.0],
    "cv_aggregation": "block28",
    "cv_min_blocks": 2,
    "abc_rule_item_bounds": [[0.10, 0.20], [0.20, 0.30]],
    "abc_rule_spend_bounds": [[0.60, 0.80], [0.20, 0.30]],
    "abc_rule_step": 0.01,
    "cv_cut_min_share": 0.20,
    "cv_cut_resolution": 100,
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


N_ITEMS = 40
YEARS = {"Y1": (1, 56), "Y2": (57, 112), "Y3": (113, 168)}


def _panel(swap_year: str | None = None, churn: bool = True):
    """40 artículos, 3 años de 2 bloques: gasto ∝ 1/rango y CV creciente con el rango.

    Con `swap_year`, los artículos 0 y 39 intercambian gasto en ese año. El artículo `late` se
    lanza en Y2 y `gone` deja de tener datos en Y3.
    """
    days = 168
    rank = np.arange(N_ITEMS)
    level = 100.0 / (rank + 1)
    amp = 0.02 + 0.6 * rank / N_ITEMS  # CV de 2 bloques = amp × 100
    units = np.empty((N_ITEMS, days))
    for d in range(days):
        sign = 1 if (d // 28) % 2 == 0 else -1
        units[:, d] = 10 * (1 + sign * amp)
    spend = np.repeat(level[:, None], days, axis=1)
    if swap_year is not None:
        a, b = YEARS[swap_year]
        spend[[0, -1], a - 1 : b] = spend[[-1, 0], a - 1 : b]
    ids = np.array([f"i{k:02d}" for k in rank], dtype=object)
    ids[5], ids[6] = "late", "gone"
    if churn:
        units[5, :56] = spend[5, :56] = np.nan
        units[6, 112:] = spend[6, 112:] = np.nan
    return {"item_ids": ids, "item_units": units, "item_spend": spend,
            "item_launch_d": np.where(ids == "late", 57, 1).astype(float)}


def test_comparison_plan_prev_and_y1():
    assert comparison_plan(["Y1", "Y2", "Y3"]) == [
        ("prev", "Y1", "Y2"), ("y1", "Y1", "Y2"), ("prev", "Y2", "Y3"), ("y1", "Y1", "Y3"),
    ]


def test_paired_comparisons_use_same_items():
    agg = _panel()
    ym = year_matrices(agg, YEARS, SEG)
    plan = comparison_plan(list(YEARS))
    measures, results = paired_comparisons(ym, YEARS, plan, agg, SEG)
    assert list(measures.columns) == STABILITY_COLUMNS
    assert len(measures) == len(plan) * 3
    items = results[("prev", "Y1", "Y2")]["items"].set_index("item_id")
    assert items.loc["late", "status"] == "new"  # sin base: fuera de la comparación
    later = results[("prev", "Y2", "Y3")]["items"].set_index("item_id")
    assert later.loc["gone", "status"] == "no_target"
    assert (measures.loc[measures["target"] == "Y2", "n"] <= N_ITEMS - 1).all()


def test_paired_comparisons_without_changes_agree_fully():
    agg = _panel(churn=False)
    ym = year_matrices(agg, YEARS, SEG)
    measures, _ = paired_comparisons(ym, YEARS, comparison_plan(list(YEARS)), agg, SEG)
    assert (measures["kappa"] == 1).all() and (measures["keep_pct"] == 100).all()
    assert (measures["max_shift_pp"] == 0).all() and measures["note"].isna().all()
    assert (measures["n"] == N_ITEMS).all()


def test_paired_comparisons_detect_swap_and_rule():
    agg = _panel(swap_year="Y3")
    ym = year_matrices(agg, YEARS, SEG)
    plan = comparison_plan(list(YEARS))
    measures, results = paired_comparisons(ym, YEARS, plan, agg, SEG)
    abc = measures[(measures["level"] == "abc") & (measures["target"] == "Y3")]
    assert (abc["kappa"] < 1).all()
    prev = [k for k in plan if k[0] == "prev"]
    st = build_item_stability(results, prev, {"stability": {"stability_rule": "abc_all"}})
    stable = st.groupby("item_id")["stable"].first()
    assert not stable["i00"] and not stable["i39"] and stable["i01"]
    assert set(st["comparison"]) == {"Y1->Y2", "Y2->Y3"}
    assert st.set_index(["item_id", "comparison"]).loc[("i00", "Y1->Y2"), "a_every_year"] == False  # noqa: E712


def test_alternating_block_cv_and_noise_baseline():
    agg = _panel()
    units = np.repeat(np.array([[10.0] * 28 + [20.0] * 28 + [10.0] * 28 + [20.0] * 28]), 2, axis=0)
    cv = alternating_block_cv(units, np.array(["a", "b"]), "Y", 1, 112, 2)
    assert np.allclose(cv["cv"], 0.0)  # cada mitad ve bloques iguales
    years = {"Y1": (1, 112)}
    noise = noise_baseline(agg, agg["item_units"], years, SEG)
    assert set(noise["base_kind"]) == {"noise"} and set(noise["level"]) == {"abc", "xyz", "cell"}
    assert noise.loc[0, "base"] == "Y1-odd" and noise.loc[0, "target"] == "Y1-even"


def test_no_period_or_psi_keys_in_stage_declarations():
    keys = " ".join(STABILITY_DECLARATION_KEYS + SEG_KEYS)
    assert "segment_" not in keys and "psi" not in keys and "agreement" not in keys


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
