import numpy as np

from src.metrics import naive_scale, rmse, rmsse, wape, wrmsse


def test_scale_starts_at_first_nonzero_sale():
    # Desde el primer valor no nulo: diferencias 2, 0, 2 → media de cuadrados 8/3.
    train = np.array([0.0, 0.0, 2.0, 4.0, 4.0, 6.0])
    assert np.isclose(naive_scale(train), 8 / 3)
    assert naive_scale(np.zeros(5)) == 0.0


def test_rmsse_by_hand():
    scale = 8 / 3
    value = rmsse(np.array([5.0, 5.0]), np.array([4.0, 7.0]), scale)
    assert np.isclose(value, np.sqrt(2.5 / scale))


def test_wrmsse_excludes_zero_scale_and_renormalizes():
    y = {"a": np.array([1.0, 1.0]), "b": np.array([2.0, 2.0]), "c": np.array([1.0])}
    y_hat = {"a": np.array([2.0, 2.0]), "b": np.array([2.0, 4.0]), "c": np.array([0.0])}
    scales = {"a": 1.0, "b": 4.0, "c": 0.0}
    weights = {"a": 3.0, "b": 1.0, "c": 6.0}
    value, excluded = wrmsse(y, y_hat, scales, weights)
    expected = 0.75 * np.sqrt(1.0 / 1.0) + 0.25 * np.sqrt(2.0 / 4.0)
    assert np.isclose(value, expected)
    assert excluded == ["c"]


def test_wape_and_rmse():
    y = np.array([2.0, 0.0, 4.0, np.nan])
    y_hat = np.array([1.0, 1.0, 4.0, 9.0])
    assert np.isclose(wape(y, y_hat), 2.0 / 6.0)
    assert np.isclose(rmse(y, y_hat), np.sqrt(2.0 / 3.0))


def test_seasonal_naive_scale():
    import math

    import numpy as np

    from src.metrics import seasonal_naive_scale

    x = np.array([1.0, 2.0, 3.0, 5.0, 2.0, 6.0])     # diferencias con m = 3: 4, 0, 3
    assert seasonal_naive_scale(x, 3) == (16 + 0 + 9) / 3
    assert math.isnan(seasonal_naive_scale(x[:3], 3))


def test_spend_weights_sum_window_and_skip_nan():
    import numpy as np
    from src.metrics import spend_weights
    spend = np.array([[1.0, 2.0, np.nan, 4.0], [0.0, 0.0, 5.0, 5.0]])
    w = spend_weights(spend, ["A", "B"], (2, 3))   # días 2 y 3
    assert w.to_dict() == {"A": 2.0, "B": 5.0} and w.index.name == "item_id"
    per_item = spend_weights(spend, ["A", "B"], {"B": (3, 4)})   # ventana propia de B
    assert per_item.to_dict() == {"B": 10.0}


def test_wrmsse_frame_matches_wrmsse():
    import numpy as np
    import pandas as pd
    from src.metrics import wrmsse, wrmsse_frame
    df = pd.DataFrame({"item_id": pd.Categorical(["A", "A", "B", "B"]), "units": [10.0, 12, 5, 7],
                       "pred": [11.0, 12, 4, 9]})
    scales, weights = pd.Series({"A": 4.0, "B": 1.0}), pd.Series({"A": 3.0, "B": 1.0})
    value, excluded = wrmsse_frame(df, "pred", scales, weights)
    expected, _ = wrmsse({"A": np.array([10.0, 12]), "B": np.array([5.0, 7])},
                         {"A": np.array([11.0, 12]), "B": np.array([4.0, 9])}, scales.to_dict(), weights.to_dict())
    assert np.isclose(value, expected) and excluded == []


def test_train_weight_windows_last_train_blocks_per_item():
    import pandas as pd
    import pytest
    from src.metrics import train_weight_windows
    dates = pd.to_datetime(["2020-01-01", "2020-01-29", "2020-02-26", "2020-03-25"])
    day_of = pd.Series([1, 29, 57, 85], index=dates)
    blocks = pd.DataFrame({"item_id": ["A"] * 4 + ["B"] * 3,
                           "split": ["train", "train", "train", "test", "train", "train", "train"],
                           "start_date": list(dates) + list(dates[1:])})
    w = train_weight_windows(blocks, day_of, n_blocks=2, block_days=28)
    assert w == {"A": (29, 84), "B": (57, 112)}   # A no usa su bloque de test
    with pytest.raises(ValueError):
        train_weight_windows(blocks, day_of, n_blocks=4, block_days=28)


def test_item_errors_by_hand_and_skips_missing_candidate():
    import numpy as np
    import pandas as pd
    from src.metrics import item_errors
    window = pd.DataFrame({"item_id": ["A", "A", "B", "B"], "units": [10.0, 12, 5, 7],
                           "m1": [11.0, 12, 4, 9], "m2": [np.nan, np.nan, 5, 8]})
    out = item_errors(window, ["m1", "m2"], pd.Series({"A": 4.0, "B": 1.0})).set_index(["item_id", "candidate"])
    assert list(out.index) == [("A", "m1"), ("B", "m1"), ("B", "m2")]   # A no tiene m2
    a = out.loc[("A", "m1")]
    assert a["n_blocks"] == 2 and np.isclose(a["rmsse"], np.sqrt(0.5 / 4))
    assert np.isclose(a["wape"], 1 / 22) and np.isclose(a["mae"], 0.5) and a["abs_error"] == 1 and a["units"] == 22


def test_segment_errors_matches_wrmsse_and_pools_wape():
    import numpy as np
    import pandas as pd
    from src.metrics import item_errors, segment_errors, wrmsse_frame
    window = pd.DataFrame({"item_id": ["A", "A", "B", "B", "C", "C"], "units": [10.0, 12, 5, 7, 2, 2],
                           "m1": [11.0, 12, 4, 9, 1, 2]})
    scales, weights = pd.Series({"A": 4.0, "B": 1.0, "C": 1.0}), pd.Series({"A": 3.0, "B": 1.0, "C": 2.0})
    out = segment_errors(item_errors(window, ["m1"], scales), weights, pd.Series({"A": "X", "B": "X", "C": "Y"}))
    assert np.isclose(out.loc[("m1", "total"), "wrmsse"], wrmsse_frame(window, "m1", scales, weights)[0])
    assert np.isclose(out.loc[("m1", "X"), "wrmsse"], wrmsse_frame(window[window["item_id"] != "C"], "m1", scales, weights)[0])
    assert np.isclose(out.loc[("m1", "total"), "wape"], 5 / 38) and out.loc[("m1", "X"), "n_items"] == 2


def test_paired_bootstrap_wrmsse():
    import numpy as np
    import pandas as pd
    import pytest
    from src.metrics import paired_bootstrap_wrmsse
    idx = pd.Index([f"i{k}" for k in range(50)])
    b = pd.Series(np.linspace(0.5, 1.0, 50), index=idx)
    w = pd.Series(np.arange(1.0, 51), index=idx)
    out = paired_bootstrap_wrmsse(b - 0.1, b, w, n_boot=200, seed=0)   # a mejora 0,1 en todos
    assert np.isclose(out["diff"], -0.1) and np.isclose(out["low"], -0.1) and np.isclose(out["high"], -0.1)
    assert out["p_a_not_better"] == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_wrmsse(b.iloc[1:], b, w, n_boot=10, seed=0)


def test_tukey_outliers_by_group():
    import pandas as pd
    import pytest

    from src.metrics import tukey_outliers
    values = pd.Series([1.0, 2, 3, 4, 100, 10, 20, 30, 40, 50, np.nan])
    groups = pd.Series(["a"] * 5 + ["b"] * 6)
    out = tukey_outliers(values, groups)
    # a: Q1 = 2, Q3 = 4, techo 7 → solo 100; b: Q1 = 20, Q3 = 40, techo 70 → nada; el nulo no se marca.
    assert out.tolist() == [False] * 4 + [True] + [False] * 6
    with pytest.raises(ValueError):
        tukey_outliers(values, groups.iloc[:5])


def test_block_totals_sums_each_block():
    import pandas as pd

    from src.metrics import block_totals
    daily = np.array([[1.0, 2, np.nan, 4, 5, 6], [10, 20, 30, 40, 50, 60]])
    day_of = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("2015-01-01", periods=6))
    rows = pd.DataFrame({"item_id": ["B", "A", "A"], "start_date": pd.to_datetime(["2015-01-04", "2015-01-01", "2015-01-04"])})
    out = block_totals(daily, ["A", "B"], rows, day_of, block_days=3)
    np.testing.assert_allclose(out, [150, 3, 15])   # el NaN no suma


def test_coverage_by_z_money_and_buffer():
    import pandas as pd
    import pytest

    from src.metrics import coverage_by_z
    df = pd.DataFrame({"item_id": ["A", "A", "B"], "units": [10.0, 10.0, 4.0], "price": [2.0, 2.0, 1.0],
                       "f": [8.0, 13.0, 0.0]})
    out = coverage_by_z(df, {"m": "f", "perfecto": "units"}, {"m": pd.Series({"A": 2.0, "B": 1.0})}, [0, 1, -10])
    a0 = out[(out.forecast == "m") & (out.z == 0) & (out.item_id == "A")].iloc[0]
    # Bloque 1: disponible 8 → captura 16, pierde 4; bloque 2: disponible 13 → captura 20, sobra 6.
    assert (a0.demand, a0.captured, a0.lost, a0.excess, a0.short_blocks) == (40, 36, 4, 6, 1)
    a1 = out[(out.forecast == "m") & (out.z == 1) & (out.item_id == "A")].iloc[0]   # disponible 10 y 15
    assert (a1.captured, a1.lost, a1.excess) == (40, 0, 10)
    b_neg = out[(out.forecast == "m") & (out.z == -10) & (out.item_id == "B")].iloc[0]   # nunca negativo
    assert (b_neg.captured, b_neg.lost, b_neg.excess) == (0, 4, 0)
    perfect = out[out.forecast == "perfecto"]
    assert (perfect.lost == 0).all() and (perfect.excess == 0).all()
    assert np.allclose(out.captured + out.lost, out.demand)
    with pytest.raises(ValueError):
        coverage_by_z(df, {"m": "f"}, {"m": pd.Series({"A": 1.0})}, [0])


def test_service_reading_and_lost_at_excess():
    import pandas as pd

    from src.metrics import lost_at_excess, service_reading
    curve = pd.DataFrame({"z": [0.0, 1, 2], "service": [0.8, 0.9, 1.0], "lost": [20.0, 10, 0], "excess": [5.0, 15, 35]})
    r = service_reading(curve, 0.95)
    assert np.isclose(r["z"], 1.5) and np.isclose(r["lost"], 5) and np.isclose(r["excess"], 25)
    assert service_reading(curve, 0.5).isna().all()                 # fuera de la curva: no extrapola
    assert np.isclose(lost_at_excess(curve, 10), 15) and np.isnan(lost_at_excess(curve, 50))
    flat = curve.assign(service=[0.8, 1.0, 1.0])                     # tramo plano: primer z que llega
    assert np.isclose(service_reading(flat, 1.0)["z"], 1)
