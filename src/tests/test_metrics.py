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
