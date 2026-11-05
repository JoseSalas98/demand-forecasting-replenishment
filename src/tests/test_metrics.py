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
