"""
Definiciones de Makridakis et al. (2022), restringidas al conjunto de productos objetivo. No se
usa MAPE (Principio VII). Justificación (Principio III, b): se usan en el backtest y en el test.
Los NaN (días enmascarados) se ignoran.
"""

import numpy as np


def naive_scale(train_units: np.ndarray) -> float:
    """Escala del RMSSE: media de las diferencias a un paso al cuadrado en train.

    Empieza en la primera venta no nula del artículo, como en M5.

    Parámetros:
        train_units (np.ndarray): unidades de train del artículo (puede tener NaN).

    Retorno:
        float: escala; 0.0 si no hay ventas no nulas o diferencias válidas.
    """
    x = np.asarray(train_units, dtype=float)
    nonzero = np.flatnonzero(np.nan_to_num(x) > 0)
    if not len(nonzero):
        return 0.0
    diffs = np.diff(x[nonzero[0] :])
    diffs = diffs[~np.isnan(diffs)]
    if not len(diffs):
        return 0.0
    return float(np.mean(diffs**2))


def _valid(y: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filtra los pares donde el real o el pronóstico son NaN."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    keep = ~np.isnan(y) & ~np.isnan(y_hat)
    return y[keep], y_hat[keep]


def rmsse(y: np.ndarray, y_hat: np.ndarray, scale: float) -> float:
    """Raíz del error cuadrático medio escalado.

    Parámetros:
        y (np.ndarray): valores reales.
        y_hat (np.ndarray): pronóstico.
        scale (float): escala de `naive_scale`.

    Retorno:
        float: RMSSE; NaN si no hay pares válidos o la escala no es positiva.
    """
    y, y_hat = _valid(y, y_hat)
    if not len(y) or not scale > 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - y_hat) ** 2) / scale))


def wrmsse(
    y: dict[str, np.ndarray],
    y_hat: dict[str, np.ndarray],
    scales: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, list[str]]:
    """RMSSE ponderado sobre un único nivel (los productos objetivo).

    Los artículos con escala 0 (o sin RMSSE válido) se excluyen y los pesos se renormalizan sobre
    los evaluados.

    Parámetros:
        y (dict): artículo → valores reales.
        y_hat (dict): artículo → pronóstico.
        scales (dict): artículo → escala.
        weights (dict): artículo → peso (gasto en la ventana de pesos).

    Retorno:
        tuple[float, list[str]]: WRMSSE y artículos excluidos.
    """
    values, w, excluded = [], [], []
    for item, actual in y.items():
        value = rmsse(actual, y_hat[item], scales.get(item, 0.0))
        if np.isnan(value) or not weights.get(item, 0.0) > 0:
            excluded.append(item)
            continue
        values.append(value)
        w.append(weights[item])
    if not values:
        return float("nan"), excluded
    w_arr = np.asarray(w) / np.sum(w)
    return float(np.dot(w_arr, values)), excluded


def wape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Error absoluto ponderado: Σ|e| / Σy.

    Parámetros:
        y (np.ndarray): valores reales.
        y_hat (np.ndarray): pronóstico.

    Retorno:
        float: WAPE; NaN si Σy = 0.
    """
    y, y_hat = _valid(y, y_hat)
    total = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - y_hat)) / total) if total > 0 else float("nan")


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Raíz del error cuadrático medio (diagnóstico, sin umbral).

    Parámetros:
        y (np.ndarray): valores reales.
        y_hat (np.ndarray): pronóstico.

    Retorno:
        float: RMSE; NaN si no hay pares válidos.
    """
    y, y_hat = _valid(y, y_hat)
    return float(np.sqrt(np.mean((y - y_hat) ** 2))) if len(y) else float("nan")
