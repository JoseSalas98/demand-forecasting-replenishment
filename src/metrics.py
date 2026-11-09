"""
Definiciones de Makridakis et al. (2022), restringidas al conjunto de productos objetivo. No se
usa MAPE (Principio VII). Justificación (Principio III, b): se usan en el backtest y en el test.
Los NaN (días enmascarados) se ignoran.
"""

import numpy as np
import pandas as pd


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


def seasonal_naive_scale(train_units: np.ndarray, m: int) -> float:
    """Escala del RMSSE con el seasonal naive: media de (y_t − y_{t−m})² en la muestra.

    Constitución (Principio VII) y FR-027: escala = error del seasonal naive en la muestra, con
    m = 13 bloques de 28 días (un año). Se calcula con todo el train del artículo (decisión del
    usuario, como en M5): dentro de un artículo todos los candidatos se dividen por la misma
    escala, así que no cambia cuál gana.

    Parámetros:
        train_units (np.ndarray): unidades de train del artículo, en orden de bloque.
        m (int): periodo estacional en bloques.

    Retorno:
        float: escala; NaN si hay menos de m + 1 bloques.
    """
    x = np.asarray(train_units, dtype=float)
    if len(x) <= m:
        return float("nan")
    diffs = x[m:] - x[:-m]
    diffs = diffs[~np.isnan(diffs)]
    return float(np.mean(diffs**2)) if len(diffs) else float("nan")


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


def spend_weights(item_spend: np.ndarray, item_ids, window) -> pd.Series:
    """Peso de cada artículo en el WRMSSE: su gasto en la ventana de pesos (días inclusive).

    Parámetros:
        item_spend (np.ndarray): gasto artículo × día (columna j = día j + 1), p. ej. de
            `load_interim`.
        item_ids: identificador de cada fila de `item_spend`.
        window: (primer día, último día) para todos los artículos, o dict artículo → (primer
            día, último día); con dict solo se devuelven esos artículos.

    Retorno:
        pd.Series: gasto por artículo (índice `item_id`); los NaN (días enmascarados) no suman.
    """
    ids = np.asarray(item_ids).astype(str)
    row = {item: k for k, item in enumerate(ids)}
    windows = window if isinstance(window, dict) else {item: window for item in ids}
    spend = {item: float(np.nansum(np.asarray(item_spend[row[item], a - 1 : b], dtype=float)))
             for item, (a, b) in windows.items()}
    return pd.Series(spend, name="weight").rename_axis("item_id")


def wrmsse_frame(df: pd.DataFrame, pred_col: str, scales: pd.Series, weights: pd.Series,
                 actual_col: str = "units") -> tuple[float, list[str]]:
    """`wrmsse` sobre una tabla larga (una fila por artículo y bloque).

    El RMSSE de cada artículo junta todos sus bloques de la tabla (con los 13 folds de la
    ventana, su error promedio en los 13 folds) y el agregado los pondera con `weights`.

    Parámetros:
        df (pd.DataFrame): con `item_id`, `actual_col` y `pred_col`.
        scales (pd.Series): escala del RMSSE por artículo.
        weights (pd.Series): peso por artículo.

    Retorno:
        tuple[float, list[str]]: WRMSSE y artículos excluidos (escala o peso no positivos).
    """
    y, y_hat = {}, {}
    for item, g in df.groupby(df["item_id"].astype(str)):
        y[item] = g[actual_col].to_numpy(float)
        y_hat[item] = g[pred_col].to_numpy(float)
    return wrmsse(y, y_hat, scales.to_dict(), weights.to_dict())
