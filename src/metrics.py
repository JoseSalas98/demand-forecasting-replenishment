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


def train_weight_windows(blocks: pd.DataFrame, day_of: pd.Series, n_blocks: int, block_days: int) -> dict:
    """Ventana de pesos de cada artículo: sus últimos `n_blocks` bloques de train (días inclusive).

    Es la regla del nb 03c (usuario, 2026-11-06): con un año ningún artículo queda con peso 0 por
    un quiebre puntual, y cada artículo usa su propio último año de train (nunca test).

    Parámetros:
        blocks (pd.DataFrame): una fila por artículo y bloque, con `item_id`, `split` y
            `start_date`.
        day_of (pd.Series): fecha → número de día d.
        n_blocks (int): bloques de la ventana.
        block_days (int): días por bloque.

    Retorno:
        dict: artículo (str) → (primer día, último día).
    """
    train = blocks[blocks["split"] == "train"]
    starts = train.groupby(train["item_id"].astype(str))["start_date"]
    out = {}
    for item, s in starts:
        s = s.sort_values()
        if len(s) < n_blocks:
            raise ValueError(f"{item}: {len(s)} bloques de train, se piden {n_blocks}")
        out[item] = (int(day_of[s.iloc[-n_blocks]]), int(day_of[s.iloc[-1]]) + block_days - 1)
    return out


def item_errors(window: pd.DataFrame, candidates: list[str], scales: pd.Series,
                actual_col: str = "units") -> pd.DataFrame:
    """Error de cada candidato por artículo en los bloques de la tabla (nb 03d, Etapa 2).

    Parámetros:
        window (pd.DataFrame): una fila por artículo y bloque, con `item_id`, `actual_col` y una
            columna de pronóstico por candidato (NaN si el artículo no tiene ese modelo).
        candidates (list[str]): columnas de pronóstico.
        scales (pd.Series): escala del RMSSE por artículo.

    Retorno:
        pd.DataFrame: una fila por artículo y candidato con `n_blocks` (pares válidos), `rmsse`,
        `wape`, `mae` y las sumas `abs_error` y `units` (para agregar el WAPE). Un artículo sin
        pronóstico de un candidato no tiene fila para ese candidato.
    """
    rows = []
    for item, g in window.groupby(window["item_id"].astype(str)):
        for cand in candidates:
            y, y_hat = _valid(g[actual_col].to_numpy(float), g[cand].to_numpy(float))
            if not len(y):
                continue
            abs_error = np.abs(y - y_hat)
            rows.append({"item_id": item, "candidate": cand, "n_blocks": len(y),
                         "rmsse": rmsse(y, y_hat, float(scales.get(item, 0.0))), "wape": wape(y, y_hat),
                         "mae": float(abs_error.mean()), "abs_error": float(abs_error.sum()),
                         "units": float(np.abs(y).sum())})
    return pd.DataFrame(rows)


def segment_errors(errors: pd.DataFrame, weights: pd.Series, segment: pd.Series) -> pd.DataFrame:
    """WRMSSE y WAPE por candidato y segmento, más el total, desde `item_errors`.

    WRMSSE: promedio de los RMSSE con los pesos renormalizados dentro del grupo (como `wrmsse`:
    sin RMSSE válido o peso no positivo, el artículo no entra). WAPE: Σ|e| / Σy del grupo, que
    pondera por unidades.

    Parámetros:
        errors (pd.DataFrame): salida de `item_errors`.
        weights (pd.Series): peso por artículo.
        segment (pd.Series): segmento por artículo (p. ej. `cell`).

    Retorno:
        pd.DataFrame: índice (`candidate`, `segment`) con `n_items`, `wrmsse` y `wape`; el
        segmento `total` junta todos los artículos.
    """
    df = errors.assign(weight=errors["item_id"].map(weights).astype(float),
                       segment=errors["item_id"].map(segment).astype(str))
    df = pd.concat([df, df.assign(segment="total")])

    def summary(g: pd.DataFrame) -> pd.Series:
        ok = g["rmsse"].notna() & (g["weight"] > 0)
        w = g.loc[ok, "weight"]
        return pd.Series({"n_items": len(g),
                          "wrmsse": float(np.dot(w / w.sum(), g.loc[ok, "rmsse"])) if ok.any() else np.nan,
                          "wape": g["abs_error"].sum() / g["units"].sum() if g["units"].sum() > 0 else np.nan})

    out = df.groupby(["candidate", "segment"], sort=False)[["rmsse", "weight", "abs_error", "units"]].apply(summary)
    out["n_items"] = out["n_items"].astype(int)
    return out


def paired_bootstrap_wrmsse(rmsse_a: pd.Series, rmsse_b: pd.Series, weights: pd.Series, n_boot: int,
                            seed: int, level: float = 0.95) -> dict:
    """Intervalo bootstrap pareado de WRMSSE(a) − WRMSSE(b), remuestreando artículos.

    Cada réplica sortea artículos con reemplazo y recalcula los dos WRMSSE con los mismos
    artículos (pareado), con los pesos renormalizados en la réplica.

    Parámetros:
        rmsse_a, rmsse_b (pd.Series): RMSSE por artículo de cada alternativa (mismo índice).
        weights (pd.Series): peso por artículo.
        n_boot (int): réplicas.
        seed (int): semilla.
        level (float): nivel del intervalo de percentiles.

    Retorno:
        dict: `diff` (diferencia observada), `low`, `high` y `p_a_not_better` (fracción de
        réplicas con diferencia ≥ 0).
    """
    items = rmsse_a.index
    if not items.equals(rmsse_b.index) or rmsse_a.isna().any() or rmsse_b.isna().any():
        raise ValueError("las dos alternativas deben tener RMSSE en los mismos artículos")
    w = weights.reindex(items).to_numpy(float)
    d = (rmsse_a - rmsse_b).to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(items), size=(n_boot, len(items)))
    boot = (w[idx] * d[idx]).sum(axis=1) / w[idx].sum(axis=1)
    tail = (1 - level) / 2
    return {"diff": float(np.dot(w, d) / w.sum()), "low": float(np.quantile(boot, tail)),
            "high": float(np.quantile(boot, 1 - tail)), "p_a_not_better": float(np.mean(boot >= 0))}


def tukey_outliers(values: pd.Series, groups: pd.Series, k: float = 1.5) -> pd.Series:
    """Marca los valores atípicos por arriba dentro de cada grupo (regla de Tukey).

    Atípico = mayor que Q3 + k·IQR de su grupo. Solo por arriba: interesa el error alto. Un valor
    nulo nunca se marca y no entra en los cuartiles.

    Parámetros:
        values (pd.Series): valor por fila (p. ej., el RMSSE de cada artículo).
        groups (pd.Series): grupo de cada fila (p. ej., el segmento), mismo índice que `values`.
        k (float): múltiplo del rango intercuartílico (1,5 es el valor clásico de Tukey).

    Retorno:
        pd.Series: booleana, mismo índice que `values`.
    """
    if not groups.index.equals(values.index):
        raise ValueError("groups debe tener el mismo índice que values")
    by = groups.astype(str)
    q1 = values.groupby(by).transform(lambda v: v.quantile(0.25))
    q3 = values.groupby(by).transform(lambda v: v.quantile(0.75))
    return (values > q3 + k * (q3 - q1)).fillna(False).astype(bool)


def block_totals(daily: np.ndarray, item_ids, rows: pd.DataFrame, day_of: pd.Series, block_days: int) -> np.ndarray:
    """Suma de un arreglo diario por artículo en el bloque de cada fila (días inclusive).

    Sirve para llevar a bloques la venta y el gasto crudos de `load_interim`. Los días NaN
    (enmascarados) no suman: cuentan como 0 venta observada.

    Parámetros:
        daily (np.ndarray): artículo × día (columna j = día j + 1).
        item_ids: identificador de cada fila de `daily`.
        rows (pd.DataFrame): una fila por artículo y bloque con `item_id` y `start_date`.
        day_of (pd.Series): fecha → número de día d.
        block_days (int): días por bloque.

    Retorno:
        np.ndarray: total de cada fila de `rows`, en su orden.
    """
    row_of = {item: k for k, item in enumerate(np.asarray(item_ids).astype(str))}
    items = rows["item_id"].astype(str).to_numpy()
    starts = day_of.reindex(pd.to_datetime(rows["start_date"])).to_numpy()
    if np.isnan(starts.astype(float)).any():
        raise ValueError("hay fechas de inicio que no están en el calendario")
    out = np.empty(len(rows))
    for k, (item, d) in enumerate(zip(items, starts.astype(int))):
        out[k] = np.nansum(np.asarray(daily[row_of[item], d - 1 : d - 1 + block_days], dtype=float))
    return out


def coverage_by_z(df: pd.DataFrame, forecasts: dict, sigma: dict, z_values) -> pd.DataFrame:
    """Venta capturada, venta perdida y exceso de cada pronóstico para cada colchón z, por artículo.

    En cada bloque: disponible = max(pronóstico + z·σ, 0); capturada = min(disponible, demanda)·p;
    perdida = max(demanda − disponible, 0)·p; exceso = max(disponible − demanda, 0)·p. Cada bloque
    se evalúa por separado (sin arrastrar lo que sobra). Los montos se suman por artículo.

    Parámetros:
        df (pd.DataFrame): una fila por artículo y bloque con `item_id`, `units` (demanda),
            `price` y las columnas de pronóstico en unidades.
        forecasts (dict): nombre del pronóstico → columna de `df`.
        sigma (dict): nombre del pronóstico → pd.Series de σ por artículo; si falta, σ = 0.
        z_values: valores de z.

    Retorno:
        pd.DataFrame: una fila por artículo, pronóstico y z con `demand`, `captured`, `lost`,
        `excess` (dinero) y `short_blocks` (bloques con disponible < demanda).
    """
    items = df["item_id"].astype(str)
    demand = df["units"].to_numpy(float)
    price = df["price"].to_numpy(float)
    out = []
    for name, col in forecasts.items():
        s = sigma.get(name)
        sd = np.zeros(len(df)) if s is None else items.map(s).to_numpy(float)
        if np.isnan(sd).any():
            raise ValueError(f"{name}: falta σ para algún artículo")
        for z in z_values:
            available = np.maximum(df[col].to_numpy(float) + z * sd, 0.0)
            frame = pd.DataFrame({"item_id": items.to_numpy(), "demand": demand * price,
                                  "captured": np.minimum(available, demand) * price,
                                  "lost": np.maximum(demand - available, 0.0) * price,
                                  "excess": np.maximum(available - demand, 0.0) * price,
                                  "short_blocks": (available < demand).astype(int)})
            out.append(frame.groupby("item_id", sort=True).sum().reset_index().assign(forecast=name, z=float(z)))
    cols = ["item_id", "forecast", "z", "demand", "captured", "lost", "excess", "short_blocks"]
    return pd.concat(out, ignore_index=True)[cols]


def service_reading(curve: pd.DataFrame, target: float) -> pd.Series:
    """Lee una curva de intercambio en un nivel de servicio: z, venta perdida y exceso interpolados.

    Parámetros:
        curve (pd.DataFrame): un pronóstico, una fila por z con `z`, `service` (0–1), `lost` y
            `excess`; el servicio no baja al subir z.
        target (float): nivel de servicio buscado (p. ej. 0,95).

    Retorno:
        pd.Series: `z`, `lost` y `excess` en ese servicio; NaN si el objetivo queda fuera de la
        curva (no se extrapola).
    """
    c = curve.sort_values("z")
    service = c["service"].to_numpy(float)
    if (np.diff(service) < -1e-12).any():
        raise ValueError("el servicio debe ser no decreciente en z")
    if not service[0] <= target <= service[-1]:
        return pd.Series({"z": np.nan, "lost": np.nan, "excess": np.nan})
    # Con tramos planos, np.interp toma el primer z que alcanza el objetivo.
    keep = np.r_[True, np.diff(service) > 0]
    return pd.Series({col: float(np.interp(target, service[keep], c[col].to_numpy(float)[keep]))
                      for col in ("z", "lost", "excess")})


def lost_at_excess(curve: pd.DataFrame, excess: float) -> float:
    """Venta perdida de una curva de intercambio para un nivel de exceso dado (interpolada).

    Parámetros:
        curve (pd.DataFrame): un pronóstico, una fila por z con `z`, `lost` y `excess`.
        excess (float): exceso en dinero.

    Retorno:
        float: venta perdida con ese exceso; NaN si el exceso queda fuera de la curva.
    """
    c = curve.sort_values("z")
    x = c["excess"].to_numpy(float)
    if not x[0] <= excess <= x[-1]:
        return float("nan")
    keep = np.r_[True, np.diff(x) > 0]
    return float(np.interp(excess, x[keep], c["lost"].to_numpy(float)[keep]))
