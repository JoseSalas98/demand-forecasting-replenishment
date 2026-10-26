"""
Los rangos de días son inclusivos y en base 1: la columna j de los arreglos es el día j + 1.
"""

import warnings

import numpy as np
import pandas as pd

MATRIX_COLUMNS = [
    "set_id",
    "item_id",
    "launch_d",
    "usage",
    "spend",
    "cum_spend",
    "cum_pct_before",
    "abc_class",
    "cv",
    "xyz_class",
    "cell",
    "exclusion_reason",
]


def _window(arr: np.ndarray, start_d: int, end_d: int) -> np.ndarray:
    """Recorta las columnas de días [start_d, end_d] (base 1) como float64."""
    return np.asarray(arr[:, start_d - 1 : end_d], dtype=np.float64)


def compute_usage_spend(
    item_units: np.ndarray, item_spend: np.ndarray, start_d: int, end_d: int
) -> pd.DataFrame:
    """Calcula uso (unidades) y gasto por artículo en un rango de días (FR-009, FR-010).

    Los días NaN (prelanzamiento o enmascarados) no cuentan. `usage_period = "computation_set"`
    significa el rango completo del conjunto de cálculo.

    Parámetros:
        item_units (np.ndarray): unidades por artículo y día.
        item_spend (np.ndarray): gasto por artículo y día (Σ unidades × precio por tienda-día).
        start_d (int): primer día del rango.
        end_d (int): último día del rango.

    Retorno:
        pd.DataFrame: columnas `usage`, `spend` y `active` (al menos un día con dato).
    """
    u = _window(item_units, start_d, end_d)
    s = _window(item_spend, start_d, end_d)
    active = (~np.isnan(u)).any(axis=1)
    return pd.DataFrame(
        {"usage": np.nansum(u, axis=1), "spend": np.nansum(s, axis=1), "active": active}
    )


def classify_abc(item_ids, spend: np.ndarray, cuts: list[float]) -> pd.DataFrame:
    """Clasifica ABC por gasto acumulado (FR-011).

    Un artículo es A si su porcentaje acumulado antes de sumarse es menor que `cuts[0]`, B si es
    menor que `cuts[1]` y C en otro caso; así el artículo que cruza el corte queda en la clase
    superior. Los empates de gasto se ordenan por `item_id` para que el resultado sea
    reproducible. Justificación (Principio III, b): se usa en el EDA, en las matrices de train,
    en la referencia y en la réplica del año 5.

    Parámetros:
        item_ids (array-like): identificadores de los artículos activos.
        spend (np.ndarray): gasto de cada artículo.
        cuts (list[float]): cortes A y B como fracción (p. ej. [0.80, 0.95]).

    Retorno:
        pd.DataFrame: `item_id, spend, cum_spend, cum_pct_before, abc_class`, ordenado por gasto.
    """
    df = pd.DataFrame({"item_id": np.asarray(item_ids, dtype=object), "spend": np.asarray(spend)})
    df = df.sort_values(["spend", "item_id"], ascending=[False, True], kind="stable")
    df = df.reset_index(drop=True)
    total = df["spend"].sum()
    df["cum_spend"] = df["spend"].cumsum()
    if total > 0:
        df["cum_pct_before"] = (df["cum_spend"] - df["spend"]) / total
        df["abc_class"] = np.select(
            [df["cum_pct_before"] < cuts[0], df["cum_pct_before"] < cuts[1]], ["A", "B"], "C"
        )
    else:
        df["cum_pct_before"] = np.nan
        df["abc_class"] = None
    return df


def _block_sums(u: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    """Suma cada bloque; un bloque con algún NaN queda NaN (incompleto)."""
    cols = []
    for a, b in bounds:
        block = u[:, a:b]
        sums = block.sum(axis=1)
        sums[np.isnan(block).any(axis=1)] = np.nan
        cols.append(sums)
    if not cols:
        return np.empty((u.shape[0], 0))
    return np.column_stack(cols)


def compute_cv(
    item_units: np.ndarray,
    start_d: int,
    end_d: int,
    aggregation: str,
    min_blocks: int,
    week_of_day: np.ndarray | None = None,
    segment_bounds: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Calcula el CV de unidades por artículo sobre bloques completos (FR-012, research R4).

    Agregaciones:
        - `block28`: bloques consecutivos de 28 días desde `start_d`; el resto final se descarta.
        - `daily`: cada día es un bloque.
        - `weekly`: semanas `wm_yr_wk` completas (7 días) dentro del rango.
        - `segment_period`: los segmentos de `segment_bounds` (días inclusivos, base 1).
    Se descarta todo bloque con algún día NaN. CV = std poblacional (ddof = 0) / media × 100; NaN
    si la media es 0 o hay menos de `min_blocks` bloques completos.

    Parámetros:
        item_units (np.ndarray): unidades por artículo y día.
        start_d (int): primer día.
        end_d (int): último día.
        aggregation (str): `block28`, `daily`, `weekly` o `segment_period`.
        min_blocks (int): mínimo de bloques completos.
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día (para `weekly`).
        segment_bounds (list[tuple[int, int]] | None): segmentos (para `segment_period`).

    Retorno:
        pd.DataFrame: columnas `cv` y `n_blocks`.
    """
    u = _window(item_units, start_d, end_d)
    n_days = u.shape[1]
    if aggregation == "block28":
        bounds = [(k * 28, (k + 1) * 28) for k in range(n_days // 28)]
    elif aggregation == "daily":
        bounds = [(k, k + 1) for k in range(n_days)]
    elif aggregation == "weekly":
        if week_of_day is None:
            raise ValueError("cv_aggregation='weekly' requiere week_of_day")
        wk = np.asarray(week_of_day)[start_d - 1 : end_d]
        starts = np.flatnonzero(np.r_[True, wk[1:] != wk[:-1]])
        ends = np.r_[starts[1:], n_days]
        full_weeks = np.asarray(week_of_day)
        bounds = []
        for a, b in zip(starts, ends):
            week_total = int((full_weeks == wk[a]).sum())
            if b - a == 7 and week_total == 7:
                bounds.append((int(a), int(b)))
    elif aggregation == "segment_period":
        if not segment_bounds:
            raise ValueError("cv_aggregation='segment_period' requiere segment_bounds")
        bounds = [(a - start_d, b - start_d + 1) for a, b in segment_bounds]
        if any(a < 0 or b > n_days for a, b in bounds):
            raise ValueError("segment_bounds fuera del rango del conjunto de cálculo")
    else:
        raise ValueError(f"cv_aggregation desconocido: {aggregation}")
    sums = _block_sums(u, bounds)
    n_blocks = (~np.isnan(sums)).sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(sums, axis=1) if sums.shape[1] else np.full(u.shape[0], np.nan)
        std = np.nanstd(sums, axis=1, ddof=0) if sums.shape[1] else np.full(u.shape[0], np.nan)
        cv = std / mean * 100
    cv = np.where((n_blocks >= min_blocks) & (mean > 0), cv, np.nan)
    return pd.DataFrame({"cv": cv, "n_blocks": n_blocks})


def classify_xyz(cv: pd.Series, thresholds: list[float]) -> pd.Series:
    """Clasifica XYZ por CV (FR-013): X < umbral X ≤ Y < umbral Z ≤ Z.

    Justificación (Principio III, b): se usa en el EDA, en las matrices de train, en la referencia
    y en la réplica del año 5.

    Parámetros:
        cv (pd.Series): CV en porcentaje (NaN si no está definido).
        thresholds (list[float]): umbrales X y Z en porcentaje (p. ej. [10.0, 25.0]).

    Retorno:
        pd.Series: "X", "Y", "Z" o None.
    """
    values = np.asarray(cv, dtype=float)
    labels = np.select(
        [values < thresholds[0], values < thresholds[1], values >= thresholds[1]],
        ["X", "Y", "Z"],
        "",
    )
    out = pd.Series(labels, index=getattr(cv, "index", None), dtype=object)
    out[np.isnan(values)] = None
    return out


def build_matrix(
    set_id: str,
    agg: dict,
    start_d: int,
    end_d: int,
    seg: dict,
    week_of_day: np.ndarray | None = None,
    segment_bounds: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Construye la matriz ABC × XYZ de un conjunto de cálculo (FR-014).

    Cubre todos los artículos: los inactivos en el conjunto quedan sin clase con
    `exclusion_reason = "inactive_in_set"`; los activos con CV indefinido quedan sin XYZ con
    `cv_undefined` y celda `A-`, `B-` o `C-`.

    Parámetros:
        set_id (str): identificador del conjunto (p. ej. `train_global`, `Y1`).
        agg (dict): agregado con `item_ids`, `item_units`, `item_spend`, `item_launch_d`.
        start_d (int): primer día.
        end_d (int): último día.
        seg (dict): `abc_cuts`, `xyz_thresholds`, `cv_aggregation`, `cv_min_blocks`.
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día.
        segment_bounds (list[tuple[int, int]] | None): segmentos para `segment_period`.

    Retorno:
        pd.DataFrame: columnas `MATRIX_COLUMNS`, ordenadas por gasto (inactivos al final).
    """
    item_ids = np.asarray(agg["item_ids"], dtype=object)
    us = compute_usage_spend(agg["item_units"], agg["item_spend"], start_d, end_d)
    cv = compute_cv(
        agg["item_units"],
        start_d,
        end_d,
        seg["cv_aggregation"],
        seg["cv_min_blocks"],
        week_of_day,
        segment_bounds,
    )
    base = pd.DataFrame(
        {
            "item_id": item_ids,
            "launch_d": np.asarray(agg["item_launch_d"], dtype=float),
            "usage": us["usage"],
            "active": us["active"],
            "cv": cv["cv"],
        }
    )
    active = base[base["active"]]
    abc = classify_abc(
        active["item_id"], us.loc[active.index, "spend"].to_numpy(), seg["abc_cuts"]
    )
    out = abc.merge(base, on="item_id", how="left")
    inactive = base[~base["active"]].sort_values("item_id")
    inactive = inactive.assign(spend=0.0, cum_spend=np.nan, cum_pct_before=np.nan, abc_class=None)
    out = pd.concat([out, inactive], ignore_index=True)
    out.loc[~out["active"], "cv"] = np.nan
    out["xyz_class"] = classify_xyz(out["cv"], seg["xyz_thresholds"]).to_numpy()
    xyz_or_dash = out["xyz_class"].fillna("-")
    out["cell"] = np.where(out["active"], out["abc_class"].fillna("") + xyz_or_dash, None)
    out["exclusion_reason"] = np.select(
        [~out["active"], out["xyz_class"].isna()], ["inactive_in_set", "cv_undefined"], ""
    )
    out["exclusion_reason"] = out["exclusion_reason"].replace("", None)
    out["set_id"] = set_id
    return out[MATRIX_COLUMNS].reset_index(drop=True)


def describe_items(agg: dict, start_d: int, end_d: int) -> pd.DataFrame:
    """Describe cada artículo en un rango: días activos, proporción de ceros, media y mediana.

    La proporción de ceros se calcula solo sobre días activos (excluye NaN de prelanzamiento).

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_units`.
        start_d (int): primer día.
        end_d (int): último día.

    Retorno:
        pd.DataFrame: `item_id, active_days, zero_share, mean_daily_units, median_daily_units,
        total_units`.
    """
    u = _window(agg["item_units"], start_d, end_d)
    active_days = (~np.isnan(u)).sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        zero_share = np.where(active_days > 0, (u == 0).sum(axis=1) / active_days, np.nan)
        mean = np.nanmean(u, axis=1)
        median = np.nanmedian(u, axis=1)
    return pd.DataFrame(
        {
            "item_id": np.asarray(agg["item_ids"], dtype=object),
            "active_days": active_days,
            "zero_share": zero_share,
            "mean_daily_units": mean,
            "median_daily_units": median,
            "total_units": np.nansum(u, axis=1),
        }
    )


def sensitivity_table(
    agg: dict, start_d: int, end_d: int, seg: dict, step_pp: float = 5.0
) -> pd.DataFrame:
    """Conteo de clases ABC y XYZ con cada corte y umbral desplazado ±`step_pp` puntos (FR-013).

    Solo se usa sobre train para revisar los valores por defecto antes de congelarlos.

    Parámetros:
        agg (dict): agregado.
        start_d (int): primer día.
        end_d (int): último día.
        seg (dict): parámetros de segmentación.
        step_pp (float): desplazamiento en puntos porcentuales (research R5 usa 5).

    Retorno:
        pd.DataFrame: `variant, dimension, class, count` en formato largo.
    """
    us = compute_usage_spend(agg["item_units"], agg["item_spend"], start_d, end_d)
    act = us[us["active"]]
    ids = np.asarray(agg["item_ids"], dtype=object)[act.index]
    cv = compute_cv(
        agg["item_units"], start_d, end_d, seg["cv_aggregation"], seg["cv_min_blocks"]
    ).loc[act.index, "cv"]
    step = step_pp / 100.0
    rows = []
    cuts = list(seg["abc_cuts"])
    abc_variants = {"base": cuts}
    for k, name in enumerate(("A", "B")):
        for sign, tag in ((-1, "-"), (1, "+")):
            new = list(cuts)
            new[k] = round(cuts[k] + sign * step, 6)
            if 0 < new[0] < new[1] < 1:
                abc_variants[f"abc_cut_{name}{tag}{step_pp:g}pp"] = new
    for variant, c in abc_variants.items():
        counts = classify_abc(ids, act["spend"].to_numpy(), c)["abc_class"].value_counts()
        rows += [(variant, "abc", cls, int(counts.get(cls, 0))) for cls in ("A", "B", "C")]
    thr = list(seg["xyz_thresholds"])
    xyz_variants = {"base": thr}
    for k, name in enumerate(("X", "Z")):
        for sign, tag in ((-1, "-"), (1, "+")):
            new = list(thr)
            new[k] = thr[k] + sign * step_pp
            if 0 <= new[0] < new[1]:
                xyz_variants[f"xyz_{name}{tag}{step_pp:g}pp"] = new
    for variant, t in xyz_variants.items():
        counts = classify_xyz(cv, t).value_counts(dropna=False)
        for cls in ("X", "Y", "Z"):
            rows.append((variant, "xyz", cls, int(counts.get(cls, 0))))
        rows.append((variant, "xyz", "undefined", int(cv.isna().sum())))
    return pd.DataFrame(rows, columns=["variant", "dimension", "class", "count"])


def run_train_matrix(cfg: dict) -> dict:
    """Etapa 3: matriz ABC × XYZ global de train, tabla de sensibilidad y congelamiento.

    Solo carga días ≤ fin de train. La agregación `segment_period` no aplica a la matriz global
    porque el periodo se elige en la etapa 4.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: `matrix`, `sensitivity` y `agg`.
    """
    from src.config import train_end, train_years
    from src.data import assert_train_only, load_interim
    from src.freeze import freeze

    seg = cfg["segmentation"]
    if seg["cv_aggregation"] == "segment_period":
        raise ValueError(
            "cv_aggregation='segment_period' no aplica a la matriz global de train: el periodo "
            "se elige en la etapa 4. Usar block28, daily o weekly."
        )
    max_day = train_end(cfg)
    assert_train_only(max_day, cfg)
    agg = load_interim(cfg, max_day)
    years = train_years(cfg)
    start = min(a for a, _ in years.values())
    matrix = build_matrix("train_global", agg, start, max_day, seg, agg["week_of_day"])
    sensitivity = sensitivity_table(agg, start, max_day, seg)
    freeze(matrix, "matrix_train", "data/processed/matrix_train.parquet", cfg)
    return {"matrix": matrix, "sensitivity": sensitivity, "agg": agg}
