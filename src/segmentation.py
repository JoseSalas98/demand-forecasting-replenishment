"""
Los rangos de días son inclusivos y en base 1: la columna j de los arreglos es el día j + 1.
"""

from itertools import combinations
from math import ceil, comb
import warnings

import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfiltic
from scipy.stats import chi2, rankdata
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.seasonal import STL

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


def _weekly_bounds(week_of_day: np.ndarray, start_d: int, end_d: int) -> list[tuple[int, int]]:
    """Semanas `wm_yr_wk` completas (7 días) dentro de [start_d, end_d], en posiciones relativas.

    Justificación (Principio III, b): la usan `compute_cv` (`weekly`) y `seasonal_strength_items`.
    """
    wk = np.asarray(week_of_day)[start_d - 1 : end_d]
    starts = np.flatnonzero(np.r_[True, wk[1:] != wk[:-1]])
    ends = np.r_[starts[1:], len(wk)]
    full_weeks = np.asarray(week_of_day)
    bounds = []
    for a, b in zip(starts, ends):
        week_total = int((full_weeks == wk[a]).sum())
        if b - a == 7 and week_total == 7:
            bounds.append((int(a), int(b)))
    return bounds


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
        bounds = _weekly_bounds(week_of_day, start_d, end_d)
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
    cv_units: np.ndarray | None = None,
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
        cv_units (np.ndarray | None): unidades para el CV, p. ej. con las rachas de ceros en NaN
            (`mask_zero_runs`, FR-032); si es None, `agg["item_units"]`. El uso, el gasto y la
            actividad salen siempre de `agg`.

    Retorno:
        pd.DataFrame: columnas `MATRIX_COLUMNS`, ordenadas por gasto (inactivos al final).
    """
    item_ids = np.asarray(agg["item_ids"], dtype=object)
    us = compute_usage_spend(agg["item_units"], agg["item_spend"], start_d, end_d)
    cv = compute_cv(
        agg["item_units"] if cv_units is None else cv_units,
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


TRAIN_DECLARATION_KEYS = [
    "segmentation.cv_aggregation",
    "segmentation.cv_min_blocks",
    "segmentation.abc_rule_item_bounds",
    "segmentation.abc_rule_spend_bounds",
    "segmentation.abc_rule_step",
    "segmentation.cv_cut_min_share",
    "segmentation.cv_cut_resolution",
    "segmentation.zero_alpha",
    "segmentation.long_run",
]


def abc_rule_kwargs(seg: dict) -> dict:
    """Argumentos de `select_abc_cuts` tomados de `[segmentation]` (o de lo congelado)."""
    return {
        "item_bounds": [tuple(b) for b in seg["abc_rule_item_bounds"]],
        "spend_bounds": [tuple(b) for b in seg["abc_rule_spend_bounds"]],
        "step": seg["abc_rule_step"],
    }


def cv_cut_kwargs(seg: dict) -> dict:
    """Argumentos de `find_cv_cuts` tomados de `[segmentation]` (o de lo congelado)."""
    return {"n_cuts": 2, "min_share": seg["cv_cut_min_share"], "resolution": seg["cv_cut_resolution"]}


def run_train_matrix(cfg: dict) -> dict:
    """Etapa 3: umbrales de ceros, cortes de train, matriz ABC × XYZ global y congelamiento.

    Orden: registra las declaraciones (`TRAIN_DECLARATION_KEYS`), carga solo días ≤ fin de train,
    estima los umbrales de racha (FR-032) y enmascara las rachas para el CV, elige los cortes ABC
    con `select_abc_cuts` y los umbrales XYZ con `find_cv_cuts` (FR-013), y congela
    `zero_thresholds`, `train_cuts` y `matrix_train`. `abc_cuts` y `xyz_thresholds` de la config
    solo se usan en la tabla de sensibilidad.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: `matrix`, `sensitivity`, `zero_thresholds`, `cuts`, `abc_scenarios`, `cv_segments`,
        `cv_raw` (CV sin enmascarar, para comparar) y `agg`.
    """
    from src.config import train_end, train_years
    from src.data import assert_train_only, load_interim
    from src.freeze import freeze, record_declarations, require_declarations, utc_now

    seg = cfg["segmentation"]
    record_declarations(TRAIN_DECLARATION_KEYS, cfg)
    require_declarations(TRAIN_DECLARATION_KEYS, utc_now(), cfg)
    max_day = train_end(cfg)
    assert_train_only(max_day, cfg)
    agg = load_interim(cfg, max_day)
    years = train_years(cfg)
    start = min(a for a, _ in years.values())
    wod = agg["week_of_day"]

    thresholds = zero_run_thresholds(agg, start, max_day, seg["zero_alpha"], seg["long_run"])
    masked = mask_zero_runs(
        agg["item_units"], agg["item_ids"], thresholds.set_index("item_id")["threshold"],
        start, max_day,
    )
    us = compute_usage_spend(agg["item_units"], agg["item_spend"], start, max_day)
    ids = np.asarray(agg["item_ids"], dtype=object)
    active = us["active"].to_numpy()
    rule = select_abc_cuts(ids[active], us.loc[active, "spend"].to_numpy(), **abc_rule_kwargs(seg))
    cv = compute_cv(masked, start, max_day, seg["cv_aggregation"], seg["cv_min_blocks"], wod)
    cv_raw = compute_cv(
        agg["item_units"], start, max_day, seg["cv_aggregation"], seg["cv_min_blocks"], wod
    )
    has_cv = active & cv["cv"].notna().to_numpy()
    cv_res = find_cv_cuts(
        cv["cv"].to_numpy()[has_cv], us["spend"].to_numpy()[has_cv], **cv_cut_kwargs(seg)
    )
    chosen = dict(seg, abc_cuts=rule["cuts"], xyz_thresholds=cv_res["thresholds"])
    matrix = build_matrix("train_global", agg, start, max_day, chosen, wod, cv_units=masked)
    sensitivity = sensitivity_table(agg, start, max_day, seg)
    cuts = {
        "abc_cuts": rule["cuts"],
        "abc_share_bounds": rule["share_bounds"],
        "xyz_thresholds": cv_res["thresholds"],
    }
    freeze(thresholds, "zero_thresholds", "data/processed/zero_thresholds.parquet", cfg)
    freeze(cuts, "train_cuts", "data/processed/train_cuts.json", cfg)
    freeze(matrix, "matrix_train", "data/processed/matrix_train.parquet", cfg)
    return {
        "matrix": matrix,
        "sensitivity": sensitivity,
        "zero_thresholds": thresholds,
        "cuts": cuts,
        "abc_scenarios": rule["scenarios"],
        "cv_segments": cv_res["segments"],
        "cv_raw": pd.Series(cv_raw["cv"].to_numpy(), index=ids),
        "agg": agg,
    }


def find_cv_cuts(
    cv,
    spend,
    n_cuts: int = 2,
    min_share: float = 0.10,
    resolution: int = 100,
    labels: list[str] | None = None,
) -> dict:
    """Busca los cortes del CV que ordenan el gasto de mayor a menor entre segmentos.

    Es una búsqueda de cambios de nivel como las de `ruptures`, pero sobre la distribución: los
    artículos se ordenan por CV y el gasto en ese orden es la señal. El costo es el L2 del gasto
    (suma de cuadrados dentro de cada segmento), pero no se minimiza sin más: solo compiten las
    particiones que cumplen (1) cada segmento tiene al menos `min_share` de los artículos y
    (2) el gasto medio por artículo baja estrictamente de un segmento al siguiente (el de CV
    más bajo gasta más). Entre ellas gana la que minimiza el costo, es decir, la que más
    separa los niveles de gasto. La búsqueda es exhaustiva sobre `resolution` posiciones de corte
    candidatas repartidas en la distribución.

    Parámetros:
        cv (array-like): CV en porcentaje; los NaN se descartan.
        spend (array-like): gasto de cada artículo, alineado con `cv`.
        n_cuts (int): número de cortes; produce `n_cuts + 1` segmentos (por defecto 2: X, Y, Z).
        min_share (float): fracción mínima de artículos por segmento.
        resolution (int): número máximo de posiciones candidatas.
        labels (list[str] | None): etiquetas de los segmentos; por defecto X, Y, Z con 2 cortes
            y S1, S2, … en otro caso.

    Retorno:
        dict: `thresholds` (cortes de CV; con `classify_xyz` un artículo es del primer segmento si
        CV < corte 1, y así sucesivamente), `segments` (una fila por segmento con artículos,
        proporción, rango de CV y gasto), `n_feasible` y `n_evaluated`.
    """
    cv = np.asarray(cv, dtype=float)
    spend = np.asarray(spend, dtype=float)
    keep = ~np.isnan(cv)
    order = np.argsort(cv[keep], kind="stable")
    cv_s, spend_s = cv[keep][order], spend[keep][order]
    n = len(cv_s)
    k = n_cuts + 1
    min_n = ceil(min_share * n)
    if n_cuts < 1 or k * min_n > n:
        raise ValueError(f"No caben {k} segmentos de al menos {min_share:.0%} de {n} artículos")
    if labels is None:
        labels = ["X", "Y", "Z"] if n_cuts == 2 else [f"S{i + 1}" for i in range(k)]
    if len(labels) != k:
        raise ValueError(f"Se esperaban {k} etiquetas y llegaron {len(labels)}")

    # Posición p = corte antes del artículo p; no se corta entre CV iguales.
    cands = np.flatnonzero(cv_s[1:] > cv_s[:-1]) + 1
    cands = cands[(cands >= min_n) & (cands <= n - min_n)]
    if len(cands) > resolution:
        cands = cands[np.unique(np.linspace(0, len(cands) - 1, resolution).round().astype(int))]
    n_evaluated = comb(len(cands), n_cuts)
    if n_evaluated > 5_000_000:
        raise ValueError(f"{n_evaluated} combinaciones: bajar `resolution` o `n_cuts`")
    if n_evaluated == 0:
        raise ValueError("No hay posiciones de corte que respeten el mínimo por segmento")

    combos = np.array(list(combinations(cands, n_cuts)), dtype=np.int64).reshape(-1, n_cuts)
    zeros = np.zeros(len(combos), dtype=np.int64)
    bounds = np.column_stack([zeros, combos, np.full(len(combos), n)])
    size = np.diff(bounds, axis=1)
    cum = np.r_[0.0, np.cumsum(spend_s)]
    mean = np.diff(cum[bounds], axis=1) / size
    ok = (size >= min_n).all(axis=1) & (mean[:, :-1] > mean[:, 1:]).all(axis=1)
    if not ok.any():
        raise ValueError(
            "Ninguna partición respeta el mínimo por segmento y un gasto medio descendente "
            "con el CV"
        )
    # Minimizar el costo L2 dentro de los segmentos equivale a maximizar el de entre segmentos.
    between = (size * (mean - spend_s.mean()) ** 2).sum(axis=1)
    best = bounds[np.flatnonzero(ok)[np.argmax(between[ok])]]

    thresholds = [float((cv_s[p - 1] + cv_s[p]) / 2) for p in best[1:-1]]
    rows = []
    for label, a, b in zip(labels, best[:-1], best[1:], strict=True):
        rows.append(
            {
                "class": label,
                "n_items": int(b - a),
                "share_items": (b - a) / n,
                "cv_min": float(cv_s[a]),
                "cv_max": float(cv_s[b - 1]),
                "mean_spend": float(spend_s[a:b].mean()),
                "total_spend": float(spend_s[a:b].sum()),
                "share_spend": float(spend_s[a:b].sum() / spend_s.sum()),
            }
        )
    return {
        "thresholds": thresholds,
        "segments": pd.DataFrame(rows),
        "n_feasible": int(ok.sum()),
        "n_evaluated": n_evaluated,
    }


def _outside(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Puntos porcentuales que `values` (en %) queda fuera del intervalo `bounds` (fracciones)."""
    lo, hi = bounds[0] * 100, bounds[1] * 100
    return np.maximum(lo - values, 0) + np.maximum(values - hi, 0)


def _center_distance(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Distancia en puntos porcentuales de `values` (en %) al centro del intervalo `bounds`."""
    return np.abs(values - (bounds[0] + bounds[1]) * 50)


def select_abc_cuts(
    item_ids,
    spend,
    item_bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.10, 0.20), (0.20, 0.30)),
    spend_bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.60, 0.80), (0.20, 0.30)),
    step: float = 0.01,
) -> dict:
    """Elige los cortes ABC que mejor cumplen una regla de % de artículos y % de gasto.

    Recorre una grilla de pares (`cut_A` < `cut_B`, paso `step`) y calcula, con la convención de
    `classify_abc`, el % de artículos y el % de gasto de A y de B. La violación es la suma de los
    puntos porcentuales que esos cuatro valores quedan fuera de sus intervalos. Gana el par de
    menor violación; entre empates, el más cercano al centro de los intervalos, y después el de
    cortes más bajos. Se minimiza la violación en lugar de exigir 0 porque la curva de Pareto ata
    el % de artículos al % de gasto y la regla puede no tener solución exacta.
    Justificación (Principio III, b): se usa en la matriz global del EDA, en cada base del
    análisis interanual y en cada año del intraanual.

    Parámetros:
        item_ids (array-like): identificadores de los artículos.
        spend (array-like): gasto de cada artículo.
        item_bounds (tuple): intervalos de % de artículos de A y de B, como fracciones.
        spend_bounds (tuple): intervalos de % de gasto de A y de B, como fracciones.
        step (float): paso de la grilla de cortes.

    Retorno:
        dict: `cuts` ([cut_A, cut_B]); `share_bounds` (fronteras de participación en el gasto de
        A y de B, a mitad de camino entre el último artículo de la clase y el primero de la
        siguiente; ver `classify_abc_share`); `abc` (salida de `classify_abc` con los cortes
        elegidos) y `scenarios` (una fila por par, ordenada de mejor a peor).
    """
    ids = np.asarray(item_ids, dtype=object)
    spend = np.asarray(spend, dtype=float)
    # Cortes 1.0 dejan todo en A: solo interesa el orden y el % acumulado antes de cada artículo.
    ranked = classify_abc(ids, spend, [1.0, 1.0])
    total = ranked["spend"].sum()
    if len(ranked) == 0 or total <= 0:
        raise ValueError("select_abc_cuts necesita artículos con gasto positivo")
    before = ranked["cum_pct_before"].to_numpy()
    cum = np.r_[0.0, np.cumsum(ranked["spend"].to_numpy())] / total
    n = len(ranked)

    grid = np.round(np.arange(1, round(1 / step)) * step, 10)
    ia, ib = np.triu_indices(len(grid), k=1)
    cut_a, cut_b = grid[ia], grid[ib]
    # Con `cum_pct_before` no decreciente, los A son los primeros `n_a` artículos.
    n_a = np.searchsorted(before, cut_a, side="left")
    n_b = np.searchsorted(before, cut_b, side="left")
    sc = pd.DataFrame(
        {
            "cut_A": cut_a,
            "cut_B": cut_b,
            "A_items_pct": n_a / n * 100,
            "A_spend_pct": cum[n_a] * 100,
            "B_items_pct": (n_b - n_a) / n * 100,
            "B_spend_pct": (cum[n_b] - cum[n_a]) * 100,
            "C_items_pct": (n - n_b) / n * 100,
            "C_spend_pct": (1 - cum[n_b]) * 100,
        }
    )
    checks = (
        ("A_items_pct", item_bounds[0]),
        ("A_spend_pct", spend_bounds[0]),
        ("B_items_pct", item_bounds[1]),
        ("B_spend_pct", spend_bounds[1]),
    )
    sc["violation"] = sum(_outside(sc[col].to_numpy(), b) for col, b in checks).round(9)
    sc["center_distance"] = sum(_center_distance(sc[col].to_numpy(), b) for col, b in checks)
    sc = sc.sort_values(
        ["violation", "center_distance", "cut_A", "cut_B"], kind="stable"
    ).reset_index(drop=True)
    cuts = [float(sc.loc[0, "cut_A"]), float(sc.loc[0, "cut_B"])]

    abc = classify_abc(ids, spend, cuts)
    share = abc["spend"].to_numpy() / total
    share_bounds = []
    for upper, lower in (("A", ("B", "C")), ("B", ("C",))):
        in_cls = abc["abc_class"].to_numpy() == upper
        below = np.isin(abc["abc_class"].to_numpy(), lower)
        if not in_cls.any():
            share_bounds.append(share_bounds[-1] if share_bounds else float(share.max()))
            continue
        low = share[in_cls].min()
        share_bounds.append(float((low + share[below].max()) / 2) if below.any() else float(low))
    return {"cuts": cuts, "share_bounds": share_bounds, "abc": abc, "scenarios": sc}


def classify_abc_share(spend, share_bounds: list[float]) -> np.ndarray:
    """Clasifica ABC por participación en el gasto con fronteras fijas, tomadas de una base.

    A si la participación es ≥ `share_bounds[0]`, B si es ≥ `share_bounds[1]` y C en otro caso.
    A diferencia de `classify_abc`, la proporción de clases puede cambiar entre conjuntos, y usar
    la participación en lugar del gasto absoluto neutraliza el crecimiento de la cadena.
    Justificación (Principio III, b): se usa en el análisis interanual y en el intraanual.

    Parámetros:
        spend (array-like): gasto de cada artículo (NaN = sin clase).
        share_bounds (list[float]): fronteras de participación de A y de B.

    Retorno:
        np.ndarray: "A", "B", "C" o None por artículo.
    """
    spend = np.asarray(spend, dtype=float)
    valid = ~np.isnan(spend)
    total = spend[valid].sum()
    out = np.full(len(spend), None, dtype=object)
    if total <= 0:
        return out
    share = spend[valid] / total
    out[valid] = np.select([share >= share_bounds[0], share >= share_bounds[1]], ["A", "B"], "C")
    return out


ZERO_RUN_COLUMNS = ["item_id", "position", "start_d", "end_d", "length"]


def zero_runs(agg: dict, start_d: int, end_d: int) -> pd.DataFrame:
    """Rachas de días con cero de cada artículo en un rango, con su posición.

    Los días NaN (sin dato) se saltan: la racha se mide en días activos y un NaN en medio no la
    corta, porque un día sin dato no es evidencia de venta. Posición: `leading` (antes de la
    primera venta del rango), `trailing` (después de la última) o `internal`. Un artículo sin
    ninguna venta en el rango tiene una sola racha `leading`.
    Justificación (Principio III, b): la usan `zero_breakdown`, `drop_zero_runs` y el histograma
    de rachas del EDA.

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_units`.
        start_d (int): primer día.
        end_d (int): último día.

    Retorno:
        pd.DataFrame: `item_id, position, start_d, end_d, length`; `start_d` y `end_d` son el
        primer y el último día con cero de la racha.
    """
    u = _window(agg["item_units"], start_d, end_d)
    ids = np.asarray(agg["item_ids"], dtype=object)
    rows = []
    for i in range(u.shape[0]):
        days = np.flatnonzero(~np.isnan(u[i]))
        if not len(days):
            continue
        zero = u[i, days] == 0
        if not zero.any():
            continue
        edges = np.diff(np.r_[0, zero.astype(np.int8), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        sales = np.flatnonzero(~zero)
        first, last = (sales[0], sales[-1]) if len(sales) else (len(zero), -1)
        for s, e in zip(starts, ends):
            position = "leading" if s < first else "trailing" if s > last else "internal"
            rows.append(
                (ids[i], position, start_d + int(days[s]), start_d + int(days[e - 1]), int(e - s))
            )
    return pd.DataFrame(rows, columns=ZERO_RUN_COLUMNS)


def _run_kind(runs: pd.DataFrame, long_run: int) -> pd.Series:
    """Tipo de cada racha: `leading`, `trailing`, `long_run` (interna ≥ `long_run`) o `short`."""
    internal = np.where(runs["length"] >= long_run, "long_run", "short")
    return pd.Series(
        np.where(runs["position"] == "internal", internal, runs["position"]), index=runs.index
    )


def zero_breakdown(
    agg: dict,
    start_d: int,
    end_d: int,
    long_run: int,
    active_stores: np.ndarray | None = None,
    runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Descompone los días con cero de cada artículo según su origen probable.

    - `leading`: ceros entre el primer día activo y la primera venta. El prelanzamiento ya es NaN,
      así que esto es el arranque: precio publicado sin ventas todavía.
    - `trailing`: ceros después de la última venta (el artículo dejó de venderse).
    - `long_run`: rachas internas de al menos `long_run` días (episodio de escasez o de falta de
      surtido).
    - `short`: ceros internos sueltos o en rachas cortas (demanda intermitente).

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_units`.
        start_d (int): primer día.
        end_d (int): último día.
        long_run (int): longitud mínima, en días activos, de una racha larga.
        active_stores (np.ndarray | None): tiendas con dato por artículo y día (salida de
            `active_stores_by_item`), alineado con `agg["item_ids"]`.
        runs (pd.DataFrame | None): salida de `zero_runs` ya calculada para el mismo rango.

    Retorno:
        pd.DataFrame: `item_id, active_days, zero_days, leading, trailing, long_run, short,
        n_long_runs, max_internal_run` y, si se pasa `active_stores`, `mean_active_stores`.
    """
    ids = np.asarray(agg["item_ids"], dtype=object)
    u = _window(agg["item_units"], start_d, end_d)
    active = ~np.isnan(u)
    if runs is None:
        runs = zero_runs(agg, start_d, end_d)
    runs = runs.assign(kind=_run_kind(runs, long_run))
    kinds = ["leading", "trailing", "long_run", "short"]
    by_kind = (
        runs.groupby(["item_id", "kind"])["length"].sum().unstack(fill_value=0)
        if len(runs)
        else pd.DataFrame(columns=kinds, dtype=int)
    )
    by_kind = by_kind.reindex(columns=kinds, fill_value=0)
    internal = runs[runs["position"] == "internal"].groupby("item_id")["length"]
    out = pd.DataFrame(
        {
            "item_id": ids,
            "active_days": active.sum(axis=1),
            "zero_days": (u == 0).sum(axis=1),
        }
    )
    out = out.join(by_kind.reindex(ids, fill_value=0).reset_index(drop=True))
    long_counts = runs[runs["kind"] == "long_run"].groupby("item_id").size()
    out["n_long_runs"] = long_counts.reindex(ids, fill_value=0).to_numpy()
    out["max_internal_run"] = internal.max().reindex(ids, fill_value=0).to_numpy()
    if active_stores is not None:
        stores = _window(active_stores, start_d, end_d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out["mean_active_stores"] = np.nanmean(np.where(active, stores, np.nan), axis=1)
    return out


def drop_zero_runs(
    item_units: np.ndarray, item_ids, runs: pd.DataFrame, kinds: set[str], long_run: int
) -> np.ndarray:
    """Copia de las unidades con los días de ciertas rachas de ceros puestos en NaN.

    Sirve para medir cuánto del CV viene de esas rachas: con `block28`, `compute_cv` descarta
    entero el bloque que contiene un día NaN.

    Parámetros:
        item_units (np.ndarray): unidades por artículo y día.
        item_ids (array-like): identificadores alineados con las filas de `item_units`.
        runs (pd.DataFrame): salida de `zero_runs`.
        kinds (set[str]): tipos a quitar (`leading`, `trailing`, `long_run`, `short`).
        long_run (int): longitud mínima de una racha larga.

    Retorno:
        np.ndarray: unidades float64 con esas rachas en NaN.
    """
    out = np.array(item_units, dtype=np.float64)
    sel = runs[_run_kind(runs, long_run).isin(kinds)]
    rows = pd.Index(np.asarray(item_ids, dtype=object)).get_indexer(sel["item_id"])
    for r, a, b in zip(rows, sel["start_d"], sel["end_d"]):
        out[r, a - 1 : b] = np.nan
    return out


def _prob_no_run(p0: float, n: int, k: int) -> float:
    """P(L_n < k): ninguna racha de ceros de largo ≥ k en n días Bernoulli(p0) independientes.

    a_m = 1 si m < k y a_m = Σ_{j=0}^{k−1} p0^j (1 − p0) a_{m−j−1} si m ≥ k (el primer día con
    venta llega después de j ceros). Es un filtro lineal, así que se resuelve con `lfilter`.
    """
    if n < k:
        return 1.0
    coef = (1 - p0) * p0 ** np.arange(k)
    den = np.r_[1.0, -coef]
    zi = lfiltic([1.0], den, y=np.ones(k))
    out, _ = lfilter([1.0], den, np.zeros(n - k + 1), zi=zi)
    return float(out[-1])


def longest_run_threshold(p0: float, n: int, alpha: float) -> int | None:
    """Menor k con P(racha de ceros más larga ≥ k) ≤ α en n días Bernoulli(p0) (FR-032, R15).

    Parámetros:
        p0 (float): tasa de ceros del artículo.
        n (int): días activos.
        alpha (float): probabilidad máxima de ver por azar una racha de largo ≥ k.

    Retorno:
        int | None: umbral; None si p0 ∈ {0, 1}, n < 1 o ningún k ≤ n cumple.
    """
    if n < 1 or not 0 < p0 < 1:
        return None
    if p0**n > alpha:  # incluso una racha de n días es probable: no hay umbral
        return None
    lo, hi = 1, n  # P(L_n ≥ k) decrece con k: búsqueda binaria del primer k que cumple
    while lo < hi:
        mid = (lo + hi) // 2
        if 1 - _prob_no_run(p0, n, mid) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


ZERO_THRESHOLD_COLUMNS = [
    "item_id", "n_days", "p0", "threshold", "source", "reason", "ww_z", "ww_pvalue",
]


def zero_run_thresholds(
    agg: dict, start_d: int, end_d: int, alpha: float, long_run: int
) -> pd.DataFrame:
    """Umbral de racha de ceros de cada artículo, estimado en [start_d, end_d] (FR-032, R15).

    n y p₀ se miden en los días activos entre la primera y la última venta del rango (las rachas
    de arranque y finales se tratan aparte). Si el umbral no se puede estimar se usa `long_run`
    y `reason` dice por qué: `no_sales`, `no_zeros` (p₀ = 0), `all_zeros` (p₀ = 1) o
    `no_threshold` (ningún k ≤ n cumple). Diagnóstico: prueba de rachas de Wald–Wolfowitz
    (venta / sin venta) sobre los mismos días; z > 0 = menos rachas que el azar, ceros agrupados.

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_units`.
        start_d (int): primer día (train).
        end_d (int): último día (train).
        alpha (float): `zero_alpha`.
        long_run (int): respaldo `long_run`.

    Retorno:
        pd.DataFrame: columnas `ZERO_THRESHOLD_COLUMNS`; `source` = `estimated` o `fallback`.
    """
    u = _window(agg["item_units"], start_d, end_d)
    ids = np.asarray(agg["item_ids"], dtype=object)
    rows = []
    for i in range(u.shape[0]):
        x = u[i][~np.isnan(u[i])]
        sales = np.flatnonzero(x > 0)
        if not len(sales):
            rows.append((ids[i], 0, np.nan, long_run, "fallback", "no_sales", np.nan, np.nan))
            continue
        zero = x[sales[0] : sales[-1] + 1] == 0
        n, p0 = len(zero), float(zero.mean())
        ww_z = ww_p = np.nan
        if 0 < p0 < 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # runstest_1samp devuelve z < 0 con pocas rachas; se cambia el signo.
                z, ww_p = runstest_1samp(zero.astype(float), cutoff=0.5, correction=False)
            ww_z = -float(z)
        k = longest_run_threshold(p0, n, alpha)
        if k is None:
            reason = "no_zeros" if p0 == 0 else "all_zeros" if p0 == 1 else "no_threshold"
            rows.append((ids[i], n, p0, long_run, "fallback", reason, ww_z, ww_p))
        else:
            rows.append((ids[i], n, p0, k, "estimated", None, ww_z, ww_p))
    out = pd.DataFrame(rows, columns=ZERO_THRESHOLD_COLUMNS)
    out["threshold"] = out["threshold"].astype("int64")
    return out


def mask_zero_runs(
    item_units: np.ndarray,
    item_ids,
    thresholds: pd.Series,
    start_d: int,
    end_d: int,
    kinds: tuple[str, ...] = ("leading", "internal", "trailing"),
) -> np.ndarray:
    """Copia de las unidades con ciertas rachas de ceros de [start_d, end_d] en NaN (FR-032).

    - `internal`: rachas internas de largo ≥ umbral del artículo (los artículos sin umbral no se
      tocan).
    - `leading` y `trailing`: ceros antes de la primera venta y después de la última del rango.
    Las posiciones se miden dentro del rango; el prelanzamiento ya es NaN y los ceros sueltos se
    conservan. En el CV de XYZ se usan los tres tipos (como en la 6.4–6.5 del nb 02); en la serie
    de participación, `leading` e `internal`.
    Justificación (Principio III, b): CV de XYZ en train y en la réplica, y serie de pronóstico.

    Parámetros:
        item_units (np.ndarray): unidades por artículo y día (desde d_1).
        item_ids (array-like): identificadores alineados con las filas.
        thresholds (pd.Series): `item_id` → umbral (columna `threshold` de `zero_run_thresholds`).
        start_d (int): primer día del rango.
        end_d (int): último día del rango.
        kinds (tuple[str, ...]): tipos de racha a enmascarar.

    Retorno:
        np.ndarray: unidades float64 del mismo tamaño que `item_units`.
    """
    ids = np.asarray(item_ids, dtype=object)
    runs = zero_runs({"item_ids": ids, "item_units": item_units}, start_d, end_d)
    thr = runs["item_id"].map(thresholds).to_numpy(dtype=float)
    internal = (runs["position"] == "internal").to_numpy() & (runs["length"].to_numpy() >= thr)
    edge = runs["position"].isin([k for k in kinds if k != "internal"]).to_numpy()
    sel = runs[(internal & ("internal" in kinds)) | edge]
    out = np.array(item_units, dtype=np.float64)
    rows = pd.Index(ids).get_indexer(sel["item_id"])
    for r, a, b in zip(rows, sel["start_d"], sel["end_d"]):
        block = out[r, a - 1 : b]
        block[block == 0] = np.nan  # los NaN previos de la racha se quedan como están
    return out


def seasonal_strength(series, period: int, robust: bool = False) -> dict:
    """Fuerza estacional y de tendencia de una serie con STL (Hyndman y Athanasopoulos).

    F_S = max(0, 1 − Var(R) / Var(S + R)) y F_T = max(0, 1 − Var(R) / Var(T + R)), con R el resto,
    S la estacionalidad y T la tendencia de STL. Se recortan los ceros y NaN del inicio y del final
    (antes de la primera venta y después de la última), y los NaN internos se interpolan
    linealmente porque STL no admite faltantes. Por defecto STL no es robusta, como en la
    definición de referencia y porque la robusta tarda unas 10 veces más (con M5, ~0,16 s frente a
    ~0,015 s por serie diaria). Con `robust=True` los valores atípicos, como las rachas de ceros,
    quedan en el resto y F_S baja. F_S no parte de 0 con ruido: con la ventana estacional por
    defecto de STL, el ruido blanco da F_S ≈ 0,36 con período 7 (1.820 días) y ≈ 0,4–0,48 con
    período 52 y 5 ciclos, así que los valores se leen contra esa línea base.
    Justificación (Principio III, b): se usa con período semanal y anual en
    `seasonal_strength_items`.

    Parámetros:
        series (array-like): valores en orden temporal.
        period (int): período estacional (7 para días, 52 para semanas).
        robust (bool): STL robusta a valores atípicos.

    Retorno:
        dict: `fs`, `ft`, `n_obs` y `n_interpolated`; `fs` y `ft` son NaN con menos de dos
        períodos completos.
    """
    x = np.asarray(series, dtype=float)
    pos = np.flatnonzero(x > 0)
    empty = {"fs": np.nan, "ft": np.nan, "n_obs": 0, "n_interpolated": 0}
    if not len(pos):
        return empty
    x = x[pos[0] : pos[-1] + 1]
    n_interp = int(np.isnan(x).sum())
    if len(x) < 2 * period:
        return dict(empty, n_obs=len(x), n_interpolated=n_interp)
    if n_interp:
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    fit = STL(x, period=period, robust=robust).fit()
    r, s, t = fit.resid, fit.seasonal, fit.trend

    def _strength(component: np.ndarray) -> float:
        total = np.var(component + r)
        return float(max(0.0, 1 - np.var(r) / total)) if total > 0 else np.nan

    return {"fs": _strength(s), "ft": _strength(t), "n_obs": len(x), "n_interpolated": n_interp}


def seasonal_strength_items(
    agg: dict, start_d: int, end_d: int, week_of_day: np.ndarray, robust: bool = False
) -> pd.DataFrame:
    """Fuerza estacional semanal y anual de cada artículo.

    - Semanal: serie diaria, período 7.
    - Anual: unidades por semana `wm_yr_wk` completa, período 52. Una semana con algún día NaN
      queda NaN y se interpola.

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_units`.
        start_d (int): primer día.
        end_d (int): último día.
        week_of_day (np.ndarray): `wm_yr_wk` de cada día.
        robust (bool): STL robusta (ver `seasonal_strength`).

    Retorno:
        pd.DataFrame: `item_id, fs_weekly, ft_weekly, fs_annual, ft_annual, n_weeks,
        n_interpolated_weeks`.
    """
    u = _window(agg["item_units"], start_d, end_d)
    weeks = _block_sums(u, _weekly_bounds(week_of_day, start_d, end_d))
    rows = []
    for i in range(u.shape[0]):
        w = seasonal_strength(u[i], 7, robust)
        a = seasonal_strength(weeks[i], 52, robust)
        rows.append((w["fs"], w["ft"], a["fs"], a["ft"], a["n_obs"], a["n_interpolated"]))
    out = pd.DataFrame(
        rows,
        columns=[
            "fs_weekly",
            "ft_weekly",
            "fs_annual",
            "ft_annual",
            "n_weeks",
            "n_interpolated_weeks",
        ],
    )
    out.insert(0, "item_id", np.asarray(agg["item_ids"], dtype=object))
    return out


def monthly_seasonal_index(
    agg: dict,
    item_groups: pd.Series,
    month_of_day: np.ndarray,
    year_ranges: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Índice estacional mensual por grupo: unidades diarias del mes / unidades diarias del año.

    Se calcula por artículo y se resume con la mediana del grupo, para que los lanzamientos no
    muevan el índice. Solo entran los artículos activos todo el año. Se usan medias diarias
    porque los años de 364 días no empiezan el día 1 de un mes.

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_units`.
        item_groups (pd.Series): grupo por `item_id` (p. ej. "AX").
        month_of_day (np.ndarray): mes (1–12) de cada día; la posición j es el día j + 1.
        year_ranges (dict): nombre del año → (día inicial, día final).

    Retorno:
        pd.DataFrame: `year, group, month, index, n_items`.
    """
    ids = np.asarray(agg["item_ids"], dtype=object)
    groups = item_groups.reindex(ids).to_numpy()
    rows = []
    for year, (a, b) in year_ranges.items():
        u = _window(agg["item_units"], a, b)
        full = ~np.isnan(u).any(axis=1)
        months = np.asarray(month_of_day)[a - 1 : b]
        year_mean = u.mean(axis=1)
        for m in np.unique(months):
            # Los artículos no activos todo el año dan NaN aquí y se filtran con `full`.
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = u[:, months == m].mean(axis=1) / year_mean
            frame = pd.DataFrame({"group": groups, "ratio": ratio})[full & (year_mean > 0)]
            summary = (
                frame.dropna(subset=["group"]).groupby("group")["ratio"].agg(["median", "size"])
            )
            for group, (median, size) in summary.iterrows():
                rows.append((year, group, int(m), float(median), int(size)))
    return pd.DataFrame(rows, columns=["year", "group", "month", "index", "n_items"])


def classical_seasonal_index(
    agg: dict,
    item_classes: pd.Series,
    dates,
    start_d: int,
    end_d: int,
) -> dict:
    """Índice estacional mensual por clase con el método clásico (razón a media móvil).

    Por clase: (1) solo artículos con dato todos los días de [start_d, end_d], para que los
    lanzamientos no creen saltos de nivel; (2) serie de unidades diarias promedio del agregado
    en cada mes calendario completo (la media diaria evita el efecto de días por mes);
    (3) media móvil centrada 2×12; (4) ratio = serie / media móvil; (5) promedio de los ratios
    de cada mes entre años; (6) normalización para que los 12 índices sumen 12.

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_units`.
        item_classes (pd.Series): clase por `item_id` (p. ej. "A", "X" o "AX"); NaN o "-" se omiten.
        dates: fecha de cada día; la posición j es el día j + 1.
        start_d (int): primer día de la ventana.
        end_d (int): último día de la ventana.

    Retorno:
        dict: `index` (clase × mes 1–12, normalizado), `ratios` (`class, period, month, ratio`),
        `n_items` (artículos usados por clase) e `items` (`item_id` → clase de los usados).
    """
    ids = np.asarray(agg["item_ids"], dtype=object)
    classes = item_classes.reindex(ids).to_numpy(dtype=object)
    u = _window(agg["item_units"], start_d, end_d)
    keep = ~np.isnan(u).any(axis=1) & pd.notna(classes) & (classes != "-")

    period = pd.DatetimeIndex(np.asarray(dates)[start_d - 1 : end_d]).to_period("M")
    counts = pd.Series(1, index=period).groupby(level=0).size()
    complete = counts.index[counts.to_numpy() == counts.index.days_in_month]
    if len(complete) < 24:
        raise ValueError("se necesitan al menos 24 meses completos")

    weights = np.r_[0.5, np.ones(11), 0.5] / 12
    index_rows, ratio_frames = {}, []
    for cls in sorted(set(classes[keep])):
        daily = u[keep & (classes == cls)].sum(axis=0)
        x = pd.Series(daily, index=period).groupby(level=0).mean().reindex(complete)
        cma = np.convolve(x.to_numpy(), weights, mode="valid")  # alineada con x[6:-6]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = x.to_numpy()[6:-6] / cma
        centered = x.index[6:-6]
        frame = pd.DataFrame(
            {"class": cls, "period": centered, "month": centered.month, "ratio": ratio}
        )
        raw = frame.groupby("month")["ratio"].mean().reindex(range(1, 13))
        index_rows[cls] = raw * 12 / raw.sum()
        ratio_frames.append(frame)

    items = pd.Series(classes[keep], index=ids[keep], name="class")
    return {
        "index": pd.DataFrame(index_rows).T.rename_axis(index="class", columns="month"),
        "ratios": pd.concat(ratio_frames, ignore_index=True),
        "n_items": items.value_counts().rename("n_items"),
        "items": items,
    }


def _friedman_stat(ranks: np.ndarray) -> np.ndarray:
    """Estadístico de Friedman sobre rangos (…, bloques, tratamientos), sin corrección por empates."""
    b, k = ranks.shape[-2:]
    r = ranks.sum(axis=-2)
    return 12 / (b * k * (k + 1)) * (r**2).sum(axis=-1) - 3 * b * (k + 1)


def friedman_seasonality(ratios: pd.DataFrame, n_perm: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Prueba de Friedman de estacionalidad estable sobre los ratios del método clásico.

    Bloques = años calendario con los 12 ratios; tratamientos = meses. H0: dentro de cada año
    el orden de los meses es aleatorio (ningún mes tiende a quedar arriba o abajo). Como hay
    pocos bloques, además del p-valor asintótico (χ² con 11 g.l.) se da uno por permutación:
    se barajan los meses dentro de cada año. Kendall W = χ² / (b·(k − 1)) mide la consistencia
    del orden entre años (0 = ninguna, 1 = el mismo orden todos los años).

    Parámetros:
        ratios (pd.DataFrame): `class, period, month, ratio` (salida de `classical_seasonal_index`).
        n_perm (int): permutaciones para el p-valor exacto aproximado.
        seed (int): semilla.

    Retorno:
        pd.DataFrame: por clase, `n_years, chi2, p_asymptotic, p_permutation, kendall_w`.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for cls, sub in ratios.groupby("class", sort=False):
        table = (
            sub.assign(year=sub["period"].dt.year)
            .pivot(index="year", columns="month", values="ratio")
            .reindex(columns=range(1, 13))
            .dropna()
        )
        b, k = table.shape
        if b < 2:
            rows[cls] = {"n_years": b}
            continue
        ranks = rankdata(table.to_numpy(), axis=1)
        stat = float(_friedman_stat(ranks))
        perm = rng.permuted(np.tile(ranks, (n_perm, 1, 1)), axis=2)
        p_perm = (1 + (_friedman_stat(perm) >= stat - 1e-9).sum()) / (n_perm + 1)
        rows[cls] = {
            "n_years": b,
            "chi2": stat,
            "p_asymptotic": float(chi2.sf(stat, k - 1)),
            "p_permutation": float(p_perm),
            "kendall_w": stat / (b * (k - 1)),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("class")


# --- Ciclo de vida y enmascarado de quiebres (llevados del nb 03, Etapa 1) -----------------------
# Justificación (Principio III, b): los usan el nb 03 (segmentación) y el nb 04 (pronóstico).
# Trabajan sobre DataFrames artículo × día con columnas `d_<día>`.


def _day_numbers(frame: pd.DataFrame) -> np.ndarray:
    """Número de día de cada columna `d_<día>`."""
    return np.array([int(c[2:]) for c in frame.columns])


def lifecycle_bounds(units: pd.DataFrame) -> pd.DataFrame:
    """Primer y último día con venta > 0 de cada artículo dentro de las columnas de `units`.

    Parámetros:
        units (pd.DataFrame): unidades artículo × día (`d_<día>`); NaN = sin dato.

    Retorno:
        pd.DataFrame: `first_sale_d`, `last_sale_d` y `lifecycle_days` por artículo (NaN si el
        artículo no vende en el rango).
    """
    days = _day_numbers(units)
    sold = units.to_numpy() > 0
    has_sale = sold.any(axis=1)
    first = np.where(has_sale, days[sold.argmax(axis=1)], np.nan)
    last = np.where(has_sale, days[len(days) - 1 - sold[:, ::-1].argmax(axis=1)], np.nan)
    out = pd.DataFrame({"first_sale_d": first, "last_sale_d": last}, index=units.index)
    out["lifecycle_days"] = out["last_sale_d"] - out["first_sale_d"] + 1
    return out


def trim_lifecycle(units: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Copia de `units` con NaN antes de `first_sale_d` y después de `last_sale_d`.

    Parámetros:
        units (pd.DataFrame): unidades artículo × día.
        bounds (pd.DataFrame): salida de `lifecycle_bounds` para los mismos artículos.

    Retorno:
        pd.DataFrame: unidades recortadas, mismo índice y columnas que `units`.
    """
    days = _day_numbers(units)
    inside = (days >= bounds["first_sale_d"].to_numpy()[:, None]) & (
        days <= bounds["last_sale_d"].to_numpy()[:, None]
    )
    return units.where(inside)


def find_zero_runs(units_lc: pd.DataFrame) -> pd.DataFrame:
    """Rachas de ceros de cada artículo con `zero_runs`, a partir de un DataFrame artículo × día.

    Parámetros:
        units_lc (pd.DataFrame): unidades artículo × día con columnas contiguas `d_<día>`.

    Retorno:
        pd.DataFrame: `item_id, position, start_d, end_d, length` (días en base 1).
    """
    days = _day_numbers(units_lc)
    start_d, end_d = int(days[0]), int(days[-1])
    full = np.full((len(units_lc), end_d), np.nan)
    full[:, start_d - 1 :] = units_lc.to_numpy()
    agg = {"item_ids": units_lc.index.to_numpy(dtype=object), "item_units": full}
    return zero_runs(agg, start_d, end_d)


def classify_runs(runs: pd.DataFrame, long_run: pd.Series) -> pd.DataFrame:
    """Etiqueta cada racha interna: `break` si su largo ≥ LONG_RUN del artículo, `demand` si no.

    Las rachas no internas (`leading`, `trailing`) conservan su posición como tipo y nunca se
    enmascaran. Lanza ValueError si un artículo con rachas no tiene LONG_RUN.

    Parámetros:
        runs (pd.DataFrame): salida de `find_zero_runs`.
        long_run (pd.Series): `item_id` → LONG_RUN congelado.

    Retorno:
        pd.DataFrame: `runs` con las columnas `long_run` y `kind`.
    """
    out = runs.assign(long_run=runs["item_id"].map(long_run))
    if out["long_run"].isna().any():
        raise ValueError("hay artículos con rachas y sin LONG_RUN.")
    internal = (out["position"] == "internal").to_numpy()
    long = (out["length"] >= out["long_run"]).to_numpy()
    out["kind"] = np.where(internal, np.where(long, "break", "demand"), out["position"])
    return out


def mask_breaks(units_lc: pd.DataFrame, runs_kind: pd.DataFrame) -> pd.DataFrame:
    """Copia de `units_lc` con los ceros de las rachas `break` en NaN (los NaN previos se quedan).

    Parámetros:
        units_lc (pd.DataFrame): unidades artículo × día.
        runs_kind (pd.DataFrame): salida de `classify_runs` sobre los mismos artículos.

    Retorno:
        pd.DataFrame: unidades con los quiebres enmascarados.
    """
    out = units_lc.to_numpy(copy=True)
    start_d = int(_day_numbers(units_lc)[0])
    breaks = runs_kind[runs_kind["kind"] == "break"]
    rows = units_lc.index.get_indexer(breaks["item_id"])
    for r, a, b in zip(rows, breaks["start_d"].to_numpy(), breaks["end_d"].to_numpy()):
        block = out[r, a - start_d : b - start_d + 1]
        block[block == 0] = np.nan
    return pd.DataFrame(out, index=units_lc.index, columns=units_lc.columns)
