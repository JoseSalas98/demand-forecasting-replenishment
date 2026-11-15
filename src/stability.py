"""
Justificación de las abstracciones (Principio III): `year_matrices`, `base_target_classes` y
`transition_stats` se usan en el EDA, en la estabilidad de train y en la réplica (b). `psi` y
`psi_band` quedan solo para la descripción del EDA (datos no pareados).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import SquareTable

from src.segmentation import (
    abc_rule_kwargs,
    build_matrix,
    classify_abc_share,
    classify_xyz,
    cv_cut_kwargs,
    find_cv_cuts,
    mask_zero_runs,
    select_abc_cuts,
)

# Valores del PSI para la descripción del EDA (nb 02, 8.1). No son parámetros de decisión.
EDA_PSI = {"psi_epsilon": 1e-4, "psi_bands": [0.10, 0.25]}


def year_matrices(
    agg: dict,
    year_ranges: dict[str, tuple[int, int]],
    seg: dict,
    week_of_day: np.ndarray | None = None,
    segment_bounds_by_year: dict[str, list[tuple[int, int]]] | None = None,
    cv_units: np.ndarray | None = None,
) -> pd.DataFrame:
    """Construye una matriz ABC × XYZ por cada año recibido.

    Parámetros:
        agg (dict): agregado.
        year_ranges (dict[str, tuple[int, int]]): nombre del año → (día inicial, día final).
        seg (dict): parámetros de segmentación.
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día.
        segment_bounds_by_year (dict | None): segmentos por año (para `segment_period`).
        cv_units (np.ndarray | None): unidades para el CV (rachas de ceros en NaN, FR-032).

    Retorno:
        pd.DataFrame: matrices concatenadas con `set_id` = nombre del año.
    """
    frames = []
    for name, (start, end) in year_ranges.items():
        bounds = (segment_bounds_by_year or {}).get(name)
        frames.append(build_matrix(name, agg, start, end, seg, week_of_day, bounds, cv_units))
    return pd.concat(frames, ignore_index=True)


def agreement_pct(a: pd.Series, b: pd.Series) -> float:
    """Porcentaje de artículos con la misma clase entre los que tienen clase en ambos lados.

    Parámetros:
        a (pd.Series): clase por artículo (índice `item_id`).
        b (pd.Series): clase por artículo (índice `item_id`).

    Retorno:
        float: acuerdo en porcentaje (0–100); NaN si no hay artículos en común.
    """
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if joined.empty:
        return float("nan")
    return float((joined["a"] == joined["b"]).mean() * 100)


def class_proportions(classes: pd.Series, labels: list[str]) -> dict[str, float]:
    """Proporción de cada clase entre los artículos con clase.

    Parámetros:
        classes (pd.Series): clase por artículo.
        labels (list[str]): clases posibles (p. ej. ["A", "B", "C"]).

    Retorno:
        dict[str, float]: clase → proporción.
    """
    valid = classes.dropna()
    if valid.empty:
        return {k: 0.0 for k in labels}
    counts = valid.value_counts(normalize=True)
    return {k: float(counts.get(k, 0.0)) for k in labels}


def psi(p_ref: dict[str, float], p_cur: dict[str, float], eps: float) -> float:
    """Índice de estabilidad poblacional sobre proporciones de clase.

    PSI = Σ (p_cur − p_ref) · ln(p_cur / p_ref); las proporciones 0 se reemplazan por `eps`.

    Parámetros:
        p_ref (dict[str, float]): proporciones de referencia.
        p_cur (dict[str, float]): proporciones actuales.
        eps (float): valor para clases vacías (`psi_epsilon`).

    Retorno:
        float: PSI.
    """
    total = 0.0
    for key in sorted(set(p_ref) | set(p_cur)):
        r = max(p_ref.get(key, 0.0), eps)
        c = max(p_cur.get(key, 0.0), eps)
        total += (c - r) * np.log(c / r)
    return float(total)


def psi_band(value: float, bands: list[float]) -> str:
    """Clasifica un PSI en `stable`, `moderate` o `shift` según `psi_bands`.

    Parámetros:
        value (float): PSI.
        bands (list[float]): umbrales [moderado, cambio].

    Retorno:
        str: banda.
    """
    if np.isnan(value):
        return "undefined"
    if value < bands[0]:
        return "stable"
    if value < bands[1]:
        return "moderate"
    return "shift"


ABC_LABELS = ["A", "B", "C"]
XYZ_LABELS = ["X", "Y", "Z"]
CELL_LABELS = [a + x for a in ABC_LABELS for x in XYZ_LABELS]
SEGMENT_PREFIX = {"quarter": "Q", "four_month": "T"}


def _fully_active(agg: dict, day_range: tuple[int, int]) -> np.ndarray:
    """Artículos con dato en todos los días del rango."""
    a, b = day_range
    u = np.asarray(agg["item_units"][:, a - 1 : b], dtype=float)
    return (~np.isnan(u)).sum(axis=1) == (b - a + 1)


def _set_column(mats: pd.DataFrame, set_id: str, col: str, item_ids: np.ndarray) -> np.ndarray:
    """Columna `col` de un conjunto, alineada con `item_ids` (NaN si el artículo no aparece)."""
    sub = mats[mats["set_id"] == set_id].set_index("item_id").reindex(item_ids)
    return sub[col].to_numpy(dtype=float)


def _join_cell(abc: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Celda ABC × XYZ; None si falta alguna de las dos clases."""
    out = np.full(len(abc), None, dtype=object)
    ok = pd.notna(abc) & pd.notna(xyz)
    out[ok] = abc[ok].astype(str) + xyz[ok].astype(str)
    return out


def base_target_classes(
    mats: pd.DataFrame,
    ranges: dict[str, tuple[int, int]],
    base_ids: list[str],
    target_id: str,
    agg: dict,
    abc_rule: dict,
    psi_cfg: dict | None = None,
    cv_cut_kw: dict | None = None,
) -> dict:
    """Selecciona los cortes en una base y clasifica con ellos la base y un conjunto objetivo.

    Igual que train → test: los cortes se aprenden en la base y se aplican al objetivo.
        - Base: promedio del gasto y del CV de cada artículo en los conjuntos de `base_ids` en
          que estuvo activo todos los días (como FR-016).
        - ABC: `select_abc_cuts` con `abc_rule` sobre el gasto de la base; el objetivo se
          clasifica por participación con las fronteras de la base (`classify_abc_share`).
        - XYZ (si `cv_cut_kw` no es None): `find_cv_cuts` sobre el CV de la base; los umbrales
          se aplican al CV del objetivo, que debe tener la misma agregación.
          Si `find_cv_cuts` no encuentra una partición válida, el XYZ queda vacío y el motivo va
          en `summary["note"]` (Principio IV).
        - PSI de los grupos homogéneos (solo si `psi_cfg` no es None, descripción del EDA): % de
          artículos en cada celda ABC × XYZ de la base frente al objetivo, con los artículos que
          tienen celda completa en los dos lados. Sin XYZ no hay PSI.
    Justificación (Principio III, b): se usa en el análisis interanual y en el intraanual.

    Parámetros:
        mats (pd.DataFrame): matrices con `set_id`, `item_id`, `spend` y `cv` (p. ej. de
            `year_matrices` o `period_matrices`); sus clases no se usan.
        ranges (dict): `set_id` → (día inicial, día final), para medir la actividad.
        base_ids (list[str]): conjuntos que forman la base.
        target_id (str): conjunto objetivo.
        agg (dict): agregado con `item_ids` e `item_units`.
        abc_rule (dict): argumentos de `select_abc_cuts` (`item_bounds`, `spend_bounds`, `step`).
        psi_cfg (dict | None): `psi_epsilon` y `psi_bands` (p. ej. `EDA_PSI`); None = sin PSI.
        cv_cut_kw (dict | None): argumentos de `find_cv_cuts` (`n_cuts` = 2, `min_share`,
            `resolution`).

    Retorno:
        dict con:
            - `items` (pd.DataFrame): por artículo, `status` (`both`, `new` = sin base,
              `no_target` = sin objetivo, `absent`), `ref_*` y `tgt_*` de gasto, CV, ABC, XYZ y
              celda.
            - `summary` (dict): cortes y fronteras usados, conteos, `psi` y `psi_band`.
            - `proportions` (pd.DataFrame | None): `cell, base_pct, target_pct` de los artículos
              comparados; None sin XYZ.
            - `abc_scenarios` (pd.DataFrame): escenarios evaluados por `select_abc_cuts`.
    """
    if cv_cut_kw is not None and cv_cut_kw.get("n_cuts", 2) != 2:
        raise ValueError("base_target_classes necesita n_cuts = 2 (clases X, Y y Z)")
    item_ids = np.asarray(agg["item_ids"], dtype=object)
    spend_cols, cv_cols = [], []
    for set_id in base_ids:
        active = _fully_active(agg, ranges[set_id])
        spend_cols.append(np.where(active, _set_column(mats, set_id, "spend", item_ids), np.nan))
        cv_cols.append(np.where(active, _set_column(mats, set_id, "cv", item_ids), np.nan))
    spend = np.column_stack(spend_cols)
    n_base_sets = (~np.isnan(spend)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ref_spend = np.nanmean(spend, axis=1)
        ref_cv = np.nanmean(np.column_stack(cv_cols), axis=1)
    in_base = n_base_sets > 0
    in_target = _fully_active(agg, ranges[target_id])
    tgt_spend = np.where(in_target, _set_column(mats, target_id, "spend", item_ids), np.nan)
    tgt_cv = np.where(in_target, _set_column(mats, target_id, "cv", item_ids), np.nan)

    rule = select_abc_cuts(item_ids[in_base], ref_spend[in_base], **abc_rule)
    ref_abc = (
        pd.Series(rule["abc"]["abc_class"].to_numpy(), index=rule["abc"]["item_id"].to_numpy())
        .reindex(item_ids)
        .to_numpy(dtype=object)
    )
    tgt_abc = classify_abc_share(tgt_spend, rule["share_bounds"])

    none = np.full(len(item_ids), None, dtype=object)
    ref_xyz, tgt_xyz, thresholds, note = none.copy(), none.copy(), [np.nan, np.nan], None
    if cv_cut_kw is not None:
        has_cv = in_base & ~np.isnan(ref_cv)
        try:
            thresholds = find_cv_cuts(ref_cv[has_cv], ref_spend[has_cv], **cv_cut_kw)[
                "thresholds"
            ]
        except ValueError as exc:
            note = f"sin XYZ: {exc}"
        else:
            ref_xyz = classify_xyz(pd.Series(np.where(in_base, ref_cv, np.nan)), thresholds)
            ref_xyz = ref_xyz.to_numpy(dtype=object)
            tgt_xyz = classify_xyz(pd.Series(tgt_cv), thresholds).to_numpy(dtype=object)
    ref_cell = _join_cell(ref_abc, ref_xyz) if cv_cut_kw is not None else ref_abc
    tgt_cell = _join_cell(tgt_abc, tgt_xyz) if cv_cut_kw is not None else tgt_abc

    status = np.select(
        [in_base & in_target, in_target, in_base], ["both", "new", "no_target"], "absent"
    )
    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "status": status,
            "n_base_sets": n_base_sets,
            "ref_spend_mean": ref_spend,
            "ref_cv_mean": ref_cv,
            "ref_abc": ref_abc,
            "ref_xyz": ref_xyz,
            "ref_cell": ref_cell,
            "tgt_spend": tgt_spend,
            "tgt_cv": tgt_cv,
            "tgt_abc": tgt_abc,
            "tgt_xyz": tgt_xyz,
            "tgt_cell": tgt_cell,
        }
    )

    value, proportions = np.nan, None
    both = (status == "both") & pd.notna(ref_cell) & pd.notna(tgt_cell)
    if cv_cut_kw is not None and note is None:
        p_ref = class_proportions(pd.Series(ref_cell[both]), CELL_LABELS)
        p_tgt = class_proportions(pd.Series(tgt_cell[both]), CELL_LABELS)
        if psi_cfg is not None:
            value = psi(p_ref, p_tgt, psi_cfg["psi_epsilon"])
        proportions = pd.DataFrame(
            {
                "cell": CELL_LABELS,
                "base_pct": [p_ref[c] * 100 for c in CELL_LABELS],
                "target_pct": [p_tgt[c] * 100 for c in CELL_LABELS],
            }
        )
    summary = {
        "target": target_id,
        "base": "+".join(base_ids),
        "cut_A": rule["cuts"][0],
        "cut_B": rule["cuts"][1],
        "share_A": rule["share_bounds"][0],
        "share_B": rule["share_bounds"][1],
        "cv_X": thresholds[0],
        "cv_Z": thresholds[1],
        "n_base": int(in_base.sum()),
        "n_target": int(in_target.sum()),
        "n_compared": int(both.sum()),
        "n_new": int((status == "new").sum()),
        "n_no_target": int((status == "no_target").sum()),
        "note": note,
    }
    if psi_cfg is not None:
        summary.update({"psi": value, "psi_band": psi_band(value, psi_cfg["psi_bands"])})
    return {
        "items": items,
        "summary": summary,
        "proportions": proportions,
        "abc_scenarios": rule["scenarios"],
    }


def segment_bounds(kind: str, start_d: int, end_d: int) -> list[tuple[int, int]]:
    """Divide un año en periodos de segmentación por semanas completas.

    `quarter` = 4 periodos de 13 semanas; `four_month` = 3 periodos de 17/17/18 semanas (en un año
    de 52 semanas). Si el año no tiene 52 semanas, se reparte igual con el resto en el último.

    Parámetros:
        kind (str): `quarter` o `four_month`.
        start_d (int): primer día del año.
        end_d (int): último día del año.

    Retorno:
        list[tuple[int, int]]: (día inicial, día final) de cada periodo, inclusivos.
    """
    n_weeks = (end_d - start_d + 1) // 7
    parts = {"quarter": 4, "four_month": 3}[kind]
    base = n_weeks // parts
    weeks = [base] * (parts - 1) + [n_weeks - base * (parts - 1)]
    bounds, cursor = [], start_d
    for k, w in enumerate(weeks):
        last = end_d if k == parts - 1 else cursor + 7 * w - 1
        bounds.append((cursor, last))
        cursor = last + 1
    return bounds


def segment_bounds_by_year(
    kind: str, year_ranges: dict[str, tuple[int, int]]
) -> dict[str, list[tuple[int, int]]]:
    """Periodos de segmentación de cada año.

    Parámetros:
        kind (str): `quarter` o `four_month`.
        year_ranges (dict): nombre del año → (día inicial, día final).

    Retorno:
        dict[str, list[tuple[int, int]]]: año → periodos.
    """
    return {name: segment_bounds(kind, a, b) for name, (a, b) in year_ranges.items()}


def _period_seg(seg: dict) -> dict:
    # Dentro de un periodo, `segment_period` sería un solo bloque; el criterio de elección usa
    # solo ABC, así que el CV de estas matrices se calcula con bloques de 28 días.
    """Ajusta la agregación del CV para matrices de periodos (bloques de 28 días)."""
    if seg["cv_aggregation"] == "segment_period":
        return dict(seg, cv_aggregation="block28")
    return seg


def period_matrices(
    agg: dict,
    year_ranges: dict[str, tuple[int, int]],
    seg: dict,
    kind: str,
    week_of_day: np.ndarray | None = None,
) -> pd.DataFrame:
    """Matrices ABC × XYZ de cada periodo de segmentación dentro de cada año.

    Parámetros:
        agg (dict): agregado.
        year_ranges (dict): años de train.
        seg (dict): parámetros de segmentación.
        kind (str): `quarter` o `four_month`.
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día.

    Retorno:
        pd.DataFrame: matrices con `set_id` como `Y1-Q1` o `Y1-T1`.
    """
    frames = []
    pseg = _period_seg(seg)
    for name, (a, b) in year_ranges.items():
        for k, (s, e) in enumerate(segment_bounds(kind, a, b), start=1):
            set_id = f"{name}-{SEGMENT_PREFIX[kind]}{k}"
            frames.append(build_matrix(set_id, agg, s, e, pseg, week_of_day))
    return pd.concat(frames, ignore_index=True)


def interleaved_halves(
    agg: dict,
    set_id: str,
    start_d: int,
    end_d: int,
    week_of_day: np.ndarray | None = None,
) -> pd.DataFrame:
    """Gasto de cada artículo en las semanas impares y en las pares de un periodo.

    Las dos mitades cubren la misma época y tienen casi el mismo largo, así que la diferencia de
    clase entre ellas mide el ruido de la ventana, no la estacionalidad. Las semanas se numeran
    por orden dentro del periodo (1.ª, 2.ª, …): impares = 1.ª, 3.ª, …
    Justificación (Principio III, b): línea base de ruido del análisis intraanual del nb 02.

    Parámetros:
        agg (dict): agregado con `item_ids` e `item_spend`.
        set_id (str): conjunto del periodo (p. ej. `Y1-Q1`).
        start_d (int): primer día del periodo.
        end_d (int): último día del periodo.
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día; si es None, bloques de 7 días
            desde `start_d`.

    Retorno:
        pd.DataFrame: `set_id` (`{set_id}-odd` y `{set_id}-even`), `item_id`, `spend` y `cv`
            (NaN: el CV de media ventana no se usa).
    """
    days = np.arange(start_d, end_d + 1)
    if week_of_day is None:
        week = (days - start_d) // 7
    else:
        week = np.unique(np.asarray(week_of_day)[days - 1], return_inverse=True)[1]
    spend = np.asarray(agg["item_spend"][:, start_d - 1 : end_d], dtype=float)
    frames = []
    for name, mask in (("odd", week % 2 == 0), ("even", week % 2 == 1)):
        frames.append(
            pd.DataFrame(
                {
                    "set_id": f"{set_id}-{name}",
                    "item_id": agg["item_ids"],
                    "spend": np.nansum(spend[:, mask], axis=1),
                    "cv": np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def period_pairs(year_names: list[str], n_periods: int, prefix: str) -> pd.DataFrame:
    """Pares base → objetivo entre periodos del mismo largo.

    - `adjacent`: cada periodo frente al siguiente (Yt-P4 → Yt+1-P1 cruza el año).
    - `same_last_year`: el mismo periodo del año anterior frente al de este año.
    Justificación (Principio III, b): comparaciones del análisis intraanual del nb 02.

    Parámetros:
        year_names (list[str]): años en orden (p. ej. Y1–Y5).
        n_periods (int): periodos por año (4 trimestres o 3 cuatrimestres).
        prefix (str): prefijo del periodo (`Q` o `T`, ver `SEGMENT_PREFIX`).

    Retorno:
        pd.DataFrame: `comparison`, `base_id` y `target_id`.
    """
    seq = [f"{y}-{prefix}{k}" for y in year_names for k in range(1, n_periods + 1)]
    rows = [("adjacent", a, b) for a, b in zip(seq, seq[1:])]
    rows += [("same_last_year", a, b) for a, b in zip(seq, seq[n_periods:])]
    return pd.DataFrame(rows, columns=["comparison", "base_id", "target_id"])


TRANSITION_LEVELS = {
    "abc": (ABC_LABELS, ("ref_spend_mean", "tgt_spend")),
    "xyz": (XYZ_LABELS, ("ref_cv_mean", "tgt_cv")),
    "cell": (CELL_LABELS, None),
}


def _stuart_maxwell(counts: np.ndarray) -> tuple[float, int, float]:
    """Prueba de Stuart–Maxwell (homogeneidad marginal pareada) sobre las clases presentes."""
    present = (counts.sum(axis=0) + counts.sum(axis=1)) > 0
    sub = counts[np.ix_(present, present)]
    k = int(present.sum())
    if k < 2:
        return np.nan, 0, np.nan
    if np.trace(sub) == sub.sum():  # nadie cambia: estadístico 0 (la covarianza es singular)
        return 0.0, k - 1, 1.0
    with warnings.catch_warnings():
        # Con casi nadie cambiando la covarianza puede ser singular; statsmodels usa la pseudoinversa.
        warnings.simplefilter("ignore")
        res = SquareTable(sub, shift_zeros=False).homogeneity(method="stuart_maxwell")
    return float(res.statistic), int(res.df), float(res.pvalue)


def _move_shares(counts: np.ndarray, level: str) -> dict:
    """% que mantiene, mejora, empeora o cambia en sentidos opuestos (solo `cell`).

    Mejor = posición más baja en el orden de cada dimensión (A < B < C, X < Y < Z). En `cell`
    cada artículo se compara en ABC y en XYZ por separado: mejora si sube en alguna sin bajar en
    la otra, empeora si baja en alguna sin subir en la otra, y es mixto si sube en una y baja en
    la otra (p. ej. AZ → BX).
    """
    i, j = np.indices(counts.shape)
    k = len(XYZ_LABELS)
    deltas = [j // k - i // k, j % k - i % k] if level == "cell" else [j - i]
    better = np.all([d <= 0 for d in deltas], axis=0)
    worse = np.all([d >= 0 for d in deltas], axis=0)
    keep = better & worse
    n = counts.sum()
    return {
        "keep_pct": counts[keep].sum() / n * 100,
        "up_pct": counts[better & ~keep].sum() / n * 100,
        "down_pct": counts[worse & ~keep].sum() / n * 100,
        "mixed_pct": counts[~better & ~worse].sum() / n * 100 if level == "cell" else np.nan,
    }


def transition_stats(items: pd.DataFrame, level: str = "abc") -> dict:
    """Medidas de cambio de clase entre la base y el objetivo de `base_target_classes`.

    Usa los artículos con `status == "both"` y clase en los dos lados. `level` elige la clase:
    `abc`, `xyz` o `cell` (ABC × XYZ).
        - `keep_pct`, `up_pct`, `down_pct`: % que mantiene, mejora (C → A, Z → X) y empeora.
          En `abc` y `xyz` son la diagonal, debajo y encima de la matriz de transición. En
          `cell` el orden AX…CZ no es ordinal, así que se mira cada dimensión por separado
          (`_move_shares`) y se agrega `mixed_pct`: sube en una dimensión y baja en la otra.
        - `kappa`: kappa de Cohen sin ponderar (plano), acuerdo por artículo descontado el azar.
        - `mobility`: índice de Shorrocks, (k − traza(P)) / (k − 1), con P normalizada por fila
          y k las clases presentes en la base; 0 = nadie cambia.
        - `spearman`: correlación de rangos del gasto (`abc`) o del CV (`xyz`) de la base y del
          objetivo; NaN en `cell`.
        - `stuart_maxwell`, `sm_df`, `sm_pvalue`: homogeneidad marginal pareada (¿cambia el % de
          artículos por clase?), sobre las clases presentes.
        - `max_shift_pp`, `max_shift_class`: mayor cambio del % de artículos de una clase, en
          puntos porcentuales; da el tamaño del cambio que Stuart–Maxwell solo declara.
    Justificación (Principio III, b): resume las comparaciones interanuales e intraanuales del
    nb 02.

    Parámetros:
        items (pd.DataFrame): `items` de `base_target_classes`.
        level (str): `abc`, `xyz` o `cell`.

    Retorno:
        dict con `n`, `keep_pct`, `up_pct`, `down_pct`, `mixed_pct` (NaN salvo en `cell`),
        `kappa`, `mobility`, `spearman`, `stuart_maxwell`, `sm_df`, `sm_pvalue`,
        `max_shift_pp` y `max_shift_class`.
    """
    labels, value_cols = TRANSITION_LEVELS[level]
    ref, tgt = f"ref_{level}", f"tgt_{level}"
    both = items[(items["status"] == "both") & items[ref].notna() & items[tgt].notna()]
    counts = (
        pd.crosstab(both[ref], both[tgt])
        .reindex(index=labels, columns=labels, fill_value=0)
        .to_numpy()
    )
    n = int(counts.sum())
    out = {"n": n}
    keys = [
        "keep_pct", "up_pct", "down_pct", "mixed_pct", "kappa", "mobility", "spearman",
        "stuart_maxwell", "sm_df", "sm_pvalue", "max_shift_pp", "max_shift_class",
    ]
    if n == 0:
        return {**out, **dict.fromkeys(keys, np.nan)}
    rows = counts.sum(axis=1)
    present = rows > 0
    p = counts[present] / rows[present, None]
    k = int(present.sum())
    trace = float(p[:, present].diagonal().sum()) if k else 0.0
    shift = (counts.sum(axis=0) - rows) / n * 100
    sm_stat, sm_df, sm_p = _stuart_maxwell(counts)
    spearman = np.nan
    if value_cols is not None:
        spearman = float(spearmanr(both[value_cols[0]], both[value_cols[1]]).statistic)
    out.update(
        {
            **_move_shares(counts, level),
            "kappa": float(cohen_kappa_score(both[ref], both[tgt], labels=labels)),
            "mobility": (k - trace) / (k - 1) if k > 1 else np.nan,
            "spearman": spearman,
            "stuart_maxwell": sm_stat,
            "sm_df": sm_df,
            "sm_pvalue": sm_p,
            "max_shift_pp": float(np.abs(shift).max()),
            "max_shift_class": labels[int(np.abs(shift).argmax())],
        }
    )
    return out


def _classes(mat: pd.DataFrame, set_id: str, col: str) -> pd.Series:
    """Devuelve la columna `col` de un conjunto indexada por artículo."""
    sub = mat[mat["set_id"] == set_id]
    return pd.Series(sub[col].to_numpy(), index=sub["item_id"].to_numpy())


def prior_years_map(year_names: list[str]) -> dict[str, list[str]]:
    """Cada año desde el segundo contra todos los anteriores (solo la usa el nb 02 original).

    Parámetros:
        year_names (list[str]): años en orden (p. ej. Y1–Y4).

    Retorno:
        dict[str, list[str]]: año comparado → años previos (Y2 ← [Y1], Y3 ← [Y1, Y2], …).
    """
    return {year_names[k]: year_names[:k] for k in range(1, len(year_names))}


def comparison_plan(year_names: list[str]) -> list[tuple[str, str, str]]:
    """Comparaciones pareadas de la estabilidad: `(base_kind, base, target)` (FR-016, R14).

    Cada año desde el segundo frente al año anterior (`prev`) y frente al primero (`y1`). Con el
    segundo año las dos bases coinciden y se reportan las dos filas.

    Parámetros:
        year_names (list[str]): años en orden.

    Retorno:
        list[tuple[str, str, str]]: comparaciones en orden.
    """
    out = []
    for k in range(1, len(year_names)):
        out.append(("prev", year_names[k - 1], year_names[k]))
        out.append(("y1", year_names[0], year_names[k]))
    return out


def alternating_block_cv(
    units: np.ndarray, item_ids, set_id: str, start_d: int, end_d: int, min_blocks: int
) -> pd.DataFrame:
    """CV de bloques de 28 días alternos (impares y pares) de un periodo: ruido del CV.

    Igual que `interleaved_halves` para el gasto: las dos mitades cubren la misma época, así que
    su diferencia de clase XYZ mide el ruido de la estimación del CV. Un bloque con algún NaN se
    descarta; CV = std poblacional / media × 100, NaN con media 0 o menos de `min_blocks` bloques.

    Parámetros:
        units (np.ndarray): unidades por artículo y día (con las rachas ya en NaN si aplica).
        item_ids (array-like): identificadores alineados con las filas.
        set_id (str): nombre del periodo (p. ej. `Y2`).
        start_d (int): primer día.
        end_d (int): último día.
        min_blocks (int): mínimo de bloques completos por mitad.

    Retorno:
        pd.DataFrame: `set_id` (`{set_id}-odd`, `{set_id}-even`), `item_id` y `cv`.
    """
    u = np.asarray(units[:, start_d - 1 : end_d], dtype=float)
    n_blocks = u.shape[1] // 28
    sums = u[:, : n_blocks * 28].reshape(u.shape[0], n_blocks, 28).sum(axis=2)
    frames = []
    for name, cols in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
        x = sums[:, cols]
        n_ok = (~np.isnan(x)).sum(axis=1)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(x, axis=1)
            cv = np.nanstd(x, axis=1) / mean * 100
        cv = np.where((n_ok >= min_blocks) & (mean > 0), cv, np.nan)
        frames.append(pd.DataFrame({"set_id": f"{set_id}-{name}", "item_id": item_ids, "cv": cv}))
    return pd.concat(frames, ignore_index=True)


STABILITY_COLUMNS = [
    "base_kind", "base", "target", "level", "n", "kappa", "stuart_maxwell", "sm_df",
    "sm_pvalue", "max_shift_pp", "max_shift_class", "keep_pct", "up_pct", "down_pct",
    "mixed_pct", "spearman", "note",
]


def _stat_rows(base_kind: str, res: dict) -> list[dict]:
    """Filas de `stability.parquet` de una comparación: una por nivel (abc, xyz, cell)."""
    s = res["summary"]
    rows = []
    for level in TRANSITION_LEVELS:
        st = transition_stats(res["items"], level)
        rows.append(
            {"base_kind": base_kind, "base": s["base"], "target": s["target"], "level": level,
             **st, "note": s.get("note")}
        )
    return rows


def paired_comparisons(
    mats: pd.DataFrame,
    ranges: dict[str, tuple[int, int]],
    plan: list[tuple[str, str, str]],
    agg: dict,
    seg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Ejecuta un plan de comparaciones pareadas con los cortes elegidos en cada base (R14).

    Parámetros:
        mats (pd.DataFrame): matrices con `set_id`, `item_id`, `spend` y `cv`.
        ranges (dict): `set_id` → (día inicial, día final).
        plan (list): `(base_kind, base, target)`.
        agg (dict): agregado.
        seg (dict): `[segmentation]` (o lo congelado) con la regla ABC y `find_cv_cuts`.

    Retorno:
        tuple[pd.DataFrame, dict]: medidas (`STABILITY_COLUMNS`) y resultados de
        `base_target_classes` por `(base_kind, base, target)`.
    """
    rows, results = [], {}
    for base_kind, base, target in plan:
        res = base_target_classes(
            mats, ranges, [base], target, agg, abc_rule_kwargs(seg), None, cv_cut_kwargs(seg)
        )
        results[(base_kind, base, target)] = res
        rows += _stat_rows(base_kind, res)
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS), results


def noise_baseline(
    agg: dict,
    cv_units: np.ndarray,
    year_ranges: dict[str, tuple[int, int]],
    seg: dict,
    week_of_day: np.ndarray | None = None,
) -> pd.DataFrame:
    """Línea base de ruido: semanas alternas (gasto) y bloques de 28 días alternos (CV) de cada año.

    Las mitades impar y par de un año se comparan como base → objetivo con las mismas medidas
    que las comparaciones entre años; lo que cambia entre ellas es ruido de la ventana.

    Parámetros:
        agg (dict): agregado.
        cv_units (np.ndarray): unidades para el CV (rachas en NaN).
        year_ranges (dict): años de train.
        seg (dict): `[segmentation]`.
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día.

    Retorno:
        pd.DataFrame: filas con `base_kind = "noise"`.
    """
    frames, ranges, plan = [], {}, []
    for name, (a, b) in year_ranges.items():
        spend = interleaved_halves(agg, name, a, b, week_of_day).drop(columns="cv")
        cv = alternating_block_cv(cv_units, agg["item_ids"], name, a, b, seg["cv_min_blocks"])
        frames.append(spend.merge(cv, on=["set_id", "item_id"]))
        ranges[f"{name}-odd"] = ranges[f"{name}-even"] = (a, b)
        plan.append(("noise", f"{name}-odd", f"{name}-even"))
    measures, _ = paired_comparisons(pd.concat(frames, ignore_index=True), ranges, plan, agg, seg)
    return measures


def _abc_all(st: pd.DataFrame) -> pd.Series:
    """Regla: misma clase ABC en todas las comparaciones."""
    return st.groupby("item_id")["same_abc"].all()


def _abc_xyz_all(st: pd.DataFrame) -> pd.Series:
    """Regla: misma clase ABC y XYZ en todas las comparaciones."""
    return (st["same_abc"] & st["same_xyz"]).groupby(st["item_id"]).all()


def _cell_majority(st: pd.DataFrame) -> pd.Series:
    """Regla: misma celda en al menos dos tercios de las comparaciones."""
    g = st.groupby("item_id")["same_cell"]
    # "Al menos 2 de 3" generalizado: al menos dos tercios de las comparaciones.
    return g.sum() * 3 >= 2 * g.size()


def _a_every_year(st: pd.DataFrame) -> pd.Series:
    """Regla: clase A en cada año de train."""
    return st.groupby("item_id")["a_every_year"].first().astype(bool)


# Reglas intercambiables de estabilidad (FR-018). Misma entrada (filas por artículo y comparación)
# y misma salida (estable sí/no por artículo). Justificación (Principio III, condición a): regla de
# negocio que cambia de forma independiente, con 4 implementaciones reales.
STABILITY_RULES = {
    "abc_all": _abc_all,
    "abc_xyz_all": _abc_xyz_all,
    "cell_majority": _cell_majority,
    "a_every_year": _a_every_year,
}


def build_item_stability(
    results: dict, comparisons: list[tuple[str, str, str]], cfg: dict
) -> pd.DataFrame:
    """Compara la clase de cada artículo con su base y aplica la regla elegida (FR-018).

    Usa las comparaciones con base año anterior. Una comparación sin clase en algún lado cuenta
    como no coincidente. `a_every_year`: A en la base de la primera comparación y en el objetivo
    de todas.

    Parámetros:
        results (dict): salida de `paired_comparisons` (por `(base_kind, base, target)`).
        comparisons (list): comparaciones que entran en la regla (las de `prev`).
        cfg (dict): configuración (`stability.stability_rule`).

    Retorno:
        pd.DataFrame: `item_id, comparison, abc, ref_abc, xyz, ref_xyz, same_abc, same_xyz,
        same_cell, a_every_year, stable, rule`.
    """
    frames = []
    for key in comparisons:
        it = results[key]["items"]
        same_abc = it["tgt_abc"].notna() & (it["tgt_abc"] == it["ref_abc"])
        same_xyz = it["tgt_xyz"].notna() & (it["tgt_xyz"] == it["ref_xyz"])
        frames.append(
            pd.DataFrame(
                {
                    "item_id": it["item_id"].to_numpy(),
                    "comparison": f"{key[1]}->{key[2]}",
                    "abc": it["tgt_abc"].to_numpy(),
                    "ref_abc": it["ref_abc"].to_numpy(),
                    "xyz": it["tgt_xyz"].to_numpy(),
                    "ref_xyz": it["ref_xyz"].to_numpy(),
                    "same_abc": same_abc.to_numpy(),
                    "same_xyz": same_xyz.to_numpy(),
                    "same_cell": (same_abc & same_xyz).to_numpy(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    first = results[comparisons[0]]["items"].set_index("item_id")["ref_abc"].eq("A")
    tgt_a = out.assign(a=out["abc"].eq("A")).groupby("item_id")["a"].all()
    out["a_every_year"] = out["item_id"].map(first & tgt_a.reindex(first.index, fill_value=False))
    out["a_every_year"] = out["a_every_year"].fillna(False).astype(bool)
    rule = cfg["stability"]["stability_rule"]
    stable = STABILITY_RULES[rule](out)
    out["stable"] = out["item_id"].map(stable).astype(bool)
    out["rule"] = rule
    return out


def build_targets(matrix_train: pd.DataFrame, item_stab: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Construye la lista objetivo: artículos A en train y estables, con su tratamiento (FR-019).

    Parámetros:
        matrix_train (pd.DataFrame): matriz global de train congelada.
        item_stab (pd.DataFrame): estabilidad por artículo (`item_id`, `stable`).
        cfg (dict): configuración (`treatment_map`).

    Retorno:
        pd.DataFrame: `item_id, cell, xyz_class, spend_train, spend_share, treatment`, por gasto.
        Lanza ValueError si una celda no tiene tratamiento o si la lista queda vacía.
    """
    stable = item_stab.groupby("item_id")["stable"].first()
    total = matrix_train["spend"].sum()
    mt = matrix_train.set_index("item_id")
    ids = [i for i in mt.index[mt["abc_class"] == "A"] if bool(stable.get(i, False))]
    if not ids:
        raise ValueError("La lista objetivo quedó vacía: ningún artículo A es estable en train.")
    sel = mt.loc[ids]
    tmap = cfg["treatment_map"]
    unmapped = sorted(set(sel["cell"]) - set(tmap))
    if unmapped:
        raise ValueError(f"Celdas sin tratamiento en treatment_map: {unmapped}")
    out = pd.DataFrame(
        {
            "item_id": ids,
            "cell": sel["cell"].to_numpy(),
            "xyz_class": sel["xyz_class"].to_numpy(),
            "spend_train": sel["spend"].to_numpy(dtype=float),
        }
    )
    out["spend_share"] = out["spend_train"] / total
    out["treatment"] = out["cell"].map(tmap)
    return out.sort_values(["spend_train", "item_id"], ascending=[False, True]).reset_index(
        drop=True
    )


STABILITY_DECLARATION_KEYS = [
    "stability.stability_rule",
    "stability.stability_rule_criterion",
    "segmentation.cv_aggregation",
    "segmentation.abc_rule_item_bounds",
    "segmentation.abc_rule_spend_bounds",
    "segmentation.abc_rule_step",
    "segmentation.cv_cut_min_share",
    "segmentation.cv_cut_resolution",
    "treatment_map",
]

SEG_KEYS = [
    "cv_aggregation",
    "cv_min_blocks",
    "abc_rule_item_bounds",
    "abc_rule_spend_bounds",
    "abc_rule_step",
    "cv_cut_min_share",
    "cv_cut_resolution",
    "zero_alpha",
    "long_run",
]


def _masked_units(agg: dict, zero_thresholds: pd.DataFrame, windows: list[tuple[int, int]]):
    """Unidades con las rachas de ceros en NaN, ventana por ventana (umbrales congelados)."""
    thr = zero_thresholds.set_index("item_id")["threshold"]
    out = np.asarray(agg["item_units"], dtype=float)
    for a, b in windows:
        out = mask_zero_runs(out, agg["item_ids"], thr, a, b)
    return out


def run_stability_stage(cfg: dict) -> dict:
    """Etapa 4: estabilidad pareada Y2–Y4, línea base de ruido, lista objetivo y congelamiento.

    Registra las declaraciones antes de calcular y solo carga días de train. El CV usa las
    unidades con las rachas de ceros en NaN según `zero_thresholds` (congelado en la etapa 3).
    Cada año se compara con el anterior y con Y1, y la matriz global de train con cada año
    (`base_kind = "global"`); la regla de estabilidad usa las comparaciones con el año anterior.
    Congela `matrices_by_set`, `stability`, `item_stability`, `targets` y `segmentation_params`.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: resultados intermedios para mostrar en el notebook.
    """
    from src.config import train_end, train_years
    from src.data import assert_train_only, load_interim
    from src.freeze import freeze, load_frozen, record_declarations, require_declarations, utc_now

    record_declarations(STABILITY_DECLARATION_KEYS, cfg)
    require_declarations(STABILITY_DECLARATION_KEYS, utc_now(), cfg)
    max_day = train_end(cfg)
    assert_train_only(max_day, cfg)
    agg = load_interim(cfg, max_day)
    wod = agg["week_of_day"]
    matrix_train = load_frozen("matrix_train")
    train_cuts = load_frozen("train_cuts")
    zero_thr = load_frozen("zero_thresholds")
    years = train_years(cfg)
    seg = cfg["segmentation"]
    start = min(a for a, _ in years.values())
    masked = _masked_units(agg, zero_thr, [(start, max_day)])

    year_seg = dict(
        seg, abc_cuts=train_cuts["abc_cuts"], xyz_thresholds=train_cuts["xyz_thresholds"]
    )
    year_mats = year_matrices(agg, years, year_seg, wod, cv_units=masked)
    names = list(years)
    plan = comparison_plan(names)
    measures, results = paired_comparisons(year_mats, years, plan, agg, seg)
    glob = matrix_train.assign(set_id="train_global")
    ranges = {**years, "train_global": (start, max_day)}
    glob_plan = [("global", "train_global", n) for n in names]
    glob_measures, _ = paired_comparisons(
        pd.concat([year_mats, glob], ignore_index=True), ranges, glob_plan, agg, seg
    )
    noise = noise_baseline(agg, masked, years, seg, wod)
    measures = pd.concat([measures, glob_measures, noise], ignore_index=True)

    prev = [k for k in plan if k[0] == "prev"]
    item_stab = build_item_stability(results, prev, cfg)
    targets = build_targets(matrix_train, item_stab, cfg)
    seg_params = {k: seg[k] for k in SEG_KEYS}
    seg_params.update(
        {
            "abc_cuts": train_cuts["abc_cuts"],
            "abc_share_bounds": train_cuts["abc_share_bounds"],
            "xyz_thresholds": train_cuts["xyz_thresholds"],
            "stability_rule": cfg["stability"]["stability_rule"],
            "treatment_map": dict(cfg["treatment_map"]),
        }
    )
    freeze(year_mats, "matrices_by_set", "data/processed/matrices_by_set.parquet", cfg)
    freeze(measures, "stability", "data/processed/stability.parquet", cfg)
    freeze(item_stab, "item_stability", "data/processed/item_stability.parquet", cfg)
    freeze(targets, "targets", "data/processed/targets.parquet", cfg)
    freeze(seg_params, "segmentation_params", "data/processed/segmentation_params.json", cfg)
    return {
        "year_mats": year_mats,
        "comparisons": results,
        "measures": measures,
        "item_stability": item_stab,
        "targets": targets,
        "segmentation_params": seg_params,
    }


def replicate_year5(
    targets: pd.DataFrame,
    seg_params: dict,
    zero_thresholds: pd.DataFrame,
    agg: dict,
    cfg: dict,
    week_of_day: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replica sobre el año 5 la matriz y la estabilidad con lo congelado (FR-030).

    Usa la regla ABC, `find_cv_cuts`, `cv_aggregation`, los cortes y `zero_thresholds` congelados,
    nunca los de la config. Las rachas se enmascaran en la ventana de train y en la de Y5 por
    separado (en Y5 las posiciones de arranque y final se miden dentro de Y5). Bases: Y4 (año
    anterior) y Y1, con las medidas pareadas de `transition_stats`. `matrix_y5` usa los cortes
    congelados de train. Solo diagnóstico.

    Parámetros:
        targets (pd.DataFrame): lista objetivo congelada.
        seg_params (dict): parámetros de segmentación congelados.
        zero_thresholds (pd.DataFrame): umbrales de racha congelados.
        agg (dict): agregado con los días del año 5.
        cfg (dict): configuración (rangos de días).
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día.

    Retorno:
        tuple: `matrix_y5`, `stability_y5` y `year5_replication`.
    """
    from src.config import all_years, train_years

    years = all_years(cfg)
    train_names = list(train_years(cfg))
    test_name = next(n for n in years if n not in train_names)
    start = years[train_names[0]][0]
    masked = _masked_units(
        agg, zero_thresholds, [(start, years[train_names[-1]][1]), years[test_name]]
    )
    year_mats = year_matrices(agg, years, seg_params, week_of_day, cv_units=masked)
    plan = [("prev", train_names[-1], test_name), ("y1", train_names[0], test_name)]
    stability_y5, results = paired_comparisons(year_mats, years, plan, agg, seg_params)
    matrix_y5 = year_mats[year_mats["set_id"] == test_name].reset_index(drop=True)
    y5 = matrix_y5.set_index("item_id")
    cmp = results[plan[0]]["items"].set_index("item_id")
    ids = targets["item_id"].to_numpy()
    rep = pd.DataFrame(
        {
            "item_id": ids,
            "frozen_cell": targets["cell"].to_numpy(),
            "ref_abc": cmp["ref_abc"].reindex(ids).to_numpy(),
            "ref_xyz": cmp["ref_xyz"].reindex(ids).to_numpy(),
            "y5_abc": cmp["tgt_abc"].reindex(ids).to_numpy(),
            "y5_xyz": cmp["tgt_xyz"].reindex(ids).to_numpy(),
            "y5_cell": cmp["tgt_cell"].reindex(ids).to_numpy(),
        }
    )
    rep["same_abc"] = rep["y5_abc"].notna() & (rep["y5_abc"] == rep["ref_abc"])
    rep["same_xyz"] = rep["y5_xyz"].notna() & (rep["y5_xyz"] == rep["ref_xyz"])
    rep["same_cell"] = rep["same_abc"] & rep["same_xyz"]
    # Con los cortes congelados de train, no con los elegidos en Y4.
    rep["same_as_frozen"] = y5["cell"].reindex(ids).to_numpy() == rep["frozen_cell"].to_numpy()
    return matrix_y5, stability_y5, rep


def run_replication(cfg: dict) -> dict:
    """Etapa 5: declara `[forecast]` y replica lo congelado sobre el año 5 (FR-030).

    Orden: carga lo congelado (se detiene si falta), declara todas las claves de `[forecast]`,
    recién entonces lee los días del año 5, replica, congela y verifica que `targets` no cambió.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: `matrix_y5`, `stability_y5`, `year5_replication` y `declared_keys`.
    """
    from src.config import forecast_keys
    from src.data import load_interim
    from src.freeze import (
        freeze,
        load_frozen,
        load_manifest,
        record_declarations,
        require_declarations,
        utc_now,
        verify_manifest,
    )

    targets = load_frozen("targets")
    seg_params = load_frozen("segmentation_params")
    zero_thr = load_frozen("zero_thresholds")
    targets_sha = load_manifest()["artifacts"]["targets"]["sha256"]
    keys = forecast_keys(cfg)
    record_declarations(keys, cfg)
    require_declarations(keys, utc_now(), cfg)
    agg = load_interim(cfg, cfg["split"]["test_days"][1])
    matrix_y5, stability_y5, rep = replicate_year5(
        targets, seg_params, zero_thr, agg, cfg, agg["week_of_day"]
    )
    freeze(matrix_y5, "matrix_y5", "data/processed/matrix_y5.parquet", cfg)
    freeze(stability_y5, "stability_y5", "data/processed/stability_y5.parquet", cfg)
    freeze(rep, "year5_replication", "data/processed/year5_replication.parquet", cfg)
    verify_manifest()
    if load_manifest()["artifacts"]["targets"]["sha256"] != targets_sha:
        raise RuntimeError("La réplica modificó la lista objetivo; no debe ocurrir (FR-030).")
    return {
        "matrix_y5": matrix_y5,
        "stability_y5": stability_y5,
        "year5_replication": rep,
        "declared_keys": keys,
    }


EDA_DIR = "reports/eda_five_years"


def run_eda_five_years(cfg: dict) -> dict:
    """Etapa 2: exploración descriptiva de cinco años (FR-029). No congela ni decide nada.

    Calcula la matriz global de Y1–Y5, las matrices anuales, el acuerdo y el PSI de cada año
    frente al anterior y frente a la global (solo como descripción) y la descripción por artículo.
    Escribe todo en `reports/eda_five_years/` y verifica que el manifiesto no cambió.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: `matrix_five_years`, `year_matrices`, `stability_descriptive` y `describe`.
    """
    from pathlib import Path

    from src.config import all_years
    from src.data import load_interim
    from src.freeze import load_manifest
    from src.segmentation import describe_items

    before = load_manifest()
    years = all_years(cfg)
    names = list(years)
    start, end = years[names[0]][0], years[names[-1]][1]
    agg = load_interim(cfg, end)
    wod = agg["week_of_day"]
    seg = cfg["segmentation"]
    matrix_5y = build_matrix(f"{names[0]}-{names[-1]}", agg, start, end, seg, wod)
    ym = year_matrices(agg, years, seg, wod)
    stab = EDA_PSI
    rows = []
    for k, name in enumerate(names):
        for dim, col, labels in (
            ("abc", "abc_class", ABC_LABELS),
            ("xyz", "xyz_class", XYZ_LABELS),
        ):
            cur = _classes(ym, name, col)
            vs_global = agreement_pct(cur, _classes(matrix_5y, matrix_5y["set_id"].iloc[0], col))
            vs_prev, value = float("nan"), float("nan")
            if k > 0:
                prev = _classes(ym, names[k - 1], col)
                vs_prev = agreement_pct(cur, prev)
                value = psi(
                    class_proportions(prev, labels),
                    class_proportions(cur, labels),
                    stab["psi_epsilon"],
                )
            rows.append((name, dim, vs_prev, vs_global, value, psi_band(value, stab["psi_bands"])))
    stability_desc = pd.DataFrame(
        rows,
        columns=[
            "year",
            "dimension",
            "agreement_pct_vs_prev",
            "agreement_pct_vs_5y",
            "psi_vs_prev",
            "psi_band",
        ],
    )
    describe = describe_items(agg, start, end)
    out_dir = Path(EDA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_5y.to_parquet(out_dir / "matrix_five_years.parquet", index=False)
    ym.to_parquet(out_dir / "year_matrices.parquet", index=False)
    stability_desc.to_parquet(out_dir / "stability_descriptive.parquet", index=False)
    describe.to_parquet(out_dir / "describe_items.parquet", index=False)
    if load_manifest() != before:
        raise RuntimeError("La exploración modificó el manifiesto; no debe congelar nada.")
    return {
        "matrix_five_years": matrix_5y,
        "year_matrices": ym,
        "stability_descriptive": stability_desc,
        "describe": describe,
        "agg": agg,
    }
