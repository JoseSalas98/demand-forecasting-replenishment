"""
Justificación de las abstracciones (Principio III): `year_matrices`, `agreement_pct`, `psi` y
`psi_band` se usan en el EDA, en la estabilidad de train y en la réplica (b).
"""

import warnings

import numpy as np
import pandas as pd

from src.segmentation import build_matrix, classify_abc, classify_xyz


def year_matrices(
    agg: dict,
    year_ranges: dict[str, tuple[int, int]],
    seg: dict,
    week_of_day: np.ndarray | None = None,
    segment_bounds_by_year: dict[str, list[tuple[int, int]]] | None = None,
) -> pd.DataFrame:
    """Construye una matriz ABC × XYZ por cada año recibido.

    Parámetros:
        agg (dict): agregado.
        year_ranges (dict[str, tuple[int, int]]): nombre del año → (día inicial, día final).
        seg (dict): parámetros de segmentación.
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día.
        segment_bounds_by_year (dict | None): segmentos por año (para `segment_period`).

    Retorno:
        pd.DataFrame: matrices concatenadas con `set_id` = nombre del año.
    """
    frames = []
    for name, (start, end) in year_ranges.items():
        bounds = (segment_bounds_by_year or {}).get(name)
        frames.append(build_matrix(name, agg, start, end, seg, week_of_day, bounds))
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
SEGMENT_PREFIX = {"quarter": "Q", "four_month": "T"}


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


def _classes(mat: pd.DataFrame, set_id: str, col: str) -> pd.Series:
    """Devuelve la columna `col` de un conjunto indexada por artículo."""
    sub = mat[mat["set_id"] == set_id]
    return pd.Series(sub[col].to_numpy(), index=sub["item_id"].to_numpy())


def choose_segment_period(
    year_mats: pd.DataFrame, period_mats_by_kind: dict[str, pd.DataFrame], candidates: list[str]
) -> tuple[str, pd.DataFrame]:
    """Elige el periodo de segmentación con el criterio `max_mean_abc_agreement`.

    Para cada candidato se mide el acuerdo ABC entre cada matriz por periodo y la matriz de su
    año, y se promedia; gana el mayor y un empate va al primer candidato de la lista.

    Parámetros:
        year_mats (pd.DataFrame): matrices anuales.
        period_mats_by_kind (dict[str, pd.DataFrame]): candidato → matrices por periodo.
        candidates (list[str]): candidatos en el orden declarado.

    Retorno:
        tuple[str, pd.DataFrame]: periodo elegido y tabla `kind, set_id, agreement_pct`.
    """
    rows = []
    for kind in candidates:
        pm = period_mats_by_kind[kind]
        for set_id in pm["set_id"].unique():
            year = set_id.split("-")[0]
            rows.append(
                (
                    kind,
                    set_id,
                    agreement_pct(
                        _classes(pm, set_id, "abc_class"), _classes(year_mats, year, "abc_class")
                    ),
                )
            )
    table = pd.DataFrame(rows, columns=["kind", "set_id", "agreement_pct"])
    means = table.groupby("kind", sort=False)["agreement_pct"].mean()
    best = candidates[0]
    for kind in candidates[1:]:
        if means[kind] > means[best]:
            best = kind
    return best, table


def prior_years_map(year_names: list[str]) -> dict[str, list[str]]:
    """Comparaciones de train: cada año desde el segundo contra todos los anteriores.

    Parámetros:
        year_names (list[str]): años en orden (p. ej. Y1–Y4).

    Retorno:
        dict[str, list[str]]: año comparado → años previos (Y2 ← [Y1], Y3 ← [Y1, Y2], …).
    """
    return {year_names[k]: year_names[:k] for k in range(1, len(year_names))}


def reference_classes(
    year_mats: pd.DataFrame,
    agg: dict,
    seg: dict,
    comparison: str,
    prior_years: list[str],
    min_active_days_year: int,
    year_ranges: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Clase de referencia de cada artículo frente al promedio de sus años previos activos (FR-016).

    El gasto anual se promedia sobre los años previos en que el artículo está activo todo el año
    (al menos `min_active_days_year` días con dato) y se vuelve a clasificar ABC sobre esos
    promedios para todos los artículos con referencia; el CV anual se promedia y se clasifica XYZ.
    Justificación (Principio III, b): se usa en las comparaciones de train y en la réplica Y5.

    Parámetros:
        year_mats (pd.DataFrame): matrices anuales.
        agg (dict): agregado (para medir los días activos).
        seg (dict): parámetros de segmentación (cortes y umbrales).
        comparison (str): año comparado (p. ej. `Y3` o `Y5`).
        prior_years (list[str]): años previos.
        min_active_days_year (int): días con dato para contar un año como activo.
        year_ranges (dict): rangos de días de cada año.

    Retorno:
        pd.DataFrame: `comparison, item_id, ref_spend_mean, ref_abc, ref_cv_mean, ref_xyz,
        n_ref_years`.
    """
    item_ids = np.asarray(agg["item_ids"], dtype=object)
    spend_cols, cv_cols = [], []
    for year in prior_years:
        a, b = year_ranges[year]
        days_with_data = (~np.isnan(np.asarray(agg["item_units"][:, a - 1 : b], dtype=float))).sum(
            axis=1
        )
        active = days_with_data >= min_active_days_year
        ym = year_mats[year_mats["set_id"] == year].set_index("item_id").reindex(item_ids)
        spend_cols.append(np.where(active, ym["spend"].to_numpy(dtype=float), np.nan))
        cv_cols.append(np.where(active, ym["cv"].to_numpy(dtype=float), np.nan))
    spend = np.column_stack(spend_cols)
    cv = np.column_stack(cv_cols)
    n_ref = (~np.isnan(spend)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spend_mean = np.nanmean(spend, axis=1)
        cv_mean = np.nanmean(cv, axis=1)
    out = pd.DataFrame(
        {
            "comparison": comparison,
            "item_id": item_ids,
            "ref_spend_mean": spend_mean,
            "ref_cv_mean": cv_mean,
            "n_ref_years": n_ref,
        }
    )
    has_ref = out["n_ref_years"] > 0
    abc = classify_abc(
        out.loc[has_ref, "item_id"], out.loc[has_ref, "ref_spend_mean"].to_numpy(), seg["abc_cuts"]
    )
    out = out.merge(
        abc[["item_id", "abc_class"]].rename(columns={"abc_class": "ref_abc"}),
        on="item_id",
        how="left",
    )
    out["ref_xyz"] = classify_xyz(out["ref_cv_mean"], seg["xyz_thresholds"]).to_numpy()
    out.loc[~has_ref, "ref_xyz"] = None
    return out[
        [
            "comparison",
            "item_id",
            "ref_spend_mean",
            "ref_abc",
            "ref_cv_mean",
            "ref_xyz",
            "n_ref_years",
        ]
    ]


def stability_measures(
    year_mats: pd.DataFrame,
    matrix_global: pd.DataFrame,
    refs: pd.DataFrame,
    comparisons: list[str],
    prior_map: dict[str, list[str]],
    cfg: dict,
) -> pd.DataFrame:
    """Acuerdo y PSI de cada año comparado frente a su referencia y a la matriz global (FR-017).

    La distribución de referencia del PSI es el promedio simple de las proporciones de clase de
    los años previos.

    Parámetros:
        year_mats (pd.DataFrame): matrices anuales.
        matrix_global (pd.DataFrame): matriz global de train.
        refs (pd.DataFrame): clases de referencia por comparación.
        comparisons (list[str]): años comparados.
        prior_map (dict): año comparado → años previos.
        cfg (dict): configuración (`psi_epsilon`, `psi_bands`).

    Retorno:
        pd.DataFrame: `comparison, dimension, agreement_pct_vs_ref, agreement_pct_vs_global, psi,
        psi_band`.
    """
    stab = cfg["stability"]
    glob = matrix_global.set_index("item_id")
    rows = []
    for comp in comparisons:
        ref = refs[refs["comparison"] == comp].set_index("item_id")
        for dim, col, ref_col, labels in (
            ("abc", "abc_class", "ref_abc", ABC_LABELS),
            ("xyz", "xyz_class", "ref_xyz", XYZ_LABELS),
        ):
            cur = _classes(year_mats, comp, col)
            p_cur = class_proportions(cur, labels)
            priors = [
                class_proportions(_classes(year_mats, y, col), labels) for y in prior_map[comp]
            ]
            p_ref = {k: float(np.mean([p[k] for p in priors])) for k in labels}
            value = psi(p_ref, p_cur, stab["psi_epsilon"])
            rows.append(
                (
                    comp,
                    dim,
                    agreement_pct(cur, ref[ref_col]),
                    agreement_pct(cur, glob[col]),
                    value,
                    psi_band(value, stab["psi_bands"]),
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "dimension",
            "agreement_pct_vs_ref",
            "agreement_pct_vs_global",
            "psi",
            "psi_band",
        ],
    )


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
    year_mats: pd.DataFrame,
    refs: pd.DataFrame,
    comparisons: list[str],
    train_year_names: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """Compara la clase de cada artículo en cada año con su referencia y aplica la regla elegida.

    Una comparación sin referencia o sin clase cuenta como no coincidente.

    Parámetros:
        year_mats (pd.DataFrame): matrices anuales.
        refs (pd.DataFrame): referencias por comparación.
        comparisons (list[str]): años comparados (Y2–Y4).
        train_year_names (list[str]): años de train (para `a_every_year`).
        cfg (dict): configuración (`stability.stability_rule`).

    Retorno:
        pd.DataFrame: `item_id, comparison, abc, ref_abc, xyz, ref_xyz, same_abc, same_xyz,
        same_cell, a_every_year, stable, rule`.
    """
    items = pd.Index(sorted(year_mats["item_id"].unique()), name="item_id")
    a_flags = pd.concat(
        [_classes(year_mats, y, "abc_class").reindex(items).eq("A") for y in train_year_names],
        axis=1,
    ).all(axis=1)
    frames = []
    for comp in comparisons:
        cur = year_mats[year_mats["set_id"] == comp].set_index("item_id").reindex(items)
        ref = refs[refs["comparison"] == comp].set_index("item_id").reindex(items)
        same_abc = cur["abc_class"].notna() & (cur["abc_class"] == ref["ref_abc"])
        same_xyz = cur["xyz_class"].notna() & (cur["xyz_class"] == ref["ref_xyz"])
        frames.append(
            pd.DataFrame(
                {
                    "item_id": items,
                    "comparison": comp,
                    "abc": cur["abc_class"].to_numpy(),
                    "ref_abc": ref["ref_abc"].to_numpy(),
                    "xyz": cur["xyz_class"].to_numpy(),
                    "ref_xyz": ref["ref_xyz"].to_numpy(),
                    "same_abc": same_abc.to_numpy(),
                    "same_xyz": same_xyz.to_numpy(),
                    "same_cell": (same_abc & same_xyz).to_numpy(),
                    "a_every_year": a_flags.reindex(items).to_numpy(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
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
    "stability.segment_candidates",
    "stability.segment_criterion",
    "stability.stability_rule",
    "stability.stability_rule_criterion",
    "segmentation.cv_aggregation",
    "segmentation.abc_cuts",
    "segmentation.xyz_thresholds",
    "treatment_map",
]

SEG_KEYS = ["abc_cuts", "xyz_thresholds", "cv_aggregation", "cv_min_blocks"]


def run_stability_stage(cfg: dict) -> dict:
    """Etapa 4: estabilidad Y2–Y4, periodo de segmentación, lista objetivo y congelamiento.

    Registra las declaraciones antes de calcular, solo carga días de train y congela
    `matrices_by_set`, `stability`, `item_stability`, `targets` y `segmentation_params`.

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
    years = train_years(cfg)
    seg = cfg["segmentation"]
    stab = cfg["stability"]

    candidates = stab["segment_candidates"]
    period_mats = {k: period_matrices(agg, years, seg, k, wod) for k in candidates}
    year_mats_abc = year_matrices(agg, years, _period_seg(seg), wod)
    chosen, choice_table = choose_segment_period(year_mats_abc, period_mats, candidates)
    bounds = (
        segment_bounds_by_year(chosen, years)
        if seg["cv_aggregation"] == "segment_period"
        else None
    )
    year_mats = year_matrices(agg, years, seg, wod, bounds)

    names = list(years)
    prior = prior_years_map(names)
    comparisons = list(prior)
    refs = pd.concat(
        [
            reference_classes(
                year_mats, agg, seg, c, prior[c], stab["min_active_days_year"], years
            )
            for c in comparisons
        ],
        ignore_index=True,
    )
    measures = stability_measures(year_mats, matrix_train, refs, comparisons, prior, cfg)
    item_stab = build_item_stability(year_mats, refs, comparisons, names, cfg)
    targets = build_targets(matrix_train, item_stab, cfg)
    seg_params = {k: seg[k] for k in SEG_KEYS}
    seg_params.update(
        {
            "segment_period": chosen,
            "stability_rule": stab["stability_rule"],
            "treatment_map": dict(cfg["treatment_map"]),
        }
    )
    matrices_by_set = pd.concat([year_mats, *period_mats.values()], ignore_index=True)
    freeze(matrices_by_set, "matrices_by_set", "data/processed/matrices_by_set.parquet", cfg)
    freeze(measures, "stability", "data/processed/stability.parquet", cfg)
    freeze(item_stab, "item_stability", "data/processed/item_stability.parquet", cfg)
    freeze(targets, "targets", "data/processed/targets.parquet", cfg)
    freeze(seg_params, "segmentation_params", "data/processed/segmentation_params.json", cfg)
    return {
        "choice_table": choice_table,
        "segment_period": chosen,
        "year_mats": year_mats,
        "refs": refs,
        "measures": measures,
        "item_stability": item_stab,
        "targets": targets,
        "segmentation_params": seg_params,
    }


def replicate_year5(
    targets: pd.DataFrame,
    seg_params: dict,
    matrix_train: pd.DataFrame,
    agg: dict,
    cfg: dict,
    week_of_day: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replica sobre el año 5 la matriz y la estabilidad con los parámetros congelados (FR-030).

    Usa los cortes, umbrales, `cv_aggregation` y periodo de `segmentation_params`, nunca los de la
    config. La referencia de Y5 es el promedio de los años activos Y1–Y4. Solo diagnóstico.

    Parámetros:
        targets (pd.DataFrame): lista objetivo congelada.
        seg_params (dict): parámetros de segmentación congelados.
        matrix_train (pd.DataFrame): matriz global de train congelada.
        agg (dict): agregado con los días del año 5.
        cfg (dict): configuración (rangos de días, `psi_*`, `min_active_days_year`).
        week_of_day (np.ndarray | None): `wm_yr_wk` de cada día.

    Retorno:
        tuple: `matrix_y5`, `stability_y5` y `year5_replication`.
    """
    from src.config import all_years, train_years

    seg = {k: seg_params[k] for k in SEG_KEYS}
    years = all_years(cfg)
    train_names = list(train_years(cfg))
    test_name = next(n for n in years if n not in train_names)
    bounds = (
        segment_bounds_by_year(seg_params["segment_period"], years)
        if seg["cv_aggregation"] == "segment_period"
        else None
    )
    year_mats = year_matrices(agg, years, seg, week_of_day, bounds)
    refs = reference_classes(
        year_mats,
        agg,
        seg,
        test_name,
        train_names,
        cfg["stability"]["min_active_days_year"],
        years,
    )
    stability_y5 = stability_measures(
        year_mats, matrix_train, refs, [test_name], {test_name: train_names}, cfg
    )
    matrix_y5 = year_mats[year_mats["set_id"] == test_name].reset_index(drop=True)
    y5 = matrix_y5.set_index("item_id")
    ref = refs.set_index("item_id")
    ids = targets["item_id"].to_numpy()
    rep = pd.DataFrame(
        {
            "item_id": ids,
            "frozen_cell": targets["cell"].to_numpy(),
            "ref_abc": ref["ref_abc"].reindex(ids).to_numpy(),
            "ref_xyz": ref["ref_xyz"].reindex(ids).to_numpy(),
            "y5_abc": y5["abc_class"].reindex(ids).to_numpy(),
            "y5_xyz": y5["xyz_class"].reindex(ids).to_numpy(),
            "y5_cell": y5["cell"].reindex(ids).to_numpy(),
        }
    )
    rep["same_abc"] = rep["y5_abc"].notna() & (rep["y5_abc"] == rep["ref_abc"])
    rep["same_xyz"] = rep["y5_xyz"].notna() & (rep["y5_xyz"] == rep["ref_xyz"])
    rep["same_cell"] = rep["same_abc"] & rep["same_xyz"]
    rep["same_as_frozen"] = rep["y5_cell"] == rep["frozen_cell"]
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
    matrix_train = load_frozen("matrix_train")
    targets_sha = load_manifest()["artifacts"]["targets"]["sha256"]
    keys = forecast_keys(cfg)
    record_declarations(keys, cfg)
    require_declarations(keys, utc_now(), cfg)
    agg = load_interim(cfg, cfg["split"]["test_days"][1])
    matrix_y5, stability_y5, rep = replicate_year5(
        targets, seg_params, matrix_train, agg, cfg, agg["week_of_day"]
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
    seg = _period_seg(cfg["segmentation"])
    matrix_5y = build_matrix(f"{names[0]}-{names[-1]}", agg, start, end, seg, wod)
    ym = year_matrices(agg, years, seg, wod)
    stab = cfg["stability"]
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
