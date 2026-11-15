"""Tratamiento por tienda del nb 03b (experimental).

Los quiebres de stock se detectan y se completan por tienda × artículo antes de sumar al
artículo: cuando el artículo se cae en casi todas las tiendas y una sigue vendiendo poco, la suma
del artículo nunca llega a cero y el quiebre queda escondido. Las reglas de bloque atípico
(4.2b, 4.4) y del ratio (6a) siguen a nivel artículo, sobre la suma ya corregida.
"""

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.features import (
    BLOCK_DAYS,
    aggregate_blocks,
    assign_windows,
    classify_extreme_ratios,
    drop_partial_first_block,
    fill_atypical_blocks,
    fill_invalid_blocks,
    impute_history_cases,
    replace_after_history,
)
from src.segmentation import longest_run_threshold

# Código de cada día-tienda en `flag_store_breaks`.
BREAK_KINDS = {0: "", 1: "run_history", 2: "run_causal", 3: "excess_zeros"}


def _days(units: pd.DataFrame) -> np.ndarray:
    return np.array([int(c[2:]) for c in units.columns])


def store_units(interim: dict, item_ids, start_d: int, end_d: int) -> pd.DataFrame:
    """Unidades diarias por tienda × artículo de los artículos pedidos.

    Parámetros:
        interim (dict): salida de `load_interim(..., series_level=True)`.
        item_ids (array-like): artículos a conservar.
        start_d (int): primer día (inclusivo).
        end_d (int): último día (inclusivo).

    Retorno:
        pd.DataFrame: float64, índice (`item_id`, `store_id`) ordenado y columnas `d_<día>`.
        NaN = sin dato (prelanzamiento, rechazo o semana sin precio en la tienda).
    """
    idx = interim["series_index"]
    keep = np.flatnonzero(idx["item_id"].isin(set(item_ids)).to_numpy())
    rows = idx.iloc[keep]
    index = pd.MultiIndex.from_arrays(
        [rows["item_id"].to_numpy(dtype=object), rows["store_id"].to_numpy(dtype=object)],
        names=["item_id", "store_id"],
    )
    out = pd.DataFrame(
        np.asarray(interim["units"][keep, start_d - 1 : end_d], dtype=float),
        index=index,
        columns=[f"d_{d}" for d in range(start_d, end_d + 1)],
    )
    return out.sort_index()


def store_zero_thresholds(units: pd.DataFrame, train_end: int, alpha: float, fallback: int) -> pd.DataFrame:
    """Umbral de racha de ceros (`long_run`) de cada tienda × artículo, estimado solo con train.

    Mismo método que el nb 03 (`zero_run_thresholds`) pero por tienda: n y p₀ se miden en los
    días observados de train entre la primera y la última venta de train de la tienda; el umbral
    es el menor k con P(racha más larga ≥ k) ≤ α. Si no se puede estimar se usa `fallback`.

    Parámetros:
        units (pd.DataFrame): salida de `store_units`.
        train_end (int): último día de train.
        alpha (float): `zero_alpha`.
        fallback (int): `long_run` de respaldo.

    Retorno:
        pd.DataFrame: mismo índice que `units` con `n`, `p0`, `threshold`, `source`
        (`estimated` o `fallback`) y `reason` (por qué se usó el respaldo).
    """
    train = _days(units) <= train_end
    rows = []
    for x in units.to_numpy():
        xt = x[train]
        xt = xt[~np.isnan(xt)]
        sales = np.flatnonzero(xt > 0)
        if len(sales) < 2:
            rows.append((0, np.nan, fallback, "fallback", "few_sales"))
            continue
        zero = xt[sales[0] : sales[-1] + 1] == 0
        n, p0 = len(zero), float(zero.mean())
        k = longest_run_threshold(p0, n, alpha)
        if k is None:
            reason = "no_zeros" if p0 == 0 else "all_zeros" if p0 == 1 else "no_threshold"
            rows.append((n, p0, fallback, "fallback", reason))
        else:
            rows.append((n, p0, k, "estimated", ""))
    out = pd.DataFrame(rows, index=units.index, columns=["n", "p0", "threshold", "source", "reason"])
    out["threshold"] = out["threshold"].astype("int64")
    return out


def flag_store_breaks(
    units: pd.DataFrame, thresholds: pd.DataFrame, window_start: pd.Series, alpha: float
) -> pd.DataFrame:
    """Marca los días-tienda en quiebre de stock, sin usar el futuro en ventana y test.

    Solo cuentan los ceros desde la primera venta de la tienda (antes no hay surtido). Reglas:
    - Racha (E1): una racha de ceros ≥ `threshold` de la tienda es quiebre. Si la tienda volvió a
      vender antes de la ventana del artículo, la racha entera se marca (`run_history`). Si no
      (la racha toca la ventana o el test, o termina sin venta), un día se marca solo cuando, al
      cierre de su bloque, la racha ya alcanzó el umbral (`run_causal`): no se sabe si la tienda
      vuelve.
    - Exceso de ceros (E2): un bloque con más ceros de lo esperable para la tienda (binomial con
      su p₀ de train, P ≤ α) marca sus ceros (`excess_zeros`). En la historia solo hasta la
      última venta anterior a la ventana.

    Parámetros:
        units (pd.DataFrame): salida de `store_units`, con días alineados a los bloques.
        thresholds (pd.DataFrame): salida de `store_zero_thresholds` (mismo índice).
        window_start (pd.Series): `item_id` → primer día de la ventana de backtesting.
        alpha (float): `zero_alpha` de la prueba binomial.

    Retorno:
        pd.DataFrame: int8 con el mismo índice y columnas que `units`; códigos de `BREAK_KINDS`
        (0 = sin quiebre).
    """
    if not thresholds.index.equals(units.index):
        raise ValueError("thresholds debe tener el mismo índice que units.")
    days = _days(units)
    n_days = len(days)
    pos = np.arange(n_days)
    block = pos // BLOCK_DAYS
    block_end = (block + 1) * BLOCK_DAYS - 1
    ws = window_start.reindex(units.index.get_level_values("item_id")).to_numpy(dtype=float)
    if np.isnan(ws).any():
        raise ValueError("hay artículos sin inicio de ventana.")
    ws_pos = ws - days[0]
    X = units.to_numpy()
    k_all = thresholds["threshold"].to_numpy()
    p0_all = thresholds["p0"].to_numpy()
    code = np.zeros(X.shape, dtype=np.int8)
    for r in range(len(X)):
        x = X[r]
        sold = np.flatnonzero(x > 0)
        if not len(sold):
            continue
        hist = pos < ws_pos[r]
        zero = x == 0
        zero[: sold[0]] = False
        edges = np.diff(np.r_[0, zero.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            seg = np.arange(s, e)
            if e < n_days and x[e] > 0 and hist[e]:
                if e - s >= k_all[r]:
                    code[r, seg] = 1
            else:
                seen = np.minimum(e - 1, block_end[seg]) - s + 1
                code[r, seg[seen >= k_all[r]]] = 2
        p0 = p0_all[r]
        if not 0 < p0 < 1:
            continue
        act = (pos >= sold[0]) & ~np.isnan(x)
        sold_hist = sold[hist[sold]]
        last_hist = sold_hist.max() if len(sold_hist) else -1
        act &= ~(hist & (pos > last_hist))
        for b in np.unique(block[act]):
            m = (block == b) & act & (code[r] == 0)
            n, z = int(m.sum()), int((zero & m).sum())
            if n and z and binom.sf(z - 1, n, p0) <= alpha:
                code[r, m & zero] = 3
    return pd.DataFrame(code, index=units.index, columns=units.columns)


def fill_store_breaks(units: pd.DataFrame, breaks: pd.DataFrame, rate_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Completa cada día en quiebre con la venta diaria media reciente de la propia tienda.

    La tasa es la media de los últimos `rate_days` días válidos (observados y sin quiebre)
    ANTERIORES al día: solo usa el pasado. Un día en quiebre sin `rate_days` días válidos previos
    no se completa.

    Parámetros:
        units (pd.DataFrame): salida de `store_units`.
        breaks (pd.DataFrame): salida de `flag_store_breaks`.
        rate_days (int): días válidos para la tasa.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]: unidades completadas y unidades agregadas (0 fuera de
        los quiebres; NaN en los días en quiebre que no se pudieron completar).
    """
    X = units.to_numpy()
    B = breaks.to_numpy() > 0
    added = np.zeros_like(X)
    for r in np.flatnonzero(B.any(axis=1)):
        valid = ~np.isnan(X[r]) & ~B[r]
        cnt = np.cumsum(valid)
        csum = np.cumsum(np.where(valid, X[r], 0.0))
        cnt_prev = np.r_[0, cnt[:-1]]
        csum_prev = np.r_[0.0, csum[:-1]]
        vpos = np.flatnonzero(valid)
        have = B[r] & (cnt_prev >= rate_days)
        j = cnt_prev[have] - rate_days  # días válidos que quedan antes de los últimos rate_days
        base = np.where(j > 0, csum[vpos[np.maximum(j - 1, 0)]], 0.0)
        added[r, have] = (csum_prev[have] - base) / rate_days
        added[r, B[r] & ~have] = np.nan
    filled = np.where(B, np.nan_to_num(X) + np.nan_to_num(added), X)
    filled = np.where(B & np.isnan(added), X, filled)
    as_frame = lambda a: pd.DataFrame(a, index=units.index, columns=units.columns)  # noqa: E731
    return as_frame(filled), as_frame(added)


def long_break_runs(breaks: pd.DataFrame, window_start: pd.Series, min_days: int,
                    chain_stores: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rachas de quiebre largas por tienda: generalizadas (caída de la cadena) o aisladas.

    Decisión del usuario (nb 03b, 5.3): una racha de quiebre de más de `min_days` días no se
    completa con la tasa congelada (inventaría meses de venta).
    - **Caída generalizada:** al menos `chain_stores` tiendas del artículo dentro de una racha
      larga a la vez, durante `min_days` días o más. La historia del artículo empieza en el primer
      bloque completo tras la vuelta a stock (fin del último episodio, `cut_d`); si no hay un
      bloque completo después, `no_return`. Es un filtro de alcance (historia suficiente), no una
      regla de pronóstico, por eso mira la serie completa.
    - **Racha aislada:** la que no cae en su mayoría (> 50 % de sus días) dentro de un episodio
      generalizado. La tienda se excluye desde `excl_start_d`: toda la racha si termina antes de la
      ventana (se sabe en la historia); si no, sin mirar el futuro, desde el bloque en cuyo cierre
      la racha ya superó `min_days` (los días anteriores se completan como siempre).

    Parámetros:
        breaks (pd.DataFrame): salida de `flag_store_breaks`.
        window_start (pd.Series): `item_id` → primer día de la ventana de backtesting.
        min_days (int): largo desde el cual una racha es larga (se cuenta `> min_days`).
        chain_stores (int): tiendas a la vez para una caída generalizada.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]:
        - rachas largas: `item_id`, `store_id`, `start_d`, `end_d` (último día), `days`,
          `generalized_share`, `kind` (`generalizada` / `aislada`) y `excl_start_d` (NaN si es
          generalizada);
        - cortes por artículo con caída generalizada: `item_id`, `episode_start_d`,
          `episode_end_d`, `cut_d` (NaN si `no_return`) y `no_return`.
    """
    days = _days(breaks)
    n = len(days)
    pos = np.arange(n)
    block_end = (pos // BLOCK_DAYS + 1) * BLOCK_DAYS - 1
    B = breaks.to_numpy() > 0
    items = breaks.index.get_level_values("item_id").to_numpy()
    stores = breaks.index.get_level_values("store_id").to_numpy()
    ws = window_start.reindex(items).to_numpy(dtype=float) - days[0]
    if np.isnan(ws).any():
        raise ValueError("hay artículos sin inicio de ventana.")

    def spans(mask, longer_than=None, at_least=None):
        edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
        out = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        return [(s, e) for s, e in out if (longer_than is None or e - s > longer_than) and (at_least is None or e - s >= at_least)]

    runs, in_long = [], np.zeros(B.shape, dtype=bool)
    for r in np.flatnonzero(B.any(axis=1)):
        for s, e in spans(B[r], longer_than=min_days):
            in_long[r, s:e] = True
            runs.append((r, s, e))
    in_gen = np.zeros(B.shape, dtype=bool)
    cuts = []
    for item, rows in pd.Series(np.arange(len(items))).groupby(items):
        rows = rows.to_numpy()
        episodes = spans(in_long[rows].sum(axis=0) >= chain_stores, at_least=min_days)
        if not episodes:
            continue
        for s, e in episodes:
            in_gen[rows, s:e] = True
        first_full = -(-episodes[-1][1] // BLOCK_DAYS) * BLOCK_DAYS  # primer bloque completo tras la vuelta
        no_return = first_full >= n
        cuts.append({"item_id": item, "episode_start_d": days[episodes[-1][0]], "episode_end_d": days[episodes[-1][1] - 1],
                     "cut_d": np.nan if no_return else days[first_full], "no_return": no_return})
    rows_out = []
    for r, s, e in runs:
        share = in_gen[r, s:e].mean()
        kind = "generalizada" if share > 0.5 else "aislada"
        if kind == "generalizada":
            excl = np.nan
        elif e <= ws[r]:
            excl = days[s]
        else:
            seen = np.minimum(e - 1, block_end[s:e]) - s + 1
            excl = days[s + int(np.argmax(seen > min_days))]
        rows_out.append({"item_id": items[r], "store_id": stores[r], "start_d": days[s], "end_d": days[e - 1],
                         "days": e - s, "generalized_share": share, "kind": kind, "excl_start_d": excl})
    runs_cols = ["item_id", "store_id", "start_d", "end_d", "days", "generalized_share", "kind", "excl_start_d"]
    cut_cols = ["item_id", "episode_start_d", "episode_end_d", "cut_d", "no_return"]
    return pd.DataFrame(rows_out, columns=runs_cols), pd.DataFrame(cuts, columns=cut_cols)


def exclude_long_runs(units: pd.DataFrame, breaks: pd.DataFrame, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Saca a la tienda de sus rachas aisladas largas, desde `excl_start_d` hasta el fin de la racha.

    Esos días quedan en NaN (como un día sin precio) y sin marca de quiebre: no se completan y no
    entran a la tasa de la tienda ni a su umbral. En la suma del artículo la tienda aporta lo que
    vendió, es decir 0.

    Parámetros:
        units (pd.DataFrame): salida de `store_units`.
        breaks (pd.DataFrame): salida de `flag_store_breaks` (mismo índice y columnas).
        runs (pd.DataFrame): primera salida de `long_break_runs`.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]: unidades y quiebres con los días excluidos.
    """
    if not (breaks.index.equals(units.index) and breaks.columns.equals(units.columns)):
        raise ValueError("breaks debe tener el mismo índice y columnas que units.")
    iso = runs[runs["kind"] == "aislada"]
    X, C = units.to_numpy(copy=True), breaks.to_numpy(copy=True)
    d0 = _days(units)[0]
    rows = units.index.get_indexer(pd.MultiIndex.from_frame(iso[["item_id", "store_id"]]))
    if (rows < 0).any():
        raise ValueError("hay rachas de series que no están en units.")
    for r, a, b in zip(rows, iso["excl_start_d"].to_numpy(dtype=int) - d0, iso["end_d"].to_numpy(dtype=int) - d0 + 1):
        X[r, a:b] = np.nan
        C[r, a:b] = 0
    return (pd.DataFrame(X, index=units.index, columns=units.columns),
            pd.DataFrame(C, index=breaks.index, columns=breaks.columns))


def cut_item_bounds(bounds: pd.DataFrame, cuts: pd.DataFrame) -> pd.DataFrame:
    """Mueve el inicio del artículo al primer bloque completo tras una caída generalizada.

    Parámetros:
        bounds (pd.DataFrame): `first_sale_d` y `last_sale_d` por `item_id`.
        cuts (pd.DataFrame): segunda salida de `long_break_runs`.

    Retorno:
        pd.DataFrame: copia de `bounds` con `first_sale_d` = máx(`first_sale_d`, `cut_d`); los
        artículos sin vuelta a stock (`no_return`) se quitan.
    """
    out = bounds.copy()
    ok = cuts[~cuts["no_return"]].set_index("item_id")["cut_d"]
    out.loc[ok.index, "first_sale_d"] = np.maximum(out.loc[ok.index, "first_sale_d"], ok)
    return out.drop(index=cuts.loc[cuts["no_return"], "item_id"])


def store_block_summary(units: pd.DataFrame, breaks: pd.DataFrame, added: pd.DataFrame) -> pd.DataFrame:
    """Resumen por tienda × artículo × bloque de 28 días (para revisar la validez del bloque de tienda).

    Retorno:
        pd.DataFrame: `item_id`, `store_id`, `block` (1, 2, …), `days_observed`, `break_days`,
        `unfilled_days`, `units_observed` y `units_added`.
    """
    n_rows, n_days = units.shape
    n_blocks = n_days // BLOCK_DAYS
    shape = (n_rows, n_blocks, BLOCK_DAYS)
    X = units.to_numpy().reshape(shape)
    B = breaks.to_numpy().reshape(shape) > 0
    A = added.to_numpy().reshape(shape)
    out = pd.DataFrame({
        "item_id": np.repeat(units.index.get_level_values("item_id").to_numpy(), n_blocks),
        "store_id": np.repeat(units.index.get_level_values("store_id").to_numpy(), n_blocks),
        "block": np.tile(np.arange(1, n_blocks + 1), n_rows),
        "days_observed": (~np.isnan(X)).sum(axis=2).ravel(),
        "break_days": B.sum(axis=2).ravel(),
        "unfilled_days": (B & np.isnan(A)).sum(axis=2).ravel(),
        "units_observed": np.nansum(X, axis=2).ravel(),
        "units_added": np.nansum(A, axis=2).ravel(),
    })
    return out


def add_store_fill(item_units: pd.DataFrame, added: pd.DataFrame, masked: pd.DataFrame | None = None) -> pd.DataFrame:
    """Suma al artículo las unidades completadas por tienda.

    En los días observados del artículo se suman a su venta (quiebres parciales, que la suma
    escondía). Si se pasa `masked`, los días que la máscara del artículo dejó en NaN por un quiebre
    total (todas las tiendas en cero) toman la suma de lo completado por tienda, siempre que alguna
    tienda tenga tasa (> 0); si no, siguen en NaN. Los demás NaN (fuera del ciclo de vida o sin dato
    de origen) no cambian.

    Parámetros:
        item_units (pd.DataFrame): unidades artículo × día (`units_treated` del nb 04).
        added (pd.DataFrame): segunda salida de `fill_store_breaks`.
        masked (pd.DataFrame | None): días del artículo enmascarados como quiebre (mismo índice y
            columnas que `item_units`).

    Retorno:
        pd.DataFrame: unidades del artículo con los quiebres por tienda completados.
    """
    if not added.columns.equals(item_units.columns):
        raise ValueError("los días de added e item_units no coinciden.")
    per_item = added.fillna(0.0).groupby(level="item_id").sum().reindex(item_units.index).fillna(0.0)
    out = item_units.where(item_units.isna(), item_units + per_item)
    if masked is not None:
        if not (masked.index.equals(item_units.index) and masked.columns.equals(item_units.columns)):
            raise ValueError("masked debe tener el mismo índice y columnas que item_units.")
        fill_total = masked.to_numpy(dtype=bool) & item_units.isna().to_numpy() & (per_item.to_numpy() > 0)
        out = out.mask(fill_total, per_item)
    return out


def item_block_pipeline(
    item_units: pd.DataFrame,
    chain_total: np.ndarray,
    bounds: pd.DataFrame,
    dates: pd.Series,
    targets: pd.DataFrame,
    fe: dict,
    min_blocks: int,
    test_blocks: int,
    val_blocks: int,
) -> dict:
    """Etapas 4.1 a 4.4 del nb 04 a nivel artículo, con las mismas funciones y parámetros.

    Se empaqueta porque el nb 03b la corre dos veces (línea base y con el tratamiento por tienda).

    Retorno:
        dict: `blocks` (4.1), `blocks_42a`, `blocks_43`, `cases`, `blocks_model` (4.4) y
        `excluded` (artículos con menos de `min_blocks` bloques).
    """
    blocks = aggregate_blocks(item_units, chain_total, bounds, dates)
    t = targets.set_index("item_id")
    valid = fill_invalid_blocks(blocks, t["min_days"], fe["low_block_share"], fe["low_block_window"])
    b42a = drop_partial_first_block(valid, min_blocks)
    base = fill_atypical_blocks(b42a, fe["atypical_level"], fe["low_block_window"], fe["atypical_min_ref"],
                                fe["atypical_years"])
    b43, excluded = assign_windows(base, test_blocks, val_blocks, min_blocks)
    cases = classify_extreme_ratios(b43, fe["atypical_level"], fe["case_low_level"], fe["case_high_level"],
                                    fe["case_masked_days"], fe["case_start_blocks"],
                                    window=fe["low_block_window"], min_ref=fe["atypical_min_ref"])
    causal = fill_atypical_blocks(b42a, fe["atypical_level"], fe["low_block_window"], fe["atypical_min_ref"],
                                  fe["atypical_years"], level_masked=fe["atypical_level_masked"],
                                  pattern_years=fe["atypical_pattern_years"], refill_level=fe["atypical_refill_level"])
    model = replace_after_history(impute_history_cases(b43, cases, fe["low_block_window"], fe["atypical_min_ref"]),
                                  causal)
    return {"blocks": blocks, "blocks_42a": b42a, "blocks_43": b43, "cases": cases,
            "blocks_model": model.sort_values(["item_id", "block"]).reset_index(drop=True), "excluded": excluded}


def extreme_blocks(blocks_model: pd.DataFrame, cases: pd.DataFrame, cell: pd.Series,
                   low: float = 0.5, high: float = 2.0) -> pd.DataFrame:
    """Bloques cuyo ratio sin tratar es extremo (< `low` o > `high`), con su tipo y tramo.

    El tipo sale de la clasificación de la 4.4 (`cases`): `estructural` y `arranque` se conservan;
    `atipico`, `propagado` y los extremos sin caso son `atipico` (los que siguen produciendo picos).

    Retorno:
        pd.DataFrame: `item_id`, `block`, `ratio`, `cell`, `kind` y `section` (historia, ventana o
        test).
    """
    k = ["item_id", "block"]
    b = blocks_model.sort_values(k).reset_index(drop=True)
    ratio = b["units"] / b.groupby("item_id")["units"].shift(1)
    ext = b.loc[(ratio > high) | (ratio < low), k + ["split", "backtest"]].assign(ratio=ratio)
    ext = ext.merge(cases[k + ["group"]].drop_duplicates(k), on=k, how="left")
    ext["cell"] = ext["item_id"].map(cell)
    ext["kind"] = np.where(ext["group"].isin(["estructural", "arranque"]), ext["group"], "atipico")
    ext["section"] = np.select([ext["split"] == "test", ext["backtest"]], ["test", "ventana"], "historia")
    return ext[k + ["ratio", "cell", "kind", "section"]]


def classify_level_changes(blocks_model: pd.DataFrame, extremes: pd.DataFrame, recover: float,
                           ref_blocks: int, smooth: int) -> pd.Series:
    """Diagnóstico (usa el futuro; no es una regla): separa los extremos estructurales en
    interrupción y cambio genuino.

    - Caída: es `interrupcion` si, más adelante, la mediana móvil centrada de `smooth` bloques
      vuelve a ≥ `recover` × la mediana de los `ref_blocks` bloques previos; si no, `genuino`.
    - Subida: es `regreso` si, antes de la caída previa, la mediana móvil ya estuvo en ≥ `recover`
      × la mediana de los `ref_blocks` bloques desde la subida; si no, `genuino`.

    Retorno:
        pd.Series: etiqueta por fila de `extremes` (solo filas con `kind == "estructural"`).
    """
    series = {i: g.sort_values("block") for i, g in blocks_model.groupby("item_id")}
    labels = {}
    for idx, row in extremes[extremes["kind"] == "estructural"].iterrows():
        g = series[row["item_id"]]
        x = g["units"].to_numpy(dtype=float)
        i = int(np.searchsorted(g["block"].to_numpy(), row["block"]))
        roll = pd.Series(x).rolling(smooth, center=True, min_periods=2).median().to_numpy()
        if row["ratio"] < 1:
            pre = np.median(x[max(0, i - ref_blocks) : i])
            labels[idx] = "interrupcion" if (roll[i + 1 :] >= recover * pre).any() else "genuino"
        else:
            post = np.median(x[i : i + ref_blocks])
            labels[idx] = "regreso" if (roll[: max(0, i - 1)] >= recover * post).any() else "genuino"
    return pd.Series(labels, dtype=object)


def share_deficit_test(units_sb: pd.DataFrame, days_sb: pd.DataFrame, window: int, min_ref: int,
                       alpha: float, max_level: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prueba de participación por tienda en cada bloque (idea de Friedman: las tiendas son los jueces).

    Por artículo y bloque, el esperado de cada tienda es su participación en las ventas del
    artículo en los `window` bloques anteriores (mínimo `min_ref`) × el total del bloque. Una
    prueba chi-cuadrado de bondad de ajuste dice si el reparto entre tiendas cambió (P ≤ α); el
    residuo estandarizado ajustado de cada tienda da su veredicto (por debajo si < −z₁₋α/₂). Una
    tienda por debajo y que vende menos de `max_level` × su esperado se completa con su esperado,
    calculado con el total que implican las demás tiendas. La participación se mide con lo
    observado, no con lo completado (si no, cada corrección agranda la participación de la tienda y
    se retroalimenta): solo usa el pasado y el propio bloque.
    Solo entran las tiendas con el bloque completo (28 días con dato) y participación previa > 0.

    Parámetros:
        units_sb (pd.DataFrame): unidades por tienda × bloque (índice `item_id`, `store_id`;
            columnas = bloques en orden), ya con los quiebres completados.
        days_sb (pd.DataFrame): días con dato por tienda × bloque (mismo índice y columnas).
        window (int): bloques previos para la participación.
        min_ref (int): mínimo de bloques previos.
        alpha (float): nivel de la prueba.
        max_level (float): la tienda se completa solo si vende < max_level × su esperado.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]: unidades por tienda × bloque corregidas y tabla de
        las tiendas-bloque completadas (`item_id`, `store_id`, `block`, `observed`, `expected`,
        `residual`, `p_value`).
    """
    from scipy.stats import chi2, norm

    if not (units_sb.index.equals(days_sb.index) and units_sb.columns.equals(days_sb.columns)):
        raise ValueError("units_sb y days_sb deben tener el mismo índice y columnas.")
    z = norm.ppf(1 - alpha / 2)
    out = units_sb.to_numpy(dtype=float).copy()
    full = days_sb.to_numpy() == BLOCK_DAYS
    items = units_sb.index.get_level_values("item_id").to_numpy()
    stores = units_sb.index.get_level_values("store_id").to_numpy()
    blocks = units_sb.columns.to_numpy()
    rows = []
    for item in pd.unique(items):
        r = np.flatnonzero(items == item)
        X = out[r]
        X_obs = X.copy()
        for j in range(min_ref, X.shape[1]):
            prev = X_obs[:, max(0, j - window):j]
            ok_prev = ~np.isnan(prev).all(axis=0)
            if ok_prev.sum() < min_ref:
                continue
            share = np.nansum(prev, axis=1)
            if share.sum() <= 0:
                continue
            share = share / share.sum()
            use = full[r, j] & (share > 0) & ~np.isnan(X_obs[:, j])
            if use.sum() < 2:
                continue
            p = share[use] / share[use].sum()
            obs = X_obs[use, j]
            total = obs.sum()
            if total <= 0:
                continue
            exp = p * total
            stat = float(((obs - exp) ** 2 / exp).sum())
            p_value = float(chi2.sf(stat, use.sum() - 1))
            if p_value > alpha:
                continue
            resid = (obs - exp) / np.sqrt(exp * (1 - p))
            low = (resid < -z) & (obs < max_level * exp)
            if not low.any() or low.all():
                continue
            implied_total = obs[~low].sum() / (1 - p[low].sum())   # total que implican las demás tiendas
            target = p * implied_total
            idx_use = np.flatnonzero(use)
            for k in np.flatnonzero(low):
                if target[k] > obs[k]:
                    rows.append((item, stores[r[idx_use[k]]], blocks[j], obs[k], target[k], resid[k], p_value))
                    X[idx_use[k], j] = target[k]
        out[r] = X
    table = pd.DataFrame(rows, columns=["item_id", "store_id", "block", "observed", "expected", "residual", "p_value"])
    return pd.DataFrame(out, index=units_sb.index, columns=units_sb.columns), table


def spread_block_extra(units: pd.DataFrame, extra_sb: pd.DataFrame) -> pd.DataFrame:
    """Reparte las unidades extra de cada tienda × bloque por igual entre los días del bloque.

    Parámetros:
        units (pd.DataFrame): unidades diarias por tienda (`store_units`), para la forma.
        extra_sb (pd.DataFrame): unidades extra por tienda × bloque (columnas = bloques 1, 2, …).

    Retorno:
        pd.DataFrame: unidades extra por tienda × día (mismo índice y columnas que `units`).
    """
    n_blocks = units.shape[1] // BLOCK_DAYS
    if extra_sb.shape != (len(units), n_blocks) or not extra_sb.index.equals(units.index):
        raise ValueError("extra_sb debe tener una columna por bloque y el índice de units.")
    daily = np.repeat(extra_sb.to_numpy(dtype=float) / BLOCK_DAYS, BLOCK_DAYS, axis=1)
    return pd.DataFrame(daily, index=units.index, columns=units.columns)
