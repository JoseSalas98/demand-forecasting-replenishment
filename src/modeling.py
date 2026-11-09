"""Modelado del nb 04: folds de backtesting, modelos y selección por artículo."""

import warnings

import numpy as np
import pandas as pd


def assign_folds(data: pd.DataFrame) -> pd.DataFrame:
    """Numera los folds de origen móvil dentro de la ventana de backtesting de cada artículo.

    Constitución 4.0.0 (Principio VI) y FR-024: el fold k predice el k-ésimo bloque de la ventana
    de cada artículo (h = 1) y entrena con todo lo anterior: los bloques previos a la ventana más
    los bloques 1 a k − 1 de la ventana (ventana expansiva). Los folds se definen por posición
    dentro de la ventana de cada artículo, no por fecha, porque la partición es por artículo.

    Parámetros:
        data (pd.DataFrame): filas por artículo y bloque con `item_id`, `block`, `split` y
            `backtest` (salida de las Etapas 4–9).

    Retorno:
        pd.DataFrame: copia ordenada por artículo y bloque con `fold` (Int64): 0 antes de la
        ventana, 1 a n dentro de la ventana y nulo en test.
    """
    out = data.sort_values(["item_id", "block"]).reset_index(drop=True)
    fold = pd.Series(pd.NA, index=out.index, dtype="Int64")
    fold[(out["split"] == "train") & ~out["backtest"]] = 0
    in_window = out["backtest"]
    fold[in_window] = out[in_window].groupby("item_id").cumcount() + 1
    out["fold"] = fold
    return out


def fold_masks(data: pd.DataFrame, k: int) -> tuple[pd.Series, pd.Series]:
    """Filas de entrenamiento y de validación del fold k.

    Parámetros:
        data (pd.DataFrame): salida de `assign_folds`.
        k (int): número de fold (1 a la cantidad de bloques de la ventana).

    Retorno:
        tuple[pd.Series, pd.Series]: máscara de entrenamiento (`fold` < k: todo lo anterior al
        bloque validado de cada artículo) y de validación (`fold` == k). El test nunca entra.
    """
    fold = data["fold"]
    if not 1 <= k <= int(fold.max()):
        raise ValueError(f"k debe estar entre 1 y {int(fold.max())}.")
    train = (fold < k).fillna(False).astype(bool)
    valid = (fold == k).fillna(False).astype(bool)
    return train, valid


def seasonal_naive(blocks: pd.DataFrame, m: int) -> pd.Series:
    """Pronóstico seasonal naive en unidades: ŷ_t = unidades del mismo artículo m bloques antes.

    Única línea base (constitución, FR-023), en unidades (decisión del usuario): el ratio se
    lleva a unidades para medir el error, así que la línea base se expresa igual. Con m = 13 es
    el mismo periodo del año anterior.

    Parámetros:
        blocks (pd.DataFrame): bloques contiguos por artículo con `item_id`, `block` y `units`
            (la serie completa, incluido el primer bloque).
        m (int): periodo estacional en bloques.

    Retorno:
        pd.Series: pronóstico alineado con `blocks` (NaN en los primeros m bloques de cada
        artículo).
    """
    order = blocks.sort_values(["item_id", "block"])
    gaps = order.groupby("item_id")["block"].diff().dropna()
    if not (gaps == 1).all():
        raise ValueError("los bloques de cada artículo deben ser contiguos.")
    return order.groupby("item_id", sort=False)["units"].shift(m).reindex(blocks.index)


# ---------------------------------------------------------------------------------------------
# SARIMAX: diferencias (KPSS y OCSB), búsqueda de órdenes y contraste entre periodos estacionales
# ---------------------------------------------------------------------------------------------


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """Desplaza `x` k posiciones hacia adelante (x_{t−k}); las primeras k quedan NaN."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def ocsb_crit(m: int) -> float:
    """Valor crítico al 5 % de la prueba OCSB para el periodo m.

    Es la aproximación por simulación que usan el paquete `forecast` de R (`ocsb.test`) y pmdarima
    (`OCSBTest`); solo existe para el 5 %.
    """
    z = np.log(m) - 0.7656451
    return float(-0.2937411 * np.exp(-0.2850853 * z - 0.05983644 * z**2) - 1.652202)


def ocsb_test(y: np.ndarray, m: int, max_lag: int = 3) -> dict | None:
    """Prueba OCSB (Osborn, Chui, Smith y Birchenhall, 1988) de raíz unitaria estacional.

    Regresión sin constante: Δ1Δm y_t = β1 Δm y_{t−1} + β2 Δ1 y_{t−m} + Σ φ_i Δ1Δm y_{t−i} + ε.
    H0: raíz unitaria estacional (β2 = 0). Si el estadístico t de β2 no baja del valor crítico,
    no se rechaza H0 y hace falta una diferencia estacional (D = 1). Los rezagos de Δ1Δm y
    (0 a `max_lag`) se eligen por AIC sobre la misma muestra, para que sean comparables. Las
    filas con NaN (ratios tratados) se dejan fuera.

    Parámetros:
        y (np.ndarray): serie, con NaN en los bloques tratados.
        m (int): periodo estacional en bloques.
        max_lag (int): máximo de rezagos de Δ1Δm y (3, el valor por defecto de pmdarima).

    Retorno:
        dict | None: `stat`, `crit`, `lag`, `rows` y `D`; None si quedan menos de 2 filas por
        coeficiente (muestra insuficiente para la regresión).
    """
    from statsmodels.regression.linear_model import OLS

    y = np.asarray(y, dtype=float)
    dm = y - _shift(y, m)
    dd = dm - _shift(dm, 1)
    X = np.column_stack([_shift(dm, 1), _shift(y - _shift(y, 1), m)] + [_shift(dd, i) for i in range(1, max_lag + 1)])
    ok = np.isfinite(dd) & np.isfinite(X).all(axis=1)
    if ok.sum() < 2 * X.shape[1]:
        return None
    fits = [OLS(dd[ok], X[ok, : 2 + lag]).fit() for lag in range(max_lag + 1)]
    lag = int(np.argmin([f.aic for f in fits]))
    stat, crit = float(fits[lag].tvalues[1]), ocsb_crit(m)
    return {"stat": stat, "crit": crit, "lag": lag, "rows": int(ok.sum()), "D": int(stat > crit)}


def kpss_d(y: np.ndarray, alpha: float = 0.05) -> dict:
    """Diferencia no estacional por KPSS (H0: estacionaria en nivel), como `ndiffs` de pmdarima.

    Se aplica a la serie sin NaN. d = 1 si se rechaza la estacionariedad (p < alpha). El máximo es
    1: el ratio ya es un cambio relativo, así que una segunda diferencia sería sobrediferenciar.
    statsmodels interpola el p-valor en una tabla (acotado entre 0,01 y 0,10); el aviso de que
    quedó en un extremo se silencia porque solo importa el lado de alpha.

    Retorno:
        dict: `kpss_stat`, `kpss_p` y `d`.
    """
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import kpss

    x = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p, _, _ = kpss(x, regression="c", nlags="auto", result_object=False)
    return {"kpss_stat": float(stat), "kpss_p": float(p), "d": int(p < alpha)}


def sarimax_differencing(y: np.ndarray, s_list: list[int], alpha: float = 0.05, max_lag: int = 3) -> pd.DataFrame:
    """Diferencias de cada periodo estacional candidato, en el orden de pmdarima: primero D, luego d.

    Para cada s: D por OCSB sobre la serie; d por KPSS sobre la serie ya diferenciada D veces con
    rezago s. La fila s = 0 es el modelo sin parte estacional (D = 0; d por KPSS sobre la serie).
    Un s es candidato solo si la muestra tiene al menos 2·s + 1 bloques (dos ciclos completos y
    uno más para la diferencia), y la regresión OCSB tiene muestra suficiente.

    Retorno:
        pd.DataFrame: una fila por s con `feasible`, `D`, `d`, el estadístico OCSB y el de KPSS.
    """
    y = np.asarray(y, dtype=float)
    rows = [{"s": 0, "feasible": True, "D": 0, **kpss_d(y, alpha)}]
    for s in s_list:
        oc = ocsb_test(y, s, max_lag) if len(y) >= 2 * s + 1 else None
        if oc is None:
            rows.append({"s": s, "feasible": False})
            continue
        z = y - _shift(y, s) if oc["D"] else y
        rows.append({"s": s, "feasible": True, "D": oc["D"], **kpss_d(z, alpha),
                     "ocsb_stat": oc["stat"], "ocsb_crit": oc["crit"], "ocsb_lag": oc["lag"]})
    out = pd.DataFrame(rows)
    return out.astype({"D": "Int64", "d": "Int64"})


def _n_params(order: tuple, seasonal_order: tuple, n_exog: int) -> int:
    """Parámetros de un SARIMAX: AR, MA, AR y MA estacionales, constante, exógenas y varianza."""
    p, d, q = order
    P, D, Q, _ = seasonal_order
    return p + q + P + Q + int(d + D == 0) + n_exog + 1


def fit_sarimax(y: np.ndarray, exog: np.ndarray, order: tuple, seasonal_order: tuple,
                start_params: np.ndarray | None = None, maxiter: int = 200) -> dict:
    """Ajusta un SARIMAX por máxima verosimilitud (filtro de Kalman, que admite NaN en `y`).

    - Constante solo sin diferencias (d + D = 0); con diferencias sería una deriva.
    - Exógenas: se dejan fuera las que no varían en la muestra (un conteo de eventos siempre 0
      no tiene coeficiente identificable).
    - Estacionariedad e invertibilidad forzadas (valor por defecto de statsmodels).
    - Muestra insuficiente si, descontadas las observaciones que consume la diferencia
      (d + s·D), no queda al menos una más que parámetros (AICc indefinido).

    Retorno:
        dict: `status` ("ok", "no_converge", "muestra_insuficiente" o "error: ..."), `res`
        (resultado de statsmodels o None), `exog_cols` (índices de las exógenas usadas), `k`,
        `n_eff`, `aic` y `aicc`.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.asarray(y, dtype=float)
    exog = np.asarray(exog, dtype=float)
    cols = [j for j in range(exog.shape[1]) if np.nanstd(exog[:, j]) > 0]
    k = _n_params(order, seasonal_order, len(cols))
    n_eff = int(np.isfinite(y[order[1] + seasonal_order[3] * seasonal_order[1]:]).sum())
    out = {"status": "muestra_insuficiente", "res": None, "exog_cols": cols, "k": k, "n_eff": n_eff,
           "aic": np.nan, "aicc": np.nan}
    if n_eff - k - 1 <= 0:
        return out
    trend = "c" if order[1] + seasonal_order[1] == 0 else "n"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y, exog=exog[:, cols] if cols else None, order=order, seasonal_order=seasonal_order, trend=trend)
            if start_params is not None and len(start_params) != model.k_params:
                start_params = None
            res = model.fit(start_params=start_params, disp=False, maxiter=maxiter)
    except Exception as exc:  # cualquier fallo de statsmodels queda registrado, no detiene la búsqueda
        out["status"] = f"error: {type(exc).__name__}: {exc}"[:200]
        return out
    ok = bool(res.mle_retvals.get("converged", False)) and np.isfinite(res.llf)
    aicc = -2 * res.llf + 2 * k + 2 * k * (k + 1) / (n_eff - k - 1)
    out.update(status="ok" if ok else "no_converge", res=res, aic=float(res.aic), aicc=float(aicc))
    return out


def sarimax_order_grid(y: np.ndarray, exog: np.ndarray, diffs: pd.DataFrame, p_list: list[int],
                       q_list: list[int], seasonal_pq: list[tuple[int, int]], maxiter: int = 200) -> pd.DataFrame:
    """Ajusta todas las combinaciones de órdenes de cada s factible (búsqueda completa, no paso a paso).

    s = 0: (p, d, q) sin parte estacional. s > 0: (p, d, q)(P, D, Q, s) con (P, Q) en
    `seasonal_pq`; si D = 1 se agrega (0, 0), que es solo la diferencia estacional (con D = 0
    repetiría el modelo de s = 0).

    Retorno:
        pd.DataFrame: una fila por combinación con `s`, `p`, `d`, `q`, `P`, `D`, `Q`, `k`,
        `n_eff`, `aic`, `aicc`, `status` y `params` (para arrancar los reajustes).
    """
    rows = []
    for t in diffs[diffs["feasible"]].itertuples():
        s, d, D = int(t.s), int(t.d), int(t.D)
        pq_seas = [(0, 0)] if s == 0 else list(seasonal_pq) + ([(0, 0)] if D == 1 else [])
        for p in p_list:
            for q in q_list:
                for P, Q in pq_seas:
                    order, sorder = (p, d, q), (P, D, Q, s)
                    fit = fit_sarimax(y, exog, order, sorder, maxiter=maxiter)
                    rows.append({"s": s, "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "k": fit["k"],
                                 "n_eff": fit["n_eff"], "aic": fit["aic"], "aicc": fit["aicc"], "status": fit["status"],
                                 "params": None if fit["res"] is None else np.asarray(fit["res"].params)})
    return pd.DataFrame(rows)


def sarimax_one_step(y: np.ndarray, exog: np.ndarray, units_prev: np.ndarray, order: tuple, seasonal_order: tuple,
                     start: int, start_params: np.ndarray | None = None, maxiter: int = 200) -> tuple[np.ndarray, int]:
    """Pronóstico a un paso en unidades desde `start` hasta el final, reestimando en cada origen.

    En el origen t se ajusta con y[:t] y exog[:t] (nunca con el bloque que se predice) y se
    pronostica el ratio de t con la exógena de t (conteos de eventos, conocidos de antemano).
    Unidades = unidades del bloque anterior × ratio pronosticado.
    Si el reajuste no converge, se filtra y[:t] con los últimos coeficientes que sí convergieron
    (sigue sin usar el bloque t) y se cuenta como respaldo; si no hay coeficientes compatibles,
    ese pronóstico queda NaN.

    Retorno:
        tuple[np.ndarray, int]: pronóstico en unidades de las posiciones `start` a len(y) − 1 y
        cantidad de orígenes resueltos con el respaldo.
    """
    out = np.full(len(y) - start, np.nan)
    last, n_backup = None if start_params is None else np.asarray(start_params), 0
    for i, t in enumerate(range(start, len(y))):
        fit = fit_sarimax(y[:t], exog[:t], order, seasonal_order, last, maxiter)
        res = fit["res"]
        if fit["status"] == "ok":
            last = np.asarray(res.params)
        elif res is not None and last is not None and len(last) == res.model.k_params:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = res.model.filter(last)
            n_backup += 1
        else:
            continue
        x_next = exog[t:t + 1, fit["exog_cols"]] if fit["exog_cols"] else None
        out[i] = units_prev[t] * float(np.asarray(res.forecast(1, exog=x_next))[0])
    return out, n_backup


def select_sarimax_item(item: pd.DataFrame, exog_cols: list[str], s_list: list[int], p_list: list[int],
                        q_list: list[int], seasonal_pq: list[tuple[int, int]], n_valid: int,
                        alpha: float = 0.05, max_lag: int = 3, maxiter: int = 200) -> dict:
    """Elige los órdenes SARIMAX de un artículo con los bloques anteriores a la ventana (fold 0).

    1. Muestra de selección: los bloques del fold 0 menos los últimos `n_valid`; contraste: esos
       `n_valid` bloques. La ventana de backtesting y el test no se usan.
    2. Con la muestra de selección: diferencias por s (OCSB y KPSS) y búsqueda completa de
       órdenes; dentro de cada s gana el menor AICc (el AIC solo se compara con la misma
       diferenciación, que es la de cada s).
    3. Entre los ganadores de cada s gana el menor RMSE en unidades del pronóstico a un paso en el
       contraste, reestimando en cada bloque (comparable aunque cambie la diferenciación). Si a un
       ganador le falla algún pronóstico, no compite.

    Parámetros:
        item (pd.DataFrame): filas de `sarimax_data` de un artículo, con `fold`.
        n_valid (int): bloques del contraste entre periodos estacionales.

    Retorno:
        dict: `diffs`, `grid` y `winners` (DataFrames, con `item_id`) y `choice` (dict con los
        órdenes elegidos o `status` = "sin_modelo").
    """
    item_id = str(item["item_id"].iloc[0])
    pre = item[item["fold"] == 0].sort_values("block")
    y = pre["ratio"].to_numpy(float)
    X = pre[exog_cols].to_numpy(float)
    n_sel = len(y) - n_valid
    diffs = sarimax_differencing(y[:n_sel], s_list, alpha, max_lag) if n_sel > 2 else pd.DataFrame(
        {"s": [0], "feasible": [False]})
    grid = sarimax_order_grid(y[:n_sel], X[:n_sel], diffs, p_list, q_list, seasonal_pq, maxiter)
    keys = ["s", "p", "d", "q", "P", "D", "Q"]
    if grid.empty or not (grid["status"] == "ok").any():
        winners = pd.DataFrame(columns=keys + ["aicc", "rmse_valid"])
    else:
        winners = grid[grid["status"] == "ok"].sort_values("aicc").groupby("s", sort=True).head(1).sort_values("s")
        rmse_valid, backups = [], []
        for w in winners.itertuples():
            f, n_backup = sarimax_one_step(y, X, pre["units_prev"].to_numpy(float), (w.p, w.d, w.q),
                                           (w.P, w.D, w.Q, w.s), n_sel, w.params, maxiter)
            err = pre["units"].to_numpy(float)[n_sel:] - f
            rmse_valid.append(float(np.sqrt(np.mean(err**2))) if np.isfinite(f).all() else np.nan)
            backups.append(n_backup)
        winners = winners.assign(rmse_valid=rmse_valid, backups=backups)[keys + ["aicc", "rmse_valid", "backups", "params"]]
    ranked = winners.dropna(subset=["rmse_valid"]).sort_values(["rmse_valid", "aicc"])
    if ranked.empty:
        choice = {"item_id": item_id, "status": "sin_modelo"}
    else:
        best = ranked.iloc[0]
        choice = {"item_id": item_id, "status": "ok", **{c: int(best[c]) for c in keys},
                  "aicc": float(best["aicc"]), "rmse_valid": float(best["rmse_valid"]), "params": best["params"]}
    choice.update(n_pre=len(y), n_sel=n_sel)
    return {"diffs": diffs.assign(item_id=item_id), "grid": grid.assign(item_id=item_id),
            "winners": winners.assign(item_id=item_id), "choice": choice}


def sarimax_backtest_item(item: pd.DataFrame, choice: dict, exog_cols: list[str], maxiter: int = 200) -> tuple[pd.DataFrame, int]:
    """Backtesting de 13 folds de un artículo con sus órdenes fijos (constitución: órdenes una vez,
    coeficientes reestimados en cada fold).

    El fold k ajusta con todo lo anterior a su bloque (fold 0 y bloques 1 a k − 1 de la ventana) y
    pronostica el bloque k (h = 1); en unidades = unidades del bloque anterior × ratio pronosticado.
    El test (fold nulo) no entra. Respaldo si un reajuste no converge, igual que en el contraste.

    Retorno:
        tuple[pd.DataFrame, int]: una fila por bloque de la ventana con `item_id`, `block`, `fold`,
        `units` y `sarimax` (NaN si el artículo quedó `sin_modelo`), y la cantidad de respaldos.
    """
    rows = item[item["fold"].notna()].sort_values("block")
    start = int((rows["fold"] == 0).sum())
    win = rows.iloc[start:]
    out = pd.DataFrame({"item_id": str(item["item_id"].iloc[0]), "block": win["block"].to_numpy(),
                        "fold": win["fold"].astype(int).to_numpy(), "units": win["units"].to_numpy(float),
                        "sarimax": np.nan})
    if choice.get("status") != "ok":
        return out, 0
    order = (choice["p"], choice["d"], choice["q"])
    sorder = (choice["P"], choice["D"], choice["Q"], choice["s"])
    f, n_backup = sarimax_one_step(rows["ratio"].to_numpy(float), rows[exog_cols].to_numpy(float),
                                   rows["units_prev"].to_numpy(float), order, sorder, start, choice["params"], maxiter)
    out["sarimax"] = f
    return out, n_backup


def search_sarimax_orders(data: pd.DataFrame, items: list[str], n_jobs: int = 1, backtest: bool = False,
                          verbose: int = 0, **kwargs) -> dict:
    """Corre `select_sarimax_item` en varios artículos, en paralelo y con un hilo por proceso.

    Con `backtest=True`, cada artículo hace además su backtesting de 13 folds con los órdenes
    elegidos (`sarimax_backtest_item`) en el mismo proceso. Para corridas largas, con guardado
    por lotes y avance, ver `search_sarimax_checkpointed`.

    Retorno:
        dict: `diffs`, `grid`, `winners` y `choice` (DataFrames de todos los artículos, `choice`
        con `seconds` por artículo) y, con `backtest`, `backtest` (pronóstico por artículo y fold).
    """
    from joblib import Parallel, delayed, parallel_config

    groups = [g for _, g in data[data["item_id"].astype(str).isin(items)].groupby("item_id", observed=True)]
    # Un hilo por proceso: joblib fija OMP/OpenBLAS/MKL antes de lanzar cada proceso (sin esto cada uno
    # abre varios hilos y los procesos se pisan).
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_search_item)(g, backtest, kwargs) for g in groups)
    return _combine_search(results, backtest)


def _search_item(df: pd.DataFrame, backtest: bool, kwargs: dict) -> dict:
    """Búsqueda de órdenes de un artículo y, si se pide, su backtesting; agrega `seconds` a `choice`."""
    import time

    t0 = time.perf_counter()
    out = select_sarimax_item(df, **kwargs)
    if backtest:
        out["backtest"], out["choice"]["backtest_backups"] = sarimax_backtest_item(
            df, out["choice"], kwargs["exog_cols"], kwargs.get("maxiter", 200))
    out["choice"]["seconds"] = time.perf_counter() - t0
    return out


def _combine_search(results: list[dict], backtest: bool) -> dict:
    """Junta los resultados de varios artículos en un DataFrame por tabla."""
    keys = ["diffs", "grid", "winners"] + (["backtest"] if backtest else [])
    out = {k: pd.concat([r[k] for r in results], ignore_index=True) for k in keys}
    out["choice"] = pd.DataFrame([r["choice"] for r in results])
    return out


def _read_batches(out_dir, ext: str) -> list[dict]:
    """Lee los lotes guardados (`batch_NNNN<ext>`) en orden de número de lote."""
    from pathlib import Path

    return [pd.read_pickle(f) for f in sorted(Path(out_dir).glob(f"batch_*{ext}"))]


def load_sarimax_checkpoints(out_dir, ext: str = ".pkl", items: list[str] | None = None) -> dict:
    """Lee y junta los lotes de `search_sarimax_checkpointed` sin volver a correr la búsqueda.

    Parámetros:
        out_dir: carpeta de los lotes.
        ext (str): extensión de los archivos; `pd.read_pickle` deduce la compresión de ella
            (".pkl" sin comprimir, ".pkl.gz" o ".pkl.xz" comprimidos).
        items (list[str] | None): si se da, solo esos artículos.

    Retorno:
        dict: las mismas tablas que `search_sarimax_orders` (`diffs`, `grid`, `winners`,
        `choice` y, si hubo backtesting, `backtest`) y `meta` (una fila por lote con `batch`,
        `n_items` y `saved_at`). Vacío si no hay lotes.
    """
    batches = _read_batches(out_dir, ext)
    if not batches:
        return {}
    keys = [k for k in batches[0] if k != "meta"]
    out = {k: pd.concat([b[k] for b in batches], ignore_index=True) for k in keys}
    if items is not None:
        keep = set(map(str, items))
        out = {k: v[v["item_id"].astype(str).isin(keep)].reset_index(drop=True) for k, v in out.items()}
    out["meta"] = pd.DataFrame([{"batch": b["meta"]["batch"], "n_items": len(b["meta"]["items"]),
                                 "saved_at": b["meta"]["saved_at"]} for b in batches])
    return out


def search_sarimax_checkpointed(data: pd.DataFrame, items: list[str], out_dir, batch_size: int = 10,
                                ext: str = ".pkl", n_jobs: int = 1, backtest: bool = True,
                                progress: bool = True, **kwargs) -> dict:
    """`search_sarimax_orders` con puntos de control: guarda cada `batch_size` artículos terminados
    y, al relanzarla, retoma donde quedó.

    - Un solo flujo paralelo (joblib, un hilo por proceso): cada proceso toma el siguiente artículo
      apenas termina el suyo, sin esperar al más lento de un lote.
    - Cada lote es un archivo `batch_NNNN<ext>` con un dict: `meta` (número de lote, artículos,
      fecha, configuración y versiones) y las tablas `diffs`, `grid`, `winners`, `choice` y
      `backtest` de sus artículos. Se escribe con un nombre temporal y luego se renombra, para no
      dejar un lote a medias si el proceso se corta.
    - Al retomar se saltan los artículos ya guardados. Si la configuración guardada no coincide con
      la actual, se detiene (no mezcla dos búsquedas); para empezar de cero se borra la carpeta.
      La configuración incluye el número de artículos pedidos (`n_items`): si cambia el conjunto
      de artículos (p. ej., otro tratamiento que deja otros artículos), también se detiene.
    - Si un artículo falla o se interrumpe la corrida, se guardan antes los terminados del lote.

    Parámetros:
        out_dir: carpeta de los lotes (se crea si no existe).
        batch_size (int): artículos por lote; es lo máximo que se pierde si algo falla.
        ext (str): extensión; la compresión se deduce de ella (".pkl", ".pkl.gz", ".pkl.xz").
        progress (bool): barra de avance (tqdm) sobre el total de artículos.
        kwargs: los de `select_sarimax_item`.

    Retorno:
        dict: igual que `load_sarimax_checkpoints` sobre `items`.
    """
    import os
    from pathlib import Path
    import platform

    from joblib import Parallel, delayed, parallel_config
    import statsmodels
    from tqdm.auto import tqdm

    if batch_size < 1:
        raise ValueError("batch_size debe ser ≥ 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"backtest": backtest, "n_items": len(set(map(str, items))), **kwargs}
    saved = _read_batches(out_dir, ext)
    for b in saved:
        if b["meta"]["config"] != config:
            raise ValueError(f"el lote {b['meta']['batch']} se guardó con otra configuración; "
                             f"para empezar de cero, borrar {out_dir}")
    done = {i for b in saved for i in b["meta"]["items"]}
    wanted = set(map(str, items))
    sub = data[data["item_id"].astype(str).isin(wanted - done)]
    groups = [g for _, g in sub.groupby("item_id", observed=True)]
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "statsmodels": statsmodels.__version__}
    state = {"batch": max((b["meta"]["batch"] for b in saved), default=-1) + 1, "buffer": []}

    def flush():
        if not state["buffer"]:
            return
        res = _combine_search(state["buffer"], backtest)
        res["meta"] = {"batch": state["batch"], "items": [str(r["choice"]["item_id"]) for r in state["buffer"]],
                       "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"), "config": config,
                       "versions": versions}
        final = out_dir / f"batch_{state['batch']:04d}{ext}"
        tmp = out_dir / f".tmp_{final.name}"   # termina en `ext`: la compresión se deduce igual
        pd.to_pickle(res, tmp)
        os.replace(tmp, final)
        state["batch"] += 1
        state["buffer"] = []

    n_done = len(done & wanted)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_search_item)(g, backtest, kwargs) for g in groups)
        try:
            for r in tqdm(gen, total=n_done + len(groups), initial=n_done, desc="SARIMAX", unit="art.",
                          disable=not progress):
                state["buffer"].append(r)
                if len(state["buffer"]) == batch_size:
                    flush()
        finally:
            flush()
    return load_sarimax_checkpoints(out_dir, ext, items=list(wanted))


def seasonal_acf(data: pd.DataFrame, lags: list[int], n_valid: int) -> pd.DataFrame:
    """Autocorrelación del ratio en los rezagos candidatos, por artículo, con su muestra de selección.

    Muestra: bloques del fold 0 menos los últimos `n_valid` (la misma de la búsqueda de órdenes).
    Significativa si |ACF| > 1,96 / √n (banda aproximada del 95 % bajo ruido blanco). Los NaN
    (ratios tratados) se saltan al calcular cada par (`missing="conservative"`).

    Retorno:
        pd.DataFrame: una fila por artículo con `n`, la ACF en cada rezago (`acf_<lag>`) y si es
        significativa (`sig_<lag>`); NaN si la muestra no alcanza el rezago.
    """
    from statsmodels.tsa.stattools import acf

    rows = []
    for item_id, g in data[data["fold"] == 0].groupby("item_id", observed=True):
        y = g.sort_values("block")["ratio"].to_numpy(float)[:-n_valid]
        n = int(np.isfinite(y).sum())
        row = {"item_id": str(item_id), "n": n}
        values = acf(y, nlags=max(lags), missing="conservative") if len(y) > max(lags) else None
        for lag in lags:
            ok = values is not None and len(y) > lag + 2
            row[f"acf_{lag}"] = float(values[lag]) if ok else np.nan
            row[f"sig_{lag}"] = bool(ok and abs(values[lag]) > 1.96 / np.sqrt(n))
        rows.append(row)
    return pd.DataFrame(rows)


# --- Boosting global: backtesting de 13 folds y búsqueda de hiperparámetros ---------------------

BOOSTING_ALGORITHMS = ("xgboost", "lightgbm", "catboost")


def make_booster(algorithm: str, params: dict, seed: int, n_jobs: int):
    """Crea el regresor de boosting con sus hiperparámetros, la semilla y los hilos.

    Parámetros:
        algorithm (str): uno de `BOOSTING_ALGORITHMS`.
        params (dict): hiperparámetros (fijos y los de la prueba).
        seed (int): semilla (R6).
        n_jobs (int): hilos del modelo.

    Retorno:
        el regresor sin entrenar.
    """
    if algorithm == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, random_state=seed, n_jobs=n_jobs)
    if algorithm == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, random_state=seed, n_jobs=n_jobs)
    if algorithm == "catboost":
        from catboost import CatBoostRegressor
        # `cat_features` va en `params`; sin archivos de registro en disco (catboost_info/).
        return CatBoostRegressor(**params, random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False)
    raise ValueError(f"algoritmo no soportado: {algorithm}")


def boosting_backtest(data: pd.DataFrame, features: list[str], algorithm: str, params: dict, seed: int,
                      n_jobs: int, n_folds: int = 13, target: str = "ratio") -> pd.DataFrame:
    """Backtesting de 13 folds de un modelo de boosting global (constitución, Principio VI).

    El fold k entrena un solo modelo con las filas de todos los artículos anteriores a su bloque
    k (`fold_masks`) y pronostica el bloque k de cada artículo (h = 1). Entrenan solo las filas
    con objetivo (el ratio tratado queda vacío); se pronostican y se miden todas las de la
    ventana. Pronóstico en unidades = unidades del bloque anterior × ratio pronosticado.

    Retorno:
        pd.DataFrame: una fila por artículo y fold con `item_id`, `block`, `fold`, `units`,
        `units_prev`, `ratio_hat` y `pred` (unidades).
    """
    X = data[features]
    out = []
    for k in range(1, n_folds + 1):
        train, valid = fold_masks(data, k)
        fit_rows = train & data[target].notna()
        model = make_booster(algorithm, params, seed, n_jobs)
        model.fit(X[fit_rows], data.loc[fit_rows, target])
        v = data.loc[valid, ["item_id", "block", "fold", "units", "units_prev"]].copy()
        v["item_id"] = v["item_id"].astype(str)
        v["fold"] = v["fold"].astype(int)
        v["ratio_hat"] = model.predict(X[valid])
        v["pred"] = v["units_prev"] * v["ratio_hat"]
        out.append(v)
    return pd.concat(out, ignore_index=True)


def suggest_params(trial, space: dict) -> dict:
    """Muestrea los hiperparámetros de una prueba de Optuna.

    `space`: nombre → ("int", bajo, alto), ("int", bajo, alto, paso), ("int_log", bajo, alto),
    ("float", bajo, alto) o ("float_log", bajo, alto) (escala logarítmica).
    """
    params = {}
    for name, spec in space.items():
        kind, low, high, *rest = spec
        if kind == "int":
            params[name] = trial.suggest_int(name, low, high, step=rest[0] if rest else 1)
        elif kind == "int_log":
            params[name] = trial.suggest_int(name, low, high, log=True)
        elif kind in ("float", "float_log"):
            params[name] = trial.suggest_float(name, low, high, log=kind == "float_log")
        else:
            raise ValueError(f"{name}: tipo desconocido {kind}")
    return params


def boosting_search(data: pd.DataFrame, features: list[str], algorithm: str, space: dict, fixed: dict,
                    scales: pd.Series, weights: pd.Series, n_trials: int, seed: int, n_jobs: int,
                    storage: str | None = None, study_name: str | None = None, n_folds: int = 13,
                    progress: bool = True):
    """Búsqueda de hiperparámetros con Optuna (TPE): cada prueba es un backtesting de 13 folds y
    su valor es el WRMSSE en la ventana (error promedio en los 13 folds, a minimizar).

    - Con `storage` (p. ej. "sqlite:///…/xgboost.db") cada prueba queda guardada al terminar; al
      relanzar se retoma y solo se corren las que faltan para llegar a `n_trials` completas.
    - Si el estudio guardado tiene otro espacio, otros fijos u otras variables, se detiene (no
      mezcla dos búsquedas); para empezar de cero se borra el archivo.
    - Cada prueba guarda en `user_attrs` el WRMSSE de cada fold, los segundos y cuántos
      pronósticos salieron negativos.
    - Semilla del sampler = `seed` + pruebas ya guardadas: reproducible si la corrida no se corta.

    Retorno:
        optuna.Study
    """
    import json
    import time

    import optuna

    from src.metrics import wrmsse_frame

    config = json.loads(json.dumps({"algorithm": algorithm, "space": space, "fixed": fixed,
                                    "features": features, "n_folds": n_folds}))
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction="minimize")
    # Semilla fija (R6) que avanza con las pruebas ya guardadas: al retomar no se repiten las
    # primeras pruebas aleatorias del TPE.
    study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials))
    if "config" in study.user_attrs and study.user_attrs["config"] != config:
        raise ValueError(f"el estudio {study_name} se guardó con otra configuración; para empezar de cero, "
                         f"borrar {storage}")
    study.set_user_attr("config", config)

    def objective(trial):
        t0 = time.perf_counter()
        bt = boosting_backtest(data, features, algorithm, {**fixed, **suggest_params(trial, space)}, seed, n_jobs,
                               n_folds)
        value, excluded = wrmsse_frame(bt, "pred", scales, weights)
        trial.set_user_attr("wrmsse_fold", [wrmsse_frame(g, "pred", scales, weights)[0] for _, g in bt.groupby("fold")])
        trial.set_user_attr("n_negative", int((bt["pred"] < 0).sum()))
        trial.set_user_attr("n_excluded", len(excluded))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return value

    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if n_trials > done:
        study.optimize(objective, n_trials=n_trials - done, show_progress_bar=progress)
    return study


def window_forecasts(base: pd.DataFrame, forecasts: dict) -> pd.DataFrame:
    """Junta en una tabla los pronósticos de la ventana de backtesting de varios candidatos.

    Parámetros:
        base (pd.DataFrame): una fila por artículo y bloque de la ventana, con `item_id`, `block`,
            `fold` y `units` (p. ej. `snaive_valid` del nb 03b, que trae el seasonal naive).
        forecasts (dict): nombre del candidato → tabla con `item_id`, `block`, `units` y una
            columna con ese nombre (pronóstico en unidades; NaN si el artículo no tiene modelo).

    Retorno:
        pd.DataFrame: `base` con una columna por candidato, mismas filas y orden.

    Raises:
        ValueError: si un candidato tiene filas repetidas, filas fuera de `base`, le faltan filas
            o sus unidades reales no coinciden con las de `base`.
    """
    out = base.assign(item_id=base["item_id"].astype(str)).reset_index(drop=True)
    for name, df in forecasts.items():
        f = df[["item_id", "block", "units", name]].assign(item_id=df["item_id"].astype(str))
        if f.duplicated(["item_id", "block"]).any():
            raise ValueError(f"{name}: filas repetidas por artículo y bloque")
        merged = out[["item_id", "block", "units"]].merge(f, on=["item_id", "block"], how="left",
                                                          suffixes=("", "_cand"), indicator=True)
        if (merged["_merge"] != "both").any() or len(f) != len(out):
            raise ValueError(f"{name}: no tiene exactamente las filas de la ventana")
        if not np.allclose(merged["units"], merged["units_cand"], equal_nan=True):
            raise ValueError(f"{name}: las unidades reales no coinciden con las de la base")
        out[name] = merged[name].to_numpy(float)
    return out


def select_by_item(errors: pd.DataFrame, order: list[str], metric: str = "rmsse") -> pd.DataFrame:
    """Elige el modelo de cada artículo con el filtro de la constitución (Principio VII).

    `order[0]` es la línea base. Un candidato compite solo si su error es estrictamente menor que
    el de la línea base; entre los que pasan gana el de menor error, y si ninguno pasa gana la
    línea base. `order` va del más simple al más complejo: con empate exacto gana el más simple.

    Parámetros:
        errors (pd.DataFrame): una fila por artículo y candidato con `item_id`, `candidate` y
            `metric` (p. ej. `item_errors`); un candidato sin fila no compite en ese artículo.
        order (list[str]): candidatos, la línea base primero.
        metric (str): columna con la que se elige.

    Retorno:
        pd.DataFrame: una fila por artículo con el error de cada candidato (NaN si no tiene),
        `winner` y `winner_<metric>`.

    Raises:
        ValueError: si algún artículo no tiene error de la línea base.
    """
    wide = errors.pivot(index="item_id", columns="candidate", values=metric).reindex(columns=order)
    if wide[order[0]].isna().any():
        raise ValueError(f"hay artículos sin error de la línea base {order[0]}")
    winner = pd.Series(order[0], index=wide.index)
    best = wide[order[0]].copy()
    for cand in order[1:]:
        better = wide[cand] < best   # estricto: empate → se queda el anterior (más simple); NaN no compite
        winner[better], best[better] = cand, wide.loc[better, cand]
    return wide.assign(winner=winner, **{f"winner_{metric}": best}).reset_index().rename_axis(columns=None)


def selected_forecast(window: pd.DataFrame, winner: pd.Series) -> np.ndarray:
    """Pronóstico del modelo elegido en cada fila de la ventana.

    Parámetros:
        window (pd.DataFrame): con `item_id` y una columna por candidato.
        winner (pd.Series): artículo → nombre del candidato elegido.

    Retorno:
        np.ndarray: pronóstico de la columna del ganador de cada artículo, fila a fila.

    Raises:
        ValueError: si a un artículo le falta ganador o el ganador no tiene pronóstico en alguna fila.
    """
    chosen = window["item_id"].astype(str).map(winner)
    if chosen.isna().any():
        raise ValueError("hay artículos de la ventana sin ganador")
    out = np.full(len(window), np.nan)
    for cand in chosen.unique():
        rows = (chosen == cand).to_numpy()
        out[rows] = window.loc[rows, cand].to_numpy(float)
    if np.isnan(out).any():
        raise ValueError("el modelo elegido no tiene pronóstico en alguna fila")
    return out
