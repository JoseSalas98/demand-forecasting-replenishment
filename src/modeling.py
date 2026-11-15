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


def assign_test_folds(data: pd.DataFrame) -> pd.DataFrame:
    """Numera los orígenes del test como folds, para medir el test con el régimen del backtesting.

    Constitución, Principio VI: en test se repite el backtesting. Todo el train (ventana incluida)
    pasa a fold 0 y el k-ésimo bloque de test de cada artículo a fold k, por posición (la partición
    es por artículo). Así `fold_masks`, `boosting_backtest` y `recursive_ratio_forecast` sirven
    igual que en la ventana.

    Parámetros:
        data (pd.DataFrame): dataset con `item_id`, `block` y `split` (train / test).

    Retorno:
        pd.DataFrame: copia ordenada por artículo y bloque con `fold` (Int64) reemplazado.
    """
    out = data.sort_values(["item_id", "block"]).reset_index(drop=True)
    is_test = out["split"] == "test"
    last_train = out[~is_test].groupby("item_id", observed=True)["block"].max()
    first_test = out[is_test].groupby("item_id", observed=True)["block"].min()
    if not (first_test == last_train.reindex(first_test.index) + 1).all():
        raise ValueError("el test de cada artículo debe empezar justo después de su train")
    fold = pd.Series(0, index=out.index, dtype="Int64")
    fold[is_test] = out[is_test].groupby("item_id", observed=True).cumcount() + 1
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


def recursive_ratio_forecast(data: pd.DataFrame, model, features: list[str], lags: list[int], windows: list[int],
                             stats: list[str], first_fold: int = 1, horizon: int = 13) -> pd.DataFrame:
    """Pronóstico encadenado de la tasa (ratio) de `horizon` bloques desde un mismo origen.

    El origen es el último bloque antes del fold `first_fold` de cada artículo. Para el paso h
    los rezagos y ventanas del ratio (`ratio_lag*`, `ratio_roll*`) se recalculan con la historia
    real hasta el origen más las tasas pronosticadas en los pasos 1 a h − 1, con las mismas
    funciones del dataset (`add_lags`, `add_rolling`). Las demás variables (calendario, edad,
    `item_id`, `cell`) se conocen de antemano y se toman del dataset. En cada paso se pronostican
    todos los artículos a la vez.

    Parámetros:
        data (pd.DataFrame): dataset del boosting con `fold`, `ratio` y `ratio_treated`.
        model: modelo ya entrenado (solo con lo anterior al origen).
        features (list[str]): variables del modelo, en su orden.
        lags (list[int]): rezagos del ratio (`target_lags`).
        windows (list[int]): ventanas móviles (`rolling_windows`).
        stats (list[str]): estadísticos móviles (`rolling_stats`).
        first_fold (int): primer bloque pronosticado (1 = inicio de la ventana).
        horizon (int): bloques hacia adelante.

    Retorno:
        pd.DataFrame: una fila por artículo y paso con `item_id`, `block`, `fold`, `step` y
        `ratio_hat` (tasa pronosticada).
    """
    from src.features import add_lags, add_rolling

    ratio_cols = [c for c in features if c.startswith("ratio_lag") or c.startswith("ratio_roll")]
    fold = data["fold"]
    hist = data.loc[fold.notna() & (fold < first_fold), ["item_id", "block", "ratio", "ratio_treated"]]
    future = data[fold.between(first_fold, first_fold + horizon - 1)].sort_values(["item_id", "block"])
    future = future.assign(step=(future["fold"] - first_fold + 1).astype(int))
    if not (future.groupby("item_id", observed=True)["step"].count() == horizon).all():
        raise ValueError(f"cada artículo necesita {horizon} bloques después del origen")
    path = future[["item_id", "block", "ratio_treated"]].assign(ratio=np.nan, ratio_treated=False)
    out = []
    for h in range(1, horizon + 1):
        frame = pd.concat([hist, path], ignore_index=True)
        frame = add_lags(frame, ["ratio"], lags)
        frame = add_rolling(frame, "ratio", windows, stats, treated_col="ratio_treated")
        rows = future[future["step"] == h]
        X = rows.reindex(columns=features).copy()   # las columnas del ratio se recalculan abajo
        recalc = rows[["item_id", "block"]].merge(frame[["item_id", "block"] + ratio_cols], on=["item_id", "block"],
                                                  how="left", validate="one_to_one")
        X[ratio_cols] = recalc[ratio_cols].to_numpy()
        ratio_hat = model.predict(X)
        step = rows[["item_id", "block", "fold", "step"]].assign(ratio_hat=ratio_hat)
        out.append(step)
        # La tasa pronosticada pasa a ser historia del paso siguiente.
        key = path[["item_id", "block"]].merge(step[["item_id", "block", "ratio_hat"]], on=["item_id", "block"], how="left")
        path["ratio"] = path["ratio"].where(key["ratio_hat"].isna().to_numpy(), key["ratio_hat"].to_numpy())
    res = pd.concat(out, ignore_index=True)
    res["item_id"] = res["item_id"].astype(str)
    res["fold"] = res["fold"].astype(int)
    return res


def chain_units(path: pd.DataFrame, origin_units: pd.Series, ratio_col: str = "ratio_hat") -> np.ndarray:
    """Unidades encadenadas desde el origen: Û(t+h) = U(t) × r̂(t+1) × … × r̂(t+h).

    Cada bloque toma como base las unidades pronosticadas del bloque anterior (en el paso 1, las
    unidades reales del origen), igual que en producción.

    Parámetros:
        path (pd.DataFrame): una fila por artículo y paso con `item_id`, `step` y `ratio_col`.
        origin_units (pd.Series): artículo → unidades reales del bloque de origen.
        ratio_col (str): columna con la tasa pronosticada.

    Retorno:
        np.ndarray: unidades pronosticadas de cada fila de `path`, en su mismo orden.

    Raises:
        ValueError: si falta el origen de algún artículo o un artículo tiene pasos repetidos.
    """
    if path.duplicated(["item_id", "step"]).any():
        raise ValueError("pasos repetidos por artículo")
    order = path.sort_values(["item_id", "step"])
    base = order["item_id"].astype(str).map(origin_units)
    if base.isna().any():
        raise ValueError("falta el origen de algún artículo")
    growth = order[ratio_col].groupby(order["item_id"].astype(str), sort=False).cumprod()
    return (base * growth).reindex(path.index).to_numpy(float)


def plot_forecast_item(data: pd.DataFrame, table: pd.DataFrame, item_id: str, rate_scale: float,
                       unit_scale: float, splits: tuple = ("train",), window_label: str = "ventana de backtesting"):
    """Gráfico de un artículo: pronóstico a un paso y encadenado, en tasas y en unidades, con su error.

    Fila de arriba, tasas: tasa real (`ratio_raw`) de todo el train, la ventana sombreada, la tasa a
    un paso (`ratio_1step`) y la encadenada (`ratio_chain`). Fila de abajo, unidades: unidades reales
    (círculo hueco = bloque completado por el tratamiento), las de un paso (`units_1step`) y las
    encadenadas (`units_chain`). En cada leyenda, el WAPE y el RMSSE de cada pronóstico en la
    ventana. A la derecha, el error de cada bloque de la ventana (pronóstico − real): en tasa, y en
    unidades como % de lo real. Solo se dibujan las particiones de `splits` (por omisión, sin test).

    Parámetros:
        data (pd.DataFrame): con `item_id`, `block`, `start_date`, `split`, `units`, `ratio_raw`,
            `imputed` y `cell`.
        table (pd.DataFrame): una fila por artículo y bloque de la ventana con `item_id`, `block`,
            `step` y las columnas `ratio_1step`, `ratio_chain`, `units_1step` y `units_chain`.
        item_id (str): artículo.
        rate_scale (float): escala del RMSSE en tasas.
        unit_scale (float): escala del RMSSE en unidades.
        splits (tuple): particiones que se dibujan; ("train", "test") para pronósticos del test.
        window_label (str): nombre de la ventana pronosticada, para el título.

    Retorno:
        matplotlib.figure.Figure: la figura.
    """
    import matplotlib.pyplot as plt

    from src.metrics import rmsse, wape

    rows = data[(data["item_id"].astype(str) == item_id) & data["split"].isin(splits)].sort_values("block")
    win = table[table["item_id"].astype(str) == item_id].drop(columns=["start_date", "units", "ratio_raw"], errors="ignore")
    win = win.merge(rows[["block", "start_date", "units", "ratio_raw"]], on="block").sort_values("block")
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [3, 1]})
    styles = {"1step": dict(color="tab:orange", ls="--", label="a un paso"), "chain": dict(color="tab:blue", ls="-", label="encadenado")}
    levels = [("ratio", "ratio_raw", rate_scale, "tasa (unidades / bloque anterior)", "error de la tasa"),
              ("units", "units", unit_scale, "unidades por bloque", "error en unidades (% de lo real)")]
    for (prefix, real_col, scale, ylabel, err_title), (ax, ax_err) in zip(levels, axes):
        ax.plot(rows["start_date"], rows[real_col], color="0.35", lw=1.2, label="real")
        if prefix == "units":
            imputed = rows["imputed"].astype(bool).to_numpy()
            ax.scatter(rows["start_date"][imputed], rows[real_col][imputed], facecolors="none", edgecolors="0.35",
                       s=22, zorder=3, label="completado")
        else:
            ax.axhline(1, color="0.6", lw=0.8, ls=":")
        ax.axvspan(win["start_date"].min(), win["start_date"].max(), color="0.5", alpha=0.08, lw=0)
        width = 0.38
        for k, (kind, st) in enumerate(styles.items()):
            pred = win[f"{prefix}_{kind}"]
            w, r = wape(win[real_col], pred), rmsse(win[real_col], pred, scale)
            ax.plot(win["start_date"], pred, color=st["color"], ls=st["ls"], lw=1.5, marker="o", ms=3,
                    label=f"{st['label']} (WAPE {w:.1%}, RMSSE {r:.2f})")
            err = pred - win[real_col]
            if prefix == "units":
                err = 100 * err / win[real_col]
            ax_err.bar(win["step"] + (k - 0.5) * width, err, width=width, color=st["color"], label=st["label"])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax_err.axhline(0, color="0.3", lw=0.8)
        ax_err.set_title(err_title, fontsize=9, loc="left")
        ax_err.set_xticks(win["step"])
        ax_err.tick_params(axis="x", labelsize=7)
    axes[0, 0].set_title(f"{item_id} · {rows['cell'].iloc[0]} · {window_label}: a un paso vs encadenado desde su inicio",
                         fontsize=10, loc="left")
    axes[0, 1].legend(fontsize=7, frameon=False)
    axes[1, 1].set_xlabel("paso (bloque de la ventana)", fontsize=8)
    return fig



# --- Modelo final: importancia de las variables con SHAP ------------------------------------------

def catboost_shap(model, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Valores SHAP de un CatBoost ya entrenado, con su cálculo nativo (sin la librería `shap`).

    Parámetros:
        model: `CatBoostRegressor` entrenado.
        X (pd.DataFrame): filas a explicar, con las mismas columnas del ajuste.

    Retorno:
        tuple: (`pd.DataFrame` con un valor SHAP por fila y variable, mismo índice y columnas que `X`;
        valor esperado del modelo). Por fila, valor esperado + suma de los SHAP = predicción.
    """
    from catboost import Pool

    pool = Pool(X, cat_features=model.get_cat_feature_indices())
    raw = model.get_feature_importance(pool, type="ShapValues")   # última columna: valor esperado
    return pd.DataFrame(raw[:, :-1], index=X.index, columns=X.columns), float(raw[0, -1])


def shap_importance(shap: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Importancia de cada variable: media de |SHAP| en todas las filas y en cada grupo.

    Parámetros:
        shap (pd.DataFrame): valores SHAP por fila y variable (`catboost_shap`).
        groups (pd.Series): grupo de cada fila (p. ej., el segmento), mismo índice que `shap`.

    Retorno:
        pd.DataFrame: una fila por variable, ordenada de mayor a menor importancia total, con
        `total` y una columna por grupo (en el orden de los grupos).
    """
    if not groups.index.equals(shap.index):
        raise ValueError("groups debe tener el mismo índice que shap")
    absolute = shap.abs()
    out = absolute.groupby(groups.astype(str).to_numpy()).mean().T
    out.insert(0, "total", absolute.mean())
    return out.sort_values("total", ascending=False).rename_axis("variable")


def plot_shap_importance(importance: pd.DataFrame, title: str = ""):
    """Importancia SHAP: barras con la total y mapa de calor por grupo.

    Izquierda, media de |SHAP| de cada variable en todas las filas. Derecha, para cada grupo (columna),
    la parte de su importancia que aporta cada variable (% de la suma de su columna), para comparar
    qué variables pesan más en cada grupo aunque la escala de los SHAP difiera entre grupos; en la
    cabecera, la suma de |SHAP| medio del grupo.

    Parámetros:
        importance (pd.DataFrame): salida de `shap_importance`.
        title (str): título de la figura.

    Retorno:
        matplotlib.figure.Figure: la figura.
    """
    import matplotlib.pyplot as plt

    groups = [c for c in importance.columns if c != "total"]
    share = 100 * importance[groups] / importance[groups].sum()
    n = len(importance)
    fig, (ax_bar, ax_map) = plt.subplots(1, 2, figsize=(13, 0.28 * n + 1.6), sharey=True,
                                         gridspec_kw={"width_ratios": [2, 1.2]})
    y = np.arange(n)
    ax_bar.barh(y, importance["total"], color="tab:blue")
    ax_bar.set_yticks(y, importance.index, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("media de |SHAP| (tasa)")
    ax_bar.set_title("importancia total", fontsize=10, loc="left")
    image = ax_map.imshow(share.to_numpy(), aspect="auto", cmap="Blues")
    for i in range(n):
        for j in range(len(groups)):
            v = share.iat[i, j]
            ax_map.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v > share.to_numpy().max() * 0.6 else "black")
    ax_map.set_xticks(range(len(groups)), [f"{g}\nΣ {importance[g].sum():.3f}" for g in groups], fontsize=8)
    ax_map.xaxis.tick_top()
    ax_map.set_title("% de la importancia de cada segmento", fontsize=10, loc="left", pad=28)
    fig.colorbar(image, ax=ax_map, fraction=0.05, pad=0.02, label="%")
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


# --- Modelo final: guardado y carga ---------------------------------------------------------------

MODEL_FILE = "catboost_global.cbm"
META_FILE = "meta.json"


def save_catboost_model(model, meta: dict, out_dir, overwrite: bool = False) -> list:
    """Guarda un CatBoost en formato nativo (`.cbm`) con su `meta.json`, sin reemplazar otro distinto.

    Si ya hay un modelo en `out_dir`: igual (mismo modelo según CatBoost y mismo `meta`, sin contar
    `saved_at`) no se escribe nada; distinto, se detiene salvo `overwrite=True`. Los bytes del
    `.cbm` no sirven para comparar: dos guardados del mismo modelo difieren.

    Parámetros:
        model: `CatBoostRegressor` entrenado.
        meta (dict): datos del modelo, serializables en JSON; se agrega `saved_at`.
        out_dir: carpeta de destino (se crea si no existe).
        overwrite (bool): reemplazar aunque lo guardado sea distinto.

    Retorno:
        list[Path]: rutas del modelo y del `meta.json`.

    Raises:
        FileExistsError: si lo guardado es distinto y `overwrite` es False.
    """
    import json
    from pathlib import Path

    from src.data import _same_content

    out_dir = Path(out_dir)
    paths = [out_dir / MODEL_FILE, out_dir / META_FILE]
    meta = json.loads(json.dumps(meta))   # falla aquí, antes de escribir, si algo no es JSON
    if paths[0].exists() and not overwrite:
        old_model, old_meta = load_catboost_model(out_dir)
        if old_model == model and _same_content(old_meta, meta):
            return paths
        raise FileExistsError(f"{out_dir} ya tiene otro modelo; para reemplazarlo usa overwrite=True.")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(paths[0]))
    meta = {"saved_at": pd.Timestamp.now().isoformat(timespec="seconds"), **meta}
    paths[1].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def load_catboost_model(out_dir):
    """Lee el modelo y el `meta.json` que escribió `save_catboost_model`.

    Retorno:
        tuple: (`CatBoostRegressor`, dict con el meta).
    """
    import json
    from pathlib import Path

    from catboost import CatBoostRegressor

    out_dir = Path(out_dir)
    model = CatBoostRegressor()
    model.load_model(str(out_dir / MODEL_FILE))
    return model, json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))


# --- Impacto en el negocio: curvas de intercambio -------------------------------------------------

FORECAST_STYLES = {"a un paso": dict(color="tab:orange", ls="--"), "encadenado": dict(color="tab:blue", ls="-"),
                   "seasonal naive": dict(color="0.45", ls=":")}


def plot_tradeoff_curves(curves: pd.DataFrame, forecasts: list, segments: list, marks: pd.DataFrame | None = None,
                         title: str = ""):
    """Curvas de intercambio: exceso (eje X) frente a venta perdida (eje Y), un panel por segmento.

    Cada pronóstico es una curva (un punto por z); más abajo y a la izquierda es mejor. El
    pronóstico perfecto es el origen (no pierde ni sobra). Se marca z = 0 con un círculo y, si se
    pasan, los puntos de `marks` (p. ej. el 95 % de servicio) con una estrella. El eje vertical se
    recorta a 1,15 veces la mayor venta perdida entre z = 0 y las estrellas: los z negativos llevan
    la venta perdida a toda la demanda y aplastarían la zona que interesa. Montos en millones.

    Parámetros:
        curves (pd.DataFrame): `segment`, `forecast`, `z`, `lost` y `excess`.
        forecasts (list): pronósticos que se dibujan, en orden (estilos en `FORECAST_STYLES`).
        segments (list): un panel por segmento.
        marks (pd.DataFrame | None): `segment`, `forecast`, `lost` y `excess` de los puntos a marcar.
        title (str): título de la figura.

    Retorno:
        matplotlib.figure.Figure: la figura.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    millions = FuncFormatter(lambda v, _: f"{v / 1e6:,.1f}")
    fig, axes = plt.subplots(1, len(segments), figsize=(4.4 * len(segments), 4.2))
    axes = np.atleast_1d(axes)
    for ax, seg in zip(axes, segments):
        top = 0.0
        for name in forecasts:
            c = curves[(curves["segment"] == seg) & (curves["forecast"] == name)].sort_values("z")
            st = FORECAST_STYLES.get(name, {})
            ax.plot(c["excess"], c["lost"], lw=1.8, label=name, **st)
            z0 = c[np.isclose(c["z"], 0)]
            ax.scatter(z0["excess"], z0["lost"], s=30, facecolors="white", edgecolors=st.get("color", "k"), zorder=3)
            top = max(top, float(z0["lost"].max()))
            if marks is not None:
                m = marks[(marks["segment"] == seg) & (marks["forecast"] == name)]
                ax.scatter(m["excess"], m["lost"], marker="*", s=120, color=st.get("color", "k"), zorder=4)
                top = max(top, float(m["lost"].max()) if len(m) else 0.0)
        ax.scatter([0], [0], color="black", s=24, zorder=4, label="perfecto")
        ax.set_title(seg, fontsize=10, loc="left")
        ax.set_xlabel("exceso, millones")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(millions)
            axis.set_major_locator(MaxNLocator(4))
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.15 * top if top > 0 else None)
        ax.text(0.97, 0.95, "↙ mejor", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.4")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("venta perdida, millones")
    axes[0].legend(fontsize=8, frameon=False, loc="upper center")
    fig.suptitle(title + "  (○ sin colchón, z = 0 · ★ 95 % de servicio)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig
