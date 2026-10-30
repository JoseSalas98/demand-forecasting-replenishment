"""
El objetivo es la participación diaria del artículo en las unidades de la cadena. Las unidades
pronosticadas se obtienen con el total real de la cadena (FR-026); el total no se pronostica.
"""

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import forecast_keys, train_end
from src.freeze import (
    TestLedger,
    freeze,
    load_frozen,
    load_manifest,
    require_declarations,
    require_declared_before,
    utc_now,
)
from src.metrics import naive_scale, rmse, rmsse, wape, wrmsse
from src.segmentation import mask_zero_runs

FAILURE_COLUMNS = ["item_id", "candidate_id", "stage", "message"]

# Rachas que pasan a NaN en la serie de participación (FR-032). Los ceros finales no: en un origen
# son lo último que se sabe del artículo y pueden ser una baja, no un faltante.
SHARE_MASK_KINDS = ("leading", "internal")


def share_series(
    item_units: np.ndarray,
    chain_total: np.ndarray,
    item_ids,
    start_d: int,
    end_d: int,
    zero_thresholds: pd.Series | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Participación diaria = unidades del artículo / unidades totales reales de la cadena (FR-022).

    Los días con total de la cadena igual a 0 quedan NaN y se registran (`zero_chain_total`).
    El prelanzamiento ya viene como NaN en `item_units`. Con `zero_thresholds`, las rachas de
    ceros de arranque y las internas de largo ≥ umbral de [start_d, end_d] quedan NaN (FR-032);
    las posiciones se miden dentro del rango, así que solo usa días ≤ `end_d`.

    Parámetros:
        item_units (np.ndarray): unidades por artículo y día.
        chain_total (np.ndarray): total de la cadena por día.
        item_ids (array-like): identificadores (para el registro y el enmascarado).
        start_d (int): primer día.
        end_d (int): último día.
        zero_thresholds (pd.Series | None): `item_id` → umbral congelado.

    Retorno:
        tuple[np.ndarray, pd.DataFrame]: participación (artículos × días del rango) y registro.
    """
    if zero_thresholds is not None:
        item_units = mask_zero_runs(
            item_units, item_ids, zero_thresholds, start_d, end_d, SHARE_MASK_KINDS
        )
    u = np.asarray(item_units[:, start_d - 1 : end_d], dtype=float)
    total = np.asarray(chain_total[start_d - 1 : end_d], dtype=float)
    zero = total <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = u / np.where(zero, np.nan, total)[None, :]
    days = np.flatnonzero(zero) + start_d
    log = pd.DataFrame(
        {
            "reason": "zero_chain_total",
            "source": "chain_total",
            "row_id": days.astype("int64"),
            "column": [f"d_{d}" for d in days],
            "detail": f"participación indefinida para {len(item_ids)} artículos",
        }
    )
    return shares, log


def candidate_grid(cfg: dict) -> list[dict]:
    """Candidatos declarados en `[forecast]` (FR-023), en orden de desempate.

    Parámetros:
        cfg (dict): configuración.

    Retorno:
        list[dict]: `candidate_id, family, hyperparams`.
    """
    fc = cfg["forecast"]
    grid = [
        {
            "candidate_id": f"snaive_{fc['snaive_m']}",
            "family": "snaive",
            "hyperparams": {"m": fc["snaive_m"]},
        }
    ]
    for w in fc["ma_windows"]:
        grid.append(
            {
                "candidate_id": f"moving_average_{w}",
                "family": "moving_average",
                "hyperparams": {"window": w},
            }
        )
    orders = [(p, d, q) for p in fc["arima_p"] for d in fc["arima_d"] for q in fc["arima_q"]]
    for p, d, q in orders:
        grid.append(
            {
                "candidate_id": f"arima_{p}_{d}_{q}",
                "family": "arima",
                "hyperparams": {"order": [p, d, q]},
            }
        )
    s = fc["sarima_s"]
    pdq = fc["sarima_PDQ"]
    for p, d, q in orders:
        for P in pdq:
            for D in pdq:
                for Q in pdq:
                    grid.append(
                        {
                            "candidate_id": f"sarima_{p}_{d}_{q}_{P}_{D}_{Q}_{s}",
                            "family": "sarima",
                            "hyperparams": {"order": [p, d, q], "seasonal_order": [P, D, Q, s]},
                        }
                    )
    return grid


def _trend(order, seasonal_order) -> str:
    # Sin diferenciación, la participación tiene un nivel distinto de cero: se incluye constante.
    """Devuelve la tendencia de SARIMAX: constante 'c' sin diferenciación, 'n' en otro caso."""
    return "c" if order[1] == 0 and seasonal_order[1] == 0 else "n"


def _series_scale(series: np.ndarray) -> float:
    # La participación es del orden de 1e-4; se normaliza por su media positiva para que el
    # optimizador trabaje en una escala estable. Es un dato derivado de train, no un parámetro.
    """Escala de normalización de la serie: media de los valores positivos de train."""
    x = np.asarray(series, dtype=float)
    pos = x[np.nan_to_num(x) > 0]
    return float(pos.mean()) if len(pos) else 1.0


def _model(history: np.ndarray, frozen: dict) -> SARIMAX:
    """Construye el SARIMAX con la orden y la escala congeladas sobre `history`."""
    return SARIMAX(
        np.asarray(history, dtype=float) / frozen["scale"],
        order=tuple(frozen["order"]),
        seasonal_order=tuple(frozen["seasonal_order"]),
        trend=frozen["trend"],
    )


def _fit_sarimax(series: np.ndarray, order, seasonal_order, family: str) -> dict:
    """Ajusta un SARIMAX y devuelve su entrada congelada; un fallo de convergencia lanza error."""
    frozen = {
        "family": family,
        "order": [int(v) for v in order],
        "seasonal_order": [int(v) for v in seasonal_order],
        "trend": _trend(order, seasonal_order),
        "scale": _series_scale(series),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _model(series, frozen).fit(disp=False)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("el ajuste no convergió (ConvergenceWarning)")
    params = np.asarray(result.params, dtype=float)
    if not np.isfinite(params).all():
        raise RuntimeError("parámetros no finitos")
    frozen["params"] = [float(v) for v in params]
    return frozen


def _fit_snaive(series: np.ndarray, hyperparams: dict) -> dict:
    """Ajusta el seasonal naive como SARIMA(0,0,0)(0,1,0)m."""
    return _fit_sarimax(series, (0, 0, 0), (0, 1, 0, hyperparams["m"]), "snaive")


def _fit_arima(series: np.ndarray, hyperparams: dict) -> dict:
    """Ajusta un ARIMA con la orden de `hyperparams`."""
    return _fit_sarimax(series, hyperparams["order"], (0, 0, 0, 0), "arima")


def _fit_sarima(series: np.ndarray, hyperparams: dict) -> dict:
    """Ajusta un SARIMA con la orden y la orden estacional de `hyperparams`."""
    return _fit_sarimax(series, hyperparams["order"], hyperparams["seasonal_order"], "sarima")


def _forecast_sarimax(history: np.ndarray, frozen: dict, h: int) -> np.ndarray:
    # Solo se actualizan los datos observados; los parámetros quedan fijos (FR-025).
    """Pronostica `h` días aplicando los parámetros congelados sin reajustar."""
    result = _model(history, frozen).smooth(np.asarray(frozen["params"], dtype=float))
    return np.asarray(result.forecast(h), dtype=float) * frozen["scale"]


def _fit_moving_average(series: np.ndarray, hyperparams: dict) -> dict:
    # La media móvil es de forma cerrada: su único "parámetro congelado" es la ventana.
    """Congela la media móvil: solo guarda la ventana."""
    return {
        "family": "moving_average",
        "window": int(hyperparams["window"]),
        "order": None,
        "seasonal_order": None,
        "trend": None,
        "scale": 1.0,
        "params": [],
    }


def _forecast_moving_average(history: np.ndarray, frozen: dict, h: int) -> np.ndarray:
    """Pronostica `h` días con la media de los últimos `window` valores observados."""
    x = np.asarray(history, dtype=float)
    x = x[~np.isnan(x)][-frozen["window"] :]
    level = float(x.mean()) if len(x) else 0.0
    return np.full(h, level)


# Familias de pronosticadores con la misma interfaz `fit(series, hyperparams) -> frozen` y
# `forecast(history, frozen, h)`. Justificación (Principio III, condición a): regla de modelado
# intercambiable con 4 implementaciones reales.
FORECASTERS = {
    "snaive": {"fit": _fit_snaive, "forecast": _forecast_sarimax},
    "moving_average": {"fit": _fit_moving_average, "forecast": _forecast_moving_average},
    "arima": {"fit": _fit_arima, "forecast": _forecast_sarimax},
    "sarima": {"fit": _fit_sarima, "forecast": _forecast_sarimax},
}


def fit_candidate(series: np.ndarray, candidate: dict) -> dict:
    """Ajusta un candidato y devuelve su versión congelable.

    Parámetros:
        series (np.ndarray): participación desde el lanzamiento.
        candidate (dict): `candidate_id, family, hyperparams`.

    Retorno:
        dict: modelo congelable con `candidate_id`, `hyperparams` y `params`.
    """
    frozen = FORECASTERS[candidate["family"]]["fit"](series, candidate["hyperparams"])
    frozen["candidate_id"] = candidate["candidate_id"]
    frozen["hyperparams"] = candidate["hyperparams"]
    return frozen


def forecast_frozen(history: np.ndarray, frozen: dict, h: int) -> np.ndarray:
    """Pronostica `h` pasos con parámetros congelados, sin reestimar.

    Parámetros:
        history (np.ndarray): participación observada hasta el día previo al origen.
        frozen (dict): modelo congelado.
        h (int): horizonte.

    Retorno:
        np.ndarray: participación pronosticada.
    """
    return FORECASTERS[frozen["family"]]["forecast"](history, frozen, h)


def fold_starts(cfg: dict) -> list[int]:
    """Días iniciales de los folds de backtest: bloques de `horizon` que terminan al final de train.

    Con `backtest_folds = 13` y `horizon = 28` los folds cubren Y4 completo (d_1214–d_1577, R16).

    Parámetros:
        cfg (dict): configuración.

    Retorno:
        list[int]: primer día de cada fold, en orden.
    """
    h = cfg["split"]["horizon"]
    end = train_end(cfg)
    n = cfg["forecast"]["backtest_folds"]
    return [end - h * (n - k) + 1 for k in range(n)]


def _selection_unit(item_id: str, level: str) -> str:
    """Devuelve la unidad de selección del artículo según `selection_level` ('item' o 'global')."""
    if level == "item":
        return item_id
    if level == "global":
        return "global"
    raise ValueError(
        f"selection_level='{level}' no está implementado; las etiquetas disponibles son "
        "'item' y 'global'."
    )


def _subset(targets: pd.DataFrame, subset_size) -> list[str]:
    """Devuelve los artículos objetivo ordenados por gasto, recortados a `subset_size`."""
    ordered = targets.sort_values(["spend_train", "item_id"], ascending=[False, True])
    ids = list(ordered["item_id"])
    return ids if subset_size == "all" else ids[: int(subset_size)]


def _launch(agg: dict, k: int) -> int:
    """Devuelve el día de lanzamiento del artículo (1 si no tiene)."""
    value = agg["item_launch_d"][k]
    return int(value) if np.isfinite(value) else 1


def _history(
    shares: np.ndarray, agg: dict, k: int, stop: int, zero_thresholds: pd.Series | None
) -> np.ndarray:
    """Participación del artículo k desde su lanzamiento hasta `stop` (inclusive).

    Sin umbrales recorta `shares` (días desde d_1). Con umbrales recalcula la serie del artículo
    enmascarando solo con los días ≤ `stop`, como se haría en ese origen.
    """
    launch = _launch(agg, k)
    if zero_thresholds is None:
        return shares[k, launch - 1 : stop]
    row = np.asarray(agg["item_units"][k : k + 1], dtype=float)
    s, _ = share_series(
        row, agg["chain_total"], [agg["item_ids"][k]], 1, stop, zero_thresholds
    )
    return s[0, launch - 1 :]


def _clip(share_hat: np.ndarray) -> np.ndarray:
    # Una participación negativa no tiene sentido como unidades (Principio II).
    """Recorta la participación pronosticada a valores mayores o iguales a cero."""
    return np.clip(np.nan_to_num(share_hat, nan=0.0), 0.0, None)


def run_backtest(
    shares: np.ndarray,
    agg: dict,
    targets: pd.DataFrame,
    cfg: dict,
    zero_thresholds: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtesting en train de todos los candidatos (FR-023, FR-024).

    Se detiene si `[forecast]` no se declaró antes de la réplica en el año 5 o si cambió desde
    entonces. Cada candidato se ajusta una vez con los datos previos al primer fold; los folds
    siguientes solo añaden observaciones con parámetros fijos (R16: con 13 folds de 28 días el
    ajuste usa los días ≤ d_1213 y los folds cubren Y4). El puntaje es el RMSSE por artículo
    sobre unidades reales (sin enmascarar), promediado por unidad de selección. Con
    `zero_thresholds`, cada historia se enmascara solo con los días previos a su origen.

    Parámetros:
        shares (np.ndarray): participación (artículos × días desde d_1 hasta fin de train).
        agg (dict): agregado de train.
        targets (pd.DataFrame): lista objetivo congelada.
        cfg (dict): configuración.
        zero_thresholds (pd.Series | None): `item_id` → umbral de racha congelado.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]: `backtest_scores` (`selection_unit, candidate_id,
        fold, rmsse_mean`) y fallos (`item_id, candidate_id, stage, message`).
    """
    keys = forecast_keys(cfg)
    require_declarations(keys, utc_now(), cfg)
    require_declared_before(keys, "matrix_y5")
    fc = cfg["forecast"]
    level = fc["selection_level"]
    _selection_unit("probe", level)
    h = cfg["split"]["horizon"]
    folds = fold_starts(cfg)
    index = {item: k for k, item in enumerate(agg["item_ids"])}
    candidates = candidate_grid(cfg)
    rows, failures = [], []
    for item in _subset(targets, fc["subset_size"]):
        k = index[item]
        launch = _launch(agg, k)
        units = np.asarray(agg["item_units"][k], dtype=float)
        chain = np.asarray(agg["chain_total"], dtype=float)
        scale = naive_scale(units[launch - 1 : folds[0] - 1])
        if not scale > 0:
            failures.append(
                (item, "*", "backtest_scale", "escala naive igual a 0; artículo omitido")
            )
            continue
        fit_history = _history(shares, agg, k, folds[0] - 1, zero_thresholds)
        histories = [_history(shares, agg, k, start - 1, zero_thresholds) for start in folds]
        for cand in candidates:
            try:
                frozen = fit_candidate(fit_history, cand)
            except Exception as exc:  # noqa: BLE001 - cada fallo se registra (Principio IV)
                failures.append((item, cand["candidate_id"], "backtest_fit", str(exc)))
                continue
            for f, start in enumerate(folds, start=1):
                try:
                    share_hat = forecast_frozen(histories[f - 1], frozen, h)
                except Exception as exc:  # noqa: BLE001
                    failures.append((item, cand["candidate_id"], f"backtest_fold_{f}", str(exc)))
                    continue
                units_hat = _clip(share_hat) * chain[start - 1 : start - 1 + h]
                score = rmsse(units[start - 1 : start - 1 + h], units_hat, scale)
                rows.append((_selection_unit(item, level), item, cand["candidate_id"], f, score))
    detail = pd.DataFrame(
        rows, columns=["selection_unit", "item_id", "candidate_id", "fold", "rmsse"]
    )
    scores = (
        detail.groupby(["selection_unit", "candidate_id", "fold"], as_index=False)["rmsse"]
        .mean()
        .rename(columns={"rmsse": "rmsse_mean"})
    )
    return scores, pd.DataFrame(failures, columns=FAILURE_COLUMNS)


def _jsonable(frozen: dict) -> dict:
    """Convierte los arreglos de una entrada congelada en listas serializables."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in frozen.items()}


def select_and_freeze(
    scores: pd.DataFrame, shares: np.ndarray, agg: dict, targets: pd.DataFrame, cfg: dict
) -> tuple[list[dict], pd.DataFrame]:
    """Elige el candidato con menor RMSSE medio por unidad de selección y lo reajusta en todo train.

    Solo compiten los candidatos con puntaje en todos los folds; un empate va al primero de
    `candidate_grid`. Si el reajuste falla, o el artículo no tiene puntajes, se registra y se usa
    el seasonal naive.

    Parámetros:
        scores (pd.DataFrame): salida de `run_backtest`.
        shares (np.ndarray): participación hasta fin de train.
        agg (dict): agregado de train.
        targets (pd.DataFrame): lista objetivo congelada.
        cfg (dict): configuración.

    Retorno:
        tuple[list[dict], pd.DataFrame]: modelos congelables (uno por artículo) y fallos.
    """
    fc = cfg["forecast"]
    level = fc["selection_level"]
    n_folds = fc["backtest_folds"]
    candidates = candidate_grid(cfg)
    by_id = {c["candidate_id"]: c for c in candidates}
    rank = {c["candidate_id"]: i for i, c in enumerate(candidates)}
    summary = scores.groupby(["selection_unit", "candidate_id"]).agg(
        rmsse=("rmsse_mean", "mean"), n=("fold", "nunique")
    )
    summary = summary[summary["n"] == n_folds].reset_index()
    summary["rank"] = summary["candidate_id"].map(rank)
    best = (
        summary.sort_values(["selection_unit", "rmsse", "rank"])
        .groupby("selection_unit")
        .first()["candidate_id"]
        .to_dict()
    )
    index = {item: k for k, item in enumerate(agg["item_ids"])}
    end = train_end(cfg)
    models, failures = [], []
    for item in _subset(targets, fc["subset_size"]):
        k = index[item]
        unit = _selection_unit(item, level)
        history = shares[k, _launch(agg, k) - 1 : end]
        chosen = best.get(unit)
        if chosen is None:
            failures.append(
                (item, "*", "selection", "sin puntaje completo; se usa seasonal naive")
            )
            chosen = candidates[0]["candidate_id"]
        try:
            frozen = fit_candidate(history, by_id[chosen])
        except Exception as exc:  # noqa: BLE001
            failures.append((item, chosen, "refit", f"{exc}; se usa seasonal naive"))
            frozen = fit_candidate(history, candidates[0])
        frozen.update({"selection_unit": unit, "item_ids": [item]})
        models.append(_jsonable(frozen))
    return models, pd.DataFrame(failures, columns=FAILURE_COLUMNS)


def baseline_models(shares: np.ndarray, agg: dict, targets: pd.DataFrame, cfg: dict) -> list[dict]:
    """Seasonal naive ajustado en todo train para cada objetivo (línea base obligatoria).

    Parámetros:
        shares (np.ndarray): participación hasta fin de train.
        agg (dict): agregado de train.
        targets (pd.DataFrame): lista objetivo congelada.
        cfg (dict): configuración.

    Retorno:
        list[dict]: modelos congelables.
    """
    snaive = candidate_grid(cfg)[0]
    index = {item: k for k, item in enumerate(agg["item_ids"])}
    end = train_end(cfg)
    out = []
    for item in _subset(targets, cfg["forecast"]["subset_size"]):
        k = index[item]
        frozen = fit_candidate(shares[k, _launch(agg, k) - 1 : end], snaive)
        frozen.update({"selection_unit": item, "item_ids": [item]})
        out.append(_jsonable(frozen))
    return out


def run_backtesting_stage(cfg: dict) -> dict:
    """Etapa 6: backtesting en train, selección y congelamiento de modelos.

    Solo carga días de train y no lee el EDA ni los artefactos de la réplica.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: `scores`, `failures`, `models` y `baseline`.
    """
    from src.data import assert_train_only, load_interim

    end = train_end(cfg)
    assert_train_only(end, cfg)
    agg = load_interim(cfg, end)
    targets = load_frozen("targets")
    zero_thr = load_frozen("zero_thresholds").set_index("item_id")["threshold"]
    shares, _ = share_series(
        agg["item_units"], agg["chain_total"], agg["item_ids"], 1, end, zero_thr
    )
    scores, fail_bt = run_backtest(shares, agg, targets, cfg, zero_thr)
    models, fail_sel = select_and_freeze(scores, shares, agg, targets, cfg)
    baseline = baseline_models(shares, agg, targets, cfg)
    failures = pd.concat([fail_bt, fail_sel], ignore_index=True)
    freeze(models, "frozen_models", "models/frozen_models.json", cfg)
    freeze(baseline, "baseline_models", "models/baseline_models.json", cfg)
    freeze(scores, "backtest_scores", "data/processed/backtest_scores.parquet", cfg)
    failures.to_parquet("models/fit_failures.parquet", index=False)
    return {"scores": scores, "failures": failures, "models": models, "baseline": baseline}


def evaluate_test(model_id: str, models_artifact: str, agg: dict, cfg: dict) -> dict:
    """Evalúa un modelo congelado una sola vez en los orígenes del test (FR-025 a FR-027).

    En cada origen el modelo se reconstruye con los datos observados hasta el día previo y
    pronostica `horizon` días sin reestimar. Las unidades usan el total real de la cadena. Los
    pesos del WRMSSE son la participación en gasto en `weight_window` y la escala sale de train.

    Parámetros:
        model_id (str): identificador único de la evaluación.
        models_artifact (str): nombre del artefacto de modelos congelados en el manifiesto.
        agg (dict): agregado con los días del test.
        cfg (dict): configuración.

    Retorno:
        dict: `model_id, wrmsse, wrmsse_o1 … wrmsse_oN, wape, rmse, n_items, excluded`.
    """
    ledger = TestLedger()
    if ledger.has(model_id):
        raise RuntimeError(f"El modelo '{model_id}' ya se evaluó en test (SC-007).")
    start_run = utc_now()
    manifest = load_manifest()
    for name, entry in manifest["artifacts"].items():
        if datetime.fromisoformat(entry["frozen_at"]) >= start_run:
            raise RuntimeError(f"'{name}' se congeló después del inicio de la evaluación.")
    models = load_frozen(models_artifact)
    targets = load_frozen("targets")
    if not {m["item_ids"][0] for m in models} <= set(targets["item_id"]):
        raise RuntimeError("Hay modelos para artículos fuera de la lista objetivo congelada.")
    index = {item: k for k, item in enumerate(agg["item_ids"])}
    end = train_end(cfg)
    test_start = cfg["split"]["test_days"][0]
    h = cfg["split"]["horizon"]
    zero_thr = load_frozen("zero_thresholds").set_index("item_id")["threshold"]
    chain = np.asarray(agg["chain_total"], dtype=float)
    a, b = cfg["metrics"]["weight_window"]
    scales, weights = {}, {}
    for m in models:
        item = m["item_ids"][0]
        k = index[item]
        units = np.asarray(agg["item_units"][k], dtype=float)
        scales[item] = naive_scale(units[_launch(agg, k) - 1 : end])
        weights[item] = float(np.nansum(np.asarray(agg["item_spend"][k, a - 1 : b], dtype=float)))
    rows, per_origin, excluded = [], {}, set()
    all_y, all_hat = [], []
    for origin in range(1, cfg["split"]["test_origins"] + 1):
        s = test_start + (origin - 1) * h
        y_d, hat_d = {}, {}
        for m in models:
            item = m["item_ids"][0]
            k = index[item]
            history = _history(None, agg, k, s - 1, zero_thr)
            share_hat = _clip(forecast_frozen(history, m, h))
            units_hat = share_hat * chain[s - 1 : s - 1 + h]
            y = np.asarray(agg["item_units"][k, s - 1 : s - 1 + h], dtype=float)
            y_d[item], hat_d[item] = y, units_hat
            all_y.append(y)
            all_hat.append(units_hat)
            days = np.arange(s, s + h)
            rows.append(
                pd.DataFrame(
                    {
                        "model_id": model_id,
                        "item_id": item,
                        "origin": origin,
                        "h": np.arange(1, h + 1),
                        "d": days,
                        "share_hat": share_hat,
                        "units_hat": units_hat,
                        "units_actual": y,
                    }
                )
            )
        value, excl = wrmsse(y_d, hat_d, scales, weights)
        per_origin[f"wrmsse_o{origin}"] = value
        excluded.update(excl)
    forecasts = pd.concat(rows, ignore_index=True)
    path = "data/processed/forecasts_test.parquet"
    try:
        previous = pd.read_parquet(path)
        forecasts = pd.concat([previous, forecasts], ignore_index=True)
    except FileNotFoundError:
        pass
    forecasts.to_parquet(path, index=False)
    y_all, hat_all = np.concatenate(all_y), np.concatenate(all_hat)
    result = {
        "model_id": model_id,
        "wrmsse": float(np.nanmean(list(per_origin.values()))),
        **per_origin,
        "wape": wape(y_all, hat_all),
        "rmse": rmse(y_all, hat_all),
        "n_items": len(models),
        "excluded": ",".join(sorted(excluded)),
    }
    ledger.record(
        model_id,
        manifest["artifacts"][models_artifact]["sha256"],
        manifest["artifacts"]["targets"]["sha256"],
    )
    return result


def run_test_evaluation_stage(cfg: dict) -> pd.DataFrame:
    """Etapa 7: evalúa una sola vez el seasonal naive y el modelo congelado, y los compara.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        pd.DataFrame: una fila por modelo con `beats_snaive`; se guarda en
        `data/processed/evaluation.parquet`.
    """
    from src.data import load_interim

    version = cfg["version"]
    agg = load_interim(cfg, cfg["split"]["test_days"][1])
    level = cfg["forecast"]["selection_level"]
    base = evaluate_test(f"snaive_{version}", "baseline_models", agg, cfg)
    chosen = evaluate_test(f"selected_{level}_{version}", "frozen_models", agg, cfg)
    evaluation = pd.DataFrame([base, chosen])
    evaluation["beats_snaive"] = evaluation["wrmsse"] < base["wrmsse"]
    path = "data/processed/evaluation.parquet"
    try:
        evaluation = pd.concat([pd.read_parquet(path), evaluation], ignore_index=True)
    except FileNotFoundError:
        pass
    evaluation.to_parquet(path, index=False)
    return evaluation[
        evaluation["model_id"].isin([base["model_id"], chosen["model_id"]])
    ].reset_index(drop=True)


def round_half_up(values: np.ndarray) -> np.ndarray:
    """Redondea a enteros con la regla de la mitad hacia arriba (para mostrar en el informe).

    Parámetros:
        values (np.ndarray): valores no negativos.

    Retorno:
        np.ndarray: enteros (int64).
    """
    return np.floor(np.asarray(values, dtype=float) + 0.5).astype("int64")


def final_report_tables(cfg: dict) -> dict:
    """Etapa 8: arma las tablas del informe final a partir de los artefactos.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: `targets` (clase, celda, gasto, tratamiento, réplica Y5 y pronóstico del último
        origen), `evaluation`, `eda_findings`, `stability`, `stability_y5` y `last_origin_daily`.
    """
    from src.stability import EDA_DIR

    version = cfg["version"]
    level = cfg["forecast"]["selection_level"]
    chosen_id = f"selected_{level}_{version}"
    targets = load_frozen("targets")
    rep = load_frozen("year5_replication")
    evaluation = pd.read_parquet("data/processed/evaluation.parquet")
    evaluation = evaluation[evaluation["model_id"].isin([f"snaive_{version}", chosen_id])]
    forecasts = pd.read_parquet("data/processed/forecasts_test.parquet")
    fc = forecasts[forecasts["model_id"] == chosen_id]
    last = fc[fc["origin"] == fc["origin"].max()].copy()
    last["units_hat_int"] = round_half_up(last["units_hat"])
    summary = last.groupby("item_id").agg(
        forecast_units_28d=("units_hat", "sum"), actual_units_28d=("units_actual", "sum")
    )
    summary["forecast_units_28d"] = round_half_up(summary["forecast_units_28d"])
    models = {m["item_ids"][0]: m["candidate_id"] for m in load_frozen("frozen_models")}
    table = targets.merge(
        rep[["item_id", "y5_cell", "same_as_frozen"]], on="item_id", how="left"
    ).merge(summary, on="item_id", how="left")
    table["model"] = table["item_id"].map(models)
    matrix_train = load_frozen("matrix_train")
    eda = pd.read_parquet(f"{EDA_DIR}/matrix_five_years.parquet")
    findings = matrix_train[["item_id", "abc_class", "xyz_class"]].merge(
        eda[["item_id", "abc_class", "xyz_class"]], on="item_id", suffixes=("_train", "_5y")
    )
    findings = findings[
        (findings["abc_class_train"] != findings["abc_class_5y"])
        | (findings["xyz_class_train"] != findings["xyz_class_5y"])
    ]
    return {
        "targets": table,
        "evaluation": evaluation.reset_index(drop=True),
        "eda_findings": findings.reset_index(drop=True),
        "stability": load_frozen("stability"),
        "stability_y5": load_frozen("stability_y5"),
        "last_origin_daily": last[["item_id", "d", "units_hat_int", "units_actual"]],
    }
