"""
Los datos se guardan como arreglos anchos (series × día), nunca en formato largo. La columna j de
cada arreglo corresponde al día d = j + 1. NaN significa celda enmascarada (prelanzamiento o dato
rechazado): no es un cero.
"""

from pathlib import Path

import numpy as np
import pandas as pd

INTERIM_DIR = Path("data/interim")
LOG_COLUMNS = ["reason", "source", "row_id", "column", "detail"]


def load_calendar(cfg: dict) -> pd.DataFrame:
    """Lee `calendar.csv` desde `data.raw_dir`.

    Parámetros:
        cfg (dict): configuración.

    Retorno:
        pd.DataFrame: calendario crudo.
    """
    return pd.read_csv(Path(cfg["data"]["raw_dir"]) / "calendar.csv")


def validate_calendar(calendar: pd.DataFrame, last_day: int) -> None:
    """Valida el calendario y detiene el proceso si no es correcto (FR-003).

    Reglas: fechas continuas y únicas, un `wm_yr_wk` por `d`, `d_1` a `d_{last_day}` presentes y
    cada `wm_yr_wk` con ≤ 7 días.

    Parámetros:
        calendar (pd.DataFrame): calendario con columnas `d`, `date` y `wm_yr_wk`.
        last_day (int): último día con ventas.

    Retorno:
        None. Lanza ValueError si alguna regla falla.
    """
    if calendar["d"].duplicated().any():
        raise ValueError("Calendario inválido: hay días 'd' duplicados (un wm_yr_wk por d).")
    dates = pd.to_datetime(calendar["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Calendario inválido: hay fechas no válidas.")
    if dates.duplicated().any():
        raise ValueError("Calendario inválido: hay fechas duplicadas.")
    day_num = calendar["d"].str.removeprefix("d_").astype(int)
    order = np.argsort(day_num.to_numpy())
    if not (np.diff(day_num.to_numpy()[order]) == 1).all():
        raise ValueError("Calendario inválido: la secuencia de días 'd' no es continua.")
    if not (
        np.diff(dates.to_numpy()[order]).astype("timedelta64[D]") == np.timedelta64(1, "D")
    ).all():
        raise ValueError("Calendario inválido: las fechas no son continuas.")
    if day_num.min() != 1 or day_num.max() < last_day:
        raise ValueError(f"Calendario inválido: debe cubrir d_1 a d_{last_day}.")
    if (calendar.groupby("wm_yr_wk").size() > 7).any():
        raise ValueError("Calendario inválido: hay semanas wm_yr_wk con más de 7 días.")


def week_of_day(calendar: pd.DataFrame, last_day: int) -> np.ndarray:
    """Devuelve el `wm_yr_wk` de cada día 1..last_day.

    Parámetros:
        calendar (pd.DataFrame): calendario validado.
        last_day (int): último día.

    Retorno:
        np.ndarray: arreglo int64 de largo `last_day`.
    """
    day_num = calendar["d"].str.removeprefix("d_").astype(int)
    wk = pd.Series(calendar["wm_yr_wk"].to_numpy(), index=day_num.to_numpy()).sort_index()
    return wk.loc[1:last_day].to_numpy(dtype=np.int64)


def load_sales(cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Lee el archivo de ventas como arreglo ancho ordenado por tienda y artículo.

    Parámetros:
        cfg (dict): configuración (`data.raw_dir`, `data.sales_file`, `data.last_day`).

    Retorno:
        tuple[np.ndarray, pd.DataFrame]: `units` float32 (series × last_day) y `series_index`
        con `store_id, item_id, csv_row` (fila de datos en el CSV, base 0).
    """
    last_day = cfg["data"]["last_day"]
    day_cols = [f"d_{d}" for d in range(1, last_day + 1)]
    path = Path(cfg["data"]["raw_dir"]) / cfg["data"]["sales_file"]
    dtypes = {c: np.float32 for c in day_cols}
    df = pd.read_csv(path, usecols=["store_id", "item_id", *day_cols], dtype=dtypes)
    keys = df[["store_id", "item_id"]].reset_index(names="csv_row")
    keys = keys.sort_values(["store_id", "item_id"], kind="stable")
    units = df[day_cols].to_numpy(dtype=np.float32)[keys["csv_row"].to_numpy()]
    series_index = keys[["store_id", "item_id", "csv_row"]].reset_index(drop=True)
    return units, series_index


def load_prices(cfg: dict) -> pd.DataFrame:
    """Lee `sell_prices.csv` y conserva la fila original de cada precio.

    Parámetros:
        cfg (dict): configuración.

    Retorno:
        pd.DataFrame: `store_id, item_id, wm_yr_wk, sell_price, csv_row`.
    """
    df = pd.read_csv(Path(cfg["data"]["raw_dir"]) / "sell_prices.csv")
    df["csv_row"] = np.arange(len(df))
    return df


def _series_positions(series_index: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Devuelve la fila de `series_index` de cada (tienda, artículo) de `frame` (-1 si no existe)."""
    key = pd.MultiIndex.from_frame(series_index[["store_id", "item_id"]])
    return key.get_indexer(pd.MultiIndex.from_frame(frame[["store_id", "item_id"]]))


def build_daily_prices(
    prices: pd.DataFrame, week_of_day_arr: np.ndarray, series_index: pd.DataFrame
) -> np.ndarray:
    """Expande los precios semanales a precio por tienda-artículo-día.

    El cruce es día → `wm_yr_wk` (calendario) → `sell_price`. Donde no hay precio queda NaN.

    Parámetros:
        prices (pd.DataFrame): precios semanales.
        week_of_day_arr (np.ndarray): `wm_yr_wk` de cada día.
        series_index (pd.DataFrame): orden de las series.

    Retorno:
        np.ndarray: arreglo float32 con la misma forma que `units`.
    """
    weeks = np.unique(week_of_day_arr)
    week_pos_of_day = np.searchsorted(weeks, week_of_day_arr)
    weekly = np.full((len(series_index), len(weeks)), np.nan, dtype=np.float32)
    rows = _series_positions(series_index, prices)
    cols = np.searchsorted(weeks, prices["wm_yr_wk"].to_numpy())
    keep = (rows >= 0) & (cols < len(weeks))
    keep &= np.isin(prices["wm_yr_wk"].to_numpy(), weeks)
    weekly[rows[keep], cols[keep]] = prices["sell_price"].to_numpy(dtype=np.float32)[keep]
    return weekly[:, week_pos_of_day]


def compute_launch(
    prices_daily: np.ndarray, series_index: pd.DataFrame, week_of_day_arr: np.ndarray
) -> pd.DataFrame:
    """Calcula la semana y el día de lanzamiento por tienda-artículo (FR-004).

    El lanzamiento es la primera semana con precio en el archivo de precios.

    Parámetros:
        prices_daily (np.ndarray): precios diarios (NaN = sin precio).
        series_index (pd.DataFrame): orden de las series.
        week_of_day_arr (np.ndarray): `wm_yr_wk` de cada día.

    Retorno:
        pd.DataFrame: `store_id, item_id, launch_wk, launch_d` (NaN si nunca tiene precio).
    """
    has_price = ~np.isnan(prices_daily)
    any_price = has_price.any(axis=1)
    first = np.argmax(has_price, axis=1)
    launch_d = np.where(any_price, first + 1, np.nan)
    launch_wk = np.where(any_price, week_of_day_arr[first], np.nan)
    out = series_index[["store_id", "item_id"]].copy()
    out["launch_wk"] = launch_wk
    out["launch_d"] = launch_d
    return out


def _log(reason, source, row_id, column, detail) -> pd.DataFrame:
    """Construye filas del registro de rechazos con motivo, fuente, fila, columna y detalle."""
    return pd.DataFrame(
        {"reason": reason, "source": source, "row_id": row_id, "column": column, "detail": detail}
    )


def apply_masks(
    units: np.ndarray,
    prices_daily: np.ndarray,
    launch: pd.DataFrame,
    series_index: pd.DataFrame,
    week_of_day_arr: np.ndarray,
    prices: pd.DataFrame,
    sales_file: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Enmascara prelanzamiento y datos inválidos, y registra cada rechazo (FR-002, research R3).

    Casos:
        - unidades no enteras o < 0 → celda NaN, `invalid_units`.
        - unidades > 0 antes del primer precio → celda NaN, `sales_before_first_price`.
        - precio faltante o ≤ 0 tras el lanzamiento con unidades > 0 en la semana → semana NaN,
          `invalid_price_with_sales` (el gasto no se imputa).
        - precio faltante tras el lanzamiento con unidades = 0 → semana NaN, `price_gap_no_sales`
          (informativo, no es rechazo).
        - prelanzamiento sin precio → NaN sin registro.

    Parámetros:
        units (np.ndarray): unidades crudas (series × día).
        prices_daily (np.ndarray): precios diarios.
        launch (pd.DataFrame): lanzamiento por serie.
        series_index (pd.DataFrame): orden de las series con `csv_row`.
        week_of_day_arr (np.ndarray): `wm_yr_wk` de cada día.
        prices (pd.DataFrame): precios semanales con `csv_row`.
        sales_file (str): nombre del archivo de ventas (fuente del registro).

    Retorno:
        tuple[np.ndarray, np.ndarray, pd.DataFrame]: unidades enmascaradas, precios enmascarados
        y registro con columnas `reason, source, row_id, column, detail`.
    """
    _n_series, n_days = units.shape
    day_idx = np.arange(n_days)
    launch_d = launch["launch_d"].to_numpy()
    launch0 = np.where(np.isnan(launch_d), n_days, launch_d - 1).astype(np.int64)
    pre = day_idx[None, :] < launch0[:, None]

    finite = np.isfinite(units)
    bad_units = finite & ((units < 0) | (units != np.floor(units)))
    bad_units |= ~finite
    before_price = pre & ~bad_units & (units > 0)
    valid_units = ~bad_units & ~pre

    # Revisión por semana: el precio es constante dentro de una semana wm_yr_wk.
    week_starts = np.flatnonzero(np.r_[True, week_of_day_arr[1:] != week_of_day_arr[:-1]])
    bad_price_day = ~pre & ~(prices_daily > 0)
    bad_week = np.add.reduceat(bad_price_day, week_starts, axis=1) > 0
    sold = np.where(valid_units, units, 0).astype(np.float64)
    sold_week = np.add.reduceat(sold, week_starts, axis=1) > 0
    week_len = np.diff(np.r_[week_starts, n_days])
    bad_week_days = np.repeat(bad_week, week_len, axis=1)

    mask = pre | bad_units | bad_week_days
    masked_units = np.where(mask, np.nan, units).astype(np.float32)
    masked_prices = np.where(mask, np.nan, prices_daily).astype(np.float32)

    csv_row = series_index["csv_row"].to_numpy()
    names = (series_index["store_id"] + "/" + series_index["item_id"]).to_numpy()
    logs = []
    r, c = np.nonzero(bad_units)
    if len(r):
        logs.append(
            _log(
                "invalid_units",
                sales_file,
                csv_row[r],
                [f"d_{j + 1}" for j in c],
                [f"{names[i]} units={units[i, j]}" for i, j in zip(r, c)],
            )
        )
    r, c = np.nonzero(before_price)
    if len(r):
        logs.append(
            _log(
                "sales_before_first_price",
                sales_file,
                csv_row[r],
                [f"d_{j + 1}" for j in c],
                [f"{names[i]} units={units[i, j]}" for i, j in zip(r, c)],
            )
        )
    week_ids = week_of_day_arr[week_starts]
    price_key = prices.set_index(["store_id", "item_id", "wm_yr_wk"])
    for reason, selector in (
        ("invalid_price_with_sales", bad_week & sold_week),
        ("price_gap_no_sales", bad_week & ~sold_week),
    ):
        r, w = np.nonzero(selector)
        if not len(r):
            continue
        stores = series_index["store_id"].to_numpy()[r]
        items = series_index["item_id"].to_numpy()[r]
        wks = week_ids[w]
        idx = pd.MultiIndex.from_arrays([stores, items, wks])
        pos = price_key.index.get_indexer(idx)
        has_row = pos >= 0
        price_rows = np.where(has_row, price_key["csv_row"].to_numpy()[np.maximum(pos, 0)], -1)
        source = np.where(has_row, "sell_prices.csv", sales_file)
        row_id = np.where(has_row, price_rows, csv_row[r])
        first_d = week_starts[w] + 1
        last_d = first_d + week_len[w] - 1
        detail = [
            f"{s}/{it} wm_yr_wk={wk} d_{a}-d_{b} " + ("price<=0" if h else "missing price")
            for s, it, wk, a, b, h in zip(stores, items, wks, first_d, last_d, has_row)
        ]
        logs.append(_log(reason, source, row_id, [f"wm_yr_wk={wk}" for wk in wks], detail))
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=LOG_COLUMNS)
    log["row_id"] = log["row_id"].astype("int64")
    return masked_units, masked_prices, log[LOG_COLUMNS]


def aggregate_items(
    units: np.ndarray, prices_daily: np.ndarray, series_index: pd.DataFrame, launch: pd.DataFrame
) -> dict:
    """Agrega las series tienda-artículo a artículo y calcula el total de la cadena.

    El gasto se calcula por tienda-día como unidades × precio antes de sumar; nunca como precio
    medio × unidades (Principio II).

    Parámetros:
        units (np.ndarray): unidades enmascaradas.
        prices_daily (np.ndarray): precios enmascarados.
        series_index (pd.DataFrame): orden de las series.
        launch (pd.DataFrame): lanzamiento por serie.

    Retorno:
        dict: `item_ids`, `item_units`, `item_spend` (artículos × día, NaN el día sin ninguna
        tienda válida), `item_launch_d` (mínimo entre tiendas) y `chain_total` (por día).
    """
    item_codes, item_ids = pd.factorize(series_index["item_id"], sort=True)
    order = np.argsort(item_codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(item_codes[order]) != 0])
    u = units[order]
    valid = ~np.isnan(u)
    spend_cells = np.nan_to_num(u * prices_daily[order])
    item_units = np.add.reduceat(np.nan_to_num(u), starts, axis=0, dtype=np.float64)
    item_spend = np.add.reduceat(spend_cells, starts, axis=0, dtype=np.float64)
    any_valid = np.add.reduceat(valid, starts, axis=0) > 0
    item_units[~any_valid] = np.nan
    item_spend[~any_valid] = np.nan
    launch_by_item = (
        pd.Series(launch["launch_d"].to_numpy(), index=series_index["item_id"].to_numpy())
        .groupby(level=0)
        .min()
        .reindex(item_ids)
        .to_numpy()
    )
    chain_total = np.nansum(units, axis=0, dtype=np.float64)
    return {
        "item_ids": np.asarray(item_ids, dtype=object),
        "item_units": item_units,
        "item_spend": item_spend,
        "item_launch_d": launch_by_item,
        "chain_total": chain_total,
    }


def save_interim(arrays: dict, tables: dict) -> None:
    """Escribe los arreglos (.npy) y tablas (Parquet) intermedios en `data/interim/`.

    Parámetros:
        arrays (dict): nombre → np.ndarray (p. ej. `units`, `item_units`, `chain_total`).
        tables (dict): nombre → pd.DataFrame (p. ej. `launch`, `rejections`).

    Retorno:
        None
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(INTERIM_DIR / f"{name}.npy", np.asarray(arr))
    for name, df in tables.items():
        df.to_parquet(INTERIM_DIR / f"{name}.parquet", index=False)


def interim_from_aggregate(agg: dict) -> tuple[dict, dict]:
    """Separa un agregado en arreglos y tablas para `save_interim`.

    Parámetros:
        agg (dict): salida de `aggregate_items`.

    Retorno:
        tuple[dict, dict]: arreglos y tablas.
    """
    arrays = {k: agg[k] for k in ("item_units", "item_spend", "chain_total")}
    tables = {
        "item_index": pd.DataFrame(
            {"item_id": agg["item_ids"].astype(str), "item_launch_d": agg["item_launch_d"]}
        )
    }
    return arrays, tables


def load_interim(cfg: dict, max_day: int, series_level: bool = False) -> dict:
    """Carga los intermedios hasta `max_day` sin leer días posteriores.

    Los `.npy` se abren con `mmap_mode="r"` y se recortan a los días 1..max_day, así las etapas
    de decisión no tocan días del test.

    Parámetros:
        cfg (dict): configuración.
        max_day (int): último día a cargar.
        series_level (bool): si es True, incluye `units` y `prices_daily` por tienda-artículo.

    Retorno:
        dict: `item_ids`, `item_units`, `item_spend`, `item_launch_d`, `chain_total`,
        `week_of_day`, `max_day` y, opcionalmente, `units`, `prices_daily` y `series_index`.
    """
    if max_day > cfg["data"]["last_day"]:
        raise ValueError("max_day supera data.last_day")
    item_index = pd.read_parquet(INTERIM_DIR / "item_index.parquet")
    out = {
        "item_ids": item_index["item_id"].to_numpy(dtype=object),
        "item_launch_d": item_index["item_launch_d"].to_numpy(),
        "max_day": max_day,
    }
    for name in ("item_units", "item_spend"):
        out[name] = np.load(INTERIM_DIR / f"{name}.npy", mmap_mode="r")[:, :max_day]
    for name in ("chain_total", "week_of_day"):
        out[name] = np.load(INTERIM_DIR / f"{name}.npy", mmap_mode="r")[:max_day]
    if series_level:
        for name in ("units", "prices_daily"):
            out[name] = np.load(INTERIM_DIR / f"{name}.npy", mmap_mode="r")[:, :max_day]
        out["series_index"] = pd.read_parquet(INTERIM_DIR / "series_index.parquet")
    # Un artículo lanzado después de max_day no tiene historia visible en este corte.
    launch = out["item_launch_d"].astype(float)
    out["item_launch_d"] = np.where(launch > max_day, np.nan, launch)
    return out


def assert_train_only(max_day: int, cfg: dict) -> None:
    """Detiene el proceso si una etapa de decisión intenta cargar días del test.

    Parámetros:
        max_day (int): último día que la etapa quiere cargar.
        cfg (dict): configuración.

    Retorno:
        None. Lanza RuntimeError si `max_day >= test_days[0]`.
    """
    if max_day >= cfg["split"]["test_days"][0]:
        raise RuntimeError(
            f"Etapa de decisión con max_day={max_day}: no puede cargar días del test "
            f"(≥ {cfg['split']['test_days'][0]}, Principio VI)."
        )


def run_data_validation(cfg: dict) -> dict:
    """Ejecuta la etapa 1 completa y guarda los intermedios.

    Parámetros:
        cfg (dict): configuración validada.

    Retorno:
        dict: `rejections`, `launch`, `series_index` y el agregado `agg`.
    """
    last_day = cfg["data"]["last_day"]
    calendar = load_calendar(cfg)
    validate_calendar(calendar, last_day)
    wod = week_of_day(calendar, last_day)
    units, series_index = load_sales(cfg)
    prices = load_prices(cfg)
    prices_daily = build_daily_prices(prices, wod, series_index)
    launch = compute_launch(prices_daily, series_index, wod)
    masked_units, masked_prices, rejections = apply_masks(
        units, prices_daily, launch, series_index, wod, prices, cfg["data"]["sales_file"]
    )
    del units, prices_daily
    agg = aggregate_items(masked_units, masked_prices, series_index, launch)
    arrays, tables = interim_from_aggregate(agg)
    arrays.update({"units": masked_units, "prices_daily": masked_prices, "week_of_day": wod})
    tables.update({"series_index": series_index, "launch": launch, "rejections": rejections})
    save_interim(arrays, tables)
    return {"rejections": rejections, "launch": launch, "series_index": series_index, "agg": agg}
