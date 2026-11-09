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

# Motivo de cada celda (serie × día) en el arreglo `mask_reason`. Los nombres coinciden con los
# `reason` del registro de rechazos. 0 = celda válida; cualquier otro código = celda en NaN.
MASK_REASONS = {
    0: "valid",
    1: "pre_launch",  # antes de la primera semana con precio y sin ventas; no es rechazo
    2: "sales_before_first_price",  # ventas antes del primer precio; rechazo
    3: "invalid_units",  # unidades no enteras, negativas o faltantes; rechazo
    4: "invalid_price_with_sales",  # semana sin precio válido y con ventas; rechazo
    5: "price_gap_no_sales",  # semana sin precio válido y sin ventas; informativo
}
REJECTION_CODES = (2, 3, 4)


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
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Enmascara prelanzamiento y datos inválidos, y registra cada rechazo (FR-002, research R3).

    Casos (código de `mask_reason` entre paréntesis):
        - prelanzamiento sin ventas → NaN sin registro (1, `pre_launch`).
        - unidades > 0 antes del primer precio → celda NaN, `sales_before_first_price` (2).
        - unidades no enteras, < 0 o faltantes → celda NaN, `invalid_units` (3).
        - precio faltante o ≤ 0 tras el lanzamiento con unidades > 0 en la semana → semana NaN,
          `invalid_price_with_sales` (4; el gasto no se imputa).
        - precio faltante tras el lanzamiento con unidades = 0 → semana NaN, `price_gap_no_sales`
          (5; informativo, no es rechazo).

    El registro tiene una fila por celda en los casos 2 y 3, y una fila por semana en los casos
    4 y 5. `mask_reason` tiene el motivo de cada celda, así se puede contar y comparar el efecto
    de cada regla sobre los arreglos.

    Parámetros:
        units (np.ndarray): unidades crudas (series × día).
        prices_daily (np.ndarray): precios diarios.
        launch (pd.DataFrame): lanzamiento por serie.
        series_index (pd.DataFrame): orden de las series con `csv_row`.
        week_of_day_arr (np.ndarray): `wm_yr_wk` de cada día.
        prices (pd.DataFrame): precios semanales con `csv_row`.
        sales_file (str): nombre del archivo de ventas (fuente del registro).

    Retorno:
        tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]: unidades enmascaradas, precios
        enmascarados, registro con columnas `reason, source, row_id, column, detail` y
        `mask_reason` int8 (series × día) con los códigos de `MASK_REASONS`.
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

    # Códigos por celda; el orden de asignación define la precedencia (el último gana).
    mask_reason = np.zeros(units.shape, dtype=np.int8)
    post = ~pre
    mask_reason[bad_week_days & post & np.repeat(~sold_week, week_len, axis=1)] = 5
    mask_reason[bad_week_days & post & np.repeat(sold_week, week_len, axis=1)] = 4
    mask_reason[pre] = 1
    mask_reason[before_price] = 2
    mask_reason[bad_units] = 3

    mask = mask_reason > 0
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
    return masked_units, masked_prices, log[LOG_COLUMNS], mask_reason


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


def active_stores_by_item(units: np.ndarray, series_index: pd.DataFrame) -> np.ndarray:
    """Cuenta, por artículo y día, las tiendas con dato válido (no NaN).

    Un cero del artículo agregado significa cero en todas las tiendas con dato. Si son pocas, el
    agregado se comporta como una serie de una sola tienda; este conteo permite verlo.

    Parámetros:
        units (np.ndarray): unidades enmascaradas por serie tienda-artículo y día.
        series_index (pd.DataFrame): orden de las series (`item_id`).

    Retorno:
        np.ndarray: int16 (artículos × día), con filas en el orden de `item_ids` de
        `aggregate_items` (artículos ordenados).
    """
    item_codes, _ = pd.factorize(series_index["item_id"], sort=True)
    order = np.argsort(item_codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(item_codes[order]) != 0])
    valid = (~np.isnan(np.asarray(units)))[order]
    return np.add.reduceat(valid, starts, axis=0, dtype=np.int16)


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


def save_tables(out_dir, tables: dict, manifest: dict | None = None) -> list[Path]:
    """Guarda varias tablas en Parquet en `out_dir`, con un `manifest.json` que las describe.

    Parquet conserva Int64 con nulos y str. Las columnas category (con sus categorías y si son
    ordenadas) se anotan en el manifiesto y `load_tables` las restaura: Parquet devuelve como
    int64 una category de enteros.

    Parámetros:
        out_dir: carpeta de destino (se crea si no existe).
        tables (dict): nombre → pd.DataFrame.
        manifest (dict | None): datos extra para el manifiesto (listas de variables, parámetros).

    Retorno:
        list[Path]: rutas escritas (las tablas y el manifiesto).
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths.append(path)
    categories = {name: {c: {"categories": df[c].cat.categories.tolist(), "ordered": bool(df[c].cat.ordered)}
                         for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)}
                  for name, df in tables.items()}
    meta = {"saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "tables": {name: list(df.shape) for name, df in tables.items()}, "categories": categories,
            **(manifest or {})}
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths + [path]


def load_tables(out_dir) -> tuple[dict, dict]:
    """Lee las tablas que escribió `save_tables`, restaura sus columnas category y comprueba sus
    dimensiones con el manifiesto.

    Retorno:
        tuple[dict, dict]: nombre → pd.DataFrame y el manifiesto.
    """
    import json

    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    tables = {name: pd.read_parquet(out_dir / f"{name}.parquet") for name in manifest["tables"]}
    for name, cols in manifest.get("categories", {}).items():
        for c, spec in cols.items():
            tables[name][c] = tables[name][c].astype(pd.CategoricalDtype(spec["categories"], ordered=spec["ordered"]))
    for name, shape in manifest["tables"].items():
        if list(tables[name].shape) != shape:
            raise ValueError(f"{name}: {tables[name].shape} en disco, {tuple(shape)} en el manifiesto")
    return tables, manifest


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


def profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnóstico rápido por columna: tipo, nulos explícitos y valores distintos.

    Sirve para revisar cualquier tabla cargada o unida; los nulos implícitos (filas ausentes,
    como una semana sin precio) no aparecen aquí hasta que un merge los vuelve explícitos.

    Parámetros:
        df (pd.DataFrame): tabla a describir.

    Retorno:
        pd.DataFrame: una fila por columna de `df` con `dtype`, `nulls`, `null_pct` (0–100,
        dos decimales) y `unique` (sin contar nulos).
    """
    nulls = df.isna()
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulls": nulls.sum(),
            "null_pct": (nulls.mean() * 100).round(2),
            "unique": df.nunique(),
        }
    )


def to_daily_frame(arr: np.ndarray, item_ids, start_d: int, end_d: int) -> pd.DataFrame:
    """Recorta un arreglo artículo × día (columna j = día j + 1) a [start_d, end_d] como DataFrame.

    Parámetros:
        arr (np.ndarray): arreglo ancho desde d_1 (por ejemplo, `item_units` de `load_interim`).
        item_ids (array-like): identificador de cada fila.
        start_d (int): primer día (inclusivo).
        end_d (int): último día (inclusivo).

    Retorno:
        pd.DataFrame: float64, índice `item_id` y columnas `d_<día>`.
    """
    return pd.DataFrame(
        np.asarray(arr[:, start_d - 1 : end_d], dtype=float),
        index=pd.Index(np.asarray(item_ids, dtype=object), name="item_id"),
        columns=[f"d_{d}" for d in range(start_d, end_d + 1)],
    )


def sales_overview(units: np.ndarray, series_index: pd.DataFrame) -> pd.Series:
    """Resumen del arreglo ancho de ventas crudas: tamaño, nulos y valores fuera de dominio.

    Cuenta por separado los negativos y los no enteros porque la regla de dominio (Principio II)
    rechaza ambos, y conviene saber si existen antes de cualquier tratamiento.

    Parámetros:
        units (np.ndarray): unidades crudas (series × día), como las devuelve `load_sales`.
        series_index (pd.DataFrame): `store_id, item_id` de cada fila de `units`.

    Retorno:
        pd.Series: `series`, `stores`, `items`, `days`, `cells`, `nan_cells`, `negative_cells`,
        `non_integer_cells`, `zero_cells`, `zero_pct` (0–100) y `total_units`.
    """
    finite = units[np.isfinite(units)]
    return pd.Series(
        {
            "series": units.shape[0],
            "stores": series_index["store_id"].nunique(),
            "items": series_index["item_id"].nunique(),
            "days": units.shape[1],
            "cells": units.size,
            "nan_cells": int(np.isnan(units).sum()),
            "negative_cells": int((finite < 0).sum()),
            "non_integer_cells": int((finite != np.round(finite)).sum()),
            "zero_cells": int((finite == 0).sum()),
            "zero_pct": round(float((finite == 0).mean() * 100), 2),
            "total_units": float(finite.sum()),
        }
    )


def mask_summary(
    mask_reason: np.ndarray, units_raw: np.ndarray, series_index: pd.DataFrame
) -> pd.DataFrame:
    """Resume cuántas celdas, series y artículos enmascara cada regla y cuántas unidades quita.

    Sirve para auditar el efecto de las máscaras: por ejemplo, ver que `pre_launch` concentra
    la mayoría de las celdas en NaN y que los rechazos son pocos y explicables.

    Parámetros:
        mask_reason (np.ndarray): códigos por celda (salida de `apply_masks`).
        units_raw (np.ndarray): unidades crudas, antes de enmascarar.
        series_index (pd.DataFrame): orden de las series (`store_id, item_id`).

    Retorno:
        pd.DataFrame: una fila por código con `code, reason, is_rejection, cells, pct_cells,
        series, items, units_removed` (unidades crudas finitas que quedan en NaN).
    """
    items = series_index["item_id"].to_numpy()
    raw = np.nan_to_num(units_raw, nan=0.0, posinf=0.0, neginf=0.0)
    rows = []
    for code, reason in MASK_REASONS.items():
        hit = mask_reason == code
        series_hit = hit.any(axis=1)
        rows.append(
            {
                "code": code,
                "reason": reason,
                "is_rejection": code in REJECTION_CODES,
                "cells": int(hit.sum()),
                "pct_cells": float(hit.mean() * 100),
                "series": int(series_hit.sum()),
                "items": int(pd.unique(items[series_hit]).size),
                "units_removed": float(raw[hit].sum()) if code else 0.0,
            }
        )
    return pd.DataFrame(rows)


def audit_series(
    result: dict,
    store_id: str,
    item_id: str,
    start_d: int | None = None,
    end_d: int | None = None,
) -> pd.DataFrame:
    """Muestra día a día una serie tienda-artículo antes y después de las máscaras.

    Permite comprobar a mano que cada regla hace lo que dice, por ejemplo alrededor del
    lanzamiento o en una semana sin precio.

    Parámetros:
        result (dict): salida de `run_data_validation`.
        store_id (str): tienda.
        item_id (str): artículo.
        start_d (int | None): primer día (base 1); por defecto, el primero.
        end_d (int | None): último día (inclusivo); por defecto, el último.

    Retorno:
        pd.DataFrame: `d, date, wm_yr_wk, units_raw, price_raw, mask_reason, units, price`.
        Lanza KeyError si la serie no existe.
    """
    si = result["series_index"]
    hits = np.flatnonzero((si["store_id"] == store_id).to_numpy() & (si["item_id"] == item_id))
    if not len(hits):
        raise KeyError(f"No existe la serie {store_id}/{item_id}.")
    i = int(hits[0])
    n_days = result["units"].shape[1]
    a, b = (start_d or 1), (end_d or n_days)
    cols = slice(a - 1, b)
    cal = result["calendar"].assign(d_num=lambda c: c["d"].str.removeprefix("d_").astype(int))
    dates = cal.set_index("d_num")["date"].reindex(range(a, b + 1)).to_numpy()
    codes = result["mask_reason"][i, cols]
    return pd.DataFrame(
        {
            "d": np.arange(a, b + 1),
            "date": dates,
            "store_id": store_id,
            "item_id": item_id,
            "wm_yr_wk": result["week_of_day"][cols],
            "units_raw": result["units_raw"][i, cols],
            "price_raw": result["prices_daily_raw"][i, cols],
            "mask_reason": [MASK_REASONS[int(c)] for c in codes],
            "units": result["units"][i, cols],
            "price": result["prices_daily"][i, cols],
        }
    )


def save_data_validation(result: dict) -> list[Path]:
    """Guarda en `data/interim/` lo que leen las etapas siguientes.

    Escribe las unidades y precios enmascarados, `week_of_day`, el agregado por artículo, el
    total de la cadena, `series_index`, `launch`, `rejections` y `mask_reason`. No guarda los
    arreglos crudos: esos se releen desde `data/raw/` (FR-001).

    Parámetros:
        result (dict): salida de `run_data_validation`.

    Retorno:
        list[Path]: archivos escritos.
    """
    arrays, tables = interim_from_aggregate(result["agg"])
    arrays.update(
        {
            "units": result["units"],
            "prices_daily": result["prices_daily"],
            "week_of_day": result["week_of_day"],
            "mask_reason": result["mask_reason"],
        }
    )
    tables.update(
        {
            "series_index": result["series_index"],
            "launch": result["launch"],
            "rejections": result["rejections"],
        }
    )
    save_interim(arrays, tables)
    return [INTERIM_DIR / f"{n}.npy" for n in arrays] + [
        INTERIM_DIR / f"{n}.parquet" for n in tables
    ]


def run_data_validation(cfg: dict, persist: bool = False) -> dict:
    """Etapa 1: lee los CSV de M5, valida, enmascara datos inválidos y agrega por artículo.

    Qué hace, en orden:
        1. Lee y valida el calendario (fechas continuas, un `wm_yr_wk` por día); si falla, se
           detiene (FR-003).
        2. Lee las ventas como arreglo ancho (serie tienda-artículo × día) y los precios
           semanales, y expande el precio a cada día de su semana.
        3. Calcula el lanzamiento de cada serie: la primera semana con precio (FR-004).
        4. Enmascara (pone en NaN) prelanzamiento, unidades inválidas y semanas sin precio
           válido, y registra cada rechazo con motivo, fila y fuente (FR-002).
        5. Agrega por artículo (suma de tiendas) las unidades y el gasto (unidades × precio por
           tienda-día), y calcula el total diario de la cadena.

    Uso:
        - Exploración en el notebook (`persist=False`, por defecto): no escribe nada. Devuelve
          los datos crudos y los enmascarados para comparar, contar el efecto de cada regla con
          `mask_summary` y revisar series con `audit_series`. Al terminar la revisión, guardar
          con `save_data_validation(result)`.
        - Proceso automático (`persist=True`): ejecuta y guarda en `data/interim/` en un paso.
        Las etapas siguientes solo leen `data/interim/`, así que alguna de las dos formas de
        guardar debe correr antes de la etapa 3.

    Cómo leer la salida:
        - NaN en `units` o `prices_daily` significa "celda excluida", no cero; el motivo está en
          `mask_reason` (códigos de `MASK_REASONS`). Un 0 en `units` es una venta real de cero.
        - `rejections` tiene una fila por dato rechazado; `pre_launch` no aparece porque no es
          un error, y `price_gap_no_sales` aparece solo como aviso.
        - En `agg`, la fila k de cada arreglo es el artículo `item_ids[k]` y la columna j es el
          día d = j + 1. Un día sin ninguna tienda válida queda NaN.

    Memoria: con M5 completo, cada arreglo serie × día ocupa ~240 MB (float32); la salida
    guarda cuatro (crudos y enmascarados) más `mask_reason` (~60 MB).

    Parámetros:
        cfg (dict): configuración validada.
        persist (bool): si es True, guarda los intermedios con `save_data_validation`.

    Retorno:
        dict con:
            - `calendar` (pd.DataFrame): calendario validado.
            - `week_of_day` (np.ndarray): `wm_yr_wk` de cada día 1..last_day.
            - `prices_raw` (pd.DataFrame): precios semanales tal como se leyeron, con `csv_row`.
            - `series_index` (pd.DataFrame): `store_id, item_id, csv_row` de cada fila de los
              arreglos (ordenado por tienda y artículo; `csv_row` es la fila en el CSV).
            - `units_raw`, `prices_daily_raw` (np.ndarray): unidades y precios diarios antes de
              enmascarar.
            - `units`, `prices_daily` (np.ndarray): los mismos arreglos enmascarados.
            - `mask_reason` (np.ndarray int8): motivo de cada celda (0 = válida).
            - `launch` (pd.DataFrame): `store_id, item_id, launch_wk, launch_d` por serie.
            - `rejections` (pd.DataFrame): `reason, source, row_id, column, detail`.
            - `agg` (dict): `item_ids`, `item_units`, `item_spend`, `item_launch_d` y
              `chain_total` (salida de `aggregate_items`).
            - `persisted` (list[Path]): archivos escritos; vacío si `persist=False`.
    """
    last_day = cfg["data"]["last_day"]
    calendar = load_calendar(cfg)
    validate_calendar(calendar, last_day)
    wod = week_of_day(calendar, last_day)
    units_raw, series_index = load_sales(cfg)
    prices_raw = load_prices(cfg)
    prices_daily_raw = build_daily_prices(prices_raw, wod, series_index)
    launch = compute_launch(prices_daily_raw, series_index, wod)
    units, prices_daily, rejections, mask_reason = apply_masks(
        units_raw,
        prices_daily_raw,
        launch,
        series_index,
        wod,
        prices_raw,
        cfg["data"]["sales_file"],
    )
    result = {
        "calendar": calendar,
        "week_of_day": wod,
        "prices_raw": prices_raw,
        "series_index": series_index,
        "units_raw": units_raw,
        "prices_daily_raw": prices_daily_raw,
        "units": units,
        "prices_daily": prices_daily,
        "mask_reason": mask_reason,
        "launch": launch,
        "rejections": rejections,
        "agg": aggregate_items(units, prices_daily, series_index, launch),
    }
    result["persisted"] = save_data_validation(result) if persist else []
    return result
