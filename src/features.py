"""
Construcción de las series y variables del nb 04 (etapas 2 a 9).

El artículo es `item_id` sumado sobre las 10 tiendas. Lo congelado en el nb 03 se carga y no se
recalcula (Principio VI).
"""

import numpy as np
import pandas as pd

BLOCK_DAYS = 28  # bloque de pronóstico (constitución 3.0.0, Principio I)

TARGET_COLUMNS = ["cell", "abc_class", "xyz_class", "cv_train", "long_run", "min_days", "treatment"]


def flag_targets(item_ids, targets: pd.DataFrame) -> pd.DataFrame:
    """Une las clases y reglas congeladas de `targets` a la lista completa de artículos.

    Parámetros:
        item_ids (array-like): todos los `item_id` del catálogo.
        targets (pd.DataFrame): artefacto congelado `targets` del nb 03.

    Retorno:
        pd.DataFrame: una fila por artículo con `item_id`, `is_target` y las columnas de
        `TARGET_COLUMNS` (nulas en los artículos que no se modelan). Lanza ValueError si algún
        artículo de `targets` no está en el catálogo o está repetido.
    """
    if targets["item_id"].duplicated().any():
        raise ValueError("targets tiene item_id repetidos.")
    catalog = pd.DataFrame({"item_id": pd.unique(pd.Series(item_ids, dtype=str))})
    missing = set(targets["item_id"]) - set(catalog["item_id"])
    if missing:
        raise ValueError(f"{len(missing)} artículos de targets no están en el catálogo.")
    out = catalog.merge(targets[["item_id", *TARGET_COLUMNS]], on="item_id", how="left")
    # Enteros con nulos: los excluidos no tienen reglas, y un float ocultaría que son días.
    out[["long_run", "min_days"]] = out[["long_run", "min_days"]].astype("Int64")
    out.insert(1, "is_target", out["item_id"].isin(targets["item_id"]))
    return out



def aggregate_blocks(
    units: pd.DataFrame,
    chain_total: np.ndarray,
    bounds: pd.DataFrame,
    dates: pd.Series,
) -> pd.DataFrame:
    """Suma diaria → bloques de 28 días por artículo, con la participación en la cadena.

    El valor del bloque es la suma de sus unidades: los días NaN (quiebres enmascarados o fuera
    del ciclo de vida) no restan ni anulan el bloque, y no se exige un mínimo de días observados.
    Un bloque sin ningún día con dato (entero dentro de un quiebre) queda en NaN: no es una venta
    de 0 unidades, sino un bloque no observado (se completa con `fill_invalid_blocks`).
    La serie de cada artículo va del bloque de su primera venta al de su última venta, aunque
    cualquiera de los dos tenga un solo día con venta (constitución 4.0.0, Principio II).

    Parámetros:
        units (pd.DataFrame): unidades artículo × día (`d_<día>`), contiguas y en múltiplos de 28
            días a partir del primer bloque.
        chain_total (np.ndarray): unidades totales de la cadena por día, desde d_1.
        bounds (pd.DataFrame): `first_sale_d` y `last_sale_d` por `item_id`.
        dates (pd.Series): día (int) → fecha, para rotular cada bloque.

    Retorno:
        pd.DataFrame: una fila por artículo y bloque con `item_id`, `block` (1, 2, … desde el
        primer día de `units`), `start_d`, `end_d`, `start_date`, `end_date`, `days_with_data`,
        `units` (NaN si el bloque no tiene días con dato) y `chain_units`.
    """
    days = np.array([int(c[2:]) for c in units.columns])
    if len(days) % BLOCK_DAYS or (np.diff(days) != 1).any():
        raise ValueError("los días deben ser contiguos y un múltiplo de 28.")
    n_items, n_blocks = len(units), len(days) // BLOCK_DAYS
    u = units.to_numpy(dtype=float).reshape(n_items, n_blocks, BLOCK_DAYS)
    starts = days[::BLOCK_DAYS]
    chain = chain_total[days - 1].reshape(n_blocks, BLOCK_DAYS).sum(axis=1)
    out = pd.DataFrame({
        "item_id": np.repeat(units.index.to_numpy(dtype=object), n_blocks),
        "block": np.tile(np.arange(1, n_blocks + 1), n_items),
        "start_d": np.tile(starts, n_items),
        "days_with_data": (~np.isnan(u)).sum(axis=2).ravel(),
        "units": np.nansum(u, axis=2).ravel(),
        "chain_units": np.tile(chain, n_items),
    })
    out["end_d"] = out["start_d"] + BLOCK_DAYS - 1
    first = out["item_id"].map(bounds["first_sale_d"])
    last = out["item_id"].map(bounds["last_sale_d"])
    if first.isna().any() or last.isna().any():
        raise ValueError("hay artículos sin primera o última venta.")
    out = out[(out["end_d"] >= first) & (out["start_d"] <= last)].reset_index(drop=True)
    out["start_date"] = out["start_d"].map(dates)
    out["end_date"] = out["end_d"].map(dates)
    out.loc[out["days_with_data"] == 0, "units"] = np.nan
    cols = ["item_id", "block", "start_d", "end_d", "start_date", "end_date", "days_with_data",
            "units", "chain_units"]
    return out[cols]


def assign_windows(
    blocks: pd.DataFrame, test_blocks: int, val_blocks: int, min_blocks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición por artículo: test = últimos bloques del ciclo de vida; el resto es train.

    Constitución 4.0.0 (Principio VI): todos los artículos tienen un test de igual longitud aunque
    su ciclo de vida termine antes del último bloque. Los `val_blocks` bloques de train anteriores
    al test son la ventana de backtesting: cada uno se predice una vez en su fold y luego pasa a
    ser dato de entrenamiento, por eso siguen siendo train. Un artículo con menos de `min_blocks`
    bloques no alcanza para entrenar, hacer backtesting y probar, y se excluye.

    Parámetros:
        blocks (pd.DataFrame): salida de `aggregate_blocks` (bloques contiguos por artículo).
        test_blocks (int): bloques de test por artículo.
        val_blocks (int): bloques de la ventana de backtesting por artículo (dentro de train).
        min_blocks (int): bloques mínimos de ciclo de vida para modelar el artículo.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]: `blocks` de los artículos que se modelan con las
        columnas `pos` (1, 2, … dentro del artículo), `split` (`train` o `test`) y `backtest`
        (True en los bloques de train de la ventana de backtesting), y `excluded` con `item_id`,
        `n_blocks` y `reason`.
    """
    if min_blocks <= test_blocks + val_blocks:
        raise ValueError("min_blocks debe dejar al menos un bloque de train.")
    out = blocks.sort_values(["item_id", "block"]).copy()
    out["pos"] = out.groupby("item_id").cumcount() + 1
    n = out.groupby("item_id")["pos"].transform("max")
    n_items = out.groupby("item_id")["pos"].max()
    short = n_items[n_items < min_blocks]
    excluded = pd.DataFrame({"item_id": short.index, "n_blocks": short.to_numpy(),
                             "reason": f"menos de {min_blocks} bloques de ciclo de vida"})
    from_end = n - out["pos"]  # 0 = último bloque del artículo
    out["split"] = np.where(from_end < test_blocks, "test", "train")
    out["backtest"] = (from_end >= test_blocks) & (from_end < test_blocks + val_blocks)
    out = out[~out["item_id"].isin(short.index)].reset_index(drop=True)
    return out, excluded.reset_index(drop=True)


EVENT_TYPES = ["Religious", "National", "Cultural", "Sporting"]


def block_calendar(calendar: pd.DataFrame, start_d: int, n_blocks: int) -> pd.DataFrame:
    """Variables de calendario de cada bloque de 28 días: año, periodo del año y eventos.

    El año y el periodo se cuentan desde el primer bloque (d_122): 13 bloques de 28 días = 364
    días = 52 semanas, así que el periodo p cae en las mismas semanas cada año (el periodo 1
    empieza a fines de mayo). Los eventos de M5 vienen en dos columnas por día
    (`event_*_1` y `event_*_2`); la segunda solo se usa cuando un día tiene dos eventos.

    Parámetros:
        calendar (pd.DataFrame): calendario de M5 (`d`, `event_name_1/2`, `event_type_1/2`).
        start_d (int): primer día del bloque 1.
        n_blocks (int): cantidad de bloques.

    Retorno:
        pd.DataFrame: una fila por bloque con `block`, `year` (1, 2, …), `period` (1–13),
        `n_events` (eventos en el bloque, sumando ambas columnas), `has_event`, `has_event_1`,
        `has_event_2` (0/1: hay algún evento, alguno en la columna 1, alguno en la columna 2) y
        `n_<tipo>` (eventos de cada tipo de `EVENT_TYPES`).
    """
    day = calendar["d"].str.removeprefix("d_").astype(int)
    end_d = start_d + BLOCK_DAYS * n_blocks - 1
    cal = calendar[(day >= start_d) & (day <= end_d)].assign(
        block=(day - start_d) // BLOCK_DAYS + 1
    )
    g = cal.groupby("block")
    out = pd.DataFrame({
        "n_event_1": g["event_name_1"].count(),
        "n_event_2": g["event_name_2"].count(),
    })
    for t in EVENT_TYPES:
        hits = (cal["event_type_1"] == t).astype(int) + (cal["event_type_2"] == t).astype(int)
        out[f"n_{t.lower()}"] = hits.groupby(cal["block"]).sum()
    out = out.reindex(range(1, n_blocks + 1), fill_value=0).rename_axis("block").reset_index()
    out.insert(1, "year", (out["block"] - 1) // 13 + 1)
    out.insert(2, "period", (out["block"] - 1) % 13 + 1)
    out.insert(3, "n_events", out["n_event_1"] + out["n_event_2"])
    out.insert(4, "has_event", (out["n_events"] > 0).astype(int))
    out.insert(5, "has_event_1", (out["n_event_1"] > 0).astype(int))
    out.insert(6, "has_event_2", (out["n_event_2"] > 0).astype(int))
    return out.drop(columns=["n_event_1", "n_event_2"])


def calendar_lags(calendar: pd.DataFrame, start_d: int, n_blocks: int, cols: list[str],
                  lags: list[int]) -> pd.DataFrame:
    """Rezagos de variables de calendario por bloque: `<col>_lag<k>` = valor k bloques antes.

    El calendario es el mismo para todos los artículos y se conoce antes del inicio de cada
    serie, así que el rezago se toma del calendario y no de la serie del artículo: el primer
    bloque de un artículo tiene sus rezagos completos. Para los primeros bloques se usan los días
    anteriores a `start_d`, que el calendario de M5 trae.

    Parámetros:
        calendar (pd.DataFrame): calendario de M5 (ver `block_calendar`).
        start_d (int): primer día del bloque 1.
        n_blocks (int): cantidad de bloques.
        cols (list[str]): columnas de `block_calendar` a rezagar.
        lags (list[int]): rezagos en bloques (≥ 1).

    Retorno:
        pd.DataFrame: una fila por bloque (1 a `n_blocks`) con `block` y las columnas rezagadas.
        Lanza ValueError si el calendario no cubre los días anteriores que hacen falta.
    """
    max_lag = max(lags)
    ext_start = start_d - BLOCK_DAYS * max_lag
    if calendar["d"].str.removeprefix("d_").astype(int).min() > ext_start:
        raise ValueError("el calendario no cubre los bloques anteriores al bloque 1.")
    ext = block_calendar(calendar, ext_start, n_blocks + max_lag)
    ext["block"] -= max_lag
    lagged = {f"{c}_lag{k}": ext[c].shift(k) for c in cols for k in lags}
    out = pd.concat([ext[["block"]], pd.DataFrame(lagged)], axis=1)
    return out[out["block"] >= 1].astype(int).reset_index(drop=True)


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """Agrega los rezagos de `cols` por artículo: `<col>_lag<k>` = valor k bloques antes.

    Con h = 1, todo rezago ≥ 1 usa solo bloques ya observados en el origen. Los bloques de cada
    artículo deben ser contiguos (sin huecos), así un desplazamiento de k filas es k bloques.

    Parámetros:
        df (pd.DataFrame): una fila por artículo y bloque, con `item_id` y `block`.
        cols (list[str]): columnas a rezagar.
        lags (list[int]): rezagos en bloques (≥ 1).

    Retorno:
        pd.DataFrame: copia de `df` ordenada por artículo y bloque, con las columnas nuevas (NaN
        en los primeros bloques de cada artículo).
    """
    out = df.sort_values(["item_id", "block"]).reset_index(drop=True)
    gaps = out.groupby("item_id")["block"].diff().dropna()
    if not (gaps == 1).all():
        raise ValueError("los bloques de cada artículo deben ser contiguos.")
    g = out.groupby("item_id", sort=False)
    for col in cols:
        for k in lags:
            out[f"{col}_lag{k}"] = g[col].shift(k)
    return out


def add_rolling(df: pd.DataFrame, col: str, windows: list[int], stats: list[str],
                treated_col: str | None = None) -> pd.DataFrame:
    """Agrega estadísticos móviles de `col` sobre los k bloques ANTERIORES de cada artículo.

    Se desplaza un bloque antes de la ventana (shift(1) antes de rolling), así el bloque del
    registro nunca entra en su propia ventana. La ventana exige sus k bloques anteriores: un
    estadístico de menos bloques no significa lo mismo, y los nulos de arranque los manejan las
    librerías de boosting. Un bloque tratado (`treated_col`, de `treat_ratio`) cuenta como
    historia, pero su valor (NaN) se omite del estadístico (decisión del usuario, nb 04, 6a): el
    estadístico queda NaN solo si todos los bloques de la ventana están tratados (la desviación
    estándar necesita 2 valores). Sin bloques tratados, el resultado es el de una ventana completa.

    Parámetros:
        df (pd.DataFrame): una fila por artículo y bloque (contiguos), con `item_id` y `block`.
        col (str): columna objetivo.
        windows (list[int]): tamaños de ventana en bloques.
        stats (list[str]): subconjunto de `mean`, `std`, `median`, `range` (máximo − mínimo).
        treated_col (str | None): columna booleana de bloques tratados; None si no hay tratamiento.

    Retorno:
        pd.DataFrame: copia de `df` con las columnas `<col>_roll<k>_<stat>`.
    """
    out = df.sort_values(["item_id", "block"]).reset_index(drop=True)
    gaps = out.groupby("item_id")["block"].diff().dropna()
    if not (gaps == 1).all():
        raise ValueError("los bloques de cada artículo deben ser contiguos.")
    by_item = out.groupby("item_id", sort=False)
    prev = by_item[col].shift(1)
    g = prev.groupby(out["item_id"], sort=False)
    # Bloques que cuentan como historia: con valor, o tratados (vacíos a propósito).
    history = out[col].notna() if treated_col is None else out[col].notna() | out[treated_col].astype(bool)
    prev_history = history.groupby(out["item_id"], sort=False).shift(1, fill_value=False).astype(float)
    h = prev_history.groupby(out["item_id"], sort=False)
    for k in windows:
        r = g.rolling(k, min_periods=1)
        full = h.rolling(k, min_periods=k).sum().reset_index(level=0, drop=True).sort_index() == k
        values = {
            "mean": r.mean(), "std": r.std(), "median": r.median(),
            "range": r.max() - r.min() if "range" in stats else None,
        }
        for stat in stats:
            v = values[stat].reset_index(level=0, drop=True).sort_index()
            out[f"{col}_roll{k}_{stat}"] = v.where(full)
    return out


def fill_invalid_blocks(blocks: pd.DataFrame, min_days: pd.Series, low_share: float,
                        window: int) -> pd.DataFrame:
    """Completa los bloques no válidos con el valor del bloque anterior del mismo artículo.

    Un bloque es no válido por dos motivos (constitución, Principio II):
    - `min_days`: tiene menos días observados que el `min_days` congelado del artículo (incluye
      los bloques sin ningún día con dato): casi todo el bloque cae en un quiebre.
    - `low_units`: pasó `min_days`, pero sus unidades no llegan a `low_share` × su referencia:
      un quiebre que las ventas sueltas de unas pocas unidades escondieron (decisión del usuario).
      La referencia es la mediana de sus últimos `window` bloques válidos; en el primer año
      (primeros `window` bloques) es la mediana de los bloques válidos de ese primer año, que
      siempre quedan antes de la ventana de backtesting. El bloque 1 no se evalúa, para no
      confundir un lanzamiento con un quiebre.

    El bloque no válido toma el valor del bloque anterior (forward fill): su ratio vale 1 ("no
    cambió") y el bloque siguiente se compara contra el último bloque válido. Si los primeros
    bloques no son válidos no hay valor que arrastrar: se eliminan y la serie empieza en el primer
    bloque válido.

    Parámetros:
        blocks (pd.DataFrame): bloques con `item_id`, `block`, `units` y `days_with_data`.
        min_days (pd.Series): `min_days` congelado por artículo (índice `item_id`, nb 03).
        low_share (float): fracción de la referencia bajo la cual un bloque es no válido.
        window (int): bloques válidos de la referencia (y largo del primer año).

    Retorno:
        pd.DataFrame: copia ordenada por artículo y bloque, sin los bloques iniciales no válidos,
        con `units` completadas, `imputed` (bloque no válido que tomó el valor del anterior) e
        `invalid_reason` (`min_days`, `low_units` o vacío).
        Lanza ValueError si falta el `min_days` de algún artículo.
    """
    out = blocks.sort_values(["item_id", "block"]).reset_index(drop=True)
    threshold = out["item_id"].map(min_days)
    if threshold.isna().any():
        raise ValueError("falta min_days para algún artículo.")
    by_days = (out["days_with_data"] < threshold.astype(int)).to_numpy()
    units = out["units"].to_numpy(dtype=float)
    low = np.zeros(len(out), dtype=bool)
    for idx in out.groupby("item_id", sort=False).indices.values():
        u, bad = units[idx], by_days[idx]
        first_year = u[:window][~bad[:window]]
        year_ref = np.median(first_year) if len(first_year) else np.nan
        history = []  # unidades de los bloques válidos ya vistos, en orden
        for pos in range(len(idx)):
            if bad[pos]:
                continue
            if pos > 0:
                ref = year_ref if pos < window else (np.median(history[-window:]) if history else np.nan)
                if ref > 0 and u[pos] < low_share * ref:
                    low[idx[pos]] = True
                    continue
            history.append(u[pos])
    out["invalid_reason"] = np.where(by_days, "min_days", np.where(low, "low_units", ""))
    out["imputed"] = by_days | low
    out["units"] = out["units"].mask(out["imputed"]).groupby(out["item_id"], sort=False).ffill()
    return out[out["units"].notna()].reset_index(drop=True)


def drop_partial_first_block(blocks: pd.DataFrame, min_blocks: int,
                             full_days: int = BLOCK_DAYS) -> pd.DataFrame:
    """Quita el primer bloque de cada artículo si está incompleto (decisión del usuario, nb 04, 4.2a).

    El primer bloque no tiene bloque anterior del cual tomar valor (4.2), y si tiene menos de
    `full_days` días con datos (lanzamiento a mitad de bloque o días enmascarados) vende una
    fracción de su nivel: el ratio del bloque siguiente sale inflado. La serie empieza entonces en
    el siguiente bloque válido: si los bloques que siguen eran no válidos (4.2), copiaron el valor
    del bloque incompleto y también se quitan, como en la 4.2. Excepción: si al quitarlos el
    artículo quedaría con menos de `min_blocks` bloques (mínimo de la partición, 4.3), se
    conservan, para no sacarlo de la modelación.

    Parámetros:
        blocks (pd.DataFrame): salida de `fill_invalid_blocks` (bloques contiguos por artículo,
            con `days_with_data`).
        min_blocks (int): bloques mínimos de ciclo de vida para modelar el artículo.
        full_days (int): días de un bloque completo.

    Retorno:
        pd.DataFrame: copia ordenada sin los primeros bloques incompletos.
    """
    out = blocks.sort_values(["item_id", "block"]).reset_index(drop=True)
    g = out.groupby("item_id")
    pos = g.cumcount()
    partial = g["days_with_data"].transform("first") < full_days
    # Nuevo inicio: primer bloque válido después del primero.
    start = pos.where((pos >= 1) & ~out["imputed"]).groupby(out["item_id"]).transform("min")
    keeps_min = g["block"].transform("size") - start >= min_blocks
    drop = partial & keeps_min & (pos < start)
    return out[~drop].reset_index(drop=True)


def fill_atypical_blocks(blocks: pd.DataFrame, low_level: float, window: int, min_ref: int,
                         years: int, period: int = 13, level_masked: float | None = None,
                         pattern_years: int = 1, refill_level: float | None = None,
                         full_days: int = BLOCK_DAYS) -> pd.DataFrame:
    """Completa los bloques atípicos: válidos para la 4.2, pero con una caída que el artículo no repite.

    Decisión del usuario (nb 04, 4.2, tras revisar la Etapa 11): un bloque que pasó `min_days` y
    `low_units` puede seguir deprimido (quiebre parcial o caída puntual) y su ratio siguiente sale
    inflado. Se trata aguas arriba, en las unidades, para que el ratio, los rezagos, las ventanas y
    el seasonal naive lo reflejen. Por artículo y en orden:
    - **Nivel** del bloque = unidades ÷ mediana de sus últimos `window` bloques válidos (unidades
      observadas; exige `min_ref` bloques). El bloque 1 no se evalúa.
    - **Deprimido:** nivel < `low_level`. Se mira el nivel del mismo periodo en los `years` años
      anteriores (bloques válidos, cada `period` bloques):
      - si en alguno también estuvo deprimido, es un patrón del artículo y no se toca;
      - si en ninguno lo estuvo, es **atípico** (`atypical`): toma mediana reciente × factor,
        con factor = mediana de los niveles de ese periodo en los años anteriores disponibles,
        con tope 1 (opción 1, decisión del usuario). La mediana reciente trae el nivel actual; la
        mediana de los años evita que un solo año raro mande, y el tope impide que el valor
        completado supere el nivel actual y cree un salto en el ratio;
      - si no hay año anterior (primer año), `atypical_first_year`: toma el valor del bloque
        anterior, como la regla de la 4.2.
    - La referencia sigue con las unidades observadas, también las del bloque atípico: un bloque
      aislado casi no la mueve y una caída persistente la baja, así que la regla se detiene sola.
    - Los bloques no válidos de la 4.2 (`min_days`, `low_units`) vuelven a tomar el valor del
      bloque anterior ya corregido.
    Solo usa bloques anteriores al evaluado: no hay fuga.

    Extensiones causales (nb 04, 4.4, opción B+; por defecto apagadas, reproducen la 4.2b):
    - `level_masked` (E1): umbral de deprimido para bloques con días enmascarados (< `full_days`
      días con datos); un quiebre parcial deja el bloque entre 50 y 80 % de su nivel. El patrón de
      años anteriores se sigue juzgando con `low_level`.
    - `pattern_years` (E7): años anteriores deprimidos que hacen falta para llamar patrón a la
      caída (con menos años disponibles, todos). Con 1, basta un año.
    - `refill_level` (E8): un bloque no válido de la 4.2 que sigue a un bloque válido sin corregir
      con nivel < `refill_level` toma la mediana reciente en vez de copiar ese valor bajo.

    Parámetros:
        blocks (pd.DataFrame): salida de `fill_invalid_blocks` (con `days_with_data`).
        low_level (float): nivel bajo el cual un bloque está deprimido (0–1).
        window (int): bloques válidos de la referencia.
        min_ref (int): mínimo de bloques válidos para evaluar un bloque.
        years (int): años anteriores que se miran.
        period (int): bloques por año.
        level_masked (float | None): umbral para bloques con días enmascarados (E1).
        pattern_years (int): años deprimidos necesarios para considerar patrón (E7).
        refill_level (float | None): nivel del bloque anterior bajo el cual el no válido toma la
            mediana reciente (E8).
        full_days (int): días de un bloque completo.

    Retorno:
        pd.DataFrame: copia ordenada con `units` corregidas, `units_observed` (las de entrada),
        `invalid_reason` (agrega `atypical` y `atypical_first_year`) e `imputed` actualizado.
    """
    if not 0 < low_level < 1 or min(window, min_ref, years, period) < 1 or min_ref > window:
        raise ValueError("parámetros fuera de rango.")
    if level_masked is not None and not low_level <= level_masked < 1:
        raise ValueError("level_masked debe estar entre low_level y 1.")
    if not 1 <= pattern_years <= years or (refill_level is not None and not 0 < refill_level <= 1):
        raise ValueError("pattern_years o refill_level fuera de rango.")
    out = blocks.sort_values(["item_id", "block"]).reset_index(drop=True)
    units = out["units"].to_numpy(dtype=float).copy()
    reason = out["invalid_reason"].to_numpy(dtype=object).copy()
    invalid = reason != ""
    days = out["days_with_data"].to_numpy() if level_masked is not None else None
    block_num = out["block"].to_numpy()
    level = np.full(len(out), np.nan)
    for idx in out.groupby("item_id", sort=False).indices.values():
        pos_of = {b: i for i, b in enumerate(block_num[idx])}
        history = []
        for i, j in enumerate(idx):
            if invalid[j]:
                prev = idx[i - 1]
                if (refill_level is not None and len(history) >= min_ref and reason[prev] == ""
                        and level[prev] < refill_level):
                    units[j] = np.median(history[-window:])
                else:
                    units[j] = units[prev]
                continue
            ref = np.median(history[-window:]) if len(history) >= min_ref else np.nan
            lv = units[j] / ref if ref > 0 else np.nan
            level[j] = lv
            observed = units[j]
            history.append(observed)
            threshold = level_masked if (days is not None and days[j] < full_days) else low_level
            if i == 0 or not lv < threshold:
                continue
            prior = [level[idx[pos_of[b]]] for b in (block_num[j] - k * period for k in range(1, years + 1))
                     if b in pos_of and not np.isnan(level[idx[pos_of[b]]])]
            if not prior:
                units[j], reason[j] = units[idx[i - 1]], "atypical_first_year"
            elif sum(pl < low_level for pl in prior) < min(pattern_years, len(prior)):
                units[j], reason[j] = ref * min(float(np.median(prior)), 1.0), "atypical"
    out["units_observed"] = out["units"]
    out["units"] = units
    out["invalid_reason"] = reason
    out["imputed"] = reason != ""
    return out


COMPLETED_REASONS = ("min_days", "low_units", "atypical", "atypical_first_year")


def classify_extreme_ratios(blocks: pd.DataFrame, depressed: float, low: float = 0.8, high: float = 1.25,
                            masked_days: int = 14, start_blocks: int = 7, extreme: tuple = (0.5, 2.0),
                            window: int = 6, min_ref: int = 3, full_days: int = BLOCK_DAYS) -> pd.DataFrame:
    """Clasifica cada ratio extremo (< 0,5 o > 2) por su causa (nb 04, 4.4; diagnóstico por segmento).

    Compara t−1, t y t+1 con el nivel previo del artículo (mediana de los `window` bloques
    anteriores a t−1, mínimo `min_ref`) y con un nivel antiguo (mediana de t−26 a t−8, mínimo 6
    bloques: el año anterior a la referencia). Grupos (`group`):
    - **arranque:** posición ≤ `start_blocks` o sin referencia.
    - **estructural:** el nivel cambia y se mantiene en t+1 (subida o bajada); no se toca.
    - **atipico:** el bloque no refleja la demanda (quiebre, pico o caída de un bloque, bloque
      deprimido que la 4.2b no corrigió). `subgroup`: `quiebre`, `pico_caida` o `deprimido`.
    - **propagado:** un bloque completado por la 4.2 o la 4.2b produce el extremo.
    Umbrales: t−1 bajo si < `low` de su nivel; t alto si > `high`; quiebre prolongado si suma
    ≥ `masked_days` días enmascarados en t−6..t−1; `depressed` = umbral de la 4.2b. Usa t+1, así
    que solo sirve para diagnosticar o para corregir la historia (nunca un bloque que se pronostica).

    Parámetros:
        blocks (pd.DataFrame): `blocks_model` (4.3) con `units`, `days_with_data`,
            `invalid_reason` y `pos`.
        depressed (float): `features.atypical_level`.
        low, high (float): niveles de t−1 bajo y t alto.
        masked_days (int): días enmascarados de un quiebre prolongado.
        start_blocks (int): posiciones del arranque.
        extreme (tuple): límites del ratio extremo.
        window, min_ref (int): ventana y mínimo de la referencia.
        full_days (int): días de un bloque completo.

    Retorno:
        pd.DataFrame: una fila por ratio extremo (subidas y luego caídas) con `item_id`, `block`,
        `direction` (`up`/`down`), `group`, `subgroup`, `detail` y `culprit` (bloque a corregir:
        `t`, `t_forward`, `run_back` o `t_minus_1`).
    """
    f = blocks.sort_values(["item_id", "block"]).reset_index(drop=True)
    g = f.groupby("item_id")
    ref = g["units"].transform(lambda s: s.shift(2).rolling(window, min_periods=min_ref).median())
    old = g["units"].transform(lambda s: s.shift(8).rolling(19, min_periods=6).median())
    u_prev, u_next = g["units"].shift(1), g["units"].shift(-1)
    masked = full_days - f["days_with_data"]
    d = pd.DataFrame({
        "item_id": f["item_id"], "block": f["block"], "pos": f["pos"], "units": f["units"],
        "ratio": f["units"] / u_prev, "ref": ref, "old": old, "reason": f["invalid_reason"],
        "lv_prev": u_prev / ref, "lv_t": f["units"] / ref, "lv_next": u_next / ref, "masked": masked,
        "masked_prev": masked.groupby(f["item_id"]).shift(1), "masked_next": masked.groupby(f["item_id"]).shift(-1),
        "masked_before": masked.groupby(f["item_id"]).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()),
        "reason_prev": g["invalid_reason"].shift(1)})

    def up(r):
        if r.pos <= start_blocks or np.isnan(r.ref):
            return "arranque", "arranque", f"bloques 2–{start_blocks}", "t_minus_1"
        den, num = r.lv_prev < low, r.lv_t > high
        side = ("den" if -np.log(r.lv_prev) >= np.log(r.lv_t) else "num") if den and num else (
            "den" if den else "num" if num else None)
        if side is None:
            return "atipico", "pico_caida", "volatilidad (ni t−1 bajo ni t alto)", "t_minus_1"
        if side == "den":
            if r.reason_prev in COMPLETED_REASONS or r.reason == "atypical":
                return "propagado", "propagado", "t−1 o t completado, con t−1 bajo", "t_minus_1"
            if r.masked_prev > 0:
                return "atipico", "quiebre", "t−1 con días de quiebre (pasa min_days)", "t_minus_1"
            if r.lv_prev >= depressed:
                return "atipico", "deprimido", f"t−1 entre {depressed:.0%} y {low:.0%} de su nivel", "t_minus_1"
            return "atipico", "deprimido", "t−1 deprimido que la 4.2b tomó como patrón", "t_minus_1"
        if r.lv_next <= high:
            return "atipico", "pico_caida", "pico de un bloque en t", "t"
        if r.masked_before >= masked_days and not np.isnan(r.old) and r.units <= high * r.old:
            return "atipico", "quiebre", "recuperación tras quiebre prolongado", "run_back"
        return "estructural", "estructural", "subida de nivel que persiste", "t"

    def down(r):
        if r.pos <= start_blocks or np.isnan(r.ref):
            return "arranque", "arranque", f"bloques 2–{start_blocks}", "t"
        low_t, high_prev = r.lv_t < low, r.lv_prev > high
        side = ("t" if -np.log(r.lv_t) >= np.log(r.lv_prev) else "prev") if low_t and high_prev else (
            "t" if low_t else "prev" if high_prev else None)
        if side is None:
            return "atipico", "pico_caida", "volatilidad (ni t bajo ni t−1 alto)", "t_minus_1"
        if side == "prev":
            if r.reason_prev in COMPLETED_REASONS:
                return "propagado", "propagado", "t−1 completado alto, t normal", "t_minus_1"
            return "atipico", "pico_caida", "vuelve tras un pico en t−1", "t_minus_1"
        if r.reason in COMPLETED_REASONS:
            return "propagado", "propagado", "t completado con un valor bajo", "t"
        if np.isnan(r.lv_next):
            return "estructural", "estructural", "caída en el último bloque (sin t+1)", "t"
        if r.lv_next >= low:
            if r.masked > 0:
                return "atipico", "quiebre", "caída de un bloque con días de quiebre", "t"
            return "atipico", "pico_caida", "caída de un bloque sin días de quiebre", "t"
        if r.masked > 0 or r.masked_next > 0:
            return "atipico", "quiebre", "caída que persiste con días de quiebre", "t_forward"
        return "estructural", "estructural", "bajada de nivel que persiste", "t"

    cols = ["group", "subgroup", "detail", "culprit"]
    parts = []
    for direction, mask, fn in [("up", d["ratio"] > extreme[1], up), ("down", d["ratio"] < extreme[0], down)]:
        x = d[mask]
        lab = pd.DataFrame([fn(r) for r in x.itertuples()], columns=cols, index=x.index)
        parts.append(x[["item_id", "block"]].assign(direction=direction).join(lab))
    return pd.concat(parts, ignore_index=True)


def _case_links(cases: pd.DataFrame, reason: pd.Series) -> set:
    """Bloques de cada caso: t, t−1 y la cadena de no válidos (4.2) que copia hacia atrás."""
    out = set()
    for r in cases.itertuples():
        out |= {(r.item_id, r.block), (r.item_id, r.block - 1)}
        j = r.block - 1
        while reason.get((r.item_id, j), "") in ("min_days", "low_units"):
            j -= 1
            out.add((r.item_id, j))
    return out


def impute_history_cases(blocks: pd.DataFrame, cases: pd.DataFrame, window: int = 6, min_ref: int = 3,
                         full_days: int = BLOCK_DAYS) -> pd.DataFrame:
    """Imputa en la historia el bloque culpable de cada caso atípico o propagado (nb 04, 4.4, B+).

    Solo en la historia (train antes de la ventana de backtesting) y solo si t+1 también está en
    ella: ahí mirar t+1 no es fuga, porque todos los folds entrenan con esa historia completa. El
    bloque culpable (`culprit`) toma la mediana de los `window` bloques válidos anteriores al
    primer bloque corregido (mínimo `min_ref`): `t` (pico o caída de un bloque), `t_forward` (t y
    los bloques siguientes con días enmascarados), `run_back` (bloques con quiebre o completados
    antes de una recuperación) o `t_minus_1` (t−1 y la cadena de no válidos que lo copió). Nunca
    toca bloques de un caso estructural (t, t−1 y su cadena).

    Parámetros:
        blocks (pd.DataFrame): `blocks_model` (4.3).
        cases (pd.DataFrame): salida de `classify_extreme_ratios`.
        window, min_ref (int): ventana y mínimo de la mediana.
        full_days (int): días de un bloque completo.

    Retorno:
        pd.DataFrame: copia con `units` imputadas, `invalid_reason` = `case_history` e `imputed`
        en los bloques corregidos.
    """
    out = blocks.sort_values(["item_id", "block"]).set_index(["item_id", "block"])
    u = out["units"].astype(float).copy()
    days, reason = out["days_with_data"], out["invalid_reason"]
    hist = (out["split"] == "train") & ~out["backtest"]
    last_hist = hist[hist].reset_index().groupby("item_id")["block"].max()
    struct = _case_links(cases[cases["group"] == "estructural"], reason)
    fixed = []
    for r in cases[cases["group"].isin(["atipico", "propagado"])].itertuples():
        i, t = r.item_id, r.block
        if t + 1 > last_hist.get(i, -1):
            continue
        if r.culprit == "t":
            bad = [t]
        elif r.culprit == "t_forward":
            bad, j = [t], t + 1
            while j <= last_hist[i] and days.get((i, j), full_days) < full_days:
                bad.append(j)
                j += 1
        elif r.culprit == "run_back":
            bad, j = [], t - 1
            while (i, j) in u.index and (days[(i, j)] < full_days or reason[(i, j)] != ""):
                bad.append(j)
                j -= 1
        else:
            bad, j = [t - 1], t - 1
            while reason.get((i, j), "") in ("min_days", "low_units"):
                j -= 1
                bad.append(j)
        bad = [b for b in bad if (i, b) in u.index and (i, b) not in struct]
        if not bad:
            continue
        start = min(bad)
        prev = [u[(i, b)] for b in range(start - window, start) if (i, b) in u.index and reason[(i, b)] == ""]
        if len(prev) < min_ref:
            continue
        for b in bad:
            u[(i, b)] = float(np.median(prev))
            fixed.append((i, b))
    out["units"] = u
    idx = pd.MultiIndex.from_tuples(sorted(set(fixed)), names=["item_id", "block"])
    out.loc[idx, "invalid_reason"] = "case_history"
    out["imputed"] = out["invalid_reason"] != ""
    return out.reset_index()


def replace_after_history(blocks: pd.DataFrame, causal: pd.DataFrame) -> pd.DataFrame:
    """En la ventana de backtesting y el test, toma las unidades de la regla causal (nb 04, 4.4, B+).

    Ahí no se puede saber si un bloque es un caso sin mirar el bloque que se pronostica, así que
    solo vale una regla que use el pasado: la 4.2b con E1, E7 y E8 (`fill_atypical_blocks`).

    Parámetros:
        blocks (pd.DataFrame): `blocks_model` (4.3), con `split` y `backtest`.
        causal (pd.DataFrame): salida de `fill_atypical_blocks` con las extensiones, sobre la misma
            serie de entrada de la 4.2b.

    Retorno:
        pd.DataFrame: copia con `units`, `invalid_reason` e `imputed` de `causal` en la ventana y el test.
    """
    out = blocks.copy()
    rest = (out["split"] == "test") | out["backtest"]
    src = causal.set_index(["item_id", "block"])[["units", "invalid_reason", "imputed"]]
    key = pd.MultiIndex.from_frame(out.loc[rest, ["item_id", "block"]])
    if not key.isin(src.index).all():
        raise ValueError("la regla causal no cubre todos los bloques de la ventana y el test.")
    for col in ["units", "invalid_reason", "imputed"]:
        out.loc[rest, col] = src.loc[key, col].to_numpy()
    return out


def add_ratio(blocks: pd.DataFrame) -> pd.DataFrame:
    """Agrega el objetivo: `ratio` = unidades del bloque / unidades del bloque anterior.

    Sin transformar: r > 1 si el bloque vendió más que el anterior y r < 1 si vendió menos. El
    pronóstico en unidades es `units_prev` × r̂. Queda en NaN en el primer bloque de cada
    artículo.

    Parámetros:
        blocks (pd.DataFrame): bloques contiguos por artículo con `item_id`, `block` y `units`
            (> 0 donde no son NaN; salida de `fill_invalid_blocks`).

    Retorno:
        pd.DataFrame: copia ordenada por artículo y bloque con `units_prev` y `ratio`.
        Lanza ValueError si hay unidades no positivas o bloques no contiguos.
    """
    out = blocks.sort_values(["item_id", "block"]).reset_index(drop=True)
    if (out["units"].dropna() <= 0).any():
        raise ValueError("las unidades deben ser positivas.")
    gaps = out.groupby("item_id")["block"].diff().dropna()
    if not (gaps == 1).all():
        raise ValueError("los bloques de cada artículo deben ser contiguos.")
    out["units_prev"] = out.groupby("item_id", sort=False)["units"].shift(1)
    out["ratio"] = out["units"] / out["units_prev"]
    return out


DATASET_KEYS = ["item_id", "block", "start_date", "split", "backtest", "imputed", "units", "units_prev",
                "ratio", "ratio_raw", "ratio_treated"]


def treat_ratio(features: pd.DataFrame, max_ratio: float, min_ratio: float | None = None,
                start_blocks: int | None = None) -> pd.DataFrame:
    """Tratamiento del objetivo ANTES del feature engineering: los ratios > `max_ratio` quedan vacíos.

    Decisión del usuario (nb 04, 6a, tras el debate de la Etapa 11): un ratio mayor que
    `max_ratio` casi siempre viene de un denominador deprimido (quiebre parcial, bloque completado,
    arranque) y, con pérdida cuadrática, sesga al alza el pronóstico de las filas normales. Se
    aplica aguas arriba para que los rezagos, las ventanas y los datasets lo reflejen:
    - `ratio` queda en NaN en esas filas: no entra al ajuste de ningún modelo (boosting: se
      entrena con las filas de ratio no nulo; SARIMAX: el filtro de Kalman lo trata como
      faltante), y sus rezagos y ventanas lo ven vacío.
    - No se borra la fila: se pronostica (`units_prev` × r̂) y se mide en unidades reales.
    - `ratio_raw` conserva el ratio sin tratar y `ratio_treated` marca la fila, para identificar y
      reportar la afectación.

    - **Arranque (E11, nb 04, 4.4):** con `min_ratio` y `start_blocks`, también queda vacío el
      ratio < `min_ratio` en las posiciones ≤ `start_blocks`; los primeros bloques no tienen
      historia para imputar y su caída no refleja la demanda.

    Parámetros:
        features (pd.DataFrame): salida de `add_ratio` (con `pos` si se usa `min_ratio`).
        max_ratio (float): mayor ratio que se conserva (`features.max_ratio`), > 1.
        min_ratio (float | None): menor ratio que se conserva en el arranque (0–1).
        start_blocks (int | None): posiciones del arranque.

    Retorno:
        pd.DataFrame: copia con `ratio_raw`, `ratio_treated` y `ratio` tratado.
        Lanza ValueError si `max_ratio` ≤ 1 o si ya se aplicó el tratamiento.
    """
    if not max_ratio > 1:
        raise ValueError("max_ratio debe ser > 1.")
    if "ratio_raw" in features:
        raise ValueError("el tratamiento ya se aplicó.")
    if (min_ratio is None) != (start_blocks is None) or (min_ratio is not None and not (0 < min_ratio < 1 and start_blocks >= 2)):
        raise ValueError("min_ratio (0–1) y start_blocks (≥ 2) van juntos.")
    out = features.copy()
    out["ratio_raw"] = out["ratio"]
    treated = out["ratio_raw"] > max_ratio
    if min_ratio is not None:
        treated |= (out["ratio_raw"] < min_ratio) & (out["pos"] <= start_blocks)
    out["ratio_treated"] = treated
    out["ratio"] = out["ratio_raw"].mask(out["ratio_treated"])
    return out


def interpolate_treated(features: pd.DataFrame, col: str = "ratio", treated_col: str = "ratio_treated") -> pd.DataFrame:
    """Agrega `<col>_plot`: el valor con los tratados interpolados, SOLO para los gráficos.

    En cada artículo, ordenado por bloque, los valores tratados (vacíos en `col`) se reemplazan por
    la interpolación lineal entre el valor válido anterior y el siguiente; así la línea del gráfico
    no se corta. Los no tratados quedan igual, y los vacíos sin un valor válido a cada lado (al
    inicio o al final de la serie) siguen vacíos. No es una variable del modelo: los
    datasets (`boosting_dataset`, `sarimax_dataset`) eligen sus columnas y no la incluyen.

    Parámetros:
        features (pd.DataFrame): filas por artículo y bloque con `item_id`, `block`, `col` y
            `treated_col`.
        col (str): columna a interpolar.
        treated_col (str): marca de los valores tratados.

    Retorno:
        pd.DataFrame: copia con `<col>_plot`.
    """
    out = features.copy()
    order = out.sort_values(["item_id", "block"])
    interp = order.groupby("item_id", sort=False)[col].transform(
        lambda s: s.interpolate(method="linear", limit_area="inside"))
    plot = order[col].where(~order[treated_col].astype(bool), interp)
    out[f"{col}_plot"] = plot.reindex(out.index)
    return out


def sarimax_dataset(features: pd.DataFrame, exog_cols: list[str]) -> pd.DataFrame:
    """Arma el dataset de SARIMAX: la serie del ratio de cada artículo y sus exógenas del bloque.

    SARIMAX no recibe rezagos ni ventanas del objetivo (su parte AR y estacional los cubre), ni
    año ni periodo. Se deja fuera la primera fila de cada artículo, que no tiene bloque anterior.
    Las filas tratadas (`ratio` NaN, `treat_ratio`) se conservan: el filtro de Kalman las trata
    como faltantes en el ajuste y se pronostican igual.

    Parámetros:
        features (pd.DataFrame): salida de la Etapa 6, con `item_id`, `block`, `ratio` y
            `units_prev`.
        exog_cols (list[str]): exógenas conocidas de antemano para el bloque t.

    Retorno:
        pd.DataFrame: una fila por artículo y bloque con identificadores, partición, `units`,
        `units_prev`, `ratio`, `ratio_raw`, `ratio_treated` y las exógenas.
        Lanza ValueError si alguna exógena tiene nulos (SARIMAX no los admite).
    """
    out = features.loc[features["units_prev"].notna(), DATASET_KEYS + exog_cols].reset_index(drop=True)
    if out[exog_cols].isna().any().any():
        raise ValueError("las exógenas de SARIMAX no pueden tener nulos.")
    return out


def boosting_dataset(features: pd.DataFrame, cell: pd.Series, feature_cols: list[str],
                     cat_cols: list[str]) -> pd.DataFrame:
    """Arma el dataset de boosting global con los tipos que piden las librerías.

    Las columnas de `cat_cols` quedan como `category` de pandas, que LightGBM usa directo,
    XGBoost con `enable_categorical=True` y CatBoost listándolas en `cat_features`. Las
    categorías de `item_id` y `cell` se fijan con todos sus valores, así son las mismas en todos
    los folds. Los nulos de arranque de las variables se conservan (las tres librerías los
    manejan); la primera fila de cada artículo, sin bloque anterior, se deja fuera. Las filas
    tratadas (`ratio` NaN) se conservan: no entran al entrenamiento, pero se pronostican.

    Parámetros:
        features (pd.DataFrame): salida de la Etapa 6.
        cell (pd.Series): segmento ABC-XYZ por `item_id` (índice).
        feature_cols (list[str]): variables del modelo (incluye `cell` y las categóricas).
        cat_cols (list[str]): variables categóricas.

    Retorno:
        pd.DataFrame: identificadores, partición, `units`, `units_prev`, `ratio`, `ratio_raw`,
        `ratio_treated` y las variables.
        Lanza ValueError si falta el segmento de algún artículo.
    """
    out = features[features["units_prev"].notna()].reset_index(drop=True)
    out["cell"] = out["item_id"].map(cell)
    if out["cell"].isna().any():
        raise ValueError("falta el segmento de algún artículo.")
    for col in cat_cols:
        if not isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = pd.Categorical(out[col], categories=sorted(out[col].unique()))
    return out[DATASET_KEYS + [c for c in feature_cols if c not in DATASET_KEYS]]
