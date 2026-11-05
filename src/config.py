import csv
import itertools
import json
from pathlib import Path
import tomllib
from typing import Any

DEFAULT_CONFIG_PATH = "src/config/params.toml"
DEFAULT_HISTORY_PATH = "src/config/params_history.csv"

# Claves obligatorias del contrato. `split.years` y `treatment_map` son secciones con claves libres
# (años y celdas), por eso se validan aparte.
REQUIRED_KEYS = [
    "version",
    "data.raw_dir",
    "data.sales_file",
    "data.last_day",
    "split.history_days",
    "split.years",
    "split.test_days",
    "split.test_origins",
    "split.horizon",
    "segmentation.usage_period",
    "segmentation.abc_cuts",
    "segmentation.xyz_thresholds",
    "segmentation.cv_aggregation",
    "segmentation.cv_min_blocks",
    "segmentation.abc_rule_item_bounds",
    "segmentation.abc_rule_spend_bounds",
    "segmentation.abc_rule_step",
    "segmentation.cv_cut_min_share",
    "segmentation.cv_cut_resolution",
    "segmentation.zero_alpha",
    "segmentation.long_run",
    "stability.min_active_days_year",
    "stability.stability_rule",
    "stability.stability_rule_criterion",
    "treatment_map",
    "forecast.selection_level",
    "forecast.subset_size",
    "forecast.backtest_folds",
    "forecast.snaive_m",
    "forecast.ma_windows",
    "forecast.arima_p",
    "forecast.arima_d",
    "forecast.arima_q",
    "forecast.sarima_PDQ",
    "forecast.sarima_s",
    "metrics.weight_window",
    "features.target_lags",
    "features.rolling_windows",
    "features.rolling_stats",
    "features.event_lags",
    "features.seed",
    "features.low_block_share",
    "features.low_block_window",
    "features.max_ratio",
    "features.atypical_level",
    "features.atypical_min_ref",
    "features.atypical_years",
    "features.atypical_level_masked",
    "features.atypical_pattern_years",
    "features.atypical_refill_level",
    "features.case_low_level",
    "features.case_high_level",
    "features.case_masked_days",
    "features.case_start_blocks",
    "features.min_ratio_start",
]

ROLLING_STATS = {"mean", "std", "median", "range"}

CV_AGGREGATIONS = {"block28", "daily", "weekly"}
STABILITY_RULE_NAMES = {"abc_all", "abc_xyz_all", "cell_majority", "a_every_year"}
RULE_CRITERIA = {"fixed"}
USAGE_PERIODS = {"computation_set"}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Lee el archivo TOML de parámetros.

    Parámetros:
        path (str | Path): ruta del archivo de configuración.

    Retorno:
        dict: configuración tal como está en el archivo.
    """
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def get_key(cfg: dict, dotted: str) -> Any:
    """Devuelve el valor de una clave con puntos (p. ej. `segmentation.abc_cuts`).

    Parámetros:
        cfg (dict): configuración.
        dotted (str): clave con puntos.

    Retorno:
        Any: valor de la clave; lanza KeyError si no existe.
    """
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Aplana la configuración en claves con puntos.

    Parámetros:
        cfg (dict): configuración anidada.
        prefix (str): prefijo acumulado en la recursión.

    Retorno:
        dict[str, Any]: mapeo clave con puntos → valor hoja.
    """
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{dotted}."))
        else:
            out[dotted] = value
    return out


def _is_int(value: Any) -> bool:
    """Indica si el valor es entero (excluye bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _int_list(values: Any, min_value: int = 0) -> bool:
    """Indica si es una lista no vacía de enteros mayores o iguales a `min_value`."""
    return (
        isinstance(values, list)
        and len(values) > 0
        and all(_is_int(v) and v >= min_value for v in values)
    )


def _fail(rule: str, detail: str) -> None:
    """Detiene el proceso con el nombre de la regla de rango incumplida."""
    raise ValueError(f"Regla de rango '{rule}' no se cumple: {detail}")


def _validate_split(cfg: dict) -> None:
    """Valida que historia, años y test sean bloques contiguos, sin solaparse y dentro de `last_day`."""
    split = cfg["split"]
    last_day = cfg["data"]["last_day"]
    blocks = [("history_days", split["history_days"])]
    years = split["years"]
    if not years:
        _fail("split", "split.years está vacío")
    blocks += sorted(years.items(), key=lambda kv: kv[1][0])
    blocks.append(("test_days", split["test_days"]))
    for name, (start, end) in blocks:
        if not (_is_int(start) and _is_int(end) and 1 <= start <= end):
            _fail("split", f"{name} = {[start, end]} no es un rango válido")
    for (prev_name, prev), (name, cur) in itertools.pairwise(blocks):
        if cur[0] != prev[1] + 1:
            _fail(
                "split",
                f"{name} no es contiguo a {prev_name} (bloques contiguos y sin solapamiento)",
            )
    test_start, test_end = split["test_days"]
    if any(end >= test_start for _, (_, end) in blocks[:-1]):
        _fail("split", "todo día de train debe ser < test_days[0]")
    if test_end - test_start + 1 != split["test_origins"] * split["horizon"]:
        _fail("split", "longitud de test_days debe ser test_origins × horizon")
    if test_end > last_day:
        _fail("split", "test_days termina después de data.last_day")
    ww = cfg["metrics"]["weight_window"]
    train_end = blocks[-2][1][1]
    if not (_is_int(ww[0]) and _is_int(ww[1]) and 1 <= ww[0] <= ww[1] <= train_end):
        _fail("metrics", "weight_window debe estar dentro de train")


def validate_config(cfg: dict) -> None:
    """Exige todas las claves del contrato y aplica las reglas de rango (FR-008).

    Parámetros:
        cfg (dict): configuración leída.

    Retorno:
        None. Lanza ValueError con el nombre de la regla que falla.
    """
    for key in REQUIRED_KEYS:
        try:
            get_key(cfg, key)
        except (KeyError, TypeError):
            raise ValueError(f"Falta la clave obligatoria '{key}' en la configuración") from None

    seg, stab, fc = cfg["segmentation"], cfg["stability"], cfg["forecast"]
    cuts = seg["abc_cuts"]
    if not (len(cuts) == 2 and 0 < cuts[0] < cuts[1] < 1):
        _fail("abc_cuts", "0 < abc_cuts[0] < abc_cuts[1] < 1")
    xyz = seg["xyz_thresholds"]
    if not (len(xyz) == 2 and 0 <= xyz[0] < xyz[1]):
        _fail("xyz_thresholds", "0 ≤ x < z")
    for key in ("abc_rule_item_bounds", "abc_rule_spend_bounds"):
        pairs = seg[key]
        if not (
            isinstance(pairs, list)
            and len(pairs) == 2
            and all(isinstance(b, list) and len(b) == 2 and 0 <= b[0] < b[1] <= 1 for b in pairs)
        ):
            _fail("abc_rule", f"{key}: dos intervalos [lo, hi] con 0 ≤ lo < hi ≤ 1")
    if not 0 < seg["abc_rule_step"] < 1:
        _fail("abc_rule", "0 < abc_rule_step < 1")
    if not 0 < seg["zero_alpha"] < 1:
        _fail("zeros", "0 < zero_alpha < 1")
    if not (_is_int(seg["long_run"]) and seg["long_run"] >= 1):
        _fail("zeros", "long_run debe ser entero ≥ 1")
    if not 0 < seg["cv_cut_min_share"] < 0.5:
        _fail("cv_cuts", "0 < cv_cut_min_share < 0.5")
    if not (_is_int(seg["cv_cut_resolution"]) and seg["cv_cut_resolution"] >= 10):
        _fail("cv_cuts", "cv_cut_resolution debe ser entero ≥ 10")
    _validate_split(cfg)

    if seg["cv_aggregation"] not in CV_AGGREGATIONS:
        _fail("enums", f"cv_aggregation desconocido: {seg['cv_aggregation']}")
    if stab["stability_rule"] not in STABILITY_RULE_NAMES:
        _fail("enums", f"stability_rule desconocida: {stab['stability_rule']}")
    if stab["stability_rule_criterion"] not in RULE_CRITERIA:
        _fail("enums", f"stability_rule_criterion desconocido: {stab['stability_rule_criterion']}")
    if seg["usage_period"] not in USAGE_PERIODS:
        _fail("enums", f"usage_period desconocido: {seg['usage_period']}")
    if not (_is_int(seg["cv_min_blocks"]) and seg["cv_min_blocks"] >= 1):
        _fail("segmentation", "cv_min_blocks debe ser entero ≥ 1")
    if not (_is_int(stab["min_active_days_year"]) and stab["min_active_days_year"] >= 1):
        _fail("stability", "min_active_days_year debe ser entero ≥ 1")

    tmap = cfg["treatment_map"]
    if not tmap or not all(isinstance(v, str) and v.strip() for v in tmap.values()):
        _fail("treatment_map", "los valores deben ser cadenas no vacías")

    for key in ("ma_windows", "arima_p", "arima_d", "arima_q", "sarima_PDQ"):
        if not _int_list(fc[key]):
            _fail("forecast", f"{key} debe ser una lista de enteros no negativos")
    if not _int_list(fc["ma_windows"], min_value=1):
        _fail("forecast", "ma_windows debe tener enteros ≥ 1")
    for key in ("backtest_folds", "snaive_m", "sarima_s"):
        if not (_is_int(fc[key]) and fc[key] >= 1):
            _fail("forecast", f"{key} debe ser entero ≥ 1")
    last_year = list(train_years(cfg).values())[-1]
    if fc["backtest_folds"] * cfg["split"]["horizon"] > last_year[1] - last_year[0] + 1:
        _fail("forecast", "backtest_folds × horizon debe caber en el último año de train")
    if not (fc["subset_size"] == "all" or (_is_int(fc["subset_size"]) and fc["subset_size"] >= 1)):
        _fail("forecast", "subset_size debe ser 'all' o entero ≥ 1")
    if not (isinstance(fc["selection_level"], str) and fc["selection_level"].strip()):
        _fail("forecast", "selection_level debe ser una cadena no vacía")

    feat = cfg["features"]
    for key in ("target_lags", "rolling_windows", "event_lags"):
        if not _int_list(feat[key], min_value=1):
            _fail("features", f"{key} debe ser una lista de enteros ≥ 1 (h = 1)")
    if not (feat["rolling_stats"] and set(feat["rolling_stats"]) <= ROLLING_STATS):
        _fail("features", f"rolling_stats debe ser un subconjunto no vacío de {sorted(ROLLING_STATS)}")
    if not (isinstance(feat["low_block_share"], (int, float)) and 0 < feat["low_block_share"] < 1):
        _fail("features", "low_block_share debe estar entre 0 y 1")
    if not (_is_int(feat["low_block_window"]) and feat["low_block_window"] >= 2):
        _fail("features", "low_block_window debe ser un entero ≥ 2")
    if not (isinstance(feat["max_ratio"], (int, float)) and feat["max_ratio"] > 1):
        _fail("features", "max_ratio debe ser un número > 1 (un ratio ≤ 1 no es una subida)")
    if not (isinstance(feat["atypical_level"], (int, float)) and 0 < feat["atypical_level"] < 1):
        _fail("features", "atypical_level debe estar entre 0 y 1")
    if not (_is_int(feat["atypical_min_ref"]) and 1 <= feat["atypical_min_ref"] <= feat["low_block_window"]):
        _fail("features", "atypical_min_ref debe ser un entero entre 1 y low_block_window")
    if not (_is_int(feat["atypical_years"]) and feat["atypical_years"] >= 1):
        _fail("features", "atypical_years debe ser un entero ≥ 1")
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if not (num(feat["atypical_level_masked"]) and feat["atypical_level"] <= feat["atypical_level_masked"] < 1):
        _fail("features", "atypical_level_masked debe estar entre atypical_level y 1")
    if not (_is_int(feat["atypical_pattern_years"]) and 1 <= feat["atypical_pattern_years"] <= feat["atypical_years"]):
        _fail("features", "atypical_pattern_years debe ser un entero entre 1 y atypical_years")
    if not (num(feat["atypical_refill_level"]) and 0 < feat["atypical_refill_level"] <= 1):
        _fail("features", "atypical_refill_level debe estar entre 0 y 1")
    if not (num(feat["case_low_level"]) and 0 < feat["case_low_level"] < 1):
        _fail("features", "case_low_level debe estar entre 0 y 1")
    if not (num(feat["case_high_level"]) and feat["case_high_level"] > 1):
        _fail("features", "case_high_level debe ser > 1")
    if not (_is_int(feat["case_masked_days"]) and 1 <= feat["case_masked_days"] <= 6 * 28):
        _fail("features", "case_masked_days debe ser un entero entre 1 y 168 (6 bloques)")
    if not (_is_int(feat["case_start_blocks"]) and feat["case_start_blocks"] >= 2):
        _fail("features", "case_start_blocks debe ser un entero ≥ 2")
    if not (num(feat["min_ratio_start"]) and 0 < feat["min_ratio_start"] < 1):
        _fail("features", "min_ratio_start debe estar entre 0 y 1")
    if not (_is_int(feat["seed"]) and feat["seed"] >= 0):
        _fail("features", "seed debe ser un entero ≥ 0")


def _same_value(logged: str, value: Any) -> bool:
    """Compara el `new_value` del historial (JSON o texto) con el valor actual."""
    try:
        return json.loads(logged) == value
    except (json.JSONDecodeError, TypeError):
        return logged == json.dumps(value)


def check_history(
    cfg: dict, snapshot: dict | None, history_path: str | Path = DEFAULT_HISTORY_PATH
) -> None:
    """Verifica que cada cambio frente al último snapshot tenga su fila en el historial (FR-007).

    Parámetros:
        cfg (dict): configuración actual.
        snapshot (dict | None): config congelada en el manifiesto; None en la primera ejecución.
        history_path (str | Path): CSV con `date, parameter, old_value, new_value, reason`.

    Retorno:
        None. Lanza RuntimeError si falta alguna fila.
    """
    with open(history_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if snapshot is None:
        if not any(r["parameter"] == "*" for r in rows):
            raise RuntimeError("El historial de parámetros no tiene la fila inicial ('*').")
        return
    current, previous = flatten(cfg), flatten(snapshot)
    missing = []
    for key in sorted(set(current) | set(previous)):
        if current.get(key) == previous.get(key):
            continue
        new_value = current.get(key)
        logged = [
            r for r in rows if r["parameter"] == key and _same_value(r["new_value"], new_value)
        ]
        if not logged:
            missing.append(key)
    if missing:
        raise RuntimeError(
            "Cambios de parámetros sin fila en el historial: "
            + ", ".join(missing)
            + f". Añadirlas en {history_path} con el valor nuevo en JSON."
        )


def load_validated_config(
    path: str | Path = DEFAULT_CONFIG_PATH, history_path: str | Path = DEFAULT_HISTORY_PATH
) -> dict:
    """Carga la configuración, valida rangos y verifica el historial contra el manifiesto.

    Parámetros:
        path (str | Path): ruta de params.toml.
        history_path (str | Path): ruta de params_history.csv.

    Retorno:
        dict: configuración validada.
    """
    from src.freeze import load_manifest

    cfg = load_config(path)
    validate_config(cfg)
    check_history(cfg, load_manifest().get("config_snapshot"), history_path)
    return cfg


def train_years(cfg: dict) -> dict[str, tuple[int, int]]:
    """Devuelve los años de train ordenados por fecha de inicio.

    Parámetros:
        cfg (dict): configuración.

    Retorno:
        dict[str, tuple[int, int]]: nombre del año → (día inicial, día final).
    """
    items = sorted(cfg["split"]["years"].items(), key=lambda kv: kv[1][0])
    return {name: (int(a), int(b)) for name, (a, b) in items}


def train_end(cfg: dict) -> int:
    """Último día de train (fin del último año de segmentación).

    Parámetros:
        cfg (dict): configuración.

    Retorno:
        int: día final de train.
    """
    return max(end for _, end in train_years(cfg).values())


def all_years(cfg: dict, test_name: str = "Y5") -> dict[str, tuple[int, int]]:
    """Años de train más el año de test, en orden.

    Parámetros:
        cfg (dict): configuración.
        test_name (str): nombre del año de test.

    Retorno:
        dict[str, tuple[int, int]]: nombre del año → (día inicial, día final).
    """
    years = train_years(cfg)
    years[test_name] = tuple(int(d) for d in cfg["split"]["test_days"])
    return years


def forecast_keys(cfg: dict) -> list[str]:
    """Claves de `[forecast]` que se declaran antes de la réplica en el año 5 (FR-023, FR-024).

    Parámetros:
        cfg (dict): configuración.

    Retorno:
        list[str]: claves con puntos, ordenadas.
    """
    return sorted(f"forecast.{k}" for k in cfg["forecast"])
