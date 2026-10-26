"""Fixtures sintéticos mínimos para las pruebas.

La estructura imita a M5 con rangos reducidos: 7 días de historia, cinco "años" de 56 días
(dos bloques de 28) y el último "año" como test (2 orígenes × 28 días).
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TINY_LAST_DAY = 287
TINY_YEARS = {"Y1": [8, 63], "Y2": [64, 119], "Y3": [120, 175], "Y4": [176, 231]}
TINY_TEST = [232, 287]


def _tiny_cfg() -> dict:
    """Construye una config pequeña a partir de params.toml.

    Retorno:
        dict: config válida con rangos de días reducidos y una grilla de modelos chica.
    """
    cfg = copy.deepcopy(load_config(PROJECT_ROOT / "src/config/params.toml"))
    cfg["data"]["last_day"] = TINY_LAST_DAY
    cfg["split"]["history_days"] = [1, 7]
    cfg["split"]["years"] = copy.deepcopy(TINY_YEARS)
    cfg["split"]["test_days"] = list(TINY_TEST)
    cfg["split"]["test_origins"] = 2
    cfg["split"]["horizon"] = 28
    cfg["stability"]["min_active_days_year"] = 56
    cfg["forecast"]["backtest_folds"] = 2
    cfg["forecast"]["ma_windows"] = [7]
    cfg["forecast"]["arima_p"] = [1]
    cfg["forecast"]["arima_d"] = [0]
    cfg["forecast"]["arima_q"] = [0]
    cfg["forecast"]["sarima_PDQ"] = [0]
    cfg["metrics"]["weight_window"] = [204, 231]
    return cfg


@pytest.fixture
def tiny_cfg() -> dict:
    """Config pequeña y válida."""
    return _tiny_cfg()


def make_raw_frames(last_day: int = TINY_LAST_DAY) -> dict[str, pd.DataFrame]:
    """Genera calendario, precios y ventas sintéticos con anomalías conocidas.

    Artículos (2 tiendas: CA_1, TX_1):
        - FOODS_A: alto gasto, precios distintos por tienda y semana.
        - FOODS_B: gasto medio.
        - FOODS_C: lanzado en la semana 10 solo en TX_1 (sin precio antes).
        - FOODS_D: bajo gasto, con una semana sin precio y ventas en CA_1 (semana 5)
          y una semana sin precio y sin ventas en TX_1 (semana 6).
    Anomalías en ventas: unidades negativas (FOODS_B, CA_1, d_20) y no enteras
    (FOODS_B, TX_1, d_21).

    Parámetros:
        last_day (int): último día con ventas.

    Retorno:
        dict[str, pd.DataFrame]: claves "calendar", "prices" y "sales".
    """
    rng = np.random.default_rng(7)
    days = np.arange(1, last_day + 1)
    dates = pd.date_range("2011-01-29", periods=last_day, freq="D")
    week_pos = (days - 1) // 7
    wm = 11101 + week_pos
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": wm,
            "weekday": dates.day_name(),
            "d": [f"d_{d}" for d in days],
        }
    )
    stores = ["CA_1", "TX_1"]
    items = ["FOODS_A", "FOODS_B", "FOODS_C", "FOODS_D"]
    base_price = {"FOODS_A": 10.0, "FOODS_B": 4.0, "FOODS_C": 3.0, "FOODS_D": 1.0}
    base_units = {"FOODS_A": 20.0, "FOODS_B": 12.0, "FOODS_C": 6.0, "FOODS_D": 3.0}
    weeks = np.unique(wm)
    price_rows = []
    for store_k, store in enumerate(stores):
        for item in items:
            for w_k, w in enumerate(weeks):
                if item == "FOODS_C" and (store == "CA_1" or w_k < 10):
                    continue
                if item == "FOODS_D" and store == "CA_1" and w_k == 5:
                    continue
                if item == "FOODS_D" and store == "TX_1" and w_k == 6:
                    continue
                price = base_price[item] * (1 + 0.1 * store_k) * (1 + 0.02 * (w_k % 3))
                price_rows.append((store, item, int(w), round(price, 2)))
    prices = pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])

    sales_rows = []
    for store in stores:
        for item in items:
            lam = base_units[item] * (1.2 if store == "TX_1" else 1.0)
            units = rng.poisson(lam, size=last_day).astype(float)
            if item == "FOODS_C":
                units[:70] = 0.0
                if store == "CA_1":
                    units[:] = 0.0
            if item == "FOODS_D" and store == "CA_1":
                units[35:42] = np.maximum(units[35:42], 1.0)
            if item == "FOODS_D" and store == "TX_1":
                units[42:49] = 0.0
            if item == "FOODS_B" and store == "CA_1":
                units[19] = -1.0
            if item == "FOODS_B" and store == "TX_1":
                units[20] = 2.5
            row = {
                "id": f"{item}_{store}_evaluation",
                "item_id": item,
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": store,
                "state_id": store[:2],
            }
            row.update({f"d_{d}": units[d - 1] for d in days})
            sales_rows.append(row)
    sales = pd.DataFrame(sales_rows)
    return {"calendar": calendar, "prices": prices, "sales": sales}


def write_raw(root: Path, cfg: dict, frames: dict[str, pd.DataFrame] | None = None) -> None:
    """Escribe los CSV sintéticos en `root/<raw_dir>`.

    Parámetros:
        root (Path): directorio base.
        cfg (dict): config con `data.raw_dir` y `data.sales_file`.
        frames (dict | None): tablas a escribir; si es None se generan.

    Retorno:
        None
    """
    frames = frames or make_raw_frames(cfg["data"]["last_day"])
    raw = root / cfg["data"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    frames["calendar"].to_csv(raw / "calendar.csv", index=False)
    frames["prices"].to_csv(raw / "sell_prices.csv", index=False)
    frames["sales"].to_csv(raw / cfg["data"]["sales_file"], index=False)


@pytest.fixture
def raw_frames() -> dict[str, pd.DataFrame]:
    """Tablas crudas sintéticas."""
    return make_raw_frames()


@pytest.fixture
def workdir(tmp_path, monkeypatch, tiny_cfg) -> Path:
    """Directorio de trabajo temporal con datos crudos sintéticos y carpetas del proyecto."""
    monkeypatch.chdir(tmp_path)
    for sub in ("data/raw", "data/interim", "data/processed", "models", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    write_raw(tmp_path, tiny_cfg)
    return tmp_path


@pytest.fixture
def staged(workdir, tiny_cfg) -> dict:
    """Ejecuta las etapas 1, 3 y 4 sobre los datos sintéticos (lista objetivo congelada)."""
    from src.data import run_data_validation
    from src.segmentation import run_train_matrix
    from src.stability import run_stability_stage

    run_data_validation(tiny_cfg)
    run_train_matrix(tiny_cfg)
    out = run_stability_stage(tiny_cfg)
    return {"cfg": tiny_cfg, "root": workdir, **out}
