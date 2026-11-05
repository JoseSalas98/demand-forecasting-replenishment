"""Modelado del nb 04: folds de backtesting, modelos y selección por artículo."""

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
