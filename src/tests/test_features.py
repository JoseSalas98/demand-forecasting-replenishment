import pandas as pd
import pytest

from src.features import TARGET_COLUMNS, flag_targets


def _targets(ids):
    return pd.DataFrame({"item_id": ids, "cell": "AX", "abc_class": "A", "xyz_class": "X",
                         "cv_train": 10.0, "long_run": 3, "min_days": 4,
                         "treatment": "regular_replenishment"})


def test_flag_targets_marks_and_keeps_catalog():
    out = flag_targets(["I1", "I2", "I3"], _targets(["I2"]))
    assert list(out["item_id"]) == ["I1", "I2", "I3"]
    assert list(out["is_target"]) == [False, True, False]
    assert out.loc[out["is_target"], "cell"].item() == "AX"
    assert out.loc[~out["is_target"], TARGET_COLUMNS].isna().all().all()


def test_flag_targets_rejects_unknown_or_repeated():
    with pytest.raises(ValueError, match="catálogo"):
        flag_targets(["I1"], _targets(["I9"]))
    with pytest.raises(ValueError, match="repetidos"):
        flag_targets(["I1"], _targets(["I1", "I1"]))


def test_aggregate_blocks_sums_and_spans_first_to_last_sale():
    import numpy as np

    from src.features import aggregate_blocks

    days = [f"d_{d}" for d in range(1, 57)]
    a = np.ones(56)
    a[10:20] = np.nan  # quiebre enmascarado: no anula ni resta
    b = np.full(56, np.nan)
    b[55] = 4  # primera venta el último día del segundo bloque
    units = pd.DataFrame([a, b], index=["A", "B"], columns=days)
    chain = np.full(56, 10.0)
    dates = pd.Series({d: f"2020-01-{d:02d}" if d < 32 else "x" for d in range(1, 57)})
    bounds = pd.DataFrame({"first_sale_d": [1, 56], "last_sale_d": [56, 56]}, index=["A", "B"])
    out = aggregate_blocks(units, chain, bounds, dates)
    assert out[["item_id", "block"]].values.tolist() == [["A", 1], ["A", 2], ["B", 2]]
    assert out["units"].tolist() == [18.0, 28.0, 4.0]
    assert out["days_with_data"].tolist() == [18, 28, 1]
    assert out["chain_units"].tolist() == [280.0, 280.0, 280.0]
    assert out["start_date"].iloc[0] == "2020-01-01"


def test_aggregate_blocks_stops_at_last_sale_block():
    import numpy as np

    from src.features import aggregate_blocks

    days = [f"d_{d}" for d in range(1, 85)]
    a = np.full(84, np.nan)
    a[:30] = 1  # última venta el día 30: bloque 2; el bloque 3 no existe para el artículo
    units = pd.DataFrame([a], index=["A"], columns=days)
    bounds = pd.DataFrame({"first_sale_d": [1], "last_sale_d": [30]}, index=["A"])
    out = aggregate_blocks(units, np.ones(84), bounds, pd.Series(range(1, 85), index=range(1, 85)))
    assert out["block"].tolist() == [1, 2] and out["units"].tolist() == [28.0, 2.0]


def test_assign_windows_counts_from_the_end_and_excludes_short():
    from src.features import assign_windows

    blocks = pd.DataFrame({"item_id": ["A"] * 8 + ["B"] * 4, "block": list(range(3, 11)) + list(range(1, 5))})
    out, excluded = assign_windows(blocks, test_blocks=2, val_blocks=2, min_blocks=5)
    a = out[out["item_id"] == "A"]
    assert a["split"].tolist() == ["train"] * 6 + ["test"] * 2
    assert a["backtest"].tolist() == [False] * 4 + [True] * 2 + [False] * 2
    assert a["pos"].tolist() == list(range(1, 9))
    assert excluded[["item_id", "n_blocks"]].values.tolist() == [["B", 4]]
    with pytest.raises(ValueError):
        assign_windows(blocks, 2, 2, 4)


def test_block_calendar_year_period_and_events():
    from src.features import block_calendar

    n_days = 28 * 14
    cal = pd.DataFrame({"d": [f"d_{d}" for d in range(1, n_days + 1)],
                        "event_name_1": None, "event_type_1": None, "event_name_2": None, "event_type_2": None})
    cal.loc[0, ["event_name_1", "event_type_1"]] = ["Xmas", "National"]
    cal.loc[1, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = ["Easter", "Cultural", "OE", "Religious"]
    cal.loc[13 * 28, ["event_name_1", "event_type_1"]] = ["SB", "Sporting"]
    out = block_calendar(cal, 1, 14).set_index("block")
    assert out.loc[1, ["year", "period"]].tolist() == [1, 1]
    assert out.loc[13, ["year", "period"]].tolist() == [1, 13]
    assert out.loc[14, ["year", "period"]].tolist() == [2, 1]
    assert out.loc[1, ["n_events", "has_event", "has_event_1", "has_event_2"]].tolist() == [3, 1, 1, 1]
    assert out.loc[1, ["n_religious", "n_national", "n_cultural", "n_sporting"]].tolist() == [1, 1, 1, 0]
    assert out.loc[2, ["n_events", "has_event"]].tolist() == [0, 0]
    assert out.loc[14, "n_sporting"] == 1


def _series():
    return pd.DataFrame({"item_id": ["A"] * 5 + ["B"] * 3, "block": [1, 2, 3, 4, 5, 7, 8, 9],
                         "y": [1.0, 2.0, 3.0, 4.0, 10.0, 5.0, 6.0, 7.0]})


def test_add_lags_by_item():
    from src.features import add_lags

    out = add_lags(_series(), ["y"], [1, 2])
    a, b = out[out["item_id"] == "A"], out[out["item_id"] == "B"]
    assert a["y_lag1"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0] and pd.isna(a["y_lag1"].iloc[0])
    assert b["y_lag2"].tolist()[2] == 5.0 and b["y_lag2"].isna().sum() == 2
    with pytest.raises(ValueError, match="contiguos"):
        add_lags(_series().drop(index=2), ["y"], [1])


def test_add_rolling_excludes_current_block():
    import numpy as np

    from src.features import add_rolling

    out = add_rolling(_series(), "y", [3], ["mean", "std", "median", "range"])
    a = out[out["item_id"] == "A"].reset_index(drop=True)
    # Bloque 4: ventana = bloques 1–3 (1, 2, 3); bloque 5: bloques 2–4 (2, 3, 4), nunca su propio 10.
    assert a["y_roll3_mean"].tolist()[3:] == [2.0, 3.0]
    assert a["y_roll3_median"].tolist()[3:] == [2.0, 3.0]
    assert a["y_roll3_range"].tolist()[3:] == [2.0, 2.0]
    assert np.isclose(a["y_roll3_std"].iloc[3], 1.0)
    assert a["y_roll3_mean"].iloc[:3].isna().all()
    b = out[out["item_id"] == "B"]
    assert b["y_roll3_mean"].isna().all(), "B tiene solo 3 bloques: ninguna ventana completa de 3 anteriores"


def test_aggregate_blocks_block_without_data_is_nan():
    import numpy as np

    from src.features import aggregate_blocks

    days = [f"d_{d}" for d in range(1, 85)]
    a = np.ones(84)
    a[28:56] = np.nan  # bloque 2 entero dentro de un quiebre
    units = pd.DataFrame([a], index=["A"], columns=days)
    bounds = pd.DataFrame({"first_sale_d": [1], "last_sale_d": [84]}, index=["A"])
    out = aggregate_blocks(units, np.ones(84), bounds, pd.Series(range(1, 85), index=range(1, 85)))
    assert out["units"].isna().tolist() == [False, True, False]
    assert out["days_with_data"].tolist() == [28, 0, 28]


def test_fill_invalid_blocks_carries_previous_value():
    import numpy as np

    from src.features import fill_invalid_blocks

    blocks = pd.DataFrame({
        "item_id": ["A"] * 5 + ["B"] * 3, "block": [1, 2, 3, 4, 5, 1, 2, 3],
        "units": [10.0, 1.0, np.nan, 40.0, 50.0, 2.0, 30.0, 35.0],
        "days_with_data": [28, 2, 0, 28, 28, 1, 28, 28],
    })
    out = fill_invalid_blocks(blocks, pd.Series({"A": 4, "B": 4}), 0.2, 13)
    a, b = out[out["item_id"] == "A"], out[out["item_id"] == "B"]
    assert a["units"].tolist() == [10.0, 10.0, 10.0, 40.0, 50.0]
    assert a["imputed"].tolist() == [False, True, True, False, False]
    assert b["block"].tolist() == [2, 3] and b["units"].tolist() == [30.0, 35.0]
    with pytest.raises(ValueError, match="min_days"):
        fill_invalid_blocks(blocks, pd.Series({"A": 4}), 0.2, 13)


def test_add_ratio_by_item():
    import numpy as np

    from src.features import add_ratio

    blocks = pd.DataFrame({
        "item_id": ["B", "A", "A", "A", "B", "B"],
        "block": [2, 1, 2, 3, 3, 1],
        "units": [10.0, 4.0, 8.0, 6.0, 5.0, np.nan],
    })
    out = add_ratio(blocks)
    a = out[out["item_id"] == "A"]
    assert a["ratio"].isna().tolist() == [True, False, False]
    assert a["ratio"].iloc[1:].tolist() == [2.0, 0.75]
    b = out[out["item_id"] == "B"]
    assert b["ratio"].isna().tolist() == [True, True, False] and b["ratio"].iloc[2] == 0.5
    ok = out["ratio"].notna()
    assert np.allclose(out.loc[ok, "units_prev"] * out.loc[ok, "ratio"], out.loc[ok, "units"])

    with pytest.raises(ValueError):
        add_ratio(blocks.assign(units=[10.0, 0.0, 8.0, 6.0, 5.0, 1.0]))


def test_calendar_lags_uses_days_before_block_1():
    from src.features import calendar_lags

    # El bloque 1 del análisis empieza en d_57; los bloques −1 (d_1–d_28) y 0 (d_29–d_56) solo
    # sirven para los rezagos.
    calendar = pd.DataFrame({"d": [f"d_{d}" for d in range(1, 141)], "event_name_1": None,
                             "event_type_1": None, "event_name_2": None, "event_type_2": None})
    calendar.loc[calendar["d"] == "d_5", ["event_name_1", "event_type_1"]] = ["X", "Cultural"]     # bloque −1
    calendar.loc[calendar["d"] == "d_40", ["event_name_1", "event_type_1"]] = ["Y", "Sporting"]    # bloque 0
    out = calendar_lags(calendar, 57, 3, ["n_cultural", "n_sporting"], [1, 2])
    assert out["block"].tolist() == [1, 2, 3]
    assert out["n_cultural_lag1"].tolist() == [0, 0, 0]
    assert out["n_cultural_lag2"].tolist() == [1, 0, 0]
    assert out["n_sporting_lag1"].tolist() == [1, 0, 0]
    assert out["n_sporting_lag2"].tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="no cubre"):
        calendar_lags(calendar, 57, 3, ["n_cultural"], [3])


def test_model_datasets_drop_first_row_and_set_types():
    import numpy as np

    from src.features import boosting_dataset, sarimax_dataset

    features = pd.DataFrame({
        "item_id": ["A", "A", "A", "B", "B"], "block": [1, 2, 3, 1, 2],
        "start_date": pd.to_datetime(["2011-05-30"] * 5), "split": ["train"] * 4 + ["test"],
        "backtest": False, "imputed": False, "units": [1.0, 2.0, 3.0, 4.0, 5.0],
        "units_prev": [np.nan, 1.0, 2.0, np.nan, 4.0], "ratio": [np.nan, np.nan, 1.5, np.nan, 1.25],
        "ratio_raw": [np.nan, 2.5, 1.5, np.nan, 1.25], "ratio_treated": [False, True, False, False, False],
        "n_national": [0, 1, 0, 0, 1], "ratio_lag1": [np.nan, np.nan, 2.0, np.nan, np.nan],
        "period": pd.Categorical([1, 2, 3, 1, 2], categories=range(1, 14), ordered=True),
    })
    sar = sarimax_dataset(features, ["n_national"])
    # La fila tratada (A, bloque 2) se conserva con ratio NaN: se pronostica aunque no entre al ajuste.
    assert len(sar) == 3 and sar["ratio_raw"].notna().all() and sar["ratio"].isna().sum() == 1
    assert "ratio_lag1" not in sar
    with pytest.raises(ValueError, match="nulos"):
        sarimax_dataset(features.assign(n_national=[0, np.nan, 0, 0, 1]), ["n_national"])

    cols = ["ratio_lag1", "n_national", "period", "item_id", "cell"]
    boost = boosting_dataset(features, pd.Series({"A": "AX", "B": "AZ"}), cols, ["item_id", "cell", "period"])
    assert len(boost) == 3 and boost["ratio_lag1"].isna().sum() == 2
    assert all(isinstance(boost[c].dtype, pd.CategoricalDtype) for c in ["item_id", "cell", "period"])
    assert list(boost["cell"].cat.categories) == ["AX", "AZ"] and boost["period"].cat.ordered
    with pytest.raises(ValueError, match="segmento"):
        boosting_dataset(features, pd.Series({"A": "AX"}), cols, ["cell"])


def test_fill_invalid_blocks_low_units_rule():
    from src.features import fill_invalid_blocks

    # Ventana de 5: primer año = bloques 1–5 (mediana 90). Bloque 1 bajo pero protegido
    # (lanzamiento); bloque 3 bajo frente al primer año; bloque 6 bajo frente a sus válidos previos.
    units = [5.0, 100.0, 10.0, 110.0, 90.0, 15.0, 95.0]
    blocks = pd.DataFrame({"item_id": "A", "block": range(1, 8), "units": units, "days_with_data": 28})
    out = fill_invalid_blocks(blocks, pd.Series({"A": 4}), 0.2, 5)
    assert out["invalid_reason"].tolist() == ["", "", "low_units", "", "", "low_units", ""]
    assert out["units"].tolist() == [5.0, 100.0, 100.0, 110.0, 90.0, 90.0, 95.0]
    # El 20 % es estricto: 20 frente a una referencia de 100 no se marca.
    edge = blocks.assign(units=[100.0, 100.0, 100.0, 100.0, 100.0, 20.0, 100.0])
    assert fill_invalid_blocks(edge, pd.Series({"A": 4}), 0.2, 5)["invalid_reason"].tolist()[5] == ""


def test_treat_ratio_empties_high_ratios_and_keeps_raw():
    import numpy as np

    from src.features import treat_ratio

    features = pd.DataFrame({"item_id": ["A"] * 4, "block": [1, 2, 3, 4], "units": [1.0, 3.0, 3.0, 6.0],
                             "units_prev": [np.nan, 1.0, 3.0, 3.0], "ratio": [np.nan, 3.0, 1.0, 2.0]})
    out = treat_ratio(features, 2.0)
    assert out["ratio_raw"].tolist()[1:] == [3.0, 1.0, 2.0]
    assert out["ratio_treated"].tolist() == [False, True, False, False]      # 2.0 se conserva
    assert out["ratio"].isna().tolist() == [True, True, False, False]
    assert features["ratio"].tolist()[1:] == [3.0, 1.0, 2.0]                 # no modifica la entrada
    with pytest.raises(ValueError):
        treat_ratio(features, 1.0)
    with pytest.raises(ValueError, match="ya se aplicó"):
        treat_ratio(out, 2.0)


def test_add_rolling_skips_treated_blocks_but_keeps_warmup():
    import numpy as np

    from src.features import add_rolling

    # 1.er bloque sin ratio (como en el pipeline); el bloque 4 está tratado (NaN).
    data = pd.DataFrame({"item_id": ["A"] * 7, "block": range(1, 8),
                         "y": [np.nan, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    data["treated"] = [False, False, False, True, False, False, False]
    out = add_rolling(data, "y", [3], ["mean", "std", "range"], treated_col="treated")
    # Arranque: el 1.er bloque no tiene valor y no es tratado → las primeras k + 1 = 4 filas quedan NaN.
    assert out["y_roll3_mean"].iloc[:4].isna().all()
    # Bloque 5: ventana = bloques 2–4 (1, 2, NaN) → se omite el tratado: media 1.5.
    assert out["y_roll3_mean"].tolist()[4:] == [1.5, 3.0, 4.5]
    assert np.isclose(out["y_roll3_std"].iloc[4], np.std([1.0, 2.0], ddof=1))
    assert out["y_roll3_range"].tolist()[4:] == [1.0, 2.0, 1.0]
    # Sin marcar el bloque 4 como tratado, su NaN deja incompletas las ventanas que lo contienen.
    plain = add_rolling(data.drop(columns="treated"), "y", [3], ["mean"])
    assert plain["y_roll3_mean"].iloc[4:7].isna().tolist() == [True, True, True]


def test_fill_atypical_blocks_by_prior_year():
    from src.features import fill_atypical_blocks

    # Periodo de 4 bloques (para el test); el nivel exige 3 bloques válidos previos, así que el
    # primer periodo evaluable es el bloque 4. A: cae en el bloque 8 y su mismo periodo del año
    # anterior (bloque 4) estuvo normal → atípico. B: cae en los bloques 4 y 8: el 4 no tiene año
    # anterior (→ valor del bloque anterior) y el 8 repite la caída del 4 (patrón, no se toca).
    rows = []
    for item, units in {"A": [100, 100, 100, 100, 100, 100, 100, 20],
                        "B": [100, 100, 100, 20, 100, 100, 100, 20]}.items():
        rows += [{"item_id": item, "block": b + 1, "units": float(u), "invalid_reason": "", "imputed": False}
                 for b, u in enumerate(units)]
    out = fill_atypical_blocks(pd.DataFrame(rows), 0.5, 6, 3, 3, period=4)
    a, b = (out[out["item_id"] == k].set_index("block") for k in "AB")
    assert a.loc[8, "invalid_reason"] == "atypical" and a.loc[8, "units"] == 100.0   # mediana 100 × nivel 1
    assert a.loc[8, "units_observed"] == 20.0 and a.loc[8, "imputed"]
    assert (a.drop(index=8)["invalid_reason"] == "").all()
    assert b.loc[4, "invalid_reason"] == "atypical_first_year" and b.loc[4, "units"] == 100.0
    assert b.loc[8, "invalid_reason"] == "" and b.loc[8, "units"] == 20.0           # se repite: no se toca


def test_fill_atypical_blocks_refills_invalid_after_corrected_block():
    from src.features import fill_atypical_blocks

    # El bloque 8 es no válido de la 4.2 y copió el 7 deprimido; tras corregir el 7, vuelve a copiarlo.
    units = [100, 100, 100, 100, 100, 100, 20, 20]
    reasons = [""] * 7 + ["min_days"]
    df = pd.DataFrame({"item_id": "A", "block": range(1, 9), "units": [float(u) for u in units],
                       "invalid_reason": reasons, "imputed": [r != "" for r in reasons]})
    out = fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4).set_index("block")
    assert out.loc[7, "units"] == 100.0 and out.loc[8, "units"] == 100.0 and out.loc[8, "invalid_reason"] == "min_days"
    with pytest.raises(ValueError):
        fill_atypical_blocks(df, 1.0, 6, 3, 3)


def test_fill_atypical_blocks_factor_is_median_of_years_capped_at_one():
    from src.features import fill_atypical_blocks

    # Periodo 4. El mismo periodo del bloque 12 estuvo alto el año anterior (bloque 8, nivel 3) y
    # normal hace dos años (bloque 4, nivel 1): mediana 2 → tope 1 → el valor completado = 100.
    units = [100, 100, 100, 100, 100, 100, 100, 300, 100, 100, 100, 20]
    df = pd.DataFrame({"item_id": "A", "block": range(1, 13), "units": [float(u) for u in units],
                       "invalid_reason": "", "imputed": False})
    out = fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4).set_index("block")
    assert out.loc[12, "invalid_reason"] == "atypical" and out.loc[12, "units"] == 100.0


def test_drop_partial_first_block():
    from src.features import drop_partial_first_block

    # A: primer bloque con 14 días → se quita. B: completo → se queda. C: incompleto, pero sin él
    # quedaría bajo el mínimo de 4 bloques → se conserva. D: incompleto y el bloque 2 no era válido
    # (copió el valor del 1) → se quitan los dos y la serie empieza en el bloque 3.
    rows = []
    for item, n, d1 in [("A", 5, 14), ("B", 5, 28), ("C", 4, 10), ("D", 6, 10)]:
        rows += [{"item_id": item, "block": b, "days_with_data": d1 if b == 1 else 28, "units": 1.0,
                  "imputed": item == "D" and b == 2} for b in range(1, n + 1)]
    out = drop_partial_first_block(pd.DataFrame(rows), min_blocks=4)
    first = out.groupby("item_id")["block"].min()
    assert first.to_dict() == {"A": 2, "B": 1, "C": 1, "D": 3}
    assert len(out) == 17


def _model_blocks(series: dict, n_test: int = 13, n_val: int = 13) -> pd.DataFrame:
    """Bloques con partición (como la salida de la 4.3) para los tests de la 4.4."""
    rows = []
    for item, units in series.items():
        n = len(units)
        for p, u in enumerate(units, start=1):
            from_end = n - p
            rows.append({"item_id": item, "block": p, "pos": p, "units": float(u), "days_with_data": 28,
                         "invalid_reason": "", "imputed": False,
                         "split": "test" if from_end < n_test else "train",
                         "backtest": n_test <= from_end < n_test + n_val})
    return pd.DataFrame(rows)


def test_fill_atypical_blocks_causal_extensions():
    from src.features import fill_atypical_blocks

    base = {"item_id": "A", "block": range(1, 9), "invalid_reason": "", "imputed": False}
    # E1: el bloque 8 vende el 60 % de su nivel con 20 días: solo con level_masked = 0.8 es atípico.
    df = pd.DataFrame({**base, "units": [100.0] * 7 + [60.0], "days_with_data": [28] * 7 + [20]})
    assert fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4).set_index("block").loc[8, "invalid_reason"] == ""
    e1 = fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4, level_masked=0.8).set_index("block")
    assert e1.loc[8, "invalid_reason"] == "atypical" and e1.loc[8, "units"] == 100.0
    # E7: el bloque 12 cae; el mismo periodo cayó un año antes (bloque 8) y no dos (bloque 4).
    units = [100, 100, 100, 100, 100, 100, 100, 20, 100, 100, 100, 20]
    df = pd.DataFrame({"item_id": "A", "block": range(1, 13), "units": [float(u) for u in units],
                       "days_with_data": 28, "invalid_reason": "", "imputed": False})
    assert fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4).set_index("block").loc[12, "invalid_reason"] == ""
    e7 = fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4, pattern_years=2).set_index("block")
    assert e7.loc[12, "invalid_reason"] == "atypical"
    # E8: el 7 queda al 60 % (no se corrige) y el 8 no es válido: con refill_level toma la mediana.
    reasons = [""] * 7 + ["min_days"]
    df = pd.DataFrame({**base, "units": [100.0] * 6 + [60.0, 60.0], "days_with_data": 28,
                       "invalid_reason": reasons, "imputed": [r != "" for r in reasons]})
    assert fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4).set_index("block").loc[8, "units"] == 60.0
    assert fill_atypical_blocks(df, 0.5, 6, 3, 3, period=4, refill_level=0.8).set_index("block").loc[8, "units"] == 100.0


def test_classify_and_impute_history_cases():
    from src.features import classify_extreme_ratios, impute_history_cases, replace_after_history

    # 40 bloques: historia 1–14, ventana 15–27, test 28–40.
    a = [100] * 40
    a[1] = 300                        # arranque (posición 2)
    a[9] = 30                         # caída de un bloque en la historia (bloque 10)
    a[29] = 30                        # caída de un bloque en el test (bloque 30)
    b = [100] * 11 + [300] * 29       # subida de nivel que persiste (bloque 12)
    blocks = _model_blocks({"A": a, "B": b})
    cases = classify_extreme_ratios(blocks, 0.5)
    c = cases.set_index(["item_id", "block", "direction"])
    assert c.loc[("A", 2, "up"), "group"] == "arranque"
    assert c.loc[("A", 10, "down"), "detail"] == "caída de un bloque sin días de quiebre"
    assert c.loc[("A", 10, "down"), "culprit"] == "t"
    assert c.loc[("B", 12, "up"), "group"] == "estructural"
    out = impute_history_cases(blocks, cases).set_index(["item_id", "block"])
    assert out.loc[("A", 10), "units"] == 100.0 and out.loc[("A", 10), "invalid_reason"] == "case_history"
    assert out.loc[("A", 30), "units"] == 30.0, "el test no se imputa con t+1"
    assert (out.loc["B", "units"] == pd.Series(b, index=range(1, 41), dtype=float)).all(), "lo estructural no se toca"
    causal = blocks.assign(units=1.0, invalid_reason="atypical", imputed=True)
    rest = replace_after_history(blocks, causal)
    after = (rest["split"] == "test") | rest["backtest"]
    assert (rest.loc[after, "units"] == 1.0).all() and rest.loc[~after, "units"].equals(blocks.loc[~after, "units"])


def test_treat_ratio_start_low_ratios():
    from src.features import treat_ratio

    df = pd.DataFrame({"ratio": [None, 0.3, 3.0, 0.3, 1.0], "pos": [1, 2, 3, 9, 10]})
    out = treat_ratio(df, 2.0, min_ratio=0.5, start_blocks=7)
    assert out["ratio_treated"].tolist() == [False, True, True, False, False]
    with pytest.raises(ValueError):
        treat_ratio(df, 2.0, min_ratio=0.5)


def test_interpolate_treated_only_fills_treated_inside():
    import numpy as np
    from src.features import interpolate_treated
    f = pd.DataFrame({"item_id": "I1", "block": [1, 2, 3, 4, 5, 6],
                      "ratio": [np.nan, 1.0, np.nan, 2.0, 1.5, np.nan],
                      "ratio_treated": [False, False, True, False, False, True]})
    out = interpolate_treated(f)
    assert np.isnan(out.loc[0, "ratio_plot"]) and out.loc[2, "ratio_plot"] == 1.5
    assert np.isnan(out.loc[5, "ratio_plot"])          # tratado al final: sin vecino siguiente
    assert out.loc[[1, 3, 4], "ratio_plot"].tolist() == [1.0, 2.0, 1.5]
    assert out["ratio"].isna().sum() == 3              # la columna del modelo no cambia
