# Modelo de datos: segmentación ABC × XYZ y pronóstico de participación (M5)

Los esquemas de los archivos en disco están en [contracts/artifacts.md](contracts/artifacts.md).
Este archivo describe las entidades, sus relaciones, las reglas de validación y el ciclo de vida.

## Entidades

| Entidad | Clave | Campos | Validación / reglas (FR) |
|---|---|---|---|
| **Venta diaria** (cruda) | `store_id, item_id, d` | `units` | Entero ≥ 0; si no, se rechaza (FR-002). Se guarda como arreglo ancho `units[series, day]`, con NaN = enmascarado. |
| **Precio semanal** (crudo) | `store_id, item_id, wm_yr_wk` | `sell_price` | > 0. Faltante o ≤ 0 tras el lanzamiento con unidades > 0 → rechazo (FR-002). |
| **Día de calendario** (crudo) | `d` | `date, wm_yr_wk, weekday, events, snap_*` | Fechas continuas y únicas, un `wm_yr_wk` por `d`; si no, el proceso se detiene (FR-003). |
| **Lanzamiento** | `store_id, item_id` | `launch_wk, launch_d`; a nivel artículo: `item_launch_d = mín. entre tiendas` | Primera semana con precio (FR-004). |
| **Artículo agregado** | `item_id` | `launch_d, usage, spend, cv, abc_class, xyz_class, cell` por conjunto de cálculo | Gasto = Σ unidades × precio diario (FR-010). |
| **Conjunto de cálculo** | `set_id` | `kind ∈ {train_global, year, segment, year5}`, `start_d, end_d` | Train global = Y1–Y4. Y5 solo se permite para `eda` y `year5_replication`. |
| **Matriz ABC × XYZ** | `set_id, item_id` | `spend, cum_spend, cum_pct_before, abc_class, cv, xyz_class, cell, exclusion_reason` | ABC por `cum_pct_before < cut`, empates resueltos por `item_id` (FR-011). Umbrales XYZ según FR-013. |
| **Referencia** | `comparison (Y2, Y3, Y4, Y5), item_id` | `ref_spend_mean, ref_abc, ref_cv_mean, ref_xyz, n_ref_years` | Media sobre los años completos activos previos (FR-016). Para Y5, los años previos son Y1–Y4 (FR-030). |
| **Medida de estabilidad** | `comparison (Y2, Y3, Y4 en train; Y5 en la réplica), dimension (abc, xyz)` | `agreement_pct_vs_ref, agreement_pct_vs_global, psi, psi_band` | PSI con ε = 1e-4. Umbrales tomados de la config (FR-017). |
| **Estabilidad por artículo** | `item_id` | por comparación: `same_abc, same_xyz, same_cell`; `a_every_year`; `stable` (bool), `rule` | Usa `stability_rule` (FR-018). |
| **Producto objetivo** | `item_id` | `abc_class = A, xyz_class, cell, spend_train, spend_share, treatment` | Debe ser A y estable; celda sin tratamiento → se detiene; lista vacía → se detiene (FR-019). |
| **Configuración** | nombre del parámetro | valor, valor por defecto, rango | Un único archivo TOML más historial (FR-006–FR-008). |
| **Registro de cambio** | fila | `date, parameter, old_value, new_value, reason` | Toda diferencia frente al snapshot necesita una fila. |
| **Registro de eventos** | fila | `reason, source, row_id, column, detail` | Rechazos, eventos informativos y fallos de ajuste (`item_id, candidate, message`). |
| **Serie de participación** | `item_id, d` | `units, chain_total, share` | `chain_total = 0` → día excluido y registrado. |
| **Réplica del año 5** | `item_id` (productos objetivo) | `frozen_cell, ref_abc, ref_xyz, y5_abc, y5_xyz, y5_cell, same_abc, same_xyz, same_cell, same_as_frozen` | Calculada con los parámetros congelados y después de declarar `[forecast]`. Solo diagnóstico: no modifica la lista ni el tratamiento, y ninguna etapa de selección la lee (FR-030). |
| **Candidato** | `candidate_id` | `family, hyperparams` | Declarado en config antes de la réplica en el año 5; cambiarlo después detiene el backtesting (FR-023). |
| **Modelo congelado** | `selection_unit` | `candidate_id, order, seasonal_order, params[]` | Ajustado solo con train (FR-024). |
| **Pronóstico** | `model_id, item_id, origin (1–13), h (1–28)` | `d, share_hat, units_hat, units_actual` | Sin reajuste (FR-025). Las unidades usan el total real de la cadena (FR-026). |
| **Resultado de evaluación** | `model_id` | `wrmsse (media de 13), wrmsse_by_origin[], wape, rmse, beats_snaive` | Se acepta solo si `wrmsse < snaive` (FR-027). |

## Relaciones

```text
Día de calendario 1─* Venta diaria *─1 (tienda,artículo) ─* Precio semanal (vía wm_yr_wk)
(tienda,artículo) ─1 Lanzamiento ──► Artículo agregado.launch_d
Artículo agregado ─* Matriz ABC × XYZ (una fila por conjunto de cálculo)
Matriz ABC × XYZ (año) + Referencia ──► Estabilidad por artículo ──► Producto objetivo (+treatment_map)
Producto objetivo + parámetros congelados ──► Réplica del año 5 (solo diagnóstico)
Producto objetivo 1─1 Serie de participación ─► Modelo congelado ─► Pronóstico ─► Resultado de evaluación
```

## Ciclo de vida

Las etapas se ejecutan en este orden (constitución v1.1.0; research R13).

| Etapa | Precondición | Estado de salida |
|---|---|---|
| 1 Validar datos | Archivos crudos presentes, calendario correcto | `validated` (arreglos intermedios, registros) |
| 2 EDA de cinco años | Etapa 1 | `described`. Ninguna etapa de decisión lo lee. |
| 3 Matriz de train | Etapa 2, config válida | `frozen: matrix_train` |
| 4 Estabilidad y lista objetivo | `matrix_train` congelado, candidatos y criterios en el manifiesto | `frozen: targets, treatment, period, cuts, thresholds, cv_aggregation, stability_rule, treatment_map` |
| 5 Réplica en el año 5 | `targets` y `segmentation_params` congelados; claves de `[forecast]` (candidatos y `selection_level`) declaradas antes de leer Y5 | `frozen: matrix_y5, stability_y5, year5_replication` (lista sin cambios, hash verificado). Ninguna etapa de selección lo lee. |
| 6 Backtest | Etapa 5 terminada; declaraciones de `[forecast]` anteriores a `matrix_y5` e iguales a la config | `frozen: models` |
| 7 Evaluación en test | Marcas de congelamiento anteriores a la ejecución, el ledger no tiene entrada para `model_id` | `evaluated` (entrada en el ledger, resultados) |

Cambiar un parámetro después de la etapa 5 exige una nueva etiqueta de versión y volver a
ejecutar desde la etapa 3.
