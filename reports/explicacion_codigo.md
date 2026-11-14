# Qué hace el código

Explicación del código del proyecto. Para cada notebook: objetivo, funciones de `src/` que usa y lo que produce. Para cada módulo de `src/`: sus funciones. Lo saqué de leer el código y las celdas markdown; no ejecuté nada.

## 1. Visión general

- **Datos:** M5 (Walmart), 3.049 artículos × 10 tiendas, días `d_1`…`d_1941`. Train = años 1–4 (d_122–d_1577) y test = año 5 (d_1578–d_1941).
- **Unidad de pronóstico:** bloques de 28 días por artículo. El objetivo es el `ratio` = unidades del bloque / unidades del bloque anterior. Las unidades pronosticadas se obtienen como unidades del bloque anterior × ratio.
- **Flujo vigente (constitución 7.1.0):**
  `01` (intermedios) → **03** segmentación ABC×XYZ y lista objetivo (592 artículos AX/AY/AZ) → **03b** tratamiento de quiebres por tienda, features y datasets (517 artículos modelables) + SARIMAX → **03c** búsqueda Optuna de XGBoost/LightGBM/CatBoost → **03d** diagnóstico: la selección por artículo pierde frente a CatBoost para todos → **04** reentrena un CatBoost global con todo el train → **05** mide el test una sola vez (a un paso y encadenado) → **06** impacto en dinero (venta capturada, perdida y exceso).
- **Resultado en test** (`data/interim/nb05_test/manifest.json`), WRMSSE en unidades: CatBoost a un paso 0,495 · encadenado 13 bloques 0,748 · seasonal naive 0,868.
- **Persistencia:** cada notebook guarda en su carpeta de `data/interim/` o `models/`. Los notebooks siguientes cargan esas salidas sin recalcularlas. Los notebooks que guardan algo tienen un candado (`guard_rerun` + `RERUN`/`OVERWRITE`).

## 2. Notebooks en orden

### 01_data_validation (requisito previo; no está en la lista del flujo de modelado)
- **Objetivo:** leer los CSV de M5, enmascarar el prelanzamiento y los datos inválidos (NaN ≠ 0) y agregar por artículo. El encabezado interno dice "00 · Tablas de entrada".
- **Usa:** `data.load_sales`, `load_prices`, `build_daily_prices`, `compute_launch`, `apply_masks`, `mask_summary`, `audit_series`, `profile_frame`, `run_data_validation`, `save_data_validation`.
- **Produce:** `data/interim/*.npy|parquet` (`item_units`, `item_spend`, `chain_total`, `units`, `prices_daily`, `mask_reason`, `launch`, `rejections`, índices). Todo el flujo lee estos archivos con `load_interim`.

### 03_abc_xyz_train (canónico de la segmentación)
- **Objetivo:** segmentar ABC × XYZ solo con train, congelar las reglas, revisarlas en test y fijar los artículos que se modelan.
- **Etapas:**
  1. Ciclo de vida (primera a última venta).
  2. Prueba de rachas de Wald–Wolfowitz, como diagnóstico.
  3. `LONG_RUN` por artículo con un umbral Bernoulli (α = 0,01). Las rachas ≥ `LONG_RUN` son quiebres y pasan a NaN.
  4. Mínimo de días observados `m` por tercil de p₀, estimado con Monte Carlo.
  5. ABC por gasto (`select_abc_cuts`).
  6. CV crudo y limpio sobre bloques de 28 días. Cortes XYZ con Jenks sobre log(CV).
  7. Perfiles, radares y rachas por segmento.
  8. Congelar (`freeze_train`, en memoria con SHA-256).
  9. Universo y CV de test con las reglas congeladas (2.886 artículos).
  10. Selección de AX/AY/AZ de train.
- **Usa de `src/`:** `config.load_validated_config`, `all_years`, `train_end`; `data.load_interim`, `active_stores_by_item`, `load_calendar`; `segmentation.zero_runs`, `zero_breakdown`, `select_abc_cuts`, `abc_rule_kwargs`, `classify_xyz`, `describe_items`; `freeze.freeze`, `load_frozen`.
- **Particularidad:** define unas 57 funciones locales en el propio notebook (`lifecycle_bounds`, `trim_lifecycle`, `simulate_block_error`, `jenks_cuts`, `cv_from_blocks`, `select_targets`, …).
- **Produce:** solo `data/processed/targets.parquet`, registrado en `manifest.json` con `freeze`. Tiene 592 artículos (AX 207, AY 251, AZ 134) con `cell`, `abc_class`, `xyz_class`, `spend_*`, `cv_train`, `long_run`, `min_days` y `treatment`.

### 03b_tratamiento_por_tienda
- **Objetivo:** detectar y completar los quiebres por **tienda × día** antes de sumar al artículo. Después aplica el pipeline por artículo (del antiguo nb 04) sobre esa suma y arma los datasets.
- **Etapas:**
  1. Carga de `targets` y de las series por tienda (`store_units`). La línea base es el pipeline por artículo (`item_block_pipeline`).
  2. Ciclo de vida por tienda (`lifecycle_bounds`).
  3. `long_run` por tienda (`store_zero_thresholds`).
  4. Quiebres sin usar el futuro (`flag_store_breaks`).
  5. Completar los días en quiebre con la venta reciente de la tienda (`fill_store_breaks`, `add_store_fill`).
     - 5.2: `share_deficit_test` se probó y se descartó.
     - 5.3: las rachas largas se recortan cuando hay caída generalizada (`long_break_runs`, `cut_item_bounds`) o se excluyen cuando son aisladas.
  6. Suma por artículo.
  7. Bloques de 28 días (`aggregate_blocks`), relleno de bloques no válidos, primer bloque parcial y atípicos (`fill_invalid_blocks`, `drop_partial_first_block`, `fill_atypical_blocks`), partición (`assign_windows`) e imputación de casos (`classify_extreme_ratios`, `impute_history_cases`, `replace_after_history`).
  8. Calendario y eventos (`block_calendar`, `calendar_lags`).
  9. Objetivo y features (`add_ratio`, `treat_ratio`, `add_lags`, `add_rolling`, `interpolate_treated`).
  10. Datasets (`sarimax_dataset`, `boosting_dataset`).
  11. Folds (`assign_folds`, `fold_masks`).
  12. Seasonal naive m = 13 y escala del RMSSE (`seasonal_naive`, `seasonal_naive_scale`).
  13. SARIMAX por artículo (`seasonal_acf`, `search_sarimax_checkpointed`, `load_sarimax_checkpoints`).
- **Produce:**
  - `data/interim/nb03b_datasets/`: `sarimax_data`, `boosting_data`, `snaive_valid`, `rmsse_scale` y `manifest.json` con las 33 variables del boosting (`cat_features` = `item_id`, `cell`, `period`). Tiene 517 artículos y 13 folds.
  - `data/interim/nb03b_sarimax/`: lotes de la búsqueda SARIMAX.

### 03c_tratamiento_por_tienda (búsqueda del boosting)
- **Objetivo:** elegir los hiperparámetros de XGBoost, LightGBM y CatBoost globales por su WRMSSE en 13 folds (h = 1, ventana expansiva). Cada uno pasa por piloto, 100 pruebas TPE y repetición del backtest de la mejor prueba.
- **Usa:** `data.load_tables`, `load_interim`, `load_calendar`, `save_pickle`, `guard_rerun`; `metrics.spend_weights`, `wrmsse_frame`; `modeling.boosting_search`, `boosting_backtest`; `features.BLOCK_DAYS`.
- **Produce:** `data/interim/nb03c_boosting/<algoritmo>.db` (estudio Optuna en SQLite, retomable) y `<algoritmo>_backtest.pkl` (`meta` + pronóstico de la ventana).
- **Resultado:** CatBoost 0,506 (prueba 43) · XGBoost 0,507 · LightGBM 0,511 · seasonal naive 0,894.
- **Nota:** el nombre del archivo repite el del 03b, aunque el contenido es la búsqueda del boosting.

### 03d_seleccion_por_articulo (diagnóstico)
- **Objetivo:** elegir el modelo por artículo entre seasonal naive, SARIMAX y CatBoost, con el filtro "RMSSE < seasonal naive", y comprobar si esa selección es optimista.
- **Usa:** `metrics.train_weight_windows`, `spend_weights`, `wrmsse_frame`, `item_errors`, `segment_errors`, `paired_bootstrap_wrmsse`; `modeling.load_sarimax_checkpoints`, `window_forecasts`, `select_by_item`, `selected_forecast`.
- **Resultado:** con 13 folds la selección da 0,481. Si se elige con los folds 1–7 y se mide en los 8–13, la selección da 0,517 frente a 0,483 de CatBoost para todos (IC 95 % [0,009; 0,056]). De ahí se decidió un único CatBoost global (constitución 7.0.0).
- **Produce:** solo `data/interim/nb03d_selection/selection.csv`, con `winner` y `winner_catboost_all`. Ningún notebook posterior lo carga.

### 04_feature_engineering_y_seleccion_de_modelo (reentrenamiento)
- **Objetivo:**
  - Cargar los datasets del 03b y la prueba 43 del 03c.
  - Reproducir exactamente el backtest del 03c (diferencia 0, WRMSSE 0,5062).
  - Comparar en la ventana el pronóstico a un paso (0,506) con el encadenado de 13 bloques (0,742).
  - Reentrenar con todo el train (`split == train`, ratio no nulo) y calcular la importancia SHAP.
  - Guardar el modelo.
- **Usa:** `modeling.boosting_backtest`, `make_booster`, `fold_masks`, `recursive_ratio_forecast`, `chain_units`, `plot_forecast_item`, `catboost_shap`, `shap_importance`, `plot_shap_importance`, `save_catboost_model`, `load_catboost_model`; `metrics.item_errors`, `segment_errors`, `seasonal_naive_scale`, `wrmsse_frame`, `spend_weights`, `train_weight_windows`. Define localmente `level_errors`.
- **Produce:**
  - `models/nb04_catboost/catboost_global.cbm` y `meta.json` (variables, hiperparámetros, semilla 42, 6 hilos, filas de ajuste, fuentes y versiones).
  - Figuras en `reports/figures/nb04_catboost_backtest/`.

### 05_evaluacion_backtesting (test, una sola vez)
- **Objetivo:** medir el test (13 bloques por artículo) frente al seasonal naive.
  - **A un paso:** 13 orígenes, reentrenando en cada uno con `boosting_backtest` sobre `assign_test_folds`.
  - **Encadenado:** un solo origen con el modelo del nb 04 (`recursive_ratio_forecast` + `chain_units`).
  - Al final marca los artículos atípicos con `tukey_outliers`.
- **Usa:** `modeling.assign_test_folds`, `fold_masks`, `load_catboost_model`, `make_booster`, `boosting_backtest`, `recursive_ratio_forecast`, `chain_units`, `seasonal_naive`, `plot_forecast_item`; `metrics.seasonal_naive_scale`, `item_errors`, `segment_errors`, `tukey_outliers`, `spend_weights`, `train_weight_windows`; `data.save_tables`. Define localmente `level_errors`.
- **Produce:**
  - `data/interim/nb05_test/`: `forecasts`, `metrics`, `mae`, `item_metrics` y `manifest.json` (`accepted`, `wrmsse_units`).
  - Figuras en `reports/figures/nb05_test/<segmento>/`.

### 06_reporte_final (impacto en el negocio)
- **Objetivo:** valorar en dinero, a precio de venta y contra la demanda tratada, el pronóstico de CatBoost frente al seasonal naive y al pronóstico perfecto, en dos escenarios: A (a un paso) y B (encadenado).
  - Disponible = max(ŷ + z·σ, 0), con σ = RMSE de la ventana de backtesting y z de −1 a 3.
  - Calcula curvas exceso–venta perdida, lecturas al 95 % de servicio, comparación A frente a B e impacto por artículo.
- **Usa:** `data.load_tables`, `load_interim`, `save_tables`; `metrics.block_totals`, `coverage_by_z`, `service_reading`, `lost_at_excess`, `seasonal_naive_scale`; `modeling.plot_tradeoff_curves`, `plot_forecast_item`. Define localmente `scenario_readings`, `scenario_items` y `concentration`.
- **Produce:** `data/interim/nb06_impact/`: `curves`, `readings`, `ab`, `item_impact`, `by_segment`, `concentration` y `manifest.json`.
- **Cifras del manifiesto, escenario A:** al 95 % de servicio ahorra 3,28 M de exceso (77 %). Con el mismo exceso recupera 0,91 M de venta.

### Notebooks sin uso en el flujo de modelado
- `02_eda_five_years.ipynb`: EDA de cinco años (ABC/XYZ, estabilidad, estacionalidad). Usa `segmentation` y `stability`. No congela nada.
- `04_feature_engineering_y_seleccion_de_modelo_superseded.ipynb`: tratamiento por artículo y selección por artículo. Lo reemplazó la cadena 03b→04 en la constitución 7.0.0.

## 3. Módulos de `src/`

Convención en todos: arreglos anchos serie × día, con la columna j = día j + 1. NaN es una celda enmascarada, no un cero.

### `config.py` — configuración
Lee `src/config/params.toml` (secciones `[data]`, `[split]`, `[segmentation]`, `[stability]`, `[treatment_map]`, `[forecast]`, `[metrics]`, `[features]`) y lo valida contra `params_history.csv`.

| Función | Qué hace |
|---|---|
| `load_validated_config(path, history_path)` | Carga, valida rangos (`validate_config`) y exige que cada cambio tenga su fila en el historial (`check_history`). Todos los notebooks la llaman. |
| `train_years`, `train_end`, `all_years` | Rangos de días de train y test. |
| `forecast_keys` | Claves de `[forecast]` que deben declararse antes de la réplica (flujo viejo). |
| `get_key`, `flatten`, `load_config` | Utilidades de lectura. |

### `data.py` — carga, máscaras y E/S
| Función | Entrada → salida |
|---|---|
| `load_calendar`, `validate_calendar`, `week_of_day` | `calendar.csv` → tabla validada / `wm_yr_wk` por día. |
| `load_sales`, `load_prices`, `build_daily_prices` | CSV → arreglo ancho de ventas / precio por tienda-artículo-día. |
| `compute_launch`, `apply_masks` | Día de lanzamiento; enmascara prelanzamiento e inválidos y registra los rechazos. |
| `aggregate_items`, `active_stores_by_item` | Tienda-artículo → artículo + total de la cadena; tiendas con dato por día. |
| `run_data_validation`, `save_data_validation`, `save_interim`, `interim_from_aggregate` | Etapa 1 completa → `data/interim/`. |
| `load_interim(cfg, max_day, series_level)` | Lee los intermedios hasta `max_day`, sin leer días posteriores. |
| `assert_train_only` | Detiene el proceso si una etapa de decisión toca el test (flujo viejo). |
| `guard_rerun(paths, rerun)` | Candado: detiene el notebook si sus salidas ya existen. |
| `save_tables` / `load_tables` | Parquet + `manifest.json` en una carpeta, sin sobrescribir con contenido distinto salvo `overwrite`. Restaura las columnas category. |
| `save_pickle` | `to_pickle` con la misma protección. |
| `to_daily_frame`, `profile_frame`, `sales_overview`, `mask_summary`, `audit_series` | Diagnóstico y recortes a DataFrame. |

### `segmentation.py` — ABC, XYZ y ceros
| Grupo | Funciones |
|---|---|
| ABC | `compute_usage_spend`, `classify_abc`, `select_abc_cuts` (rejilla de cortes contra la regla de % de artículos y % de gasto), `classify_abc_share`, `abc_rule_kwargs` |
| XYZ | `compute_cv` (CV sobre bloques completos), `classify_xyz`, `find_cv_cuts`, `cv_cut_kwargs`, `sensitivity_table` |
| Matriz | `build_matrix`, `describe_items` |
| Rachas de ceros | `zero_runs`, `zero_breakdown`, `drop_zero_runs`, `longest_run_threshold` (umbral Bernoulli con α), `zero_run_thresholds`, `mask_zero_runs` |
| Ciclo de vida y quiebres (las usa el 03b) | `lifecycle_bounds`, `trim_lifecycle`, `find_zero_runs`, `classify_runs` (`break`/`demand`), `mask_breaks` |
| Estacionalidad (EDA) | `seasonal_strength`, `seasonal_strength_items`, `monthly_seasonal_index`, `classical_seasonal_index`, `friedman_seasonality` |
| Flujo viejo | `run_train_matrix` (Etapa 3 vieja: umbrales, cortes y congelamiento en `data/processed/`) |

### `stability.py` — estabilidad de clases (EDA y flujo viejo)
- **Matrices por año y periodo:** `year_matrices`, `period_matrices`, `segment_bounds`, `segment_bounds_by_year`.
- **Medidas de cambio de clase:** `agreement_pct`, `class_proportions`, `psi`, `psi_band` (solo EDA), `transition_stats` (kappa y Stuart–Maxwell), `base_target_classes`.
- **Plan de comparaciones pareadas:** `comparison_plan`, `paired_comparisons`, `prior_years_map`.
- **Línea base de ruido:** `noise_baseline`, `interleaved_halves`, `alternating_block_cv`.
- **Reglas y orquestadores viejos:** `build_item_stability`, `build_targets`, y las etapas `run_stability_stage`, `replicate_year5`, `run_replication`, `run_eda_five_years`.
- **Uso actual:** el flujo vigente no usa este módulo. Solo lo usa el nb 02.

### `features.py` — series de bloques y variables (pipeline por artículo)
| Función | Qué hace |
|---|---|
| `aggregate_blocks` | Unidades diarias → bloques de 28 días por artículo, con su participación en la cadena. |
| `fill_invalid_blocks` | Bloques con menos de `min_days` o muy bajos toman el valor del bloque anterior. |
| `drop_partial_first_block` | Quita el primer bloque si está incompleto. |
| `fill_atypical_blocks` | Rellena caídas que el artículo no repite en años anteriores. |
| `classify_extreme_ratios`, `impute_history_cases`, `replace_after_history` | Regla 4.4 "B+": clasifica los ratios < 0,5 o > 2 por causa, imputa el bloque culpable en la historia y aplica una regla causal en la ventana y el test. |
| `assign_windows` | Partición por artículo: los últimos 13 bloques son test y los 13 anteriores, la ventana de backtesting. |
| `block_calendar`, `calendar_lags`, `EVENT_TYPES` | Año, periodo y conteo de eventos por bloque, con sus rezagos. |
| `add_ratio`, `treat_ratio` | Objetivo `ratio`. Los ratios > `max_ratio` (y < `min_ratio_start` en el arranque) quedan vacíos. |
| `add_lags`, `add_rolling` | Rezagos 1, 2, 3 y 13; medias, desviación, mediana y rango móviles de 3, 6 y 13 bloques anteriores. |
| `interpolate_treated` | Columna `_plot`, solo para los gráficos. |
| `sarimax_dataset`, `boosting_dataset` | Tablas finales con los tipos que pide cada librería. |
| `flag_targets` | Une las clases congeladas a la lista de artículos. Solo la usa el nb 04 superseded. |

### `store_treatment.py` — tratamiento por tienda (nb 03b)
| Función | Qué hace |
|---|---|
| `store_units` | Unidades diarias tienda × artículo de los artículos pedidos. |
| `store_zero_thresholds` | `long_run` por tienda-artículo, estimado solo con train y con un valor de respaldo. |
| `flag_store_breaks` | Marca días-tienda en quiebre (`BREAK_KINDS`: `run_history`, `run_causal`, `excess_zeros`) sin usar el futuro. |
| `fill_store_breaks`, `add_store_fill` | Completa con la venta media reciente de la tienda y lo suma al artículo. |
| `long_break_runs`, `exclude_long_runs`, `cut_item_bounds` | Rachas largas: caída generalizada → mueve el inicio del artículo; racha aislada → excluye la tienda. |
| `store_block_summary` | Resumen tienda × bloque. |
| `item_block_pipeline` | Pasos 4.1–4.4 de `features.py` encadenados (línea base). |
| `extreme_blocks`, `classify_level_changes` | Diagnóstico de los ratios extremos. `classify_level_changes` usa el futuro y no es una regla. |
| `share_deficit_test`, `spread_block_extra` | Prueba tipo Friedman por tienda. Se probó y se descartó. |

### `modeling.py` — modelos, backtesting y persistencia del modelo
| Grupo | Funciones |
|---|---|
| Folds | `assign_folds` (13 folds de origen móvil en la ventana), `assign_test_folds` (13 orígenes del test), `fold_masks` |
| Línea base | `seasonal_naive(blocks, m)` → ŷ_t = unidades de hace m bloques |
| SARIMAX (sin pmdarima) | `ocsb_test`/`ocsb_crit` (D estacional), `kpss_d` (d), `sarimax_differencing`, `fit_sarimax`, `sarimax_order_grid` (búsqueda completa por AICc), `select_sarimax_item`, `sarimax_one_step`, `sarimax_backtest_item`, `search_sarimax_orders` (en paralelo), `search_sarimax_checkpointed` / `load_sarimax_checkpoints` (lotes), `seasonal_acf` |
| Boosting | `make_booster` (xgboost/lightgbm/catboost), `boosting_backtest` (reentrena en cada fold y pronostica el ratio a h = 1 → unidades), `suggest_params` + `boosting_search` (Optuna TPE con SQLite) |
| Selección (03d) | `window_forecasts`, `select_by_item` (filtro frente al seasonal naive), `selected_forecast` |
| Encadenado | `recursive_ratio_forecast` (13 bloques recalculando rezagos y ventanas con lo pronosticado), `chain_units` (Û(t+h) = U(t)·∏r̂) |
| Interpretación | `catboost_shap` (SHAP nativo de CatBoost), `shap_importance`, `plot_shap_importance` |
| Persistencia | `save_catboost_model` / `load_catboost_model` (`.cbm` + `meta.json`) |
| Gráficos | `plot_forecast_item`, `plot_tradeoff_curves` |

### `metrics.py` — métricas
| Función | Qué hace |
|---|---|
| `rmsse`, `wrmsse`, `wrmsse_frame` | RMSSE y su versión ponderada por gasto (no se usa MAPE). |
| `seasonal_naive_scale`, `naive_scale` | Escala del RMSSE. La primera es la vigente; `naive_scale` solo la usa `forecasting.py`. |
| `wape`, `rmse` | Diagnóstico. |
| `spend_weights`, `train_weight_windows` | Pesos = gasto del artículo en sus últimos bloques de train. |
| `item_errors`, `segment_errors` | Error por artículo y por segmento AX/AY/AZ/total. |
| `paired_bootstrap_wrmsse` | IC bootstrap pareado de la diferencia de WRMSSE (03d). |
| `tukey_outliers` | Artículos atípicos por segmento (05). |
| `block_totals` | Suma diaria → bloque, para el precio en el 06. |
| `coverage_by_z`, `service_reading`, `lost_at_excess` | Venta capturada, perdida y exceso por colchón z; lecturas sobre la curva (06). |

### `freeze.py` — congelamiento con hash
- `freeze(obj, name, path, cfg)` escribe un artefacto una sola vez y lo registra con SHA-256 en `data/processed/manifest.json`.
- `load_frozen(name)` lo carga y verifica el hash. `verify_manifest` revisa todos.
- `record_declarations`, `require_declarations` y `require_declared_before`: exigen que los parámetros se declaren antes de usarse (flujo viejo).
- `TestLedger`: registro de "una evaluación en test por modelo" (flujo viejo).
- El flujo vigente solo usa `freeze`/`load_frozen`, para `targets`.

### `forecasting.py` — flujo viejo (participación diaria)
- **Pronóstico:** participación diaria del artículo en la cadena (`share_series`) con candidatos SARIMAX/ARIMA/media móvil (`candidate_grid`, `fit_candidate`, `forecast_frozen`).
- **Backtest, test e informe:** `run_backtest`, `select_and_freeze`, `run_backtesting_stage`, `evaluate_test`, `run_test_evaluation_stage`, `final_report_tables`.
- **Uso actual:** ningún notebook lo importa. Solo lo cubren los tests.

### `src/tests/`
- 16 archivos pytest (`test_<módulo>.py`, más `test_eda`, `test_lifecycle_masking`, `test_measures`, `test_replication`, `test_reproducibility`) con fixtures sintéticas en `conftest.py`.
- Cubren casi todo `src/`, incluidos los módulos del flujo viejo.
- No verifiqué si pasan: no los ejecuté.

## 4. Puntos a notar

- **Código muerto o del flujo viejo:**
  - `forecasting.py` completo.
  - La mayor parte de `stability.py`; solo lo usa el EDA del nb 02.
  - En `segmentation.py`: `run_train_matrix`, `sensitivity_table`, `monthly_seasonal_index`.
  - En `freeze.py`: las declaraciones y `TestLedger`.
  - En `config.py`: `forecast_keys`.
  - En `metrics.py`: `naive_scale`.
  - En `features.py`: `flag_targets`.
  - Sus salidas antiguas siguen en `data/processed/`: `matrix_train.parquet`, `train_cuts.json`, `zero_thresholds.parquet`, `nb04_*.csv`.
- **Experimentos que siguen en el código:**
  - SARIMAX (`modeling.py`, ~la mitad del módulo), XGBoost y LightGBM: documentados y sin uso tras la constitución 7.0.0.
  - `select_by_item`/`selected_forecast`: solo los usa el 03d.
  - `share_deficit_test`/`spread_block_extra`: descartadas.
- **Duplicados:**
  - El nb 03 define localmente `to_daily_frame`, `lifecycle_bounds`, `trim_lifecycle`, `find_zero_runs`, `classify_runs`, `mask_breaks` y `block_calendar`, con los mismos nombres que en `src/` (`data`, `segmentation`, `features`). El código no es idéntico al de `src/`. No verifiqué si dan el mismo resultado.
  - `level_errors` está definida igual en los nb 04 y 05.
  - En `forecasting.py`/`stability.py` (orquestadores viejos) y en `features.py`/`store_treatment.py` (pasos 4.1–4.4) hay dos implementaciones del pipeline de bloques.
- **Dependencias entre módulos:**
  - `store_treatment` → `features`, `segmentation`.
  - `stability` → `segmentation`.
  - `forecasting` → `config`, `freeze`, `metrics`, `segmentation`.
  - `config` → `freeze`, dentro de una función (import local).
  - `freeze` → `config.get_key`.
  - `modeling` → `metrics`, `features` y `data._same_content`, con imports locales dentro de las funciones.
  - `data`, `features` y `metrics` no importan otros módulos de `src/`.
- **Nombres y docstrings desactualizados:**
  - `03c_tratamiento_por_tienda.ipynb` contiene la búsqueda del boosting.
  - `05_evaluacion_backtesting` es la evaluación en test.
  - `06_reporte_final` mide el impacto en el negocio.
  - El docstring de `features.py` habla de "nb 04 (etapas 2 a 9)" y el de `modeling.py` de "selección por artículo": son la numeración y el plan del nb 04 superseded.
- **Cadena de datos:** `data/interim/` (nb 01) → `targets.parquet` (03) → `nb03b_datasets` (03b) → `nb03c_boosting/catboost_backtest.pkl` (03c) → `models/nb04_catboost` (04) → `nb05_test` (05) → `nb06_impact` (06). `nb03d_selection` y `nb03b_sarimax` no alimentan al 04 ni a los siguientes.
