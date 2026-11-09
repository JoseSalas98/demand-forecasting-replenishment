# nb 04 (superseded) · Registro de etapas

> **Superseded (constitución 7.0.0, 2026-11-09).** Sin uso; lo reemplaza el nuevo nb 04 (`nb04-etapas.md`).

Notebook: `notebooks/04_feature_engineering_y_seleccion_de_modelo_superseded.ipynb`.
Reglas de trabajo: ver `CLAUDE.md` (R1–R8). Decisiones de la Etapa 0: spec.md, Session 2026-11-02,
y constitución 6.0.0 (objetivo = ratio de unidades entre bloques).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas

- [x] **0 · Revisión y plan.** Aprobada el 2026-11-02. Constitución 3.0.0 y spec actualizados.
- [x] **1 · Carga de datos crudos.** Aprobada el 2026-11-02. Ventas, calendario y precios: shapes,
  rango de fechas, tipos y nulos. `src/data.py`: `sales_overview` (nueva) y su test. Decisiones:
  semilla en `params.toml` desde la 6c; desde la Etapa 3 se usa `load_interim`.
- [x] **2 · Artefactos congelados del nb 03.** Aprobada el 2026-11-02. Clases por artículo desde `targets.parquet` (merge
  por `item_id`), `long_run` y `min_days`; conteos iguales a los del nb 03 (592: AX 207, AY 251,
  AZ 134). No se persisten los cortes. `src/features.py` (nuevo): `flag_targets` y
  `src/tests/test_features.py`.
- [x] **3 · Tratamiento de ceros y filtro (diario).** Aprobada el 2026-11-02 (constitución
  4.0.0): ciclo de vida de la primera a la última venta (se recortan ceros iniciales y finales);
  quiebres enmascarados con `long_run` congelado (no alteran la suma del bloque). Verificado contra
  el nb 03. `src/segmentation.py`: `lifecycle_bounds`, `trim_lifecycle`, `find_zero_runs`,
  `classify_runs`, `mask_breaks`; `src/data.py`: `to_daily_frame`; tests en
  `src/tests/test_lifecycle_masking.py`.
- [x] **4 · Bloques de 28 días, objetivo y partición por artículo.** Aprobada el 2026-11-02. Valor del bloque = suma de
  unidades; participación = artículo ÷ cadena en el bloque; serie del bloque de la primera venta
  al de la última. Partición por artículo: test = últimos 13 bloques, validación = 13 anteriores,
  train = resto (los 13 bloques previos al test son la ventana de backtesting, `backtest = True`); mínimo 39 bloques → 580 artículos (12 excluidos). `src/features.py`:
  `aggregate_blocks`, `assign_windows` y sus tests.
  - [x] Corrección aprobada el 2026-11-02 (constitución 6.0.0). Se quita `share`. 4.2 · Bloque no
    válido (días observados < `min_days` congelado, incluido el de 0 días) toma el valor del bloque
    anterior: 1,215 bloques en 219 artículos; los bloques iniciales no válidos se eliminan (95 en
    50 artículos). 4.3 · Partición sobre la serie recortada: 579 artículos (AX 203, AY 248,
    AZ 128; 58,87 % del gasto de train), 13 excluidos (sale HOUSEHOLD_1_457). Se miden todos los
    bloques. `src/features.py`: `aggregate_blocks` (sin `share`), `fill_invalid_blocks` (nueva,
    reemplaza a `impute_gaps`) y sus tests.
  - [x] 4.2 ampliada el 2026-11-02 (constitución 6.2.0, tras revisar ratios > 10 en la Etapa 11):
    bloque válido que vende < 10 % de la mediana de sus últimos 6 bloques válidos = quiebre
    escondido (`invalid_reason = low_units`; bloque 1 no se evalúa). 960 bloques (AX 0,2 %,
    AY 1,8 %, AZ 8,1 %); ratios > 10 en train: 272 → 33. Siguen 579 artículos. Parámetros
    `low_block_share = 0.1`, `low_block_window = 6` con historial. `src/features.py`:
    `fill_invalid_blocks` (modificada) y su test.
  - [~] 4.2a · (escrita 2026-11-04, decisión del usuario) Primer bloque incompleto (< 28 días con
    datos): se quita, junto con los bloques no válidos que lo copiaron, y la serie empieza en el
    siguiente bloque válido; se conserva si quitarlo deja < 39 bloques (FOODS_3_499 y
    HOUSEHOLD_1_338 siguen en la modelación). 157 artículos recortados (193 bloques); ratios > 2 en
    posiciones 2–7: 146 → 74 (posición 2: 79 → 20). Siguen 579 artículos. `src/features.py`:
    `drop_partial_first_block` (nueva) y su test.
  - [~] 4.2b · (escrita 2026-11-04, decisión del usuario) Bloques atípicos: bloque válido con
    unidades < 50 % de la mediana de sus últimos 6 bloques válidos (mínimo 3). Si el mismo periodo
    no estuvo deprimido en ninguno de los 3 años anteriores → `atypical`, valor = mediana reciente ×
    factor (mediana de los niveles de ese periodo en los años anteriores, tope 1; corregido el
    2026-11-04: con el año más reciente sin tope la regla creaba 29 saltos); sin año anterior → `atypical_first_year`, valor del bloque
    anterior; si ya estuvo deprimido antes → patrón, no se toca. Aplica a todos los artículos.
    Con la 4.2a: 950 bloques completados (702 atypical, 248 first_year); ratios 2–10: 719 → 336,
    > 10: 34 → 15;
    siguen 579 artículos. Parámetros `atypical_level = 0.5`, `atypical_min_ref = 3`,
    `atypical_years = 3` con historial. `src/features.py`: `fill_atypical_blocks` (nueva) y tests.
  - [~] 4.4 · (escrita 2026-11-04, opción B+ del usuario) Casos atípicos. `classify_extreme_ratios`
    separa los ratios extremos (> 2 o < 0,5) en estructurales (no se tocan), atípicos, propagados y
    arranque. Historia: `impute_history_cases` imputa el bloque culpable (mediana de 6 válidos
    previos; t+1 en la historia, sin fuga). Ventana y test: 4.2b causal con E1 (0,8 con días
    enmascarados), E7 (patrón = 2 de 3 años) y E8 (no válido tras bloque < 80 % toma la mediana),
    vía `replace_after_history`. Arranque: E11 en la 6a (ratio < 0,5 vacío en posiciones ≤ 7).
    Atípicos + propagados 309 → 122 (AX 22 → 6, AY 126 → 39, AZ 161 → 77); 697 bloques tocados
    (96 historia, 319 ventana, 282 test). Proxy: RMSSE 0,5400 → 0,5344 (AX y AY mejoran; AZ +0,007,
    IC ancho: vigilar en Etapas 12–13). 8 parámetros nuevos con historial. `src/features.py`:
    `fill_atypical_blocks` (E1/E7/E8), `classify_extreme_ratios`, `impute_history_cases`,
    `replace_after_history` (nuevas), `treat_ratio` (E11); tests.
    Cierre de la reducción (2026-11-04, usuario): quedan 143 atípicos (AX 6, AY 41, AZ 96): 89 no
    detectables sin mirar el futuro en ventana y test y 21 creados por las reglas, más 33 en la
    historia. Se probaron y descartaron: tope a la imputación de la historia (110 → 98, impacto no
    significativo) y comparación con el ratio mediano del segmento (toca 67–77 estructurales y
    reescribe cientos de bloques). Abierto para la Etapa 14: medir contra unidades observadas o
    corregidas en ventana y test (B+ cambia 601 bloques ahí).
- [x] **5 · Variables de calendario.** Aprobada el 2026-11-02. Año entero (1–5) y periodo 1–13 ordinal, ya unidos a
  `blocks_model`. Eventos (decisión del usuario): cantidad de eventos de cada tipo en el bloque
  (`n_religious`, `n_national`, `n_cultural`, `n_sporting`), unidos a `blocks_model`. Sin precio ni
  SNAP (decisión del usuario). `src/features.py`: `block_calendar`, `EVENT_TYPES` y su test.
- [x] **6 · Feature engineering (boosting).** Aprobada el 2026-11-02.
  - [x] 6a · (aprobada 2026-11-02) Rezagos y ventanas del objetivo (shift(1) antes de rolling). Parámetros en
    `params.toml [features]` con historial: `target_lags = [1, 2, 3, 13]`, `rolling_windows =
    [3, 6, 13]`, `rolling_stats = [mean, std, median, range]`, `event_lags = [1, 2]`. Rehecha
    (2026-11-02) sobre el objetivo `ratio` = unidades_t / unidades_{t−1}, sin transformar
    (constitución 6.0.0). `src/features.py`: `add_ratio` (nueva) y su test. `src/features.py`: `add_lags`,
    `add_rolling`; `src/config.py`: validación de `[features]`.
  - [x] 6b · (aprobada 2026-11-02) Rezagos t−1 y t−2 de los conteos de eventos por tipo (`n_religious`, `n_national`,
    `n_cultural`, `n_sporting`), calculados sobre el calendario (sin nulos: usa los días previos a
    d_122). 8 columnas. Adelanto t+1 descartado por el usuario (2026-11-02). `src/features.py`:
    `calendar_lags` (nueva) y su test.
  - [x] 6c · (aprobada 2026-11-02) Reglas comunes: cálculo antes del split (por artículo, serie completa); chequeo
    manual de 200 filas al azar (ratio_lag1, ratio_lag13, ratio_roll3_mean, ratio_roll13_std,
    n_national_lag1): 200/200 coinciden; nulos de arranque por columna y tramo (test sin nulos).
    Semilla `features.seed = 42` en `params.toml` con historial; `src/config.py`: validación de
    `seed`.
  - [x] 6d · Sin efecto (2026-11-02): el objetivo es el ratio sin transformar (decisión del
    usuario, constitución 5.0.0).
  - [~] 6e · (escrita 2026-11-03; propuesta B de la Etapa 11) Edad del artículo `age` = `pos`
    (bloques desde el primer bloque válido), solo boosting, sin tope. Entra a `BOOST_FEATURES`
    (Etapa 9: 33 variables). Sin código nuevo en `src/`.
- [x] **7 · Split temporal.** Aprobada el 2026-11-02. `train_set` (28,391 filas, con la ventana de backtesting marcada) y
  `test_set` (7,527 filas), 579 artículos en ambos. Memoria verificada en las dos fronteras para
  los 579 artículos (units_prev, ratio_lag1, ratio_lag13, ratio_roll3_mean). Sin código nuevo en
  `src/`.
- [x] **8 · Análisis bivariado.** Aprobada el 2026-11-02. PENDIENTE de definir: solo una celda markdown (`04-e8-md`).
- [x] **9 · Datasets por modelo.** Aprobada el 2026-11-02. SARIMAX: ratio + 4 conteos de eventos del bloque t (sin
  rezagos, ventanas, año ni periodo; decisión del usuario). Boosting: 32 variables, `item_id`,
  `cell` y `period` como `category`. 35,339 filas cada uno (sin la primera fila de cada artículo).
  Export a CSV en `data/processed/` (decisión del usuario: CSV, no parquet ni Excel).
  `src/features.py`: `sarimax_dataset`, `boosting_dataset` y su test.
- [x] **10 · Folds.** Aprobada el 2026-11-02. 13 folds de origen móvil sobre los 13 bloques de la ventana de cada artículo,
  ventana expansiva, h = 1 (por posición en la ventana; 576 artículos comparten fecha). 579 filas
  de validación por fold; entrenamiento de 20,285 (fold 1) a 27,233 (fold 13). Tabla y gráfico.
  `src/modeling.py` (nuevo, aprobado en la Etapa 0): `assign_folds`, `fold_masks`;
  `src/tests/test_modeling.py`. Sin early stopping: `n_estimators` entra como hiperparámetro en la Etapa 13 (decisión del usuario, constitución 6.1.0).
- [~] **11 · Seasonal naive.** En unidades: ŷ_t = unidades_{t−13} (decisión del usuario). Escala del
  RMSSE = media de (y_t − y_{t−13})² con todo el train del artículo (decisión del usuario, como
  M5). Error por fold (MAE, RMSE, WAPE) y RMSSE por artículo (mediana ≈ 0,92). `snaive_m` 7 → 13
  con historial. `src/metrics.py`: `seasonal_naive_scale` (nueva); `src/modeling.py`:
  `seasonal_naive` (nueva); tests; `test_forecasting.py` (flujo viejo) lee `snaive_m`.
  11.1: gráficos de 10 artículos al azar (semilla), ratio y unidades, real vs seasonal naive, con
  su error en la ventana en la leyenda (solo train). Ejemplo FOODS_3_586 con error absoluto y
  porcentual por bloque (diagnóstico).
  PENDIENTE (2026-11-02): ratios 2–10 en train = 589 (causas: bloque anterior con días
  enmascarados 202, primeros 6 bloques 112, anterior completado 110, anterior < 50 % 61, normal 104).
  Propuesta A: unidades × 28 ÷ días observados en bloques válidos (regla general, no por rango de
  ratio; requiere enmienda a la constitución): 2–5 479→377, 5–10 110→37, >10 33→25, p99 3,3→2,43.
  Propuesta B: variable edad del artículo en bloques. Esperando decisión del usuario.
  DECISIÓN (2026-11-03): A descartada (el usuario: hace más difícil identificar las afectaciones;
  el objetivo no se toca). B aprobada → 6e. Debate de 5 propuestas con protocolo común (boosting
  proxy sklearn, 13 folds, 3 semillas): gana dejar los ratios > 2 fuera del ajuste.
  CORRECCIÓN (2026-11-04, usuario): el tratamiento va AGUAS ARRIBA, antes de rezagos y ventanas.
  6a (escrita): `treat_ratio` deja `ratio` vacío si `ratio_raw` > `features.max_ratio = 2.0`
  (con historial); `ratio_raw`, `ratio_treated` y `denominator_cause` identifican la afectación;
  la fila se pronostica y se mide. Ventanas: el bloque tratado cuenta como historia y su valor se
  omite (decisión del usuario). Tras la 4.2a y la 4.2b: 337 filas tratadas (182 train previo,
  55 ventana, 100 test); train estricto: ratio > 2 en 0,46 % (denominador limpio) vs 2,34 % (afectado). Datasets (9)
  conservan las filas tratadas. 11.1/11.2: una sola leyenda, etiquetas cortas, error en % (WAPE).
  La 11.3 se eliminó (absorbida por la 6a). `src/features.py`: `treat_ratio` (nueva),
  `add_rolling` (`treated_col`), `sarimax_dataset`/`boosting_dataset` (filtran por `units_prev`,
  `DATASET_KEYS`); `src/config.py`: validación de `max_ratio`; tests. `fit_mask` retirada.
  Diagnóstico por segmento de los ratios > 2 (2026-11-04): arranque (AX), cambios de nivel
  legítimos (AZ), denominador con días enmascarados o entre 50 y 80 % (todos). Pendiente: decidir
  si la 6a debe vaciar los cambios de nivel legítimos y qué hacer con t−1 con días enmascarados,
  Separación (2026-11-04, ratios > 2 y < 0,5, 576 casos): estructurales 188 (subidas 123, bajadas 65),
  atípicos 294 (157 ligados a días de quiebre), propagados por reglas 15, arranque 79 (solo train).
  sensibilidad del umbral en SARIMAX (Etapa 12) y posible enmienda de la constitución.
- [ ] **12 · SARIMAX.** Órdenes por AIC con los bloques de train (pmdarima o búsqueda propia); coeficientes
  reestimados en cada fold. Piloto antes (R7).
- [ ] **13 · Boosting.** 13a XGBoost, 13b LightGBM, 13c CatBoost (detenerse en cada una). Piloto
  con variante por artículo en 20–30 artículos frente al global.
- [ ] **14 · Filtro frente al seasonal naive y selección por artículo.** Con error de validación;
  tablas por modelo y por segmento.
- [ ] **15 · Reentrenamiento de los ganadores con todo lo anterior al test.** Sin evaluar en test (eso es del nb 05).
- [ ] **16 · Persistencia y cierre.** Modelos, hiperparámetros, órdenes, selección, métricas y
  metadatos; lo que queda para el nb 05.

- Enmienda de la constitución (Principio II): se pospone hasta terminar de reducir los atípicos (usuario, 2026-11-04).
