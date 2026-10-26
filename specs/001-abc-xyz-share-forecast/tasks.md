---
description: "Lista de tareas: segmentación ABC × XYZ y pronóstico de participación (M5)"
---

# Tareas: segmentación ABC × XYZ y pronóstico de participación (M5)

**Entrada**: documentos de diseño en `specs/001-abc-xyz-share-forecast/`
**Prerrequisitos**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Orden de etapas (obligatorio, constitución v1.1.0 y research R13)**: validación → EDA de cinco
años → matriz de train → estabilidad y lista objetivo (congeladas) → declaración de `[forecast]`
y réplica en el año 5 → backtesting → evaluación en test → informe. Las historias US1–US5 siguen
ese orden.

**Pruebas**: se incluyen porque plan.md y quickstart.md definen pytest con casos clave. Van en
`src/tests/` sobre datos sintéticos mínimos y se escriben antes de la implementación (deben fallar
primero).

**Convenciones (Principio III)**: identificadores de código en inglés (`snake_case`, clases en
`CamelCase`, constantes en `UPPER_SNAKE_CASE`); docstrings y comentarios en español con la
estructura descripción / parámetros con tipo / retorno. Ningún corte, umbral, periodo ni rango de
días queda fijo en el código: todo se lee de `src/config/params.toml`. Los notebooks solo
orquestan y muestran; la lógica vive en `src/`. Las funciones reciben los rangos de días como
argumentos para poder probarlas con datos sintéticos.

## Formato: `[ID] [P?] [Historia] Descripción`

- **[P]**: se puede ejecutar en paralelo (archivo distinto, sin dependencias pendientes)
- **[USn]**: historia de usuario de spec.md a la que pertenece la tarea

---

## Fase 1: Setup (infraestructura compartida)

**Propósito**: dependencias, configuración de herramientas y archivos de parámetros.

- [X] T001 Añadir dependencias con `uv add --group ds statsmodels pyarrow` y `uv add --group dev pytest`; confirmar que `pyproject.toml` y `uv.lock` quedan actualizados
- [X] T002 Ajustar `pyproject.toml`: en `[tool.ruff]` usar `src = ["src"]` e `include = ["pyproject.toml", "src/**/*.py"]`; en `[tool.ruff.lint.isort]` usar `known-first-party = ["src"]`; añadir `[tool.pytest.ini_options]` con `testpaths = ["src/tests"]` y `pythonpath = ["."]`. Ejecutar `uv sync --all-groups`; si flit falla porque no encuentra el módulo `main`, añadir `[tool.flit.module]` con `name = "src"`
- [X] T003 [P] Crear `src/config/params.toml` con el contenido exacto del bloque TOML de `contracts/params.md` (todas las claves y valores por defecto, con sus comentarios en español)
- [X] T004 [P] Crear `src/config/params_history.csv` con la cabecera `date,parameter,old_value,new_value,reason` y la fila inicial `2026-10-24,*,,initial,"valores por defecto iniciales"`
- [X] T005 [P] Crear `src/tests/__init__.py` y `src/tests/conftest.py` con fixtures sintéticos: `tiny_cfg` (copia de params.toml con rangos de días reducidos: 5 "años" de 56 días y test = último "año"), calendario con `wm_yr_wk` (semana empieza en sábado), 2 tiendas × 4 artículos con precios distintos por tienda y semana, un artículo lanzado a mitad del periodo en una sola tienda, una celda con unidades negativas, una con unidades no enteras y una semana sin precio con ventas

---

## Fase 2: Fundacional (prerrequisitos bloqueantes)

**Propósito**: configuración validada, congelamiento con manifiesto, datos M5 validados y las funciones de clasificación que comparten el EDA (US1), la matriz de train (US2), la estabilidad (US3) y la réplica en el año 5 (US4).

**⚠️ CRÍTICO**: ninguna historia puede empezar antes de terminar esta fase.

### Pruebas de la fase fundacional

- [X] T006 [P] Escribir `src/tests/test_config.py`: cada regla de rango de `contracts/params.md` falla con su entrada inválida (incluido `abc_cuts = [0.95, 0.80]`, que debe detener antes de calcular); una clave faltante falla; un valor distinto al snapshot sin fila en el historial falla; con la fila correspondiente pasa
- [X] T007 [P] Escribir `src/tests/test_freeze.py`: `freeze` registra SHA-256, `config_sha256` y `frozen_at`; un segundo `freeze` del mismo nombre en la misma `version` falla; `load_frozen` falla si el archivo cambió; `require_declarations` falla si una clave no está declarada, si su `declared_at` es posterior al inicio de la etapa o si su `value` difiere de la config actual; `require_declared_before(keys, "matrix_y5")` falla si alguna declaración es posterior al `frozen_at` de ese artefacto; `TestLedger` rechaza un `model_id` repetido
- [X] T008 [P] Escribir `src/tests/test_data.py`: calendario con fecha duplicada, faltante o `d` con dos `wm_yr_wk` detiene el proceso; los días previos al lanzamiento por tienda quedan NaN (no 0) y no se registran; `invalid_units`, `sales_before_first_price`, `invalid_price_with_sales` y `price_gap_no_sales` se registran con `reason, source, row_id, column, detail`; el gasto agregado es Σ unidades × precio por tienda-día y difiere de precio medio × unidades cuando los precios varían (US2-2); `chain_total` excluye celdas enmascaradas; `assert_train_only` falla con `max_day ≥ test_days[0]`
- [X] T009 [P] Escribir `src/tests/test_segmentation.py`: el artículo que cruza el 80 % es A (`cum_pct_before < cut`); los empates de gasto se ordenan por `item_id`; un CV de exactamente 25 % es Z y uno de 10 % es Y; media 0 o menos de `cv_min_blocks` bloques completos → sin XYZ con `cv_undefined`; con `block28` se descartan los bloques incompletos por lanzamiento o enmascarado; CV con ddof = 0; `daily` y `weekly` calculan sobre días y semanas `wm_yr_wk` completas; `cell` vale `A-` cuando falta XYZ; los artículos inactivos tienen `exclusion_reason = "inactive_in_set"`
- [X] T010 [P] Escribir `src/tests/test_measures.py`: `agreement_pct` ignora artículos sin clase en alguno de los dos lados; `psi` con una clase vacía usa ε; `psi_band` en los límites 0,10 y 0,25; `year_matrices` produce un `set_id` por cada rango recibido

### Implementación de la fase fundacional

- [X] T011 Implementar en `src/config.py` la función `load_config(path: str = "src/config/params.toml") -> dict` con `tomllib`, y `validate_config(cfg: dict) -> None`, que exige todas las claves de `contracts/params.md` y aplica las reglas de rango textuales: "`0 < abc_cuts[0] < abc_cuts[1] < 1`", "`0 ≤ x < z`" para `xyz_thresholds`, "`psi_bands` ascendente, todos ≥ 0, `psi_epsilon > 0`", "`0 < agreement_min ≤ 1`", "Bloques contiguos y sin solapamiento. Todo día de train < `test_days[0]`. Longitud de `test_days` = `test_origins × horizon`", enumeraciones conocidas (`cv_aggregation ∈ {block28, daily, weekly, segment_period}`, `stability_rule ∈ {abc_all, abc_xyz_all, cell_majority, a_every_year}`, `segment_candidates ⊆ {quarter, four_month}`), valores de `treatment_map` como cadenas no vacías, ventanas y órdenes enteros no negativos y `backtest_folds ≥ 1`. Cada fallo lanza `ValueError` con el nombre de la regla
- [X] T012 Implementar en `src/config.py` la función `check_history(cfg: dict, snapshot: dict | None, history_path: str = "src/config/params_history.csv") -> None`: aplana las claves con puntos (p. ej. `segmentation.abc_cuts`); sin snapshot exige la fila `parameter = "*"`; con snapshot, cada clave cuyo valor difiera exige una fila con ese `parameter` y `new_value == json.dumps(valor)`; si no, lanza `RuntimeError`. Añadir `load_validated_config()`, que encadena `load_config`, `validate_config` y `check_history` contra `config_snapshot` del manifiesto (depende de T013)
- [X] T013 Implementar en `src/freeze.py`: `sha256_file(path) -> str`; `load_manifest()` y `save_manifest(manifest)` sobre `data/processed/manifest.json` con la estructura de `contracts/artifacts.md` (`version`, `declarations`, `artifacts`, `config_snapshot`); `record_declarations(keys: list[str], cfg: dict) -> None` (guarda `key`, `value` y `declared_at` UTC ISO-8601; si la clave ya está declarada en la misma `version` con otro valor, lanza `RuntimeError`); `require_declarations(keys, stage_start: datetime, cfg: dict) -> None` (cada clave declarada, `declared_at < stage_start` y `value` igual al de `cfg`); `require_declared_before(keys, artifact_name: str) -> None` (cada `declared_at` anterior al `frozen_at` del artefacto); `freeze(obj, name: str, path: str, cfg: dict) -> str`, que escribe DataFrame en Parquet (pyarrow, sin índice) o dict/list en JSON (`sort_keys=True`), falla si `name` ya existe para la misma `version`, registra `sha256`, `config_sha256` y `frozen_at` y actualiza `config_snapshot`; `load_frozen(name: str)`, que verifica el hash y lanza `RuntimeError` si no coincide; `verify_manifest() -> None`, que verifica todos los hashes. Justificación de la abstracción en el docstring: (b) y (c)
- [X] T014 Implementar en `src/freeze.py` la clase `TestLedger` sobre `data/processed/test_ledger.json` con `has(model_id) -> bool` y `record(model_id, frozen_models_sha256, targets_sha256) -> None` (añade `evaluated_at` UTC; lanza `RuntimeError` si `model_id` ya existe, SC-007). Justificación en el docstring: (c), estado que persiste entre ejecuciones
- [X] T015 Implementar en `src/data.py` la función `load_calendar(cfg) -> pd.DataFrame` (lee `calendar.csv` de `cfg["data"]["raw_dir"]`) y `validate_calendar(calendar, last_day: int) -> None`: "Fechas continuas y únicas, un `wm_yr_wk` por `d`; si no, el proceso se detiene (FR-003)"; además, cada `wm_yr_wk` tiene ≤ 7 días y existen `d_1` a `d_{last_day}`. Lanza `ValueError`
- [X] T016 Implementar en `src/data.py` la función `load_sales(cfg) -> tuple[np.ndarray, pd.DataFrame]`: lee `cfg["data"]["sales_file"]` y devuelve `units` float64 de forma (series, `last_day`) y `series_index` (`store_id, item_id`), ordenado por `store_id, item_id`, sin pasar a formato largo; y `load_prices(cfg) -> pd.DataFrame` desde `sell_prices.csv`
- [X] T017 Implementar en `src/data.py` la función `build_daily_prices(prices, calendar, series_index) -> np.ndarray`, con la misma forma que `units`: precio por tienda-artículo-día según el cruce día → `wm_yr_wk` → `sell_price`, y NaN donde no hay precio
- [X] T018 Implementar en `src/data.py` la función `compute_launch(prices_daily, series_index, calendar) -> pd.DataFrame` con columnas `store_id, item_id, launch_wk, launch_d`: "Primera semana con precio (FR-004)"; NaN si nunca tiene precio
- [X] T019 Implementar en `src/data.py` la función `apply_masks(units, prices_daily, launch, series_index) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]`, que devuelve unidades y precios enmascarados (NaN) y el registro. Reglas de research R3: unidades no enteras o < 0 → celda NaN, `invalid_units`; unidades > 0 antes del primer precio → celda NaN, `sales_before_first_price`; precio faltante/≤ 0 tras el lanzamiento con unidades > 0 en la semana → tienda-artículo-semana NaN, `invalid_price_with_sales`, gasto sin imputar; precio faltante tras el lanzamiento con unidades = 0 → tienda-artículo-semana NaN, `price_gap_no_sales` (informativo); prelanzamiento sin precio → NaN sin registro. El registro tiene columnas `reason, source, row_id, column, detail` (`source` = archivo, `row_id` = fila en el CSV, `column` = `d_n` o `wm_yr_wk`)
- [X] T020 Implementar en `src/data.py` la función `aggregate_items(units, prices_daily, series_index) -> dict` con claves `item_ids` (ordenados), `item_units` y `item_spend` (artículos × día; nansum sobre las tiendas; NaN el día en que ninguna tienda del artículo tiene dato válido), `item_launch_d` (mínimo entre tiendas) y `chain_total` (día; nansum de todas las series). El gasto se calcula por tienda-día como unidades × precio antes de sumar; prohibido precio medio × unidades (Principio II)
- [X] T021 Implementar en `src/data.py` las funciones `save_interim(arrays: dict, tables: dict) -> None`, que escribe `data/interim/units.npy`, `prices_daily.npy`, `series_index.parquet`, `launch.parquet` y `rejections.parquet` (además de `item_units.npy`, `item_spend.npy`, `chain_total.npy`, `week_of_day.npy` e `item_index.parquet`), y `load_interim(cfg, max_day: int) -> dict`, que abre los `.npy` con `mmap_mode="r"` y devuelve solo las columnas de días `1..max_day`. Añadir `assert_train_only(max_day, cfg)`, que lanza `RuntimeError` si `max_day >= cfg["split"]["test_days"][0]`; lo usan las etapas de decisión (notebooks 03, 04 y 06)
- [X] T022 Implementar en `src/segmentation.py` la función `compute_usage_spend(item_units, item_spend, start_d: int, end_d: int) -> pd.DataFrame` con columnas `usage`, `spend` y `active` (al menos un día no NaN en el rango), con nansum sobre `[start_d, end_d]`; los días NaN (prelanzamiento o enmascarados) no cuentan. `usage_period = "computation_set"` significa el rango completo del conjunto
- [X] T023 Implementar en `src/segmentation.py` la función `classify_abc(item_ids, spend, cuts: list[float]) -> pd.DataFrame`, con columnas `item_id, spend, cum_spend, cum_pct_before, abc_class`. Orden por `spend` descendente y `item_id` ascendente; "ABC por `cum_pct_before < cut`, empates resueltos por `item_id` (FR-011)": A si `cum_pct_before < cuts[0]`, B si `< cuts[1]`, C en otro caso. Solo recibe artículos activos. Justificación en el docstring: (b)
- [X] T024 Implementar en `src/segmentation.py` la función `compute_cv(item_units, start_d, end_d, aggregation: str, min_blocks: int, week_of_day: np.ndarray | None = None, segment_bounds: list[tuple[int, int]] | None = None) -> pd.DataFrame`, con columnas `cv` y `n_blocks`. `block28`: bloques consecutivos de 28 días desde `start_d`; `daily`: cada día; `weekly`: semanas `wm_yr_wk` completas dentro del rango; `segment_period`: los segmentos de `segment_bounds`. Se descarta todo bloque con algún NaN. CV = std(ddof=0) / media × 100; NaN si la media es 0 o `n_blocks < min_blocks`
- [X] T025 Implementar en `src/segmentation.py` la función `classify_xyz(cv: pd.Series, thresholds: list[float]) -> pd.Series`: X si CV < `thresholds[0]`, Y si `thresholds[0] ≤ CV < thresholds[1]`, Z si CV ≥ `thresholds[1]`, NaN si CV es NaN. Justificación en el docstring: (b)
- [X] T026 Implementar en `src/segmentation.py` la función `build_matrix(set_id: str, agg: dict, start_d: int, end_d: int, seg: dict, week_of_day=None, segment_bounds=None) -> pd.DataFrame`, donde `seg` contiene `abc_cuts`, `xyz_thresholds`, `cv_aggregation` y `cv_min_blocks` (de `cfg["segmentation"]` o de `segmentation_params` congelado). Columnas de `contracts/artifacts.md`: `item_id, launch_d, usage, spend, cum_spend, cum_pct_before, abc_class, cv, xyz_class, cell, exclusion_reason` (más `set_id`). Cubre todos los artículos: inactivos → `exclusion_reason = "inactive_in_set"`; CV indefinido → `cv_undefined`; `cell = abc_class + (xyz_class or "-")`
- [X] T027 Implementar en `src/stability.py` la función `year_matrices(agg, year_ranges: dict[str, tuple[int, int]], seg: dict, week_of_day) -> pd.DataFrame`, que concatena un `build_matrix` por cada año recibido, y las funciones `agreement_pct(a: pd.Series, b: pd.Series) -> float` (% de artículos con la misma clase entre los que tienen clase en ambos), `psi(p_ref: dict, p_cur: dict, eps: float) -> float` (Σ (p_cur − p_ref) · ln(p_cur / p_ref), con ε en las proporciones 0) y `psi_band(value, bands) -> str` (`stable` < bands[0] ≤ `moderate` < bands[1] ≤ `shift`). Justificación en el docstring: (b), las usan el EDA, la estabilidad de train y la réplica
- [X] T028 Crear `notebooks/01_data_validation.ipynb`: `load_validated_config`, carga, `validate_calendar`, precios diarios, lanzamiento, máscaras, agregación y `save_interim`; mostrar conteos de rechazos por `reason` y verificar que cada rechazo tiene motivo, fuente y fila (SC-002)

**Checkpoint**: config validada, manifiesto operativo, arreglos intermedios y funciones de clasificación listos.

---

## Fase 3: Historia 1 — exploración descriptiva de cinco años (Prioridad: P1)

**Objetivo**: describir los productos con Y1–Y5 como primera etapa del análisis, sin alimentar ninguna decisión.

**Prueba independiente**: el notebook 02 genera `reports/eda_five_years/` sin escribir en `data/processed/` ni congelar nada; ningún notebook posterior de decisión (03, 04 y 06) lee esa carpeta (US1-2).

- [X] T029 [P] [US1] Escribir `src/tests/test_eda.py`: `describe_items` calcula la proporción de ceros solo sobre días activos (excluye NaN de prelanzamiento) y la media y la mediana de unidades diarias
- [X] T030 [US1] Implementar en `src/segmentation.py` la función `describe_items(agg, start_d, end_d) -> pd.DataFrame`, con columnas `item_id, active_days, zero_share, mean_daily_units, median_daily_units, total_units`
- [X] T031 [US1] Crear `notebooks/02_eda_five_years.ipynb`. Primera celda en markdown: "Solo descriptivo; ninguna etapa de decisión lee esta salida (Principio VI)". Guardar una copia del manifiesto al inicio; `load_interim(cfg, max_day=1941)`; matriz global de cinco años con `build_matrix("Y1-Y5", ..., 122, 1941, cfg["segmentation"])`; matrices anuales Y1–Y5 con `year_matrices` (Y1–Y4 de `cfg["split"]["years"]` e Y5 = `cfg["split"]["test_days"]`); acuerdo y PSI de cada año frente al anterior y frente a la global (solo como descripción); `describe_items`; distribuciones de gasto, CV, ceros y media frente a mediana. Guardar tablas y figuras en `reports/eda_five_years/`, incluida `matrix_five_years.parquet`, que se compara con `matrix_train` en el informe final (US1-3). Al final, comprobar que el manifiesto no cambió

**Checkpoint**: contexto descriptivo disponible; nada congelado.

---

## Fase 4: Historia 2 — ranking y matriz ABC × XYZ sobre train (Prioridad: P2) 🎯 MVP

**Objetivo**: matriz ABC × XYZ por artículo agregado, calculada solo con Y1–Y4 (d_122–d_1577) y congelada.

**Prueba independiente**: con los datos de train y la config por defecto se genera `matrix_train.parquet`; se verifica a mano en ≥ 10 artículos el gasto, el porcentaje acumulado, el CV y la clase (SC-003), y el 100 % de los artículos activos tiene clase o `exclusion_reason` (SC-001).

- [X] T032 [US2] Añadir a `src/tests/test_segmentation.py` el caso de `sensitivity_table`: con cortes desplazados ±5 pp los conteos de clases coinciden con un cálculo a mano sobre el fixture
- [X] T033 [US2] Implementar en `src/segmentation.py` la función `sensitivity_table(agg, start_d, end_d, seg: dict) -> pd.DataFrame`: conteo de clases ABC y XYZ con cada corte y umbral desplazado ±5 pp (R5, FR-013), solo sobre train
- [X] T034 [US2] Crear `notebooks/03_abc_xyz_train.ipynb`: `load_validated_config`; `load_interim(cfg, max_day=1577)` con `assert_train_only`; no leer `reports/eda_five_years/`; `build_matrix("train_global", ..., 122, 1577, cfg["segmentation"])` tomando los días de `cfg["split"]["years"]`; mostrar ranking, matriz de conteos ABC × XYZ, la tabla de sensibilidad y una verificación manual de 10 artículos (gasto recalculado desde `units × prices_daily` por tienda, % acumulado, CV y clase); comprobar SC-001; `freeze(matrix, "matrix_train", "data/processed/matrix_train.parquet", cfg)`

**Checkpoint**: `matrix_train` congelado y verificable por sí solo.

---

## Fase 5: Historia 3 — estabilidad estacional, lista objetivo y tratamiento (Prioridad: P3)

**Objetivo**: comparaciones Y2/Y1, Y3/prom(Y1,Y2) y Y4/prom(Y1–Y3), acuerdo, PSI, regla de estabilidad y lista objetivo congelada con tratamiento, sin tocar Y5.

**Prueba independiente**: con `matrix_train` congelado, el notebook 04 genera `targets.parquet` y `segmentation_params.json`; cada objetivo es A en train, estable según `stability_rule` y con tratamiento; las declaraciones son anteriores a la ejecución y no se carga ningún día ≥ 1578 (FR-021).

### Pruebas de la Historia 3

- [X] T035 [P] [US3] Escribir `src/tests/test_stability.py`: cada una de las 4 reglas de `STABILITY_RULES` con entradas construidas a mano (incluido `cell_majority` con 2 de 3 y `a_every_year` con un año en B); la referencia excluye los años no activos completos y una comparación sin referencia falla para ese artículo; `choose_segment_period` resuelve empates a favor del primer candidato; `build_targets` lanza error con una celda sin `treatment_map` y con la lista vacía

### Implementación de la Historia 3

- [X] T036 [US3] Implementar en `src/stability.py` la función `period_matrices(agg, cfg, kind: str, week_of_day) -> pd.DataFrame`: `quarter` = 4 bloques de 91 días por año; `four_month` = bloques de 119/119/126 días (17/17/18 semanas) por año, sobre Y1–Y4; `set_id` como `"Y1-Q1"` o `"Y1-T1"`
- [X] T037 [US3] Implementar en `src/stability.py` la función `choose_segment_period(year_mats, period_mats_by_kind: dict, candidates: list[str]) -> tuple[str, pd.DataFrame]`, con el criterio `max_mean_abc_agreement`: mayor acuerdo ABC medio entre cada matriz por periodo y la matriz de su año sobre Y1–Y4; empate → primer candidato de la lista; devuelve el periodo elegido y la tabla de acuerdos
- [X] T038 [US3] Implementar en `src/stability.py` la función `reference_classes(year_mats, agg, seg: dict, comparison: str, prior_years: list[str], min_active_days_year: int) -> pd.DataFrame`, con columnas `comparison, item_id, ref_spend_mean, ref_abc, ref_cv_mean, ref_xyz, n_ref_years`: promedia el gasto anual sobre los `prior_years` en que el artículo está activo (activo = sin NaN en sus `min_active_days_year` días) y vuelve a aplicar `classify_abc` sobre esos promedios para todos los artículos con referencia; promedia el CV anual y aplica `classify_xyz`. Sin años activos previos → `ref_abc` NaN, y la comparación falla (FR-016). En train: Y2 ← [Y1], Y3 ← [Y1, Y2], Y4 ← [Y1, Y2, Y3]; la réplica usa Y5 ← [Y1…Y4]. Justificación en el docstring: (b)
- [X] T039 [US3] Implementar en `src/stability.py` la función `stability_measures(year_mats, matrix_global, refs, comparisons: list[str], cfg) -> pd.DataFrame`, con columnas `comparison, dimension, agreement_pct_vs_ref, agreement_pct_vs_global, psi, psi_band` para `dimension ∈ {abc, xyz}`; la distribución de referencia del PSI es el promedio simple de las proporciones de los años previos; los umbrales vienen de `cfg["stability"]` (FR-017). La reutiliza la réplica con `comparisons = ["Y5"]`
- [X] T040 [US3] Implementar en `src/stability.py` el dict `STABILITY_RULES = {"abc_all", "abc_xyz_all", "cell_majority", "a_every_year"}` → funciones con la misma firma `(item_stab: pd.DataFrame) -> pd.Series[bool]` indexada por `item_id`, según FR-018: misma ABC en las 3 comparaciones; misma ABC y XYZ en las 3; misma celda en ≥ 2 de 3; clase A en cada año Y1–Y4. Una comparación sin referencia cuenta como no coincidente. Justificación en un comentario: (a), regla de negocio intercambiable con 4 implementaciones
- [X] T041 [US3] Implementar en `src/stability.py` la función `build_item_stability(year_mats, refs, cfg) -> pd.DataFrame`, con columnas de `contracts/artifacts.md`: `item_id, comparison, abc, ref_abc, xyz, ref_xyz, same_abc, same_xyz, same_cell, stable, rule`; `stable` sale de `STABILITY_RULES[cfg["stability"]["stability_rule"]]` y `a_every_year` queda disponible para el reporte
- [X] T042 [US3] Implementar en `src/stability.py` la función `build_targets(matrix_train, item_stab, cfg) -> pd.DataFrame`, con columnas `item_id, cell, xyz_class, spend_train, spend_share, treatment`: "Debe ser A y estable; celda sin tratamiento → se detiene; lista vacía → se detiene (FR-019)"; `cell` y `xyz_class` salen de `matrix_train`; `spend_share = spend / Σ spend de todos los artículos en train`; `treatment = cfg["treatment_map"][cell]`, y si falta la celda lanza `ValueError`
- [X] T043 [US3] Crear `notebooks/04_stability_target_list.ipynb`: `record_declarations` de `stability.segment_candidates`, `stability.segment_criterion`, `stability.stability_rule`, `stability.stability_rule_criterion`, `segmentation.cv_aggregation`, `segmentation.abc_cuts`, `segmentation.xyz_thresholds` y `treatment_map` antes de calcular, y `require_declarations` con la hora de inicio; `load_interim(cfg, max_day=1577)` con `assert_train_only`; `load_frozen("matrix_train")`; calcular matrices anuales Y1–Y4 y por periodo, periodo elegido, referencias Y2–Y4, medidas, estabilidad por artículo y lista objetivo; mostrar acuerdos, PSI, cambios de XYZ y conteo de objetivos por celda; congelar `matrices_by_set`, `stability`, `item_stability`, `targets` y `segmentation_params` (`data/processed/segmentation_params.json` con periodo elegido, `abc_cuts`, `xyz_thresholds`, `cv_aggregation`, `cv_min_blocks`, `stability_rule` y `treatment_map`, FR-020)

**Checkpoint**: lista objetivo y parámetros de segmentación congelados.

---

## Fase 6: Historia 4 — réplica sobre el año 5 de lo congelado en train (Prioridad: P4)

**Objetivo**: con la lista congelada y `[forecast]` ya declarado, replicar sobre Y5 la matriz y la estabilidad con los parámetros congelados, antes de pronosticar.

**Prueba independiente**: sin `targets` congelado o sin `[forecast]` declarado, el notebook 05 se detiene (US4-2); con ambos, produce `matrix_y5`, `stability_y5` y `year5_replication`, y el hash de `targets` no cambia.

- [X] T044 [P] [US4] Escribir `src/tests/test_replication.py`: `replicate_year5` usa los cortes, umbrales y `cv_aggregation` de `segmentation_params` (no los de la config); `same_as_frozen` es correcto para un objetivo que cambia de celda en Y5 y para uno que no; `targets` no se modifica; la referencia de Y5 promedia los años activos de Y1–Y4; el notebook-flujo `run_replication` se detiene si `targets` no está en el manifiesto o si falta la declaración de alguna clave de `[forecast]`
- [X] T045 [US4] Implementar en `src/stability.py` la función `replicate_year5(targets, seg_params: dict, matrix_train, agg, cfg, week_of_day) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: `matrix_y5 = build_matrix("Y5", agg, *cfg["split"]["test_days"], seg_params, week_of_day)`; matrices anuales Y1–Y4 con `year_matrices` y `seg_params`; `reference_classes(..., comparison="Y5", prior_years=["Y1","Y2","Y3","Y4"])`; `stability_y5 = stability_measures(..., matrix_global=matrix_train, comparisons=["Y5"])`; `year5_replication` solo para los objetivos, con columnas `item_id, frozen_cell, ref_abc, ref_xyz, y5_abc, y5_xyz, y5_cell, same_abc, same_xyz, same_cell, same_as_frozen` (`same_*` frente a la referencia; `same_as_frozen = y5_cell == frozen_cell`)
- [X] T046 [US4] Implementar en `src/stability.py` la función `run_replication(cfg) -> None`, que ejecuta la etapa en este orden: `load_frozen("targets")`, `load_frozen("segmentation_params")` y `load_frozen("matrix_train")` (falla si alguno no existe); `record_declarations` de todas las claves de `[forecast]` y `require_declarations`; recién entonces `load_interim(cfg, max_day=1941)`; `replicate_year5`; `freeze` de `matrix_y5`, `stability_y5` y `year5_replication` en `data/processed/`; `verify_manifest()` para comprobar que el hash de `targets` no cambió
- [X] T047 [US4] Crear `notebooks/05_year5_replication.ipynb`: celda de markdown con la regla "solo diagnóstico; no modifica la lista ni los candidatos; ninguna etapa de selección lo lee"; mostrar los valores de `[forecast]` que se van a declarar; `run_replication(cfg)`; mostrar acuerdo y PSI de Y5 frente a la referencia y a `matrix_train`, la matriz de conteos de Y5 y la tabla de objetivos que cambian de clase o celda

**Checkpoint**: réplica en Y5 congelada; los candidatos de pronóstico quedan fijados antes de pronosticar.

---

## Fase 7: Historia 5 — pronóstico de participación a 28 días evaluado en test (Prioridad: P5)

**Objetivo**: elegir un modelo por backtesting en train, congelarlo y evaluarlo una sola vez en 13 orígenes de Y5 con WRMSSE frente al seasonal naive.

**Prueba independiente**: con `targets` y la réplica congelados, los notebooks 06 y 07 producen `frozen_models.json` y `evaluation.parquet`; el WRMSSE del elegido < el del seasonal naive en los mismos objetivos y orígenes (SC-006); repetir con el mismo `model_id` se detiene (SC-007); cambiar `[forecast]` después de la réplica detiene el backtest (US5-2).

### Pruebas de la Historia 5

- [X] T048 [P] [US5] Escribir `src/tests/test_metrics.py`: RMSSE y WRMSSE de una serie de juguete contra un cálculo hecho a mano; la escala empieza en la primera venta no nula; un artículo con escala 0 se excluye y se registra; WAPE = Σ|e| / Σy; los pesos se normalizan sobre los artículos evaluados
- [X] T049 [P] [US5] Escribir `src/tests/test_forecasting.py`: `share_series` excluye y registra los días con `chain_total = 0` (`zero_chain_total`); el seasonal naive con m = 7 repite la semana anterior; la media móvil de ventana w es la media de las últimas w observaciones; un ajuste que lanza excepción se registra en fallos y el proceso sigue; `forecast_frozen` no cambia `params` (comparación exacta antes y después); los folds caen dentro de d_1466–d_1577; `run_backtest` se detiene si una clave de `[forecast]` difiere de su declaración o si la declaración es posterior a `matrix_y5`

### Implementación de la Historia 5

- [X] T050 [P] [US5] Implementar en `src/metrics.py` las funciones `naive_scale(train_units: np.ndarray) -> float` (media de las diferencias a un paso al cuadrado desde la primera venta no nula hasta el fin de train), `rmsse(y, y_hat, scale) -> float`, `wrmsse(y: dict, y_hat: dict, scales: dict, weights: dict) -> tuple[float, list[str]]` (excluye escala 0 y devuelve los excluidos; renormaliza los pesos), `wape(y, y_hat) -> float` y `rmse(y, y_hat) -> float`. Justificación en el docstring: (b). Sin MAPE
- [X] T051 [US5] Implementar en `src/forecasting.py` la función `share_series(item_units, chain_total, item_ids, item_launch_d, start_d, end_d) -> tuple[np.ndarray, pd.DataFrame]`: participación = unidades del artículo / total de la cadena desde el lanzamiento del artículo; días con total 0 → NaN y fila `zero_chain_total` en el registro
- [X] T052 [US5] Implementar en `src/forecasting.py` la función `candidate_grid(cfg) -> list[dict]` (`candidate_id, family, hyperparams`) a partir de `cfg["forecast"]`: `snaive` (m = `snaive_m`), `moving_average` (cada ventana de `ma_windows`), `arima` (producto de `arima_p × arima_d × arima_q`) y `sarima` (grilla ARIMA × (P, D, Q) ∈ `sarima_PDQ`³, s = `sarima_s`); `candidate_id` es legible, p. ej. `sarima_1_0_1_1_0_1_7`
- [X] T053 [US5] Implementar en `src/forecasting.py` el dict `FORECASTERS` (`snaive`, `moving_average`, `arima`, `sarima`) → funciones `fit(series, hyperparams) -> dict` y `forecast_frozen(history, frozen: dict, h: int) -> np.ndarray`. `snaive`: `SARIMAX(order=(0,0,0), seasonal_order=(0,1,0,m))`; `arima` y `sarima`: `SARIMAX(order, seasonal_order, trend="n")` sin exógenas; `forecast_frozen` reconstruye con `SARIMAX(history, ...).smooth(frozen["params"])` y pronostica `h` pasos sin reestimar; `moving_average` es de forma cerrada (media de las últimas w observaciones no NaN, sin parámetros estimados). Los NaN de la serie se pasan como faltantes al filtro de Kalman. Justificación en un comentario: (a)
- [X] T054 [US5] Corregir `specs/001-abc-xyz-share-forecast/research.md` R8: la media móvil no es un modelo `SARIMAX`; se calcula de forma cerrada y su "parámetro congelado" es la ventana
- [X] T055 [US5] Implementar en `src/forecasting.py` la función `run_backtest(shares, item_units, chain_total, targets, agg, cfg) -> tuple[pd.DataFrame, pd.DataFrame]`. Primero llama a `require_declarations(forecast_keys, stage_start, cfg)` y `require_declared_before(forecast_keys, "matrix_y5")`. `backtest_folds` folds de 28 días sin solapamiento que terminan en d_1577 (por defecto d_1466–d_1577); cada candidato se ajusta una vez con los datos previos al primer fold y los folds siguientes solo añaden observaciones con `forecast_frozen`; `units_hat = share_hat × chain_total real`; puntaje = RMSSE por artículo sobre unidades (escala con `naive_scale` hasta el día previo al primer fold), promediado por `selection_unit` (`selection_level = "item"` → `item_id`; `"global"` → una unidad para todos; otra etiqueta → `ValueError`). Respeta `subset_size` (los N primeros por `spend_train`). Devuelve `backtest_scores` (`selection_unit, candidate_id, fold, rmsse_mean`) y `fit_failures` (`item_id, candidate_id, stage, message`); cualquier excepción o `ConvergenceWarning` se registra y omite ese candidato para ese artículo
- [X] T056 [US5] Implementar en `src/forecasting.py` la función `select_and_freeze(scores, shares, targets, cfg) -> list[dict]`: por `selection_unit`, el candidato con menor RMSSE medio entre folds (empate → primero en `candidate_grid`); reajuste por artículo con todo train hasta d_1577; si el reajuste falla, se registra y se usa `snaive`; devuelve entradas `{selection_unit, item_ids[], candidate_id, family, order, seasonal_order, params[]}` (una por artículo, con `params` como lista de floats) para `models/frozen_models.json`
- [X] T057 [US5] Implementar en `src/forecasting.py` la función `evaluate_test(model_id: str, models: list[dict], cfg) -> dict`: `TestLedger.has(model_id)` → `RuntimeError`; todo `frozen_at` del manifiesto debe ser anterior al inicio de la ejecución; carga `targets` y `frozen_models` con `load_frozen`; para cada origen k = 1…13 el modelo pronostica d_(1578 + 28(k−1)) … +27 con `forecast_frozen` sobre los datos observados hasta el día previo al origen; `units_hat = share_hat × chain_total real` (sin redondeo); WRMSSE por origen con pesos = participación en gasto en `weight_window` (d_1550–d_1577) y escala de train; escribe `forecasts_test.parquet` (`model_id, item_id, origin, h, d, share_hat, units_hat, units_actual`) y devuelve `model_id, wrmsse (media de 13), wrmsse_o1 … wrmsse_o13, wape, rmse`; registra en `TestLedger`
- [X] T058 [US5] Crear `notebooks/06_backtesting.ipynb`: no registra declaraciones nuevas; `load_interim(cfg, max_day=1577)` con `assert_train_only`; `load_frozen("targets")`; no leer `matrix_y5`, `stability_y5`, `year5_replication` ni `reports/eda_five_years/`; series de participación, `run_backtest` y `select_and_freeze`; mostrar el ranking de candidatos y los fallos; congelar `frozen_models` (`models/frozen_models.json`) y `backtest_scores`; guardar `models/fit_failures.parquet`
- [X] T059 [US5] Crear `notebooks/07_test_evaluation.ipynb`: `load_interim(cfg, max_day=1941)`; `evaluate_test` del seasonal naive (`model_id = "snaive_<version>"`, con los mismos objetivos y orígenes) y del modelo congelado (`model_id = "<family>_<selection_level>_<version>"`); comprobar que los `params` usados son idénticos a `frozen_models.json`; `beats_snaive = wrmsse < wrmsse_snaive`; escribir `data/processed/evaluation.parquet`; si no gana, reportar que el RMV no se cierra, sin cambiar parámetros (Principio VI)

**Checkpoint**: RMV completo.

---

## Fase 8: Pulido y aspectos transversales

- [X] T060 [P] Escribir `src/tests/test_reproducibility.py`: ejecutar dos veces las etapas 1, 3 y 4 sobre los fixtures sintéticos en directorios temporales (`tmp_path`) y comprobar que el SHA-256 de `matrix_train` y `targets` coincide (SC-004)
- [X] T061 Crear `notebooks/08_final_report.ipynb` y exportarlo con `uv run jupyter nbconvert --to html --output-dir reports notebooks/08_final_report.ipynb` a `reports/final_report.html`. Debe incluir: hallazgos del EDA, es decir, artículos cuya clase en `reports/eda_five_years/matrix_five_years.parquet` difiere de `matrix_train`, presentados como hallazgo y sin efecto sobre la matriz (US1-3); por objetivo, clase, celda, `spend_share`, tratamiento y pronóstico en unidades enteras redondeadas hacia arriba en la mitad (half-up) (SC-008); réplica en Y5 del 100 % de los objetivos (SC-009); WRMSSE frente al seasonal naive, WAPE y RMSE como diagnóstico; advertencia de backtesting y sesgo de selección (FR-031); aclaración de que no es una orden de compra (Principio I); versiones de Python y librerías tomadas de `uv.lock` (R12) — **Nota (2026-10-26)**: exportación probada sobre datos sintéticos; `reports/final_report.html` real se genera al correr con M5.
- [X] T062 [P] Ejecutar `uv run ruff check src` y `uv run ruff format --check src`, y corregir los hallazgos
- [X] T063 Revisar que todas las funciones de `src/` tienen docstrings en español con descripción, parámetros con tipo y retorno, y que no hay valores de cortes, umbrales o días fijos en el código (Principio III)
- [X] T064 Ejecutar la validación completa de `specs/001-abc-xyz-share-forecast/quickstart.md`: `uv run pytest src/tests -q`, notebooks 01–08 en orden y las cinco verificaciones negativas — **Nota (2026-10-26)**: validado de punta a punta sobre un proyecto sintético (`PROJECT_ROOT`); falta repetirlo con los CSV reales de M5 en `data/raw/`.

---

## Dependencias y orden de ejecución

### Orden de ejecución de los notebooks (obligatorio)

01 validación → 02 EDA de cinco años → 03 matriz de train → 04 estabilidad y lista objetivo →
05 réplica en Y5 → 06 backtesting → 07 evaluación en test → 08 informe. Ningún notebook se
ejecuta antes que el anterior.

### Dependencias entre fases

- **Setup (Fase 1)**: sin dependencias.
- **Fundacional (Fase 2)**: depende del Setup y bloquea todas las historias.
- **US1 (Fase 3)**: depende de la Fase 2.
- **US2 (Fase 4)**: el código depende de la Fase 2; el notebook 03 se ejecuta después del 02 (T031).
- **US3 (Fase 5)**: necesita `matrix_train` congelado (T034).
- **US4 (Fase 6)**: necesita `targets` y `segmentation_params` congelados (T043).
- **US5 (Fase 7)**: el código puede escribirse en paralelo con US3/US4, pero el notebook 06 solo se ejecuta después de la réplica (T047).
- **Pulido (Fase 8)**: T061 necesita US1–US5.

### Dependencias dentro de las historias

- Las pruebas se escriben primero y fallan antes de implementar.
- Fundacional: `src/config.py` (T011–T012) → `src/freeze.py` (T013–T014) → `src/data.py` (T015–T021) → `src/segmentation.py` (T022–T026) → `src/stability.py` (T027) → notebook 01 (T028).
- US3: T036 → T037 → T038 → T039 → T040 → T041 → T042 → T043 (mismo archivo, en secuencia).
- US4: T045 → T046 → T047 (reutiliza T038 y T039).
- US5: T050 en paralelo con T051–T053; T053 → T055 → T056 → T057 → T058 → T059.

### Oportunidades de paralelismo

- Setup: T003, T004 y T005.
- Fundacional: T006–T010 (pruebas en archivos distintos).
- US1: T029 en paralelo con las pruebas de otras historias.
- US5: T048, T049 y T050 en paralelo; el código de `src/forecasting.py` puede escribirse mientras se ejecutan los notebooks 02–05.
- Pulido: T060 y T062.

---

## Ejemplo de paralelismo: Historia 5

```bash
# Pruebas y métricas en paralelo (archivos distintos):
Tarea: "T048 Escribir src/tests/test_metrics.py"
Tarea: "T049 Escribir src/tests/test_forecasting.py"
Tarea: "T050 Implementar src/metrics.py"
# Luego, en secuencia sobre src/forecasting.py: T051 → T052 → T053 → T055 → T056 → T057
```

---

## Estrategia de implementación

### MVP primero

1. Fase 1 (Setup) y Fase 2 (Fundacional).
2. Fase 3 (US1): EDA de cinco años como contexto.
3. Fase 4 (US2): `matrix_train` congelado.
4. **Parar y validar**: verificación manual de 10 artículos (SC-003) y cobertura del 100 % (SC-001).

### Entrega incremental

1. Setup + Fundacional → datos validados.
2. US1 → contexto descriptivo de cinco años.
3. US2 → matriz ABC × XYZ de train (MVP).
4. US3 → lista objetivo congelada con tratamiento.
5. US4 → `[forecast]` declarado y réplica en Y5.
6. US5 → pronóstico evaluado en test; cierre del RMV si supera al seasonal naive.
7. Pulido → informe final y validación con quickstart.

---

## Notas

- [P] = archivos distintos y sin dependencias pendientes.
- Un solo bloque de trabajo activo a la vez (constitución, Flujo de trabajo); cerrar cada fase en su checkpoint.
- Las etapas de decisión (notebooks 03, 04 y 06) no cargan días ≥ 1578 (`load_interim(cfg, max_day=1577)` con `assert_train_only`) y no leen `reports/eda_five_years/` ni los artefactos de la réplica.
- Cambiar un parámetro después de la réplica en Y5 (notebook 05) exige una nueva `version` en `params.toml`, su fila en el historial y repetir desde el notebook 03.
- Hacer commit al cerrar cada tarea o grupo lógico.

## Phase 9: Convergence

- [ ] T065 CRITICAL: Exigir en `run_train_matrix` (`src/segmentation.py`) que exista `reports/eda_five_years/matrix_five_years.parquet` con fecha de modificación anterior al inicio de la etapa 3, y detenerse si falta, sin leer su contenido; agregar el caso a `src/tests/test_eda.py` per Constitution, Flujo de trabajo (orden obligatorio) y FR-029 (partial)
- [ ] T066 Registrar con `record_declarations` las claves `segmentation.cv_aggregation`, `segmentation.abc_cuts`, `segmentation.xyz_thresholds` y `segmentation.cv_min_blocks` al inicio de `run_train_matrix`, antes de calcular; mantener `require_declarations` en la etapa 4; agregar la prueba en `src/tests/test_segmentation.py` per FR-012 y FR-015 (partial)
- [ ] T067 Soportar `cv_aggregation = "segment_period"` en la matriz de train usando los segmentos del periodo congelado, o rechazarlo en `validate_config` (`src/config.py`) con su regla de rango documentada en `contracts/params.md` per FR-012 (partial)
- [ ] T068 Implementar en `_selection_unit` (`src/forecasting.py`) un nivel de selección por grupo (por ejemplo, por celda ABC × XYZ de `targets`), con su prueba en `src/tests/test_forecasting.py` per FR-024 (partial)
- [ ] T069 Corregir en `final_report_tables` (`src/forecasting.py`) la comparación de clases train frente a cinco años para que NaN en ambos lados no cuente como diferencia; agregar la prueba en `src/tests/test_eda.py` per US1/AC3 (partial)
- [ ] T070 Alinear el tipo de `units` en `load_sales` (`src/data.py`) con T016: dejar float32 y justificarlo por memoria en el docstring, o cambiar a float64 per T016 (contradicts)
