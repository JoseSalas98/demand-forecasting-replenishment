# nb 03c · Registro de etapas (experimental)

Notebook: `notebooks/03c_tratamiento_por_tienda.ipynb`.
Búsqueda de hiperparámetros del boosting global (XGBoost, LightGBM, CatBoost) sobre las series con tratamiento por
tienda del nb 03b. Carga las entradas que exporta el 03b (13.3, `data/interim/nb03b_datasets/`) y solo persiste su
búsqueda en `data/interim/nb03c_boosting/`; el nb 04 no carga nada de aquí. El nb 03d (pendiente) juntará SARIMAX,
boosting y seasonal naive y hará solo la selección por artículo (usuario, 2026-11-06).
Reglas de trabajo: ver `CLAUDE.md` (R1–R8).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas

- [x] **0 · Decisiones y reglas.** Aprobada el 2026-11-06. D1 entradas del 03b, persiste solo la búsqueda; D2 boosting global,
  hiperparámetros uno por algoritmo; D3 error promedio en los 13 folds, sin early stopping (`n_estimators` en la
  búsqueda); D4 Optuna con SQLite (reanuda), semilla `features.seed`; D5 una etapa por algoritmo con piloto (R7);
  D6 hasta 6 hilos mientras corre el SARIMAX del 03b. Límite: hiperparámetros elegidos con los mismos folds con los que
  compite (optimista frente a SARIMAX). Pendiente para la Etapa 2: métrica del promedio, espacio, pruebas, filas tratadas.
- [x] **1 · Carga de las entradas del nb 03b.** Aprobada el 2026-11-06 (escrita; falta ejecutarla). `load_tables` (`src/data.py`, nueva en el 03b 13.3).
  Probada en seco con los datasets del nb 04 (mismas columnas); falta que el usuario la ejecute tras correr el 03b.
  Decisiones de la búsqueda (usuario, 2026-11-06): WRMSSE; modelos simples, regularizados y amigables con la RAM;
  filas con ratio tratado no entrenan pero se pronostican y miden.
- [x] **2 · Métrica: WRMSSE.** Aprobada el 2026-11-06 (escrita; falta ejecutarla). RMSSE por artículo con sus 13 bloques de la ventana (escala del 03b) y
  pesos = gasto en sus últimos 13 bloques de train (usuario): con los 28 días de M5 (`weight_window`) 18 artículos con
  quiebre pesaban 0; 576 artículos usan Y4 y 3 (test antes del día 1578) su propio último año. Referencia: WRMSSE del
  seasonal naive con los mismos pesos. `src/metrics.py`: `spend_weights`, `wrmsse_frame` (nuevas) y sus tests.
- [x] **3 · XGBoost** (nb 04, 13a). Aprobada el 2026-11-06 como escrita; falta ejecutarla (piloto 3.1 antes de la 3.2, R7). `xgboost-cpu` (sin GPU) y `optuna` agregados con uv. Fijos: hist,
  `max_bin` 64, error cuadrático, category. Espacio: profundidad 2–4, 100–600 árboles, tasa 0,01–0,2, 5–100 filas por
  hoja, 60–90 % filas, 50–90 % variables, L2 1–100, L1 0,01–10. Piloto 5 pruebas en memoria (3.1); búsqueda 100
  pruebas TPE en SQLite con reanudación (3.2; medido 4–20 s por prueba con 6 hilos → ~20–30 min, ~270 MB); backtesting
  de la mejor y guardado en `xgboost_backtest.pkl` (3.3). Probada en seco con los datos del nb 04 (pocas pruebas;
  reanudación y repetición determinista verificadas). `src/modeling.py`: `make_booster`, `boosting_backtest`,
  `suggest_params`, `boosting_search` (nuevas) y sus tests.
- [x] **4 · LightGBM** (nb 04, 13b). Aprobada el 2026-11-06 como escrita; falta ejecutarla (piloto 4.1 antes de la 4.2). `lightgbm` agregado con uv. Fijos: error cuadrático, `max_depth` 4
  de tope, `max_bin` 63, `subsample_freq` 1, `deterministic` + `force_row_wise`; category con su regularización por
  defecto. Espacio: 3–15 hojas (crece hoja por hoja), el resto como XGBoost (`min_child_samples` 5–100 log). Piloto 5,
  búsqueda 100 en `lightgbm.db`, mejor en `lightgbm_backtest.pkl`. Medido 5–20 s por prueba con 6 hilos (~20–30 min,
  ~260 MB); repetición determinista verificada. Probada en seco. `src/modeling.py`: `make_booster` (rama lightgbm),
  `suggest_params` (tipo `int_log`) modificadas; tests parametrizados por algoritmo y de `suggest_params`.
- [~] **5 · CatBoost** (nb 04, 13c; 2026-11-06). `catboost` agregado con uv. Fijos: `RMSE`, `border_count` 64,
  `Bernoulli`, `boosting_type` Plain (medido: Plain 6–42 s y ~500 MB por prueba; Ordered 5–62 s y ~600 MB; las
  categóricas se codifican ordenadas en los dos), `cat_features` = item_id, cell, period; sin archivos `catboost_info/`.
  Espacio: profundidad 2–4 (árboles simétricos, 4–16 hojas), 100–600 árboles, tasa 0,01–0,2, L2 1–100,
  `random_strength` 0,1–10, 60–90 % filas, 50–90 % variables (`rsm`). Piloto 5, búsqueda 100 (~35–45 min) en
  `catboost.db`, mejor en `catboost_backtest.pkl`; repetición determinista verificada. Cierre del notebook escrito.
  Probada en seco. `src/modeling.py`: `make_booster` (rama catboost) modificada; test parametrizado con CatBoost.
- [x] **Ajuste por la 5.3 del 03b (2026-11-06).** Aprobado el 2026-11-06. El 03b ahora exporta 517 artículos (antes 579). La Etapa 1 compara
  con `n_items` del manifiesto en vez de 579; los conteos citados arriba (576 + 3 en los pesos) son de los 579 y se
  recalculan al correr. Antes de cada búsqueda (3.2, 4.2, 5.2) se imprimen los artículos, las filas, la fecha de la
  exportación y el estudio, para validar el conjunto cargado; el número de artículos no va en la configuración del
  estudio (no es un hiperparámetro, usuario). Sin cambios en `src/`.

- [~] **Candado y guardado protegido** (2026-11-09, usuario). Celda tras la configuración: `guard_rerun` detiene el
  notebook si sus salidas existen, salvo `RERUN = True`; el guardado no reemplaza contenido distinto salvo
  `OVERWRITE = True` (si es igual, no escribe). No cambia ningún resultado. `src/data.py`: `guard_rerun`, `save_pickle`,
  `_same_content` (nuevas), `save_tables` (`overwrite`) y sus tests en `src/tests/test_data.py`.
