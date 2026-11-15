# nb 05 · Registro de etapas

Notebook: `notebooks/05_evaluacion_backtesting.ipynb`.
Constitución 7.0.0: evaluación única en test del CatBoost global del nb 04 frente al seasonal naive, a un paso (13
orígenes con reentrenamiento, régimen del backtesting) y encadenado 13 bloques desde el último bloque de train (uso en
producción, usuario 2026-11-10). Nada se busca, elige ni ajusta con el test.
Reglas de trabajo: ver `CLAUDE.md` (R1–R8).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas

- [x] **0 · Decisiones y reglas** (2026-11-10). Aprobada el 2026-11-10 (usuario: el encadenado, solo como medida de producción, sin umbral). Notebook reescrito (título, Etapa 0, configuración y candado). D1 cargar
  03b, 03c y el modelo del nb 04 (`meta.json` igual al 03c); D2 bloques de test numerados 1–13 por artículo, train =
  fold 0, mismas funciones del backtesting; D3 a un paso: 13 orígenes con reentrenamiento; D4 encadenado: un origen
  (último bloque de train), modelo del nb 04, y comprobación de que el modelo del origen 1 es igual al del nb 04; D5
  seasonal naive t−13, mismo para ambas formas; D6 WRMSSE/WAPE/MAE en unidades y tasas, por segmento y paso, pesos y
  escalas solo de train; D7 aceptación: a un paso < seasonal naive en unidades, encadenado sin umbral; D8 una sola
  evaluación; D9 `data/interim/nb05_test/` (`forecasts`, `metrics`, manifiesto) con candado; D10 sin piloto (~45 s).
  Sin cambios en `src/`.
  **Cambio del usuario (2026-11-10), D11:** resumen de errores global y por artículo, sin comparar cada artículo con
  el seasonal naive. Marca para revisar si el RMSSE en unidades es atípico en su segmento o si el cociente test /
  ventana de backtesting lo es (Tukey, Q3 + 1,5·IQR), por separado a un paso y encadenado; los marcados se grafican
  en pantalla (hasta 5 por segmento). D9 suma `item_metrics.parquet`.
- [x] **1 · Carga y comprobaciones** (2026-11-10). Aprobada el 2026-11-10. Carga 03b, 03c y el modelo del nb 04; exige que variables,
  hiperparámetros, semilla, versión de CatBoost y origen coincidan. `test_data`: train = fold 0, test = folds 1–13 por
  posición (517 por fold). El modelo del origen 1 (24.348 filas, misma semilla e hilos) es igual al del nb 04. Pesos
  del WRMSSE como en el nb 04. No se pronostica el test. Probada en seco (~4 s). `src/modeling.py`:
  `assign_test_folds` (nueva) y su test.
- [x] **2 · Pronósticos del test** (2026-11-10). Aprobada el 2026-11-10. `forecasts`: 6.721 filas (517 × 13) con lo real y, en tasas y
  unidades, a un paso (`boosting_backtest` sobre `test_data`, 13 reentrenamientos, ~41 s), encadenado
  (`recursive_ratio_forecast` con el modelo del nb 04, ~9 s; `chain_units`) y seasonal naive (m = 13, una columna
  para ambas formas). Chequeos: sin nulos ni duplicados; paso 1 encadenado = a un paso; unidades = base × tasa;
  seasonal naive = real del bloque − 13. 10 filas con ratio tratado, se miden igual. Probada en seco con el código
  del notebook; sin métricas. Gráficos por artículo como en el nb 04 (usuario): 5 al azar por segmento en pantalla y
  los 517 en `reports/figures/nb05_test/<segmento>/` (código escrito, sin correr). `src/modeling.py`:
  `plot_forecast_item` (modificada: `splits` y `window_label` para dibujar el test) y su test.
- [x] **3 · Métricas globales y por artículo, aceptación y artículos para revisar** (2026-11-10). Aprobada el 2026-11-10 (Tukey sobre el log del cociente). `summary`
  (WRMSSE/WAPE por nivel, forma y segmento), `by_step`, `mae`, `accepted` (a un paso < seasonal naive en unidades).
  `item_metrics`: artículo × forma (a un paso, encadenado; sin seasonal naive) con RMSSE/WAPE/MAE en unidades y tasas,
  RMSSE de su ventana (a un paso del 03c, encadenado repetido como en el nb 04; chequeo 0,5062) y marcas Tukey por
  forma y segmento: `outlier_segment` (RMSSE) y `outlier_window` (log del cociente test / ventana). Marcados
  graficados en pantalla, hasta 5 por segmento. Código escrito, sin correr (usuario). `src/metrics.py`:
  `tukey_outliers` (nueva) y su test.
- [x] **4 · Guardado y cierre** (2026-11-10). Aprobada el 2026-11-10 (corrida: a un paso 0,495, encadenado 0,748, seasonal naive 0,868; cumple). `save_tables` en `data/interim/nb05_test/`: `forecasts`, `metrics`
  (nivel × forma × segmento; `step` vacío = 13 bloques, 1–13 = por paso), `mae`, `item_metrics` y manifiesto
  (aceptación, WRMSSE, origen 03b/03c/nb 04, semilla, k de Tukey, versiones), con candado. Chequeo: lo leído = lo
  guardado. Cierre con resumen desde lo guardado. Código escrito, sin correr. Sin cambios en `src/`.
