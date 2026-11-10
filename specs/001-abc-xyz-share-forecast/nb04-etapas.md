# nb 04 · Registro de etapas

Notebook: `notebooks/04_feature_engineering_y_seleccion_de_modelo.ipynb` (nuevo; el anterior quedó como
`04_feature_engineering_y_seleccion_de_modelo_superseded.ipynb`, registro en `nb04-superseded-etapas.md`).
Constitución 7.0.0: carga los datasets del nb 03b y los hiperparámetros del nb 03c y reentrena el CatBoost global con
todo el train (años 1–4, incluida la ventana de backtesting), sin ninguna búsqueda. La evaluación en test es del nb 05.
Reglas de trabajo: ver `CLAUDE.md` (R1–R8).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas

- [x] **0 · Decisiones y reglas** (2026-11-09). Aprobada el 2026-11-09 (reproducción exacta con la misma semilla e hilos). Notebook creado (título, Etapa 0, configuración y candado). D1 cargar
  datasets del 03b y la prueba 43 del 03c; D2 reproducir el backtest del 03c (pronósticos ≤ 1e-6, WRMSSE 0,5062,
  CatBoost 1.2.10) o detenerse; D3 ajuste con todo el train con objetivo (ratio tratado fuera); D4 hiperparámetros sin
  cambios, semilla 42, 6 hilos; D5 `models/nb04_catboost/catboost_global.cbm` + `meta.json` con candado; D6 sin
  búsqueda ni test, importancia de variables como diagnóstico; D7 sin piloto (~50 s + ~4 s). Sin cambios en `src/`.
- [x] **1 · Carga y reproducción del backtest del 03c** (2026-11-09). Aprobada el 2026-11-09. Datasets del 03b (517 artículos; train 24.443
  filas, 24.348 con objetivo; test 6.721), prueba 43 del 03c y pesos del WRMSSE (últimos 13 bloques de train).
  Probada en seco: el backtest de 13 folds repetido da diferencia máxima 0 unidades frente al 03c y WRMSSE 0,5062;
  misma versión de CatBoost (1.2.10); ~22 s. Sin cambios en `src/`.
- [x] **2 · Pronóstico a un paso y encadenado, en tasas y en unidades** (2026-11-09). Aprobada el 2026-11-09 (usuario: reemplaza a las
  1.1, 1.2 y a la Etapa 2 anterior, en una sola sección). Sobre los 13 bloques de la ventana: a un paso (folds de la
  Etapa 1, unidades desde el real anterior) y encadenado desde el inicio de la ventana con el modelo del fold 1 (tasas
  con rezagos y ventanas recalculados con lo pronosticado; unidades = pronóstico anterior × tasa). WRMSSE, WAPE y MAE
  en los dos niveles (escala del RMSSE en tasas = misma definición sobre la tasa de todo el train). Probada en seco:
  WRMSSE en unidades 0,5062 a un paso vs 0,7418 encadenado (WAPE 10,2 % vs 20,2 %); en tasas 0,6711 vs 0,6959 (WAPE
  10,4 % vs 10,7 %). Por paso, en unidades el encadenado sube de 0,37 (paso 1) a ~0,70; en tasas casi no cambia.
  Gráfico por artículo (tasas arriba, unidades abajo, error por bloque); 5 por segmento en pantalla y los 517 en
  `reports/figures/nb04_catboost_backtest/<segmento>/` (~82 MB, ~6 min). `src/modeling.py`: `recursive_ratio_forecast`,
  `chain_units`, `plot_forecast_item` (nuevas) y sus tests.
- [x] **3 · Reentrenamiento e importancia SHAP** (2026-11-09). Aprobada el 2026-11-09 (gráficos solo en el notebook). CatBoost global con todo el train (24.348 filas con
  objetivo, bloques 2–52, ratio tratado fuera, sin test), prueba 43 sin cambios, semilla 42, 6 hilos. Importancia con
  SHAP nativo de CatBoost sobre las filas de ajuste (usuario): barras de media |SHAP| y mapa de calor por segmento (% de
  la importancia de cada segmento). Probada en seco: ajuste 2,6 s, SHAP 0,8 s, aditividad 9e-16; lideran
  `ratio_lag13`, `ratio_roll13_mean`, `period`, `ratio_lag1`, `ratio_roll6_mean` e `item_id` (~70 % en los tres
  segmentos); `cell` pesa más en AZ (3,9 %) que en AX (1,3 %). `src/modeling.py`: `catboost_shap`,
  `shap_importance`, `plot_shap_importance` (nuevas) y sus tests.
- [x] **4 · Guardado del modelo y cierre** (2026-11-09). Aprobada el 2026-11-10. `models/nb04_catboost/catboost_global.cbm` (~4,6 MB) y
  `meta.json` (variables, categóricas, hiperparámetros, semilla, hilos, filas de ajuste, origen 03b/03c, versiones).
  Candado: modelo igual (igualdad de CatBoost, no bytes) y meta igual → no escribe; distinto → se detiene salvo
  `OVERWRITE`. Chequeo: el modelo leído es igual y predice lo mismo. Probada en seco (fuera de `models/`): reentrenar
  da el mismo modelo y el candado no reescribe. Celda de cierre con el resumen para el nb 05. `src/modeling.py`:
  `save_catboost_model`, `load_catboost_model` (nuevas) y su test.

**Decisión del usuario (2026-11-09) · uso en producción:** cada mes se pronostican los próximos 13 bloques desde el
último bloque real, **encadenados** (cada bloque usa el pronóstico del anterior), y el pronóstico del mes anterior se
reemplaza por el nuevo. No se cambia nada de lo hecho (tratamiento, búsqueda, selección, modelo); el encadenado se
aplica al pronosticar con este mismo modelo.
