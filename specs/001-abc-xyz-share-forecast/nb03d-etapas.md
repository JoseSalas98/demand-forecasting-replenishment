# nb 03d · Registro de etapas (experimental)

Notebook: `notebooks/03d_seleccion_por_articulo.ipynb`.
Selección del modelo por artículo sobre las series con tratamiento por tienda: junta el seasonal naive y SARIMAX
(nb 03b) con CatBoost (nb 03c), aplica el filtro de la constitución (Principio VII) y mide el conjunto elegido. No
entrena ni busca nada: solo lee pronósticos ya guardados de la ventana de backtesting. Persiste solo la tabla de
selección en `data/interim/nb03d_selection/`; el nb 04 no carga nada de aquí.
Reglas de trabajo: ver `CLAUDE.md` (R1–R8).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas

- [x] **0 · Decisiones y reglas** (2026-11-07). Aprobada el 2026-11-08 con las opciones propuestas: empate → el más
  simple; corte 7/6; CSV en `data/interim/nb03d_selection/`. D1 independiente; lee lo guardado por el 03b y el 03c; persiste
  solo `selection.csv`. D2 candidatos: seasonal naive, SARIMAX y CatBoost (usuario: solo el mejor boosting del
  03c, para no inflar el sesgo de selección con tres modelos casi empatados). D3 error por artículo = RMSSE en sus 13
  bloques de la ventana, con la escala del 03b; filtro y regla de la constitución; empate exacto → el más simple.
  D4 aceptación: WRMSSE del conjunto elegido < seasonal naive (pesos del 03c); WAPE secundaria, MAE diagnóstico.
  D5 diagnóstico de optimismo (usuario): elegir con los folds 1–7 y medir en los 8–13. D6 cómputo liviano.
  Configuración escrita. Sin cambios en `src/`.
- [x] **1 · Carga** (2026-11-08). Aprobada el 2026-11-08. Entradas del 03b, lotes de SARIMAX, backtest de CatBoost y pesos (regla del 03c);
  tabla `window` artículo × fold con `snaive`, `sarimax` y `catboost` en unidades. Probada en seco con los datos
  reales: 517 artículos × 13 folds; SARIMAX en 511 (6 `sin_modelo`, 1,4 % del peso), 231 respaldos en 170 artículos
  (máx. 4 de 13), 1 pronóstico negativo; CatBoost reproduce su WRMSSE del 03c (0,5062) y el seasonal naive da 0,8940.
  `src/metrics.py`: `train_weight_windows` (nueva); `src/modeling.py`: `window_forecasts` (nueva); y sus tests.
- [x] **2 · Error por candidato** (2026-11-08). Aprobada el 2026-11-08 (sin recorte de negativos; MAE solo por artículo). RMSSE (escala del 03b), WAPE y MAE por artículo en sus 13 bloques;
  WRMSSE (pesos renormalizados por segmento) y WAPE agrupado (Σ|e| / Σy) por candidato y segmento, comparados sobre los
  511 artículos con los tres candidatos. Probada en seco: WRMSSE total snaive 0,8925 · SARIMAX 0,6640 · CatBoost 0,5046;
  WAPE 24,8 % · 14,1 % · 10,2 %; CatBoost es el mejor en AX, AY y AZ. Cola de SARIMAX: RMSSE máx. 5,24
  (HOUSEHOLD_1_162, sesgo de +300 unidades en los folds 1–11); el filtro de la Etapa 3 lo descarta. Sin recorte de
  negativos. `src/metrics.py`: `item_errors`, `segment_errors` (nuevas) y sus tests.
- [x] **3 · Selección por artículo y aceptación agregada** (2026-11-08). Aprobada el 2026-11-08. Filtro estricto contra el snaive, menor
  RMSSE en los 13 folds, empate → el más simple. Probada en seco: CatBoost 394 artículos (74,2 % del peso), SARIMAX 83
  (17,4 %), snaive 40 (8,4 %). Aceptación: WRMSSE del conjunto 0,4808 < snaive 0,8940 (−46,2 %); WAPE 9,9 % vs.
  24,7 %. Frente a CatBoost solo (0,5062) gana 0,025, ganancia medida con los mismos folds con que se eligió (la
  Etapa 4 mide cuánto es optimismo). `src/modeling.py`: `select_by_item`, `selected_forecast` (nuevas) y sus tests.
- [x] **4 · Diagnóstico de optimismo** (2026-11-08). Aprobada el 2026-11-08 (opción b para la Etapa 5). Elegir con folds 1–7, medir en 8–13; bootstrap pareado por
  artículos (2.000 réplicas). Probada en seco (WRMSSE en 8–13): honesto 0,5165 · optimista 0,4363 (optimismo 0,080) ·
  CatBoost para todos 0,4831 · snaive 0,8265. **La selección honesta pierde contra CatBoost para todos:** +0,033,
  IC 95 % [0,009; 0,056], en AX, AY y AZ; la pérdida viene de los 64 artículos que con 1–7 eligen snaive (0,025) y los
  106 que eligen SARIMAX (0,008). Le gana al snaive (−0,310, IC [−0,355; −0,268]). Mismo ganador que la selección final
  en 78,1 % de los artículos. `src/metrics.py`: `paired_bootstrap_wrmsse` (nueva) y su test.
- [~] **5 · Guardado de la tabla de selección y cierre** (2026-11-08). `data/interim/nb03d_selection/selection.csv`,
  517 filas: segmento, peso, estado SARIMAX, RMSSE de los tres candidatos, `winner` / `winner_rmsse` / `winner_wape`
  (selección de la constitución), `winner_folds_1_7` (Etapa 4) y `winner_catboost_all` (opción b del usuario: CatBoost
  para todos, evidencia para la enmienda pospuesta del Principio VII). Se relee y compara; probada en seco con salida
  al scratchpad (el CSV lo escribe el usuario al ejecutar). Celda de cierre con conclusiones. Sin cambios en `src/`.

**Decisión del usuario (2026-11-09):** se adopta **CatBoost para todos** (sin selección por artículo), con la evidencia de
la Etapa 4. Requiere enmendar el Principio VII (filtro por artículo) antes de aplicarlo en el flujo canónico.
