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

- [~] **0 · Decisiones y reglas** (2026-11-07). D1 independiente; lee lo guardado por el 03b y el 03c; persiste
  solo `selection.csv`. D2 candidatos: seasonal naive, SARIMAX y CatBoost (usuario: solo el mejor boosting del
  03c, para no inflar el sesgo de selección con tres modelos casi empatados). D3 error por artículo = RMSSE en sus 13
  bloques de la ventana, con la escala del 03b; filtro y regla de la constitución; empate exacto → el más simple.
  D4 aceptación: WRMSSE del conjunto elegido < seasonal naive (pesos del 03c); WAPE secundaria, MAE diagnóstico.
  D5 diagnóstico de optimismo (usuario): elegir con los folds 1–7 y medir en los 8–13. D6 cómputo liviano.
  Configuración escrita. Sin cambios en `src/`.
- [ ] **1 · Carga** de las entradas del 03b, los lotes de SARIMAX, el backtest de CatBoost y los pesos; tabla larga
  artículo × bloque con los tres pronósticos.
- [ ] **2 · Error por candidato:** RMSSE, WAPE y MAE por artículo; WRMSSE y WAPE por candidato y segmento.
- [ ] **3 · Selección por artículo y aceptación agregada.**
- [ ] **4 · Diagnóstico de optimismo** (folds 1–7 → 8–13).
- [ ] **5 · Guardado de la tabla de selección y cierre.**
