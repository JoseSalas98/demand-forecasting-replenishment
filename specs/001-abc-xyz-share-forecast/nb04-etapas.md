# nb 04 · Registro de etapas

Notebook: `notebooks/04_feature_engineering_y_seleccion_de_modelo.ipynb` (nuevo; el anterior quedó como
`04_feature_engineering_y_seleccion_de_modelo_superseded.ipynb`, registro en `nb04-superseded-etapas.md`).
Constitución 7.0.0: carga los datasets del nb 03b y los hiperparámetros del nb 03c y reentrena el CatBoost global con
todo el train (años 1–4, incluida la ventana de backtesting), sin ninguna búsqueda. La evaluación en test es del nb 05.
Reglas de trabajo: ver `CLAUDE.md` (R1–R8).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas (propuesta, a confirmar en la Etapa 0)

- [ ] **0 · Decisiones y reglas.**
- [ ] **1 · Carga** de los datasets del 03b y de la mejor configuración de CatBoost del 03c; chequeos de coherencia.
- [ ] **2 · Reentrenamiento** del CatBoost global con todo el train (filas con ratio tratado fuera del ajuste).
- [ ] **3 · Guardado del modelo y cierre.**
