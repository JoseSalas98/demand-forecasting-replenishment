# Instrucciones del proyecto

- La constitución (`.specify/memory/constitution.md`) prevalece sobre cualquier otra práctica.
- Notebook canónico de la segmentación: `notebooks/03_abc_xyz_train.ipynb`.
- **Flujo de modelado (constitución 7.1.0):** 03 → 03b (tratamiento por tienda y datasets) → 03c (búsqueda del
  boosting) → 03d (diagnóstico de la selección) → 04 (reentrenamiento de CatBoost global) → 05 (test) → 06 (impacto
  en el negocio). Lo que guardan el 03b, el 03c, el 03d y el 05 se carga, no se recalcula; sus notebooks tienen
  candado (`RERUN`/`OVERWRITE`).
- **nb 06** (`notebooks/06_reporte_final.ipynb`): impacto del pronóstico en dinero (venta capturada, perdida y
  exceso) contra la demanda tratada; guarda en `data/interim/nb06_impact/` (cerrado). Registro:
  `specs/001-abc-xyz-share-forecast/nb06-etapas.md`.
- **Siguiente:** informe final en `reports/`, desde lo guardado por el nb 05 y el nb 06 (constitución 7.1.0).
- **nb 05** (`notebooks/05_evaluacion_backtesting.ipynb`): evaluación única en test, a un paso y encadenado;
  guarda en `data/interim/nb05_test/`. Registro: `specs/001-abc-xyz-share-forecast/nb05-etapas.md`.
- **nb 04** (`notebooks/04_feature_engineering_y_seleccion_de_modelo.ipynb`): reentrena el CatBoost global y lo guarda
  en `models/nb04_catboost/` (cerrado). Registro: `specs/001-abc-xyz-share-forecast/nb04-etapas.md`. El nb 04 anterior
  quedó como `..._superseded.ipynb`, sin uso (registro `nb04-superseded-etapas.md`).
- **nb 03b** (`notebooks/03b_tratamiento_por_tienda.ipynb`): tratamiento por tienda; persiste los datasets y los
  puntos de control SARIMAX. Registro: `specs/001-abc-xyz-share-forecast/nb03b-etapas.md`.
- **nb 03c** (`notebooks/03c_tratamiento_por_tienda.ipynb`): búsqueda de hiperparámetros del boosting; persiste solo
  su búsqueda. Registro: `specs/001-abc-xyz-share-forecast/nb03c-etapas.md`.
- **nb 03d** (`notebooks/03d_seleccion_por_articulo.ipynb`): diagnóstico de la selección por artículo; persiste solo
  `selection.csv`. Registro: `specs/001-abc-xyz-share-forecast/nb03d-etapas.md`.
- Mismas reglas R1–R8 en todos.

## Reglas de trabajo por etapas (nb 04 en adelante)

- **R1.** Una etapa a la vez. Al terminarla: resumen, resultados y chequeos, dudas pendientes, y
  DETENTE hasta que el usuario escriba "aprobado" o dé feedback (con feedback se corrige la misma
  etapa y se vuelve a detener).
- **R2.** El código vive en `src/` y se escribe de forma incremental: solo lo que la etapa
  necesita; nada de etapas futuras ni pipelines completos. Al cerrar, indicar qué funciones se
  agregaron o modificaron y en qué archivo. Módulo nuevo solo con aprobación.
- **R3.** Celdas atómicas en este orden: (a) markdown con objetivo, entradas, salidas y decisión
  justificada; (b) importación desde `src/` (con `%autoreload`); (c) ejecución, una llamada
  principal; (d) validación (shapes, nulos, rangos, asserts, muestra).
- **R4.** Ante ambigüedad, preguntar. Todo número (ventanas, rezagos, n_splits, espacios de
  búsqueda) con su justificación.
- **R5.** Sin fuga: nada calculado con test entra a train. Lo congelado en el nb 03 se carga, no se
  recalcula.
- **R6.** Reproducibilidad: semillas fijas, versiones impresas al inicio, rutas relativas.
- **R7.** Antes de un proceso largo, piloto con pocos artículos, tiempo por artículo y estimación
  total; esperar aprobación.
- **R8.** Sin commit ni push. Comentarios y textos en español; identificadores de código en inglés.
- El usuario ejecuta los notebooks; no ejecutarlos con nbconvert sin que lo pida.
