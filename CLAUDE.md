# Instrucciones del proyecto

- La constitución (`.specify/memory/constitution.md`) prevalece sobre cualquier otra práctica.
- Notebook canónico de la segmentación: `notebooks/03_abc_xyz_train.ipynb`.
- **Notebook en curso: nb 04.** Registro de etapas y su estado:
  `specs/001-abc-xyz-share-forecast/nb04-etapas.md`. Léelo al empezar y márcalo al cerrar cada
  etapa.
- **Notebook experimental en curso: nb 03b** (`notebooks/03b_tratamiento_por_tienda.ipynb`),
  tratamiento por tienda; independiente del nb 04, no persiste nada. Registro:
  `specs/001-abc-xyz-share-forecast/nb03b-etapas.md`. Mismas reglas R1–R8.

## Reglas de trabajo por etapas (nb 04)

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
