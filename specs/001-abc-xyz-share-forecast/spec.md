# Feature Specification: Segmentación ABC × XYZ y pronóstico de participación (M5)

**Feature Branch**: `main` (no se creó rama; no hay hook `before_specify` registrado)

**Created**: 2026-10-24

**Status**: Draft

**Input**: User description: "Especificación: Rotación y abastecimiento con M5 (v2.2, preliminar).
Decidir qué artículos de mayor relevancia priorizar en promoción y abastecimiento, según su peso en
el gasto estimado y su variabilidad de demanda."

## Pregunta de negocio

¿Qué artículos son de alto y lento movimiento, considerando la estacionalidad, y cuánto aporta cada
uno al gasto estimado? ¿Cuáles necesitan más promoción?

## Clarifications

### Session 2026-10-24

- Q: ¿Con qué frecuencia temporal se agregan las unidades de cada artículo para calcular su CV? → A: Por periodos de 28 días por defecto, como parámetro configurable (`cv_aggregation`) para experimentar con otras agregaciones (diaria, semanal, periodo de segmentación) solo dentro de train.
- Q: ¿Qué regla individual hace que un artículo se considere "estable" en train? → A: Por defecto, misma clase ABC en las 3 comparaciones (XYZ puede variar y se reporta). La regla se elige con el parámetro `stability_rule`, y las alternativas (ABC y XYZ en las 3; misma celda en ≥2 de 3; clase A en los años 1 a 4) quedan implementadas como reglas intercambiables para experimentar con ellas.

## User Scenarios & Testing *(mandatory)*

Actor principal: analista de categoría o abastecimiento que decide qué artículos priorizar en
promoción y abastecimiento. Todos los artículos se analizan agregados sobre todas las tiendas.

### User Story 1 - Ranking y matriz ABC × XYZ sobre train (Priority: P1)

El analista obtiene el ranking de artículos por gasto estimado y por variabilidad de unidades, con
la clase ABC, la clase XYZ y la celda de la matriz ABC × XYZ de cada artículo, calculado solo con
datos de train (años 1 a 4).

**Why this priority**: es la base de todo lo demás. Por sí sola responde cuánto aporta cada
artículo al gasto estimado y cuáles son de demanda estable o errática.

**Independent Test**: con los datos de train y la configuración por defecto, se genera la matriz.
Se verifica a mano, para una muestra de artículos, el gasto (unidades × precio de cada
tienda-semana), el porcentaje acumulado, el CV y la clase asignada.

**Acceptance Scenarios**:

1. **Given** los datos de train validados y los cortes por defecto (80 % y 95 %), **When** se
   calcula la clasificación ABC, **Then** los artículos aparecen ordenados de mayor a menor gasto
   estimado, con gasto acumulado, porcentaje acumulado y clase A, B o C según los cortes.
2. **Given** un artículo vendido en varias tiendas con precios distintos por semana, **When** se
   calcula su gasto, **Then** el gasto es la suma de unidades × precio de cada tienda-día y no
   coincide con precio medio × unidades totales cuando los precios difieren.
3. **Given** los umbrales XYZ por defecto, **When** se calcula el CV de cada artículo, **Then**
   cada artículo recibe X, Y o Z según su CV y los umbrales configurados.
4. **Given** un artículo lanzado a mitad de train, **When** se calculan uso, gasto y CV, **Then**
   las semanas previas a su lanzamiento no entran en ningún cálculo.
5. **Given** la matriz calculada, **When** se cierra este análisis, **Then** la matriz se
   serializa una vez y las etapas siguientes la cargan sin recalcularla.

---

### User Story 2 - Estabilidad estacional, lista objetivo y tratamiento (Priority: P2)

El analista evalúa con datos de train si la clasificación de cada artículo se mantiene año a año y
dentro de los periodos de segmentación candidatos. Con eso obtiene la lista congelada de productos
objetivo (clase A y estables en train) y el tratamiento de cada uno según su gasto y su
predictibilidad (CV).

**Why this priority**: decide sobre qué artículos se modela y cómo se tratan. Sin esta lista no
existe el pronóstico del RMV.

**Independent Test**: con la matriz de train congelada, se ejecuta el análisis de estabilidad.
Se verifica que solo use los años 1 a 4, que las comparaciones sigan el esquema declarado y que
la lista resultante solo contenga artículos de clase A que cumplen el criterio de estabilidad.

**Acceptance Scenarios**:

1. **Given** candidatos de periodo de segmentación y un criterio de elección declarados antes de
   ejecutar, **When** se ejecuta el análisis, **Then** se elige un periodo según ese criterio y el
   registro muestra que los candidatos y el criterio se declararon antes de la ejecución.
2. **Given** los años 1 a 4, **When** se comparan las distribuciones, **Then** se compara el año 2
   contra el año 1, el año 3 contra el promedio de los años 1 y 2, y el año 4 contra el promedio de
   los años 1 a 3. El promedio de referencia es el promedio simple de las distribuciones previas.
3. **Given** las matrices anuales y la matriz global de train, **When** se miden los acuerdos,
   **Then** se reportan el porcentaje de artículos con la misma clase en cada año y en la global,
   y el PSI sobre la proporción de clases, comparados con sus umbrales configurados.
4. **Given** el criterio de estabilidad por artículo, **When** se construye la lista objetivo,
   **Then** la lista contiene solo artículos de clase A estables en train, cada uno con su
   tratamiento, y queda serializada y congelada.
5. **Given** cualquier dato del año 5, **When** se ejecuta este análisis, **Then** el año 5 no
   participa en ningún cálculo.

---

### User Story 3 - Pronóstico de participación a 28 días evaluado en test (Priority: P3)

El analista obtiene para cada producto objetivo de clase A el pronóstico de su participación en
unidades a 28 días con el mejor modelo entre las líneas base (seasonal naive y medias móviles),
ARIMA y SARIMA. El modelo se elige solo con backtesting en train y se evalúa una única vez en el
test (año 5, 13 orígenes cada 28 días, sin solapamiento).

**Why this priority**: cierra el RMV. Depende de la lista congelada de la historia 2.

**Independent Test**: con la lista congelada, se ejecuta la selección en train y la evaluación en
test. Se verifica que el WRMSSE del modelo elegido sea menor que el del seasonal naive en el
mismo conjunto y los mismos 13 orígenes.

**Acceptance Scenarios**:

1. **Given** los candidatos de búsqueda declarados antes de experimentar, **When** se ejecuta el
   backtesting en train, **Then** se elige un modelo por producto o grupo usando solo train, y el
   modelo y sus parámetros quedan congelados.
2. **Given** el modelo congelado, **When** se evalúa en test, **Then** se producen pronósticos en
   13 orígenes consecutivos cada 28 días, sin reentrenar. Solo se actualizan los datos observados
   y los parámetros se mantienen fijos.
3. **Given** los pronósticos de participación, **When** se convierten a unidades, **Then** se usa
   el total real de unidades de la cadena del periodo; el total de la cadena no se pronostica.
4. **Given** los resultados de test, **When** se compara con la línea base, **Then** el modelo
   solo se acepta si su WRMSSE es menor que el del seasonal naive. WAPE y RMSE se reportan como
   diagnóstico sin umbral, y MAPE no se usa.
5. **Given** un ajuste de modelo que falla, **When** ocurre el fallo, **Then** se registra con el
   artículo y el candidato, y el proceso continúa con los demás candidatos.

---

### User Story 4 - Exploración descriptiva de cinco años (Priority: P4)

El analista describe los productos con los cinco años: ABC por gasto, XYZ, distribuciones,
proporción de ceros, media frente a mediana, y estabilidad anual y global. Es solo descriptivo.

**Why this priority**: da contexto y hallazgos, pero no decide nada del modelo.

**Independent Test**: se genera el informe descriptivo y se verifica que ningún corte, candidato,
periodo, lista o partición dependa de él.

**Acceptance Scenarios**:

1. **Given** los cinco años de datos, **When** se genera la exploración, **Then** el informe
   incluye las descripciones listadas y marca como hallazgo cada diferencia con la matriz de
   train.
2. **Given** una diferencia entre la exploración y la matriz de train, **When** se reporta,
   **Then** la matriz congelada no cambia.

---

### User Story 5 - Diagnóstico posterior del año 5 (Priority: P5)

Después de evaluar el test, el analista recibe la estabilidad de cada producto objetivo congelado
en el año 5.

**Why this priority**: informa la confianza en la lista para la siguiente iteración, sin cambiarla.

**Independent Test**: se ejecuta después de la evaluación en test y se verifica que la lista y el
tratamiento sean idénticos antes y después.

**Acceptance Scenarios**:

1. **Given** la evaluación en test terminada, **When** se calcula el diagnóstico, **Then** se
   reporta la estabilidad del año 5 de cada producto objetivo y la lista congelada no cambia.
2. **Given** que la evaluación en test no terminó, **When** se pide el diagnóstico, **Then** el
   proceso se detiene y lo indica.

---

### Edge Cases

- **Unidades no enteras o negativas:** la fila se rechaza y se registra con motivo, fila y fuente.
- **Precio ausente o no positivo con unidades vendidas:** la combinación tienda-artículo-semana se
  rechaza y se registra. El gasto de esas filas no se imputa.
- **Fechas faltantes o duplicadas en el calendario:** el proceso se detiene.
- **Artículo con media de unidades igual a cero en el periodo (CV indefinido):** no recibe clase
  XYZ y se registra con motivo.
- **Artículo sin ventas en un año de train después de su lanzamiento:** se conservan los ceros
  como ceros ambiguos; no se interpretan como quiebre de stock.
- **Artículo cuyo gasto acumulado cruza un corte ABC:** recibe la clase del tramo en que está
  su porcentaje acumulado antes de sumarse (ver Assumptions).
- **Empates de gasto en el límite de un corte:** el orden se desempata de forma determinista por
  identificador de artículo para que el resultado sea reproducible.
- **Día con total de unidades de la cadena igual a cero:** la participación está indefinida; el día
  se registra y no entra en el cálculo de participación.
- **Lista objetivo vacía:** el proceso se detiene y lo reporta. No se relajan criterios después de
  ver el test.
- **Ningún modelo supera al seasonal naive:** el RMV no se cierra; se reporta el resultado.
- **Parámetro fuera de rango** (cortes ABC no crecientes, umbrales XYZ negativos o no crecientes,
  umbrales PSI negativos): el proceso se detiene antes de calcular.
- **Cambio de parámetro después de ver el test:** exige una nueva versión y repetir desde la matriz
  sobre train.

## Requirements *(mandatory)*

### Functional Requirements

**Datos y validación**

- **FR-001**: El sistema DEBE usar como fuente los archivos originales de M5 (ventas diarias por
  artículo y tienda, precios semanales y calendario) sin modificarlos; cada transformación genera
  un archivo nuevo.
- **FR-002**: El sistema DEBE rechazar y registrar, con motivo, fila y fuente, las unidades que no
  sean enteros no negativos y los precios ausentes o no positivos.
- **FR-003**: El sistema DEBE validar que el calendario tenga fechas válidas y continuas, y
  detenerse si no las tiene.
- **FR-004**: El sistema DEBE identificar la semana de lanzamiento de cada artículo y excluir las
  semanas previas de entrenamiento, evaluación y cálculos de uso, gasto y CV.
- **FR-005**: El sistema DEBE partir los datos en un único train (años 1 a 4) y un único test
  (año 5, 13 bloques de 28 días sin solapamiento), con fechas exactas fijadas en la configuración.

**Configuración**

- **FR-006**: Todos los cortes, umbrales, periodos y candidatos DEBEN leerse de un único archivo
  de configuración, cada uno con valor por defecto documentado.
- **FR-007**: Cada cambio de parámetro DEBE registrarse con fecha, valor anterior, valor nuevo y
  motivo.
- **FR-008**: El sistema DEBE detenerse ante parámetros fuera de rango.

**Clasificación ABC × XYZ (train)**

- **FR-009**: El sistema DEBE calcular el uso de cada artículo agregado como las unidades vendidas
  en el periodo de análisis (por defecto, el periodo completo del conjunto de cálculo).
- **FR-010**: El sistema DEBE calcular el gasto estimado de cada artículo como la suma, sobre
  tiendas y días, de unidades × precio de venta de esa tienda y semana. No DEBE usar precio
  medio × unidades totales.
- **FR-011**: El sistema DEBE ordenar los artículos de mayor a menor gasto, calcular gasto
  acumulado y porcentaje acumulado, y asignar A hasta el corte A (por defecto 80 %), B hasta el
  corte B (por defecto 95 %) y C al resto.
- **FR-012**: El sistema DEBE calcular el CV de cada artículo agregado como desviación estándar /
  media de sus unidades, en porcentaje, sobre la serie de unidades agregadas según el parámetro
  `cv_aggregation` (por defecto, bloques de 28 días alineados con el horizonte de pronóstico;
  alternativas: diaria, semanal `wm_yr_wk` o periodo de segmentación). Los bloques incompletos
  por lanzamiento o por el borde del periodo se excluyen del cálculo. Las alternativas a
  comparar y el criterio de elección DEBEN declararse antes de experimentar, la elección se hace
  solo con train y el valor elegido queda congelado con la matriz.
- **FR-013**: El sistema DEBE asignar X si CV < umbral X (por defecto 10 %), Y si umbral X ≤ CV <
  umbral Z (por defecto 25 %) y Z si CV ≥ umbral Z. Los valores por defecto son de práctica y
  DEBEN verificarse antes de usarse para decidir.
- **FR-014**: El sistema DEBE entregar el ranking con gasto, porcentaje acumulado, CV, clase ABC,
  clase XYZ y celda ABC × XYZ de cada artículo, y serializarlo una vez.

**Estabilidad estacional y lista objetivo (train)**

- **FR-015**: El sistema DEBE exigir que los candidatos de periodo de segmentación y el criterio de
  elección estén declarados en la configuración antes de ejecutar el análisis de estabilidad.
- **FR-016**: El sistema DEBE comparar el año 2 contra el año 1, el año 3 contra el promedio simple
  de los años 1 y 2, y el año 4 contra el promedio simple de los años 1 a 3.
- **FR-017**: El sistema DEBE calcular la matriz global de train y contrastarla con cada matriz
  anual mediante el porcentaje de artículos con la misma clase y el PSI sobre la proporción de
  clases, con umbrales configurables.
- **FR-018**: El sistema DEBE marcar como estable en train a un artículo según la regla elegida
  en el parámetro `stability_rule`. Por defecto (`abc_all`), un artículo es estable si tiene la
  misma clase ABC en las 3 comparaciones (año 2, 3 y 4 frente a su referencia). La clase XYZ
  puede variar, y sus cambios se reportan. DEBEN estar disponibles como reglas intercambiables,
  con la misma entrada (clases por artículo y comparación) y la misma salida (estable sí/no):
  - `abc_all` (por defecto): misma clase ABC en las 3 comparaciones.
  - `abc_xyz_all`: misma clase ABC y misma clase XYZ en las 3 comparaciones.
  - `cell_majority`: misma celda ABC × XYZ en al menos 2 de las 3 comparaciones.
  - `a_every_year`: clase A en cada uno de los años 1 a 4, sin mirar XYZ.
  Las reglas a comparar y el criterio de elección DEBEN declararse antes de experimentar. La
  elección se hace solo con train y la regla elegida queda congelada con la lista objetivo.
  Justificación de la abstracción (Principio III, condición a): es una regla de negocio que
  cambia de forma independiente y tiene varias implementaciones reales.
- **FR-019**: El sistema DEBE construir la lista de productos objetivo con los artículos de clase A
  estables en train y asignar a cada uno un tratamiento según su gasto y su CV, siguiendo
  [NEEDS CLARIFICATION: ¿qué tratamientos existen y qué celda o regla lleva a cada uno? Por
  ejemplo, AX abastecimiento regular, AY abastecimiento con revisión, AZ prioridad de promoción].
- **FR-020**: El sistema DEBE serializar y congelar la lista objetivo, su tratamiento, el periodo
  de segmentación elegido, los cortes ABC, los umbrales XYZ, la agregación del CV y la regla de estabilidad. Las etapas siguientes DEBEN
  cargarlos sin recalcularlos.
- **FR-021**: El análisis de estabilidad NO DEBE usar ningún dato del año 5.

**Pronóstico y evaluación**

- **FR-022**: El sistema DEBE pronosticar la participación en unidades a 28 días de cada producto
  objetivo de clase A: unidades del artículo agregado / unidades totales reales de la cadena.
- **FR-023**: El sistema DEBE comparar como candidatos las líneas base (seasonal naive y medias
  móviles), ARIMA y SARIMA, con candidatos de búsqueda declarados antes de experimentar.
- **FR-024**: La selección de modelos y órdenes DEBE hacerse solo con backtesting dentro de train.
- **FR-025**: El sistema DEBE evaluar en test cada modelo una única vez, en 13 orígenes cada 28 días,
  sin reentrenar y con parámetros fijos, actualizando solo los datos observados.
- **FR-026**: El sistema DEBE convertir la participación a unidades usando el total real de la
  cadena; no DEBE pronosticar el total de la cadena.
- **FR-027**: El sistema DEBE calcular WRMSSE como criterio principal, y WAPE y RMSE como
  diagnóstico. Un modelo DEBE rechazarse si su WRMSSE no es menor que el del seasonal naive.
- **FR-028**: El sistema DEBE registrar cada fallo de ajuste con el artículo y el candidato.

**Exploración, diagnóstico e informe**

- **FR-029**: El sistema DEBE producir una exploración descriptiva de cinco años que no alimente
  ninguna decisión y que reporte como hallazgo sus diferencias con la matriz de train.
- **FR-030**: Después de evaluar el test, el sistema DEBE reportar la estabilidad del año 5 de cada
  producto objetivo sin modificar la lista ni el tratamiento.
- **FR-031**: El informe final DEBE presentar los resultados como backtesting sobre datos
  históricos, con la advertencia de sesgo de selección: valen para el subconjunto evaluado, no
  para todo el catálogo.

### Key Entities

- **Artículo agregado**: artículo sumado sobre todas las tiendas. Atributos: identificador, fecha de
  lanzamiento, uso, gasto estimado, CV, clase ABC, clase XYZ.
- **Venta diaria**: unidades de un artículo en una tienda y un día. Se relaciona con el calendario
  y con el precio semanal.
- **Precio semanal**: precio de venta por tienda, artículo y semana (`wm_yr_wk`). Proxy del costo
  unitario.
- **Calendario**: fechas, semana (`wm_yr_wk`), eventos y SNAP, conocidos de antemano.
- **Matriz ABC × XYZ**: asignación de clases por artículo para un conjunto de cálculo (train
  global, año o periodo de segmentación).
- **Medida de estabilidad**: porcentaje de acuerdo y PSI entre una matriz y su referencia.
- **Producto objetivo**: artículo de clase A estable en train, con su tratamiento. Congelado.
- **Configuración**: parámetros con valor por defecto e historial de cambios.
- **Registro de rechazos y fallos**: motivo, fila, fuente, o artículo y candidato del modelo.
- **Pronóstico**: participación y unidades por producto objetivo, origen y día del horizonte.
- **Resultado de evaluación**: WRMSSE, WAPE y RMSE por modelo, frente al seasonal naive.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: El 100 % de los artículos con actividad en train tiene clase ABC y XYZ, o un motivo de
  exclusión registrado.
- **SC-002**: El 100 % de los datos rechazados aparece en el registro con motivo, fila y fuente.
- **SC-003**: En una verificación manual de al menos 10 artículos, el gasto, el porcentaje
  acumulado, el CV y la clase coinciden con el cálculo reportado.
- **SC-004**: Repetir el proceso con la misma configuración produce la misma matriz, la misma lista
  objetivo y el mismo tratamiento.
- **SC-005**: La lista objetivo y su tratamiento tienen fecha de congelamiento anterior a la primera
  evaluación en test, y ninguna decisión usa datos del año 5.
- **SC-006**: El modelo elegido tiene un WRMSSE menor que el del seasonal naive sobre los mismos
  productos objetivo y los mismos 13 orígenes.
- **SC-007**: Cada modelo se evalúa en test exactamente una vez.
- **SC-008**: Con el informe final, un analista identifica para cada producto objetivo su clase,
  su aporte al gasto estimado, su tratamiento y su pronóstico, sin consultar otra fuente.
- **SC-009**: El informe incluye la advertencia de backtesting y sesgo de selección, WAPE y RMSE
  como diagnóstico, y la estabilidad del año 5 del 100 % de los productos objetivo.

## Entregables

- **Entregable 1 (esta especificación):** el RMV: matriz ABC × XYZ sobre train, lista objetivo con
  tratamiento y pronóstico de participación a 28 días de los productos objetivo de clase A.
- **Entregable 2 (otra iteración, fuera de alcance):** cantidad sugerida de abastecimiento,
  desagregación artículo-tienda, clasificación SBC y extensión a todo el catálogo.

**Criterio de cierre del RMV:** la matriz ABC × XYZ y la lista objetivo están calculadas con train
y congeladas, y el pronóstico de los productos objetivo se evaluó en test con WRMSSE y el modelo
elegido supera al seasonal naive.

## Out of Scope

- Optimizar precios o promociones.
- Modelar inventario o costos logísticos.
- Calcular margen, rotación de inventario o gasto con costo real.
- Construir tablero o interfaz.
- Variables exógenas (ARIMAX/SARIMAX).
- Pronosticar el total de la cadena.
- Pronosticar a nivel artículo-tienda o extender a todo el catálogo.

## Assumptions

- **Datos:** M5 (Kaggle) es la única fuente. No incluye inventario, costos de compra, costo
  unitario, lead time ni pedidos; por eso el gasto usa el precio de venta como proxy.
- **Rango temporal:** M5 cubre aproximadamente del 2011-01-29 al 2016-06-19. Las fechas exactas de
  train y test, y cómo se reparten los días que sobran de los cinco años, se fijan en el plan tras
  verificarlas en la documentación oficial.
- **Ceros:** los ceros dentro del periodo activo son ambiguos y se mantienen como ceros.
- **Corte ABC:** un artículo es A si su porcentaje acumulado antes de sumarse es menor que el
  corte A; así el artículo que cruza el 80 % queda en A. Ídem para B.
- **Límites XYZ:** se usa X < 10 %, 10 % ≤ Y < 25 %, Z ≥ 25 %. La descripción original decía
  "Z > 25 %", lo que dejaba sin clase un CV de exactamente 25 %.
- **Promedio de referencia:** promedio simple de las distribuciones de los años previos.
- **Ponderación del WRMSSE:** sigue la definición de Makridakis et al. (2022), con pesos por gasto
  estimado de los últimos 28 días de train y escala por el error naive de un paso en train.
- **Periodo de segmentación:** intervalo dentro del año (por ejemplo, trimestre o cuatrimestre),
  distinto del periodo estacional m de SARIMA. Los candidatos por defecto son trimestral y
  cuatrimestral.
- **Pendientes que se resuelven en el plan o la experimentación:** valor del periodo de uso,
  umbrales PSI y de acuerdo, significado y cobertura de `wm_yr_wk`, forma de serialización,
  redondeo al convertir participación a unidades, ventanas de medias móviles en días, tamaño del
  subconjunto y candidatos de búsqueda, y versiones de Python y librerías.
- **Referencias:** Makridakis et al. (2022) para WRMSSE; las fuentes para ABC-XYZ, umbrales y
  estabilidad se citan en el plan.
