# Feature Specification: Segmentación ABC × XYZ y pronóstico de unidades por ratio entre bloques (M5)

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

- Q: ¿Con qué frecuencia temporal se agregan las unidades de cada artículo para calcular su CV? → A: Por periodos de 28 días por defecto, como parámetro configurable (`cv_aggregation`) para experimentar con otras agregaciones (diaria, semanal, periodo de segmentación) solo dentro de train. **Desde el 2026-10-29** ya no hay periodo de segmentación.
- Q: ¿Qué regla individual hace que un artículo se considere "estable" en train? → A: Por defecto, misma clase ABC en las 3 comparaciones (XYZ puede variar y se reporta). La regla se elige con el parámetro `stability_rule`, y las alternativas (ABC y XYZ en las 3; misma celda en ≥2 de 3; clase A en los años 1 a 4) quedan implementadas como reglas intercambiables para experimentar con ellas.
- Q: ¿Qué tratamientos existen para los productos objetivo y qué regla asigna cada uno? → A: Mapeo por celda en la configuración (`treatment_map`): AX → abastecimiento regular (guiado por pronóstico), AY → abastecimiento con revisión periódica, AZ → prioridad de promoción / revisión de demanda.
- Q: ¿Cómo se obtiene la clase de referencia de un artículo frente al promedio de años previos? → A: Se promedia el gasto anual del artículo sobre los años previos activos y se reclasifica ABC sobre ese promedio; para XYZ se promedia su CV anual. **Reemplazada el 2026-10-29** (base año anterior y Y1 fija).
- Q: ¿Cómo se identifica la fecha de lanzamiento de un artículo? → A: Primera `wm_yr_wk` con precio en el archivo de precios, por tienda; el artículo agregado empieza en el lanzamiento más temprano entre tiendas.
- Q: ¿El modelo se elige por producto objetivo o por grupo? → A: No se fija en la spec; es el parámetro `selection_level`, cuyas etiquetas define el analista durante la experimentación, declarado antes del backtesting y congelado con el modelo.

### Session 2026-10-26

- Q: ¿En qué orden se ejecutan la exploración de cinco años, el análisis sobre train, la mirada al año 5 y el pronóstico? → A: Primero la exploración descriptiva global de los cinco años; después la matriz y la estabilidad sobre train, cuyo resultado se congela; luego se replica sobre el año 5 lo congelado; y recién después se hace el pronóstico (backtesting en train y evaluación única en test).
- Q: ¿Cómo se evita que la réplica en el año 5 influya en la elección del modelo? → A: Los candidatos de pronóstico y `selection_level` se declaran antes de la réplica; cambiarlos después exige una nueva versión. La réplica no modifica la lista ni el tratamiento, y ninguna etapa de selección la lee.
- Q: ¿Qué calcula la réplica en el año 5? → A: La matriz ABC × XYZ del año 5 con los cortes, umbrales y agregación del CV congelados; el acuerdo y el PSI del año 5 frente al promedio de los años 1 a 4 y frente a la matriz global de train; y, por producto objetivo, si mantiene su clase y su celda. **Actualizada el 2026-10-29:** bases Y4 y Y1 y medidas pareadas, sin PSI.

### Session 2026-10-27

- Q: ¿Los cortes ABC y los umbrales XYZ son valores fijos? → A: No. Los valores por defecto (80/95 % y 10/25 %) son un punto de partida experimental. El valor definitivo se elige solo con train y se congela (FR-020). El método para elegirlo no se fija en la spec: puede ser determinista o algorítmico, con el criterio declarado antes de ejecutarlo.
- Q: ¿Puede la exploración de cinco años usar el año 5? → A: Sí, solo como línea base descriptiva para saber qué esperar al explorar por años (incluida la exploración inter e intraanual). Probar cortes en ella es exploración y no decide nada; los cortes que se congelan salen de train.

### Session 2026-10-29

Alineación con la constitución 2.0.0, a partir de lo aprendido en la exploración (nb 01–02).

- Q: ¿Contra qué se compara cada año en la estabilidad de train? → A: Con dos bases y los mismos artículos: el año anterior (Yt−1 → Yt) y el año 1 fijo (Y1 → Yt), sin promediar años. Los cortes se eligen en la base y se aplican al objetivo por participación en el gasto. Motivo: el promedio de años previos mueve los cortes y el crecimiento del catálogo encoge la clase A sin que los artículos cambien.
- Q: ¿Con qué se mide la estabilidad? → A: Con medidas pareadas, porque base y objetivo son los mismos artículos: kappa de Cohen plano, Stuart–Maxwell y mayor cambio en puntos porcentuales, más mantiene / mejora / empeora (y mixto en ABC × XYZ). El PSI se deja de usar.
- Q: ¿Se elige un periodo de segmentación (trimestre o cuatrimestre)? → A: No. La exploración intraanual no mostró que la clase dependa de la época del año; la clase se define por año.
- Q: ¿Cómo se tratan los ceros internos? → A: Las rachas de ceros iguales o más largas que el umbral del artículo se tratan como no observadas (NaN) en el CV de XYZ y en la serie de pronóstico. El umbral es el menor k tal que la probabilidad de que la racha más larga sea ≥ k, con días independientes y la tasa de ceros observada del artículo en train, es ≤ α (por defecto 0,01). Wald–Wolfowitz por artículo como diagnóstico; si no se puede estimar, `LONG_RUN` = 7.
- Q: ¿Dónde caen los folds del backtesting? → A: 13 orígenes cada 28 días repartidos en el año 4, igual que el test, para que la selección vea todo el ciclo anual. **Actualizada el 2026-11-02:** con reentrenamiento en cada fold (ver Session 2026-11-02).
- Q: ¿Cómo se eligen los cortes de train? → A: ABC con la regla de % de artículos y % de gasto, y XYZ con la búsqueda de cortes sobre la distribución del CV; se ejecutan sobre train y se congelan.

### Session 2026-11-02

Decisiones de la Etapa 0 del nb 04 (constitución 3.0.0).

- Q: ¿Qué es el artículo y qué se pronostica? → A: `item_id` sumado sobre las 10 tiendas. Se pronostican sus unidades por bloque mediante el ratio r_t = unidades_t ÷ unidades_{t−1} (corregido, constituciones 5.0.0 y 6.0.0: antes, participación en la cadena).
- Q: ¿Con qué periodo y horizonte? → A: Bloques de 28 días (52 en train, 13 en test), un bloque adelante (h = 1), con pronóstico en cada bloque. Los meses calendario no encajan con el split (mayo 2015 queda partido).
- Q: ¿Cómo se pasa de diario a bloque? → A: El valor del bloque es la suma de sus unidades. `min_days` no se aplica en el modelado (solo sirvió para el CV) y ningún bloque se descarta. La serie va del bloque de la primera venta (aunque sea un solo día) al de la última venta; se recortan los ceros iniciales y finales (corregido, constitución 4.0.0).
- Q: ¿Qué candidatos se comparan? → A: Seasonal naive (ŷ_t = y_{t−13}, única línea base), SARIMAX con covariables de calendario y eventos del bloque, y boosting global (XGBoost, LightGBM, CatBoost) con `item_id` y `cell`. En el piloto se compara el boosting global con una variante por artículo en 20–30 artículos antes de decidir.
- Q: ¿Cómo se codifica el calendario? → A: Año como entero; periodo del año 1–13 como categórica ordinal. Eventos por bloque: cantidad de eventos de cada tipo (`n_religious`, `n_national`, `n_cultural`, `n_sporting`; corregido en la Etapa 5 del nb 04, decisión del usuario), con rezagos t−1 y t−2 tomados del calendario. Sin adelanto t+1 (descartado por el usuario, Etapa 6b).
- Q: ¿Cómo se valida? → A: Partición por artículo: test = últimos 13 bloques de su ciclo de vida, validación = los 13 anteriores, train = el resto; mínimo 39 bloques (3 años). 13 folds de origen móvil con ventana expansiva sobre la validación y reentrenamiento en cada bloque (FR-024). El test (nb 05) usa el mismo régimen (corregido, constitución 4.0.0).
- Q: ¿Cómo se elige el modelo de cada artículo? → A: RMSSE por artículo (escala = error del seasonal naive en la muestra, m = 13). Un candidato compite solo si su error en las ventanas de validación es menor que el del seasonal naive en esas ventanas; gana el de menor error y, si ninguno pasa, el seasonal naive. WRMSSE agregado; WAPE, MAE y RMSE como diagnóstico.
- Q: ¿Qué transformación del objetivo se usa? → A: Ninguna: el objetivo es el ratio sin transformar; rezagos y ventanas se calculan sobre él. El pronóstico es unidades_{t−1} × r̂_t y el error se mide en unidades (constitución 5.0.0; la Etapa 6d queda sin efecto).
- Q: ¿Qué pasa con un bloque que pasó `min_days` pero casi no vendió? → A: Si vende menos del 10 % de la mediana de sus últimos 6 bloques válidos (en sus primeros 6 bloques, de la mediana de los válidos de esos 6; el bloque 1 no se evalúa), es un quiebre escondido por ventas sueltas y se trata como no válido (constitución 6.2.0; parámetros `low_block_share` y `low_block_window`).
- Q: ¿Qué pasa con un bloque no válido (menos días observados que `min_days`, incluido el de 0 días)? → A: Toma el valor del bloque anterior del mismo artículo (ratio = 1, "no cambió") y el bloque siguiente se compara contra el último bloque válido. Los bloques no válidos al inicio del ciclo de vida, sin bloque anterior, se eliminan y la serie empieza en el primer bloque válido (antes de la partición y del mínimo de 39 bloques). Se miden todos los bloques (constitución 6.0.0; reemplaza la interpolación lineal de la 5.0.0).
- Q: ¿Y la réplica en el año 5, hecha antes de declarar los candidatos? → A: Se declaran ahora, antes de cualquier backtesting; la réplica no alimentó ninguna selección.

## Orden de ejecución

Las historias están numeradas en el orden en que se ejecutan:

1. Exploración descriptiva de cinco años (US1).
2. Matriz ABC × XYZ sobre train, congelada (US2).
3. Estabilidad, lista objetivo y tratamiento, congelados (US3).
4. Declaración de los candidatos de pronóstico y de `selection_level`, y réplica en el año 5 (US4).
5. Backtesting y selección por artículo en train, y reentrenamiento de los ganadores con los años
   1–4 (US5, nb 04).
6. Evaluación única en test, 13 orígenes del año 5 (US5, nb 05).

## User Scenarios & Testing *(mandatory)*

Actor principal: analista de categoría o abastecimiento que decide qué artículos priorizar en
promoción y abastecimiento. Todos los artículos se analizan agregados sobre todas las tiendas.

### User Story 1 - Exploración descriptiva de cinco años (Priority: P1)

El analista describe primero los productos con los cinco años: ABC por gasto, XYZ,
distribuciones, proporción de ceros, media frente a mediana, y estabilidad anual y global, además de
la exploración inter e intraanual, el origen de los ceros (arranque, rachas largas, ceros finales o
demanda intermitente) y la fuerza estacional. Usa el año 5 solo como línea base. Es solo
descriptivo.

**Why this priority**: es la primera etapa del análisis y da el contexto global antes de trabajar
sobre la partición de train. No decide nada del modelo.

**Independent Test**: se genera el informe descriptivo y se verifica que ningún corte, candidato,
periodo, lista o partición dependa de él.

**Acceptance Scenarios**:

1. **Given** los cinco años de datos validados, **When** se genera la exploración, **Then** el
   informe incluye las descripciones listadas.
2. **Given** la exploración terminada, **When** se ejecutan las etapas siguientes, **Then** ninguna
   lee su salida para decidir cortes, umbrales, periodos, lista objetivo ni modelos.
3. **Given** la matriz de train congelada (US2), **When** se compara con la exploración, **Then**
   cada diferencia se reporta como hallazgo y la matriz congelada no cambia.

---

### User Story 2 - Ranking y matriz ABC × XYZ sobre train (Priority: P2)

El analista obtiene el ranking de artículos por gasto estimado y por variabilidad de unidades, con
la clase ABC, la clase XYZ y la celda de la matriz ABC × XYZ de cada artículo, calculado solo con
datos de train (años 1 a 4).

**Why this priority**: es la base de todas las decisiones. Responde cuánto aporta cada artículo al
gasto estimado y cuáles son de demanda estable o errática, solo con train.

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

### User Story 3 - Estabilidad estacional, lista objetivo y tratamiento (Priority: P3)

El analista evalúa con datos de train si la clasificación de cada artículo se mantiene de un año a
otro. Con eso obtiene la lista congelada de productos
objetivo (clase A y estables en train) y el tratamiento de cada uno según su gasto y su
predictibilidad (CV).

**Why this priority**: decide sobre qué artículos se modela y cómo se tratan. Sin esta lista no
existen la réplica en el año 5 ni el pronóstico del RMV.

**Independent Test**: con la matriz de train congelada, se ejecuta el análisis de estabilidad.
Se verifica que solo use los años 1 a 4, que las comparaciones sigan el esquema declarado y que
la lista resultante solo contenga artículos de clase A que cumplen el criterio de estabilidad.

**Acceptance Scenarios**:

1. **Given** la regla de estabilidad y los parámetros declarados antes de ejecutar, **When** se
   ejecuta el análisis, **Then** el registro muestra que se declararon antes de la ejecución.
2. **Given** los años 1 a 4, **When** se comparan, **Then** cada año t (2 a 4) se compara con el
   año anterior y con el año 1, solo con los artículos activos todos los días de los dos años, con
   los cortes elegidos en la base y aplicados al objetivo por participación en el gasto.
3. **Given** cada comparación, **When** se miden los cambios, **Then** se reportan para ABC, XYZ y
   ABC × XYZ el kappa de Cohen plano, Stuart–Maxwell, el mayor cambio en puntos porcentuales y el %
   que mantiene, mejora, empeora (y mixto en ABC × XYZ), junto con una línea base de ruido
   dentro del mismo año. La matriz global de train se compara con las anuales con las mismas
   medidas.
4. **Given** el criterio de estabilidad por artículo, **When** se construye la lista objetivo,
   **Then** la lista contiene solo artículos de clase A estables en train, cada uno con su
   tratamiento, y queda serializada y congelada.
5. **Given** cualquier dato del año 5, **When** se ejecuta este análisis, **Then** el año 5 no
   participa en ningún cálculo.

---

### User Story 4 - Réplica sobre el año 5 de lo congelado en train (Priority: P4)

Con la lista objetivo congelada y los candidatos de pronóstico ya declarados, el analista replica
sobre el año 5 la matriz ABC × XYZ y la estabilidad con los cortes, umbrales y agregación
congelados en train, antes de pronosticar.

**Why this priority**: muestra, antes de pronosticar, si la segmentación congelada en train se
sostiene en el periodo de test, sin cambiarla.

**Independent Test**: se ejecuta con la lista congelada y los candidatos de pronóstico declarados.
Se verifica que la lista, el tratamiento y los candidatos sean idénticos antes y después, y que
ninguna etapa de selección de modelos lea el resultado.

**Acceptance Scenarios**:

1. **Given** la lista objetivo congelada y los candidatos de pronóstico y `selection_level`
   declarados, **When** se ejecuta la réplica, **Then** se calcula la matriz ABC × XYZ del año 5
   con los parámetros congelados, las medidas pareadas del año 5 frente al año 4 y frente al
   año 1 (FR-017), y por producto objetivo si mantiene su clase y su celda. La lista y el
   tratamiento no cambian.
2. **Given** que la lista objetivo no está congelada o que los candidatos de pronóstico no están
   declarados, **When** se pide la réplica, **Then** el proceso se detiene y lo indica.
3. **Given** la réplica terminada, **When** se ejecuta el backtesting, **Then** ninguna etapa de
   selección lee el resultado de la réplica ni datos del año 5.

---

### User Story 5 - Pronóstico de unidades por bloque de 28 días evaluado en test (Priority: P5)

El analista obtiene para cada producto objetivo de clase A el pronóstico de sus unidades del
bloque de 28 días siguiente, vía el ratio entre bloques, con el mejor modelo del artículo entre seasonal naive,
SARIMAX, XGBoost, LightGBM y CatBoost. El modelo se elige solo con backtesting en train y se
evalúa una única vez en el test (año 5, 13 orígenes cada 28 días, sin solapamiento).

**Why this priority**: es la última etapa y cierra el RMV. Depende de la lista congelada (US3) y
de la réplica en el año 5 (US4).

**Independent Test**: con la lista congelada, se ejecuta la selección en train y la evaluación en
test. Se verifica que el WRMSSE del conjunto elegido sea menor que el del seasonal naive en el
mismo conjunto y los mismos 13 orígenes.

**Acceptance Scenarios**:

1. **Given** los candidatos de búsqueda declarados antes de la réplica en el año 5, **When** se
   ejecuta el backtesting en train, **Then** se elige un modelo por artículo usando solo train, y
   el modelo y sus hiperparámetros u órdenes quedan congelados.
2. **Given** un candidato de búsqueda cambiado después de la réplica en el año 5, **When** se pide
   el backtesting, **Then** el proceso se detiene y exige una nueva versión.
3. **Given** el modelo congelado, **When** se evalúa en test, **Then** se producen pronósticos en
   13 orígenes consecutivos cada 28 días (h = 1 bloque). En cada origen el modelo se reentrena con
   todo lo observado hasta ese origen, con hiperparámetros y órdenes congelados.
4. **Given** el ratio pronosticado r̂_t, **When** se convierte a unidades, **Then** se usa
   unidades_{t−1} × r̂_t con las unidades observadas del bloque anterior.
5. **Given** los resultados de test, **When** se compara con la línea base, **Then** el conjunto
   elegido solo se acepta si su WRMSSE es menor que el del seasonal naive. RMSSE por artículo,
   WAPE, MAE y RMSE se reportan como diagnóstico sin umbral, y MAPE no se usa.
6. **Given** un ajuste de modelo que falla, **When** ocurre el fallo, **Then** se registra con el
   artículo y el candidato, y el proceso continúa con los demás candidatos.

---

### Edge Cases

- **Unidades no enteras o negativas:** la fila se rechaza y se registra con motivo, fila y fuente.
- **Precio ausente o no positivo con unidades vendidas:** la combinación tienda-artículo-semana se
  rechaza y se registra. El gasto de esas filas no se imputa.
- **Fechas faltantes o duplicadas en el calendario:** el proceso se detiene.
- **Artículo con media de unidades igual a cero en el periodo (CV indefinido):** no recibe clase
  XYZ y se registra con motivo.
- **Artículo sin ventas en un año de train después de su lanzamiento:** se conservan los ceros
  como ceros ambiguos; no se interpretan como quiebre de stock, salvo las rachas que superan el
  umbral del artículo (FR-032), que se tratan como no observadas.
- **Artículo sin días con cero o con ceros en todos los días activos:** la tasa de ceros es 0 o 1
  y el umbral no se puede estimar; se usa `LONG_RUN` y se registra el motivo.
- **Artículo cuyo gasto acumulado cruza un corte ABC:** recibe la clase del tramo en que está
  su porcentaje acumulado antes de sumarse (ver Assumptions).
- **Empates de gasto en el límite de un corte:** el orden se desempata de forma determinista por
  identificador de artículo para que el resultado sea reproducible.
- **Día con total de unidades de la cadena igual a cero:** la participación está indefinida; el día
  se registra y no entra en el cálculo de participación.
- **Bloque con menos días observados que `min_days` (incluido el de 0 días), o que vende menos del
  10 % de la mediana de sus últimos 6 bloques válidos:** toma el valor del bloque anterior; al
  inicio del ciclo de vida se elimina. Se mide como cualquier otro bloque.
- **Lista objetivo vacía:** el proceso se detiene y lo reporta. No se relajan criterios después de
  ver el año 5.
- **Producto objetivo que cambia de clase en el año 5:** se reporta en la réplica; la lista y el
  tratamiento no cambian.
- **Artículo con menos de 39 bloques de ciclo de vida:** se excluye del modelado con su motivo
  (FR-035).
- **Ningún candidato le gana al seasonal naive en un artículo (validación):** el artículo usa el
  seasonal naive.
- **El conjunto elegido no supera al seasonal naive en WRMSSE:** el RMV no se cierra; se reporta
  el resultado.
- **Parámetro fuera de rango** (cortes ABC no crecientes, umbrales XYZ negativos o no crecientes,
  α fuera de (0, 1) o `LONG_RUN` < 1): el proceso se detiene antes de calcular.
- **Cambio de parámetro después de la réplica en el año 5 o de ver el test:** exige una nueva
  versión y repetir desde la matriz sobre train.

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
- **FR-004**: El sistema DEBE identificar la semana de lanzamiento de cada artículo por tienda
  como la primera `wm_yr_wk` con precio en el archivo de precios, y excluir las semanas previas
  de esa tienda de entrenamiento, evaluación y cálculos de uso, gasto y CV. El lanzamiento del
  artículo agregado es el más temprano entre sus tiendas. Las filas de prelanzamiento sin precio
  no se cuentan como rechazos por precio ausente.
- **FR-005**: El sistema DEBE partir los datos en un único train (años 1 a 4) y un único test
  (año 5, 13 bloques de 28 días sin solapamiento), con fechas exactas fijadas en la configuración,
  para la segmentación. En el pronóstico la partición es por artículo: test = últimos 13 bloques
  de su ciclo de vida (el año 5 si vende hasta d_1941), validación = los 13 anteriores y train =
  el resto.

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
  corte B (por defecto 95 %) y C al resto. Los cortes por defecto son un punto de partida
  experimental; los de train salen de la regla de % de artículos y % de gasto (ver FR-013).
- **FR-012**: El sistema DEBE calcular el CV de cada artículo agregado como desviación estándar /
  media de sus unidades, en porcentaje, sobre la serie de unidades agregadas según el parámetro
  `cv_aggregation` (por defecto, bloques de 28 días alineados con el horizonte de pronóstico;
  alternativas: diaria o semanal `wm_yr_wk`). Los bloques incompletos por lanzamiento, por el
  borde del periodo o por días no observados (FR-032) se excluyen del cálculo. Las alternativas a
  comparar y el criterio de elección DEBEN declararse antes de experimentar, la elección se hace
  solo con train y el valor elegido queda congelado con la matriz.
- **FR-013**: El sistema DEBE asignar X si CV < umbral X (por defecto 10 %), Y si umbral X ≤ CV <
  umbral Z (por defecto 25 %) y Z si CV ≥ umbral Z. Los valores por defecto son de práctica y
  son un punto de partida experimental: los definitivos se eligen solo con train y se congelan
  con la matriz (FR-020). Método declarado: ABC con la regla de % de artículos y % de gasto
  (A 10–20 % y 60–80 %; B 20–30 % y 20–30 %; menor violación y, en empate, el más cercano al
  centro) y XYZ con la búsqueda de cortes sobre la distribución del CV (mínimo de artículos por
  segmento y gasto medio descendente de X a Z).
- **FR-032**: El sistema DEBE estimar con train, para cada artículo, un umbral de racha de ceros:
  el menor k tal que la probabilidad de que la racha de ceros más larga sea ≥ k, en sus n días
  activos y con días independientes con su tasa de ceros observada p₀, sea ≤ α (por defecto
  0,01, en la configuración). Las rachas internas de ceros de largo ≥ k se tratan como no
  observadas (NaN) en el CV de XYZ y en la serie de pronóstico; en la segmentación, los ceros
  de arranque y finales se tratan igual que en la exploración. En la serie de pronóstico (nb 04 y
  nb 05) el ciclo de vida va de la primera a la última venta, igual que en la segmentación. La
  prueba de rachas de
  Wald–Wolfowitz por artículo se
  reporta como diagnóstico. Si el umbral no se puede estimar, se usa `LONG_RUN` (por defecto 7) y
  se registra el motivo. Los umbrales se congelan (FR-020).
- **FR-014**: El sistema DEBE entregar el ranking con gasto, porcentaje acumulado, CV, clase ABC,
  clase XYZ y celda ABC × XYZ de cada artículo, y serializarlo una vez.

**Estabilidad y lista objetivo (train)**

- **FR-015**: El sistema DEBE exigir que la regla de estabilidad, α, `LONG_RUN` y el mapeo de
  tratamientos estén declarados en la configuración antes de ejecutar el análisis de estabilidad.
  La clase se define por año; NO se elige un periodo de segmentación dentro del año.
- **FR-016**: El sistema DEBE comparar cada año t de train (2 a 4) con dos bases: el año anterior
  (t − 1 → t) y el año 1 (1 → t). En cada comparación entran solo los artículos activos todos los
  días de los dos años, sin promediar años. Los cortes ABC y XYZ se eligen en la base con el
  método de FR-013; el objetivo se clasifica ABC por su participación en el gasto con las
  fronteras de la base, y XYZ con los umbrales de la base.
- **FR-017**: El sistema DEBE medir cada comparación, y la de la matriz global de train frente a
  cada matriz anual, para ABC, XYZ y ABC × XYZ, con: kappa de Cohen plano; Stuart–Maxwell
  (homogeneidad marginal pareada); el mayor cambio del % de artículos de una clase, en puntos
  porcentuales; y el % que mantiene, mejora o empeora (en ABC × XYZ, por dimensión, con `mixto`
  cuando sube en una y baja en la otra). Como línea base de ruido, DEBE medir lo mismo entre
  semanas alternas del mismo año. El PSI NO se usa.
- **FR-018**: El sistema DEBE marcar como estable en train a un artículo según la regla elegida
  en el parámetro `stability_rule`. Por defecto (`abc_all`), un artículo es estable si tiene la
  misma clase ABC en las 3 comparaciones con la base del año anterior (año 2, 3 y 4). La clase XYZ
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
  estables en train y asignar a cada uno un tratamiento según su celda ABC × XYZ, con un mapeo
  celda → tratamiento declarado en el archivo de configuración (parámetro `treatment_map`). Por
  defecto: **AX** → abastecimiento regular (guiado por pronóstico), **AY** → abastecimiento con
  revisión periódica, **AZ** → prioridad de promoción / revisión de demanda. Un producto objetivo
  cuya celda no tenga tratamiento en el mapeo detiene el proceso.
- **FR-020**: El sistema DEBE serializar y congelar la lista objetivo, su tratamiento, los cortes
  ABC, los umbrales XYZ, la agregación del CV, los umbrales de racha por artículo, la regla de
  estabilidad y el mapeo de tratamientos. Las etapas siguientes DEBEN cargarlos sin recalcularlos.
- **FR-021**: El análisis de estabilidad NO DEBE usar ningún dato del año 5.

**Pronóstico y evaluación**

- **FR-022**: El sistema DEBE pronosticar, para cada producto objetivo de clase A, el ratio del
  bloque de 28 días siguiente (h = 1): r_t = unidades del artículo agregado en el bloque t /
  unidades en el bloque t − 1, sin transformar. El pronóstico en unidades es unidades_{t−1} × r̂_t.
- **FR-023**: El sistema DEBE comparar como candidatos el seasonal naive (ŷ_t = y_{t−13}, única
  línea base), SARIMAX con covariables exógenas del bloque, y boosting global (XGBoost, LightGBM y
  CatBoost, con `item_id` y `cell` como variables). Los candidatos y sus espacios de búsqueda se
  declaran antes del backtesting; un candidato cambiado después DEBE detener el backtesting.
- **FR-024**: La selección de modelos, órdenes e hiperparámetros DEBE hacerse solo con backtesting
  dentro de train, con 13 folds de origen móvil y ventana expansiva sobre los 13 bloques de
  validación de cada artículo (FR-005): el fold k entrena con todo lo anterior a la validación más
  los bloques 1 a k − 1 de la validación y predice el bloque k, con reentrenamiento en cada fold. Una configuración de boosting se elige por su error promedio en
  los 13 folds; sin early stopping: `n_estimators` es un hiperparámetro más de la
  búsqueda (decisión del usuario, constitución 6.1.0). Los órdenes SARIMAX se eligen una vez por AIC con los bloques anteriores
  a la validación y sus coeficientes se reestiman en cada fold. El modelo se elige por artículo
  (`selection_level` = item) y los ganadores se reentrenan con todo lo anterior al test.
- **FR-025**: El sistema DEBE evaluar en test cada modelo una única vez, en 13 orígenes cada 28
  días, con el régimen del backtesting: en cada origen se reentrena con todo lo observado hasta
  ese origen, con hiperparámetros y órdenes congelados. Ninguna selección se repite con test.
- **FR-027**: El sistema DEBE calcular RMSSE por artículo (escala = error del seasonal naive en la
  muestra, m = 13). Un candidato compite en un artículo solo si su error en las ventanas de
  validación es menor que el del seasonal naive en esas mismas ventanas; entre los que pasan gana
  el de menor error y, si ninguno pasa, el artículo usa el seasonal naive. WRMSSE agregado es el
  criterio principal; WAPE, MAE y RMSE son diagnóstico. El conjunto elegido DEBE rechazarse si su
  WRMSSE no es menor que el del seasonal naive. Todo error se calcula en unidades.
- **FR-028**: El sistema DEBE registrar cada fallo de ajuste con el artículo y el candidato.
- **FR-033**: Las covariables DEBEN conocerse de antemano para el bloque pronosticado (año entero,
  periodo del año 1–13 como categórica ordinal, banderas de evento del bloque) o estar rezagadas.
  Las ventanas móviles usan solo bloques anteriores al registro (shift(1) antes de rolling).
  Rezagos y ventanas se calculan por artículo sobre la serie completa antes del split, de modo que
  los primeros bloques de test toman su memoria del final de train.
- **FR-034**: El objetivo es el ratio sin transformar; rezagos y ventanas se calculan sobre
  él, y cada pronóstico se reconstruye a unidades antes de medir el error.
- **FR-035**: Solo se modelan los artículos con al menos 39 bloques de ciclo de vida (3 años: train
  ≥ 13, validación 13 y test 13). Los demás se excluyen y se registran con su motivo. Si el test de
  un artículo cae en fechas que para otros son train o validación, se documenta.

**Exploración, réplica en el año 5 e informe**

- **FR-029**: El sistema DEBE producir, como primera etapa del análisis y antes de la matriz sobre
  train, una exploración descriptiva de cinco años, incluido el año 5 como línea base, que no alimente
  ninguna decisión; probar cortes en ella es solo exploración. Sus
  diferencias con la matriz de train congelada se reportan como hallazgo.
- **FR-030**: Una vez congelados la lista objetivo y su tratamiento (FR-020), y declarados los
  candidatos de pronóstico y `selection_level` (FR-023, FR-024), y antes del backtesting, el
  sistema DEBE replicar sobre el año 5 la matriz ABC × XYZ con los cortes, umbrales y agregación
  del CV congelados, y reportar las medidas de FR-017 del año 5 frente al año 4 y frente al año
  1, y por producto objetivo si mantiene su clase y su celda.
  La réplica NO DEBE modificar la lista, el tratamiento ni los candidatos, y ninguna etapa de
  selección de modelos DEBE leerla.
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
  global o año).
- **Medida de estabilidad**: kappa, Stuart–Maxwell, cambio en puntos porcentuales y movimientos
  entre una matriz y su base, por nivel (ABC, XYZ, ABC × XYZ).
- **Umbral de racha**: por artículo, largo mínimo de una racha de ceros no observada, con su tasa
  de ceros, el diagnóstico de Wald–Wolfowitz y el motivo si se usó `LONG_RUN`. Congelado.
- **Producto objetivo**: artículo de clase A estable en train, con su tratamiento. Congelado.
- **Réplica del año 5**: matriz y estabilidad del año 5 calculadas con los parámetros congelados,
  y permanencia de clase y celda de cada producto objetivo. Solo diagnóstico.
- **Configuración**: parámetros con valor por defecto e historial de cambios.
- **Registro de rechazos y fallos**: motivo, fila, fuente, o artículo y candidato del modelo.
- **Pronóstico**: ratio y unidades por producto objetivo, origen y bloque pronosticado.
- **Selección por artículo**: candidato ganador, su error de validación y el del seasonal naive.
- **Resultado de evaluación**: RMSSE por artículo, WRMSSE, WAPE, MAE y RMSE, frente al seasonal
  naive.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: El 100 % de los artículos con actividad en train tiene clase ABC y XYZ, o un motivo de
  exclusión registrado.
- **SC-002**: El 100 % de los datos rechazados aparece en el registro con motivo, fila y fuente.
- **SC-003**: En una verificación manual de al menos 10 artículos, el gasto, el porcentaje
  acumulado, el CV y la clase coinciden con el cálculo reportado.
- **SC-004**: Repetir el proceso con la misma configuración produce la misma matriz, la misma lista
  objetivo y el mismo tratamiento.
- **SC-005**: La lista objetivo, su tratamiento y la declaración de los candidatos de pronóstico
  tienen fecha anterior a la réplica en el año 5, y la réplica tiene fecha anterior al
  backtesting y a la primera evaluación en test. Ninguna decisión usa datos del año 5.
  Excepción registrada (2026-11-02): los candidatos se declararon después de la réplica (nb 03,
  Etapa 13) y antes de cualquier backtesting; la réplica no alimentó ninguna selección.
- **SC-006**: El conjunto de modelos elegidos por artículo tiene un WRMSSE menor que el del
  seasonal naive sobre los mismos productos objetivo y los mismos 13 orígenes.
- **SC-007**: Cada modelo se evalúa en test exactamente una vez.
- **SC-008**: Con el informe final, un analista identifica para cada producto objetivo su clase,
  su aporte al gasto estimado, su tratamiento y su pronóstico, sin consultar otra fuente.
- **SC-009**: El informe incluye la advertencia de backtesting y sesgo de selección, WAPE y RMSE
  como diagnóstico, y la réplica en el año 5 del 100 % de los productos objetivo.

## Entregables

- **Entregable 1 (esta especificación):** el RMV: matriz ABC × XYZ sobre train, lista objetivo con
  tratamiento y pronóstico de unidades a 28 días (por ratio entre bloques) de los productos objetivo de clase A.
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
- Pronosticar el total de la cadena.
- Pronosticar a nivel artículo-tienda o extender a todo el catálogo.

## Assumptions

- **Datos:** M5 (Kaggle) es la única fuente. No incluye inventario, costos de compra, costo
  unitario, lead time ni pedidos; por eso el gasto usa el precio de venta como proxy.
- **Rango temporal:** M5 cubre aproximadamente del 2011-01-29 al 2016-06-19. Las fechas exactas de
  train y test, y cómo se reparten los días que sobran de los cinco años, se fijan en el plan tras
  verificarlas en la documentación oficial.
- **Ceros:** los ceros dentro del periodo activo son ambiguos y se mantienen como ceros, salvo
  las rachas que superan el umbral del artículo (FR-032). El umbral es conservador: la tasa de
  ceros observada incluye esas mismas rachas.
- **Corte ABC:** un artículo es A si su porcentaje acumulado antes de sumarse es menor que el
  corte A; así el artículo que cruza el 80 % queda en A. Ídem para B.
- **Límites XYZ:** se usa X < 10 %, 10 % ≤ Y < 25 %, Z ≥ 25 %. La descripción original decía
  "Z > 25 %", lo que dejaba sin clase un CV de exactamente 25 %.
- **Bases de comparación:** año anterior y año 1 (FR-016). Para la réplica en el año 5, el año 4 y
  el año 1.
- **Ponderación del WRMSSE:** sigue la definición de Makridakis et al. (2022), con pesos por gasto
  estimado de los últimos 28 días de train (el último bloque) y escala por el error del seasonal
  naive (m = 13 bloques) en la muestra.
- **Pendientes que se resuelven en el plan o la experimentación:** valor del periodo de uso,
  umbral de kappa para leer la estabilidad, significado y cobertura de `wm_yr_wk`, forma de serialización,
  redondeo del pronóstico en unidades, rezagos y ventanas, espacios de búsqueda de boosting, y versiones de
  Python y librerías.
- **Referencias:** Makridakis et al. (2022) para WRMSSE; las fuentes para ABC-XYZ, umbrales y
  estabilidad se citan en el plan.
