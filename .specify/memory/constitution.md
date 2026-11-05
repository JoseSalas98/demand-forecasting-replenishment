<!--
Sync Impact Report
- Version change: 6.1.0 → 6.2.0 (MINOR, 2026-11-02)
- Motivo: decisión del usuario en el nb 04 (Etapa 4.2, revisada tras la Etapa 11): un bloque que
  pasó `min_days` pero vende menos del 10 % de la mediana de sus últimos 6 bloques válidos es un
  quiebre escondido por ventas sueltas de pocas unidades, y se trata como no válido. Ejemplo:
  HOUSEHOLD_1_327, bloque 40: 10 días observados, 13 unidades frente a ~2.000 normales.
- Modified principles: II. Reglas de dominio ("Bloque no válido" suma el criterio de unidades).
- Datos (nb 04): 960 bloques marcados por unidades (AX 0,2 %, AY 1,8 %, AZ 8,1 % de sus bloques);
  ratios > 10 en el train bajan de 272 a 33. Siguen 579 artículos.
- Parámetros: `features.low_block_share = 0.1`, `features.low_block_window = 6` (con historial).
- Impacto sobre artefactos congelados: ninguno.
- Artefactos por actualizar: spec.md.

Historial 6.1.0
- Version change: 6.0.0 → 6.1.0 (MINOR, 2026-11-02)
- Motivo: decisión del usuario en la Etapa 10 del nb 04: el boosting no usa early stopping; la
  cantidad de árboles (`n_estimators`) entra como hiperparámetro de la búsqueda. Así ningún bloque
  del entrenamiento de un fold queda fuera del ajuste.
- Modified principles: VI. Método y congelamiento (se reemplaza la frase del early stopping).
- Impacto sobre artefactos congelados: ninguno.
- Artefactos por actualizar: spec.md (FR-024).

Historial 6.0.0
- Version change: 5.0.0 → 6.0.0 (MAJOR, 2026-11-02)
- Motivo del MAJOR: se redefine la regla de bloques sin dato del Principio II (decisión del
  usuario): ya no se interpola ni se excluye de la medición. Todo bloque no válido (días
  observados < `min_days` congelado del artículo, incluidos los de 0 días) toma el valor del
  bloque anterior, y se miden todos los bloques. El objetivo pasa a llamarse "ratio".
- Modified principles:
  - I. Propósito y alcance: "rodamiento" → "ratio".
  - II. Reglas de dominio: "Rodamiento" → "Ratio"; "Bloque sin dato" → "Bloque no válido".
- Datos (nb 04): los bloques iniciales no válidos se eliminan (no hay valor anterior): 95
  bloques de 50 artículos; la serie empieza en el primer bloque válido y la partición y el mínimo
  de 39 bloques se aplican después. 1,215 bloques no válidos completados en 219 artículos (587 sin
  ningún día con dato). Quedan 579 artículos (58,87 % del gasto de train); HOUSEHOLD_1_457 sale
  (48 → 38 bloques).
- Impacto sobre artefactos congelados: ninguno (`min_days` se carga del nb 03).
- Artefactos por actualizar: spec.md (este cambio), plan.md, data-model.md, tasks.md.

Historial 5.0.0
- Version change: 4.0.0 → 5.0.0 (MAJOR, 2026-11-02)
- Motivo del MAJOR: se redefine lo que se pronostica (Principio I): ya no la participación del
  artículo en la cadena, sino sus unidades por bloque, modeladas con el rodamiento
  r_t = unidades_t / unidades_{t−1} sin transformar (decisión del usuario).
- Modified principles:
  - I. Propósito y alcance: pronóstico de unidades por rodamiento.
  - II. Reglas de dominio: nuevas reglas "Rodamiento" y "Bloque sin dato" (interpolación lineal;
    no se miden los bloques imputados ni el primero tras cada hueco).
  - III. Parámetros configurables y VI. Método: se retira la transformación del objetivo (la
    Etapa 6d del nb 04 queda sin efecto).
- Datos (nb 04): 595 bloques sin ningún día con dato en 268 huecos de 150 artículos.
- Impacto sobre artefactos congelados: ninguno.
- Artefactos por actualizar: spec.md (este cambio; se elimina FR-026), plan.md, data-model.md
  (entidad Pronóstico cita FR-026), tasks.md.

Historial 4.0.0
- Version change: 3.0.0 → 4.0.0 (MAJOR, 2026-11-02)
- Motivo del MAJOR: se redefine la partición del Principio VI (NO NEGOCIABLE): pasa de un único
  test (año 5) a una partición por artículo: test = últimos 13 bloques de 28 días de su ciclo de
  vida, validación = los 13 anteriores, con un mínimo de 39 bloques (3 años).
- Modified principles:
  - II. Reglas de dominio: en el pronóstico el ciclo de vida va de la primera a la última venta
    (se recortan los ceros finales, como en la segmentación). Corrige la 3.0.0.
  - VI. Método y congelamiento: partición por artículo, historial mínimo y folds sobre los 13
    bloques de validación de cada artículo.
- Datos (nb 04): quedan 580 de los 592 artículos de `targets` (58,93 % del gasto de train); 12
  salen por tener menos de 39 bloques. 3 artículos tienen un test distinto del año 5.
- Impacto sobre artefactos congelados: ninguno (`targets` no cambia; el filtro se aplica en nb 04).
- Artefactos por actualizar: spec.md (este cambio), plan.md, data-model.md, tasks.md.

Historial 3.0.0
- Version change: 2.1.0 → 3.0.0 (MAJOR, 2026-11-02)
- Motivo del MAJOR: se redefine una regla del Principio VI (NO NEGOCIABLE): en test y en el
  backtesting los modelos se reentrenan en cada origen (antes: sin reentrenar, solo se actualizaban
  los datos). Además cambian los candidatos, se admiten covariables y la selección pasa a ser por
  artículo con filtro frente al seasonal naive.
- Modified principles:
  - I. Propósito y alcance: pronóstico por bloques de 28 días, un bloque adelante (h = 1), con
    covariables.
  - II. Reglas de dominio: covariables solo conocidas de antemano o rezagadas; los ceros finales no
    se enmascaran en la serie de pronóstico.
  - III. Parámetros configurables: entran candidatos de modelo, espacios de búsqueda, rezagos,
    ventanas, transformación del objetivo y semillas.
  - VI. Método y congelamiento: folds de origen móvil con ventana expansiva sobre el año 4 y
    reentrenamiento en cada bloque; el mismo régimen en test (nb 05); candidatos declarados.
  - VII. Métricas y aceptación: RMSSE por artículo con filtro frente al seasonal naive
    (ŷ_t = y_{t−13}); WRMSSE agregado; WAPE, MAE y RMSE como diagnóstico.
- Modified sections: Stack tecnológico (SARIMAX, pmdarima o búsqueda AIC propia, XGBoost,
  LightGBM, CatBoost, Optuna); Flujo de trabajo (nb 04 selecciona y reentrena; nb 05 evalúa test).
- Gobierno: la réplica en el año 5 (nb 03, Etapa 13) se hizo antes de declarar los candidatos de
  pronóstico. Mitigación aprobada: los candidatos se declaran ahora, antes de cualquier
  backtesting, y la réplica no alimenta ninguna selección.
- Pendiente: la transformación del objetivo se elige en la Etapa 6d del nb 04 y se registra con
  una enmienda PATCH.
- Removed sections: ninguna.
- Impacto sobre artefactos congelados: ninguno (`targets` no cambia).
- Artefactos por actualizar: spec.md (este cambio), plan.md, research.md, data-model.md,
  contracts/ (artefactos de modelos), tasks.md y params.toml (`[forecast]`).

Historial 2.1.0
- Version change: 2.0.0 → 2.1.0 (MINOR, 2026-11-01)
- Motivo: guía ampliada. Durante la experimentación en train se definen tantos parámetros como
  exija el método adecuado, aunque no estén previstos en esta constitución ni en la config.
- Modified principles:
  - III. Parámetros configurables: se admiten parámetros de método definidos durante la
    experimentación, documentados donde se estiman (notebook canónico) con su motivo.
  - VI. Método y congelamiento: el notebook 03 por etapas es la versión canónica de la
    segmentación sobre train; lo que el flujo anterior exportaba o congelaba y el nb 03 no produce
    deja de tomarse en cuenta. Se congelan los cortes y reglas que surgieron en el nb 03 y la lista
    de artículos para modelado.
- Removed sections: ninguna.
- Impacto sobre artefactos congelados: `matrix_train`, `train_cuts` y `zero_thresholds` (v1, del
  flujo `run_train_matrix`) quedan sin uso; no se borran. `targets` se congela desde el nb 03.
- Artefactos por actualizar: tasks.md (estado de las etapas 3–5 y del modelado), código que lee
  los artefactos sin uso (`src/forecasting.py`, `src/stability.py`).

Historial 2.0.0
- Version change: 1.2.0 → 2.0.0 (MAJOR, 2026-10-29)
- Motivo del MAJOR: se redefinen reglas del Principio VI (NO NEGOCIABLE) de forma incompatible
  con la versión anterior: cambian las comparaciones de estabilidad y se elimina la elección del
  periodo de segmentación.
- Modified principles:
  - II. Reglas de dominio: los ceros internos en rachas estadísticamente improbables (criterio del
    Principio VI) se tratan como no observados (NaN) en el CV y en la serie de pronóstico, sin
    afirmar que sean quiebres. Los demás ceros siguen siendo ambiguos y se conservan.
  - III. Parámetros configurables: sale el periodo de segmentación; entran `LONG_RUN` (respaldo) y
    el α de la prueba de rachas.
  - V. Fuente de la verdad: sale el periodo de segmentación de lo que se serializa; entran los
    umbrales de racha por artículo.
  - VI. Método y congelamiento: (1) estabilidad con la base del año anterior (Yt−1 → Yt) y con Y1
    fija, mismos artículos, cortes elegidos en la base y aplicados al objetivo por participación;
    se elimina la comparación contra el promedio de años previos. (2) Medidas pareadas: kappa de
    Cohen plano, Stuart–Maxwell y cambio en puntos porcentuales; sin PSI sobre datos pareados.
    (3) Se elimina la elección del periodo de segmentación. (4) Umbral de rachas de ceros por
    artículo estimado solo con train. (5) Backtesting con 13 orígenes cada 28 días en el año 4.
    (6) La exploración puede orientar el método (qué medidas encajan), nunca los valores.
- Removed sections: ninguna (se eliminan reglas dentro de VI).
- Motivo: lo aprendido en la exploración (nb 02): el promedio de años previos sesga los cortes y
  el crecimiento del catálogo diluye la clase A; los datos de estabilidad son pareados; la clase
  ABC no depende de la época del año; los ceros estructurales inflan el CV (Z pasa de 702 a 146
  sin ellos); hay estacionalidad anual que 4 folds de fin de año no cubren.
- Impacto sobre artefactos congelados: ninguno (todavía no hay artefactos congelados).
- Artefactos por actualizar: spec.md (FR-015 a FR-019, FR-023/24, FR-030 y supuestos),
  research.md (folds, R5), data-model.md, contracts/, tasks.md y params.toml.

Historial 1.2.0
- Version change: 1.1.0 → 1.2.0 (MINOR, 2026-10-27)
- Modified principles:
  - III. Parámetros configurables: los cortes ABC y los umbrales XYZ son parámetros experimentales;
    su valor por defecto es un punto de partida y el valor definitivo es el que se elige en train.
  - VI. Método y congelamiento: la exploración de cinco años (que incluye el año 5) es solo una
    línea base descriptiva, también inter e intraanual; y el método para elegir cortes no se fija de
    antemano (determinista o algorítmico), pero siempre se ejecuta solo con train, con el criterio
    declarado antes y con el resultado congelado.
- Motivo: evitar que la experimentación con cortes y la exploración con el año 5 se lean como una
  decisión sobre el test, sin atar el proyecto a una metodología que depende de la naturaleza de
  los datos.
- Impacto sobre artefactos congelados: ninguno (todavía no hay artefactos congelados).
- Artefactos actualizados: spec.md (aclaraciones, FR-011, FR-013 y FR-029) y research.md (R5).

Historial 1.1.0
- Version change: 1.0.0 → 1.1.0 (MINOR, 2026-10-26)
- Modified principles:
  - VI. Método y congelamiento: el "Diagnóstico posterior (año 5)" pasa a ser la "Réplica en el
    año 5", que se ejecuta antes del pronóstico, una vez congelada la lista objetivo y declarados
    los candidatos de pronóstico y el nivel de selección. Sigue siendo solo diagnóstico.
- Modified sections: Flujo de trabajo y cierre (se añade el orden obligatorio de etapas: la
  exploración de cinco años va primero, y el pronóstico va después de la réplica en el año 5).
- Motivo: el orden anterior ponía la exploración y la mirada al año 5 después del pronóstico, lo
  que no corresponde al flujo de análisis del proyecto.
- Impacto sobre artefactos congelados: ninguno (todavía no hay artefactos congelados).
- Artefactos actualizados: spec.md, plan.md, research.md, data-model.md, contracts/,
  quickstart.md y tasks.md de 001-abc-xyz-share-forecast.

Historial previo
- Version change: (plantilla sin ratificar) → 1.0.0
- Modified principles: ninguno (primera adopción). Placeholders de plantilla reemplazados por:
  - [PRINCIPLE_1_NAME] → I. Propósito y alcance
  - [PRINCIPLE_2_NAME] → II. Reglas de dominio
  - [PRINCIPLE_3_NAME] → III. Estilo de código, parámetros y abstracciones
  - [PRINCIPLE_4_NAME] → IV. Manejo de errores y trazabilidad de rechazos
  - [PRINCIPLE_5_NAME] → V. Fuente de la verdad
  - (nuevo) VI. Método y congelamiento (NO NEGOCIABLE)
  - (nuevo) VII. Métricas y aceptación
- Added sections: Stack tecnológico (SECTION_2), Flujo de trabajo y cierre (SECTION_3), Governance
- Removed sections: ninguna
- Templates: no se modifican (leen la constitución en tiempo de ejecución).
- Follow-up TODOs:
  - TODO(STACK_VERSIONS): statsmodels no figura en pyproject.toml; añadirlo y fijar versiones
    exactas en uv.lock antes del bloque de modelado.
-->

# Rotación y abastecimiento con M5 Constitution

## Core Principles

### I. Propósito y alcance

- El proyecto identifica los artículos de mayor relevancia de la cadena: clase A por gasto
  estimado, con su clasificación XYZ por variabilidad de unidades.
- El proyecto pronostica las unidades de esos artículos por bloques de 28 días, un bloque
  adelante (h = 1), modeladas como ratio de unidades entre un bloque y el anterior,
  con modelos que pueden usar covariables.
- El resultado orienta decisiones de promoción y abastecimiento. NO es una orden de compra ni un
  pronóstico operativo completo; ningún entregable DEBE presentarse como tal.

**Razón:** acotar el alcance evita que el pronóstico de unidades se use como cantidad de
reposición, algo que esta etapa no valida.

### II. Reglas de dominio

- **Unidades vendidas:** DEBEN ser enteros no negativos. Cualquier otro valor se rechaza y se
  registra.
- **Precio de venta:** DEBE ser positivo y definido por tienda, artículo y semana (`wm_yr_wk`).
  Un precio ausente o no positivo se rechaza y se registra.
- **Uso:** unidades vendidas de un artículo agregado en el periodo de análisis. El periodo es un
  parámetro configurable (Principio III).
- **Gasto estimado por artículo:** suma sobre tiendas y días de (unidades × precio de venta
  semanal de esa tienda y semana). El precio de venta sustituye al costo unitario, que M5 no
  trae. Es un proxy; NO es costo real ni margen.
- Está PROHIBIDO calcular el gasto como precio medio × unidades totales. Cada combinación
  tienda-día-artículo DEBE aportar su propio precio.
- **Participación:** unidades del artículo agregado / unidades totales reales de la cadena, por
  periodo. Se usa en la segmentación; no es el objetivo del pronóstico.
- **Ratio:** r_t = unidades del artículo en el bloque t / unidades en el bloque t − 1, sin
  transformar. r > 1: el bloque vendió más que el anterior; r < 1: menos. El pronóstico en
  unidades es unidades_{t−1} × r̂_t y todo error se mide en unidades.
- **Bloque no válido:** un bloque con menos días observados que el `min_days` congelado del
  artículo (incluido el bloque sin ningún día observado) cae casi todo en un quiebre y su suma no
  representa la venta del periodo. También es no válido el bloque que pasó `min_days` pero vende
  menos del 10 % de la mediana de sus últimos 6 bloques válidos (en sus primeros 6 bloques, de la
  mediana de los válidos de esos 6; el bloque 1 no se evalúa): es un quiebre escondido por ventas
  sueltas de pocas unidades. Toma el valor del bloque anterior del mismo artículo: su ratio
  vale 1 ("no cambió") y el bloque siguiente se compara contra el último bloque válido. Si no hay
  bloque anterior (bloques no válidos al inicio del ciclo de vida), se eliminan y la serie
  empieza en el primer bloque válido. Se miden todos los bloques de la ventana de backtesting y
  del test.
- **Calendario:** las fechas DEBEN ser válidas y continuas. Eventos y SNAP son variables
  conocidas de antemano.
- **Prelanzamiento:** las semanas previas al lanzamiento de un artículo DEBEN excluirse del
  entrenamiento, la evaluación y los cálculos de uso, gasto y CV. NO se tratan como cero.
- **Ceros en periodo activo:** son ambiguos (quiebre de stock o baja demanda). NO DEBEN
  interpretarse como quiebre sin evidencia externa.
- **Rachas improbables de ceros:** una racha interna de ceros de largo igual o mayor que el
  umbral del artículo (Principio VI) se trata como **no observada** (NaN) en el CV de XYZ y en la
  serie de pronóstico. No se afirma su causa: se excluye porque es improbable como demanda de días
  independientes. El resto de los ceros se conserva como demanda observada.
- **Ciclo de vida en el pronóstico:** la serie de cada artículo va de su primera a su última
  venta; los ceros anteriores y posteriores no forman parte de la serie, igual que en la
  segmentación.
- **Covariables:** solo variables conocidas de antemano para el periodo pronosticado (calendario,
  eventos, SNAP) o valores rezagados. Ninguna usa información que no se conozca en el origen.

**Razón:** el gasto con precio por tienda-semana y la exclusión del prelanzamiento evitan sesgos
en ABC y en el CV que alterarían la clasificación. Las rachas improbables de ceros inflan el CV y
enseñan al modelo una demanda nula que no se observó.

### III. Estilo de código, parámetros y abstracciones

- **Idioma:** el código (funciones, métodos, variables, clases y firmas) DEBE estar en inglés.
- **Nombres:** funciones, métodos y variables en `snake_case`; clases en `CamelCase`;
  constantes en `UPPER_SNAKE_CASE`.
- **Documentación:** docstrings y comentarios en español, con una única estructura:
  descripción, parámetros con tipo y breve descripción, y retorno.
- Los comentarios DEBEN explicar el porqué de una decisión, no repetir lo que hace el código.
- **Parámetros configurables:** todos los cortes, umbrales y periodos (cortes ABC, umbrales XYZ,
  periodo de uso, candidatos de modelo, espacios de búsqueda, rezagos, ventanas, semillas, α de la prueba de rachas y `LONG_RUN` de respaldo) DEBEN vivir en
  un único archivo de configuración. Cada parámetro DEBE tener un valor por defecto documentado.
  Ningún valor queda fijo en el código.
- Los cortes ABC y los umbrales XYZ son parámetros experimentales: su valor por defecto es un punto
  de partida, no una decisión. El valor definitivo es el que se elige en train y se congela
  (Principio VI).
- Cada cambio de parámetro DEBE quedar registrado con fecha, valor anterior, valor nuevo y
  motivo.
- **Parámetros de método definidos en la experimentación:** se definen tantos parámetros como sean
  necesarios para llevar a cabo el método adecuado durante la experimentación en train, aunque no
  estén previstos en esta constitución ni en la config (por ejemplo, el mínimo de días observados
  por bloque o el mínimo de bloques válidos del CV). Cada uno se documenta, con su valor y su
  motivo, en el notebook canónico donde se estima, y se congela con el resultado.
- **Abstracciones:** una clase o función auxiliar se crea solo si cumple al menos una condición:
  (a) encapsula una regla de negocio que cambia de forma independiente, (b) se usa en al menos
  dos lugares del flujo, o (c) mantiene un estado que debe persistir entre llamadas. Si no, la
  lógica va en una sola función.
- NO se crean clases que solo envuelven funciones sin estado, interfaces sin al menos dos
  implementaciones reales, ni capas de configuración, fábricas o registros para algo que se usa
  una vez. La configuración es un archivo, no una capa de código.
- Antes de crear una abstracción, DEBE quedar escrito cuál de las condiciones (a), (b) o (c) la
  justifica.
- La estructura de carpetas del proyecto es fija: `data/` (con `raw/`, `interim/`, `processed/`, `external/`), `notebooks/`, `models/`, `reports/` y `src/`. No crear, mover ni renombrar carpetas de la raíz sin aprobación explícita.
- El código fuente de producción va en `src/`.
- Los notebooks de exploración y análisis van en `notebooks/`. No se generan en la raíz ni
  dentro de `src/`.
- No se escribe código de análisis en la raíz del proyecto.
- Las dependencias se gestionan con uv por grupos (`ds`, `deploy`, `back`, `algo`) en `pyproject.toml`. Agregar paquetes con `uv add --group <grupo>`.

**Razón:** parámetros centralizados y con historial hacen auditable cada decisión; la regla de
abstracciones mantiene el código mínimo y legible.

### IV. Manejo de errores y trazabilidad de rechazos

- Ningún dato rechazado desaparece en silencio. Cada rechazo DEBE registrarse con motivo, fila
  y fuente.
- Un dato fuera de dominio se aísla o detiene el proceso, según la regla de cada etapa. Nunca se
  continúa como si nada.
- Los fallos de ajuste de modelos DEBEN registrarse con el artículo y el candidato que fallaron.
- Un parámetro fuera de rango (por ejemplo, un corte ABC que no sea creciente o un umbral XYZ
  negativo) DEBE detener el proceso.

**Razón:** los rechazos silenciosos cambian las bases de cálculo de gasto, CV y participación
sin dejar rastro.

### V. Fuente de la verdad

- Los archivos originales de M5 son inmutables. Las transformaciones DEBEN producir archivos
  nuevos.
- El archivo de configuración y su historial de cambios son la referencia para los parámetros.
- Los cortes XYZ, la asignación ABC, los umbrales de racha por artículo, la lista de productos
  objetivo y su tratamiento se serializan una vez y se cargan en memoria. NO se recalculan.
- La lista de productos objetivo congelada es la única fuente para el modelado y la evaluación
  en test.
- El repositorio es la referencia para la especificación, la constitución y el plan.

**Razón:** una sola fuente serializada impide que la segmentación cambie entre etapas.

### VI. Método y congelamiento (NO NEGOCIABLE)

- **Partición por artículo:** el test son los últimos 13 bloques de 28 días del ciclo de vida de
  cada artículo (el año 5 si vende hasta d_1941); la validación, los 13 bloques anteriores; el
  resto es train. Solo se modelan artículos con ≥ 39 bloques de ciclo de vida (3 años); los
  demás se excluyen con su motivo. Si el test de un artículo cae en fechas que para otros son
  train o validación, se documenta.
- **Exploración descriptiva:** es la primera etapa del análisis, antes de la matriz sobre train.
  Puede usar los cinco años, incluido el año 5, para describir los productos: ABC por
  gasto, XYZ, distribuciones, proporción de ceros, media frente a mediana, y estabilidad anual y
  global solo como descripción, además de la exploración inter e intraanual. Es una línea base
  para saber qué esperar al explorar por años. NO selecciona productos, cortes, periodos ni
  partición, y NO informa ninguna decisión del modelo; probar cortes aquí es solo exploración.
  Sus diferencias con la matriz de train se reportan como hallazgo y no corrigen la matriz
  congelada. Lo que sí puede orientar es el **método** (qué medidas encajan con los datos); los
  valores se calculan de nuevo en train.
- **Estabilidad:** todo lo que influya en una decisión DEBE calcularse solo con train (años 1 a
  4). Dos bases: el año anterior (año t − 1 → año t) y el año 1 fijo (año 1 → año t). Se comparan
  los mismos artículos (activos todos los días de los años comparados), sin promediar años. Los
  cortes se eligen en la base y se aplican al objetivo por participación en el gasto. El año 5 NO
  entra en ninguna comparación de este análisis.
- **Medidas pareadas:** como base y objetivo son los mismos artículos, la estabilidad se mide con
  kappa de Cohen plano (acuerdo por artículo), Stuart–Maxwell (cambio en el % de artículos por
  clase) y el mayor cambio en puntos porcentuales. El PSI NO se usa sobre datos pareados.
- **Sin periodo de segmentación:** la clase se define por año; NO se elige un periodo dentro del
  año (trimestre o cuatrimestre).
- **Umbral de rachas de ceros:** se estima por artículo solo con train. Es el menor k tal que la
  probabilidad de que la racha de ceros más larga sea ≥ k, bajo días independientes con la tasa de
  ceros observada del artículo, sea ≤ α (α declarado; por defecto 0,01). La prueba de rachas de
  Wald–Wolfowitz por artículo se reporta como diagnóstico. Si el umbral no se puede estimar, se usa
  `LONG_RUN` (por defecto 7).
- **Backtesting:** 13 folds de origen móvil con ventana expansiva sobre los 13 bloques de
  validación de cada artículo, igual que el test, para que la selección cubra todo el ciclo
  anual. El fold k entrena con todo lo anterior a la validación más los bloques 1 a k − 1 de la
  validación y predice el bloque k (h = 1); nunca se entrena con el bloque que se predice. Se reentrena en cada fold. Una configuración de hiperparámetros se evalúa por su error
  promedio en los 13 folds, nunca por un fold aislado. El boosting no usa early stopping: la
  cantidad de árboles (`n_estimators`) es un hiperparámetro más de la búsqueda. Los órdenes SARIMAX se eligen una vez (AIC) con los bloques
  anteriores a la validación y sus coeficientes se reestiman en cada fold.
- **Candidatos de pronóstico:** seasonal naive (única línea base), SARIMAX y boosting global
  (XGBoost, LightGBM y CatBoost, con `item_id` y `cell` como variables). Se declaran antes del
  backtesting. La réplica en el año 5 (nb 03, Etapa 13) se hizo antes de esta declaración y no
  alimenta ninguna selección.
- **Réplica en el año 5:** una vez congelados la lista objetivo y su tratamiento, y declarados
  los candidatos de pronóstico y el nivel de selección del modelo, se replica sobre el año 5 la
  matriz ABC × XYZ y la estabilidad con los cortes, umbrales y agregación congelados. Es solo
  diagnóstico: NO modifica la lista, el tratamiento ni los candidatos declarados, y ninguna etapa
  de selección de modelos la lee. Cambiar un candidato declarado después de la réplica exige una
  nueva versión.
- Cualquier selección (órdenes SARIMAX, hiperparámetros de boosting, modelo por artículo, subconjunto, cortes ABC y XYZ, umbrales de racha, lista de productos
  objetivo) DEBE hacerse solo con backtesting o análisis interno dentro del train.
- El método para elegir cortes ABC y XYZ no se fija de antemano: puede ser determinista (valores
  fijos, sensibilidad, percentiles) o algorítmico (por ejemplo, una búsqueda de cortes sobre la
  distribución del CV), según lo que muestren los datos. Cualquiera que se use DEBE ejecutarse solo
  con train, con su criterio declarado antes de ejecutarlo, y sus valores finales se congelan.
- Los candidatos de búsqueda DEBEN declararse antes de ejecutar la experimentación. Sus valores
  concretos se definen en esa fase.
- **Versión canónica de la segmentación:** `notebooks/03_abc_xyz_train.ipynb`, por etapas
  auditables. Sus cortes, reglas y la lista de artículos para modelado son los que se congelan; lo
  que el flujo anterior exportaba o congelaba y el nb 03 no produce deja de tomarse en cuenta.
- Una vez seleccionados, todos los elementos quedan congelados. El test se evalúa una sola vez
  por modelo.
- Ningún parámetro se cambia después de ver resultados de test. Un cambio posterior a la
  congelación exige una nueva versión y repetir desde la matriz sobre train.
- En test se repite el régimen del backtesting: en cada uno de los 13 orígenes el modelo elegido
  por artículo se reentrena con todo lo observado hasta ese origen, con hiperparámetros y órdenes
  congelados. Ninguna selección se repite con datos de test.

**Razón:** cualquier uso del año 5 en una decisión filtra información del test e invalida la
evaluación.

### VII. Métricas y aceptación

- **WRMSSE:** criterio principal.
- **WAPE:** métrica secundaria.
- **RMSE:** diagnóstico interno, sin umbral.
- **MAPE:** fuera del criterio de aceptación.
- **RMSSE por artículo:** escala = error del seasonal naive dentro de la muestra (m = 13 bloques).
- **Línea base obligatoria:** seasonal naive (ŷ_t = y_{t−13}, bloques de 28 días).
- **Filtro por artículo:** un candidato compite solo si su error en las ventanas de validación es
  menor que el del seasonal naive en esas mismas ventanas; entre los que pasan gana el de menor
  error, y si ninguno pasa el artículo usa el seasonal naive.
- **Aceptación agregada:** el conjunto elegido DEBE tener un WRMSSE menor que el del seasonal naive.
- **MAE:** diagnóstico, sin umbral.

**Razón:** WRMSSE es la métrica de referencia de M5 y es robusta a series intermitentes, donde
MAPE no está definido o se distorsiona.

## Stack tecnológico

- Lenguaje: Python (`requires-python = "~=3.12.0"` en `pyproject.toml`).
- Librerías: pandas, NumPy, scikit-learn, statsmodels (SARIMAX), pmdarima (o búsqueda AIC propia
  con statsmodels si no es compatible), XGBoost, LightGBM, CatBoost, Optuna y herramientas de
  visualización habituales (matplotlib, seaborn, plotly).
- Las versiones DEBEN estar fijadas en el entorno y registradas en el repositorio
  (`pyproject.toml` y `uv.lock`).

## Flujo de trabajo y cierre

- Un solo bloque de trabajo activo a la vez. Cada bloque se cierra cuando cumple su criterio
  verificable.
- **Orden de etapas (obligatorio):** validación de datos → exploración descriptiva de cinco años
  → matriz ABC × XYZ sobre train → estabilidad y lista objetivo (congeladas) → declaración de
  candidatos de pronóstico → réplica en el año 5 → backtesting en train → evaluación única en
  test → informe final. El nb 04 cierra con la selección por artículo y el reentrenamiento de los
  ganadores con los años 1–4; la evaluación en test (13 orígenes del año 5) es del nb 05.
- El RMV se cierra cuando el ranking ABC × XYZ y el pronóstico de clase A cumplen los criterios
  de aceptación de la especificación.
- Todo plan (`/speckit-plan`) DEBE pasar una verificación de cumplimiento contra los Principios
  I a VII antes de iniciar la implementación.

## Governance

- Esta constitución prevalece sobre cualquier otra práctica del proyecto. La especificación, el
  plan y las tareas DEBEN ser coherentes con ella.
- **Enmiendas:** se proponen por escrito en el repositorio, indicando el principio afectado, el
  motivo y el impacto sobre artefactos congelados. Una enmienda que afecte elementos congelados
  (Principio VI) exige nueva versión y repetir desde la matriz sobre train.
- **Versionado semántico:** MAJOR para eliminación o redefinición incompatible de principios;
  MINOR para principios o secciones nuevas o guía ampliada; PATCH para aclaraciones y redacción.
- **Revisión de cumplimiento:** cada revisión de cambios DEBE verificar las reglas de dominio,
  la centralización de parámetros, el registro de rechazos, el aislamiento del test y la
  justificación escrita de cada abstracción.

**Version**: 6.2.0 | **Ratified**: 2026-10-24 | **Last Amended**: 2026-11-02
