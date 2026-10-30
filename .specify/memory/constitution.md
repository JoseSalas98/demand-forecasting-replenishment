<!--
Sync Impact Report
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
- El proyecto pronostica la participación en unidades de esos artículos a 28 días.
- El resultado orienta decisiones de promoción y abastecimiento. NO es una orden de compra ni un
  pronóstico operativo completo; ningún entregable DEBE presentarse como tal.

**Razón:** acotar el alcance evita que el pronóstico de participación se use como cantidad de
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
  periodo.
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
  periodo de uso, candidatos, α de la prueba de rachas y `LONG_RUN` de respaldo) DEBEN vivir en
  un único archivo de configuración. Cada parámetro DEBE tener un valor por defecto documentado.
  Ningún valor queda fijo en el código.
- Los cortes ABC y los umbrales XYZ son parámetros experimentales: su valor por defecto es un punto
  de partida, no una decisión. El valor definitivo es el que se elige en train y se congela
  (Principio VI).
- Cada cambio de parámetro DEBE quedar registrado con fecha, valor anterior, valor nuevo y
  motivo.
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

- **Partición:** un único train (años 1 a 4) y un único test (año 5, 1 año), evaluado en 13
  orígenes cada 28 días, sin solapamiento.
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
- **Backtesting:** 13 orígenes cada 28 días repartidos en el año 4, igual que el test, para que la
  selección de modelos cubra todo el ciclo anual.
- **Réplica en el año 5:** una vez congelados la lista objetivo y su tratamiento, y declarados
  los candidatos de pronóstico y el nivel de selección del modelo, se replica sobre el año 5 la
  matriz ABC × XYZ y la estabilidad con los cortes, umbrales y agregación congelados. Es solo
  diagnóstico: NO modifica la lista, el tratamiento ni los candidatos declarados, y ninguna etapa
  de selección de modelos la lee. Cambiar un candidato declarado después de la réplica exige una
  nueva versión.
- Cualquier selección (línea base, órdenes ARIMA/SARIMA, subconjunto, cortes ABC y XYZ, umbrales
  de racha, lista de productos objetivo) DEBE hacerse solo con backtesting o análisis
  interno dentro del train.
- El método para elegir cortes ABC y XYZ no se fija de antemano: puede ser determinista (valores
  fijos, sensibilidad, percentiles) o algorítmico (por ejemplo, una búsqueda de cortes sobre la
  distribución del CV), según lo que muestren los datos. Cualquiera que se use DEBE ejecutarse solo
  con train, con su criterio declarado antes de ejecutarlo, y sus valores finales se congelan.
- Los candidatos de búsqueda DEBEN declararse antes de ejecutar la experimentación. Sus valores
  concretos se definen en esa fase.
- Una vez seleccionados, todos los elementos quedan congelados. El test se evalúa una sola vez
  por modelo.
- Ningún parámetro se cambia después de ver resultados de test. Un cambio posterior a la
  congelación exige una nueva versión y repetir desde la matriz sobre train.
- Los modelos NO se reentrenan en el test. Solo se actualizan los datos observados con
  parámetros fijos.

**Razón:** cualquier uso del año 5 en una decisión filtra información del test e invalida la
evaluación.

### VII. Métricas y aceptación

- **WRMSSE:** criterio principal.
- **WAPE:** métrica secundaria.
- **RMSE:** diagnóstico interno, sin umbral.
- **MAPE:** fuera del criterio de aceptación.
- **Línea base obligatoria:** seasonal naive. Un modelo que no le gane en WRMSSE NO se acepta.

**Razón:** WRMSSE es la métrica de referencia de M5 y es robusta a series intermitentes, donde
MAPE no está definido o se distorsiona.

## Stack tecnológico

- Lenguaje: Python (`requires-python = "~=3.12.0"` en `pyproject.toml`).
- Librerías: pandas, NumPy, scikit-learn, statsmodels (ARIMA y SARIMA) y herramientas de
  visualización habituales (matplotlib, seaborn, plotly).
- Las versiones DEBEN estar fijadas en el entorno y registradas en el repositorio
  (`pyproject.toml` y `uv.lock`).
- TODO(STACK_VERSIONS): statsmodels aún no figura en las dependencias; las versiones exactas
  quedan por verificar y fijar.

## Flujo de trabajo y cierre

- Un solo bloque de trabajo activo a la vez. Cada bloque se cierra cuando cumple su criterio
  verificable.
- **Orden de etapas (obligatorio):** validación de datos → exploración descriptiva de cinco años
  → matriz ABC × XYZ sobre train → estabilidad y lista objetivo (congeladas) → declaración de
  candidatos de pronóstico → réplica en el año 5 → backtesting en train → evaluación única en
  test → informe final.
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

**Version**: 2.0.0 | **Ratified**: 2026-10-24 | **Last Amended**: 2026-10-29
