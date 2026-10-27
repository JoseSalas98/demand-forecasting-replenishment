# Research: segmentación ABC × XYZ y pronóstico de participación (M5)

Cada entrada da la decisión, la razón y las alternativas consideradas. La fuente de los datos de
M5 es Makridakis, Spiliotis & Assimakopoulos (2022), "The M5 accuracy competition: Results,
findings and conclusions", IJF 38(4).

## R1. Fechas exactas de train/test

**Decisión**: usar los días con ventas en `sales_train_evaluation.csv` (d_1 = 2011-01-29 a
d_1941 = 2016-05-22). Los años son bloques de 364 días (13 × 28 días, así cada año tiene los
mismos días de la semana) contados hacia atrás desde el final.

| Bloque | Días | Fechas |
|---|---|---|
| Solo historia | d_1–d_121 | 2011-01-29 → 2011-05-29 |
| Y1 | d_122–d_485 | 2011-05-30 → 2012-05-27 |
| Y2 | d_486–d_849 | 2012-05-28 → 2013-05-26 |
| Y3 | d_850–d_1213 | 2013-05-27 → 2014-05-25 |
| Y4 | d_1214–d_1577 | 2014-05-26 → 2015-05-24 |
| Y5 = test | d_1578–d_1941 | 2015-05-25 → 2016-05-22 |

El origen de test *k* (1 a 13) pronostica de d_(1578 + 28(k−1)) a +27. El origen 13 es
exactamente la ventana de validación de M5 (d_1914–d_1941). Los pesos de WRMSSE usan
d_1550–d_1577.

**Razón**: el paper confirma 1.969 días de calendario (hasta 2016-06-19). Las ventas de
d_1942–d_1969 no están en el archivo de Kaggle. Los bloques de 364 días se alinean con los
bloques de 28 días del CV y con las ventanas de test. Los 121 días sobrantes quedan antes de Y1.
Se excluyen de la segmentación, para que todos los años tengan la misma longitud, y se conservan
como historia de ajuste; son datos de train, así que no filtran información.

**Alternativas**: años calendario (longitud desigual, con distinta mezcla de días de la semana);
incluir d_1942–d_1969 (requiere un archivo extra que no forma parte de la descarga estándar);
descartar d_1–d_121 por completo (desperdicia historia).

## R2. Archivos de datos y `wm_yr_wk`

**Decisión**: las entradas son `calendar.csv`, `sell_prices.csv` (`store_id, item_id, wm_yr_wk,
sell_price`) y `sales_train_evaluation.csv` (una fila por tienda-artículo, con columnas d_1 a
d_1941). `wm_yr_wk` es el identificador semanal de Walmart (las semanas empiezan en sábado). Los
precios diarios salen del cruce día → `wm_yr_wk` (calendario) → precio. La validación verifica
que cada `d` corresponda a exactamente una fecha y un `wm_yr_wk`, que las fechas sean continuas
y que las semanas tengan ≤ 7 días.

**Razón**: en M5 los precios solo existen desde la semana en que un artículo está a la venta en
una tienda; por eso esa semana marca el lanzamiento (aclaración de la spec).

**Alternativas**: `sales_train_validation.csv` (solo llega a d_1913, así se pierde el origen de
test 13).

## R3. Lanzamiento, enmascarado y rechazos

**Decisión**: el lanzamiento en una tienda es el primer `wm_yr_wk` con precio. Los días previos
se enmascaran (NaN) para esa tienda. El artículo agregado empieza en su lanzamiento más temprano
entre tiendas, y las tiendas que aún no lanzan no aportan nada. Casos:

| Caso | Acción | Código de motivo |
|---|---|---|
| unidades no enteras o < 0 | celda enmascarada, registrada como rechazo | `invalid_units` |
| unidades > 0 antes del primer precio | celda enmascarada, registrada como rechazo | `sales_before_first_price` |
| precio faltante/≤ 0 tras el lanzamiento, unidades > 0 | tienda-artículo-semana enmascarada, registrada como rechazo, gasto sin imputar | `invalid_price_with_sales` |
| precio faltante tras el lanzamiento, unidades = 0 | tienda-artículo-semana enmascarada, registrada (no es rechazo) | `price_gap_no_sales` |
| prelanzamiento sin precio | enmascarado, sin registro (FR-004) | — |

Las celdas enmascaradas quedan fuera del uso, el gasto, el CV, los numeradores de participación
y el total de la cadena.

## R4. Periodo de uso y CV

**Decisión**: el periodo de uso por defecto es todo el conjunto de cálculo (train global = Y1–Y4,
o el año / segmento). El CV usa `cv_aggregation = "block28"`: bloques completos de 28 días dentro
del conjunto de cálculo (13 por año, 52 para train global). Se descartan los bloques incompletos
por lanzamiento o enmascarado. CV = desviación estándar poblacional (ddof = 0) / media × 100. Un
artículo queda sin clase XYZ (motivo `cv_undefined`) si su media es 0 o tiene menos de
`cv_min_blocks` = 2 bloques completos.

**Alternativas**: `daily`, `weekly` y `segment_period` quedan como opciones de config para
comparar solo en train (FR-012).

## R5. Valores por defecto de ABC y XYZ

**Decisión**: cortes ABC en 80 / 95 % del gasto acumulado. Un artículo pertenece a una clase
según su participación acumulada *antes* de sumar su propio gasto; los empates se resuelven por
`item_id`. XYZ usa X < 10 %, 10 % ≤ Y < 25 %, Z ≥ 25 %. Son valores habituales en la práctica.
Son un punto de partida experimental, no valores fijos (FR-013). El método para elegir los
definitivos no se fija de antemano; depende de la naturaleza de los datos y puede ser determinista
(sensibilidad ±5 pp, percentiles) o algorítmico (por ejemplo, `find_cv_cuts`, que busca cortes del
CV con gasto medio descendente y un mínimo de artículos por segmento). Reglas comunes: se ejecuta
solo con train, el criterio se declara antes de ejecutarlo y los valores elegidos se congelan.
Probar cortes en la exploración de cinco años es solo exploración y no decide nada.

## R6. Estabilidad, referencia y periodo de segmentación

**Decisión**:
- **Clase de referencia** (por artículo): promediar el gasto anual del artículo sobre los años
  *activos* previos y volver a aplicar ABC sobre esos promedios para todos los artículos.
  Promediar su CV anual y aplicar los umbrales XYZ. Un año cuenta como activo solo si el
  artículo está activo todo el año (`min_active_days_year = 364`). Si un artículo no tiene
  referencia para una comparación, esa comparación falla.
- **Acuerdo**: % de artículos con la misma clase en el año frente a la referencia, y en el año
  frente a train global. Umbral por defecto `agreement_min = 80 %`.
- **PSI** sobre las proporciones de clase (A/B/C y X/Y/Z), con ε = 1e-4 para clases vacías. La
  distribución de referencia es el promedio simple de las proporciones de los años previos.
  Umbrales: < 0,10 estable, 0,10–0,25 moderado, ≥ 0,25 cambio (la convención usual, Siddiqi
  2006).
- **Regla**: `stability_rule = "abc_all"`, con las alternativas de FR-018. Criterio de selección
  de la regla por defecto: `"fixed"` (usar la regla configurada).
- **Candidatos de periodo de segmentación**: `quarter` (4 × 13 semanas) y `four_month`
  (17 / 17 / 18 semanas), ambos dentro de cada año de 364 días. Criterio: el mayor acuerdo ABC
  medio entre las matrices por periodo y la matriz de su año sobre Y1–Y4. Los empates van al
  primer candidato. El periodo elegido se usa en el reporte de estabilidad y en la opción
  `cv_aggregation = "segment_period"`. La pertenencia a la lista objetivo sale de las
  comparaciones anuales.

## R7. Tratamiento

**Decisión**: `treatment_map = {AX = "regular_replenishment", AY = "periodic_review_replenishment",
AZ = "promotion_priority"}`. Si la celda de un artículo objetivo (por ejemplo `A-` cuando XYZ no
está definido) no tiene tratamiento, el proceso se detiene.

## R8. Pronosticadores y backtesting

**Decisión**:
- **Serie objetivo**: participación diaria = unidades del artículo / unidades totales de la
  cadena, desde el lanzamiento del artículo. Los días con total de cadena 0 se excluyen y se
  registran (`zero_chain_total`).
- **Candidatos**:

  | Candidato | Grilla |
  |---|---|
  | Seasonal naive | m = 7, como en M5 |
  | Media móvil | ventanas {7, 14, 28, 56} |
  | ARIMA | p, q ∈ {0, 1, 2}, d ∈ {0, 1} |
  | SARIMA | grilla ARIMA × (P, D, Q) ∈ {0, 1}³, s = 7 |

  Seasonal naive, ARIMA y SARIMA usan `SARIMAX` de statsmodels sin variables exógenas (con
  constante cuando d = D = 0). La media móvil no es un modelo `SARIMAX`: se calcula de forma
  cerrada y su "parámetro congelado" es la ventana. La serie se normaliza por su media positiva
  de train (guardada como `scale` en el modelo congelado) para estabilizar el optimizador, y la
  participación pronosticada se recorta a ≥ 0.
- **Backtest**: `backtest_folds = 4` folds de 28 días sin solapamiento que terminan en d_1577
  (d_1466–d_1577). Cada candidato se ajusta una vez con los datos previos al primer fold. Los
  folds siguientes solo añaden observaciones (`smooth` con parámetros fijos), igual que el
  protocolo de test. El puntaje es el RMSSE medio por artículo, agregado al `selection_level`
  que declara el analista (etiqueta por defecto `"item"`).
- **Congelamiento**: el candidato elegido se reajusta con todo train (hasta d_1577). Su orden y
  su vector de parámetros se guardan en JSON.
- **Test**: en cada origen el modelo congelado se reconstruye con los datos observados hasta el
  origen mediante `SARIMAX(...).smooth(frozen_params)` y pronostica 28 pasos. Nada se reestima.
- **Fallos**: cualquier excepción o falta de convergencia se registra como (artículo, candidato,
  mensaje) y ese candidato se omite para ese artículo. El seasonal naive nunca falla, así que
  todo artículo siempre tiene respaldo.
- **Costo**: `subset_size` (por defecto `all`) puede limitar los objetivos a los N primeros por
  gasto en train. Debe declararse antes del backtest.

**Alternativas**: `auto_arima` (pmdarima: una dependencia extra, con una búsqueda stepwise
difícil de declarar de antemano); reajustes móviles en test (prohibidos por el Principio VI).

## R9. Conversión a unidades y métricas

**Decisión**:
- units_hat = share_hat × total real de la cadena del día. Sin redondeo para las métricas. El
  reporte muestra enteros redondeados hacia arriba en la mitad (half-up).
- **RMSSE** por artículo se calcula sobre unidades. La escala es la media de las diferencias
  naive a un paso al cuadrado en train, desde la primera venta no nula del artículo hasta
  d_1577.
- **WRMSSE**: un solo nivel (los artículos objetivo), ponderado por la participación en gasto
  sobre d_1550–d_1577 entre los artículos evaluados. Los artículos con escala cero se excluyen y
  se registran.
- **Puntaje**: la media de los 13 valores de WRMSSE por origen. WAPE (Σ|e| / Σy) y RMSE son
  diagnósticos. No se usa MAPE.

**Razón**: sigue las definiciones del paper (pesos de los últimos 28 días de train, escala desde
la primera demanda no nula), restringidas al conjunto objetivo.

## R10. Serialización, congelamiento y regla de un solo test

**Decisión**:
- Las tablas van en Parquet (pyarrow). Snapshot de config, manifiesto y parámetros de modelos van
  en JSON.
- `freeze()` escribe cada artefacto una vez y registra su SHA-256, el hash de la config y una
  marca de tiempo UTC en `data/processed/manifest.json`. `load_frozen()` verifica el hash y se
  detiene si cambió.
- La evaluación en test exige que las marcas de congelamiento del manifiesto sean anteriores a la
  ejecución.
- El backtest exige que las declaraciones de `[forecast]` sean anteriores al `frozen_at` de
  `matrix_y5` y que sus valores coincidan con la config actual (R13).
- `TestLedger` (`data/processed/test_ledger.json`) rechaza una segunda evaluación del mismo
  `model_id` (SC-007).

**Alternativas**: pickle (opaco y frágil entre versiones); una base de datos (excesiva para
archivos de este tamaño).

## R11. Configuración e historial de cambios

**Decisión**:
- Los parámetros viven en `src/config/params.toml`, leído con `tomllib` de la librería estándar.
- El historial vive en `src/config/params_history.csv` (`date, parameter, old_value, new_value,
  reason`).
- Al iniciar, `config.py` compara el TOML con el último snapshot del manifiesto. Se detiene si
  algún cambio no tiene su fila en el historial o si un valor está fuera de rango. Las reglas de
  rango están en [contracts/params.md](contracts/params.md).

**Alternativas**: YAML (requiere un parser como dependencia en tiempo de ejecución).

## R12. Versiones

**Decisión**: Python 3.12. Añadir statsmodels y pyarrow a `ds` y pytest a `dev` con
`uv add --group`. Las versiones exactas son las que resuelva `uv.lock` al añadirlas, y se
registran en el reporte final.

## R13. Orden de etapas y réplica en el año 5

**Decisión**: las etapas se ejecutan en este orden (constitución v1.1.0, spec "Orden de
ejecución"):

| # | Etapa | Notebook | Datos que lee |
|---|---|---|---|
| 1 | Validación de datos | `01_data_validation` | d_1–d_1941 (solo validación) |
| 2 | EDA de cinco años | `02_eda_five_years` | d_122–d_1941 (Y1–Y5), solo descriptivo |
| 3 | Matriz ABC × XYZ de train | `03_abc_xyz_train` | ≤ d_1577 |
| 4 | Estabilidad y lista objetivo | `04_stability_target_list` | ≤ d_1577 |
| 5 | Declaración de `[forecast]` y réplica en Y5 | `05_year5_replication` | d_1578–d_1941 más lo congelado |
| 6 | Backtesting | `06_backtesting` | ≤ d_1577 |
| 7 | Evaluación en test | `07_test_evaluation` | d_1–d_1941 |
| 8 | Informe final | `08_final_report` | artefactos |

La réplica (etapa 5) usa los cortes, umbrales, `cv_aggregation` y `treatment_map` congelados en
la etapa 4 y calcula:
- la matriz ABC × XYZ de Y5 (d_1578–d_1941) para todos los artículos (`matrix_y5`);
- el acuerdo y el PSI de Y5 frente a la referencia (promedio de los años activos Y1–Y4, con la
  misma regla de R6) y frente a `matrix_train` (`stability_y5`);
- por producto objetivo, su clase y celda en Y5, su referencia y si coincide con lo congelado
  (`year5_replication`).

Resguardos:
- La etapa 5 exige que `targets` y `segmentation_params` estén congelados, y registra las
  declaraciones de todas las claves de `[forecast]` antes de leer cualquier día de Y5.
- El backtest (etapa 6) se detiene si alguna declaración de `[forecast]` es posterior al
  `frozen_at` de `matrix_y5` o si su valor difiere de la config actual. Cambiarla exige una nueva
  `version` y repetir desde la etapa 3.
- Las etapas 3, 4 y 6 usan `assert_train_only` y no leen `reports/eda_five_years/` ni los
  artefactos de la réplica.

**Razón**: la exploración global da contexto antes de trabajar sobre train. Replicar en Y5 antes
de pronosticar muestra si la segmentación congelada se sostiene en el periodo que se va a
pronosticar. Declarar antes los candidatos evita que lo visto en Y5 influya en la elección del
modelo.

**Alternativas**: diagnóstico del año 5 después del test (orden original de la v1.0.0; deja la
información sobre la segmentación en Y5 para después del pronóstico, lo que no corresponde al
flujo de análisis); réplica antes del pronóstico sin declarar antes los candidatos (más flexible,
pero lo visto en Y5 podría sesgar la elección del modelo).
