# Plan de implementación: segmentación ABC × XYZ y pronóstico de participación (M5)

**Rama**: `001-abc-xyz-share-forecast` (rama git: `spec`) | **Fecha**: 2026-10-24 | **Spec**: [spec.md](spec.md)

## Resumen

Clasificar los 3.049 artículos de M5 (sumados sobre 10 tiendas) por gasto estimado (ABC) y
variabilidad de unidades (XYZ), solo con train. Congelar la lista objetivo: artículos de clase A
que se mantienen estables año a año, cada uno con su tratamiento. Pronosticar su participación
diaria en unidades a 28 días, elegir el modelo con backtesting en train y evaluar una sola vez en
el año 5 (13 orígenes) con WRMSSE frente al seasonal naive.
Detalle: [research.md](research.md) · [data-model.md](data-model.md) · [contracts/](contracts/) ·
[quickstart.md](quickstart.md).

## Contexto técnico

| Elemento | Decisión |
|---|---|
| Lenguaje | Python 3.12 |
| Dependencias | pandas, NumPy, SciPy, statsmodels (`SARIMAX`), pyarrow; gráficos con matplotlib/seaborn/plotly. Config leída con `tomllib` de la librería estándar. Añadir statsmodels y pyarrow al grupo `ds` y pytest al grupo `dev` con `uv add --group`. |
| Almacenamiento | Archivos locales: `data/raw` (CSV de M5, solo lectura), `data/interim`, `data/processed` (artefactos congelados y manifiesto), `models/`, `reports/` |
| Pruebas | pytest sobre datos sintéticos mínimos en `src/tests/` |
| Plataforma | Ejecuciones batch locales en Linux/WSL2 |
| Tipo de proyecto | Pipeline de análisis de datos: paquete `src/` más un notebook por etapa |
| Escala | 30.490 series tienda-artículo × 1.941 días. Se guardan como arreglos NumPy anchos (~0,4 GB), nunca en formato largo de 59 M filas. |
| Partición | Train d_1–d_1577. Años de segmentación = bloques de 364 días Y1–Y4 (d_122–d_1577). d_1–d_121 es solo historia para los modelos. Test = Y5, d_1578–d_1941 (2015-05-25 → 2016-05-22), 13 × 28 días. Ver research R1. |
| Rendimiento | Cada etapa corre en una sesión local. El costo de SARIMA queda acotado por `subset_size` y la grilla de candidatos. |

No queda ningún NEEDS CLARIFICATION. Los valores que la spec deja a la experimentación son
parámetros de config con valores por defecto documentados ([contracts/params.md](contracts/params.md)).

## Verificación de la constitución

| Principio | Cómo cumple el diseño | Antes | Después |
|---|---|---|---|
| I Alcance | Las salidas son la matriz, la lista objetivo y el pronóstico de participación. El reporte aclara que no es una orden de compra. | ✅ | ✅ |
| II Dominio | Unidades enteras ≥ 0. Precio > 0 por tienda-artículo-`wm_yr_wk`. Gasto = Σ unidades × precio diario, nunca precio medio × unidades. El prelanzamiento por tienda se enmascara (NaN, no 0). Los ceros con actividad se conservan. La participación usa el total real de la cadena. | ✅ | ✅ |
| III Código/parámetros | Código en inglés, docstrings en español. Un único `src/config/params.toml` más `params_history.csv`. Código en `src/`, notebooks en `notebooks/`, sin carpetas nuevas en la raíz. | ✅ | ✅ |
| IV Errores | `rejections.parquet` y `fit_failures.parquet`. Calendario inválido, parámetro fuera de rango, lista vacía o celda sin tratamiento detienen el proceso. | ✅ | ✅ |
| V Fuente de la verdad | Los artefactos congelados se escriben una vez, con hash SHA-256 en `manifest.json`. Las etapas posteriores los cargan y verifican, nunca los recalculan. | ✅ | ✅ |
| VI Método/congelamiento | Todas las decisiones usan solo train. Candidatos y criterios se declaran en config antes de ejecutar. `test_ledger.json` permite una sola evaluación en test por modelo. Test usa parámetros congelados (`SARIMAX.smooth`), sin reajuste. Ninguna etapa de decisión lee la salida del EDA de 5 años. | ✅ | ✅ |
| VII Métricas | WRMSSE (Makridakis et al., 2022) es la principal, WAPE secundaria, RMSE de diagnóstico, sin MAPE. Se acepta solo si WRMSSE < seasonal naive. | ✅ | ✅ |

**Abstracciones (Principio III):**

| Abstracción | Justificación |
|---|---|
| dict `STABILITY_RULES` (4 reglas) | (a) |
| dict `FORECASTERS` (snaive, moving_average, arima, sarima) | (a) |
| `classify_abc` / `classify_xyz` | (b): se usan en las matrices global, anuales y por periodo, en la referencia y en el EDA |
| métricas (`wrmsse`, `wape`, `rmse`) | (b): se usan en backtest y en test |
| `freeze` / `load_frozen` + manifiesto | (b), (c) |
| `TestLedger` | (c) |

Puerta: **APROBADA** antes de la Fase 0 y después de la Fase 1.

## Estructura del proyecto

```text
specs/001-abc-xyz-share-forecast/
├── plan.md  research.md  data-model.md  quickstart.md
└── contracts/  params.md  artifacts.md

src/
├── config/params.toml, params_history.csv
├── config.py         # carga, validación de rangos, verificación del historial
├── data.py           # carga M5, validación, rechazos, lanzamiento, arreglos anchos, partición
├── segmentation.py   # uso, gasto, ABC, CV, XYZ, matriz
├── stability.py      # matrices anuales/por periodo, referencia, acuerdo, PSI, reglas, lista objetivo, tratamiento
├── forecasting.py    # series de participación, pronosticadores, backtest, evaluación en test
├── metrics.py        # RMSSE/WRMSSE, WAPE, RMSE
├── freeze.py         # serialización, manifiesto, TestLedger
└── tests/            # pytest, fixtures sintéticos

notebooks/  01_data_validation · 02_abc_xyz_train · 03_stability_target_list · 04_backtesting ·
            05_test_evaluation · 06_eda_five_years · 07_year5_diagnostic · 08_final_report
```

**Decisión de estructura:** un solo proyecto con la estructura fija de la constitución. Los
notebooks solo orquestan y muestran; toda la lógica de cálculo vive en `src/`. Pendiente para
tasks: `[tool.ruff]` hoy apunta a `main/**` y debe apuntar a `src/**`.

## Seguimiento de complejidad

Sin violaciones.
