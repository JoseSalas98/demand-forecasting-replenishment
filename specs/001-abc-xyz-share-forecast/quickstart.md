# Quickstart: validar la funcionalidad de punta a punta

## Requisitos previos

1. Descargar los archivos de M5 desde Kaggle (competencia `m5-forecasting-accuracy`) en `data/raw/`:
   `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv`.
2. Instalar el entorno:
   ```bash
   uv sync --all-groups
   uv add --group ds statsmodels pyarrow   # una vez, si faltan
   uv add --group dev pytest               # una vez, si falta
   ```
3. Verificar que `src/config/params.toml` coincide con [contracts/params.md](contracts/params.md).

## Pruebas unitarias

```bash
uv run pytest src/tests -q
```

Casos clave:
- gasto ≠ precio medio × unidades cuando los precios varían (US1-2)
- el artículo que cruza el corte del 80 % es clase A
- un CV de exactamente 25 % es Z
- los días de prelanzamiento se enmascaran y no se ponen en cero
- cada regla de estabilidad con entradas construidas a mano
- WRMSSE sobre una serie de juguete frente a un cálculo manual
- el ledger rechaza una segunda evaluación

## Ejecución por etapas (correr los notebooks en orden)

| # | Notebook | Resultado esperado | Verificaciones |
|---|---|---|---|
| 1 | `01_data_validation` | Arreglos intermedios, `rejections.parquet`, `launch.parquet` | Calendario continuo. Cada rechazo tiene motivo, fuente y fila (SC-002). |
| 2 | `02_abc_xyz_train` | `matrix_train.parquet` congelado | El 100 % de los artículos activos tiene clase o motivo de exclusión (SC-001). Revisión manual de ≥ 10 artículos (SC-003). Solo se cargan días ≤ 1577. |
| 3 | `03_stability_target_list` | `targets.parquet` con tratamiento, congelado | Declaraciones registradas antes de la ejecución. Comparaciones Y2/Y1, Y3/prom(Y1,Y2), Y4/prom(Y1–Y3). Todo objetivo es A, estable y con tratamiento. Ningún día de Y5 se toca (FR-021). |
| 4 | `04_backtesting` | `frozen_models.json`, `backtest_scores.parquet` | Folds dentro de d_1466–d_1577. Fallos registrados. |
| 5 | `05_test_evaluation` | `evaluation.parquet`, entrada en el ledger | 13 orígenes × 28 días. Parámetros idénticos al archivo congelado. WRMSSE del elegido < seasonal naive (SC-006). Repetir con el mismo `model_id` detiene el proceso (SC-007). |
| 6 | `06_eda_five_years` | `reports/eda_five_years/` | Los hashes congelados no cambian tras la ejecución (US4-2). |
| 7 | `07_year5_diagnostic` | `year5_diagnostic.parquet` | Se detiene si la etapa 5 no se ejecutó. El hash de targets no cambia (US5). |
| 8 | `08_final_report` | `reports/final_report.*` | Clase, participación en gasto, tratamiento y pronóstico de cada objetivo (SC-008). Advertencia sobre backtesting y sesgo de selección. WAPE/RMSE. Estabilidad en Y5 para el 100 % de los objetivos (SC-009). |

## Verificación de reproducibilidad (SC-004)

Repetir las etapas 1–3 en una carpeta de salida temporal con la misma config. El SHA-256 de
`matrix_train.parquet` y `targets.parquet` debe coincidir con el manifiesto.

## Verificaciones negativas

- Fijar `abc_cuts = [0.95, 0.80]`: el proceso se detiene antes de cualquier cálculo.
- Editar un valor sin añadir una fila al historial: el proceso se detiene.
- Quitar `AZ` de `treatment_map` cuando existe un objetivo AZ: el proceso se detiene.
