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
- gasto ≠ precio medio × unidades cuando los precios varían (US2-2)
- el artículo que cruza el corte del 80 % es clase A
- un CV de exactamente 25 % es Z
- los días de prelanzamiento se enmascaran y no se ponen en cero
- cada regla de estabilidad con entradas construidas a mano
- WRMSSE sobre una serie de juguete frente a un cálculo manual
- el ledger rechaza una segunda evaluación
- la réplica en Y5 se detiene si `[forecast]` no está declarado, y el backtest se detiene si
  `[forecast]` cambió después de la réplica

## Ejecución por etapas (correr los notebooks en orden)

El orden es obligatorio (constitución v1.1.0, research R13).

| # | Notebook | Resultado esperado | Verificaciones |
|---|---|---|---|
| 1 | `01_data_validation` | Arreglos intermedios, `rejections.parquet`, `launch.parquet` | Calendario continuo. Cada rechazo tiene motivo, fuente y fila (SC-002). |
| 2 | `02_eda_five_years` | `reports/eda_five_years/` | Solo descriptivo (Y1–Y5). Ninguna etapa posterior lee esta carpeta (US1-2). |
| 3 | `03_abc_xyz_train` | `matrix_train.parquet` congelado | El 100 % de los artículos activos tiene clase o motivo de exclusión (SC-001). Revisión manual de ≥ 10 artículos (SC-003). Solo se cargan días ≤ 1577. Diferencias con el EDA reportadas como hallazgo (US1-3). |
| 4 | `04_stability_target_list` | `targets.parquet` y `segmentation_params.json` congelados | Declaraciones registradas antes de la ejecución. Comparaciones Y2/Y1, Y3/prom(Y1,Y2), Y4/prom(Y1–Y3). Todo objetivo es A, estable y con tratamiento. Ningún día de Y5 se toca (FR-021). |
| 5 | `05_year5_replication` | `matrix_y5`, `stability_y5`, `year5_replication` congelados | `[forecast]` declarado antes de leer Y5. Parámetros congelados aplicados sin cambios. Réplica del 100 % de los objetivos. El hash de `targets` no cambia (US4). |
| 6 | `06_backtesting` | `frozen_models.json`, `backtest_scores.parquet` | Declaraciones de `[forecast]` anteriores a `matrix_y5` e iguales a la config. Folds dentro de d_1466–d_1577. Fallos registrados. |
| 7 | `07_test_evaluation` | `evaluation.parquet`, entrada en el ledger | 13 orígenes × 28 días. Parámetros idénticos al archivo congelado. WRMSSE del elegido < seasonal naive (SC-006). Repetir con el mismo `model_id` detiene el proceso (SC-007). |
| 8 | `08_final_report` | `reports/final_report.*` | Clase, participación en gasto, tratamiento y pronóstico de cada objetivo (SC-008). Advertencia sobre backtesting y sesgo de selección. WAPE/RMSE. Réplica en Y5 del 100 % de los objetivos (SC-009). |

## Verificación de reproducibilidad (SC-004)

Repetir las etapas 1, 3 y 4 en una carpeta de salida temporal con la misma config. El SHA-256 de
`matrix_train.parquet` y `targets.parquet` debe coincidir con el manifiesto.

## Verificaciones negativas

- Fijar `abc_cuts = [0.95, 0.80]`: el proceso se detiene antes de cualquier cálculo.
- Editar un valor sin añadir una fila al historial: el proceso se detiene.
- Quitar `AZ` de `treatment_map` cuando existe un objetivo AZ: el proceso se detiene.
- Ejecutar `05_year5_replication` antes de congelar `targets`: el proceso se detiene.
- Cambiar `forecast.ma_windows` (con su fila en el historial) después de la réplica: el backtest
  se detiene y exige una nueva `version`.
