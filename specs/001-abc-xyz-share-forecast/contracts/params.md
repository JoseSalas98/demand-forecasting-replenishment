# Contrato: `src/config/params.toml`

Es la única fuente de parámetros (FR-006). Todas las claves de abajo son obligatorias y se muestra
su valor por defecto. `config.py` se detiene con error si falta una clave, si un valor rompe una
regla de rango (FR-008) o si un valor difiere del último snapshot y no tiene su fila en
`params_history.csv` (FR-007).

```toml
version = "v1"                       # se incrementa ante cualquier cambio posterior al congelamiento

[data]
raw_dir = "data/raw"
sales_file = "sales_train_evaluation.csv"
last_day = 1941

[split]
history_days = [1, 121]
years = { Y1 = [122, 485], Y2 = [486, 849], Y3 = [850, 1213], Y4 = [1214, 1577] }
test_days = [1578, 1941]
test_origins = 13
horizon = 28

[segmentation]
usage_period = "computation_set"
abc_cuts = [0.80, 0.95]
xyz_thresholds = [10.0, 25.0]        # porcentaje
cv_aggregation = "block28"           # block28 | daily | weekly | segment_period
cv_min_blocks = 2

[stability]
segment_candidates = ["quarter", "four_month"]
segment_criterion = "max_mean_abc_agreement"
min_active_days_year = 364
agreement_min = 0.80
psi_bands = [0.10, 0.25]
psi_epsilon = 1e-4
stability_rule = "abc_all"           # abc_all | abc_xyz_all | cell_majority | a_every_year
stability_rule_criterion = "fixed"

[treatment_map]
AX = "regular_replenishment"
AY = "periodic_review_replenishment"
AZ = "promotion_priority"

[forecast]
selection_level = "item"             # etiquetas definidas por el analista durante la experimentación
subset_size = "all"                  # "all" o entero N (los N primeros por gasto en train)
backtest_folds = 4
snaive_m = 7
ma_windows = [7, 14, 28, 56]
arima_p = [0, 1, 2]
arima_d = [0, 1]
arima_q = [0, 1, 2]
sarima_PDQ = [0, 1]
sarima_s = 7

[metrics]
weight_window = [1550, 1577]
```

## Reglas de rango (el proceso se detiene si alguna falla)

| Regla | Verificación |
|---|---|
| Cortes ABC | `0 < abc_cuts[0] < abc_cuts[1] < 1` |
| Umbrales XYZ | `0 ≤ x < z` |
| PSI | `psi_bands` ascendente, todos ≥ 0, `psi_epsilon > 0` |
| Acuerdo | `0 < agreement_min ≤ 1` |
| Partición | Bloques contiguos y sin solapamiento. Todo día de train < `test_days[0]`. Longitud de `test_days` = `test_origins × horizon`. |
| Enumeraciones | `cv_aggregation`, `stability_rule` y `segment_candidates` deben ser nombres conocidos |
| Mapa de tratamientos | Los valores son cadenas no vacías |
| Pronóstico | Ventanas y órdenes son enteros no negativos, `backtest_folds ≥ 1` |

## `src/config/params_history.csv`

```text
date,parameter,old_value,new_value,reason
2026-10-24,*,,initial,"valores por defecto iniciales"
```

`parameter` es una clave con puntos, por ejemplo `segmentation.abc_cuts`. Los valores se
codifican en JSON.
