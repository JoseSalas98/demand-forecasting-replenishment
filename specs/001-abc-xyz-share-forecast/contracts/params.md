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
cv_aggregation = "block28"           # block28 | daily | weekly
cv_min_blocks = 2
abc_rule_item_bounds = [[0.10, 0.20], [0.20, 0.30]]    # % de artículos A y B
abc_rule_spend_bounds = [[0.60, 0.80], [0.20, 0.30]]   # % de gasto A y B
abc_rule_step = 0.01
cv_cut_min_share = 0.20              # find_cv_cuts: mínimo de artículos por segmento
cv_cut_resolution = 100
zero_alpha = 0.01                    # FR-032: P(racha más larga ≥ k) ≤ α
long_run = 7                         # respaldo si el umbral no se puede estimar

[stability]
min_active_days_year = 364
stability_rule = "abc_all"           # abc_all | abc_xyz_all | cell_majority | a_every_year
stability_rule_criterion = "fixed"

[treatment_map]
AX = "regular_replenishment"
AY = "periodic_review_replenishment"
AZ = "promotion_priority"

[forecast]                           # se declara antes de la réplica en el año 5; cambiarlo después exige nueva version
selection_level = "item"             # etiquetas definidas por el analista durante la experimentación
subset_size = "all"                  # "all" o entero N (los N primeros por gasto en train)
backtest_folds = 13                  # 13 × 28 días = Y4 completo (R16)
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
| Regla ABC | Intervalos `[lo, hi]` con `0 ≤ lo < hi ≤ 1`; `0 < abc_rule_step < 1` |
| Ceros | `0 < zero_alpha < 1`; `long_run ≥ 1` entero |
| Cortes CV | `0 < cv_cut_min_share < 0.5`; `cv_cut_resolution ≥ 10` |
| Partición | Bloques contiguos y sin solapamiento. Todo día de train < `test_days[0]`. Longitud de `test_days` = `test_origins × horizon`. |
| Enumeraciones | `cv_aggregation` y `stability_rule` deben ser nombres conocidos |
| Mapa de tratamientos | Los valores son cadenas no vacías |
| Pronóstico | Ventanas y órdenes son enteros no negativos, `backtest_folds × horizon` ≤ días de Y4 |

## `src/config/params_history.csv`

```text
date,parameter,old_value,new_value,reason
2026-10-24,*,,initial,"valores por defecto iniciales"
```

`parameter` es una clave con puntos, por ejemplo `segmentation.abc_cuts`. Los valores se
codifican en JSON.
