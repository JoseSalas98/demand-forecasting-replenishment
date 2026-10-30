# Contrato: artefactos serializados

Todas las tablas son Parquet. Los artefactos congelados los escribe una sola vez `freeze()`, que
los registra en el manifiesto. `load_frozen()` verifica el hash y se detiene si no coincide
(Principio V).

| Ruta | Congelado | Columnas / contenido |
|---|---|---|
| `data/interim/units.npy`, `prices_daily.npy`, `series_index.parquet` | no | Arreglos anchos (serie × día, NaN = enmascarado) y `store_id, item_id` por fila |
| `data/interim/launch.parquet` | no | `store_id, item_id, launch_wk, launch_d` |
| `data/interim/rejections.parquet` | no | `reason, source, row_id, column, detail` |
| `data/processed/zero_thresholds.parquet` | sí | `item_id, n_days, p0, threshold, source, reason, ww_z, ww_pvalue` (FR-032) |
| `data/processed/train_cuts.json` | sí | `abc_cuts`, `abc_share_bounds` (de `select_abc_cuts`) y `xyz_thresholds` (de `find_cv_cuts`) elegidos en train (etapa 3, FR-013) |
| `data/processed/matrix_train.parquet` | sí | `item_id, launch_d, usage, spend, cum_spend, cum_pct_before, abc_class, cv, xyz_class, cell, exclusion_reason` |
| `data/processed/matrices_by_set.parquet` | sí | Mismas columnas más `set_id` (años), con los cortes de `train_cuts` y el CV con rachas en NaN |
| `data/processed/stability.parquet` | sí | `base_kind (prev, y1, noise, global), base, target, level, n, kappa, stuart_maxwell, sm_df, sm_pvalue, max_shift_pp, max_shift_class, keep_pct, up_pct, down_pct, mixed_pct, spearman, note` (`note` = motivo si `find_cv_cuts` no encontró partición y la comparación quedó sin XYZ) |
| `data/processed/item_stability.parquet` | sí | `item_id, comparison (Yt−1->Yt), abc, ref_abc, xyz, ref_xyz, same_abc, same_xyz, same_cell, a_every_year, stable, rule` |
| `data/processed/targets.parquet` | sí | `item_id, cell, xyz_class, spend_train, spend_share, treatment` |
| `data/processed/segmentation_params.json` | sí | `abc_cuts`, `abc_share_bounds` y `xyz_thresholds` elegidos en train; regla ABC (`abc_rule_*`), `cv_cut_min_share`, `cv_cut_resolution`, `zero_alpha`, `long_run`, `cv_aggregation`, `cv_min_blocks`, `stability_rule`, `treatment_map` (FR-020) |
| `data/processed/matrix_y5.parquet` | sí | Mismas columnas que `matrix_train` más `set_id = "Y5"`. Solo diagnóstico; ninguna etapa de selección la lee. |
| `data/processed/stability_y5.parquet` | sí | Mismas columnas que `stability.parquet`, con `target = "Y5"` y bases Y4 y Y1 |
| `data/processed/year5_replication.parquet` | sí | `item_id, frozen_cell, ref_abc, ref_xyz, y5_abc, y5_xyz, y5_cell, same_abc, same_xyz, same_cell, same_as_frozen` (solo productos objetivo) |
| `models/frozen_models.json` | sí | `[{selection_unit, item_ids[], candidate_id, family, hyperparams, order, seasonal_order, trend, scale, params[]}]` (más `window` en la media móvil) |
| `models/baseline_models.json` | sí | Seasonal naive ajustado en todo train para cada objetivo, mismo formato que `frozen_models.json` |
| `models/fit_failures.parquet` | no | `item_id, candidate_id, stage, message` |
| `data/processed/backtest_scores.parquet` | sí | `selection_unit, candidate_id, fold, rmsse_mean` |
| `data/processed/forecasts_test.parquet` | no | `model_id, item_id, origin, h, d, share_hat, units_hat, units_actual` |
| `data/processed/evaluation.parquet` | no | `model_id, wrmsse, wape, rmse, beats_snaive, n_items, excluded` más `wrmsse_o1 … wrmsse_o13` |
| `reports/eda_five_years/` | no | Salida descriptiva de la etapa 2 (primera etapa de análisis). Ninguna etapa de decisión la lee. |

## `data/processed/manifest.json`

```json
{
  "version": "v1",
  "declarations": [{"key": "stability.stability_rule", "value": "abc_all",
                    "declared_at": "2026-10-25T10:00:00Z"}],
  "artifacts": {"targets": {"path": "data/processed/targets.parquet", "sha256": "…",
                            "config_sha256": "…", "frozen_at": "2026-10-25T11:00:00Z"}},
  "config_snapshot": {"…": "params.toml completo como JSON"}
}
```

Antes de ejecutar una etapa, sus declaraciones deben estar registradas con una marca de tiempo
anterior al inicio de la etapa (FR-015, FR-023 y FR-024). Las claves de `[forecast]` se declaran
al inicio de la réplica en el año 5, antes de leer cualquier día de Y5. El backtest se detiene si
alguna de esas declaraciones es posterior al `frozen_at` de `matrix_y5` o si su `value` difiere
de la config actual.

## `data/processed/test_ledger.json`

```json
[{"model_id": "sarima_item_v1", "evaluated_at": "…", "frozen_models_sha256": "…",
  "targets_sha256": "…"}]
```

`evaluate_test(model_id)` se detiene si `model_id` ya está en el ledger (SC-007) o si alguna
marca de congelamiento es posterior a la ejecución actual.
