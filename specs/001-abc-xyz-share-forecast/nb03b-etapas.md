# nb 03b · Registro de etapas (experimental)

Notebook: `notebooks/03b_tratamiento_por_tienda.ipynb`.
Independiente y solo de experimentación: no persiste resultados y el nb 04 no carga nada de aquí. D1 ajustada
(usuario, 2026-11-06), dos excepciones: las entradas de los nb 03c y 03d en `data/interim/nb03b_datasets/` (13.3) y los
puntos de control de la búsqueda SARIMAX en `data/interim/nb03b_sarimax/` (14.3).
Reglas de trabajo: ver `CLAUDE.md` (R1–R8).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas

- [x] **0 · Decisiones y reglas.** Aprobada el 2026-11-05. Detección por tienda × día, validez e
  imputación por tienda × bloque, reglas de bloque atípico solo a nivel artículo; módulo nuevo
  `src/store_treatment.py` aprobado.
- [~] **1–13 · Escritas el 2026-11-05** (el usuario pidió desarrollarlas de una vez). Código probado como
  script; falta que el usuario ejecute el notebook. `src/store_treatment.py` (nuevo): `store_units`,
  `store_zero_thresholds`, `flag_store_breaks`, `fill_store_breaks`, `store_block_summary`,
  `add_store_fill`, `item_block_pipeline` (solo para la línea base), `extreme_blocks`,
  `classify_level_changes`; tests en `src/tests/test_store_treatment.py`.
  - 1 · Carga y línea base: reproduce el nb 04 (579 artículos; 143 atípicos: AX 6, AY 41, AZ 96).
  - 2–5 · Por tienda: ciclo de vida (primera venta; días sin precio ya son NaN), `long_run` por tienda
    (Bernoulli, α = 0,01, solo train; mediana 10 / 11 / 12 días), quiebres (1,366,481 días-tienda;
    851,907 escondidos en la suma), imputación con la media de los 28 días válidos previos.
  - 6 · Suma por artículo: +7,0 % de unidades (AX 3,1 %, AY 6,8 %, AZ 13,0 %).
  - 7–13 · Desde la suma, las celdas del nb 04 una por una (Etapas 4, 5, 6, 7, 9, 10 y 11 del nb 04),
    sin exportar datasets. 7.5 compara con la línea base: atípicos 143 → 83 (AX 6 → 1, AY 41 → 13,
    AZ 96 → 69); 12 cambios genuinos borrados; extremos que llegan al modelo 133 → 52.
- [~] **Punto 1 · Quiebres totales (2026-11-05).** `add_store_fill` acepta `masked`: los días de quiebre total
  que enmascaró el nb 04 toman la suma de lo completado por tienda (95 % de esos días en AZ). Atípicos
  143 → 31 (AX 6 → 0, AY 41 → 3, AZ 96 → 28); extremos que llegan al modelo 133 → 36; bloques
  completados por reglas del artículo 3,396 → 228; unidades finales +4,6 % frente a la línea base ya
  imputada (AZ +4,5 % a +7,7 %). Cambios genuinos borrados: 12 → 19. Siguiente: prueba de participación
  por tienda (chi-cuadrado y residuos, idea del usuario a partir de Friedman).
- [~] **5.2 · Prueba de participación por tienda (Friedman), descartada (2026-11-05).** Queda documentada en el
  notebook sin alimentar la Etapa 6 (decisión del usuario): atípicos 31 → 31, genuinos borrados 19 → 17,
  ~6.800 tiendas-bloque completadas y esperados absurdos. `src/store_treatment.py`: `share_deficit_test`,
  `spread_block_extra` y su test. Cierre de la reducción de atípicos: 143 → 31 (AX 0, AY 3, AZ 28);
  los estacionales se dejan para el modelo.
- [x] **5.3 · Rachas largas (2026-11-06).** Aprobada el 2026-11-06. Decisión del usuario: racha larga = más de 6 bloques de quiebre en una
  tienda; caída generalizada = ≥ 9 de 10 tiendas en racha larga a la vez durante ≥ 6 bloques (k = 9) → la historia
  empieza en el primer bloque completo tras la vuelta a stock y, si quedan < 39 bloques, el artículo sale (filtro de
  alcance, no regla de pronóstico); racha aislada → la tienda se excluye en ese periodo (NaN, sin completar; causal en
  ventana y test). Resultado: salen 62 de 579 (AX 12, AY 19, AZ 31). En los mismos 517: el recorte solo no cambia
  los atípicos (28); la exclusión de aisladas los sube a 49 (71 % de los nuevos junto a la salida o vuelta de la
  tienda) y los extremos que llegan al modelo de 27 a 46, con completado 9,5 % → 8,2 %. **Decisión final (usuario,
  2026-11-06): opción 1**, las rachas aisladas se siguen completando; la exclusión queda documentada sin alimentar.
  **Integrado:** la Etapa 6 usa `added_s` y `bounds_x` (`units_treated` sin los 2 sin vuelta y en NaN antes de la
  vuelta a stock; `units_full` guarda los 592); la 4.1 usa `bounds_x`; los conteos fijos 592/579 pasan a
  `len(bounds_x)` y `N_MODEL` (517); la 7.5 compara con la línea base en los mismos artículos (`base_bm`,
  `ext_base_c`, `atyp_base_c`); el manifiesto de la 13.3 lleva `n_items` y el 03c lo usa en su chequeo. La Etapa 14
  (método SARIMAX) no cambia: solo sus chequeos de conteo y el tiempo estimado (~3–3,5 h). `src/store_treatment.py`:
  `long_break_runs`, `exclude_long_runs`, `cut_item_bounds` (nuevas) y sus tests.
- [~] **6a.1 · Ratio interpolado solo para gráficos (2026-11-05).** `src/features.py`: `interpolate_treated`
  (nueva) y su test. Columna `ratio_plot` en `features`; no entra a los datasets (chequeo en la Etapa 11).
  Los gráficos de la Etapa 13 dibujan `ratio_plot` y marcan los tratados con un círculo hueco.
- [~] **13.3 · Exportación para los nb 03c y 03d (2026-11-06).** `sarimax_data`, `boosting_data` (con folds),
  `snaive_valid` y `rmsse_scale` en Parquet + `manifest.json` (listas de variables, `n_folds`, `snaive_m`, `seed` y las
  category, que Parquet no conserva en enteros). `src/data.py`: `save_tables`, `load_tables` (nuevas) y su test en
  `src/tests/test_data.py`.
- [~] **14 · SARIMAX: órdenes por artículo (2026-11-05).** Sin pmdarima (búsqueda propia con statsmodels).
  En el fold 0: selección = todo menos los últimos 6 bloques; contraste = esos 6. s ∈ {sin estacional, 3, 6, 7, 13}
  (un s cabe si la selección tiene ≥ 2·s + 1 bloques); D por OCSB (5 %) y d por KPSS (α = 0,05), máximo 1 cada una;
  búsqueda completa p, q ∈ {0, 1, 2}, (P, Q) ∈ {(1,0), (0,1), (1,1)} y (0,0) si D = 1; dentro de cada s gana el menor
  **AICc** (decisión del usuario); entre s, el menor RMSE en unidades a un paso en el contraste, reestimando
  (respaldo con los últimos coeficientes convergidos). Espacio como constantes del notebook (no en `params.toml`).
  14.1 ACF en los rezagos candidatos (~4 % significativos en cada rezago: casi sin estacionalidad lineal en el ratio);
  14.2 piloto de 30 artículos (5 % sin converger, d = 1 casi nunca, AIC ≠ AICc en 52 %; 157 s de CPU por artículo en
  promedio); la sección se eliminó del notebook (usuario, 2026-11-06). 14.3: búsqueda en los 517 (579 antes de la 5.3) y
  backtesting de 13 folds en la misma corrida (espacio completo, decisión del usuario; ~3,5–4 h con 8 procesos);
  con puntos de control (usuario, 2026-11-06): un lote `.pkl` cada 10 artículos terminados, retoma al relanzar y
  se detiene si cambia la configuración o el número de artículos (`n_items` en la configuración guardada, usuario, 2026-11-06); barra de avance con tqdm (`tqdm` e `ipywidgets` agregados con uv);
  probada en seco con 9 artículos. Hilos: un hilo por proceso con `parallel_config(inner_max_num_threads=1)`. `src/modeling.py`: `ocsb_crit`, `ocsb_test`, `kpss_d`,
  `sarimax_differencing`, `fit_sarimax`, `sarimax_order_grid`, `sarimax_one_step`, `select_sarimax_item`,
  `search_sarimax_orders`, `sarimax_backtest_item`, `seasonal_acf`, `search_sarimax_checkpointed`, `load_sarimax_checkpoints`,
  `_search_item`, `_combine_search`, `_read_batches` (nuevas) y sus tests en `src/tests/test_modeling.py`.
