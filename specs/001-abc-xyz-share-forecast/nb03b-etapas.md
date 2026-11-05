# nb 03b · Registro de etapas (experimental)

Notebook: `notebooks/03b_tratamiento_por_tienda.ipynb`.
Independiente y solo de experimentación: no persiste nada y el nb 04 no carga nada de aquí.
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
- [~] **6a.1 · Ratio interpolado solo para gráficos (2026-11-05).** `src/features.py`: `interpolate_treated`
  (nueva) y su test. Columna `ratio_plot` en `features`; no entra a los datasets (chequeo en la Etapa 11).
  Los gráficos de la Etapa 13 dibujan `ratio_plot` y marcan los tratados con un círculo hueco.
