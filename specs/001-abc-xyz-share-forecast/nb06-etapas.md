# nb 06 · Registro de etapas

Notebook: `notebooks/06_reporte_final.ipynb` (el nombre se conserva; el propósito pasa a ser el impacto en el negocio).
Constitución 7.1.0: estima en el test el impacto del pronóstico en dinero (venta capturada, venta perdida y exceso)
contra la demanda tratada, del CatBoost (a un paso y encadenado) frente al seasonal naive y al pronóstico perfecto.
Solo lee lo guardado por el nb 05; no pronostica ni mide el error otra vez. Es descriptivo.
Reglas de trabajo: ver `CLAUDE.md` (R1–R8).

Estados: `[ ]` pendiente · `[~]` escrita, en auditoría · `[x]` aprobada.
Al cerrar una etapa se anotan la fecha y las funciones agregadas o modificadas en `src/`.

## Etapas

- [x] **0 · Decisiones y reglas** (2026-11-10). Aprobada el 2026-11-10 (servicio 95 %; desajuste = perdida + exceso, también por separado). Notebook reescrito (título, Etapa 0, configuración y candado). D1
  cargar nb 05, `rmsse_scale` y `snaive_valid` del 03b e intermedios diarios; D2 demanda tratada (usuario), cruda
  solo como advertencia; D3 precio = gasto crudo / unidades crudas del bloque; D4 dos escenarios con el mismo peso
  (usuario): A a un paso (renovación mensual, producción) y B encadenado, cada uno con su sección, más A frente a B;
  seasonal naive y perfecto como referencias; D5 disponible = max(pronóstico + z·σ, 0), σ = RMSE en la ventana de backtesting; D6 z de
  −1 a 3 cada 0,25, bloque a bloque, sin tiempo de entrega ni stock (usuario); D7 venta capturada, perdida, exceso y
  servicio en dinero; D8 curvas exceso vs venta perdida, lecturas al 95 % de servicio y a z = 0; D9 impacto por
  artículo = desajuste seasonal naive − CatBoost, top/bottom 10 y gráficos 5 + 5; D10 `data/interim/nb06_impact/`
  con candado; D11 descriptivo; D12 sin piloto. Sin cambios en `src/`.
- [x] **1 · Carga, precio por bloque y demanda cruda frente a tratada** (2026-11-10). Aprobada el 2026-11-10. **Corrección (2026-11-10, tras la
  Etapa 3):** σ del CatBoost = RMSSE × √escala (la escala es un error al cuadrado; antes se multiplicaba por la escala
  y el σ salía ~170 veces mayor). Chequeo nuevo: mediana σ a un paso / σ seasonal naive entre 0,2 y 2 (da 0,55). `demand` (artículo × bloque
  de test: demanda tratada, venta y gasto crudos, precio, pronósticos en unidades), `sigma` (σ por artículo:
  CatBoost = RMSSE de la ventana × escala; seasonal naive = RMSE de `snaive_valid`) y `raw_gap` (tratada − cruda en
  unidades y dinero, por segmento). Precio = gasto crudo / unidades crudas, arrastrando el último con ventas. Código
  escrito, sin correr. `src/metrics.py`: `block_totals` (nueva) y su test.
- [x] **2 · Disponible y métricas en dinero para cada colchón z** (2026-11-10). Aprobada el 2026-11-10. `coverage`: artículo × pronóstico
  (a un paso, encadenado, seasonal naive, perfecto) × z (−1 a 3 cada 0,25) con demanda, venta capturada, perdida,
  exceso (dinero) y bloques cortos. Chequeos: capturada + perdida = demanda; perfecto sin pérdida ni exceso;
  monotonía en z. Totales a z = 0 y servicio por z. Código escrito, sin correr. `src/metrics.py`: `coverage_by_z`
  (nueva) y su test.
- [x] **3 · Curvas de intercambio e impacto global y por segmento** (2026-11-10). Aprobada el 2026-11-10. `curves` (segmento × pronóstico ×
  z, con servicio), `readings` por escenario (A, B) y segmento: z y exceso al 95 % (interpolado, sin extrapolar),
  ahorro de exceso, venta recuperada a igual exceso, perdida y exceso a z = 0 y recuperación de lo alcanzable; `ab`
  (exceso extra al 95 % y desajuste extra a z = 0 del encadenado). Secciones A, B y A frente a B con curvas y frases
  de resumen. Código escrito, sin correr. `src/metrics.py`: `service_reading`, `lost_at_excess` (nuevas);
  `src/modeling.py`: `plot_tradeoff_curves`, `FORECAST_STYLES` (nuevas); con sus tests. Gráfico corregido tras feedback (no
  se entendía): montos en millones, rejilla, "↙ mejor", eje vertical recortado entre z = 0 y el 95 %.
- [x] **4 · Impacto por artículo, mayores y menores** (2026-11-10). Aprobada el 2026-11-10. `item_impact` (artículo × escenario): cada
  pronóstico en el z exacto que le da el 95 % en el total; impacto = desajuste naive − desajuste CatBoost (perdida +
  exceso), sus dos partes, % de la demanda, RMSSE y marca del nb 05. Impacto por segmento, concentración (artículos
  para el 50 % y el 80 % del impacto positivo; cuántos empeoran), 10 mayores y 10 menores por dinero y gráficos del
  nb 05 de 5 + 5, en secciones A y B. Código escrito, sin correr. Sin cambios en `src/`.
- [x] **5 · Guardado y cierre** (2026-11-10). Aprobada el 2026-11-10; nb 06 cerrado. `save_tables` en `data/interim/nb06_impact/`: `curves`, `readings`,
  `ab`, `item_impact`, `by_segment`, `concentration` y manifiesto (objetivo de servicio, z, origen nb 05/03b, demanda
  total, cifras principales, versiones), con candado. Chequeo: lo leído = lo guardado. Cierre con advertencias para el
  informe y resumen desde lo guardado. Código escrito, sin correr. Sin cambios en `src/`.
