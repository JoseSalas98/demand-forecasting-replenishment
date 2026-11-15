---
title: "Reponer con pronóstico"
modo: negocio
estilo: infografico
ejercicios: false
entrega: pptx
---

<!-- Slide number: 1 -->
PRONÓSTICO DE DEMANDA · REPOSICIÓN
Reponer con pronóstico
Qué encontramos en los datos, cómo tratamos los quiebres de stock, qué modelo elegimos y cuánto vale para el negocio
Equipo de analítica · 11 de noviembre de 2026

### Notes:
Presentación de resultados para negocio. Duración sugerida: 30 minutos más preguntas.

<!-- Slide number: 2 -->
RESUMEN EJECUTIVO
Renovar cada mes el pronóstico reduce el error a la mitad
18,4 → 9,3 %
error sobre lo vendido: repetir el año anterior frente al modelo
3,28 M US$
menos inventario de más al 95 % de servicio: 15 % de la demanda del año
2,27 M US$
vale renovar cada mes frente a pronosticar 13 bloques de una vez
490 de 517
artículos en los que el modelo mejora al punto de partida
RECOMENDACIÓN
Reponer los artículos A con el CatBoost global, renovando el pronóstico cada mes con lo último vendido.
El pronóstico encadenado a 13 bloques apenas mejora al de repetir el año anterior: sirve para planear, no para reponer.
POR QUÉ FUNCIONA
Datos limpios: los quiebres de stock se completan tienda por tienda antes de sumar.
Elección justa: el modelo se eligió con bloques que no vio al entrenar.
Año de prueba: 517 artículos clase A (52 % del gasto), 10 tiendas, 25-may-2015 a 22-may-2016. Montos a precio de venta.
2

### Notes:
Mensaje central: el valor viene de renovar el pronóstico cada mes. Las cuatro cifras se detallan en los temas 4 y 5.

<!-- Slide number: 3 -->
AGENDA
Cinco preguntas, de los datos al dinero
1
¿Qué datos hay y cómo se reparte la venta?
2
¿Qué hacemos con los ceros que no son demanda?
3
¿Qué se pronostica y con qué modelos?
4
¿Cómo se eligió el modelo y cuánto se equivoca?
5
¿Cuánto vale para el negocio?
3

<!-- Slide number: 4 -->
TEMA 1
EN ESTE TEMA
1   Tres tablas y cómo se unen
2   Clase ABC: quién concentra el gasto
3   Clase XYZ: qué tan regular es la venta
4   La matriz ABC × XYZ
5   ¿Se mantienen las clases en el tiempo?
1
¿Qué datos hay y cómo se reparte la venta?
4

<!-- Slide number: 5 -->
TEMA 1 · DATOS · 1 DE 5
Tres tablas unidas por tienda, artículo y fecha
Ventas
30.490 series tienda × artículo
1.941 días: unidades vendidas por día
Llave: tienda, artículo, día d
68 % de las celdas son 0
Calendario
1.969 días: d → fecha y semana
Eventos: 30, en 4 tipos (religioso, nacional, cultural, deportivo)
Días SNAP por estado
Precios
6,84 M filas: precio de venta por tienda, artículo y semana
Sin fila = no estaba a la venta esa semana
día d
semana
tienda + artículo + semana: precio de cada venta → gasto = unidades × precio
JERARQUÍA
10 tiendas en 3 estados · 3 categorías (alimentos, hogar, hobbies) en 7 departamentos · 3.049 artículos, cada uno en las 10 tiendas.
UNIDAD DE ANÁLISIS: EL BLOQUE
28 días seguidos (4 semanas). Los 5 años de análisis son 65 bloques: 52 de entrenamiento (años 1–4) y 13 de prueba (año 5).
Datos de una cadena de supermercados de EE. UU.: 3 estados (CA, TX, WI), enero de 2011 a mayo de 2016.
5

### Notes:
Recorrer las cajas de izquierda a derecha. Recalcar que el bloque de 28 días es la unidad de todo lo que sigue.

<!-- Slide number: 6 -->
TEMA 1 · ABC · 2 DE 5
El 19 % de los artículos concentra el 60 % del gasto
![Curva de Pareto del gasto con las clases A, B y C](Image 0)
DEFINICIÓN 1.1 · CLASE ABC
Se ordenan los artículos por gasto (unidades × precio). A: los que acumulan el primer 60 % del gasto; B: hasta el 84 %; C: el resto.
CÓMO SE ELIGIERON LOS CORTES
Regla: A con 10–20 % de los artículos y 60–80 % del gasto; B con 20–30 % de cada uno. En una grilla de pasos de 1 %, el par 60 % / 84 % la cumple sin desvío.
Gasto de los años 1–4: 138,4 M US$ en 3.049 artículos. Los cortes se fijan solo con estos años.
6

### Notes:
Cómo leerlo: el eje horizontal ordena los artículos del que más vende al que menos; la curva acumula su gasto.

<!-- Slide number: 7 -->
TEMA 1 · XYZ · 3 DE 5
XYZ mide qué tan regular es la venta de cada artículo
![Dos artículos: uno regular (X) y uno errático (Z)](Image 0)
SIN TRATAR LOS CEROS
X: 829 artículos
Y: 1.252
Z: 829
Sin CV: 139 (menos de 13 bloques válidos)
DEFINICIÓN 1.2 · CLASE XYZ
CV = desviación estándar ÷ media × 100 de la venta diaria promedio de cada bloque de 28 días (años 1–4). X: CV < 26,6 % · Y: 26,6–50,3 % · Z: > 50,3 %.
Los cortes agrupan log(CV) en tres tramos lo más compactos posible (método de Jenks), con al menos 20 % de los artículos en cada uno.
Aquí los ceros aún no se tratan: un quiebre de stock cuenta como venta 0 y sube el CV.
7

### Notes:
Mostrar la diferencia visual entre una serie X y una Z antes de la definición.

<!-- Slide number: 8 -->
TEMA 1 · MATRIZ ABC × XYZ · 4 DE 5
Sin tratar los ceros, Z es un cuarto de artículos y gasto
Artículos
% del gasto

|  | X | Y | Z | Sin CV | Total |
| --- | --- | --- | --- | --- | --- |
| A | 198 | 255 | 140 | 0 | 593 |
| B | 261 | 337 | 197 | 0 | 795 |
| C | 370 | 660 | 492 | 139 | 1.661 |
| Total | 829 | 1.252 | 829 | 139 | 3.049 |

|  | X | Y | Z | Sin CV | Total |
| --- | --- | --- | --- | --- | --- |
| A | 19,3 | 26,0 | 14,7 | 0,0 | 60,0 |
| B | 8,0 | 10,1 | 5,9 | 0,0 | 24,0 |
| C | 4,0 | 6,8 | 4,7 | 0,5 | 16,0 |
| Total | 31,3 | 42,9 | 25,3 | 0,5 | 100 |

LO QUE SE MODELA
El segmento A: 593 artículos que suman el 60 % del gasto. Un A es tan probable de ser X como Z: el valor de un artículo no dice qué tan regular es su venta.
CÓMO LEER LA MATRIZ
Cada celda combina las dos clases: AX = mucho gasto y venta regular; AZ = mucho gasto y venta errática. El color resalta la fila que se modela.
Años 1–4, ceros sin tratar. Sin CV: artículos con menos de 13 bloques válidos para calcularlo.
8

### Notes:
La fila A es la que se modela. AX, AY y AZ serán los segmentos con que se leen los resultados.

<!-- Slide number: 9 -->
TEMA 1 · ESTABILIDAD · 5 DE 5
ABC se mantiene año a año; XYZ cambia más
![Porcentaje de artículos por clase en cada año](Image 0)
ABC
κ 0,68–0,74: 81–84 % conserva su clase; ninguna clase cambia más de 2,4 puntos.
XYZ
κ 0,33–0,37: 59–63 % conserva su clase; X sube 6,7 puntos en el año 5.
DEFINICIÓN 1.3 · KAPPA DE COHEN (Κ)
Acuerdo entre la clase de un artículo en dos años, descontado el que habría por azar: 1 = todos conservan su clase; 0 = el mismo acuerdo que al azar.
Mismos 1.682 artículos. Cada año se clasifica con los cortes de su año base (el anterior) y se compara con él.
9

### Notes:
Esto justifica usar ABC para decidir qué se modela y XYZ solo para describir. El kappa descuenta el acuerdo que se daría por azar. Si se fija el año 1 como base para todos, el kappa cae con el tiempo: ABC de 0,68 a 0,55 y XYZ de 0,33 a 0,18.

<!-- Slide number: 10 -->
TEMA 2
EN ESTE TEMA
1   Cero de demanda o quiebre de stock
2   Caso guía: la venta total
3   Caso guía: tienda por tienda
4   La serie tratada
5   Embudo: de 30.490 series a 517 artículos
6   Efecto en los segmentos
2
¿Qué hacemos con los ceros que no son demanda?
10

<!-- Slide number: 11 -->
TEMA 2 · QUIEBRES · 1 DE 6
Una racha de ceros demasiado larga para ser azar es quiebre
Demanda intermitente
día con venta
cero normal
Quiebre de stock
día de quiebre
14 ceros seguidos ≥ umbral de la tienda (10 días)
DEFINICIÓN 2.1 · UMBRAL DE RACHA LARGA
p₀ = % de días con venta 0 de la tienda en el entrenamiento.
Umbral = el largo k más corto tal que, si cada día vendiera 0 al azar con probabilidad p₀, ver alguna racha de k ceros seguidos en todo el periodo tendría probabilidad ≤ 1 %.
EJEMPLO · FOODS_3_655
TX_2 vende 0 el 29 % de los días → umbral de 10 días.
CA_1: 38 % → 12 días.
CA_2: 57 % → 20 días.
QUÉ NO ES QUIEBRE
Ceros antes de la primera venta o sin precio publicado: no estaba a la venta.
En validación y prueba se decide al cierre de cada bloque, sin mirar el futuro.
Cada tienda tiene su propio umbral: una tienda que vende poco tiene más ceros normales.
11

### Notes:
Explicar con las dos filas: arriba ceros sueltos (demanda baja), abajo una racha larga (no había producto).

<!-- Slide number: 12 -->
TEMA 2 · CASO GUÍA · 2 DE 6
FOODS_3_655 cae de ~950 a ~150 por bloque y vuelve de golpe
![Venta total por bloque de FOODS_3_655](Image 0)
SUMA OBSERVADA
Lo vendido en cada bloque de 28 días sumando las 10 tiendas, sin ninguna corrección.
LA PREGUNTA
¿Cayó la demanda casi un año (bloques 15–25) y volvió de un día para otro? Mirando solo la suma no se puede saber.
Artículo AZ (mucho gasto, venta errática). Es el caso que sigue en las próximas diapositivas.
12

### Notes:
Dejar la pregunta abierta: ¿cayó la demanda o faltó producto?

<!-- Slide number: 13 -->
TEMA 2 · CASO GUÍA · 3 DE 6
Tienda por tienda: 7 de las 10 tiendas se quedan sin stock
![Venta por tienda y bloque de FOODS_3_655](Image 0)
CALIFORNIA Y WISCONSIN
~709 → ~5 unidades por bloque, con precio publicado: tenían el artículo a la venta, pero sin stock.
TEXAS SIGUE VENDIENDO
~205 → ~147 por bloque: la suma nunca llega a cero.
CONCLUSIÓN
El quiebre se trata en cada tienda, antes de sumar.
Bloques 5 a 34 del caso guía. Azules: California; naranjas: Texas; verdes: Wisconsin.
13

### Notes:
La suma no llega a cero porque Texas sigue vendiendo: el quiebre queda escondido en el total.

<!-- Slide number: 14 -->
TEMA 2 · SERIE TRATADA · 4 DE 6
Completar cada tienda antes de sumar recupera la venta del artículo
![Suma observada frente a serie tratada por tienda](Image 0)
PASO A PASO
1. Marcar los días de quiebre de cada tienda con su umbral.
2. Completar cada día marcado con la venta media de los últimos 28 días válidos de esa tienda (solo el pasado).
3. Sumar las 10 tiendas.
4. Corregir en el artículo los bloques que aún quedan deprimidos.
QUÉ SE LOGRA EN EL TOTAL
Una serie continua del artículo, sin pozos ni saltos falsos, que es la que aprende el modelo.
118 → 28
bloques atípicos (−76 %)
+11,3 %
unidades completadas (AZ: +22 %)
Las dos líneas son versiones de la misma venta total; no se suman entre sí. Atípico: bloque con ratio < 0,5 o > 2 que no es un cambio real.
14

### Notes:
Costo del método: suaviza 14 cambios reales de nivel. Se vigiló al validar el modelo.

<!-- Slide number: 15 -->
TEMA 2 · EMBUDO · 5 DE 6
De 30.490 series a 517 artículos, el 52 % del gasto
Qué queda
% gasto
POR QUÉ SALEN 75
2 nunca vuelven a tener stock.
71 tuvieron una caída generalizada (≥ 9 de 10 tiendas en quiebre ≥ 6 bloques). Completar casi un año sería inventar la venta: su historia empieza tras la vuelta a stock.
73 quedan con < 39 bloques (13 de historia + 13 de validación + 13 de prueba); 61 de ellos por ese recorte.
30.490
series tienda × artículo (10 tiendas × 3.049 artículos)
100 %
3.049
artículos: la suma de sus 10 tiendas
100 %
2.886
con clase XYZ medible tras quitar los quiebres
99,2 %
592
del segmento A: AX 207 · AY 251 · AZ 134
59,9 %
590
vuelven a tener stock tras una caída (salen 2)
59,1 %
517
con ≥ 39 bloques de historia: AX 191 · AY 229 · AZ 97
52,2 %
% del gasto de los años 1–4 sobre el total de la cadena (138,4 M US$).
15

### Notes:
Los 75 artículos A que salen: 2 nunca vuelven a tener stock y 73 quedan con historia insuficiente tras el recorte.

<!-- Slide number: 16 -->
TEMA 2 · SEGMENTOS · 6 DE 6
Al quitar los quiebres, Z baja de 829 a 683 artículos
Ceros sin tratar
Quiebres quitados
LECTURA
Z casi siempre era un artículo con quiebres, no con demanda errática: al quitarlos, el CV mediano baja de 36,5 % a 30,1 %.
829  X
893  X
1.252  Y
1.311  Y
829  Z
683  Z
SEGMENTO A, ANTES → DESPUÉS
AX 198 → 207 · AY 255 → 251 · AZ 140 → 134.
ABC no cambia: los días en cero no aportan gasto.
139  Sin CV
162  Sin CV
36,5 %  CV mediano
30,1 %  CV mediano
Años 1–4. Los cortes de XYZ se recalculan con la serie limpia: X < 24,1 %, Z > 41,9 %.
16

### Notes:
La mayor parte de lo 'errático' era falta de producto, no demanda. El segmento A modelado usa la clase limpia.

<!-- Slide number: 17 -->
TEMA 3
EN ESTE TEMA
1   La variable a predecir: el ratio
2   Tres familias de modelos
3   Los datos de cada modelo
4   Validación: 13 bloques, uno a la vez
3
¿Qué se pronostica y con qué modelos?
17

<!-- Slide number: 18 -->
TEMA 3 · OBJETIVO · 1 DE 4
Se pronostica el ratio: el cambio frente al bloque anterior
![Ratio de FOODS_3_655 con la suma observada y con la serie tratada](Image 0)
DEFINICIÓN 3.1 · RATIO
r_t = U_t ÷ U_{t−1}
Û_t = U_{t−1} × r̂_t
U: unidades del artículo en el bloque t (serie tratada).
EN EL CASO GUÍA
Observada: 0,45 y 5,6. Tratada: entre 0,75 y 1,26.
POR QUÉ EL RATIO
Es comparable entre artículos de 10 o de 10.000 unidades: un solo modelo aprende de todos. Un ratio > 2 no entra al entrenamiento.
Caso guía FOODS_3_655. Las unidades del pronóstico salen de multiplicar el ratio por la venta del bloque anterior.
18

### Notes:
Volver al caso guía: el ratio de la suma observada tiene los picos del quiebre; el de la serie tratada no.

<!-- Slide number: 19 -->
TEMA 3 · MODELOS · 2 DE 4
Tres familias de modelos, de la más simple a la más flexible
SEASONAL NAIVE · LÍNEA BASE
Repite el mismo bloque del año anterior:
Û_t = U_{t−13}
Sin parámetros: es lo que hay que mejorar.
SARIMAX · UN MODELO POR ARTÍCULO
Modelo estadístico de series de tiempo: usa el ratio de bloques pasados, sus errores y un ciclo (3, 6, 7 o 13 bloques, elegido por artículo).
Los eventos del bloque entran como variables externas. 511 de 517 artículos con modelo.
BOOSTING GLOBAL · UN MODELO PARA LOS 517
Cientos de árboles de decisión en secuencia; cada árbol corrige el error de los anteriores.
Aprende patrones comunes a todos los artículos; el artículo y su segmento son variables. Se probaron XGBoost, LightGBM y CatBoost.
Los tres pronostican el ratio de cada bloque y se miden en unidades.
19

### Notes:
El seasonal naive es la vara: cualquier modelo tiene que ganarle para valer la pena.

<!-- Slide number: 20 -->
TEMA 3 · DATOS POR MODELO · 3 DE 4
Cada modelo recibe los datos que su arquitectura necesita

| Variables | Seasonal naive | SARIMAX | Boosting global |
| --- | --- | --- | --- |
| Mismo bloque del año anterior | único dato | — | — |
| Ratio (objetivo) | — | serie por artículo | una fila por bloque |
| Rezagos del ratio: 1, 2, 3 y 13 bloques | — | en su estructura | 4 |
| Ventanas de 3, 6 y 13 bloques: media, desviación, mediana, rango | — | — | 12 |
| Eventos del bloque por tipo y sus rezagos de 1 y 2 bloques | — | 4 (sin rezagos) | 12 |
| Año, periodo del año (1–13) y edad del artículo | — | ciclo propio | 3 |
| Artículo y segmento (AX, AY, AZ) | — | un modelo por artículo | 2 |
| Total de variables | 0 | 4 | 33 |

Boosting: 31.164 filas (517 artículos × sus bloques). Se evaluaron el precio y los días SNAP y no se agregaron.
20

### Notes:
SARIMAX no necesita rezagos explícitos: su estructura autorregresiva y estacional los cubre.

<!-- Slide number: 21 -->
TEMA 3 · VALIDACIÓN · 4 DE 4
La validación pronostica 13 bloques, uno a la vez
historia
validación (13 bloques)
prueba (13 bloques)
CÓMO SE MIDE
En cada bloque el modelo se reentrena con todo lo anterior y pronostica el siguiente.
Su error es el de los 13 bloques juntos.
Bloque 1
Bloque 2
Bloque 3
…
Bloque 13
Prueba
entrena
se pronostica y se mide
prueba final
aún no se conoce
DEFINICIÓN 3.2 · WRMSSE Y WAPE
WRMSSE: error cuadrático de cada artículo dividido por el del seasonal naive en su historia, promediado con peso = gasto del último año; < 1 le gana a repetir el año. WAPE = Σ|error| ÷ Σ unidades: % de lo vendido.
Ventana de validación: los 13 últimos bloques del entrenamiento de cada artículo. La prueba no se toca hasta el final.
21

### Notes:
El bloque validado pasa a ser dato de entrenamiento del siguiente: así se simula renovar el pronóstico cada mes.

<!-- Slide number: 22 -->
TEMA 4
EN ESTE TEMA
1   Comparación justa entre candidatos
2   Error global en la prueba
3   Los 10 artículos con mayor y menor error
4
¿Cómo se eligió el modelo y cuánto se equivoca?
22

<!-- Slide number: 23 -->
TEMA 4 · SELECCIÓN · 1 DE 3
En una comparación justa, un CatBoost para todos gana
![WRMSSE de cada alternativa en la comparación justa](Image 0)
LA TRAMPA DEL OPTIMISTA
Si cada artículo elige su mejor modelo con los mismos bloques con que se mide, el error sale bajo por azar: 0,436.
COMPARACIÓN JUSTA
Se elige con los bloques 1–7 y se mide en los 8–13, que no participaron.
Elegir por artículo: 0,516. CatBoost para todos: 0,483 (diferencia 0,033; IC 95 % de 0,009 a 0,056).
GANADOR
Un CatBoost global: WRMSSE 0,506 en la validación frente a 0,894 del seasonal naive, reentrenado con los años 1–4.
WRMSSE en los bloques 8–13 de la validación (517 artículos). La barra rayada mira los mismos bloques con que mide: es optimista.
23

### Notes:
Los tres boosting quedaron casi empatados (0,506 a 0,511 en los 13 bloques); CatBoost fue el mejor y es el que compite.

<!-- Slide number: 24 -->
TEMA 4 · PRUEBA · 2 DE 3
En el año de prueba, el error baja de 18,4 % a 9,3 % de lo vendido

| Pronóstico | WAPE AX | WAPE AY | WAPE AZ | WAPE total | WRMSSE total |
| --- | --- | --- | --- | --- | --- |
| Renovado cada mes | 7,5 % | 9,7 % | 11,5 % | 9,3 % | 0,495 |
| Encadenado 13 bloques | 11,3 % | 15,8 % | 20,6 % | 15,3 % | 0,748 |
| Seasonal naive | 12,8 % | 18,5 % | 27,3 % | 18,4 % | 0,868 |

RENOVADO CADA MES
Cada bloque se pronostica con lo vendido hasta el bloque anterior; el modelo se reentrena cada mes. Es la forma que se recomienda para reponer.
ENCADENADO 13 BLOQUES
Los 13 bloques de una vez desde el inicio del año: cada bloque parte del pronóstico del anterior y el error se acumula. Útil para planear.
517 artículos × sus 13 últimos bloques, medidos una sola vez. Error en unidades.
24

### Notes:
El encadenado acumula el error de cada paso: en tasas casi no cambia, pero en unidades se multiplica.

<!-- Slide number: 25 -->
TEMA 4 · POR ARTÍCULO · 3 DE 3
Por artículo, 9 de cada 10 quedan bajo 15 % de error
Menor error
Mayor error

| Artículo | Seg. | WAPE | RMSSE |
| --- | --- | --- | --- |
| HOUSEHOLD_1_124 | AY | 2,4 % | 0,12 |
| HOUSEHOLD_1_146 | AX | 3,3 % | 0,36 |
| FOODS_3_228 | AX | 3,3 % | 0,25 |
| HOUSEHOLD_1_066 | AY | 3,6 % | 0,17 |
| FOODS_3_756 | AX | 3,8 % | 0,25 |
| FOODS_1_012 | AY | 4,1 % | 0,17 |
| FOODS_2_233 | AY | 4,2 % | 0,71 |
| FOODS_3_377 | AX | 4,2 % | 0,63 |
| HOUSEHOLD_1_498 | AY | 4,2 % | 0,18 |
| FOODS_3_080 | AX | 4,3 % | 0,30 |

| Artículo | Seg. | WAPE | RMSSE |
| --- | --- | --- | --- |
| FOODS_3_069 * | AZ | 41,1 % | 1,20 |
| FOODS_1_055 | AZ | 26,6 % | 1,06 |
| HOUSEHOLD_1_439 | AY | 22,9 % | 0,88 |
| FOODS_1_180 | AZ | 21,7 % | 0,64 |
| HOUSEHOLD_1_441 * | AY | 21,7 % | 2,22 |
| HOUSEHOLD_1_184 | AZ | 21,7 % | 1,07 |
| HOUSEHOLD_1_005 * | AY | 21,5 % | 1,47 |
| HOUSEHOLD_1_029 * | AZ | 20,4 % | 1,85 |
| FOODS_2_316 | AZ | 19,7 % | 0,62 |
| HOUSEHOLD_1_357 * | AZ | 19,4 % | 1,39 |

WAPE del pronóstico renovado cada mes en los 13 bloques de prueba (517 artículos; mediana 8,7 %). * Marcado para revisión por error atípico.
25

### Notes:
7 de los 10 con mayor error son AZ, la demanda más errática; los 10 de menor error son AX y AY.

<!-- Slide number: 26 -->
TEMA 5
EN ESTE TEMA
1   Venta perdida frente a inventario de más
2   Cifras al 95 % de servicio
3   Supuestos y límites
5
¿Cuánto vale para el negocio?
26

<!-- Slide number: 27 -->
TEMA 5 · IMPACTO · 1 DE 3
Con cualquier colchón, menos venta perdida y menos sobrante
![Curva de exceso frente a venta perdida para cada pronóstico](Image 0)
DEFINICIÓN 5.1 · DISPONIBLE Y SERVICIO
disponible = pronóstico + z·σ
σ: error del artículo en la validación. Venta perdida = demanda no atendida; exceso = disponible − demanda.
Servicio = venta atendida ÷ demanda, en dinero: al 95 % se pierde el 5 % de la demanda.
EXCESO AL 95 % DE SERVICIO
Renovado cada mes: 0,95 M (4,4 % de la demanda)
Encadenado: 3,22 M (14,7 %)
Seasonal naive: 4,23 M (19,3 %)
Año de prueba, 517 artículos, demanda de 21,86 M US$ a precio de venta. Cada bloque se evalúa por separado.
27

### Notes:
Cómo leerlo: cada punto es un colchón z (de 0 a 3); más abajo y a la izquierda es mejor; la estrella es el pronóstico perfecto.

<!-- Slide number: 28 -->
TEMA 5 · IMPACTO · 2 DE 3
Al 95 % de servicio, el exceso baja de 19 % a 4 % de la demanda
3,28 M US$
liberados frente al seasonal naive: 15 % de la demanda del año
0,91 M US$
venta recuperada (4,2 % de la demanda) con el exceso del naive
2,27 M US$
menos exceso que encadenar 13 bloques (10,4 % de la demanda)
490 de 517
artículos mejoran frente al naive

| Exceso al 95 % de servicio | US$ | % demanda del año | % venta año anterior | % venta de la cadena | Liberado frente al naive |
| --- | --- | --- | --- | --- | --- |
| Seasonal naive | 4,23 M | 19,3 % | 23,0 % | 9,4 % | — |
| Encadenado 13 bloques | 3,22 M | 14,7 % | 17,5 % | 7,2 % | 1,01 M (−24 %) |
| Renovado cada mes | 0,95 M | 4,4 % | 5,2 % | 2,1 % | 3,28 M (−77,5 %) |

Bases: demanda del año de prueba de los 517 artículos, 21,86 M US$ · su venta del año anterior, 18,35 M · venta de toda la cadena en el año de prueba (3.049 artículos), 45,04 M.
Por segmento, el modelo libera el 67 % del exceso del naive en AX, el 76 % en AY y el 82 % en AZ.
A igual servicio (95 %), la venta perdida es la misma en los tres: 5 % de la demanda (1,09 M US$). Montos a precio de venta.
28

### Notes:
55 artículos explican la mitad del impacto; 27 empeoran, por montos pequeños (el peor, 7.400 US$).

<!-- Slide number: 29 -->
TEMA 5 · SUPUESTOS · 3 DE 3
Las cifras son una estimación histórica, no un efecto medido
DEMANDA ESTIMADA
En los bloques con quiebre la demanda es la serie tratada; la venta registrada es 9,3 % menor (AZ: 15,6 %).
SIN OPERACIÓN LOGÍSTICA
Cada bloque se evalúa solo: sin tiempos de entrega, stock inicial ni arrastre de sobrantes.
VENTA, NO GANANCIA
Montos a precio de venta: los datos no traen costo ni margen.
INVENTARIO COMPARTIDO
Se pronostica el total del artículo en las 10 tiendas, como si compartieran inventario.
ALCANCE
517 artículos A (52 % del gasto). B y C no se modelaron.
ENCADENADO
143 de 517 artículos empeoran frente al naive: no conviene para reponer.
Antes de llevarlo a operación conviene un piloto con tiempos de entrega e inventario real.
29

<!-- Slide number: 30 -->
CIERRE
Conclusiones y próximos pasos
1   La venta se concentra: 19 % de los artículos es el 60 % del gasto. ABC es estable; XYZ depende de cómo se traten los ceros.
PRÓXIMOS PASOS
Piloto en operación con tiempos de entrega e inventario real.
Revisar los 22 artículos marcados con error atípico.
Extender a B y C con el mismo flujo.
2   Los quiebres se esconden en la suma: completarlos por tienda baja 76 % los bloques atípicos y es la base del pronóstico.
3   Un CatBoost global le gana a elegir modelo por artículo cuando la comparación es justa.
4   En la prueba, el error baja de 18,4 % a 9,3 %; al 95 % de servicio se liberan 3,28 M US$ (15 % de la demanda del año).
5   Recomendación: reponer con el pronóstico renovado cada mes; el encadenado a 13 bloques, solo para planear.
30

### Notes:
Cerrar con la recomendación y pedir la decisión sobre el piloto.
