// Construye pronostico_reposicion.pptx con formas nativas (diseño predeterminado del skill).
// Uso: NODE_PATH=<carpeta con pptxgenjs>/node_modules node construir_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");

const DIR = __dirname;
const FIG = (f) => path.join(DIR, "figuras", f);
const OUTFILE = path.join(DIR, "pronostico_reposicion.pptx");

// ------------------------------------------------------------------ paleta y medidas
const P = {
  bg: "F7F4EE", navy: "1C3150", teal: "2A7F7A", tealSoft: "D6E8E6", peach: "B5602A",
  peachSoft: "F6D9C4", sand: "E9E3D5", text: "1E2430", grey: "6B7078", line: "DDD6C8",
  onDark: "9FD3CC", number: "E9A375", panel: "263D5F", sub: "D8E0EC", white: "FFFFFF",
};
const FONT = "Calibri", MONO = "Consolas";
const W = 13.333, M = 0.6, CW = W - 2 * M;       // ancho útil 12.133
const TOP = 1.55, BOTTOM = 6.72;
const RATIO = {                                      // ancho / alto de cada figura
  "pareto_abc.png": 1.683, "xyz_ejemplos.png": 2.916, "estabilidad_anual.png": 2.685,
  "caso_suma_observada.png": 1.973, "caso_por_tienda.png": 1.688, "caso_tratada.png": 1.973,
  "caso_ratio.png": 1.864, "seleccion_justa.png": 2.185, "curva_impacto.png": 1.706,
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Pronóstico de demanda para la reposición";
pres.author = "Equipo de analítica";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

// ------------------------------------------------------------------ diseños (layouts)
pres.defineSlideMaster({
  title: "Contenido",
  background: { color: P.bg },
  objects: [
    { placeholder: { options: { name: "rotulo", type: "body", x: M, y: 0.42, w: CW, h: 0.26, fontFace: FONT,
      fontSize: 11, bold: true, color: P.teal, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.66, w: CW, h: 0.8, fontFace: FONT,
      fontSize: 30, bold: true, color: P.navy, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "pie", type: "body", x: M, y: 6.86, w: 11.2, h: 0.32, fontFace: FONT,
      fontSize: 12, color: P.grey, margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.33, y: 6.86, w: 0.4, h: 0.3, fontFace: FONT, fontSize: 11, color: P.grey, align: "right" },
});
pres.defineSlideMaster({ title: "Portadilla", background: { color: P.navy },
  slideNumber: { x: 12.33, y: 6.86, w: 0.4, h: 0.3, fontFace: FONT, fontSize: 11, color: P.onDark, align: "right" } });
pres.defineSlideMaster({ title: "Portada", background: { color: P.navy } });

// ------------------------------------------------------------------ utilidades
let section = "Apertura";
const CARD = {
  dark: { fill: P.navy, line: P.navy, text: P.white, label: P.onDark },
  light: { fill: P.white, line: P.line, text: P.text, label: P.teal },
  sand: { fill: P.sand, line: P.sand, text: P.text, label: P.grey },
  peach: { fill: P.peachSoft, line: P.peachSoft, text: P.text, label: P.peach },
  teal: { fill: P.tealSoft, line: P.tealSoft, text: P.text, label: P.teal },
};

// Párrafos: cada elemento es un string o un arreglo de runs [{text, b, c, f}].
function runs(paragraphs, base) {
  const out = [];
  paragraphs.forEach((p, i) => {
    const parts = typeof p === "string" ? [{ text: p }] : p;
    parts.forEach((r, j) => {
      const o = { fontFace: r.f || FONT, fontSize: r.s || base.size, color: r.c || base.color, bold: !!r.b,
        paraSpaceAfter: 5, lineSpacingMultiple: 1.1 };
      if (r.bullet) o.bullet = { indent: 14 };
      if (j === parts.length - 1 && i < paragraphs.length - 1) o.breakLine = true;
      out.push({ text: r.text, options: o });
    });
  });
  return out;
}

function card(slide, x, y, w, h, kind, label, paragraphs, opt = {}) {
  const k = CARD[kind];
  const size = opt.size || 15;
  const items = [];
  if (label) items.push({ text: label.toUpperCase(), options: { fontFace: FONT, fontSize: 11, bold: true,
    color: k.label, charSpacing: 1.5, breakLine: true, paraSpaceAfter: 6 } });
  const body = runs(paragraphs, { size, color: k.text });
  slide.addText(items.concat(body), {
    x, y, w, h, shape: pres.ShapeType.roundRect, rectRadius: 0.06,
    fill: { color: k.fill }, line: { color: k.line, width: 1, dashType: opt.dash || "solid" },
    margin: 11, valign: opt.valign || "top", align: "left", isTextBox: true,
    objectName: opt.name || `card ${label || ""}`.trim(),
  });
}

function content(rotulo, title, pie, notes) {
  const s = pres.addSlide({ masterName: "Contenido", sectionTitle: section });
  s.addText(rotulo.toUpperCase(), { placeholder: "rotulo", charSpacing: 1.5 });
  s.addText(title, { placeholder: "title" });
  if (pie) s.addText(pie, { placeholder: "pie" });
  if (notes) s.addNotes(notes);
  return s;
}

function figure(slide, file, x, y, w, alt) {
  const h = w / RATIO[file];
  slide.addImage({ path: FIG(file), x, y, w, h, altText: alt || file });
  return h;
}

function kpi(slide, x, y, w, h, cifra, label, tone) {
  const dark = tone === "acento";
  slide.addText([
    { text: cifra, options: { fontFace: FONT, fontSize: 28, bold: true, color: dark ? P.white : P.navy, breakLine: true, paraSpaceAfter: 4 } },
    { text: label, options: { fontFace: FONT, fontSize: 14, color: dark ? P.white : P.text } },
  ], { x, y, w, h, shape: pres.ShapeType.roundRect, rectRadius: 0.06,
       fill: { color: dark ? P.peach : P.white }, line: { color: dark ? P.peach : P.line, width: 1 },
       margin: 12, valign: "top", isTextBox: true, objectName: `kpi ${cifra}` });
}

function divider(n, question, items) {
  section = `Tema ${n}`;
  pres.addSection({ title: section });
  const s = pres.addSlide({ masterName: "Portadilla", sectionTitle: section });
  s.addText(`TEMA ${n}`, { x: M, y: 1.0, w: 6, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: P.onDark, charSpacing: 2, margin: 0 });
  s.addText(String(n), { x: M, y: 1.3, w: 3, h: 1.6, fontFace: FONT, fontSize: 96, bold: true, color: P.number, margin: 0, valign: "top" });
  s.addText(question, { x: M, y: 3.05, w: 7.6, h: 2.6, fontFace: FONT, fontSize: 34, bold: true, color: P.white, margin: 0, valign: "top" });
  s.addShape(pres.ShapeType.roundRect, { x: 8.75, y: 1.0, w: 3.98, h: 5.5, rectRadius: 0.06, fill: { color: P.panel }, line: { color: P.panel } });
  const list = [{ text: "EN ESTE TEMA", options: { fontFace: FONT, fontSize: 11, bold: true, color: P.onDark, charSpacing: 1.5, breakLine: true, paraSpaceAfter: 10 } }];
  items.forEach((t, i) => {
    list.push({ text: `${i + 1}   `, options: { fontFace: FONT, fontSize: 14, bold: true, color: P.number } });
    list.push({ text: t, options: { fontFace: FONT, fontSize: 14, color: P.white, paraSpaceAfter: 8, breakLine: i < items.length - 1 } });
  });
  s.addText(list, { x: 9.0, y: 1.25, w: 3.5, h: 5.0, valign: "top", margin: 0, isTextBox: true });
  return s;
}

// Tabla con cabecera azul marino.
function table(slide, header, rows, x, y, w, colW, opt = {}) {
  const size = opt.size || 13;
  const head = header.map((t) => ({ text: t, options: { bold: true, color: P.white, fill: { color: P.navy }, fontSize: size, fontFace: FONT, align: "center", valign: "middle" } }));
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    const hl = opt.highlightRow === i;
    return { text: cell.text, options: { fontSize: size, fontFace: FONT, color: cell.c || P.text, bold: !!cell.b || (opt.boldFirst && j === 0),
      fill: { color: cell.fill || (hl ? P.peachSoft : (i % 2 ? "FBFAF6" : P.white)) }, align: j === 0 ? "left" : "center", valign: "middle" } };
  }));
  slide.addTable([head, ...body], { x, y, w, colW, rowH: opt.rowH || 0.36, border: { type: "solid", pt: 0.75, color: P.line },
    margin: [3, 6, 3, 6], autoPage: false });
}

// ================================================================== APERTURA
pres.addSection({ title: "Apertura" });
{
  const s = pres.addSlide({ masterName: "Portada", sectionTitle: "Apertura" });
  s.addText("PRONÓSTICO DE DEMANDA · REPOSICIÓN", { x: M, y: 1.3, w: 11, h: 0.35, fontFace: FONT, fontSize: 12, bold: true, color: P.onDark, charSpacing: 2, margin: 0 });
  s.addText("Reponer con pronóstico", { x: M, y: 1.8, w: 12, h: 1.1, fontFace: FONT, fontSize: 54, bold: true, color: P.white, margin: 0, valign: "top" });
  s.addText("Qué encontramos en los datos, cómo tratamos los quiebres de stock, qué modelo elegimos y cuánto vale para el negocio",
    { x: M, y: 3.05, w: 10.5, h: 1.2, fontFace: FONT, fontSize: 22, color: P.sub, margin: 0, valign: "top" });
  s.addText("Equipo de analítica · 11 de noviembre de 2026", { x: M, y: 6.2, w: 10, h: 0.4, fontFace: FONT, fontSize: 14, color: P.sub, margin: 0 });
  s.addNotes("Presentación de resultados para negocio. Duración sugerida: 30 minutos más preguntas.");
}
{
  const s = content("Resumen ejecutivo", "Renovar cada mes el pronóstico reduce el error a la mitad",
    "Año de prueba: 517 artículos clase A (52 % del gasto), 10 tiendas, 25-may-2015 a 22-may-2016. Montos a precio de venta.",
    "Mensaje central: el valor viene de renovar el pronóstico cada mes. Las cuatro cifras se detallan en los temas 4 y 5.");
  const y = 1.75, h = 1.75, w = (CW - 3 * 0.25) / 4;
  kpi(s, M, y, w, h, "18,4 → 9,3 %", "error sobre lo vendido: repetir el año anterior frente al modelo");
  kpi(s, M + (w + 0.25), y, w, h, "3,28 M US$", "menos inventario de más al 95 % de servicio: 15 % de la demanda del año", "acento");
  kpi(s, M + 2 * (w + 0.25), y, w, h, "2,27 M US$", "vale renovar cada mes frente a pronosticar 13 bloques de una vez");
  kpi(s, M + 3 * (w + 0.25), y, w, h, "490 de 517", "artículos en los que el modelo mejora al punto de partida");
  card(s, M, 3.8, 7.4, 2.75, "dark", "Recomendación", [
    "Reponer los artículos A con el CatBoost global, renovando el pronóstico cada mes con lo último vendido.",
    "El pronóstico encadenado a 13 bloques apenas mejora al de repetir el año anterior: sirve para planear, no para reponer.",
  ], { size: 16 });
  card(s, M + 7.65, 3.8, CW - 7.65, 2.75, "sand", "Por qué funciona", [
    [{ text: "Datos limpios: ", b: true }, { text: "los quiebres de stock se completan tienda por tienda antes de sumar." }],
    [{ text: "Elección justa: ", b: true }, { text: "el modelo se eligió con bloques que no vio al entrenar." }],
  ]);
}
{
  const s = content("Agenda", "Cinco preguntas, de los datos al dinero", null, null);
  const qs = ["¿Qué datos hay y cómo se reparte la venta?", "¿Qué hacemos con los ceros que no son demanda?",
    "¿Qué se pronostica y con qué modelos?", "¿Cómo se eligió el modelo y cuánto se equivoca?", "¿Cuánto vale para el negocio?"];
  const w = (CW - 4 * 0.25) / 5;
  qs.forEach((q, i) => {
    s.addText([
      { text: String(i + 1), options: { fontFace: FONT, fontSize: 48, bold: true, color: P.number, breakLine: true, paraSpaceAfter: 12 } },
      { text: q, options: { fontFace: FONT, fontSize: 18, bold: true, color: P.white } },
    ], { x: M + i * (w + 0.25), y: 1.9, w, h: 4.2, shape: pres.ShapeType.roundRect, rectRadius: 0.06,
         fill: { color: P.navy }, line: { color: P.navy }, margin: 14, valign: "top", isTextBox: true, objectName: `agenda ${i + 1}` });
  });
}

// ================================================================== TEMA 1
divider(1, "¿Qué datos hay y cómo se reparte la venta?", ["Tres tablas y cómo se unen", "Clase ABC: quién concentra el gasto",
  "Clase XYZ: qué tan regular es la venta", "La matriz ABC × XYZ", "¿Se mantienen las clases en el tiempo?"]);
{
  const s = content("Tema 1 · Datos · 1 de 5", "Tres tablas unidas por tienda, artículo y fecha",
    "Datos de una cadena de supermercados de EE. UU.: 3 estados (CA, TX, WI), enero de 2011 a mayo de 2016.",
    "Recorrer las cajas de izquierda a derecha. Recalcar que el bloque de 28 días es la unidad de todo lo que sigue.");
  const box = (x, y, w, h, title, lines, color) => {
    s.addText([{ text: title, options: { fontFace: FONT, fontSize: 18, bold: true, color: P.white, breakLine: true, paraSpaceAfter: 6 } },
      ...lines.map((t, i) => ({ text: t, options: { fontFace: FONT, fontSize: 14, color: P.white, bullet: { indent: 12 }, breakLine: i < lines.length - 1, paraSpaceAfter: 3 } }))],
    { x, y, w, h, shape: pres.ShapeType.roundRect, rectRadius: 0.06, fill: { color }, line: { color }, margin: 12, valign: "top", isTextBox: true, objectName: `tabla ${title}` });
  };
  box(M, 1.75, 3.7, 2.55, "Ventas", ["30.490 series tienda × artículo", "1.941 días: unidades vendidas por día", "Llave: tienda, artículo, día d", "68 % de las celdas son 0"], P.navy);
  box(M + 4.4, 1.75, 3.6, 2.55, "Calendario", ["1.969 días: d → fecha y semana", "Eventos: 30, en 4 tipos (religioso, nacional, cultural, deportivo)", "Días SNAP por estado"], P.teal);
  box(M + 8.7, 1.75, CW - 8.7, 2.55, "Precios", ["6,84 M filas: precio de venta por tienda, artículo y semana", "Sin fila = no estaba a la venta esa semana"], P.peach);
  // conectores
  s.addShape(pres.ShapeType.line, { x: M + 3.7, y: 3.0, w: 0.7, h: 0, line: { color: P.text, width: 1.5, endArrowType: "triangle", beginArrowType: "triangle" } });
  s.addText("día d", { x: M + 3.65, y: 2.6, w: 0.8, h: 0.3, fontFace: FONT, fontSize: 12, color: P.text, align: "center", margin: 0 });
  s.addShape(pres.ShapeType.line, { x: M + 8.0, y: 3.0, w: 0.7, h: 0, line: { color: P.text, width: 1.5, endArrowType: "triangle", beginArrowType: "triangle" } });
  s.addText("semana", { x: M + 7.95, y: 2.6, w: 0.8, h: 0.3, fontFace: FONT, fontSize: 12, color: P.text, align: "center", margin: 0 });
  s.addShape(pres.ShapeType.line, { x: M + 1.85, y: 4.3, w: 0, h: 0.35, line: { color: P.text, width: 1.5 } });
  s.addShape(pres.ShapeType.line, { x: M + 1.85, y: 4.65, w: 8.85, h: 0, line: { color: P.text, width: 1.5 } });
  s.addShape(pres.ShapeType.line, { x: M + 10.7, y: 4.3, w: 0, h: 0.35, line: { color: P.text, width: 1.5, beginArrowType: "triangle" } });
  s.addText("tienda + artículo + semana: precio de cada venta → gasto = unidades × precio", { x: M + 2.2, y: 4.35, w: 8, h: 0.28, fontFace: FONT, fontSize: 12, color: P.text, margin: 0, align: "center" });
  card(s, M, 4.95, 5.9, 1.65, "light", "Jerarquía", ["10 tiendas en 3 estados · 3 categorías (alimentos, hogar, hobbies) en 7 departamentos · 3.049 artículos, cada uno en las 10 tiendas."], { size: 14 });
  card(s, M + 6.15, 4.95, CW - 6.15, 1.65, "teal", "Unidad de análisis: el bloque", ["28 días seguidos (4 semanas). Los 5 años de análisis son 65 bloques: 52 de entrenamiento (años 1–4) y 13 de prueba (año 5)."], { size: 14 });
}
{
  const s = content("Tema 1 · ABC · 2 de 5", "El 19 % de los artículos concentra el 60 % del gasto",
    "Gasto de los años 1–4: 138,4 M US$ en 3.049 artículos. Los cortes se fijan solo con estos años.",
    "Cómo leerlo: el eje horizontal ordena los artículos del que más vende al que menos; la curva acumula su gasto.");
  figure(s, "pareto_abc.png", M, 1.6, 7.4, "Curva de Pareto del gasto con las clases A, B y C");
  card(s, M + 7.7, 1.6, CW - 7.7, 2.45, "dark", "Definición 1.1 · Clase ABC", [
    "Se ordenan los artículos por gasto (unidades × precio). A: los que acumulan el primer 60 % del gasto; B: hasta el 84 %; C: el resto."]);
  card(s, M + 7.7, 4.25, CW - 7.7, 2.35, "light", "Cómo se eligieron los cortes", [
    "Regla: A con 10–20 % de los artículos y 60–80 % del gasto; B con 20–30 % de cada uno. En una grilla de pasos de 1 %, el par 60 % / 84 % la cumple sin desvío."], { size: 14 });
}
{
  const s = content("Tema 1 · XYZ · 3 de 5", "XYZ mide qué tan regular es la venta de cada artículo",
    "Aquí los ceros aún no se tratan: un quiebre de stock cuenta como venta 0 y sube el CV.",
    "Mostrar la diferencia visual entre una serie X y una Z antes de la definición.");
  figure(s, "xyz_ejemplos.png", M, 1.55, 9.0, "Dos artículos: uno regular (X) y uno errático (Z)");
  card(s, M + 9.25, 1.6, CW - 9.25, 3.0, "light", "Sin tratar los ceros", [
    [{ text: "X: ", b: true }, { text: "829 artículos" }], [{ text: "Y: ", b: true }, { text: "1.252" }],
    [{ text: "Z: ", b: true }, { text: "829" }], [{ text: "Sin CV: ", b: true }, { text: "139 (menos de 13 bloques válidos)" }]], { size: 14 });
  card(s, M, 4.85, CW, 1.75, "dark", "Definición 1.2 · Clase XYZ", [
    [{ text: "CV = desviación estándar ÷ media × 100", f: MONO, b: true }, { text: " de la venta diaria promedio de cada bloque de 28 días (años 1–4). X: CV < 26,6 % · Y: 26,6–50,3 % · Z: > 50,3 %." }],
    "Los cortes agrupan log(CV) en tres tramos lo más compactos posible (método de Jenks), con al menos 20 % de los artículos en cada uno."], { size: 14 });
}
{
  const s = content("Tema 1 · Matriz ABC × XYZ · 4 de 5", "Sin tratar los ceros, Z es un cuarto de artículos y gasto",
    "Años 1–4, ceros sin tratar. Sin CV: artículos con menos de 13 bloques válidos para calcularlo.",
    "La fila A es la que se modela. AX, AY y AZ serán los segmentos con que se leen los resultados.");
  s.addText("Artículos", { x: M, y: 1.6, w: 5.9, h: 0.35, fontFace: FONT, fontSize: 16, bold: true, color: P.navy, margin: 0 });
  table(s, ["", "X", "Y", "Z", "Sin CV", "Total"], [
    ["A", "198", "255", "140", "0", { text: "593", b: true }], ["B", "261", "337", "197", "0", { text: "795", b: true }],
    ["C", "370", "660", "492", "139", { text: "1.661", b: true }],
    [{ text: "Total", b: true }, { text: "829", b: true }, { text: "1.252", b: true }, { text: "829", b: true }, { text: "139", b: true }, { text: "3.049", b: true }]],
    M, 2.0, 5.9, [0.9, 1.0, 1.0, 1.0, 1.0, 1.0], { highlightRow: 0, boldFirst: true, rowH: 0.42, size: 14 });
  s.addText("% del gasto", { x: M + 6.25, y: 1.6, w: 5.9, h: 0.35, fontFace: FONT, fontSize: 16, bold: true, color: P.navy, margin: 0 });
  table(s, ["", "X", "Y", "Z", "Sin CV", "Total"], [
    ["A", "19,3", "26,0", "14,7", "0,0", { text: "60,0", b: true }], ["B", "8,0", "10,1", "5,9", "0,0", { text: "24,0", b: true }],
    ["C", "4,0", "6,8", "4,7", "0,5", { text: "16,0", b: true }],
    [{ text: "Total", b: true }, { text: "31,3", b: true }, { text: "42,9", b: true }, { text: "25,3", b: true }, { text: "0,5", b: true }, { text: "100", b: true }]],
    M + 6.25, 2.0, CW - 6.25, [0.88, 1.0, 1.0, 1.0, 1.0, 1.0], { highlightRow: 0, boldFirst: true, rowH: 0.42, size: 14 });
  card(s, M, 4.45, 5.9, 2.15, "peach", "Lo que se modela", ["El segmento A: 593 artículos que suman el 60 % del gasto. Un A es tan probable de ser X como Z: el valor de un artículo no dice qué tan regular es su venta."], { size: 14 });
  card(s, M + 6.25, 4.45, CW - 6.25, 2.15, "sand", "Cómo leer la matriz", ["Cada celda combina las dos clases: AX = mucho gasto y venta regular; AZ = mucho gasto y venta errática. El color resalta la fila que se modela."], { size: 14 });
}
{
  const s = content("Tema 1 · Estabilidad · 5 de 5", "ABC se mantiene año a año; XYZ cambia más",
    "Mismos 1.682 artículos. Cada año se clasifica con los cortes de su año base (el anterior) y se compara con él.",
    "Esto justifica usar ABC para decidir qué se modela y XYZ solo para describir. El kappa descuenta el acuerdo que se daría por azar. Si se fija el año 1 como base para todos, el kappa cae con el tiempo: ABC de 0,68 a 0,55 y XYZ de 0,33 a 0,18.");
  figure(s, "estabilidad_anual.png", M, 1.55, 9.2, "Porcentaje de artículos por clase en cada año");
  card(s, M + 9.45, 1.6, CW - 9.45, 1.55, "light", "ABC", [[{ text: "κ 0,68–0,74", b: true }, { text: ": 81–84 % conserva su clase; ninguna clase cambia más de 2,4 puntos." }]], { size: 14 });
  card(s, M + 9.45, 3.3, CW - 9.45, 1.55, "peach", "XYZ", [[{ text: "κ 0,33–0,37", b: true }, { text: ": 59–63 % conserva su clase; X sube 6,7 puntos en el año 5." }]], { size: 14 });
  card(s, M, 5.15, CW, 1.45, "dark", "Definición 1.3 · Kappa de Cohen (κ)", [
    "Acuerdo entre la clase de un artículo en dos años, descontado el que habría por azar: 1 = todos conservan su clase; 0 = el mismo acuerdo que al azar."], { size: 14 });
}

// ================================================================== TEMA 2
divider(2, "¿Qué hacemos con los ceros que no son demanda?", ["Cero de demanda o quiebre de stock", "Caso guía: la venta total",
  "Caso guía: tienda por tienda", "La serie tratada", "Embudo: de 30.490 series a 517 artículos", "Efecto en los segmentos"]);
{
  const s = content("Tema 2 · Quiebres · 1 de 6", "Una racha de ceros demasiado larga para ser azar es quiebre",
    "Cada tienda tiene su propio umbral: una tienda que vende poco tiene más ceros normales.",
    "Explicar con las dos filas: arriba ceros sueltos (demanda baja), abajo una racha larga (no había producto).");
  // Tiras de días
  const sq = 0.26, gap = 0.05, x0 = M + 2.1;
  const rowsDef = [
    { y: 1.85, label: "Demanda intermitente", zeros: [3, 4, 9, 15, 16, 22, 26], breakRun: [] },
    { y: 2.65, label: "Quiebre de stock", zeros: [], breakRun: [8, 21] },
  ];
  rowsDef.forEach((r) => {
    s.addText(r.label, { x: M, y: r.y - 0.02, w: 2.0, h: 0.32, fontFace: FONT, fontSize: 14, bold: true, color: P.text, margin: 0, valign: "middle" });
    for (let d = 0; d < 28; d++) {
      const isZ = r.zeros.includes(d) || (r.breakRun.length && d >= r.breakRun[0] && d <= r.breakRun[1]);
      const isBreak = r.breakRun.length && d >= r.breakRun[0] && d <= r.breakRun[1];
      s.addShape(pres.ShapeType.rect, { x: x0 + d * (sq + gap), y: r.y, w: sq, h: sq,
        fill: { color: isBreak ? P.peach : (isZ ? P.white : P.navy) }, line: { color: isZ ? P.grey : P.navy, width: 0.75 } });
    }
  });
  s.addText("14 ceros seguidos ≥ umbral de la tienda (10 días)", { x: x0 + 8 * (sq + gap), y: 3.0, w: 5.0, h: 0.3, fontFace: FONT, fontSize: 12, color: P.peach, bold: true, margin: 0 });
  // leyenda
  const lx = x0 + 28 * (sq + gap) + 0.3;
  [["día con venta", P.navy, P.navy], ["cero normal", P.white, P.grey], ["día de quiebre", P.peach, P.peach]].forEach(([t, f, l], i) => {
    s.addShape(pres.ShapeType.rect, { x: lx, y: 1.88 + i * 0.4, w: 0.22, h: 0.22, fill: { color: f }, line: { color: l, width: 0.75 } });
    s.addText(t, { x: lx + 0.32, y: 1.84 + i * 0.4, w: 1.7, h: 0.3, fontFace: FONT, fontSize: 12, color: P.text, margin: 0, valign: "middle" });
  });
  card(s, M, 3.55, 5.0, 3.05, "dark", "Definición 2.1 · Umbral de racha larga", [
    "p₀ = % de días con venta 0 de la tienda en el entrenamiento.",
    "Umbral = el largo k más corto tal que, si cada día vendiera 0 al azar con probabilidad p₀, ver alguna racha de k ceros seguidos en todo el periodo tendría probabilidad ≤ 1 %."], { size: 14 });
  card(s, M + 5.25, 3.55, 3.3, 3.05, "light", "Ejemplo · FOODS_3_655", [
    "TX_2 vende 0 el 29 % de los días → umbral de 10 días.", "CA_1: 38 % → 12 días.", "CA_2: 57 % → 20 días."], { size: 14 });
  card(s, M + 8.8, 3.55, CW - 8.8, 3.05, "sand", "Qué no es quiebre", [
    "Ceros antes de la primera venta o sin precio publicado: no estaba a la venta.",
    "En validación y prueba se decide al cierre de cada bloque, sin mirar el futuro."], { size: 14 });
}
{
  const s = content("Tema 2 · Caso guía · 2 de 6", "FOODS_3_655 cae de ~950 a ~150 por bloque y vuelve de golpe",
    "Artículo AZ (mucho gasto, venta errática). Es el caso que sigue en las próximas diapositivas.",
    "Dejar la pregunta abierta: ¿cayó la demanda o faltó producto?");
  figure(s, "caso_suma_observada.png", M, 1.6, 8.4, "Venta total por bloque de FOODS_3_655");
  card(s, M + 8.7, 1.6, CW - 8.7, 2.2, "light", "Suma observada", ["Lo vendido en cada bloque de 28 días sumando las 10 tiendas, sin ninguna corrección."], { size: 14 });
  card(s, M + 8.7, 4.0, CW - 8.7, 2.6, "peach", "La pregunta", ["¿Cayó la demanda casi un año (bloques 15–25) y volvió de un día para otro? Mirando solo la suma no se puede saber."], { size: 15 });
}
{
  const s = content("Tema 2 · Caso guía · 3 de 6", "Tienda por tienda: 7 de las 10 tiendas se quedan sin stock",
    "Bloques 5 a 34 del caso guía. Azules: California; naranjas: Texas; verdes: Wisconsin.",
    "La suma no llega a cero porque Texas sigue vendiendo: el quiebre queda escondido en el total.");
  figure(s, "caso_por_tienda.png", M, 1.55, 7.5, "Venta por tienda y bloque de FOODS_3_655");
  card(s, M + 7.8, 1.6, CW - 7.8, 1.6, "light", "California y Wisconsin", [[{ text: "~709 → ~5", b: true }, { text: " unidades por bloque, con precio publicado: tenían el artículo a la venta, pero sin stock." }]], { size: 14 });
  card(s, M + 7.8, 3.35, CW - 7.8, 1.6, "light", "Texas sigue vendiendo", [[{ text: "~205 → ~147", b: true }, { text: " por bloque: la suma nunca llega a cero." }]], { size: 14 });
  card(s, M + 7.8, 5.1, CW - 7.8, 1.5, "dark", "Conclusión", ["El quiebre se trata en cada tienda, antes de sumar."], { size: 16 });
}
{
  const s = content("Tema 2 · Serie tratada · 4 de 6", "Completar cada tienda antes de sumar recupera la venta del artículo",
    "Las dos líneas son versiones de la misma venta total; no se suman entre sí. Atípico: bloque con ratio < 0,5 o > 2 que no es un cambio real.",
    "Costo del método: suaviza 14 cambios reales de nivel. Se vigiló al validar el modelo.");
  figure(s, "caso_tratada.png", M, 1.55, 7.5, "Suma observada frente a serie tratada por tienda");
  card(s, M + 7.8, 1.6, CW - 7.8, 3.0, "dark", "Paso a paso", [
    "1. Marcar los días de quiebre de cada tienda con su umbral.",
    "2. Completar cada día marcado con la venta media de los últimos 28 días válidos de esa tienda (solo el pasado).",
    "3. Sumar las 10 tiendas.", "4. Corregir en el artículo los bloques que aún quedan deprimidos."], { size: 14 });
  kpi(s, M, 5.35, 3.6, 1.3, "118 → 28", "bloques atípicos (−76 %)", "acento");
  kpi(s, M + 3.85, 5.35, 3.65, 1.3, "+11,3 %", "unidades completadas (AZ: +22 %)");
  card(s, M + 7.8, 4.8, CW - 7.8, 1.85, "teal", "Qué se logra en el total", ["Una serie continua del artículo, sin pozos ni saltos falsos, que es la que aprende el modelo."], { size: 14 });
}
{
  const s = content("Tema 2 · Embudo · 5 de 6", "De 30.490 series a 517 artículos, el 52 % del gasto",
    "% del gasto de los años 1–4 sobre el total de la cadena (138,4 M US$).",
    "Los 75 artículos A que salen: 2 nunca vuelven a tener stock y 73 quedan con historia insuficiente tras el recorte.");
  const stages = [
    ["30.490", "series tienda × artículo (10 tiendas × 3.049 artículos)", "100 %"],
    ["3.049", "artículos: la suma de sus 10 tiendas", "100 %"],
    ["2.886", "con clase XYZ medible tras quitar los quiebres", "99,2 %"],
    ["592", "del segmento A: AX 207 · AY 251 · AZ 134", "59,9 %"],
    ["590", "vuelven a tener stock tras una caída (salen 2)", "59,1 %"],
    ["517", "con ≥ 39 bloques de historia: AX 191 · AY 229 · AZ 97", "52,2 %"],
  ];
  s.addText("Qué queda", { x: M + 5.1, y: 1.55, w: 3.4, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: P.grey, margin: 0 });
  s.addText("% gasto", { x: M + 8.4, y: 1.55, w: 0.9, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: P.grey, margin: 0, align: "right" });
  const widths = [4.8, 4.25, 3.7, 3.15, 2.6, 2.05];
  stages.forEach(([n, d, g], i) => {
    const y = 1.92 + i * 0.78, w = widths[i], x = M + (4.8 - w) / 2;
    const last = i === stages.length - 1;
    s.addText(n, { x, y, w, h: 0.64, shape: pres.ShapeType.roundRect, rectRadius: 0.06, fill: { color: last ? P.peach : P.navy },
      line: { color: last ? P.peach : P.navy }, fontFace: FONT, fontSize: 22, bold: true, color: P.white, align: "center", valign: "middle", margin: 0, objectName: `embudo ${n}` });
    s.addText(d, { x: M + 5.1, y, w: 3.3, h: 0.64, fontFace: FONT, fontSize: 13, color: P.text, margin: 0, valign: "middle" });
    s.addText(g, { x: M + 8.4, y, w: 0.9, h: 0.64, fontFace: FONT, fontSize: 15, bold: true, color: last ? P.peach : P.navy, margin: 0, valign: "middle", align: "right" });
  });
  card(s, M + 9.6, 1.6, CW - 9.6, 5.05, "sand", "Por qué salen 75", [
    [{ text: "2 ", b: true }, { text: "nunca vuelven a tener stock." }],
    [{ text: "71 ", b: true }, { text: "tuvieron una caída generalizada (≥ 9 de 10 tiendas en quiebre ≥ 6 bloques). Completar casi un año sería inventar la venta: su historia empieza tras la vuelta a stock." }],
    [{ text: "73 ", b: true }, { text: "quedan con < 39 bloques (13 de historia + 13 de validación + 13 de prueba); 61 de ellos por ese recorte." }]], { size: 13 });
}
{
  const s = content("Tema 2 · Segmentos · 6 de 6", "Al quitar los quiebres, Z baja de 829 a 683 artículos",
    "Años 1–4. Los cortes de XYZ se recalculan con la serie limpia: X < 24,1 %, Z > 41,9 %.",
    "La mayor parte de lo 'errático' era falta de producto, no demanda. El segmento A modelado usa la clase limpia.");
  const rowsL = [["X", "829", "893"], ["Y", "1.252", "1.311"], ["Z", "829", "683"], ["Sin CV", "139", "162"], ["CV mediano", "36,5 %", "30,1 %"]];
  [[0, "Ceros sin tratar", false], [1, "Quiebres quitados", true]].forEach(([k, t, hl]) => {
    const x = M + k * 3.65, w = 3.4;
    const col = hl ? P.peach : P.grey;
    s.addText(t, { x, y: 1.65, w, h: 0.5, shape: pres.ShapeType.roundRect, rectRadius: 0.06, fill: { color: col }, line: { color: col },
      fontFace: FONT, fontSize: 16, bold: true, color: P.white, align: "center", valign: "middle", margin: 0 });
    rowsL.forEach((r, i) => {
      s.addText([{ text: r[k + 1], options: { fontSize: 24, bold: true, color: hl ? P.peach : P.text, fontFace: FONT } },
        { text: `  ${r[0]}`, options: { fontSize: 14, color: P.text, fontFace: FONT } }],
      { x, y: 2.3 + i * 0.66, w, h: 0.56, fill: { color: P.white }, line: { color: P.line, width: 1 }, margin: 10, valign: "middle", isTextBox: true });
    });
  });
  card(s, M + 7.6, 1.65, CW - 7.6, 2.4, "dark", "Lectura", ["Z casi siempre era un artículo con quiebres, no con demanda errática: al quitarlos, el CV mediano baja de 36,5 % a 30,1 %."], { size: 15 });
  card(s, M + 7.6, 4.25, CW - 7.6, 2.35, "light", "Segmento A, antes → después", ["AX 198 → 207 · AY 255 → 251 · AZ 140 → 134.", "ABC no cambia: los días en cero no aportan gasto."], { size: 14 });
}

// ================================================================== TEMA 3
divider(3, "¿Qué se pronostica y con qué modelos?", ["La variable a predecir: el ratio", "Tres familias de modelos",
  "Los datos de cada modelo", "Validación: 13 bloques, uno a la vez"]);
{
  const s = content("Tema 3 · Objetivo · 1 de 4", "Se pronostica el ratio: el cambio frente al bloque anterior",
    "Caso guía FOODS_3_655. Las unidades del pronóstico salen de multiplicar el ratio por la venta del bloque anterior.",
    "Volver al caso guía: el ratio de la suma observada tiene los picos del quiebre; el de la serie tratada no.");
  figure(s, "caso_ratio.png", M, 1.55, 7.6, "Ratio de FOODS_3_655 con la suma observada y con la serie tratada");
  card(s, M + 7.9, 1.6, CW - 7.9, 2.0, "dark", "Definición 3.1 · Ratio", [
    [{ text: "r_t = U_t ÷ U_{t−1}", f: MONO, b: true }], [{ text: "Û_t = U_{t−1} × r̂_t", f: MONO, b: true }],
    "U: unidades del artículo en el bloque t (serie tratada)."], { size: 14 });
  card(s, M + 7.9, 3.75, CW - 7.9, 1.25, "light", "En el caso guía", ["Observada: 0,45 y 5,6. Tratada: entre 0,75 y 1,26."], { size: 14 });
  card(s, M + 7.9, 5.15, CW - 7.9, 1.5, "sand", "Por qué el ratio", ["Es comparable entre artículos de 10 o de 10.000 unidades: un solo modelo aprende de todos. Un ratio > 2 no entra al entrenamiento."], { size: 13 });
}
{
  const s = content("Tema 3 · Modelos · 2 de 4", "Tres familias de modelos, de la más simple a la más flexible",
    "Los tres pronostican el ratio de cada bloque y se miden en unidades.",
    "El seasonal naive es la vara: cualquier modelo tiene que ganarle para valer la pena.");
  const cols = [
    ["Seasonal naive", "Línea base", ["Repite el mismo bloque del año anterior:", [{ text: "Û_t = U_{t−13}", f: MONO, b: true }], "Sin parámetros: es lo que hay que mejorar."], "sand"],
    ["SARIMAX", "Un modelo por artículo", ["Modelo estadístico de series de tiempo: usa el ratio de bloques pasados, sus errores y un ciclo (3, 6, 7 o 13 bloques, elegido por artículo).", "Los eventos del bloque entran como variables externas. 511 de 517 artículos con modelo."], "light"],
    ["Boosting global", "Un modelo para los 517", ["Cientos de árboles de decisión en secuencia; cada árbol corrige el error de los anteriores.", "Aprende patrones comunes a todos los artículos; el artículo y su segmento son variables. Se probaron XGBoost, LightGBM y CatBoost."], "dark"],
  ];
  const w = (CW - 2 * 0.25) / 3;
  cols.forEach(([t, sub, body, kind], i) => {
    card(s, M + i * (w + 0.25), 1.65, w, 4.95, kind, `${t} · ${sub}`, body, { size: 15 });
  });
}
{
  const s = content("Tema 3 · Datos por modelo · 3 de 4", "Cada modelo recibe los datos que su arquitectura necesita",
    "Boosting: 31.164 filas (517 artículos × sus bloques). Se evaluaron el precio y los días SNAP y no se agregaron.",
    "SARIMAX no necesita rezagos explícitos: su estructura autorregresiva y estacional los cubre.");
  const ok = (t) => ({ text: t, c: P.teal, b: true });
  const no = { text: "—", c: P.grey };
  table(s, ["Variables", "Seasonal naive", "SARIMAX", "Boosting global"], [
    ["Mismo bloque del año anterior", ok("único dato"), no, no],
    ["Ratio (objetivo)", no, ok("serie por artículo"), ok("una fila por bloque")],
    ["Rezagos del ratio: 1, 2, 3 y 13 bloques", no, { text: "en su estructura" }, ok("4")],
    ["Ventanas de 3, 6 y 13 bloques: media, desviación, mediana, rango", no, no, ok("12")],
    ["Eventos del bloque por tipo y sus rezagos de 1 y 2 bloques", no, ok("4 (sin rezagos)"), ok("12")],
    ["Año, periodo del año (1–13) y edad del artículo", no, { text: "ciclo propio" }, ok("3")],
    ["Artículo y segmento (AX, AY, AZ)", no, { text: "un modelo por artículo" }, ok("2")],
    [{ text: "Total de variables", b: true }, { text: "0", b: true }, { text: "4", b: true }, { text: "33", b: true }],
  ], M, 1.65, CW, [5.0, 2.1, 2.5, 2.533], { rowH: 0.5, size: 14 });
}
{
  const s = content("Tema 3 · Validación · 4 de 4", "La validación pronostica 13 bloques, uno a la vez",
    "Ventana de validación: los 13 últimos bloques del entrenamiento de cada artículo. La prueba no se toca hasta el final.",
    "El bloque validado pasa a ser dato de entrenamiento del siguiente: así se simula renovar el pronóstico cada mes.");
  const sq = 0.2, gap = 0.04, lx = M, hx = M + 1.35, hw = 1.0, x0 = hx + hw + 0.08;
  const rowsB = [["Bloque 1", 1], ["Bloque 2", 2], ["Bloque 3", 3], ["…", 0], ["Bloque 13", 13], ["Prueba", -1]];
  s.addText("historia", { x: hx, y: 1.6, w: hw, h: 0.28, fontFace: FONT, fontSize: 12, color: P.grey, margin: 0, align: "center" });
  s.addText("validación (13 bloques)", { x: x0, y: 1.6, w: 13 * (sq + gap), h: 0.28, fontFace: FONT, fontSize: 12, color: P.grey, margin: 0, align: "center" });
  s.addText("prueba (13 bloques)", { x: x0 + 13 * (sq + gap) + 0.1, y: 1.6, w: 13 * (sq + gap), h: 0.28, fontFace: FONT, fontSize: 12, color: P.grey, margin: 0, align: "center" });
  rowsB.forEach(([lab, k], i) => {
    const y = 1.98 + i * 0.5;
    s.addText(lab, { x: lx, y, w: 1.3, h: sq, fontFace: FONT, fontSize: 13, bold: true, color: P.text, margin: 0, valign: "middle" });
    if (k === 0) return;
    s.addShape(pres.ShapeType.rect, { x: hx, y, w: hw, h: sq, fill: { color: P.navy }, line: { color: P.navy } });
    for (let b = 0; b < 26; b++) {
      let fill = P.white, line = P.line;
      if (k > 0) {
        if (b < k - 1) { fill = P.navy; line = P.navy; } else if (b === k - 1) { fill = P.peach; line = P.peach; }
      } else if (b < 13) { fill = P.navy; line = P.navy; } else { fill = P.teal; line = P.teal; }
      s.addShape(pres.ShapeType.rect, { x: x0 + b * (sq + gap) + (b >= 13 ? 0.1 : 0), y, w: sq, h: sq, fill: { color: fill }, line: { color: line, width: 0.75 } });
    }
  });
  const ly = 5.05;
  [["entrena", P.navy], ["se pronostica y se mide", P.peach], ["prueba final", P.teal], ["aún no se conoce", P.white]].forEach(([t, c], i) => {
    s.addShape(pres.ShapeType.rect, { x: M + i * 2.3, y: ly, w: 0.22, h: 0.22, fill: { color: c }, line: { color: c === P.white ? P.line : c } });
    s.addText(t, { x: M + 0.3 + i * 2.3, y: ly - 0.04, w: 2.0, h: 0.3, fontFace: FONT, fontSize: 12, color: P.text, margin: 0 });
  });
  card(s, M + 9.2, 1.6, CW - 9.2, 3.6, "light", "Cómo se mide", ["En cada bloque el modelo se reentrena con todo lo anterior y pronostica el siguiente.", "Su error es el de los 13 bloques juntos."], { size: 14 });
  card(s, M, 5.45, CW, 1.2, "dark", "Definición 3.2 · WRMSSE y WAPE", [
    "WRMSSE: error cuadrático de cada artículo dividido por el del seasonal naive en su historia, promediado con peso = gasto del último año; < 1 le gana a repetir el año. WAPE = Σ|error| ÷ Σ unidades: % de lo vendido."], { size: 13 });
}

// ================================================================== TEMA 4
divider(4, "¿Cómo se eligió el modelo y cuánto se equivoca?", ["Comparación justa entre candidatos", "Error global en la prueba",
  "Los 10 artículos con mayor y menor error"]);
{
  const s = content("Tema 4 · Selección · 1 de 3", "En una comparación justa, un CatBoost para todos gana",
    "WRMSSE en los bloques 8–13 de la validación (517 artículos). La barra rayada mira los mismos bloques con que mide: es optimista.",
    "Los tres boosting quedaron casi empatados (0,506 a 0,511 en los 13 bloques); CatBoost fue el mejor y es el que compite.");
  figure(s, "seleccion_justa.png", M, 1.6, 7.6, "WRMSSE de cada alternativa en la comparación justa");
  card(s, M + 7.9, 1.6, CW - 7.9, 1.9, "peach", "La trampa del optimista", ["Si cada artículo elige su mejor modelo con los mismos bloques con que se mide, el error sale bajo por azar: 0,436."], { size: 14 });
  card(s, M + 7.9, 3.65, CW - 7.9, 2.95, "dark", "Comparación justa", ["Se elige con los bloques 1–7 y se mide en los 8–13, que no participaron.",
    [{ text: "Elegir por artículo: 0,516. CatBoost para todos: 0,483", b: true }, { text: " (diferencia 0,033; IC 95 % de 0,009 a 0,056)." }]], { size: 14 });
  card(s, M, 5.45, 7.6, 1.2, "teal", "Ganador", ["Un CatBoost global: WRMSSE 0,506 en la validación frente a 0,894 del seasonal naive, reentrenado con los años 1–4."], { size: 14 });
}
{
  const s = content("Tema 4 · Prueba · 2 de 3", "En el año de prueba, el error baja de 18,4 % a 9,3 % de lo vendido",
    "517 artículos × sus 13 últimos bloques, medidos una sola vez. Error en unidades.",
    "El encadenado acumula el error de cada paso: en tasas casi no cambia, pero en unidades se multiplica.");
  const pct = (v) => (v * 100).toFixed(1).replace(".", ",") + " %";
  const n3 = (v) => v.toFixed(3).replace(".", ",");
  const rowsT = [
    ["Renovado cada mes", 0.0746, 0.0970, 0.1147, 0.0930, 0.4948],
    ["Encadenado 13 bloques", 0.1129, 0.1577, 0.2064, 0.1527, 0.7479],
    ["Seasonal naive", 0.1275, 0.1847, 0.2726, 0.1839, 0.8679],
  ].map((r, i) => [{ text: r[0], b: true }, pct(r[1]), pct(r[2]), pct(r[3]), { text: pct(r[4]), b: true }, { text: n3(r[5]), b: true }]);
  table(s, ["Pronóstico", "WAPE AX", "WAPE AY", "WAPE AZ", "WAPE total", "WRMSSE total"], rowsT, M, 1.7, CW, [3.4, 1.7, 1.7, 1.7, 1.75, 1.883], { rowH: 0.55, size: 16, highlightRow: 0 });
  card(s, M, 4.15, 5.95, 2.5, "dark", "Renovado cada mes", ["Cada bloque se pronostica con lo vendido hasta el bloque anterior; el modelo se reentrena cada mes. Es la forma que se recomienda para reponer."], { size: 14 });
  card(s, M + 6.2, 4.15, CW - 6.2, 2.5, "light", "Encadenado 13 bloques", ["Los 13 bloques de una vez desde el inicio del año: cada bloque parte del pronóstico del anterior y el error se acumula. Útil para planear."], { size: 14 });
}
{
  const s = content("Tema 4 · Por artículo · 3 de 3", "Por artículo, 9 de cada 10 quedan bajo 15 % de error",
    "WAPE del pronóstico renovado cada mes en los 13 bloques de prueba (517 artículos; mediana 8,7 %). * Marcado para revisión por error atípico.",
    "7 de los 10 con mayor error son AZ, la demanda más errática; los 10 de menor error son AX y AY.");
  const best = [["HOUSEHOLD_1_124", "AY", 2.4, 0.122], ["HOUSEHOLD_1_146", "AX", 3.3, 0.365], ["FOODS_3_228", "AX", 3.3, 0.246],
    ["HOUSEHOLD_1_066", "AY", 3.6, 0.170], ["FOODS_3_756", "AX", 3.8, 0.249], ["FOODS_1_012", "AY", 4.1, 0.166],
    ["FOODS_2_233", "AY", 4.2, 0.711], ["FOODS_3_377", "AX", 4.2, 0.634], ["HOUSEHOLD_1_498", "AY", 4.2, 0.184], ["FOODS_3_080", "AX", 4.3, 0.305]];
  const worst = [["FOODS_3_069 *", "AZ", 41.1, 1.198], ["FOODS_1_055", "AZ", 26.6, 1.063], ["HOUSEHOLD_1_439", "AY", 22.9, 0.884],
    ["FOODS_1_180", "AZ", 21.7, 0.637], ["HOUSEHOLD_1_441 *", "AY", 21.7, 2.223], ["HOUSEHOLD_1_184", "AZ", 21.7, 1.068],
    ["HOUSEHOLD_1_005 *", "AY", 21.5, 1.466], ["HOUSEHOLD_1_029 *", "AZ", 20.4, 1.853], ["FOODS_2_316", "AZ", 19.7, 0.623], ["HOUSEHOLD_1_357 *", "AZ", 19.4, 1.389]];
  const fmt = (r) => [r[0], r[1], r[2].toFixed(1).replace(".", ",") + " %", r[3].toFixed(2).replace(".", ",")];
  const half = (CW - 0.35) / 2;
  s.addText("Menor error", { x: M, y: 1.55, w: half, h: 0.32, fontFace: FONT, fontSize: 16, bold: true, color: P.teal, margin: 0 });
  table(s, ["Artículo", "Seg.", "WAPE", "RMSSE"], best.map(fmt), M, 1.92, half, [2.6, 0.9, 1.15, 1.24], { rowH: 0.42, size: 13 });
  s.addText("Mayor error", { x: M + half + 0.35, y: 1.55, w: half, h: 0.32, fontFace: FONT, fontSize: 16, bold: true, color: P.peach, margin: 0 });
  table(s, ["Artículo", "Seg.", "WAPE", "RMSSE"], worst.map(fmt), M + half + 0.35, 1.92, half, [2.6, 0.9, 1.15, 1.24], { rowH: 0.42, size: 13 });
}

// ================================================================== TEMA 5
divider(5, "¿Cuánto vale para el negocio?", ["Venta perdida frente a inventario de más", "Cifras al 95 % de servicio", "Supuestos y límites"]);
{
  const s = content("Tema 5 · Impacto · 1 de 3", "Con cualquier colchón, menos venta perdida y menos sobrante",
    "Año de prueba, 517 artículos, demanda de 21,86 M US$ a precio de venta. Cada bloque se evalúa por separado.",
    "Cómo leerlo: cada punto es un colchón z (de 0 a 3); más abajo y a la izquierda es mejor; la estrella es el pronóstico perfecto.");
  figure(s, "curva_impacto.png", M, 1.55, 7.6, "Curva de exceso frente a venta perdida para cada pronóstico");
  card(s, M + 7.9, 1.6, CW - 7.9, 2.85, "dark", "Definición 5.1 · Disponible y servicio", [
    [{ text: "disponible = pronóstico + z·σ", f: MONO, b: true }],
    "σ: error del artículo en la validación. Venta perdida = demanda no atendida; exceso = disponible − demanda.",
    "Servicio = venta atendida ÷ demanda, en dinero: al 95 % se pierde el 5 % de la demanda."], { size: 13 });
  card(s, M + 7.9, 4.6, CW - 7.9, 2.05, "light", "Exceso al 95 % de servicio", [
    [{ text: "Renovado cada mes: ", b: true }, { text: "0,95 M (4,4 % de la demanda)" }],
    [{ text: "Encadenado: ", b: true }, { text: "3,22 M (14,7 %)" }], [{ text: "Seasonal naive: ", b: true }, { text: "4,23 M (19,3 %)" }]], { size: 14 });
}
{
  const s = content("Tema 5 · Impacto · 2 de 3", "Al 95 % de servicio, el exceso baja de 19 % a 4 % de la demanda",
    "A igual servicio (95 %), la venta perdida es la misma en los tres: 5 % de la demanda (1,09 M US$). Montos a precio de venta.",
    "55 artículos explican la mitad del impacto; 27 empeoran, por montos pequeños (el peor, 7.400 US$).");
  const w = (CW - 3 * 0.25) / 4;
  kpi(s, M, 1.65, w, 1.6, "3,28 M US$", "liberados frente al seasonal naive: 15 % de la demanda del año", "acento");
  kpi(s, M + (w + 0.25), 1.65, w, 1.6, "0,91 M US$", "venta recuperada (4,2 % de la demanda) con el exceso del naive");
  kpi(s, M + 2 * (w + 0.25), 1.65, w, 1.6, "2,27 M US$", "menos exceso que encadenar 13 bloques (10,4 % de la demanda)");
  kpi(s, M + 3 * (w + 0.25), 1.65, w, 1.6, "490 de 517", "artículos mejoran frente al naive");
  table(s, ["Exceso al 95 % de servicio", "US$", "% demanda del año", "% venta año anterior", "% venta de la cadena", "Liberado frente al naive"], [
    ["Seasonal naive", "4,23 M", "19,3 %", "23,0 %", "9,4 %", "—"],
    ["Encadenado 13 bloques", "3,22 M", "14,7 %", "17,5 %", "7,2 %", "1,01 M (−24 %)"],
    [{ text: "Renovado cada mes", b: true }, { text: "0,95 M", b: true }, { text: "4,4 %", b: true }, { text: "5,2 %", b: true }, { text: "2,1 %", b: true }, { text: "3,28 M (−77,5 %)", b: true }]],
    M, 3.55, CW, [2.75, 1.25, 1.85, 2.0, 2.0, 2.283], { rowH: 0.46, size: 14, boldFirst: true, highlightRow: 2 });
  s.addText([
    { text: "Bases: ", options: { bold: true } },
    { text: "demanda del año de prueba de los 517 artículos, 21,86 M US$ · su venta del año anterior, 18,35 M · venta de toda la cadena en el año de prueba (3.049 artículos), 45,04 M.", options: { breakLine: true } },
    { text: "Por segmento, el modelo libera el 67 % del exceso del naive en AX, el 76 % en AY y el 82 % en AZ.", options: {} }],
    { x: M, y: 5.5, w: CW, h: 1.1, fontFace: FONT, fontSize: 14, color: P.text, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true });
}
{
  const s = content("Tema 5 · Supuestos · 3 de 3", "Las cifras son una estimación histórica, no un efecto medido",
    "Antes de llevarlo a operación conviene un piloto con tiempos de entrega e inventario real.", null);
  const items = [
    ["Demanda estimada", "En los bloques con quiebre la demanda es la serie tratada; la venta registrada es 9,3 % menor (AZ: 15,6 %)."],
    ["Sin operación logística", "Cada bloque se evalúa solo: sin tiempos de entrega, stock inicial ni arrastre de sobrantes."],
    ["Venta, no ganancia", "Montos a precio de venta: los datos no traen costo ni margen."],
    ["Inventario compartido", "Se pronostica el total del artículo en las 10 tiendas, como si compartieran inventario."],
    ["Alcance", "517 artículos A (52 % del gasto). B y C no se modelaron."],
    ["Encadenado", "143 de 517 artículos empeoran frente al naive: no conviene para reponer."],
  ];
  const w = (CW - 0.25) / 2, h = 1.5;
  items.forEach(([t, d], i) => {
    const x = M + (i % 2) * (w + 0.25), y = 1.6 + Math.floor(i / 2) * (h + 0.2);
    card(s, x, y, w, h, "sand", t, [d], { size: 14 });
  });
}

// ================================================================== CIERRE
section = "Cierre";
pres.addSection({ title: "Cierre" });
{
  const s = content("Cierre", "Conclusiones y próximos pasos", null,
    "Cerrar con la recomendación y pedir la decisión sobre el piloto.");
  const pts = [
    "La venta se concentra: 19 % de los artículos es el 60 % del gasto. ABC es estable; XYZ depende de cómo se traten los ceros.",
    "Los quiebres se esconden en la suma: completarlos por tienda baja 76 % los bloques atípicos y es la base del pronóstico.",
    "Un CatBoost global le gana a elegir modelo por artículo cuando la comparación es justa.",
    "En la prueba, el error baja de 18,4 % a 9,3 %; al 95 % de servicio se liberan 3,28 M US$ (15 % de la demanda del año).",
    "Recomendación: reponer con el pronóstico renovado cada mes; el encadenado a 13 bloques, solo para planear.",
  ];
  pts.forEach((t, i) => {
    const dark = i % 2 === 0, y = 1.6 + i * 1.0;
    s.addText([{ text: `${i + 1}   `, options: { bold: true, fontSize: 18, color: dark ? P.number : P.peach, fontFace: FONT } },
      { text: t, options: { fontSize: 15, color: dark ? P.white : P.text, fontFace: FONT, bold: i === 4 } }],
    { x: M, y, w: 8.2, h: 0.88, shape: pres.ShapeType.roundRect, rectRadius: 0.06, fill: { color: dark ? P.navy : P.white },
      line: { color: dark ? P.navy : P.line }, margin: 10, valign: "middle", isTextBox: true, objectName: `conclusión ${i + 1}` });
  });
  card(s, M + 8.45, 1.6, CW - 8.45, 4.88, "peach", "Próximos pasos", [
    [{ text: "Piloto en operación", b: true, bullet: true }, { text: " con tiempos de entrega e inventario real." }],
    [{ text: "Revisar los 22 artículos", b: true, bullet: true }, { text: " marcados con error atípico." }],
    [{ text: "Extender a B y C", b: true, bullet: true }, { text: " con el mismo flujo." }]], { size: 14 });
}

(async () => {
  await pres.writeFile({ fileName: OUTFILE });
  const { applyTheme } = require(process.env.APPLY_THEME);
  await applyTheme(OUTFILE, {
    name: "Reposición", headFontFace: FONT, bodyFontFace: FONT,
    colors: { dk1: P.text, lt1: P.white, dk2: P.navy, lt2: P.bg, accent1: P.navy, accent2: P.teal, accent3: P.peach,
      accent4: P.grey, accent5: P.tealSoft, accent6: P.peachSoft, hlink: P.teal, folHlink: P.grey },
  });
  console.log("escrito:", OUTFILE);
})();
