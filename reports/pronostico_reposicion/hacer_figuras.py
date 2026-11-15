"""Figuras de la presentación de resultados. Lee solo lo guardado por los notebooks.

Uso (desde la raíz del proyecto): python reports/pronostico_reposicion/hacer_figuras.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path("reports/pronostico_reposicion/figuras")
DATA = Path("reports/pronostico_reposicion/datos")
OUT.mkdir(parents=True, exist_ok=True)

# Paleta del deck.
NAVY, TEAL, PEACH, TEXT, GREY, LINE, BG = "#1C3150", "#2A7F7A", "#B5602A", "#1E2430", "#6B7078", "#DDD6C8", "#F7F4EE"
TEAL_SOFT, PEACH_SOFT, SAND = "#D6E8E6", "#F6D9C4", "#E9E3D5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.formatter.use_locale": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figura:", OUT / name)


def es(x, nd=0):
    """Número con formato español: punto de miles, coma decimal."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


TRAIN = (122, 1577)          # días d_122–d_1577 (años 1–4)
N_BLOCKS = 65

# ---------------------------------------------------------------- datos base
item_index = pd.read_parquet("data/interim/item_index.parquet")
item_ids = item_index["item_id"].astype(str).to_numpy() if "item_id" in item_index else item_index.index.astype(str).to_numpy()
item_spend = np.load("data/interim/item_spend.npy", mmap_mode="r")
item_units = np.load("data/interim/item_units.npy", mmap_mode="r")

# ---------------------------------------------------------------- 1. Pareto ABC (train)
spend = np.nansum(np.asarray(item_spend[:, TRAIN[0] - 1:TRAIN[1]]), axis=1)
order = np.argsort(-spend, kind="stable")
s = spend[order]
cum = np.cumsum(s) / s.sum()
before = np.concatenate([[0], cum[:-1]])
cls = np.where(before < 0.60, "A", np.where(before < 0.84, "B", "C"))
n = len(s)
x = np.arange(1, n + 1) / n * 100
counts = {k: int((cls == k).sum()) for k in "ABC"}
shares = {k: float(s[cls == k].sum() / s.sum() * 100) for k in "ABC"}
fig, ax = plt.subplots(figsize=(7.6, 4.4))
colors = {"A": NAVY, "B": TEAL, "C": GREY}
for k in "ABC":
    m = cls == k
    ax.fill_between(x[m], 0, cum[m] * 100, color=colors[k], alpha=0.85 if k != "C" else 0.45, lw=0)
ax.plot(x, cum * 100, color=TEXT, lw=1.6)
xa = counts["A"] / n * 100
xb = (counts["A"] + counts["B"]) / n * 100
for xv, yv in [(xa, 60), (xb, 84)]:
    ax.plot([xv, xv], [0, yv], color=TEXT, lw=0.8, ls="--")
    ax.plot([0, xv], [yv, yv], color=TEXT, lw=0.8, ls="--")
ax.text(xa / 2 + 1.5, 22, f"A\n{es(counts['A'])} art.\n{shares['A']:.0f} %", ha="center", color="white", fontsize=12, weight="bold")
ax.text((xa + xb) / 2, 30, f"B\n{es(counts['B'])} art.\n{shares['B']:.0f} %", ha="center", color="white", fontsize=12, weight="bold")
ax.text((xb + 100) / 2, 45, f"C\n{es(counts['C'])} art.\n{shares['C']:.0f} %", ha="center", color=TEXT, fontsize=12, weight="bold")
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.set_xlabel("% de artículos, del que más vende al que menos")
ax.set_ylabel("% acumulado del gasto")
save(fig, "pareto_abc.png")
pd.Series({**{f"n_{k}": v for k, v in counts.items()}, **{f"gasto_{k}": v for k, v in shares.items()}}).to_csv(DATA / "abc_train.csv")

# ---------------------------------------------------------------- 2. Ejemplos X y Z (bloques de train, sin tratar)
def block_means(units_row, first_day=122, n_blocks=52):
    seg = np.asarray(units_row[first_day - 1:first_day - 1 + 28 * n_blocks], dtype=float).reshape(n_blocks, 28)
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(seg).all(axis=1), np.nan, np.nanmean(seg, axis=1))

pos = {k: i for i, k in enumerate(item_ids)}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=False)
for ax, (iid, cv, lab, col) in zip(axes, [("FOODS_1_002", 12.8, "X", NAVY), ("FOODS_1_005", 68.1, "Z", PEACH)]):
    bm = block_means(item_units[pos[iid]])
    # ciclo de vida: desde la primera venta
    first = np.argmax(np.nan_to_num(bm) > 0)
    bm[:first] = np.nan
    ax.plot(np.arange(1, 53), bm, color=col, lw=2, marker="o", ms=3)
    ax.set_title(f"{iid} · CV = {str(cv).replace('.', ',')} % → {lab}", color=col, weight="bold")
    ax.set_xlabel("bloque de 28 días (años 1–4)")
    ax.set_ylim(0, np.nanmax(bm) * 1.15)
axes[0].set_ylabel("unidades por día")
fig.tight_layout()
save(fig, "xyz_ejemplos.png")

# ---------------------------------------------------------------- 3. Estabilidad por año
abc_y = pd.read_csv(DATA / "abc_por_anio.csv", index_col=0)
xyz_y = pd.read_csv(DATA / "xyz_por_anio.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharey=True)
years = list(abc_y.columns)
for ax, tab, title, cols in [(axes[0], abc_y, "ABC: el reparto casi no se mueve", {"A": NAVY, "B": TEAL, "C": GREY}),
                             (axes[1], xyz_y, "XYZ: se mueve más que ABC", {"X": NAVY, "Y": TEAL, "Z": PEACH})]:
    for k, c in cols.items():
        ax.plot(years, tab.loc[k], color=c, lw=2.4, marker="o")
        ax.text(len(years) - 1 + 0.12, tab.loc[k].iloc[-1], f"{k} {tab.loc[k].iloc[-1]:.0f} %", color=c, va="center", weight="bold")
    ax.set_title(title, loc="left", color=TEXT, weight="bold")
    ax.set_xlim(-0.2, len(years) - 0.2)
    ax.set_xlabel("año")
axes[0].set_ylabel("% de los 1.682 artículos")
axes[0].set_ylim(0, 62)
fig.tight_layout()
save(fig, "estabilidad_anual.png")

# ---------------------------------------------------------------- 4–6. Caso guía FOODS_3_655
CASE = "FOODS_3_655"
si = pd.read_parquet("data/interim/series_index.parquet")
units = np.load("data/interim/units.npy", mmap_mode="r")
rows = si.index[si["item_id"] == CASE].to_numpy()
stores = si.loc[rows, "store_id"].tolist()
xs = np.asarray(units[rows, 121:121 + 28 * N_BLOCKS], dtype=float).reshape(len(rows), N_BLOCKS, 28)
by_store = np.nansum(xs, axis=2)                      # tienda × bloque, suma observada
observed = by_store.sum(axis=0)
blocks = np.arange(1, N_BLOCKS + 1)
bd = pd.read_parquet("data/interim/nb03b_datasets/boosting_data.parquet")
case = bd[bd["item_id"].astype(str) == CASE].set_index("block").sort_index()
treated = pd.Series(np.nan, index=blocks)
treated.loc[case.index] = case["units"].to_numpy()
treated.loc[1] = observed[0]                           # el bloque 1 no tiene ratio, pero sí unidades
pd.DataFrame({"observada": observed, "tratada": treated.values}, index=blocks).to_csv(DATA / "caso_foods_3_655.csv")

def shade_test(ax):
    ax.axvspan(52.5, 65.5, color=SAND, zorder=0)
    ax.text(59, ax.get_ylim()[1] * 0.95, "test (año 5)", ha="center", va="top", color=GREY, fontsize=11)

fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.plot(blocks, observed, color=GREY, lw=2.2, marker="o", ms=3.5)
ax.annotate("bloque 15: cae a 431", (15, observed[14]), (19, 1250), color=PEACH, weight="bold",
            arrowprops=dict(arrowstyle="->", color=PEACH))
ax.annotate("bloques 16–25: ~150", (20, observed[19]), (27, 1000), color=PEACH, weight="bold",
            arrowprops=dict(arrowstyle="->", color=PEACH))
ax.annotate("bloque 26: vuelve a 724", (26, observed[25]), (33, 1350), color=PEACH, weight="bold",
            arrowprops=dict(arrowstyle="->", color=PEACH))
ax.set_ylim(0, 1700)
shade_test(ax)
ax.set_xlabel("bloque de 28 días")
ax.set_ylabel("unidades por bloque")
save(fig, "caso_suma_observada.png")

store_colors = {"CA_1": "#1C3150", "CA_2": "#34507A", "CA_3": "#5574A0", "CA_4": "#8CA3C4",
                "TX_1": "#B5602A", "TX_2": "#D08A55", "TX_3": "#E8B98F",
                "WI_1": "#2A7F7A", "WI_2": "#5CA8A2", "WI_3": "#A3CFCB"}
fig, ax = plt.subplots(figsize=(8.6, 4.0))
bottom = np.zeros(N_BLOCKS)
sel = slice(4, 34)   # bloques 5–34 alrededor del quiebre
for i, st in enumerate(stores):
    ax.bar(blocks[sel], by_store[i, sel], bottom=bottom[sel], color=store_colors[st], width=0.85, label=st)
    bottom += by_store[i]
ax.axvspan(14.5, 25.5, color=PEACH_SOFT, zorder=0)
ax.text(20, 1080, "7 tiendas sin stock", ha="center", color=PEACH, weight="bold")
ax.set_xlabel("bloque de 28 días")
ax.set_ylabel("unidades por bloque")
ax.set_ylim(0, 1200)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=10)
save(fig, "caso_por_tienda.png")
tx = [i for i, st in enumerate(stores) if st.startswith("TX")]
other = [i for i in range(len(stores)) if i not in tx]
pd.DataFrame({"tx_antes": [by_store[tx, 4:14].sum(0).mean()], "tx_quiebre": [by_store[tx, 15:25].sum(0).mean()],
              "otras_antes": [by_store[other, 4:14].sum(0).mean()], "otras_quiebre": [by_store[other, 15:25].sum(0).mean()]}
             ).round(0).to_csv(DATA / "caso_tiendas_resumen.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.plot(blocks, observed, color=GREY, lw=1.8, marker="o", ms=3, label="suma observada")
ax.plot(blocks, treated.values, color=TEAL, lw=2.6, label="serie tratada por tienda")
ax.fill_between(blocks, observed, treated.values, where=treated.values > observed, color=TEAL_SOFT, alpha=0.9)
ax.set_ylim(0, 1700)
shade_test(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0), frameon=False)
ax.set_xlabel("bloque de 28 días")
ax.set_ylabel("unidades por bloque")
save(fig, "caso_tratada.png")

# ---------------------------------------------------------------- 7. Ratio del caso
r_obs = pd.Series(observed[1:] / observed[:-1], index=blocks[1:])
r_tr = case["ratio_raw"]
fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.plot(r_obs.index, r_obs.values, color=GREY, lw=1.6, marker="o", ms=3, label="ratio de la suma observada")
ax.plot(r_tr.index, r_tr.values, color=NAVY, lw=2.6, label="ratio de la serie tratada (objetivo)")
ax.axhline(1, color=TEXT, lw=0.8)
ax.annotate("5,6", (26, r_obs.loc[26]), (29, 5.3), color=PEACH, weight="bold", arrowprops=dict(arrowstyle="->", color=PEACH))
ax.annotate("0,45", (15, r_obs.loc[15]), (8, 0.15), color=PEACH, weight="bold", arrowprops=dict(arrowstyle="->", color=PEACH))
ax.set_ylim(0, 6)
shade_test(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.6, 0.88), frameon=False)
ax.set_xlabel("bloque de 28 días")
ax.set_ylabel("ratio = bloque ÷ bloque anterior")
save(fig, "caso_ratio.png")
pd.Series({"r_tr_min": r_tr.min(), "r_tr_max": r_tr.max(), "r_obs_15": r_obs.loc[15], "r_obs_26": r_obs.loc[26]}).to_csv(DATA / "caso_ratio.csv")

# ---------------------------------------------------------------- 8. Selección: comparación justa (folds 8–13)
sel_vals = [("Elegir por artículo\n(mirando 8–13: optimista)", 0.4363, SAND),
            ("CatBoost para todos", 0.4831, TEAL),
            ("Elegir por artículo\n(con 1–7: justo)", 0.5165, GREY),
            ("SARIMAX para todos", 0.6257, GREY),
            ("Seasonal naive", 0.8265, GREY)]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
labels = [v[0] for v in sel_vals][::-1]
vals = [v[1] for v in sel_vals][::-1]
cols = [v[2] for v in sel_vals][::-1]
bars = ax.barh(labels, vals, color=cols, edgecolor=[GREY if c == SAND else c for c in cols], hatch=None)
bars[-1].set_hatch("//")
for b, v in zip(bars, vals):
    ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}".replace(".", ","), va="center", weight="bold")
ax.set_xlim(0, 0.95)
ax.set_xlabel("WRMSSE en los bloques 8–13 de la ventana (menor es mejor)")
ax.grid(axis="y", visible=False)
save(fig, "seleccion_justa.png")

# ---------------------------------------------------------------- 9. Curva exceso vs venta perdida
cur = pd.read_parquet("data/interim/nb06_impact/curves.parquet")
rd = pd.read_parquet("data/interim/nb06_impact/readings.parquet")
tot = cur[cur["segment"] == "total"]
fig, ax = plt.subplots(figsize=(7.8, 4.3))
for f, c, lab in [("seasonal naive", GREY, "Seasonal naive"), ("encadenado", PEACH, "CatBoost encadenado 13 bloques"),
                  ("a un paso", TEAL, "CatBoost renovado cada mes")]:
    t = tot[(tot["forecast"] == f) & (tot["z"] >= 0)].sort_values("z")
    ax.plot(t["excess"] / 1e6, t["lost"] / 1e6, color=c, lw=2.6, marker="o", ms=3.5, label=lab)
# puntos al 95 %
A = rd[(rd["scenario"].str.startswith("A")) & (rd["segment"] == "total")].iloc[0]
B = rd[(rd["scenario"].str.startswith("B")) & (rd["segment"] == "total")].iloc[0]
demand = float(tot["demand"].iloc[0])
lost95 = 0.05 * demand / 1e6
for xval, c in [(A["exceso95_modelo"], TEAL), (B["exceso95_modelo"], PEACH), (A["exceso95_naive"], GREY)]:
    ax.plot(xval / 1e6, lost95, "o", ms=10, mfc="white", mec=c, mew=2.5, zorder=5)
ax.axhline(lost95, color=TEXT, lw=0.8, ls="--")
ax.text(5.6, lost95 + 0.05, "95 % de servicio", ha="right", va="bottom", fontsize=11)
ax.plot(0, 0, "*", ms=14, color=NAVY)
ax.text(0.1, 0.08, "perfecto", color=NAVY, weight="bold")
ax.set_xlim(-0.1, 5.7); ax.set_ylim(-0.05, 2.8)
ax.set_xlabel("exceso: inventario de más (millones de US$, a precio de venta)")
ax.set_ylabel("venta perdida (millones de US$)")
ax.legend(loc="upper right", frameon=False)
save(fig, "curva_impacto.png")

# ---------------------------------------------------------------- tablas para el deck
met = pd.read_parquet("data/interim/nb05_test/metrics.parquet")
met[(met["nivel"] == "unidades") & met["step"].isna()].to_csv(DATA / "test_global.csv", index=False)
im = pd.read_parquet("data/interim/nb05_test/item_metrics.parquet")
one = im[im["form"] == "a un paso"].sort_values("wape_units")
one.head(10).to_csv(DATA / "top10_menor_error.csv", index=False)
one.tail(10).iloc[::-1].to_csv(DATA / "top10_mayor_error.csv", index=False)

t = pd.read_parquet("data/processed/targets.parquet")
print("gasto A total:", t["spend_share"].sum())
mod = set(bd["item_id"].astype(str))
print("gasto 517:", t[t["item_id"].astype(str).isin(mod)]["spend_share"].sum())
print(t[t["item_id"].astype(str).isin(mod)].groupby("cell").size())
