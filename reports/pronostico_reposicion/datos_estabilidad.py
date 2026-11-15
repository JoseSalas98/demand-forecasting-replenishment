"""Composición ABC/XYZ por año con los cortes del año anterior y kappa año base → año objetivo (mismos artículos)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
import numpy as np, pandas as pd
from src.config import all_years, load_validated_config
from src.data import load_interim
from src.stability import base_target_classes, year_matrices, transition_stats

cfg = load_validated_config(); seg = cfg["segmentation"]; last_day = cfg["data"]["last_day"]
src = load_interim(cfg, last_day)
item_ids = src["item_ids"]; years = all_years(cfg)
start_d, end_d = years["Y1"][0], years["Y5"][1]
ABC_RULE = {"item_bounds": ((0.10, 0.20), (0.20, 0.30)), "spend_bounds": ((0.60, 0.80), (0.20, 0.30)), "step": 0.01}
CV_CUT_KW = {"n_cuts": 2, "min_share": 0.20, "resolution": 100}
year_mats = year_matrices(src, years, seg, src["week_of_day"])
panel = ~np.isnan(src["item_units"][:, start_d - 1:end_d]).any(axis=1)
ym = year_mats[year_mats["item_id"].isin(set(item_ids[panel]))]
names = list(years)
schemes = {"Y1 fija": {t: [names[0]] for t in names[1:]},
           "año anterior": {t: [names[i]] for i, t in enumerate(names[1:])}}
res = {s: {t: base_target_classes(ym, years, b, t, src, ABC_RULE, None, CV_CUT_KW) for t, b in pm.items()}
       for s, pm in schemes.items()}
# Composición con el esquema año base (año anterior) → año objetivo: cada año se clasifica con los
# cortes elegidos en el año anterior, igual que los kappas de la diapositiva.
prev = res["año anterior"]
props = pd.concat({t: r["proportions"].set_index("cell") for t, r in prev.items()}, axis=1)
comp = pd.DataFrame({names[0]: props[(names[1], "base_pct")], **{t: props[(t, "target_pct")] for t in names[1:]}})
abc = comp.groupby(comp.index.str[0]).sum(); xyz = comp.groupby(comp.index.str[1]).sum()
rows = [{"base": s, "nivel": lv, "objetivo": t, **transition_stats(r["items"], lv)}
        for s, rr in res.items() for lv in ["abc", "xyz", "cell"] for t, r in rr.items()]
stats = pd.DataFrame(rows)
print("panel:", int(panel.sum())); print(abc.round(1)); print(xyz.round(1))
print(stats[["base", "nivel", "objetivo", "n", "kappa", "max_shift_pp", "sm_pvalue"]].round(3))
out = Path("reports/pronostico_reposicion/datos")
abc.to_csv(out / "abc_por_anio.csv"); xyz.to_csv(out / "xyz_por_anio.csv")
stats.drop(columns=[c for c in stats.columns if stats[c].dtype == object and c not in ("base", "nivel", "objetivo", "max_shift_class")]).to_csv(out / "estabilidad_kappa.csv", index=False)
