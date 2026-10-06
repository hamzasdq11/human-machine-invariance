"""Robustness of A12 (models as examinees): per-prompt single answers, one model per
family, leave-one-model-out, and a guessing-floor (c = 1/8) reference."""
import os, sys, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import scripts.a12_eight_examinees25 as T
from scripts.sapa_common import load_machine, ITEMS, RES
R_m, meta, _, _ = load_machine("data/responses_free25.jsonl")
models = sorted(meta.model.unique())
def modal(mask_rows):
    P = np.array([np.nanmean(R_m[((meta.model == m) & mask_rows).values], 0) for m in models])
    return np.where(np.abs(P - .5) < 1e-9, np.nan, (P > .5).astype(float))
out = {}
for v in ["v1", "v2", "v3", "v4", "v5"]:
    t = T.item_test(modal(meta.prompt_variant == v), T.A0, T.B0)
    out[f"prompt_{v}"] = dict(rate=float(t.flag.mean()), flagged=list(t.item[t.flag]))
X = modal(pd.Series(True, index=meta.index))
fam = {"Qwen": [i for i, m in enumerate(models) if "Qwen" in m], "OLMo": [i for i, m in enumerate(models) if "OLMo" in m]}
others = [i for i, m in enumerate(models) if "Qwen" not in m and "OLMo" not in m]
for q, o in itertools.product(fam["Qwen"], fam["OLMo"]):
    t = T.item_test(X[sorted(others + [q, o])], T.A0, T.B0)
    out[f"family_{models[q].split('/')[1]}+{models[o].split('/')[1]}"] = dict(rate=float(t.flag.mean()), flagged=list(t.item[t.flag]))
for k, m in enumerate(models):
    t = T.item_test(np.delete(X, k, 0), T.A0, T.B0)
    out[f"drop_{m.split('/')[1]}"] = dict(rate=float(t.flag.mean()), flagged=list(t.item[t.flag]))
# guessing floor
from dif.irt import prob_2pl as _p2
C = 0.125
T.prob_2pl = lambda th, a, b: np.clip(C + (1 - C) * _p2(th, a, b), 1e-12, 1 - 1e-12)
g = pd.read_csv(os.path.join(RES, "sapa_calibration_3pl_c125.csv")).set_index("item").loc[ITEMS]
t = T.item_test(X, g.a.values, g.b.values)
out["guess_floor_c125"] = dict(rate=float(t.flag.mean()), flagged=list(t.item[t.flag]))
json.dump(out, open(os.path.join(RES, "a12_variants25.json"), "w"), indent=2)
for k, v in out.items(): print(f"{k:55s} {v['rate']:.2f} {v['flagged']}")
