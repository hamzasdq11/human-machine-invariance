"""Primary analysis on the 25-item SAPA calibration: human reference vs machine focal.
python3 scripts/run_invariance25.py data/responses_free25.jsonl [tag]
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from dif.anchored import anchored_dif
from dif.mh import mantel_haenszel
from dif.irt import eap_scores
from scripts.sapa_common import load_sapa, load_machine, load_reference_params, ITEMS, RES

path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "main"
R_h, demo = load_sapa()
a_ref, b_ref = load_reference_params()
R_m, meta, long, excluded = load_machine(path)
print(f"[data] human {R_h.shape} machine {R_m.shape} excluded={excluded}")
print(f"[machine] accuracy {np.nanmean(long['y']):.3f}  parse {long['parsable'].mean():.3f}")
n_item = (~np.isnan(R_m)).sum(0)
print(f"[machine] per-item parsed N: median {np.median(n_item):.0f} range {n_item.min()}-{n_item.max()}")

res = anchored_dif(R_h, R_m, item_names=ITEMS, ref_params=(a_ref, b_ref),
                   purify=True, verbose=True)
mh = mantel_haenszel(R_h, R_m, item_names=ITEMS)
tab = res.table.assign(ets=mh["ets"].values, delta_mh=mh["delta_mh"].values,
                       p_machine=np.nanmean(R_m, 0), n_machine=n_item)
tab.to_csv(os.path.join(RES, f"inv25_{tag}_table.csv"), index=False)
s = res.summary()
s.update(n_flagged=int(tab.flag_total.sum()), anchors_final=res.anchors,
         n_purify_iter=res.n_purify_iter,
         ets_BC_rate=float(mh["ets"].isin(["B", "C"]).mean()),
         n_respondents=int(R_m.shape[0]), excluded=excluded,
         accuracy=float(np.nanmean(long["y"])), parse_rate=float(long["parsable"].mean()),
         median_item_n=float(np.median(n_item)))
inv = ~tab["flag_total"].values
th_h, _ = eap_scores(R_h[:, inv], a_ref[inv], b_ref[inv])
th_m, _ = eap_scores(R_m[:, inv], a_ref[inv], b_ref[inv])
s.update(raw_gap=float(np.nanmean(np.nanmean(R_m, 1)) - np.nanmean(np.nanmean(R_h, 1))),
         purified_theta_gap=float(np.nanmean(th_m) - np.nanmean(th_h)))
json.dump(s, open(os.path.join(RES, f"inv25_{tag}_summary.json"), "w"), indent=2, default=float)
print(tab[["item", "b_ref", "b_foc", "a_foc", "lr_total", "p_total", "flag_total", "flag_uniform", "ets", "p_machine"]].round(3).to_string(index=False))
print(json.dumps(s, indent=2, default=float))
