"""Step 1: 2PL calibration of the 25 text items on SAPA (theta ~ N(0,1))."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from dif.irt import fit_2pl_mml
from scripts.sapa_common import load_sapa, ITEMS, RES

R, demo = load_sapa()
t = time.time()
fit = fit_2pl_mml(R, names=ITEMS, n_points=41, tol=1e-7)
n = (~np.isnan(R)).sum(0); p = np.nanmean(R, 0)
out = pd.DataFrame({"item": ITEMS, "N": n, "p": p, "a": fit.a, "b": fit.b})
out.to_csv(os.path.join(RES, "sapa_calibration.csv"), index=False)
print(f"N={R.shape[0]} obs={int(n.sum())} EM iters={fit.n_iter} conv={fit.converged} {time.time()-t:.0f}s")
print(out.round(3).to_string(index=False))
