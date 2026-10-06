"""Power of the models-as-examinees test (A12): 8 simulated 2PL examinees at the models'
thetas; 5 items at a time shifted by d logits for all 8; 40 data sets per group and shift.
Reports per-shift median per-item power and type-I on the rest.
Camera-ready (Tier 1): the same design is also run with the 5 items made HARDER for the
examinees ("harder" key, separate seeds); the easier-direction results are unchanged."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
import scripts.a12_eight_examinees25 as T
from dif.irt import prob_2pl
from scripts.sapa_common import ITEMS, RES
TH = list(json.load(open(os.path.join(RES, "a12_eight_examinees25.json")))["model_thetas"].values())
def one(job):
    d, g, rep, sgn = job
    seed = int(1e6 * d) + 1000 * g + rep               # easier: the original seeds
    rng = np.random.default_rng(seed if sgn > 0 else [seed, 2])
    inj = (np.arange(25) % 5) == g; b = T.B0.copy(); b[inj] -= sgn * d
    X = (rng.random((8, 25)) < prob_2pl(TH, T.A0, b)).astype(float)
    f = T.item_test(X, T.A0, T.B0).flag.values
    return dict(d=d, g=g, rep=rep, sgn=sgn, flags="".join(map(str, f.astype(int))))
def summarise(df, d):
    F = np.array([[int(c) for c in s] for s in df["flags"]]); I = np.array([(np.arange(25) % 5) == g for g in df.g])
    s = (df.d == d).values; pw = (F[s] * I[s]).sum(0) / I[s].sum(0); t1 = (F[s] * ~I[s]).sum() / (~I[s]).sum()
    return dict(median_power=float(np.median(pw)), items_ge_08=int((pw >= .8).sum()), typeI=float(t1),
                power_by_item=dict(zip(ITEMS, np.round(pw, 2).tolist())))
if __name__ == "__main__":
    jobs = [(d, g, r, s) for s in (1, -1) for d in (0.5, 1.0, 2.0, 3.0) for g in range(5) for r in range(40)]
    with Pool(4) as p: df = pd.DataFrame(p.map(one, jobs, chunksize=4))
    res = {}
    for d in (0.5, 1.0, 2.0, 3.0):
        res[str(d)] = summarise(df[df.sgn == 1], d)
        res[str(d)]["harder"] = summarise(df[df.sgn == -1], d)
        print(d, res[str(d)]["median_power"], res[str(d)]["items_ge_08"], round(res[str(d)]["typeI"], 4),
              "| harder", res[str(d)]["harder"]["median_power"], res[str(d)]["harder"]["items_ge_08"], round(res[str(d)]["harder"]["typeI"], 4))
    json.dump(res, open(os.path.join(RES, "a12_power25.json"), "w"), indent=2)
