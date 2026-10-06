"""A6: human controls in the machine analysis's exact regime (one design, N = 600).

The machine analysis uses item parameters fixed from a large human calibration and
a focal group of 600 that played no part in that calibration. The controls now
match it:
  placebo: per replicate, 600 random respondents are held out as the focal group
           and the 2PL is recalibrated on the remaining 94,566.
  gender : the reference is every woman (calibrated once, N(0,1) in women); the
           focal group is 600 random men, who are never in the calibration.
200 replicates each; DIF flagged by the published pipeline unchanged.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from dif.anchored import anchored_dif
from dif.irt import fit_2pl_mml
from scripts.sapa_common import load_sapa, ITEMS, RES

R, demo = load_sapa(); g = demo.gender.values
NF = 600; REPS = int(os.environ.get("REPS", 200))
women = np.where(g == "female")[0]; men = np.where(g == "male")[0]
_FIT_W = None

def one(job):
    global _FIT_W
    contrast, rep = job
    rng = np.random.default_rng((1 if contrast == "gender" else 2) * 100000 + rep)
    if contrast == "gender":
        if _FIT_W is None:
            f = fit_2pl_mml(R[women], names=ITEMS, tol=1e-7); _FIT_W = (f.a, f.b)
        a, b = _FIT_W
        foc = rng.choice(men, NF, replace=False)
    else:
        foc = rng.choice(len(R), NF, replace=False)
        keep = np.ones(len(R), bool); keep[foc] = False
        f = fit_2pl_mml(R[keep], names=ITEMS, tol=1e-7); a, b = f.a, f.b
    r = anchored_dif(np.zeros((2, 25)), R[foc], item_names=ITEMS, ref_params=(a, b))
    return dict(contrast=contrast, rep=rep, rate=r.dif_rate, n_flag=int(r.table.flag_total.sum()),
                flags="".join("1" if x else "0" for x in r.table.flag_total.values),
                mu=r.focal_mu, sigma=r.focal_sigma, status=r.purify_status)

if __name__ == "__main__":
    t = time.time()
    jobs = [(c, k) for c in ("gender", "placebo") for k in range(REPS)]
    with Pool(4) as p: out = p.map(one, jobs, chunksize=4)
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, "a6_matched_controls25.csv"), index=False)
    print(f"{len(df)} runs in {time.time()-t:.0f}s")
