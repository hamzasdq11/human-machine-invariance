"""Human control quantities on the 25 SAPA items (published design, Table V).

Published design as best reconstructed: reference = random 20,000 respondents,
recalibrated per run; focal = N respondents held out of the reference.
  gender:  reference drawn from women, focal from men
  placebo: reference and focal both drawn at random from everyone
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from multiprocessing import Pool
from dif.anchored import anchored_dif
from scripts.sapa_common import load_sapa, ITEMS, RES

R, demo = load_sapa()
g = demo.gender.values
NREF = 20000

def one(args):
    contrast, nfoc, rep = args
    rng = np.random.default_rng(10_000 * nfoc + rep + (0 if contrast == "placebo" else 7))
    if contrast == "gender":
        ref = rng.choice(np.where(g == "female")[0], NREF, replace=False)
        foc = rng.choice(np.where(g == "male")[0], nfoc, replace=False)
    else:
        idx = rng.permutation(len(R)); ref, foc = idx[:NREF], idx[NREF:NREF + nfoc]
    r = anchored_dif(R[ref], R[foc], item_names=ITEMS)
    return dict(contrast=contrast, nfoc=nfoc, rep=rep, rate=r.dif_rate,
                mu=r.focal_mu, sigma=r.focal_sigma, status=r.purify_status)

if __name__ == "__main__":
    reps = int(os.environ.get("REPS", 40))
    jobs = [(c, n, k) for c in ("gender", "placebo") for n in (300, 500) for k in range(reps)]
    t = time.time()
    with Pool(4) as p: out = p.map(one, jobs, chunksize=2)
    import pandas as pd
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, "human_controls25.csv"), index=False)
    def ci(x, B=2000):
        rng = np.random.default_rng(0); bs = [rng.choice(x, len(x)).mean() for _ in range(B)]
        return np.percentile(bs, [2.5, 97.5])
    for (c, n), s in df.groupby(["contrast", "nfoc"]):
        lo, hi = ci(s.rate.values)
        print(f"{c:8s} N={n}: rate {s.rate.mean():.3f} [{lo:.3f},{hi:.3f}]  mu {s.mu.mean():+.3f}  runs {len(s)}")
    print(f"{time.time()-t:.0f}s")
