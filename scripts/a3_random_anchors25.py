"""A3 (part 2): anchor-set sensitivity. 500 random anchor sets of 4-8 items,
no purification; every item is tested against the given anchors exactly as in
the published pipeline (BH at 0.05 on the 2-df LR test)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import stats
from multiprocessing import Pool
from dif.irt import estimate_group_dist, _prepare
from dif.anchored import _AnchorCache, _item_ll, _fit_studied_item, benjamini_hochberg
from scripts.sapa_common import load_machine, load_reference_params, ITEMS, RES
path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "main"
R_m, meta, _, _ = load_machine(path); A, B = load_reference_params(); Rf, Mf = _prepare(R_m)

def rate_with(anchors):
    mu, sg, _ = estimate_group_dist(R_m[:, anchors], A[anchors], B[anchors], n_points=41)
    c = _AnchorCache(Rf, Mf, A, B, anchors, mu, sg, 41); p = []
    for i in range(25):
        base = c.base_for(i)
        llc = c.marginal(base, _item_ll(Rf[:, i], Mf[:, i], c.X, A[i], B[i]))
        llab, _ = _fit_studied_item(c, base, Rf[:, i], Mf[:, i], A[i], B[i])
        p.append(stats.chi2.sf(max(2 * (llab - llc), 0), 2))
    rej = benjamini_hochberg(np.array(p))
    return float(rej.mean()), mu

def job(k):
    rng = np.random.default_rng(30000 + k); size = int(rng.integers(4, 9))
    anc = sorted(rng.choice(25, size, replace=False).tolist())
    r, mu = rate_with(anc)
    return dict(k=k, size=size, anchors=",".join(ITEMS[i] for i in anc), rate=r, mu=mu)

if __name__ == "__main__":
    with Pool(4) as p: out = p.map(job, range(500), chunksize=5)
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, f"a3_random_anchors25_{tag}.csv"), index=False)
    print(df.rate.describe().round(3).to_string())
    print("quantiles 2.5/50/97.5:", df.rate.quantile([.025, .5, .975]).round(2).tolist())
    print(df.groupby("size").rate.agg(["mean", "min", "max", "count"]).round(3).to_string())
