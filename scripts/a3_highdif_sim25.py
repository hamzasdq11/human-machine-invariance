"""A3 (part 3): does the published pipeline recover prevalence when most items carry DIF?
Observed structure (8 model points, real missingness); 20 or 22 of 25 items (0.80, 0.88)
get a difficulty shift of random sign and size 0.5-1.5 logits. 50 data sets each."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from dif.anchored import anchored_dif
import scripts.sim_observed25 as S
from scripts.sapa_common import ITEMS, RES

def one(job):
    k_dif, rep = job
    rng = np.random.default_rng(70000 + 100 * k_dif + rep)
    inj = np.zeros(25, bool); inj[rng.choice(25, k_dif, replace=False)] = True
    b_f = S.B.copy(); b_f[inj] += rng.choice([-1, 1], inj.sum()) * rng.uniform(0.5, 1.5, inj.sum())
    R = S.simulate(rng, S.A, b_f)
    r = anchored_dif(np.zeros((2, 25)), R, item_names=ITEMS, ref_params=(S.A, S.B))
    f = r.table.flag_total.values
    return dict(true=k_dif / 25, rate=f.mean(), power=f[inj].mean(), typeI=f[~inj].mean(), status=r.purify_status)

if __name__ == "__main__":
    with Pool(4) as p: out = p.map(one, [(k, r) for k in (20, 22) for r in range(50)], chunksize=2)
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, "a3_highdif_sim25.csv"), index=False)
    print(df.groupby("true").agg(rate=("rate", "mean"), rate_lo=("rate", lambda x: x.quantile(.05)),
          rate_hi=("rate", lambda x: x.quantile(.95)), power=("power", "mean"), typeI=("typeI", "mean")).round(3))
    print(df.groupby("true").status.value_counts())
