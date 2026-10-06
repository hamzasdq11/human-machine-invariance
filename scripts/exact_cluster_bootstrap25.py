"""A5: exact cluster bootstrap over the 8 models.

Enumerates all C(15,8) = 6,435 distinct multisets of 8 models drawn with
replacement, runs the full pipeline (purification included) on each, and
weights each by its multinomial probability under the ordinary bootstrap.
"""
import os, sys, json, time, itertools, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from dif.anchored import anchored_dif
from scripts.sapa_common import load_machine, load_reference_params, ITEMS, RES
path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "main"
R_m, meta, _, _ = load_machine(path); a, b = load_reference_params()
models = sorted(meta.model.unique()); K = len(models)
blocks = {m: R_m[(meta.model == m).values] for m in models}

def run(counts):
    R = np.vstack([blocks[m] for m, c in zip(models, counts) for _ in range(c)])
    r = anchored_dif(np.zeros((2, 25)), R, item_names=ITEMS, ref_params=(a, b))
    w = math.factorial(K) / np.prod([math.factorial(c) for c in counts]) / K ** K
    return dict(counts="".join(map(str, counts)), weight=w, rate=r.dif_rate,
                n_flag=int(r.table.flag_total.sum()), status=r.purify_status,
                distinct=int(sum(c > 0 for c in counts)))

def multisets():
    for comb in itertools.combinations_with_replacement(range(K), K):
        yield tuple(comb.count(i) for i in range(K))

if __name__ == "__main__":
    jobs = list(multisets()); assert len(jobs) == 6435
    t = time.time()
    with Pool(4) as p: out = p.map(run, jobs, chunksize=8)
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, f"exact_bootstrap25_{tag}.csv"), index=False)
    assert abs(df.weight.sum() - 1) < 1e-9
    d = df.sort_values("rate"); cw = d.weight.cumsum().values
    q = lambda p: float(d.rate.values[np.searchsorted(cw, p)])
    s = dict(n_resamples=len(df), mean=float((df.rate * df.weight).sum()),
             ci95=[q(0.025), q(0.975)], median=q(0.5),
             p_rate_below_0_5=float(df.weight[df.rate < 0.5].sum()),
             min_rate=float(df.rate.min()), seconds=time.time() - t)
    json.dump(s, open(os.path.join(RES, f"exact_bootstrap25_{tag}.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))
