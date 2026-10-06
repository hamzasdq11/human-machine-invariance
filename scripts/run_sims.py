"""Validation studies A-C.  Produces the paper's simulation tables/figures."""
import sys, os, time, json, itertools
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, hashlib
from multiprocessing import Pool
from sim_core import run_cell

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
REPS = int(os.environ.get("REPS", 60))


def grid(name, varying, base, reps=REPS):
    cells = []
    for v in varying:
        for r in range(reps):
            kw = dict(base); kw.update(v)
            # deterministic across runs: Python's hash() is salted per process
            h = int(hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()[:6], 16)
            kw["seed"] = 100000 + (h % 10000) * 100 + r
            cells.append(kw)
    t0 = time.time()
    with Pool(2) as pool:
        rows = pool.map(run_cell, cells, chunksize=4)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, f"sim_{name}.csv")
    df.to_csv(path, index=False)
    print(f"[{name}] {len(df)} replicates in {time.time()-t0:.0f}s -> {path}")
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    base = dict(I=20, n_ref=2000, n_foc=400, prop_dif=0.20, b_shift=0.8, a_ratio=0.45)

    if which in ("all", "A"):
        # Study A: separation envelope -- how far apart can the groups be?
        grid("A_separation", [dict(foc_mu=m) for m in
                              (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)], base)
    if which in ("all", "B"):
        # Study B: focal sample size -- the regime-independence claim
        grid("B_focal_n", [dict(n_foc=n) for n in (30, 50, 100, 200, 400, 800, 1600)], base)
    if which in ("all", "C"):
        # Study C: null calibration -- Type-I error with no DIF present
        grid("C_null", [dict(prop_dif=0.0, foc_mu=m) for m in (-2.0, -1.0, 0.0)],
             base, reps=max(REPS, 100))
