"""A4 (null calibration) and A9 (simulated power) at the observed structure.

Generator: 8 models, each a point on the human scale at its fitted theta
(sigma = 0, as estimated), 75 respondents per model with each model's REAL
missingness mask, responses drawn from the SAPA 2PL. The published pipeline is
then run unchanged (fixed reference parameters, purification, BH at 0.05).

  null  : no item altered -> how much DIF does the pipeline report on its own?
  unif  : 5 items at a time get b_focal = b - 0.5 (the paper's fixed effect size)
  nonu  : 5 items at a time get a_focal = a - 0.4
Items are rotated through 5 fixed groups (i % 5), as in the validation studies.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import optimize
from multiprocessing import Pool
from dif.anchored import anchored_dif
from dif.irt import prob_2pl
from scripts.sapa_common import load_machine, load_reference_params, ITEMS, RES

A, B = load_reference_params()
R_m, meta, _, _ = load_machine("data/responses_free25.jsonl")
MODELS = sorted(meta.model.unique())
MASKS = [~np.isnan(R_m[(meta.model == m).values]) for m in MODELS]

def theta_hat(R):
    M = ~np.isnan(R); y = np.nansum(R, 0); n = M.sum(0)
    def nll(t):
        p = prob_2pl([t], A, B)[0]
        return -(y * np.log(p) + (n - y) * np.log1p(-p)).sum()
    return float(optimize.minimize_scalar(nll, bounds=(-6, 6), method="bounded").x)

THETA = np.array([theta_hat(R_m[(meta.model == m).values]) for m in MODELS])

def simulate(rng, a_f, b_f):
    blocks = []
    for th, M in zip(THETA, MASKS):
        P = prob_2pl(np.full(M.shape[0], th), a_f, b_f)
        Y = (rng.random(M.shape) < P).astype(float); Y[~M] = np.nan
        blocks.append(Y)
    return np.vstack(blocks)

def one(job):
    mode, grp, rep = job
    rng = np.random.default_rng({"null": 0, "unif": 1, "nonu": 2}[mode] * 100000 + (grp + 1) * 1000 + rep)
    a_f, b_f = A.copy(), B.copy(); inj = np.zeros(25, bool)
    if mode != "null":
        inj = (np.arange(25) % 5) == grp
        if mode == "unif": b_f[inj] -= 0.5
        else: a_f[inj] = np.maximum(a_f[inj] - 0.4, 0.1)
    R = simulate(rng, a_f, b_f)
    r = anchored_dif(np.zeros((2, 25)), R, item_names=ITEMS, ref_params=(A, B))
    f = r.table.flag_total.values
    return dict(mode=mode, grp=grp, rep=rep, rate=float(f.mean()),
                flags="".join("1" if x else "0" for x in f), inj="".join("1" if x else "0" for x in inj),
                typeI=float(f[~inj].mean()), power=float(f[inj].mean()) if inj.any() else np.nan,
                mu=r.focal_mu, sigma=r.focal_sigma, status=r.purify_status)

if __name__ == "__main__":
    which = sys.argv[1]; reps = int(sys.argv[2])
    t = time.time()
    print("theta per model:", dict(zip([m.split('/')[1] for m in MODELS], THETA.round(3))), flush=True)
    jobs = ([("null", -1, k) for k in range(reps)] if which == "null" else
            [(m, g, k) for m in ("unif", "nonu") for g in range(5) for k in range(reps)])
    with Pool(4) as p: out = p.map(one, jobs, chunksize=2)
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, f"sim_observed25_{which}.csv"), index=False)
    print(f"{which}: {len(df)} runs in {time.time()-t:.0f}s", flush=True)
