"""C4: what the published run-level pipeline reports when invariance holds.

Eight synthetic humans, invariant by construction, sit at the eight models'
abilities (theta from each model's modal answers, all observed items, ML on
[-4, 4]). Each is "re-run" 75 times with that model's real missing cells, in
three ways that differ ONLY in how a model's runs depend on each other:

  A  fixed answer + flips : the human answers once; every run copies that answer
     vector and flips each answer with the model's own run-to-run rate f_m,
     where f_m(1 - f_m) equals the model's mean within-cell variance (as in a13).
  B  fresh answers        : every run is a new, independent 2PL draw
     (the stochastic-subject reading of the 2PL).
  C  fixed propensities   : the human has a propensity for each item, drawn from a
     Beta with mean P_i(theta_m) and a spread set so the mean within-cell
     variance equals the model's observed one; runs are draws from it.

In all three, E[response | theta] is the human 2PL, i.e. invariance holds. The
published pipeline (dif/anchored.py, unchanged) is applied to each data set.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
import common as C
from dif.anchored import anchored_dif

REPS = int(os.environ.get("REPS", 200))
d = C.load_long(os.path.join(C.ROOT, "data", "responses_free25.jsonl"))
R, meta = C.run_matrix(d)
models = sorted(meta.model.unique())
Pm, Nm, _ = C.model_props(d, models)
X = C.modal(Pm)
TH = np.array([C.theta_ml(X[m][~np.isnan(X[m])], np.where(~np.isnan(X[m]))[0]) for m in range(8)])
MASK = [~np.isnan(R[(meta.model == m).values]) for m in models]
PQ = np.nanmean(Pm * (1 - Pm), axis=1)                      # mean within-cell variance
FLIP = (1 - np.sqrt(1 - 4 * PQ)) / 2
PH = C.p2pl(TH)                                             # (8, 25) human 2PL at theta_m
RATIO = PQ / (PH * (1 - PH)).mean(1)
KAPPA = np.where(RATIO >= 1, np.inf, RATIO / np.maximum(1 - RATIO, 1e-12))


def make(kind, rng):
    blocks = []
    for m in range(8):
        M = MASK[m]; n = M.shape[0]
        if kind == "A":
            base = (rng.random(25) < PH[m]).astype(float)
            Y = np.tile(base, (n, 1))
            Y = np.where(rng.random(M.shape) < FLIP[m], 1 - Y, Y)
        elif kind == "B":
            Y = (rng.random(M.shape) < PH[m]).astype(float)
        else:
            k = KAPPA[m]
            p = PH[m] if np.isinf(k) else rng.beta(PH[m] * k, (1 - PH[m]) * k)
            Y = (rng.random(M.shape) < p).astype(float)
        Y[~M] = np.nan
        blocks.append(Y)
    return np.vstack(blocks)


def one(kind, rep):
    rng = np.random.default_rng({"A": 4100000, "B": 4200000, "C": 4300000}[kind] + rep)
    Rn = make(kind, rng)
    r = anchored_dif(np.zeros((2, 25)), Rn, item_names=C.ITEMS, ref_params=(C.A, C.B))
    return dict(kind=kind, rep=rep, rate=r.dif_rate, mu=r.focal_mu, sigma=r.focal_sigma,
                n_anchors=len(r.anchors), status=r.purify_status,
                flags="".join(str(int(v)) for v in r.table["flag_total"].values))


def summarise(df):
    summ = {}
    for kind, g in df.groupby("kind"):
        summ[kind] = dict(reps=len(g), rate_mean=g.rate.mean(), rate_median=g.rate.median(),
                          rate_q05=g.rate.quantile(.05), rate_q95=g.rate.quantile(.95),
                          share_ge_observed_084=float((g.rate >= 0.84 - 1e-9).mean()),
                          sigma_zero_share=float((g.sigma < 1e-3).mean()), mu_median=g.mu.median())
    summ["setup"] = dict(thetas=dict(zip(models, TH)), flip=dict(zip(models, FLIP)),
                         kappa=dict(zip(models, KAPPA)), observed_rate=0.84)
    return summ


if __name__ == "__main__":
    # usage: python3 c4_runlevel_nulls.py START END  -> results/c4_part_START.csv ;  python3 c4_runlevel_nulls.py merge
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        parts = sorted(f for f in os.listdir(C.OUT) if f.startswith("c4_part_"))
        df = pd.concat([pd.read_csv(os.path.join(C.OUT, f)) for f in parts]).drop_duplicates(["kind", "rep"])
        df = df.sort_values(["kind", "rep"])
        df.to_csv(os.path.join(C.OUT, "c4_runlevel_nulls.csv"), index=False)
        s = summarise(df); C.save_json(s, "c4_runlevel_nulls.json")
        print(json.dumps({k: v for k, v in s.items() if k != "setup"}, indent=1, default=str))
        sys.exit(0)
    start, end = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, REPS)
    out_path = os.path.join(C.OUT, f"c4_part_{start:03d}.csv")
    rows, t0 = [], time.time()
    print("thetas", dict(zip(map(C.short, models), TH.round(3))), "flip", FLIP.round(3), "kappa", np.round(KAPPA, 2), flush=True)
    for rep in range(start, end):
        for kind in ("A", "B", "C"):
            rows.append(one(kind, rep))
        if (rep - start) % 5 == 4:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"rep {rep+1}/{end}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("done", start, end, flush=True)
