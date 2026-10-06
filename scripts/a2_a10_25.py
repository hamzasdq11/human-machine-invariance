"""A2: anchor-free dispersion (tau) of item-location shifts; A10: coefficient alpha.

A2. For a group, fit N(mu, sigma) on the human scale (as in A1). Then for each item
solve the location shift d_i that makes the model-implied proportion equal the
observed one (continuity-corrected). The mean of d is absorbed by mu and is not
identified without anchors; its SPREAD is. tau^2 = between-item variance of d beyond
sampling error (DerSimonian-Laird), reported in logits.
A10. Alpha on the 8 items used by administration arms 1-3, from the pairwise
covariance matrix (machine group and SAPA alike), plus listwise for the machines.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import optimize
import scripts.a1_model_fit25 as A1
from dif.irt import prob_2pl
from scripts.sapa_common import load_sapa, load_machine, ITEMS, RES

XN, WN = A1.XN, A1.WN
def p_item(mu, sig, a, b):
    return float(WN @ prob_2pl(mu + sig * XN, [a], [b])[:, 0])

def tau(R):
    M = ~np.isnan(R); y = np.nansum(R, 0); n = M.sum(0)
    mu, sig, _, _ = A1.fit(y, n, True)
    d, se = [], []
    for i in range(25):
        if n[i] < 10: continue
        p = (y[i] + 0.5) / (n[i] + 1)
        f = lambda x: p_item(mu, sig, A1.A[i], A1.B[i] + x) - p
        di = optimize.brentq(f, -20, 20)
        h = 1e-3; slope = (p_item(mu, sig, A1.A[i], A1.B[i] + di + h) - p_item(mu, sig, A1.A[i], A1.B[i] + di - h)) / (2 * h)
        d.append(di); se.append(np.sqrt(p * (1 - p) / n[i]) / abs(slope))
    d, se = np.array(d), np.array(se); w = 1 / se ** 2
    dbar = (w * d).sum() / w.sum(); Q = (w * (d - dbar) ** 2).sum(); k = len(d)
    t2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    return dict(tau=float(np.sqrt(t2)), sd_robust=float(mad), Q=float(Q), k=k, mu=mu, sigma=sig)

def alpha_pairwise(X):
    C = pd.DataFrame(X).cov(min_periods=20).values; k = C.shape[0]
    return float(k / (k - 1) * (1 - np.trace(C) / C.sum()))

if __name__ == "__main__":
    R_m, meta, _, _ = load_machine("data/responses_free25.jsonl")
    R_h, demo = load_sapa(); rng = np.random.default_rng(5)
    rows = [dict(group=m, kind="model", **tau(R_m[(meta.model == m).values])) for m in sorted(meta.model.unique())]
    rows.append(dict(group="all 8 models", kind="machine_pooled", **tau(R_m)))
    men = np.where(demo.gender.values == "male")[0]
    for k in range(100):
        rows.append(dict(group=f"r{k}", kind="human_random_221", **tau(R_h[rng.choice(len(R_h), 221, replace=False)])))
        rows.append(dict(group=f"m{k}", kind="human_men_221", **tau(R_h[rng.choice(men, 221, replace=False)])))
    for k in range(50):
        rows.append(dict(group=f"R{k}", kind="human_random_600", **tau(R_h[rng.choice(len(R_h), 600, replace=False)])))
        rows.append(dict(group=f"M{k}", kind="human_men_600", **tau(R_h[rng.choice(men, 600, replace=False)])))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "a2_tau25.csv"), index=False)
    print(df[df.kind.isin(["model", "machine_pooled"])][["group", "tau", "sd_robust", "mu", "sigma"]].round(3).to_string(index=False))
    print(df[~df.kind.isin(["model", "machine_pooled"])].groupby("kind")[["tau", "sd_robust"]].describe(percentiles=[.5, .95]).round(3).T.to_string())
    eight = ["VR.04", "VR.16", "VR.17", "VR.19", "LN.07", "LN.33", "LN.34", "LN.58"]
    idx = [ITEMS.index(i) for i in eight]
    X = R_m[:, idx]; lw = X[~np.isnan(X).any(1)]
    a10 = dict(items=eight, machine_alpha_pairwise=alpha_pairwise(X),
               machine_alpha_listwise=alpha_pairwise(lw), machine_listwise_n=int(len(lw)),
               sapa_alpha_pairwise_8=alpha_pairwise(R_h[:, idx]), human_alpha_psychtools_published=0.766,
               machine_alpha_pairwise_25=alpha_pairwise(R_m), sapa_alpha_pairwise_25=alpha_pairwise(R_h))
    json.dump(a10, open(os.path.join(RES, "a10_alpha25.json"), "w"), indent=2)
    print(json.dumps(a10, indent=2))
