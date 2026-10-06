"""A11: the model as ONE examinee. Person fit of each model's modal response vector.

The anchored DIF test and A1 treat a model's 75 (prompt, seed) runs as 75 examinees.
They are re-runs of one entity, so the right comparison for a model is a single
human answering once. Here each model is reduced to its modal answer per item
(majority over its runs; exact ties dropped) and its fit to the human 2PL is
judged by Snijders' lz* person-fit statistic, with theta by ML.

Two references for lz*:
  sim   : 20,000 simulated single humans at the model's own theta-hat (25 items)
  real  : real SAPA respondents, on their own item sets (>= 6 items): for every
          such human, lz* of the human and lz* of the model on the SAME items.
          Reported: share of humans whose fit is worse than the model's.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import optimize
from dif.irt import prob_2pl
from scripts.sapa_common import load_sapa, load_machine, load_reference_params, ITEMS, RES
A, B = load_reference_params()

def mle(x, idx):
    a, b = A[idx], B[idx]
    if x.sum() == 0 or x.sum() == len(x): return None
    nll = lambda t: -(x * np.log(prob_2pl([t], a, b)[0]) + (1 - x) * np.log1p(-prob_2pl([t], a, b)[0])).sum()
    return float(optimize.minimize_scalar(nll, bounds=(-6, 6), method="bounded").x)

def lzstar(x, idx):
    """Snijders (2001) lz* with ML theta (corrected for estimation)."""
    th = mle(x, idx)
    if th is None: return None, None
    a, b = A[idx], B[idx]; P = prob_2pl([th], a, b)[0]; Q = 1 - P
    w = np.log(P / Q); r = a                     # r_i = P'_i/(P_i Q_i) = a_i for 2PL
    c = (P * Q * w * r).sum() / (P * Q * r * r).sum()     # c_n(theta); r0 = 0 for ML
    wt = w - c * r
    W = ((x - P) * wt).sum()
    V = (wt ** 2 * P * Q).sum()
    return float(W / np.sqrt(V)), th

if __name__ == "__main__":
    R_m, meta, _, _ = load_machine("data/responses_free25.jsonl")
    R_h, demo = load_sapa(); rng = np.random.default_rng(3)
    obs = ~np.isnan(R_h); n_items = obs.sum(1)
    elig = np.where(n_items >= 6)[0]
    human_lz = {}
    rows = []
    for m in sorted(meta.model.unique()):
        p = np.nanmean(R_m[(meta.model == m).values], 0)
        keep = np.where(np.abs(p - 0.5) > 1e-9)[0]; x = (p[keep] > 0.5).astype(float)
        lz, th = lzstar(x, keep)
        # simulated single humans at theta-hat on the same items
        P = prob_2pl([th], A[keep], B[keep])[0]
        sims = []
        for _ in range(20000):
            xs = (rng.random(len(keep)) < P).astype(float); v, _ = lzstar(xs, keep)
            if v is not None: sims.append(v)
        sims = np.array(sims)
        # real humans: same item sets
        worse, cnt, mod_lz, hum_lz = 0, 0, [], []
        for j in rng.choice(elig, 4000, replace=False):
            idx = np.where(obs[j])[0]; idx_m = np.intersect1d(idx, keep)
            if len(idx_m) < 6: continue
            if j not in human_lz: human_lz[j] = lzstar(R_h[j, idx_m], idx_m)[0] if len(idx_m) == len(idx) else lzstar(R_h[j, idx_m], idx_m)[0]
            hz = human_lz[j]; mz, _ = lzstar((p[idx_m] > 0.5).astype(float), idx_m)
            if hz is None or mz is None: continue
            cnt += 1; worse += hz < mz; mod_lz.append(mz); hum_lz.append(hz)
        rows.append(dict(model=m, items=len(keep), theta=th, lz_star=lz,
                         p_sim=float((sims <= lz).mean()), sim_q05=float(np.quantile(sims, .05)),
                         real_share_humans_worse=worse / cnt, real_pairs=cnt,
                         real_model_lz_median=float(np.median(mod_lz)), real_human_lz_median=float(np.median(hum_lz)),
                         guttman_hard_right_easy_wrong=int(((p > .5) & (B > th)).sum()), easy_wrong=int(((p < .5) & (B < th - 1)).sum())))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "a11_person_fit25.csv"), index=False)
    print(df.round(3).to_string(index=False))
