"""C11b: the letter-series bundle under a 2-D null that regresses towards the human mean (7 Oct 2026).

c11's 2-D null centres each model's letter-series ability on its verbal ability. In people, letter-series
ability given verbal ability regresses towards the population mean: with standard abilities and latent
correlation rho, theta_L | theta_V ~ N(rho * theta_V, 1 - rho^2). For these below-average models that null
expects more letter-series successes than verbal ability alone, so most null data sets show an excess, and the
two-sided statistic of c3/c11 (centred on verbal ability) is then calibrated against mostly excesses. This script
reports, for the run-mean bundle with the full verbal anchor:
  - the calibrated two-sided p (as in c11), and
  - the calibrated one-sided p for a deficit (share of null data sets with a letter-series total at most the
    observed one, relative to the verbal-ability expectation).
None of these p-values allows for the bundle having been chosen after inspecting Table IX.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common as C

rng = np.random.default_rng(7102026)
NREP = int(os.environ.get("NREP", 4000))
c7 = json.load(open(os.path.join(C.OUT, "c7_humans.json")))
RHO = c7["rho"]["VR_letter"]["rho"]


def bundle_stat(X, bundle, base):
    probs, s = [], 0.0
    for m in range(X.shape[0]):
        o = base[~np.isnan(X[m, base])]
        th = C.theta_ml(X[m, o], o)
        for i in bundle:
            if not np.isnan(X[m, i]):
                probs.append(1 / (1 + np.exp(-C.A[i] * (th - C.B[i]))))
                s += X[m, i]
    probs = np.array(probs)
    return s, probs.sum(), C.stoploss_p_two_sided(probs, s)


def simulate(thV, kind, masks, PQ):
    Rn = []
    for k in range(8):
        th = np.full(25, thV[k])
        th[C.LN] = RHO * thV[k] + rng.normal(0, np.sqrt(1 - RHO ** 2))     # regression towards the mean
        ph = 1 / (1 + np.exp(-C.A * (th - C.B))); M = masks[k]
        if kind == "fixed":
            Y = np.tile((rng.random(25) < ph).astype(float), (M.shape[0], 1))
        elif kind == "fresh":
            Y = (rng.random(M.shape) < ph).astype(float)
        else:
            r_ = PQ[k] / (ph * (1 - ph)).mean(); kap = np.inf if r_ >= 1 else r_ / (1 - r_)
            pm_ = ph if np.isinf(kap) else rng.beta(ph * kap, (1 - ph) * kap)
            Y = (rng.random(M.shape) < pm_).astype(float)
        Y[~M] = np.nan; Rn.append(Y)
    return np.array([np.nanmean(r, 0) for r in Rn])


dd = C.load_long(os.path.join(C.ROOT, "data", "responses_free25.jsonl"))
R, meta = C.run_matrix(dd); models = sorted(meta.model.unique())
masks = [~np.isnan(R[(meta.model == m).values]) for m in models]
P, _, _ = C.model_props(dd, models)
PQ = np.nanmean(P * (1 - P), 1)
base = np.array(C.VR)
thV = np.array([C.theta_ml(P[k, base], base) for k in range(8)])
s_obs, e_obs, p_obs = bundle_stat(P, C.LETTER, base)
out = dict(rho=RHO, nrep=NREP, observed=s_obs, expected_from_verbal=e_obs, raw_two_sided_p=p_obs, null={})
for kind in ("fixed", "fresh", "matched"):
    t0 = time.time(); ps, devs = [], []
    for _ in range(NREP):
        xp = simulate(thV, kind, masks, PQ)
        s, e, p = bundle_stat(xp, C.LETTER, base)
        ps.append(p); devs.append(s - e)
    ps, devs = np.array(ps), np.array(devs)
    out["null"][kind] = dict(p_two_sided=float((1 + np.sum(ps <= p_obs)) / (NREP + 1)),
                             p_one_sided_deficit=float((1 + np.sum(devs <= s_obs - e_obs)) / (NREP + 1)),
                             share_null_excess=float(np.mean(devs > 0)))
    print(kind, out["null"][kind], f"{time.time() - t0:.0f}s", flush=True)
C.save_json(out, "c11b_bundle_regression_null.json")
