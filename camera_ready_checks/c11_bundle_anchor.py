"""C11: robustness grid for the exploratory letter-series bundle test (Tier 2).

The bundle test (c3) compares each model's letter-series total with what its verbal ability predicts.
Its calibrated p-value depends on three choices, each varied here:

  run dependence  the three run-dependence nulls of c2/c3: fixed (one 2PL answer vector copied to every
                  run), fresh (independent runs), matched (Beta propensities matched to each model's
                  run-to-run variance)
  dimensionality  1-D: letter-series ability equals verbal ability (invariance across domains);
                  2-D: letter-series ability = verbal ability + N(0, 2(1 - rho_hat)), centred on the verbal
                  ability, with the spread of the difference between two human abilities correlated rho_hat
                  (SAPA latent correlation between verbal and letter-series ability, c7)
  verbal anchor   all 16 VR items; VR minus VR.36 and VR.39 (flagged under every reduction); VR minus
                  VR.26, VR.36, VR.39 and VR.42 (the verbal items flagged under any reduction)

Statistic: two-sided bundle p-value (run mean: stop-loss bound; modal: exact Poisson-binomial).
Calibrated p = (1 + k) / (n + 1), k = null data sets with a p-value at most the observed one.
The whole LN subscale (9 items, all-VR anchor) is run on the same grid for comparison, its 2-D null at people's
latent correlation between verbal and LN ability (0.79).
None of these p-values accounts for the letter-series bundle having been chosen after inspecting Table IX.

Centring: the 2-D null centres letter-series ability on verbal ability. Regressing it towards the human mean
instead would raise the expected letter-series ability of these below-average models; one-sided, that makes
the observed deficit rarer still, but with the two-sided statistic most null data sets would then show an
excess (an independent check found a calibrated two-sided p of about 0.08; that figure evaluates a statistic
centred on verbal ability under a null centred elsewhere, and under that null none of 4,000 data sets showed a
deficit as large as the one observed). The centred null is the reference
for invariance across the two domains, not a conservative bound. A second caveat from the same check: rho_hat
comes from per-domain calibrations whose letter-series slopes are about 1.2 times the 25-item ones; generating
the null on those scales gave p = 0.019 instead of about 0.012 (2-D, matched).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common as C

rng = np.random.default_rng(11102026)
NREP = int(os.environ.get("NREP", 4000))
c7 = json.load(open(os.path.join(C.OUT, "c7_humans.json")))
RHO = c7["rho"]["VR_letter"]["rho"]       # verbal vs letter series (0.73)
RHO_LN = c7["rho"]["VR_LN"]["rho"]        # verbal vs the whole LN subscale (0.79), for the comparison row


def bundle_stat(X, bundle, base, stoploss):                 # as in c3_bundles.py
    probs, s = [], 0.0
    for m in range(X.shape[0]):
        o = base[~np.isnan(X[m, base])]
        th = C.theta_ml(X[m, o], o)
        for i in bundle:
            if not np.isnan(X[m, i]):
                probs.append(1 / (1 + np.exp(-C.A[i] * (th - C.B[i]))))
                s += X[m, i]
    probs = np.array(probs)
    p = C.stoploss_p_two_sided(probs, s) if stoploss else C.pb_p_two_sided(probs, int(round(s)))
    return s, probs.sum(), p


def simulate(thV, rho, kind, masks, PQ):
    Rn = []
    for k in range(8):
        th = np.full(25, thV[k])
        if rho < 1:
            th[C.LN] = thV[k] + rng.normal(0, np.sqrt(2 * (1 - rho)))
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


ANCHORS = {"VR16": [], "VRminus2": ["VR.36", "VR.39"], "VRminus4": ["VR.26", "VR.36", "VR.39", "VR.42"]}
out = dict(rho=RHO, rho_LN=RHO_LN, reps_main=NREP, reps_temp0=NREP // 2)
for path, label, nrep in ((os.path.join(C.ROOT, "data", "responses_free25.jsonl"), "main", NREP),
                          (os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"), "temp0", NREP // 2)):
    dd = C.load_long(path); R, meta = C.run_matrix(dd); models = sorted(meta.model.unique())
    masks = [~np.isnan(R[(meta.model == m).values]) for m in models]
    P, _, _ = C.model_props(dd, models)
    PQ = np.nanmean(P * (1 - P), 1)
    res = {}
    for name, drop in ANCHORS.items():
        base = np.array([i for i in C.VR if C.ITEMS[i] not in drop])
        thV = np.array([C.theta_ml(P[k, base], base) for k in range(8)])
        tests = {"letter_mean": (C.LETTER, True), "letter_modal": (C.LETTER, False)}
        if name == "VR16":
            tests["LN_mean"] = (C.LN, True)
        obs = {t: bundle_stat(P if sl else C.modal(P), bun, base, sl) for t, (bun, sl) in tests.items()}
        r = {t: dict(observed=float(o[0]), expected=float(o[1]), raw_p=float(o[2]), calibrated={}) for t, o in obs.items()}
        for dim, rho in (("1D", 1.0), ("2D", RHO)):
            for kind in ("fixed", "fresh", "matched"):
                t0 = time.time()
                pv = {t: [] for t in tests}
                for rep in range(nrep):
                    xp = simulate(thV, rho, kind, masks, PQ)
                    for t, (bun, sl) in tests.items():
                        if t == "LN_mean" and rho < 1:
                            continue
                        pv[t].append(bundle_stat(xp if sl else C.modal(xp), bun, base, sl)[2])
                    if "LN_mean" in tests and rho < 1:          # the LN row uses people's VR-LN correlation
                        xl = simulate(thV, RHO_LN, kind, masks, PQ)
                        pv["LN_mean"].append(bundle_stat(xl, C.LN, base, True)[2])
                for t in tests:
                    a = np.array(pv[t])
                    r[t]["calibrated"][f"{dim}_{kind}"] = dict(p=float((1 + np.sum(a <= obs[t][2])) / (nrep + 1)),
                                                              typeI_005=float(np.mean(a < .05)))
                print(label, name, dim, kind, {t: round(r[t]["calibrated"][f"{dim}_{kind}"]["p"], 4) for t in tests},
                      f"{time.time() - t0:.0f}s", flush=True)
        res[name] = dict(dropped=drop, n_anchor=int(len(base)), **r)
    out[label] = res
C.save_json(out, "c11_bundle_anchor.json")
