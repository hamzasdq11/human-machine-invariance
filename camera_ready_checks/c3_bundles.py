"""C3: differential bundle functioning with the model as the examinee.

Item-level tests with 8 models detect only ~3-logit shifts. A shift shared by a
bundle of items accumulates across items, so it can be detected at much smaller
sizes. Bundles:
  pre-specified : the two ICAR subscales, LN (9 items) and VR (16 items)
  exploratory   : the 7 LETTER series inside LN (LN.05-LN.58); the 2 number
                  series (LN.01, LN.03) are the rest of LN
For a bundle D, each model's ability comes from the items OUTSIDE D (for LN and
letter: the 16 VR items). Statistic: the bundle total over models and items,
against sum_m sum_{i in D} P_i(theta_m). Reductions: modal (exact
Poisson-binomial), mean proportion (stop-loss bound, valid for any dependence
among runs) and one random run.

Calibration:
  * the three run-dependence nulls of c2 (A0, B, C), unidimensional
  * a two-dimensional human null: theta_L = theta_V + N(0, 2(1 - rho)), i.e. the
    bundle and the rest measure abilities that differ as much as two human
    abilities correlated rho, centred on the verbal ability (no regression to the
    mean; for a two-sided test this is not conservative: see c11_bundle_anchor.py)
  * power: all bundle items shifted by d logits for every model
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import common as C

rng = np.random.default_rng(3102026)
NNULL = int(os.environ.get("NNULL", 2000))

BUNDLES = {"LN_given_VR": (C.LN, C.VR), "VR_given_LN": (C.VR, C.LN),
           "letter_given_VR": (C.LETTER, C.VR), "number_given_VR": (C.NUMBER, C.VR)}


def bundle_stat(X, bundle, base, stoploss):
    probs, s = [], 0.0
    for m in range(X.shape[0]):
        o = base[~np.isnan(X[m, base])]
        th = C.theta_ml(X[m, o], o)
        for i in bundle:
            if not np.isnan(X[m, i]):
                probs.append(1 / (1 + np.exp(-C.A[i] * (th - C.B[i]))))
                s += X[m, i]
    probs = np.array(probs)
    if stoploss:
        p = C.stoploss_p_two_sided(probs, s)
    else:
        p = C.pb_p_two_sided(probs, int(round(s)))
    return s, probs.sum(), p


def reduce_runs(Rlist, rng):
    mean = np.array([np.nanmean(r, 0) for r in Rlist])
    return C.modal(mean), mean, np.array([r[rng.integers(len(r))] for r in Rlist])


def all_stats(Xmod, Xmean, Xsingle, bundle, base):
    return (bundle_stat(Xmod, bundle, base, False)[2], bundle_stat(Xmean, bundle, base, True)[2],
            bundle_stat(Xsingle, bundle, base, False)[2])


if __name__ == "__main__":
    out = {}
    for path, label in ((os.path.join(C.ROOT, "data", "responses_free25.jsonl"), "main"),
                        (os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"), "temp0")):
        d = C.load_long(path)
        R, meta = C.run_matrix(d)
        models = sorted(meta.model.unique())
        Rl = [R[(meta.model == m).values] for m in models]
        masks = [~np.isnan(r) for r in Rl]
        P, _, _ = C.model_props(d, models)
        Xmod = C.modal(P)
        res = {}
        # observed, per bundle and reduction; single-run: distribution over 2000 draws
        for name, (bun, base) in BUNDLES.items():
            sm, Em, pm = bundle_stat(Xmod, bun, base, False)
            sp, Ep, pp = bundle_stat(P, bun, base, True)
            ps = [bundle_stat(np.array([r[rng.integers(len(r))] for r in Rl]), bun, base, False)[2] for _ in range(1000)]
            res[name] = dict(modal=dict(total=sm, expected=Em, p=pm),
                             mean=dict(total=sp, expected=Ep, p=pp),
                             single=dict(p_median=float(np.median(ps)), share_p_lt_005=float(np.mean(np.array(ps) < .05)),
                                         share_p_lt_001=float(np.mean(np.array(ps) < .001))))
        # per-model domain abilities (mean proportions)
        res["per_model"] = [dict(model=C.short(m), theta_VR=C.theta_ml(P[k, C.VR], C.VR, -6, 6),
                                 theta_LN=C.theta_ml(P[k, C.LN], C.LN, -6, 6),
                                 theta_letter=C.theta_ml(P[k, C.LETTER], C.LETTER, -6, 6),
                                 letter_mean=float(np.nanmean(P[k, C.LETTER])),
                                 letter_expected_from_VR=float(C.p2pl(C.theta_ml(P[k, C.VR], C.VR))[C.LETTER].mean()))
                            for k, m in enumerate(models)]
        # nulls: unidimensional run-dependence nulls (A0, B, C) at the models' VR-based thetas
        thV = np.array([C.theta_ml(P[k, C.VR], C.VR) for k in range(8)])
        PQ = np.nanmean(P * (1 - P), 1)
        cal = {}
        for kind in ("A0", "B", "C") + tuple(f"C_rho{r}" for r in (0.9, 0.8, 0.7, 0.6, 0.5)):
            rho = float(kind.split("rho")[1]) if "rho" in kind else 1.0
            pv = {k2: {nm: [] for nm in ("letter_given_VR", "LN_given_VR")} for k2 in ("modal", "mean", "single")}
            for rep in range(NNULL if label == "main" else NNULL // 4):
                Rn = []
                for k in range(8):
                    th = np.full(25, thV[k])
                    if rho < 1:
                        th[C.LN] = thV[k] + rng.normal(0, np.sqrt(2 * (1 - rho)))
                    ph = 1 / (1 + np.exp(-C.A * (th - C.B)))
                    M = masks[k]
                    if kind == "A0":
                        Y = np.tile((rng.random(25) < ph).astype(float), (M.shape[0], 1))
                    elif kind == "B":
                        Y = (rng.random(M.shape) < ph).astype(float)
                    else:
                        r_ = PQ[k] / (ph * (1 - ph)).mean()
                        kap = np.inf if r_ >= 1 else r_ / (1 - r_)
                        pm_ = ph if np.isinf(kap) else rng.beta(ph * kap, (1 - ph) * kap)
                        Y = (rng.random(M.shape) < pm_).astype(float)
                    Y[~M] = np.nan
                    Rn.append(Y)
                xm, xp, xs = reduce_runs(Rn, rng)
                for nm in ("letter_given_VR", "LN_given_VR"):
                    bun, base = BUNDLES[nm]
                    a_, b_, c_ = all_stats(xm, xp, xs, bun, base)
                    pv["modal"][nm].append(a_); pv["mean"][nm].append(b_); pv["single"][nm].append(c_)
            cal[kind] = {}
            for k2 in pv:
                for nm in pv[k2]:
                    arr = np.array(pv[k2][nm])
                    obs_p = res[nm][k2]["p"] if k2 != "single" else res[nm]["single"]["p_median"]
                    cal[kind][f"{k2}:{nm}"] = dict(typeI_005=float(np.mean(arr < .05)),
                                                   calibrated_p=float((1 + np.sum(arr <= obs_p)) / (len(arr) + 1)))
            print(label, kind, {k: (round(v["typeI_005"], 3), round(v["calibrated_p"], 4)) for k, v in cal[kind].items()}, flush=True)
        res["calibration"] = cal
        # power of the letter bundle (mean + stop-loss) vs a shift d on all 7 letter items, null C dependence
        pw = {}
        if label == "main":
            for dd in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                hits = {"bundle": 0, "any_item": 0}
                for rep in range(300):
                    Rn = []
                    for k in range(8):
                        bb = C.B.copy(); bb[C.LETTER] += dd
                        ph = 1 / (1 + np.exp(-C.A * (thV[k] - bb)))
                        M = masks[k]
                        r_ = PQ[k] / (ph * (1 - ph)).mean(); kap = np.inf if r_ >= 1 else r_ / (1 - r_)
                        pm_ = ph if np.isinf(kap) else rng.beta(ph * kap, (1 - ph) * kap)
                        Y = (rng.random(M.shape) < pm_).astype(float); Y[~M] = np.nan; Rn.append(Y)
                    xm, xp, xs = reduce_runs(Rn, rng)
                    hits["bundle"] += bundle_stat(xp, C.LETTER, C.VR, True)[2] < .05
                    hits["any_item"] += C.fast_item_test(xp, stoploss=True).flag.to_numpy()[C.LETTER].any()
                pw[str(dd)] = {k: v / 300 for k, v in hits.items()}
                print("power", dd, pw[str(dd)], flush=True)
        res["power_letter_shift"] = pw
        out[label] = res
        print(label, json.dumps({k: v for k, v in res.items() if k in BUNDLES}, indent=1, default=float), flush=True)
        print(pd.DataFrame(res["per_model"]).round(2).to_string(index=False), flush=True)
    C.save_json(out, "c3_bundles.json")
