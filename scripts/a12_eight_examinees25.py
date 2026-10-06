"""A12: item-level DIF with the MODEL as the unit (N = 8 examinees, one answer each).

Each model answers each item once: its modal answer over its runs (ties -> missing).
For item i, theta_m is estimated by ML from the model's OTHER items (leave-item-out),
so the item under test does not inform the ability it is judged against. Under
invariance the number of models answering item i correctly is Poisson-binomial
with success probabilities P_i(theta_m) from the SAPA 2PL. Two-sided exact p-value
(sum of outcome probabilities no larger than the observed one); BH at 0.05.

Calibration:
  sim    : 1,000 groups of 8 simulated 2PL humans at the models' own thetas
  real   : 1,000 groups of 8 REAL SAPA respondents per item, drawn from those who
           answered the item and >= 5 other items (theta from those other items)
  gender : the same with 8 real men, against the women-only calibration
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import optimize
from dif.irt import prob_2pl, fit_2pl_mml
from dif.anchored import benjamini_hochberg
from scripts.sapa_common import load_sapa, load_machine, load_reference_params, ITEMS, RES
A0, B0 = load_reference_params()

def mle(x, idx, A, B):
    x = np.asarray(x, float); a, b = A[idx], B[idx]
    # bounded ML; all-correct/all-wrong patterns land on the +/-4 bound (rare at >= 5 items)
    nll = lambda t: -(x * np.log(prob_2pl([t], a, b)[0]) + (1 - x) * np.log1p(-prob_2pl([t], a, b)[0])).sum()
    return float(optimize.minimize_scalar(nll, bounds=(-4, 4), method="bounded").x)

def poibin_p(probs, k):
    dist = np.array([1.0])
    for p in probs: dist = np.convolve(dist, [1 - p, p])
    return float(min(1.0, dist[dist <= dist[k] * (1 + 1e-9)].sum()))

def item_test(X, A, B):
    """X: (n_examinees, 25) 0/1/NaN. Returns per-item p, observed and expected counts."""
    out = []
    for i in range(25):
        rows = np.where(~np.isnan(X[:, i]))[0]; probs = []; k = 0
        for r in rows:
            other = np.where(~np.isnan(X[r]))[0]; other = other[other != i]
            th = mle(X[r, other], other, A, B)
            probs.append(prob_2pl([th], [A[i]], [B[i]])[0, 0]); k += int(X[r, i])
        out.append(dict(item=ITEMS[i], n=len(rows), k=k, expected=float(np.sum(probs)), p=poibin_p(probs, k)))
    t = pd.DataFrame(out); t["flag"] = benjamini_hochberg(t.p.values); return t

if __name__ == "__main__":
    R_m, meta, _, _ = load_machine("data/responses_free25.jsonl")
    models = sorted(meta.model.unique())
    P = np.array([np.nanmean(R_m[(meta.model == m).values], 0) for m in models])
    X = np.where(np.abs(P - .5) < 1e-9, np.nan, (P > .5).astype(float))
    obs = item_test(X, A0, B0); obs.to_csv(os.path.join(RES, "a12_eight_examinees25.csv"), index=False)
    print(obs.round(4).to_string(index=False)); print("flagged", int(obs.flag.sum()), "rate", obs.flag.mean())
    T = pd.read_json("data/responses_free25_temp0.jsonl", lines=True)
    from scripts.sapa_common import HARNESS_TO_ICAR
    # temperature-0 single administration, prompt v1, seed 0: one literal answer per model
    t0 = T[(T.prompt_variant == "v1") & (T.seed == 0)].copy(); t0["icar"] = t0.item_id.map(HARNESS_TO_ICAR)
    t0["y"] = np.where(t0.parsable, t0.correct, np.nan)
    X0 = t0.pivot_table(index="model", columns="icar", values="y", aggfunc="first").reindex(index=models, columns=ITEMS).values
    o0 = item_test(X0.astype(float), A0, B0); o0.to_csv(os.path.join(RES, "a12_eight_examinees25_temp0_v1s0.csv"), index=False)
    print("temp0 v1 seed0 single answers: flagged", int(o0.flag.sum()), list(o0.item[o0.flag]))
    # calibration
    rng = np.random.default_rng(12)
    thetas = [mle(X[m][~np.isnan(X[m])], np.where(~np.isnan(X[m]))[0], A0, B0) for m in range(8)]
    sim = []
    for k in range(int(os.environ.get("NSIM", 1000))):
        Xs = (rng.random((8, 25)) < prob_2pl(thetas, A0, B0)).astype(float); Xs[np.isnan(X)] = np.nan
        sim.append(item_test(Xs, A0, B0).flag.mean())
    sim = np.array(sim)
    R_h, demo = load_sapa(); obsm = ~np.isnan(R_h); nit = obsm.sum(1)
    women = demo.gender.values == "female"; men = ~women
    fw = fit_2pl_mml(R_h[women], names=ITEMS, tol=1e-7)
    def real_groups(pool_mask, A, B, nrep):
        rates = []
        for k in range(nrep):
            Xr = np.full((8, 25), np.nan)
            # one draw of 8 humans per item: each item gets its own 8 respondents
            ps = []
            for i in range(25):
                cand = np.where(pool_mask & obsm[:, i] & (nit >= 6))[0]
                pick = rng.choice(cand, 8, replace=False); probs = []; kk = 0
                for j in pick:
                    other = np.where(obsm[j])[0]; other = other[other != i]
                    th = mle(R_h[j, other], other, A, B)
                    probs.append(prob_2pl([th], [A[i]], [B[i]])[0, 0]); kk += int(R_h[j, i])
                ps.append(poibin_p(probs, kk))
            rates.append(benjamini_hochberg(np.array(ps)).mean())
        return np.array(rates)
    NR = int(os.environ.get("NREAL", 300))
    real = real_groups(np.ones(len(R_h), bool), A0, B0, NR)
    gend = real_groups(men, fw.a, fw.b, NR)
    summ = dict(observed_rate=float(obs.flag.mean()), observed_flagged=list(obs.item[obs.flag]),
                temp0_single_rate=float(o0.flag.mean()), temp0_single_flagged=list(o0.item[o0.flag]),
                model_thetas=dict(zip(models, np.round(thetas, 3).tolist())),
                sim_null_mean=float(sim.mean()), sim_null_q95=float(np.quantile(sim, .95)), sim_null_max=float(sim.max()),
                real_random8_mean=float(real.mean()), real_random8_q95=float(np.quantile(real, .95)), real_random8_max=float(real.max()),
                real_men8_mean=float(gend.mean()), real_men8_q95=float(np.quantile(gend, .95)), real_men8_max=float(gend.max()),
                p_observed_vs_real_random=float((1 + (real >= obs.flag.mean()).sum()) / (len(real) + 1)),
                p_observed_vs_real_men=float((1 + (gend >= obs.flag.mean()).sum()) / (len(gend) + 1)))
    json.dump(summ, open(os.path.join(RES, "a12_eight_examinees25.json"), "w"), indent=2, default=float)
    print(json.dumps(summ, indent=2, default=float))
