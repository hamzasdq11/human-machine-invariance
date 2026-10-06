"""A1: anchor-free, model-level test of human-machine invariance.

Question per group g: does ANY normal ability distribution on the human scale,
N(mu, sigma^2) with the SAPA item parameters fixed, reproduce g's 25 item
proportions? No anchors are chosen, no item is assumed invariant, and the test
treats each model as the unit, so clustering is not an issue.

  Statistic: G^2 of the fitted item proportions against the 25 observed ones
             (df = 23 for free mu, sigma; df = 24 for a single point, sigma = 0).
  Calibration: parametric bootstrap that simulates whole respondents with their
             observed missingness, then refits, so within-respondent dependence
             and the boundary at sigma = 0 are both reproduced.
  Effect size: RMSD between observed and fitted item proportions.

The same statistic is applied to human groups sized to give the same responses
per item as one model (~71): random groups and men-only groups.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import optimize
from multiprocessing import Pool
from dif.irt import gh_nodes, prob_2pl
from scripts.sapa_common import load_sapa, load_machine, load_reference_params, ITEMS, RES

A, B = load_reference_params()
XN, WN = gh_nodes(41)                 # standard normal nodes


def expected_p(mu, sigma):
    return WN @ prob_2pl(mu + sigma * XN, A, B)            # (I,)


def g2(y, n, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(y / (n * p)), 0.0)
        t2 = np.where(n - y > 0, (n - y) * np.log((n - y) / (n * (1 - p))), 0.0)
    return 2 * float((t1 + t2).sum())


def fit(y, n, free_sigma=True):
    ok = n > 0
    if free_sigma:
        f = lambda q: g2(y[ok], n[ok], expected_p(q[0], np.exp(np.clip(q[1], -8, 2)))[ok])
        best = None
        for s0 in (-3.0, -0.5, 0.0):                         # sigma ~ 0.05, 0.6, 1
            r = optimize.minimize(f, [0.0, s0], method="Nelder-Mead",
                                  options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 3000})
            if best is None or r.fun < best.fun: best = r
        mu, sig = best.x[0], float(np.exp(np.clip(best.x[1], -8, 2)))
        if sig < 1e-3: sig = 0.0
    else:
        f = lambda q: g2(y[ok], n[ok], expected_p(float(q), 0.0)[ok])
        best = optimize.minimize_scalar(f, bounds=(-6, 6), method="bounded")
        mu, sig = float(best.x), 0.0
    p = expected_p(mu, sig)
    return mu, sig, g2(y[ok], n[ok], p[ok]), p


def stats_of(R, free_sigma=True):
    M = ~np.isnan(R)
    y = np.nansum(R, 0); n = M.sum(0)
    mu, sig, G, p = fit(y, n, free_sigma)
    ok = n > 0
    rmsd = float(np.sqrt(np.mean((y[ok] / n[ok] - p[ok]) ** 2)))
    return dict(mu=mu, sigma=sig, G2=G, rmsd=rmsd, df=int(ok.sum()) - (2 if free_sigma else 1),
                n_resp=int(M.sum()), n_item_median=float(np.median(n)))


def boot(R, mu, sig, Bn, rng, free_sigma=True):
    M = ~np.isnan(R); out = []
    for _ in range(Bn):
        th = rng.normal(mu, sig, R.shape[0]) if sig > 0 else np.full(R.shape[0], mu)
        P = prob_2pl(th, A, B)
        Y = (rng.random(R.shape) < P).astype(float); Y[~M] = np.nan
        out.append(stats_of(Y, free_sigma)["G2"])
    return np.array(out)


def test_group(args):
    label, kind, R, seed, Bn = args
    rng = np.random.default_rng(seed); res = {"group": label, "kind": kind}
    for free in (True, False):
        s = stats_of(R, free)
        bs = boot(R, s["mu"], s["sigma"], Bn, rng, free)
        tag = "ms" if free else "pt"
        res.update({f"{tag}_{k}": v for k, v in s.items()})
        res[f"{tag}_p_boot"] = float((1 + (bs >= s["G2"]).sum()) / (Bn + 1))
        res[f"{tag}_G2_null_q95"] = float(np.quantile(bs, 0.95))
        res[f"{tag}_G2_null_mean"] = float(bs.mean())
    return res


if __name__ == "__main__":
    t = time.time()
    arm = sys.argv[1] if len(sys.argv) > 1 else "data/responses_free25.jsonl"
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    NHUM = int(os.environ.get("NHUM", 100)); BM = int(os.environ.get("BM", 1000))
    BH = int(os.environ.get("BH", 200))
    R_m, meta, _, _ = load_machine(arm)
    R_h, demo = load_sapa()
    jobs = []
    for k, m in enumerate(sorted(meta.model.unique())):
        jobs.append((m, "model", R_m[(meta.model == m).values], 100 + k, BM))
    # human group size chosen to match a model's responses per item
    target = np.median((~np.isnan(R_m[(meta.model == meta.model.iloc[0]).values])).sum(0))
    frac = np.mean(~np.isnan(R_h)); n_h = int(round(target / frac))
    rng = np.random.default_rng(7)
    men = np.where(demo.gender.values == "male")[0]
    for k in range(NHUM):
        jobs.append((f"random{k}", "human_random", R_h[rng.choice(len(R_h), n_h, replace=False)], 1000 + k, BH))
        jobs.append((f"men{k}", "human_men", R_h[rng.choice(men, n_h, replace=False)], 5000 + k, BH))
    with Pool(4) as p: out = p.map(test_group, jobs, chunksize=1)
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, f"a1_model_fit25_{tag}.csv"), index=False)
    print(f"human group size {n_h} respondents (target {target:.0f} responses/item); {time.time()-t:.0f}s")
    mods = df[df.kind == "model"]
    print(mods[["group", "ms_mu", "ms_sigma", "ms_G2", "ms_df", "ms_G2_null_q95", "ms_p_boot", "ms_rmsd",
                "pt_G2", "pt_p_boot", "pt_rmsd"]].round(4).to_string(index=False))
    for kind in ("human_random", "human_men"):
        h = df[df.kind == kind]
        print(f"{kind}: reject@0.05 {np.mean(h.ms_p_boot < 0.05):.3f}  reject@0.001 {np.mean(h.ms_p_boot <= 0.001):.3f}  "
              f"RMSD median {h.ms_rmsd.median():.4f} [95% {h.ms_rmsd.quantile(.95):.4f}] max {h.ms_rmsd.max():.4f}  "
              f"G2 median {h.ms_G2.median():.1f}  sigma median {h.ms_sigma.median():.2f}")
