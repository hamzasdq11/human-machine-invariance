"""C7: checks that need the SAPA human data (sapaICARData18aug2010thru20may2013.csv in data/).

1. Sample: respondents with >= 1 of the 25 text items; N, observed responses, per-item N.
2. Independent 2PL calibration (MML-EM written here, 41-point Gauss-Hermite) against results/sapa_calibration.csv.
3. Real-human groups of eight for the model-level item test: for each item, 8 SAPA respondents who
   answered it and >= 5 other items, theta by ML from their other items (as in a12). Flag rates with
   the exact Poisson-binomial test (comparator for the modal and single-run reductions) and with the
   stop-loss bound (comparator for the run-mean reduction). Also a stricter pool (>= 8 other items;
   SAPA forms give no item enough respondents with 10 or more).
4. Verbal / letter-series structure in people: domain-specific calibrations, then the latent
   correlation rho by marginal ML on everyone who answered both kinds of item (bootstrap SE).
5. The letter-series bundle recalibrated at the estimated rho (null C dependence, as in c3).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy import optimize
import common as C

rng = np.random.default_rng(7102026)
SAPA = os.path.join(C.ROOT, "data", "sapaICARData18aug2010thru20may2013.csv")
NREP = int(os.environ.get("NREP", 1000))
out = {}

# ---------------------------------------------------------------- 1. sample
d = pd.read_csv(SAPA, index_col=0)
X = d[C.ITEMS].to_numpy(float)
keep = ~np.isnan(X).all(1)
X, demo = X[keep], d.loc[keep, ["gender", "age"]].reset_index(drop=True)
O = ~np.isnan(X)
out["sample"] = dict(rows_in_file=int(len(d)), N=int(len(X)), observed=int(O.sum()),
                     per_item_N_min=int(O.sum(0).min()), per_item_N_max=int(O.sum(0).max()),
                     items_per_person_median=float(np.median(O.sum(1))))
print(out["sample"], flush=True)

# ---------------------------------------------------------------- 2. MML-EM calibration
xq, wq = hermegauss(41); wq = wq / wq.sum()


def mml(Xs, a0=None, b0=None, tol=1e-7, max_iter=2000):
    Os = ~np.isnan(Xs); X0 = np.where(Os, Xs, 0.0); Of = Os.astype(float)
    I = Xs.shape[1]
    p = np.clip(np.nansum(Xs, 0) / Os.sum(0), .02, .98)
    a = np.ones(I) if a0 is None else a0.copy()
    b = -np.log(p / (1 - p)) / 1.7 if b0 is None else b0.copy()
    prev = -np.inf
    for it in range(max_iter):
        Z = a * (xq[:, None] - b)                                  # (Q, I)
        lp, lq = -np.logaddexp(0, -Z), -np.logaddexp(0, Z)
        LL = X0 @ lp.T + (Of - X0) @ lq.T                          # (N, Q)
        m = LL.max(1, keepdims=True)
        post = np.exp(LL - m) * wq
        s = post.sum(1, keepdims=True)
        ll = float((m[:, 0] + np.log(s[:, 0])).sum())
        post /= s
        n = post.T @ Of; r = post.T @ X0                           # (Q, I)
        for _ in range(3):                                         # Newton on (a, b) per item, vectorised
            Z = a * (xq[:, None] - b); P = 1 / (1 + np.exp(-Z))
            res = r - n * P; W = n * P * (1 - P)
            d_ = xq[:, None] - b
            ga, gb = (res * d_).sum(0), (-a * res).sum(0)
            haa = -(W * d_ ** 2).sum(0); hbb = -(W * a ** 2).sum(0)
            hab = (W * a * d_).sum(0) - res.sum(0)
            det = haa * hbb - hab ** 2
            da = (hbb * ga - hab * gb) / det; db = (haa * gb - hab * ga) / det
            a = np.clip(a - np.clip(da, -.5, .5), 0.05, 6); b = np.clip(b - np.clip(db, -.5, .5), -6, 6)
        if abs(ll - prev) < tol * (1 + abs(prev)):
            break
        prev = ll
    return a, b, ll, it


t0 = time.time()
a_hat, b_hat, ll, iters = mml(X)
ref = pd.read_csv(os.path.join(C.ROOT, "results", "sapa_calibration.csv")).set_index("item").loc[C.ITEMS]
cal = pd.DataFrame(dict(item=C.ITEMS, a_ref=ref.a.values, a_here=a_hat, b_ref=ref.b.values, b_here=b_hat,
                        N=O.sum(0), p=np.nanmean(X, 0)))
cal.to_csv(os.path.join(C.OUT, "c7_calibration_check.csv"), index=False)
out["calibration"] = dict(iterations=int(iters), seconds=round(time.time() - t0, 1),
                          max_abs_diff_a=float(np.abs(cal.a_here - cal.a_ref).max()),
                          max_abs_diff_b=float(np.abs(cal.b_here - cal.b_ref).max()),
                          a_range=[float(a_hat.min()), float(a_hat.max())], b_range=[float(b_hat.min()), float(b_hat.max())])
print(out["calibration"], flush=True)

# ---------------------------------------------------------------- 3. real-human groups of eight
G = np.linspace(-4, 4, 1601)
ZG = C.A * (G[:, None] - C.B)
LPG, LQG = -np.logaddexp(0, -ZG), -np.logaddexp(0, ZG)


def theta_rows(M):
    """ML theta for each row of M (0/1/NaN), bounded [-4, 4]: grid search + Newton."""
    Om = ~np.isnan(M); M0 = np.where(Om, M, 0.0)
    LL = M0 @ LPG.T + (Om - M0) @ LQG.T
    th = G[LL.argmax(1)]
    for _ in range(6):
        P = 1 / (1 + np.exp(-C.A * (th[:, None] - C.B)))
        sc = (Om * C.A * (M0 - P)).sum(1); inf = (Om * C.A ** 2 * P * (1 - P)).sum(1)
        th = np.clip(th + np.clip(np.where(inf > 0, sc / np.maximum(inf, 1e-12), 0), -.5, .5), -4, 4)
    return th


nit = O.sum(1)
pools = {"ge5_other": nit >= 6, "ge8_other": nit >= 9}
groups = {}
for pname, pmask in pools.items():
    cand = [np.where(pmask & O[:, i])[0] for i in range(25)]
    F_pb, F_sl = np.zeros((NREP, 25), bool), np.zeros((NREP, 25), bool)
    for k in range(NREP):
        rows, tgt = [], []
        for i in range(25):
            pick = rng.choice(cand[i], 8, replace=False)
            rows.append(pick); tgt += [i] * 8
        rows = np.concatenate(rows); tgt = np.array(tgt)
        M = X[rows].copy(); y = M[np.arange(len(rows)), tgt].copy(); M[np.arange(len(rows)), tgt] = np.nan
        th = theta_rows(M)
        pr = 1 / (1 + np.exp(-C.A[tgt] * (th - C.B[tgt])))
        p_pb, p_sl = np.ones(25), np.ones(25)
        for i in range(25):
            s = slice(8 * i, 8 * i + 8)
            p_pb[i] = C.pb_p_two_sided(pr[s], int(y[s].sum()))
            p_sl[i] = C.stoploss_p_two_sided(pr[s], float(y[s].sum()))
        F_pb[k], F_sl[k] = C.bh(p_pb), C.bh(p_sl)
    rp, rs = F_pb.mean(1), F_sl.mean(1)
    obs = dict(modal=0.20, single_mean=0.12, run_mean=0.08)
    groups[pname] = dict(
        pool_size_median=int(np.median([len(c) for c in cand])),
        exact_pb=dict(mean=float(rp.mean()), q95=float(np.quantile(rp, .95)), max=float(rp.max()),
                      p_ge_modal_020=float((1 + (rp >= 0.20 - 1e-9).sum()) / (NREP + 1)),
                      p_ge_single_012=float((1 + (rp >= 0.12 - 1e-9).sum()) / (NREP + 1)),
                      item_rate=dict(zip(C.ITEMS, np.round(F_pb.mean(0), 4)))),
        stoploss=dict(mean=float(rs.mean()), q95=float(np.quantile(rs, .95)), max=float(rs.max()),
                      p_ge_runmean_008=float((1 + (rs >= 0.08 - 1e-9).sum()) / (NREP + 1))))
    print(pname, {k: v for k, v in groups[pname].items() if k != "exact_pb"},
          {k: v for k, v in groups[pname]["exact_pb"].items() if k != "item_rate"}, flush=True)
out["real_groups_of_eight"] = groups

# ---------------------------------------------------------------- 4. verbal / letter-series correlation
def domain_rho(dA, dB, label, nboot=200):
    XA, XB = X[:, dA], X[:, dB]
    aA, bA, _, _ = mml(XA[~np.isnan(XA).all(1)])
    aB, bB, _, _ = mml(XB[~np.isnan(XB).all(1)])
    both = (~np.isnan(XA).all(1)) & (~np.isnan(XB).all(1))
    Gq = np.linspace(-6, 6, 81)

    def lik(Xd, a, b):
        Od = ~np.isnan(Xd); X0 = np.where(Od, Xd, 0.0)
        Z = a * (Gq[:, None] - b)
        LL = X0 @ (-np.logaddexp(0, -Z)).T + (Od - X0) @ (-np.logaddexp(0, Z)).T
        return np.exp(LL - LL.max(1, keepdims=True)), LL.max(1)

    LA, mA = lik(XA[both], aA, bA); LB, mB = lik(XB[both], aB, bB)
    g1, g2 = np.meshgrid(Gq, Gq, indexing="ij")

    def W(rho):
        w = np.exp(-(g1 ** 2 - 2 * rho * g1 * g2 + g2 ** 2) / (2 * (1 - rho ** 2)))
        return w / w.sum()

    def nll(rho, idx=None):
        LAi, LBi = (LA, LB) if idx is None else (LA[idx], LB[idx])
        return -np.log(((LAi @ W(rho)) * LBi).sum(1)).sum()

    r = optimize.minimize_scalar(nll, bounds=(-0.98, 0.98), method="bounded", options={"xatol": 1e-5})
    boots = []
    n = int(both.sum())
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        boots.append(optimize.minimize_scalar(lambda q: nll(q, idx), bounds=(-0.98, 0.98), method="bounded",
                                              options={"xatol": 1e-4}).x)
    res = dict(label=label, persons_with_both=n, rho=float(r.x), se_boot=float(np.std(boots)),
               ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))])
    print(res, flush=True)
    return res


t0 = time.time()
out["rho"] = dict(VR_letter=domain_rho(C.VR, C.LETTER, "VR vs letter series"),
                  VR_LN=domain_rho(C.VR, C.LN, "VR vs all LN"),
                  VR_number=domain_rho(C.VR, C.NUMBER, "VR vs number series", nboot=100))
print("rho seconds", round(time.time() - t0), flush=True)
C.save_json(out, "c7_humans.json")

# ---------------------------------------------------------------- 5. letter bundle at the estimated rho
import c3_bundles as B3
res5 = {}
for path, label, nrep in ((os.path.join(C.ROOT, "data", "responses_free25.jsonl"), "main", 2000),
                          (os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"), "temp0", 1000)):
    dd = C.load_long(path); R, meta = C.run_matrix(dd); models = sorted(meta.model.unique())
    masks = [~np.isnan(R[(meta.model == m).values]) for m in models]
    P, _, _ = C.model_props(dd, models)
    thV = np.array([C.theta_ml(P[k, C.VR], C.VR) for k in range(8)])
    PQ = np.nanmean(P * (1 - P), 1)
    obs_p = {"mean": B3.bundle_stat(P, C.LETTER, C.VR, True)[2],
             "modal": B3.bundle_stat(C.modal(P), C.LETTER, C.VR, False)[2]}
    for rho_key in ("rho", "ci_low"):
        rho = out["rho"]["VR_letter"]["rho"] if rho_key == "rho" else out["rho"]["VR_letter"]["ci95"][0]
        pv = {"mean": [], "modal": []}
        for rep in range(nrep):
            Rn = []
            for k in range(8):
                th = np.full(25, thV[k]); th[C.LN] = thV[k] + rng.normal(0, np.sqrt(2 * (1 - rho)))
                ph = 1 / (1 + np.exp(-C.A * (th - C.B))); M = masks[k]
                r_ = PQ[k] / (ph * (1 - ph)).mean(); kap = np.inf if r_ >= 1 else r_ / (1 - r_)
                pm_ = ph if np.isinf(kap) else rng.beta(ph * kap, (1 - ph) * kap)
                Y = (rng.random(M.shape) < pm_).astype(float); Y[~M] = np.nan; Rn.append(Y)
            mean_ = np.array([np.nanmean(r, 0) for r in Rn])
            pv["mean"].append(B3.bundle_stat(mean_, C.LETTER, C.VR, True)[2])
            pv["modal"].append(B3.bundle_stat(C.modal(mean_), C.LETTER, C.VR, False)[2])
        res5[f"{label}_{rho_key}"] = dict(rho=rho, reps=nrep, **{
            f"calibrated_p_{k}": float((1 + np.sum(np.array(v) <= obs_p[k])) / (nrep + 1)) for k, v in pv.items()},
            **{f"typeI_005_{k}": float(np.mean(np.array(v) < .05)) for k, v in pv.items()})
        print(label, rho_key, res5[f"{label}_{rho_key}"], flush=True)
out["letter_bundle_at_rho"] = res5
C.save_json(out, "c7_humans.json")
print("done", flush=True)
