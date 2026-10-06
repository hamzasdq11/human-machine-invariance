"""Design effects, variance components, dispersion checks, leave-one-model-out."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scripts.sapa_common import load_sapa, load_machine, load_reference_params, ITEMS, RES

path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "main"
R_m, meta, long, _ = load_machine(path)
R_h, _ = load_sapa()
models = meta["model"].values
out = {}

# raw gap variants
out["raw_gap_resp_means"] = float(np.nanmean(np.nanmean(R_m, 1)) - np.nanmean(np.nanmean(R_h, 1)))
out["raw_gap_cells"] = float(np.nanmean(R_m) - np.nanmean(R_h))
out["raw_gap_item_means"] = float(np.mean(np.nanmean(R_m, 0)) - np.mean(np.nanmean(R_h, 0)))

# ICC / design effect per item (one-way ANOVA estimator, clusters = models)
rows = []
for j, it in enumerate(ITEMS):
    y = R_m[:, j]; ok = ~np.isnan(y); y = y[ok]; g = models[ok]
    grp = pd.Series(y).groupby(g)
    k = grp.ngroups; n = len(y); ni = grp.size().values; means = grp.mean().values
    n0 = (n - (ni ** 2).sum() / n) / (k - 1)
    msb = (ni * (means - y.mean()) ** 2).sum() / (k - 1)
    msw = ((y - pd.Series(means, index=grp.mean().index).loc[g].values) ** 2).sum() / (n - k)
    s2b = max((msb - msw) / n0, 0.0)
    icc = s2b / (s2b + msw) if (s2b + msw) > 0 else 0.0
    deff = 1 + (n / k - 1) * icc
    rows.append(dict(item=it, n=n, p=y.mean(), icc=icc, deff=deff, neff=n / deff,
                     pmin=means.min(), pmax=means.max()))
cl = pd.DataFrame(rows).sort_values("deff", ascending=False)
cl.to_csv(os.path.join(RES, f"clustering25_{tag}.csv"), index=False)
out.update(icc_median=float(cl.icc.median()), icc_max=float(cl.icc.max()),
           deff_median=float(cl.deff.median()), deff_max=float(cl.deff.max()),
           neff_median=float(cl.neff.median()), n_median=float(cl.n.median()),
           neff_at_median_deff=float(cl.n.median() / cl.deff.median()),
           se_inflation=float(np.sqrt(cl.deff.median())))
print(cl.head(8).round(2).to_string(index=False))

# variance components of respondent score: incremental R^2 per facet
d = meta.copy(); d["score"] = np.nanmean(R_m, 1)
def r2(facs):
    X = pd.get_dummies(d[facs].astype(str), drop_first=True).astype(float)
    X.insert(0, "c", 1.0)
    beta, *_ = np.linalg.lstsq(X.values, d.score.values, rcond=None)
    res = d.score.values - X.values @ beta
    return 1 - res.var() / d.score.var(ddof=0)
F = ["model", "prompt_variant", "seed"]; full = r2(F)
out["varcomp"] = {f: float(full - r2([g for g in F if g != f])) for f in F}
out["varcomp"]["residual"] = float(1 - full)

# respondent score dispersion vs binomial
sc = np.nanmean(R_m, 1); nobs = (~np.isnan(R_m)).sum(1); pbar = np.nanmean(R_m)
out["score_sd_observed"] = float(sc.std(ddof=1))
out["score_sd_binomial"] = float(np.sqrt(np.mean(pbar * (1 - pbar) / nobs)))
# The binomial at the pooled mean ignores that items differ in difficulty. The right reference is
# Poisson-binomial: each run answers its own observed items independently, (a) at the pooled item
# proportions, (b) at its own model's item proportions (adds the spread between models).
# Var(score) = mean over runs of Var(score_r) + variance over runs of E(score_r).  [camera-ready, Tier 0]
obs = ~np.isnan(R_m)
def _pb_sd(P):
    Pm = np.where(obs, P, np.nan)
    return float(np.sqrt((np.nansum(Pm * (1 - Pm), 1) / nobs ** 2).mean() + np.nanmean(Pm, 1).var(ddof=1)))
out["score_sd_poisson_binomial"] = _pb_sd(np.tile(np.nanmean(R_m, 0), (len(R_m), 1)))
_pm = {m: np.nanmean(R_m[models == m], 0) for m in np.unique(models)}
out["score_sd_poisson_binomial_models"] = _pb_sd(np.vstack([_pm[m] for m in models]))
# within (model,item) cell variance vs binomial expectation
ratios = []
for m in np.unique(models):
    sub = R_m[models == m]
    for j in range(len(ITEMS)):
        y = sub[:, j]; y = y[~np.isnan(y)]
        if len(y) > 1 and 0 < y.mean() < 1:
            ratios.append((y.var(ddof=1), y.mean() * (1 - y.mean())))
v = np.array(ratios); out["within_cell_var_ratio"] = float(v[:, 0].sum() / v[:, 1].sum())
# NB: for 0/1 data var(ddof=1) = n/(n-1) * p(1-p) exactly, so this ratio is ~1 + 1/(n-1) by construction
# and carries no information about dispersion. Kept for continuity; not cited in the camera-ready text.

# Within a model, do frames or seeds move the run score beyond independent answering?  [camera-ready, Tier 0]
# Null: every run answers each of its own observed items independently at its model's item rate
# (missing cells kept). One-way F within each model; p from B simulated data sets; BH across models.
_B = 2000; _rng = np.random.default_rng(20261005)
_P = np.vstack([_pm[m] for m in models])
_sims = np.empty((_B, len(R_m)))
for _b in range(_B):
    _Y = (_rng.random(R_m.shape) < _P).astype(float); _Y[~obs] = np.nan
    _sims[_b] = np.nanmean(_Y, 1)
def _F(score, idx, g):
    x, g = score[idx], g[idx]; lv = np.unique(g); gm = x.mean()
    ssb = sum((g == l).sum() * (x[g == l].mean() - gm) ** 2 for l in lv)
    ssw = sum(((x[g == l] - x[g == l].mean()) ** 2).sum() for l in lv)
    return (ssb / (len(lv) - 1)) / (ssw / (len(x) - len(lv)))
def _bh(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); k = np.where(p[o] <= q * np.arange(1, len(p) + 1) / len(p))[0]
    return 0 if len(k) == 0 else int(k.max() + 1)
wm = {}
for m in np.unique(models):
    idx = models == m; wm[m] = {}
    for fac in ("prompt_variant", "seed"):
        g = meta[fac].astype(str).values; Fo = _F(sc, idx, g)
        Fn = np.array([_F(_sims[b], idx, g) for b in range(_B)])
        wm[m][fac] = dict(F=float(Fo), p=float((1 + (Fn >= Fo).sum()) / (_B + 1)))
    fm = pd.Series(sc[idx]).groupby(meta["prompt_variant"].values[idx]).mean()
    wm[m]["frame_means"] = {k: float(v) for k, v in fm.items()}
out["within_model_tests"] = wm
out["n_models_prompt_bh"] = _bh([wm[m]["prompt_variant"]["p"] for m in wm])
out["n_models_seed_bh"] = _bh([wm[m]["seed"]["p"] for m in wm])
_rng_m = max(wm, key=lambda m: max(wm[m]["frame_means"].values()) - min(wm[m]["frame_means"].values()))
out["frame_range_max"] = dict(model=_rng_m, lo=min(wm[_rng_m]["frame_means"].values()), hi=max(wm[_rng_m]["frame_means"].values()))

# per-model accounting
acc = long.groupby("model").agg(n=("y", "size"), parse=("parsable", "mean"), acc=("y", "mean"))
out["per_model"] = acc.round(4).reset_index().to_dict(orient="records")
json.dump(out, open(os.path.join(RES, f"clustering25_{tag}.json"), "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "per_model"}, indent=2, default=float))
print(acc.round(3))
