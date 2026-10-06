"""Model-specification robustness.
  domain : DIF and A1 within verbal (16 items) and within series (9 items) separately
  refboot: 50 nonparametric bootstrap recalibrations of SAPA -> DIF rate and A1 misfit
  guess  : reference model with a guessing floor c = 1/8 (8 options), recalibrated -> DIF and A1
python3 scripts/robust_model25.py domain|refboot|guess
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import optimize
MODE = sys.argv[1]
C = 0.125
if MODE == "guess":       # patch the 2PL response function to a 3PL with fixed floor, everywhere
    import dif.irt as irt, dif.anchored as anc
    _p2 = irt.prob_2pl
    def p3(theta, a, b): return np.clip(C + (1 - C) * _p2(theta, a, b), 1e-12, 1 - 1e-12)
    def fit_one(nq, rq, X, a0, b0):
        def nll(p):
            pr = p3(X, np.array([p[0]]), np.array([p[1]]))[:, 0]
            return -(rq * np.log(pr) + (nq - rq) * np.log1p(-pr)).sum()
        return optimize.minimize(nll, np.array([a0, b0]), method="L-BFGS-B",
                                 bounds=[(0.05, 6.0), (-6.0, 6.0)]).x
    irt.prob_2pl = p3; anc.prob_2pl = p3; irt._fit_one_item = fit_one
from dif.irt import fit_2pl_mml
from dif.anchored import anchored_dif
import scripts.a1_model_fit25 as A1
if MODE == "guess": A1.prob_2pl = p3
from scripts.sapa_common import load_sapa, load_machine, ITEMS, RES
from scripts.robust_runner25 import a1_one
from multiprocessing import Pool

R_m, meta, _, _ = load_machine("data/responses_free25.jsonl")
models = sorted(meta.model.unique())

def a1_models(Bn=500, cols=None):
    jobs = []
    for k, m in enumerate(models):
        R = R_m[(meta.model == m).values].copy()
        if cols is not None: R[:, [i for i in range(25) if i not in cols]] = np.nan
        jobs.append((m, R, 300 + k, Bn))
    with Pool(4) as p: return pd.DataFrame(p.map(a1_one, jobs))

out = {}
if MODE == "domain":
    for name, cols in (("verbal", list(range(16))), ("series", list(range(16, 25)))):
        r = anchored_dif(np.zeros((2, len(cols))), R_m[:, cols], item_names=[ITEMS[i] for i in cols],
                         ref_params=(A1.A[cols], A1.B[cols]))
        a = a1_models(cols=cols)
        out[name] = dict(n_items=len(cols), dif_rate=r.dif_rate, n_flag=int(r.table.flag_total.sum()),
                         invariant=[ITEMS[cols[i]] for i in np.where(~r.table.flag_total.values)[0]],
                         a1_reject=int((a.p <= 0.002).sum()), a1_df=int(a.df.iloc[0]),
                         rmsd_min=float(a.rmsd.min()), rmsd_max=float(a.rmsd.max()),
                         G2_over_q95_min=float((a.G2 / a.q95).min()))
elif MODE == "refboot":
    R_h, _ = load_sapa(); rows = []
    for k in range(int(os.environ.get("REPS", 50))):
        rng = np.random.default_rng(90000 + k)
        f = fit_2pl_mml(R_h[rng.integers(0, len(R_h), len(R_h))], names=ITEMS, tol=1e-7)
        r = anchored_dif(np.zeros((2, 25)), R_m, item_names=ITEMS, ref_params=(f.a, f.b))
        A1.A, A1.B = f.a, f.b
        g = [A1.stats_of(R_m[(meta.model == m).values], True) for m in models]
        rows.append(dict(rep=k, dif_rate=r.dif_rate, rmsd_min=min(x["rmsd"] for x in g),
                         G2_min=min(x["G2"] for x in g), a_sd=float(np.std(f.a)), b0=float(f.b[0])))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "robust_refboot25.csv"), index=False)
    out = dict(reps=len(d), dif_rate_min=float(d.dif_rate.min()), dif_rate_max=float(d.dif_rate.max()),
               dif_rate_values=d.dif_rate.value_counts().to_dict(), rmsd_min=float(d.rmsd_min.min()),
               G2_min=float(d.G2_min.min()))
elif MODE == "guess":
    R_h, _ = load_sapa()
    f = fit_2pl_mml(R_h, names=ITEMS, tol=1e-7)
    pd.DataFrame({"item": ITEMS, "a": f.a, "b": f.b}).to_csv(os.path.join(RES, "sapa_calibration_3pl_c125.csv"), index=False)
    r = anchored_dif(np.zeros((2, 25)), R_m, item_names=ITEMS, ref_params=(f.a, f.b))
    A1.A, A1.B = f.a, f.b
    a = a1_models()
    # human random groups under the same reference, to show the test stays calibrated
    rng = np.random.default_rng(11); hum = []
    for k in range(40):
        Rg = R_h[rng.choice(len(R_h), 221, replace=False)]
        s = A1.stats_of(Rg, True); bs = A1.boot(Rg, s["mu"], s["sigma"], 200, rng, True)
        hum.append(dict(p=float((1 + (bs >= s["G2"]).sum()) / 201), rmsd=s["rmsd"]))
    h = pd.DataFrame(hum)
    out = dict(c=C, em_iters=f.n_iter, loglik=f.loglik, dif_rate=r.dif_rate, n_flag=int(r.table.flag_total.sum()),
               mu=r.focal_mu, sigma=r.focal_sigma,
               invariant=[ITEMS[i] for i in np.where(~r.table.flag_total.values)[0]],
               a1_reject=int((a.p <= 0.002).sum()), rmsd_min=float(a.rmsd.min()), rmsd_max=float(a.rmsd.max()),
               G2_over_q95_min=float((a.G2 / a.q95).min()),
               human_reject_05=float((h.p < 0.05).mean()), human_rmsd_max=float(h.rmsd.max()))
    f2 = fit_2pl_mml.__wrapped__ if hasattr(fit_2pl_mml, "__wrapped__") else None
json.dump(out, open(os.path.join(RES, f"robust_{MODE}25.json"), "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
