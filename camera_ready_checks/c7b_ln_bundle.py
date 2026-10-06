"""C7b: the pre-specified LN-subscale bundle recalibrated at the SAPA verbal-LN correlation (from c7)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common as C
import c3_bundles as B3

rng = np.random.default_rng(7202026)
c7 = json.load(open(os.path.join(C.OUT, "c7_humans.json")))
rho = c7["rho"]["VR_LN"]["rho"]
res = {}
for path, label, nrep in ((os.path.join(C.ROOT, "data", "responses_free25.jsonl"), "main", 2000),
                          (os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"), "temp0", 1000)):
    dd = C.load_long(path); R, meta = C.run_matrix(dd); models = sorted(meta.model.unique())
    masks = [~np.isnan(R[(meta.model == m).values]) for m in models]
    P, _, _ = C.model_props(dd, models)
    thV = np.array([C.theta_ml(P[k, C.VR], C.VR) for k in range(8)])
    PQ = np.nanmean(P * (1 - P), 1)
    obs = B3.bundle_stat(P, C.LN, C.VR, True)[2]
    pv = []
    for rep in range(nrep):
        Rn = []
        for k in range(8):
            th = np.full(25, thV[k]); th[C.LN] = thV[k] + rng.normal(0, np.sqrt(2 * (1 - rho)))
            ph = 1 / (1 + np.exp(-C.A * (th - C.B))); M = masks[k]
            r_ = PQ[k] / (ph * (1 - ph)).mean(); kap = np.inf if r_ >= 1 else r_ / (1 - r_)
            pm_ = ph if np.isinf(kap) else rng.beta(ph * kap, (1 - ph) * kap)
            Y = (rng.random(M.shape) < pm_).astype(float); Y[~M] = np.nan; Rn.append(Y)
        pv.append(B3.bundle_stat(np.array([np.nanmean(r, 0) for r in Rn]), C.LN, C.VR, True)[2])
    res[label] = dict(rho=rho, reps=nrep, observed_p=obs, calibrated_p=float((1 + np.sum(np.array(pv) <= obs)) / (nrep + 1)))
    print(label, res[label], flush=True)
c7["ln_bundle_at_rho"] = res
C.save_json(c7, "c7_humans.json")
