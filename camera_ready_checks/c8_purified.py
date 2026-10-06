"""C8: is the model-level item test robust to DIF in the items that estimate ability? (Tier 1, A4)

The model-level test estimates each model's ability for item i from its other 24 items. If some
of those items function differently, the ability estimate absorbs part of their shift. This check
purifies it: ability for every test is re-estimated without the currently flagged items, the tests
are re-run, and the flagged set is updated until it no longer changes (or cycles).

Reductions as in c2: modal answer (exact Poisson-binomial), run mean (stop-loss bound), and one
random run per model (exact test; NDRAW draws). Both arms.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import common as C

rng = np.random.default_rng(8102026)
NDRAW = int(os.environ.get("NDRAW", 500))


def test_excluding(X, F, stoploss=False):
    """Item test with each examinee's ability for item i estimated from its observed items
    other than i and other than the set F (boolean mask over items)."""
    Xm = X.copy()
    Xm[:, F] = np.nan                       # never used for ability
    TH = C.loo_theta(Xm)                    # TH[m, i]: ability from observed non-F items except i
    PI = 1 / (1 + np.exp(-C.A * (TH - C.B)))
    rows = []
    for i in range(X.shape[1]):
        ex = ~np.isnan(X[:, i])
        probs = PI[ex, i]
        s = float(X[ex, i].sum())
        p = C.stoploss_p_two_sided(probs, s) if stoploss else C.pb_p_two_sided(probs, int(round(s)))
        rows.append((C.ITEMS[i], ex.sum(), s, probs.sum(), p))
    t = pd.DataFrame(rows, columns=["item", "n", "k", "expected", "p"])
    t["flag"] = C.bh(t["p"].to_numpy())
    return t


def purify(X, stoploss=False, maxit=20):
    F = np.zeros(X.shape[1], bool)
    seen, path = [], []
    for it in range(maxit):
        t = test_excluding(X, F, stoploss)
        Fn = t.flag.to_numpy()
        path.append([C.ITEMS[j] for j in np.where(Fn)[0]])
        key = Fn.tobytes()
        if np.array_equal(Fn, F):
            return t, path, "converged", it
        if key in seen:
            return t, path, "cycle", it
        seen.append(key)
        F = Fn
    return t, path, "maxit", maxit


out = {}
for path, label in ((os.path.join(C.ROOT, "data", "responses_free25.jsonl"), "main"),
                    (os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"), "temp0")):
    d = C.load_long(path)
    R, meta = C.run_matrix(d)
    models = sorted(meta.model.unique())
    P, _, _ = C.model_props(d, models)
    res = {}
    for red, X, sl in (("modal", C.modal(P), False), ("mean", P, True)):
        t0 = test_excluding(X, np.zeros(25, bool), sl)
        t, trail, status, it = purify(X, sl)
        # how much is left to estimate ability from at the fixed point (degenerate if models sit at the bound)
        Fm = t.flag.to_numpy(); keep = ~Fm
        th_left = [C.theta_ml(x[keep & ~np.isnan(x)], np.where(keep & ~np.isnan(x))[0]) for x in X]
        at_bound = int(sum(abs(abs(v) - 4.0) < 1e-3 for v in th_left))
        res_diag = dict(items_left=int(keep.sum()), models_at_bound=at_bound,
                        mean_theta_left=float(np.mean(th_left)))
        res[red] = dict(start=[C.ITEMS[j] for j in np.where(t0.flag)[0]], start_rate=float(t0.flag.mean()), **res_diag,
                        final=[C.ITEMS[j] for j in np.where(t.flag)[0]], final_rate=float(t.flag.mean()),
                        status=status, iterations=it, path=trail,
                        p_final={r.item: float(r.p) for r in t.itertuples() if r.item in ("VR.26", "VR.36", "VR.39", "VR.42", "LN.01")})
        print(label, red, res[red]["start"], "->", res[red]["final"], status, flush=True)
    # single random run per model
    Fs0, Fs = np.zeros((NDRAW, 25), bool), np.zeros((NDRAW, 25), bool)
    for r in range(NDRAW):
        Xs = np.array([R[(meta.model == m).values][rng.integers(((meta.model == m).values).sum())] for m in models])
        Fs0[r] = test_excluding(Xs, np.zeros(25, bool)).flag.to_numpy()
        Fs[r] = purify(Xs)[0].flag.to_numpy()
    res["single"] = dict(draws=NDRAW,
                         start_rate=float(Fs0.mean()), final_rate=float(Fs.mean()),
                         start_share={it: float(Fs0[:, C.IDX[it]].mean()) for it in ("VR.26", "VR.36", "VR.39", "VR.42", "LN.01")},
                         final_share={it: float(Fs[:, C.IDX[it]].mean()) for it in ("VR.26", "VR.36", "VR.39", "VR.42", "LN.01")})
    print(label, "single", res["single"], flush=True)
    out[label] = res
C.save_json(out, "c8_purified.json")
