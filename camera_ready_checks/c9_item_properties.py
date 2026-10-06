"""C9: are the model-level misfits related to properties of the items? (Tier 2, plan item B3b)

For each item, the models' departure from the human curve is summarised as an implied shift on the
logit scale: the d_i that solves sum_m P_i(theta_m(-i); b_i - d_i) = S_i, where S_i is the run-mean
total over the eight models and theta_m(-i) the model's ability from its other items (as in Fig. 4).
d_i > 0: the item is easier for the models than the human curve predicts.

Properties: human discrimination a, human difficulty b, stem length (words, characters) and answer
type (number, word or phrase, letter). Spearman correlations with 95% bootstrap intervals over items
(resampling items) and the range of the correlation when each item is left out in turn; all 25 items
and the 16 verbal items alone (so that answer type and the letter-series result do not drive it).

Mechanical caution: the models sit about 1.5 logits below the human mean, so the expected count is
small for hard items and the shift is less precisely determined there; d_i is bounded at +-6.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
import common as C

rng = np.random.default_rng(9102026)
NBOOT = int(os.environ.get("NBOOT", 4000))


def implied_shifts(X):
    TH = C.loo_theta(X)
    out = []
    for i in range(X.shape[1]):
        ex = ~np.isnan(X[:, i])
        th, s = TH[ex, i], float(X[ex, i].sum())
        f = lambda d: (1 / (1 + np.exp(-C.A[i] * (th - (C.B[i] - d))))).sum() - s
        lo, hi = -6.0, 6.0
        d = lo if f(lo) > 0 else hi if f(hi) < 0 else brentq(f, lo, hi)
        out.append(dict(item=C.ITEMS[i], S=s, E=float((1 / (1 + np.exp(-C.A[i] * (th - C.B[i])))).sum()), shift=d))
    return pd.DataFrame(out)


def answer_type(it):
    """number, letter or word; derived from the key when the index was built (data/items25_index.json)."""
    return C.ANSWER_TYPE[it]


def corr_summary(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = spearmanr(x, y).statistic
    bs = []
    for _ in range(NBOOT):
        j = rng.integers(0, n, n)
        if np.ptp(x[j]) > 0 and np.ptp(y[j]) > 0:
            bs.append(spearmanr(x[j], y[j]).statistic)
    loo = [spearmanr(np.delete(x, k), np.delete(y, k)).statistic for k in range(n)]
    # exact-ish permutation p-value (two-sided)
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(NBOOT)])
    p = (1 + np.sum(np.abs(perm) >= abs(r) - 1e-12)) / (NBOOT + 1)
    return dict(rho=float(r), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                loo_min=float(min(loo)), loo_max=float(max(loo)), perm_p=float(p), n=n)


out = {}
for path, label in ((os.path.join(C.ROOT, "data", "responses_free25.jsonl"), "main"),
                    (os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"), "temp0")):
    d = C.load_long(path)
    models = sorted(d.model.unique())
    P, _, _ = C.model_props(d, models)
    t = implied_shifts(P)
    t["a"], t["b"] = C.A, C.B
    t["words"] = [C.STEM_WORDS[it] for it in t.item]
    t["chars"] = [C.STEM_CHARS[it] for it in t.item]
    t["type"] = [answer_type(it) for it in t.item]
    t["domain"] = ["VR" if it.startswith("VR") else ("letter" if C.IDX[it] in C.LETTER else "number") for it in t.item]
    res = {"items": t.round(4).to_dict(orient="records")}
    for scope, mask in (("all25", np.ones(25, bool)), ("verbal16", t.item.str.startswith("VR").to_numpy())):
        tt = t[mask]
        res[scope] = {prop: corr_summary(tt[prop], tt["shift"]) for prop in ("a", "b", "words", "chars")}
    # the three antonym items ("The opposite of ...": VR.11, VR.26, VR.36) have the shortest verbal stems
    ant = t.item.isin(["VR.11", "VR.26", "VR.36"]).to_numpy()
    res["antonyms"] = dict(items=["VR.11", "VR.26", "VR.36"], shifts=[float(v) for v in t["shift"][ant]],
                           words=[int(v) for v in t["words"][ant]])
    res["all25_excl_antonyms"] = {prop: corr_summary(t[prop][~ant], t["shift"][~ant]) for prop in ("a", "b", "words")}
    # answer type: mean shift by type, and a permutation test of letter vs the rest / number vs word
    g = t.groupby("type")["shift"].agg(["mean", "median", "count"]).round(3)
    res["by_type"] = g.to_dict(orient="index")
    def perm_diff(m1):
        obs = t["shift"][m1].mean() - t["shift"][~m1].mean()
        perm = []
        for _ in range(NBOOT):
            pm = rng.permutation(m1)
            perm.append(t["shift"][pm].mean() - t["shift"][~pm].mean())
        return dict(diff=float(obs), perm_p=float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (NBOOT + 1)))
    res["letter_vs_rest"] = perm_diff((t.type == "letter").to_numpy())
    vr = t[t.item.str.startswith("VR")]
    m = (vr.type == "number").to_numpy()
    obs = vr["shift"][m].mean() - vr["shift"][~m].mean()
    perm = [vr["shift"][pm].mean() - vr["shift"][~pm].mean() for pm in (rng.permutation(m) for _ in range(NBOOT))]
    res["verbal_number_vs_word"] = dict(diff=float(obs), n_number=int(m.sum()),
                                        perm_p=float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (NBOOT + 1)))
    out[label] = res
    print(label)
    print(t[["item", "S", "E", "shift", "a", "b", "words", "type"]].round(2).to_string(index=False))
    for scope in ("all25", "verbal16", "all25_excl_antonyms"):
        for prop, v in res[scope].items():
            print(f"  {scope:9s} {prop:6s} rho={v['rho']:+.2f} CI[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] "
                  f"LOO[{v['loo_min']:+.2f},{v['loo_max']:+.2f}] perm p={v['perm_p']:.3f}")
    print("  by type", res["by_type"])
    print("  letter vs rest", res["letter_vs_rest"], " verbal number vs word", res["verbal_number_vs_word"], flush=True)
C.save_json(out, "c9_item_properties.json")
