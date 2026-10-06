"""Shared code for the independent camera-ready checks (written 5 Oct 2026).

Written from scratch on purpose: it does not import scripts/ or dif/, so that
agreement with the draft's numbers is a genuine second implementation, not a
re-run of the same code. (c4 is the exception: it calls the published pipeline,
because its point is what that pipeline does under a null.)
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from scipy import optimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

# Item text and keys are not distributed (PsychArchives Scientific Use Licence); the index carries
# only identifiers and the derived properties c9 uses.
_bank = json.load(open(os.path.join(ROOT, "data", "items25_index.json")))
HARNESS_TO_ICAR = {it["item_id"]: it["icar_id"] for it in _bank}
STEM_WORDS = {it["icar_id"]: it["stem_words"] for it in _bank}
STEM_CHARS = {it["icar_id"]: it["stem_chars"] for it in _bank}
ANSWER_TYPE = {it["icar_id"]: it["answer_type"] for it in _bank}
_cal = pd.read_csv(os.path.join(ROOT, "results", "sapa_calibration.csv"))
ITEMS = list(_cal["item"])                          # calibration order: 16 VR, then 9 LN
A = _cal["a"].to_numpy(float)
B = _cal["b"].to_numpy(float)
P_HUMAN = _cal["p"].to_numpy(float)                 # SAPA proportion correct
IDX = {it: i for i, it in enumerate(ITEMS)}
VR = np.array([i for i, it in enumerate(ITEMS) if it.startswith("VR")])
LN = np.array([i for i, it in enumerate(ITEMS) if it.startswith("LN")])
NUMBER = np.array([IDX["LN.01"], IDX["LN.03"]])     # number series
LETTER = np.array([i for i in LN if i not in NUMBER])  # letter series (7)


def short(m):
    return m.split("/")[-1].replace("-Instruct", "").replace("-1124", "")


def load_long(path):
    """One row per (model, frame, seed, item). y = 1/0 when parsed, NaN otherwise."""
    d = pd.read_json(path, lines=True)
    d["item"] = d["item_id"].map(HARNESS_TO_ICAR)
    assert d["item"].notna().all()
    d["y"] = np.where(d["parsable"].astype(bool), d["correct"].astype(float), np.nan)
    d = d.rename(columns={"prompt_variant": "frame"})
    return d[["model", "frame", "seed", "item", "y", "parsable"]]


def run_matrix(d):
    """(run x item) array, runs = (model, frame, seed), plus run metadata."""
    w = d.pivot_table(index=["model", "frame", "seed"], columns="item", values="y",
                      aggfunc="first", dropna=False).reindex(columns=ITEMS)
    return w.to_numpy(float), w.index.to_frame(index=False)


def model_props(d, models=None):
    """Per-model proportion correct over parsed runs, and parsed-run counts (8 x 25)."""
    g = d.groupby(["model", "item"])["y"]
    p = g.mean().unstack().reindex(columns=ITEMS)
    n = g.count().unstack().reindex(columns=ITEMS)
    if models is not None:
        p, n = p.reindex(models), n.reindex(models)
    return p.to_numpy(float), n.to_numpy(float), list(p.index)


def modal(p):
    """Modal answer: 1 if p > .5, 0 if p < .5, NaN on an exact tie or no data."""
    x = np.where(p > 0.5, 1.0, 0.0)
    x[np.isnan(p) | (np.abs(p - 0.5) < 1e-12)] = np.nan
    return x


def p2pl(theta, a=None, b=None):
    """2PL probabilities; theta of shape (...) -> (..., I)."""
    a = A if a is None else np.asarray(a, float)
    b = B if b is None else np.asarray(b, float)
    t = np.asarray(theta, float)
    return 1.0 / (1.0 + np.exp(-a * (t[..., None] - b)))


def theta_ml(x, idx, lo=-4.0, hi=4.0, a=None, b=None):
    """Bounded ML ability from responses x (0/1, or fractional expected scores) on items idx."""
    a = A[idx] if a is None else a[idx]
    b = B[idx] if b is None else b[idx]
    x = np.asarray(x, float)

    def nll(t):
        z = a * (t - b)
        return -(x * z - np.logaddexp(0.0, z)).sum()

    return float(optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                          options={"xatol": 1e-7}).x)


def pb_pmf(probs):
    pmf = np.array([1.0])
    for q in probs:
        pmf = np.convolve(pmf, [1.0 - q, q])
    return pmf


def pb_p_two_sided(probs, k):
    """Exact two-sided p: total probability of outcomes no more likely than k."""
    pmf = pb_pmf(probs)
    return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-9)].sum()))


def stoploss_p_two_sided(probs, s):
    """Valid two-sided p for a sum s of independent [0,1] scores with means probs.

    Each score is smaller than Bernoulli(mean) in the convex order, so the sum is
    smaller than the Poisson-binomial in the convex order. Markov on stop-loss
    transforms then bounds each tail: P(S >= s) <= E(PB - c)+ / (s - c) for c < s,
    and P(S <= s) <= E(c - PB)+ / (c - s) for c > s. Two-sided = 2 x the smaller.
    """
    pmf = pb_pmf(probs)
    ks = np.arange(len(pmf))
    up, lo = 1.0, 1.0
    for c in np.concatenate([ks, np.linspace(0, len(pmf) - 1, 4 * len(pmf))]):
        if c < s - 1e-12:
            up = min(up, (pmf * np.maximum(ks - c, 0)).sum() / (s - c))
        if c > s + 1e-12:
            lo = min(lo, (pmf * np.maximum(c - ks, 0)).sum() / (c - s))
    return float(min(1.0, 2 * min(up, lo)))


def bh(p, alpha=0.05):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    rej = np.zeros(n, bool)
    if ok.any():
        rej[o[: np.max(np.where(ok)[0]) + 1]] = True
    return rej


def item_test(X, lo=-4.0, hi=4.0, stoploss=False, a=None, b=None):
    """Model-as-examinee item test. X: (examinees x 25), 0/1 (or [0,1] scores), NaN = missing.

    For item i, each examinee's theta comes from its OTHER observed items; under
    invariance the item total is Poisson-binomial with P_i(theta_m).
    """
    a = A if a is None else a
    b = B if b is None else b
    rows = []
    for i in range(X.shape[1]):
        ex = np.where(~np.isnan(X[:, i]))[0]
        probs = []
        for m in ex:
            oth = np.where(~np.isnan(X[m]))[0]
            oth = oth[oth != i]
            th = theta_ml(X[m, oth], oth, lo, hi, a, b)
            probs.append(1 / (1 + np.exp(-a[i] * (th - b[i]))))
        s = float(np.nansum(X[ex, i]))
        p = stoploss_p_two_sided(probs, s) if stoploss else pb_p_two_sided(probs, int(round(s)))
        rows.append(dict(item=ITEMS[i], n=len(ex), k=s, expected=float(np.sum(probs)), p=p))
    t = pd.DataFrame(rows)
    t["flag"] = bh(t["p"].to_numpy())
    return t


def lz_star(x, idx, lo=-6.0, hi=6.0):
    """Snijders (2001) lz* with ML theta, 2PL (r_i = a_i, r0 = 0 for ML)."""
    x = np.asarray(x, float)
    if x.sum() in (0, len(x)):
        return np.nan, np.nan
    th = theta_ml(x, idx, lo, hi)
    a, b = A[idx], B[idx]
    P = 1 / (1 + np.exp(-a * (th - b)))
    Q = 1 - P
    w = np.log(P / Q)
    c = (P * Q * w * a).sum() / (P * Q * a * a).sum()
    wt = w - c * a
    return float(((x - P) * wt).sum() / np.sqrt((wt ** 2 * P * Q).sum())), th


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --------------------------------------------------------------------------
# vectorised leave-one-item-out ML ability (grid search + Newton polish)
# --------------------------------------------------------------------------
_GRID = np.linspace(-4.0, 4.0, 1601)


def loo_theta(X, a=None, b=None, lo=-4.0, hi=4.0, newton=6):
    """TH[m, i] = ML ability of examinee m from its observed items other than i.

    X: (M, I) responses in [0, 1] (fractional scores allowed), NaN = missing.
    Bounded on [lo, hi]; all-correct / all-wrong patterns land on the bound.
    """
    a = A if a is None else np.asarray(a, float)
    b = B if b is None else np.asarray(b, float)
    O = ~np.isnan(X)
    X0 = np.where(O, X, 0.0)
    G = _GRID if (lo, hi) == (-4.0, 4.0) else np.linspace(lo, hi, 1601)
    Z = a * (G[:, None] - b)                                  # (K, I)
    lp, lq = -np.logaddexp(0, -Z), -np.logaddexp(0, Z)
    Cmat = (X0[:, None, :] * lp[None] + (1 - X0)[:, None, :] * lq[None]) * O[:, None, :]
    loo = Cmat.sum(2)[:, :, None] - Cmat                      # (M, K, I)
    th = G[loo.argmax(1)]                                     # (M, I)
    I = X.shape[1]
    W = (O[:, None, :] & ~np.eye(I, dtype=bool)[None]).astype(float)   # (M, i, j): j used for target i
    for _ in range(newton):
        P = 1 / (1 + np.exp(-a * (th[:, :, None] - b)))       # (M, i, j)
        score = (W * a * (X0[:, None, :] - P)).sum(2)
        info = (W * a * a * P * (1 - P)).sum(2)
        step = np.where(info > 1e-12, score / np.maximum(info, 1e-12), 0.0)
        th = np.clip(th + np.clip(step, -0.5, 0.5), lo, hi)
    return th


def fast_item_test(X, a=None, b=None, stoploss=False, lo=-4.0, hi=4.0, alpha=0.05):
    a = A if a is None else np.asarray(a, float)
    b = B if b is None else np.asarray(b, float)
    TH = loo_theta(X, a, b, lo, hi)
    PI = 1 / (1 + np.exp(-a * (TH - b)))                      # P_i(theta_m(-i))
    out = []
    for i in range(X.shape[1]):
        ex = ~np.isnan(X[:, i])
        probs = PI[ex, i]
        s = float(X[ex, i].sum())
        p = stoploss_p_two_sided(probs, s) if stoploss else pb_p_two_sided(probs, int(round(s)))
        out.append((ex.sum(), s, probs.sum(), p))
    t = pd.DataFrame(out, columns=["n", "k", "expected", "p"])
    t.insert(0, "item", ITEMS)
    t["flag"] = bh(t["p"].to_numpy(), alpha)
    return t
