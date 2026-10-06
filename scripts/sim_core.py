"""Shared simulation machinery for the validation studies."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dif.irt import prob_2pl
from dif.anchored import anchored_dif

DIF_KINDS = ("none", "uniform", "nonuniform")


def make_bank(I, rng):
    a = rng.uniform(0.7, 2.2, I)
    b = rng.normal(0.0, 1.0, I)
    return a, b


def inject(a, b, idx_uniform, idx_nonuniform, b_shift=0.8, a_ratio=0.45):
    af, bf = a.copy(), b.copy()
    bf[idx_uniform] -= b_shift
    af[idx_nonuniform] *= a_ratio
    return af, bf


def simulate(I=20, n_ref=2000, n_foc=400, prop_dif=0.20, b_shift=0.8,
             a_ratio=0.45, foc_mu=0.0, foc_sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    a, b = make_bank(I, rng)
    n_dif = int(round(prop_dif * I))
    n_u = n_dif // 2
    idx = rng.permutation(I)
    iu, inu = idx[:n_u], idx[n_u:n_dif]
    af, bf = inject(a, b, iu, inu, b_shift, a_ratio)

    th_r = rng.normal(0, 1, n_ref)
    th_f = rng.normal(foc_mu, foc_sigma, n_foc)
    Rr = (rng.random((n_ref, I)) < prob_2pl(th_r, a, b)).astype(float)
    Rf = (rng.random((n_foc, I)) < prob_2pl(th_f, af, bf)).astype(float)

    truth = np.array(["none"] * I, dtype=object)
    truth[iu] = "uniform"
    truth[inu] = "nonuniform"
    return Rr, Rf, truth, (a, b, af, bf)


def run_cell(kwargs):
    """One simulation replicate -> summary dict."""
    Rr, Rf, truth, _ = simulate(**kwargs)
    res = anchored_dif(Rr, Rf, purify=True, verbose=False)
    t = res.table
    is_dif = truth != "none"
    flag = t["flag_total"].values
    tp = int((flag & is_dif).sum()); fn = int((~flag & is_dif).sum())
    fp = int((flag & ~is_dif).sum()); tn = int((~flag & ~is_dif).sum())

    # kind classification among detected items
    correct_kind = 0
    for i in np.where(flag & is_dif)[0]:
        pred = "uniform" if (t["flag_uniform"].values[i] and not t["flag_nonuniform"].values[i]) \
            else ("nonuniform" if (t["flag_nonuniform"].values[i] and not t["flag_uniform"].values[i])
                  else "both")
        if pred == truth[i]:
            correct_kind += 1
    # uniformity index: share of total DIF evidence carried by the difficulty shift.
    # Contamination (memorisation) should sit near 1, cognitive divergence lower.
    ui = t["lr_uniform"].values / np.maximum(t["lr_total"].values, 1e-9)
    ui = np.clip(ui, 0, 1)
    auc = np.nan
    pos = ui[truth == "uniform"]; neg = ui[truth == "nonuniform"]
    if len(pos) and len(neg):
        auc = float((pos[:, None] > neg[None, :]).mean()
                    + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return dict(
        power=tp / max(tp + fn, 1),
        ui_uniform=float(np.nanmean(ui[truth == "uniform"])) if (truth=="uniform").any() else np.nan,
        ui_nonuniform=float(np.nanmean(ui[truth == "nonuniform"])) if (truth=="nonuniform").any() else np.nan,
        ui_none=float(np.nanmean(ui[truth == "none"])),
        auc_kind=auc,
        type1=fp / max(fp + tn, 1),
        kind_acc=correct_kind / max(tp, 1),
        n_detected=int(flag.sum()),
        mu_hat=res.focal_mu, sigma_hat=res.focal_sigma,
        mu_true=kwargs.get("foc_mu", 0.0), sigma_true=kwargs.get("foc_sigma", 1.0),
        n_foc=kwargs.get("n_foc", 400), purified=res.purified,
        **{k: v for k, v in kwargs.items() if k in ("seed", "prop_dif", "b_shift", "I", "n_ref")}
    )
