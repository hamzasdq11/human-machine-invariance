"""Mantel-Haenszel DIF with the ETS A/B/C effect-size classification.

Included as the second, non-parametric estimator required by the analysis plan:
conclusions that depend on which DIF estimator was used are not conclusions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["mantel_haenszel", "ets_class"]


def _strata(total, n_bins):
    """Thin-then-bin the matched total score into strata with usable counts."""
    qs = np.unique(np.quantile(total, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.digitize(total, qs[1:-1]), 0, len(qs) - 2)


def mantel_haenszel(R_ref, R_foc, item_names=None, n_bins=10, alpha=0.05):
    """Mantel-Haenszel odds ratio DIF, matching on observed total score.

    Returns a table with alpha_MH, the ETS delta metric, chi-square with
    Holland-Thayer continuity correction, and the A/B/C classification.
    """
    R_ref, R_foc = np.asarray(R_ref, float), np.asarray(R_foc, float)
    I = R_ref.shape[1]
    names = list(item_names) if item_names is not None else [f"item{i}" for i in range(I)]

    tot_ref = np.nansum(R_ref, axis=1)
    tot_foc = np.nansum(R_foc, axis=1)
    allt = np.concatenate([tot_ref, tot_foc])
    bins = _strata(allt, n_bins)
    s_ref, s_foc = bins[:len(tot_ref)], bins[len(tot_ref):]

    rows = []
    for i in range(I):
        num = den = 0.0
        E = V = Fsum = 0.0
        for s in np.unique(bins):
            r = R_ref[s_ref == s, i]; f = R_foc[s_foc == s, i]
            r = r[~np.isnan(r)];      f = f[~np.isnan(f)]
            Nr, Nf = len(r), len(f)
            if Nr == 0 or Nf == 0:
                continue
            Ar, Br = r.sum(), Nr - r.sum()
            Af, Bf = f.sum(), Nf - f.sum()
            T = Nr + Nf
            if T == 0:
                continue
            num += Ar * Bf / T
            den += Af * Br / T
            m1 = Ar + Af                      # total correct in stratum
            E += Nr * m1 / T
            Fsum += Ar
            if T > 1:
                V += (Nr * Nf * m1 * (T - m1)) / (T ** 2 * (T - 1))
        if den <= 0 or num <= 0 or V <= 0:
            rows.append(dict(item=names[i], alpha_mh=np.nan, delta_mh=np.nan,
                             chi2=np.nan, p=np.nan))
            continue
        alpha_mh = num / den
        delta = -2.35 * np.log(alpha_mh)
        chi2 = (abs(Fsum - E) - 0.5) ** 2 / V
        rows.append(dict(item=names[i], alpha_mh=alpha_mh, delta_mh=delta,
                         chi2=chi2, p=stats.chi2.sf(chi2, 1)))
    tab = pd.DataFrame(rows)
    tab["sig"] = tab["p"] < alpha
    tab["ets"] = [ets_class(d, s) for d, s in zip(tab["delta_mh"], tab["sig"])]
    return tab


def ets_class(delta, significant):
    """ETS A (negligible) / B (moderate) / C (large) classification."""
    if not np.isfinite(delta):
        return "NA"
    ad = abs(delta)
    if not significant or ad < 1.0:
        return "A"
    return "B" if ad < 1.5 else "C"
