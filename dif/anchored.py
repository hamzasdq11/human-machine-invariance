"""Anchored likelihood-ratio differential item functioning.

The design question this module exists to answer:  standard IRT-based DIF
requires calibrating item parameters in *both* groups.  When one group is a
small pool of machine respondents that is exactly the regime in which IRT
estimation has been shown to fail (few respondents, many items).

The anchored design avoids that failure mode entirely.  Item parameters are
calibrated once, on the reference (human) group, where N is large and the
estimation problem is the one IRT was validated for.  The focal group enters
only through (i) a two-parameter latent distribution N(mu, sigma^2) identified
from anchor items and (ii) one item at a time under test.  Consistency of the
anchored estimator therefore depends on the *reference* sample size, not on the
number of focal respondents.

Tests per studied item i (anchors held fixed throughout):
    total DIF      : (a_i, b_i) free      vs both constrained   -> chi2(2)
    uniform DIF    : b_i free, a_i fixed  vs both constrained   -> chi2(1)
    non-uniform DIF: (a_i, b_i) free      vs b_i free only      -> chi2(1)

Purification iteratively removes flagged items from the anchor set and
re-identifies the focal distribution until the anchor set is stable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import optimize, stats

from .irt import (prob_2pl, gh_nodes, fit_2pl_mml, estimate_group_dist,
                  _prepare, _resp_loglik_matrix)

__all__ = ["anchored_dif", "raju_areas", "benjamini_hochberg", "DIFResult"]


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------

def raju_areas(a_ref, b_ref, a_foc, b_foc):
    """Raju (1988) signed and unsigned area between two 2PL item curves."""
    a1, b1, a2, b2 = map(np.asarray, (a_ref, b_ref, a_foc, b_foc))
    signed = b1 - b2                      # focal easier => positive
    da = a2 - a1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        arg = a1 * a2 * (b2 - b1) / np.where(np.abs(da) < 1e-8, np.nan, da)
        # log1p(exp(x)) computed stably
        soft = np.where(arg > 30, arg, np.log1p(np.exp(np.clip(arg, -700, 30))))
        unsigned = np.abs((2 * da / (a1 * a2)) * soft - (b2 - b1))
    unsigned = np.where(np.abs(da) < 1e-8, np.abs(b2 - b1), unsigned)
    return signed, unsigned


def benjamini_hochberg(p, alpha=0.05):
    """Return boolean rejection vector under BH control at level alpha."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    rej = np.zeros(n, dtype=bool)
    if passed.any():
        k = np.max(np.where(passed)[0])
        rej[order[:k + 1]] = True
    return rej


# --------------------------------------------------------------------------
# focal-group likelihood with anchors fixed
# --------------------------------------------------------------------------

def _focal_loglik(R, M, a, b, mu, sigma, n_points):
    X, W = gh_nodes(n_points, mu, sigma)
    ll = _resp_loglik_matrix(R, M, prob_2pl(X, a, b))
    mx = ll.max(axis=1, keepdims=True)
    return float((mx[:, 0] + np.log(np.exp(ll - mx) @ W)).sum())


def _item_ll(R_i, M_i, X, a, b):
    """Log-likelihood contribution of ONE item at every quadrature node -> (N, Q)."""
    p = prob_2pl(X, np.array([a]), np.array([b]))[:, 0]
    return np.outer(R_i * M_i, np.log(p)) + np.outer((1 - R_i) * M_i, np.log1p(-p))


class _AnchorCache:
    """Precomputed anchor log-likelihood surface for a fixed (mu, sigma).

    The studied item's parameters move thousands of times during the
    likelihood-ratio search while the anchors never move.  Recomputing the
    anchor contribution on every evaluation is the dominant cost of the naive
    implementation; caching it makes each search step depend only on the single
    item under test.
    """

    def __init__(self, Rf, Mf, a_ref, b_ref, anchors, mu, sigma, n_points):
        self.X, self.W = gh_nodes(n_points, mu, sigma)
        self.Rf, self.Mf = Rf, Mf
        self.total = np.zeros((Rf.shape[0], len(self.X)))
        self._per_item = {}
        for j in anchors:
            c = _item_ll(Rf[:, j], Mf[:, j], self.X, a_ref[j], b_ref[j])
            self._per_item[j] = c
            self.total += c

    def base_for(self, i):
        """Anchor surface excluding item i, so an anchor can also be studied."""
        return self.total - self._per_item[i] if i in self._per_item else self.total

    def marginal(self, base, item_surface):
        ll = base + item_surface
        mx = ll.max(axis=1, keepdims=True)
        return float((mx[:, 0] + np.log(np.exp(ll - mx) @ self.W)).sum())


def _fit_studied_item(cache, base, R_i, M_i, a0, b0, free_a=True, free_b=True):
    """Maximise the focal marginal log-likelihood over the studied item only."""
    X = cache.X

    def unpack(p):
        if free_a and free_b: return p[0], p[1]
        if free_b:            return a0, p[0]
        if free_a:            return p[0], b0
        return a0, b0

    def nll(p):
        a, b = unpack(p)
        return -cache.marginal(base, _item_ll(R_i, M_i, X, a, b))

    if free_a and free_b:
        p0, bounds = np.array([a0, b0]), [(0.05, 6.0), (-6.0, 6.0)]
    elif free_b:
        p0, bounds = np.array([b0]), [(-6.0, 6.0)]
    elif free_a:
        p0, bounds = np.array([a0]), [(0.05, 6.0)]
    else:
        return -nll(np.array([])), (a0, b0)

    res = optimize.minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    a, b = unpack(res.x)
    return -float(res.fun), (float(a), float(b))


@dataclass
class DIFResult:
    table: pd.DataFrame
    ref_params: pd.DataFrame
    focal_mu: float
    focal_sigma: float
    anchors: list
    n_purify_iter: int
    purified: bool
    purify_status: str = "converged"   # converged | cycled | capped | off

    @property
    def dif_rate(self):
        return float(self.table["flag_total"].mean())

    def summary(self):
        t = self.table
        return {
            "n_items": len(t),
            "n_anchors": len(self.anchors),
            "dif_rate_total": self.dif_rate,
            "dif_rate_uniform": float(t["flag_uniform"].mean()),
            "dif_rate_nonuniform": float(t["flag_nonuniform"].mean()),
            "focal_mu": self.focal_mu,
            "focal_sigma": self.focal_sigma,
            "mean_abs_signed_area": float(t["signed_area"].abs().mean()),
            "purified": self.purified,
            "purify_status": self.purify_status,
        }


def anchored_dif(R_ref, R_foc, item_names=None, alpha=0.05, n_points=41,
                 purify=True, max_purify=5, min_anchors=4,
                 ref_params=None, free_sigma=True, verbose=False):
    """Run anchored likelihood-ratio DIF of a focal group against a reference.

    Parameters
    ----------
    R_ref, R_foc : (N, I) arrays of 0/1/NaN responses on the same I items.
    ref_params   : optional (a, b) tuple to skip reference calibration.
    """
    R_ref = np.asarray(R_ref, dtype=float)
    R_foc = np.asarray(R_foc, dtype=float)
    I = R_ref.shape[1]
    names = list(item_names) if item_names is not None else [f"item{i}" for i in range(I)]

    # 1. calibrate on the reference group (identification: theta ~ N(0,1))
    if ref_params is None:
        fit = fit_2pl_mml(R_ref, names=names, n_points=n_points)
        a_ref, b_ref = fit.a.copy(), fit.b.copy()
    else:
        a_ref, b_ref = map(np.asarray, ref_params)

    Rf, Mf = _prepare(R_foc)

    anchors = list(range(I))
    it, purified, status = 0, False, ("off" if not purify else "capped")
    seen = {}                     # anchor set -> order seen, for cycle detection

    while True:
        it += 1
        # 2. identify the focal latent distribution from anchors only
        mu, sigma, _ = estimate_group_dist(R_foc[:, anchors], a_ref[anchors],
                                           b_ref[anchors], n_points=n_points,
                                           free_sigma=free_sigma)

        cache = _AnchorCache(Rf, Mf, a_ref, b_ref, anchors, mu, sigma, n_points)

        rows = []
        for i in range(I):
            base = cache.base_for(i)
            R_i, M_i = Rf[:, i], Mf[:, i]

            ll_c = cache.marginal(base, _item_ll(R_i, M_i, cache.X, a_ref[i], b_ref[i]))
            ll_b, (_, b_b) = _fit_studied_item(cache, base, R_i, M_i, a_ref[i], b_ref[i],
                                               free_a=False, free_b=True)
            # Fix (3 Oct 2026): the (a, b) fit could stall at a bound below the nested
            # b-only optimum (seen once, VR.11). Multi-start, including from the b-only
            # solution, and never accept a value below the nested model's.
            ll_ab, (a_f, b_f) = ll_b, (a_ref[i], b_b)
            for a0, b0 in ((a_ref[i], b_ref[i]), (a_ref[i], b_b), (0.5, b_b), (2.0, b_b)):
                ll_t, par_t = _fit_studied_item(cache, base, R_i, M_i, a0, b0,
                                                free_a=True, free_b=True)
                if ll_t > ll_ab:
                    ll_ab, (a_f, b_f) = ll_t, par_t

            lr_tot = max(2 * (ll_ab - ll_c), 0.0)
            lr_uni = max(2 * (ll_b - ll_c), 0.0)
            lr_non = max(2 * (ll_ab - ll_b), 0.0)
            rows.append(dict(
                item=names[i], a_ref=a_ref[i], b_ref=b_ref[i], a_foc=a_f, b_foc=b_f,
                lr_total=lr_tot, p_total=stats.chi2.sf(lr_tot, 2),
                lr_uniform=lr_uni, p_uniform=stats.chi2.sf(lr_uni, 1),
                lr_nonuniform=lr_non, p_nonuniform=stats.chi2.sf(lr_non, 1),
            ))

        tab = pd.DataFrame(rows)
        rej = benjamini_hochberg(tab["p_total"].values, alpha)
        tab["flag_total"] = rej
        tab["flag_uniform"] = benjamini_hochberg(tab["p_uniform"].values, alpha)
        tab["flag_nonuniform"] = benjamini_hochberg(tab["p_nonuniform"].values, alpha)

        new_anchors = [i for i in range(I) if not rej[i]]
        if len(new_anchors) < min_anchors:                 # keep the least-suspect items
            new_anchors = sorted(np.argsort(tab["p_total"].values)[::-1][:min_anchors].tolist())

        key = tuple(sorted(new_anchors))
        if not purify:
            anchors, status = new_anchors, "off"
            break
        if key == tuple(sorted(anchors)):                  # fixed point
            anchors, purified, status = new_anchors, True, "converged"
            break
        if key in seen:
            # The anchor set is cycling.  Rather than let an arbitrary iteration
            # cap decide the answer, take the intersection of the sets in the
            # cycle -- the items no iteration ever flagged.  Conservative, and
            # unlike stopping at the cap it does not depend on max_purify.
            keys = list(seen)
            cyc = [set(k) for k in keys[seen[key]:]] + [set(key)]
            inter = sorted(set.intersection(*cyc))
            if len(inter) < min_anchors:
                inter = sorted(np.argsort(tab["p_total"].values)[::-1][:min_anchors].tolist())
            anchors, status = inter, "cycled"
            break
        if it >= max_purify:
            anchors, status = new_anchors, "capped"
            break
        seen[key] = len(seen)
        anchors = new_anchors
        if verbose:
            print(f"  purify {it}: {len(anchors)} anchors, mu={mu:.3f} sigma={sigma:.3f}")

    sa, ua = raju_areas(tab["a_ref"], tab["b_ref"], tab["a_foc"], tab["b_foc"])
    tab["signed_area"], tab["unsigned_area"] = sa, ua

    return DIFResult(table=tab,
                     ref_params=pd.DataFrame({"item": names, "a": a_ref, "b": b_ref}),
                     focal_mu=mu, focal_sigma=sigma, anchors=[names[i] for i in anchors],
                     n_purify_iter=it, purified=purified, purify_status=status)
