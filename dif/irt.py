"""Two-parameter logistic IRT: marginal maximum likelihood (MML-EM) calibration.

Implements the estimation machinery required for anchored differential item
functioning analysis.  Written from scratch (no R dependency) so the whole
pipeline runs on CPU with numpy/scipy only.

Model
-----
    P(X_ij = 1 | theta_j) = 1 / (1 + exp(-a_i (theta_j - b_i)))

Latent ability is assumed normal, N(mu, sigma^2).  For the reference group the
distribution is fixed at N(0, 1) for identification; for a focal group in an
anchored design mu and sigma are free parameters estimated from anchor items.

Missing responses are handled by omission from the likelihood (missing at
random conditional on theta), which is standard for planned-missing designs
such as SAPA.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy import optimize
from numpy.polynomial.hermite_e import hermegauss

__all__ = ["ItemParams", "gh_nodes", "prob_2pl", "fit_2pl_mml", "eap_scores",
           "loglik_2pl", "estimate_group_dist"]

# --------------------------------------------------------------------------
# quadrature
# --------------------------------------------------------------------------

from functools import lru_cache


@lru_cache(maxsize=32)
def _base_nodes(n_points: int):
    """hermegauss solves an eigenproblem; it depends only on n_points, so cache it."""
    x, w = hermegauss(n_points)
    return x, w / w.sum()


def gh_nodes(n_points: int = 41, mu: float = 0.0, sigma: float = 1.0):
    """Gauss-Hermite nodes/weights for a N(mu, sigma^2) integrating measure.

    Returns (X, W) with W summing to 1 so that sum_q W_q f(X_q) approximates
    E[f(theta)] under theta ~ N(mu, sigma^2).
    """
    x, w = _base_nodes(int(n_points))    # weight function exp(-x^2/2)
    return mu + sigma * x, w


# --------------------------------------------------------------------------
# item response function
# --------------------------------------------------------------------------

def prob_2pl(theta, a, b):
    """P(correct). theta: (Q,) or (N,); a,b: (I,) -> returns (len(theta), I)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    z = a[None, :] * (theta[:, None] - b[None, :])
    # stable logistic
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return np.clip(out, 1e-12, 1 - 1e-12)


@dataclass
class ItemParams:
    a: np.ndarray
    b: np.ndarray
    names: list = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    loglik: float = np.nan

    def as_frame(self):
        import pandas as pd
        return pd.DataFrame({"item": self.names or list(range(len(self.a))),
                             "a": self.a, "b": self.b})


# --------------------------------------------------------------------------
# likelihood helpers
# --------------------------------------------------------------------------

def _resp_loglik_matrix(R, M, P):
    """log P(response pattern | theta_q) for every examinee x node.

    R : (N, I) responses, 0/1, NaN encoded as 0 with mask
    M : (N, I) bool, True where observed
    P : (Q, I) item probabilities at each node
    returns (N, Q)
    """
    logP = np.log(P)              # (Q, I)
    logQ = np.log1p(-P)           # (Q, I)
    # (N,I) @ (I,Q) -> (N,Q)
    return (R * M) @ logP.T + ((1 - R) * M) @ logQ.T


def loglik_2pl(R, M, a, b, mu=0.0, sigma=1.0, n_points=41):
    X, W = gh_nodes(n_points, mu, sigma)
    P = prob_2pl(X, a, b)
    ll = _resp_loglik_matrix(R, M, P)                       # (N, Q)
    mx = ll.max(axis=1, keepdims=True)
    marg = mx[:, 0] + np.log(np.exp(ll - mx) @ W)
    return float(marg.sum())


def _posterior(R, M, a, b, X, W):
    ll = _resp_loglik_matrix(R, M, prob_2pl(X, a, b))
    mx = ll.max(axis=1, keepdims=True)
    num = np.exp(ll - mx) * W[None, :]
    post = num / num.sum(axis=1, keepdims=True)
    marg = float((mx[:, 0] + np.log(num.sum(axis=1))).sum())
    return post, marg


# --------------------------------------------------------------------------
# MML-EM calibration
# --------------------------------------------------------------------------

def _prepare(responses):
    R = np.asarray(responses, dtype=float)
    M = ~np.isnan(R)
    R = np.where(M, R, 0.0)
    return R, M.astype(float)


def _fit_one_item(nq, rq, X, a0, b0):
    """M-step for a single item given expected counts at each node."""
    def nll(p):
        a, b = p
        pr = prob_2pl(X, np.array([a]), np.array([b]))[:, 0]
        return -(rq * np.log(pr) + (nq - rq) * np.log1p(-pr)).sum()

    def grad(p):
        a, b = p
        pr = prob_2pl(X, np.array([a]), np.array([b]))[:, 0]
        resid = rq - nq * pr
        da = -(resid * (X - b)).sum()
        db = -(resid * (-a)).sum()
        return np.array([da, db])

    res = optimize.minimize(nll, np.array([a0, b0]), jac=grad, method="L-BFGS-B",
                            bounds=[(0.05, 6.0), (-6.0, 6.0)])
    return res.x


def fit_2pl_mml(responses, names=None, n_points=41, max_iter=500, tol=1e-6,
                mu=0.0, sigma=1.0, verbose=False):
    """Calibrate 2PL item parameters by marginal maximum likelihood (EM)."""
    R, M = _prepare(responses)
    N, I = R.shape
    X, W = gh_nodes(n_points, mu, sigma)

    # sensible starts: classical p-value -> b, uniform a
    p = np.clip((R * M).sum(0) / np.maximum(M.sum(0), 1), .02, .98)
    a = np.full(I, 1.0)
    b = -np.log(p / (1 - p))

    prev = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        post, marg = _posterior(R, M, a, b, X, W)          # (N,Q)
        # expected counts per item per node
        nq = post.T @ M                                    # (Q, I)
        rq = post.T @ (R * M)                              # (Q, I)
        for i in range(I):
            a[i], b[i] = _fit_one_item(nq[:, i], rq[:, i], X, a[i], b[i])
        if verbose and it % 25 == 0:
            print(f"  EM {it:4d}  loglik={marg:.4f}")
        if abs(marg - prev) < tol * (1 + abs(prev)):
            converged = True
            break
        prev = marg

    ll = loglik_2pl(R, M, a, b, mu, sigma, n_points)
    return ItemParams(a=a, b=b, names=list(names) if names is not None else [],
                      n_iter=it, converged=converged, loglik=ll)


def eap_scores(responses, a, b, n_points=61, mu=0.0, sigma=1.0):
    """Expected a posteriori ability estimates and posterior SDs."""
    R, M = _prepare(responses)
    X, W = gh_nodes(n_points, mu, sigma)
    post, _ = _posterior(R, M, a, b, X, W)
    th = post @ X
    var = post @ (X ** 2) - th ** 2
    return th, np.sqrt(np.maximum(var, 0))


def estimate_group_dist(responses, a, b, n_points=61, free_sigma=True):
    """Estimate (mu, sigma) of a group's latent distribution with items fixed.

    This is the step that makes the anchored design work: the focal group's
    ability distribution is identified from the anchor items alone, so the
    number of focal-group respondents never has to identify item parameters.
    """
    R, M = _prepare(responses)

    def nll(p):
        mu = p[0]
        sigma = np.exp(p[1]) if free_sigma else 1.0
        return -loglik_2pl(R, M, a, b, mu, sigma, n_points)

    p0 = np.array([0.0, 0.0])
    res = optimize.minimize(nll, p0, method="Nelder-Mead",
                            options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
    mu = float(res.x[0])
    sigma = float(np.exp(res.x[1])) if free_sigma else 1.0
    return mu, sigma, -float(res.fun)
