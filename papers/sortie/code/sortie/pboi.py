"""Module C: Poisson-binomial order-statistic evaluator (THEORY Thm 2, Prop 3, Prop 4)."""

import math

import numpy as np
from scipy.special import gammainc, gammaincinv, gammaln, ndtr, ndtri

Q_TAIL = 1e-12
ORDERED_MEASURES = ("max", "range", "gmd", "mad")


def lam_vector(name, m):
    """Weight vector lambda of an ordered measure on m drivers (nondecreasing)."""
    m = int(m)
    lam = np.zeros(m)
    if name == "max":
        lam[-1] = 1.0
    elif name == "range":
        if m < 2:
            raise ValueError("range needs m >= 2")
        lam[-1] = 1.0
        lam[0] = -1.0
    elif name == "gmd":
        j = np.arange(1, m + 1, dtype=np.float64)
        lam = 2.0 * (2.0 * j - m - 1.0) / float(m) ** 2
    elif name == "mad":
        if m % 2 == 0:
            raise ValueError("mad needs odd m")
        q = (m + 1) // 2
        lam[: q - 1] = -1.0 / m
        lam[q:] = 1.0 / m
    elif name == "var":
        raise ValueError("'var' is not an ordered measure; use expost_var")
    else:
        raise ValueError(f"unknown measure {name!r}")
    return lam


def _resolve_lam(lam, m):
    if isinstance(lam, str):
        return lam_vector(lam, m)
    lam = np.asarray(lam, dtype=np.float64)
    if lam.shape != (m,):
        raise ValueError(f"lambda must have shape ({m},)")
    return lam


def ordered(a, lam):
    """l_lambda(a): sum_j lam_j a_(j). Vector input gives a scalar; (S,m) gives (S,)."""
    a = np.asarray(a, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    return np.sort(a, axis=-1) @ lam


def exante(L, lam):
    L = np.asarray(L, dtype=np.float64)
    return ordered(L, _resolve_lam(lam, L.shape[-1]))


class Marginals:
    """Independent workload marginals with mean L and standard deviation sigma.

    gamma:  shape L^2/sigma^2, scale sigma^2/L (support [0, inf)).
    normal: N(L, sigma^2).
    sigma == 0 is replaced by 1e-9 * L (near point mass).
    """

    def __init__(self, L, sigma, family="gamma"):
        L = np.asarray(L, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sd = np.where(sigma > 0, sigma, 1e-9 * L)
        self.family = family
        self.mean = L
        self.sd = sd
        if family == "gamma":
            self.alpha = (L / sd) ** 2
            self.theta = sd ** 2 / L
        elif family != "normal":
            raise ValueError(f"unknown family {family!r}")

    def cdf(self, t):
        """(m, G) matrix of F_k(t_g) for a 1-D grid t."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if self.family == "gamma":
            x = np.maximum(t, 0.0)[None, :] / self.theta[:, None]
            return gammainc(self.alpha[:, None], x)
        z = (t[None, :] - self.mean[:, None]) / self.sd[:, None]
        return ndtr(z)

    def pdf(self, t):
        """(m, G) matrix of f_k(t_g) for a 1-D grid t."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if self.family == "gamma":
            pos = t > 0
            x = np.where(pos, t, 1.0)[None, :] / self.theta[:, None]
            logf = ((self.alpha - 1.0)[:, None] * np.log(x) - x
                    - gammaln(self.alpha)[:, None] - np.log(self.theta)[:, None])
            return np.where(pos[None, :], np.exp(logf), 0.0)
        z = (t[None, :] - self.mean[:, None]) / self.sd[:, None]
        return np.exp(-0.5 * z * z) / (self.sd[:, None] * math.sqrt(2.0 * math.pi))

    def ppf(self, q):
        """Quantiles per driver: (m,) for scalar q, (m, len(q)) for an array."""
        q_arr = np.asarray(q, dtype=np.float64)
        q1 = np.atleast_1d(q_arr)
        if self.family == "gamma":
            res = gammaincinv(self.alpha[:, None], q1[None, :]) * self.theta[:, None]
        else:
            res = self.mean[:, None] + self.sd[:, None] * ndtri(q1[None, :])
        return res[:, 0] if q_arr.ndim == 0 else res


def _bounds(marg):
    lo = float(np.min(marg.ppf(Q_TAIL)))
    hi = float(np.max(marg.ppf(1.0 - Q_TAIL)))
    return lo, hi


def _simpson_weights(G):
    if G < 3 or G % 2 == 0:
        raise ValueError("G must be odd and >= 3")
    w = np.full(G, 2.0)
    w[1:-1:2] = 4.0
    w[0] = 1.0
    w[-1] = 1.0
    return w


def _pb_pmf(P):
    """Poisson-binomial pmf on a grid. P: (m, G) success probabilities -> (G, m+1)."""
    m, G = P.shape
    pmf = np.zeros((G, m + 1))
    pmf[:, 0] = 1.0
    for k in range(m):
        p = P[k][:, None]
        new = pmf * (1.0 - p)
        new[:, 1:] += pmf[:, :-1] * p
        pmf = new
    return pmf


def pboi(L, sigma, lam, G=257, family="gamma"):
    """E l_lambda(X) for independent X_k with means L and sds sigma (Theorem 2)."""
    L = np.asarray(L, dtype=np.float64)
    m = L.shape[0]
    lam = _resolve_lam(lam, m)
    marg = Marginals(L, sigma, family)
    Lam = np.zeros(m + 1)
    Lam[:m] = np.cumsum(lam[::-1])[::-1]
    lam_tot = Lam[0]
    lo, hi = _bounds(marg)
    w = _simpson_weights(G)

    def g_of(t):
        return _pb_pmf(marg.cdf(t)) @ Lam

    total = lam_tot * max(lo, 0.0)
    tpos = max(lo, 0.0)
    if hi > tpos:
        t = np.linspace(tpos, hi, G)
        h = (hi - tpos) / (G - 1)
        total += h / 3.0 * float(np.dot(w, g_of(t)))
    if lo < 0.0:
        t = np.linspace(lo, 0.0, G)
        h = (0.0 - lo) / (G - 1)
        total += h / 3.0 * float(np.dot(w, g_of(t) - lam_tot))
    return float(total)


def rank_prices(L, sigma, lam, G=257, family="gamma"):
    """varpi_k = sum_j lam_j P(R_k = j) (Proposition 3)."""
    L = np.asarray(L, dtype=np.float64)
    m = L.shape[0]
    lam = _resolve_lam(lam, m)
    marg = Marginals(L, sigma, family)
    lo, hi = _bounds(marg)
    t = np.linspace(lo, hi, G)
    h = (hi - lo) / (G - 1)
    w = _simpson_weights(G)
    P = marg.cdf(t)
    f = marg.pdf(t)
    out = np.zeros(m)
    for k in range(m):
        pmf = _pb_pmf(np.delete(P, k, axis=0))
        Pr = (h / 3.0) * (w[:, None] * f[k][:, None] * pmf).sum(axis=0)
        out[k] = float(np.dot(lam, Pr))
    return out


def sigma_prices(L, sigma, lam, h=0.05, G=257, family="gamma"):
    """psi_k = d E l_lambda / d sigma_k by forward difference with step h*sigma_k.

    For lam == "var" the exact derivative 2(m-1)/m^2 * sigma_k is returned.
    """
    L = np.asarray(L, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    m = L.shape[0]
    if isinstance(lam, str) and lam == "var":
        return 2.0 * (m - 1) / float(m) ** 2 * sigma
    lamv = _resolve_lam(lam, m)
    sig_eff = np.where(sigma > 0, sigma, 1e-9 * L)
    base = pboi(L, sigma, lamv, G=G, family=family)
    psi = np.empty(m)
    for k in range(m):
        step = h * sig_eff[k]
        s2 = sigma.copy()
        s2[k] = sig_eff[k] + step
        psi[k] = (pboi(L, s2, lamv, G=G, family=family) - base) / step
    return psi


def _abs_normal_mean(d, s):
    """E|Z| for Z ~ N(d, s^2), elementwise; s == 0 gives |d|."""
    d = np.asarray(d, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    safe = np.where(s > 0, s, 1.0)
    z = d / safe
    val = safe * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * z * z) + d * (1.0 - 2.0 * ndtr(-z))
    return np.where(s > 0, val, np.abs(d))


def gmd_normal_closed(L, sigma):
    """E GMD(X) for independent normals (Proposition 4): (1/m^2) sum_{i != j} E|X_i - X_j|."""
    L = np.asarray(L, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    m = L.shape[0]
    d = L[:, None] - L[None, :]
    s = np.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
    vals = _abs_normal_mean(d, s)
    np.fill_diagonal(vals, 0.0)
    return float(vals.sum() / float(m) ** 2)


def expost_var(L, sigma):
    """E Var_emp(X) = Var_emp(L) + (m-1)/m^2 sum sigma_k^2 (population variance)."""
    L = np.asarray(L, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    m = L.shape[0]
    var_emp = float(np.mean((L - L.mean()) ** 2))
    return var_emp + (m - 1) / float(m) ** 2 * float(np.sum(sigma ** 2))


def objective(L, W, v, measure, G=257, family="gamma"):
    """Ex-post objective of a plan with lengths L and quadratic lengths W (sigma = v sqrt W)."""
    L = np.asarray(L, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    sigma = v * np.sqrt(W)
    if measure == "var":
        return expost_var(L, sigma)
    return pboi(L, sigma, lam_vector(measure, L.shape[0]), G=G, family=family)
