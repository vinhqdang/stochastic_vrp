"""Value-innovation monitoring of a fixed recourse policy (paper 2 support).

Setting. A vehicle serves stops k = 1..m in order. At stop k the net load
change is g_k = p_k - d_k (pickup minus delivery, independent gamma
variables under the planning law). The running load is W_k = W_{k-1} + g_k
with W_0 = 0. A breach (W_k > B) costs E[k-1] and ends the route; otherwise
a fitted policy (``core.costs.fit_lsm_actions`` models, executed as in
``core.costs.simulate_actions``) may hand off (cost H[k-1]), restock
(cost R[k-1], load reset to 0) or continue.

What this module provides.

* ``RouteValue`` computes, by backward recursion on a load grid, the exact
  expected remaining cost of that FIXED policy under the planning law:
  ``U[k](w)`` is the pre-observation value at stop k given load w, and
  ``post[k](w)`` is the value just after observing stop k's demand.
* ``RouteValue.innovation(k, w, g)`` returns the value innovation
  ``I = post(k, w + g) - U(k, w)`` and a predictable bound ``L = U(k, w) -
  (lower bound on post over the demand range)`` so that ``I >= -L``.
* ``ValueInnovationMonitor`` runs a bank of betting e-processes on the
  normalised innovations ``x = (I - eps) / (L + eps)``.
* ``run_route_day`` executes the policy on realised demands and feeds the
  innovations to a monitor.

Null hypothesis and validity. Let F_{k-1} be the history before stop k.
The null is ``E[I_k | F_{k-1}] <= eps`` for every stop k that is reached.
Under the planning law this holds with eps = 0, because U is exactly the
conditional mean of post(k, w + g) over the planning law of g_k. Since
``I >= -L`` and ``L >= 0`` are functions of the past only (they depend on
the load before stop k, not on g_k), the bet ``x = (I - eps)/(L + eps)``
satisfies ``x >= -1`` and ``E[x | F_{k-1}] <= 0`` under the null. Hence each
factor ``1 + f * x`` with ``0 <= f <= 1`` is a nonnegative supermartingale
factor with conditional mean at most 1, and so is the average over the
fractions. Ville's inequality then gives ``P(sup_n wealth_n >= 1/alpha) <=
alpha`` for every stopping rule, which is the anytime-valid alarm.
Clipping ``x`` at -1 only matters outside the demand range used for ``L``.

Numerical notes. The load grid is ``W_lo + j*h`` for ``j = 0..G-1`` with
``W_lo < 0`` so that the grid covers all loads that can occur. Demand pmfs
live on the same lattice ``h * Z``, so ``w + g`` for a grid point ``w`` is
again a grid point. Realised loads are continuous, so ``post`` and ``U`` are
evaluated by linear interpolation. Loads below ``W_lo`` are clipped to
``W_lo``; loads above ``B`` are breaches.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.signal import fftconvolve


# ============================================================
# Discrete demand laws
# ============================================================


def _trim(pmf: np.ndarray, offset: int):
    """Keep the lattice points where pmf > 1e-12 * max; renormalise."""
    keep = np.flatnonzero(pmf > 1e-12 * pmf.max())
    lo, hi = int(keep[0]), int(keep[-1])
    out = pmf[lo:hi + 1]
    return out / out.sum(), offset + lo


def _gamma_pmf(mean: float, cv: float, h: float):
    """pmf of a gamma(shape 1/cv^2, scale mean*cv^2) variable on h*Z_{>=0}.

    Returns (pmf, start) with mass pmf[t] at lattice index start + t.
    A zero mean is a point mass at 0.
    """
    if mean <= 0.0:
        return np.array([1.0]), 0
    sd = cv * mean
    jmax = int(math.ceil((mean + 20.0 * sd) / h)) + 1
    x = np.arange(jmax + 1) * h
    pdf = stats.gamma.pdf(x, a=1.0 / cv ** 2, scale=mean * cv ** 2)
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    return _trim(pdf / pdf.sum(), 0)


def _g_pmf(pbar: float, dbar: float, cv: float, h: float):
    """pmf of g = p - d on the lattice h*Z, returned as (pmf, g_start)."""
    p, ap = _gamma_pmf(pbar, cv, h)
    d, ad = _gamma_pmf(dbar, cv, h)
    conv = fftconvolve(p, d[::-1])
    conv = np.clip(conv, 0.0, None)
    conv = conv / conv.sum()
    # Index r of conv corresponds to lattice index (ap - ad - (len(d)-1)) + r.
    return _trim(conv, ap - ad - (len(d) - 1))


# ============================================================
# Route value
# ============================================================


class RouteValue:
    """Exact expected cost and value innovations of a fixed route policy.

    Parameters
    ----------
    pbar, dbar : per-stop mean pickups / deliveries in route order (length m).
    B : departure load slack; a breach is W > B.
    H, E, R : handoff, breach and restock prices, length m (0-based k-1).
    models : {k: (iso, F_k)} as returned by ``core.costs.fit_lsm_actions``.
        The policy at stop k is applied exactly as in ``simulate_actions``.
    cv : coefficient of variation of both gamma components.
    G : approximate number of load-grid points.
    """

    def __init__(self, pbar, dbar, B, H, E, R, models, cv=0.30, G=4000):
        pbar = np.asarray(pbar, dtype=float).ravel()
        dbar = np.asarray(dbar, dtype=float).ravel()
        m = len(pbar)
        if len(dbar) != m:
            raise ValueError("pbar and dbar must have the same length")
        H = np.asarray(H, dtype=float).ravel()
        E = np.asarray(E, dtype=float).ravel()
        R = np.asarray(R, dtype=float).ravel()
        if min(len(H), len(E), len(R)) < m:
            raise ValueError("H, E, R must have length m")
        self.m = m
        self.B = float(B)
        self.H, self.E, self.R = H, E, R
        self.models = models or {}
        self.cv = float(cv)
        self.G = int(G)

        # Load grid: W_lo + j*h, j = 0..G-1, last point exactly B.
        W_lo0 = -float(dbar.sum()) * (1.0 + 6.0 * self.cv)
        h0 = (self.B - W_lo0) / (self.G - 1)
        self.W_lo = W_lo0 - h0
        self.h = (self.B - self.W_lo) / (self.G - 1)
        self.grid = np.linspace(self.W_lo, self.B, self.G)

        # Demand pmfs of g_k, k = 1..m (index k-1 in the arrays).
        self._pmf = [None] * (m + 1)
        self._g0 = [0] * (m + 1)
        for k in range(1, m + 1):
            pmf, g0 = _g_pmf(pbar[k - 1], dbar[k - 1], self.cv, self.h)
            self._pmf[k] = pmf
            self._g0[k] = g0

        # Backward recursion. _U[k] and _post[k] are arrays on the grid.
        self._U = [None] * (m + 2)
        self._post = [None] * (m + 2)
        for k in range(m, 0, -1):
            if k == m:
                post = np.zeros(self.G)
            else:
                post = self._policy_post(k)
            self._post[k] = post
            self._U[k] = self._expect(k, post)

    # --- recursion pieces ------------------------------------------------

    def _policy_post(self, k: int) -> np.ndarray:
        """post[k] on the grid for a non-final stop k (uses U[k+1])."""
        U_next = self._U[k + 1]
        entry = self.models.get(k)
        if entry is None:
            return U_next.copy()
        iso, F = entry
        chat = np.asarray(iso.predict(self.grid), dtype=float).ravel()
        v_ho = self.H[k - 1]
        v_rs = self.R[k - 1] + F
        do_ho = (v_ho < chat) & (v_ho <= v_rs)
        do_rs = (v_rs < chat) & ~do_ho
        restock_value = self.R[k - 1] + self.U(k + 1, 0.0)
        return np.where(do_ho, v_ho,
                        np.where(do_rs, restock_value, U_next))

    def _expect(self, k: int, post: np.ndarray) -> np.ndarray:
        """U[k](w_j) = sum_g pmf_k(g) post[k](w_j + g) on the grid."""
        pmf = self._pmf[k]
        g0 = self._g0[k]
        G = self.G
        idx = (np.arange(G)[:, None] + g0 + np.arange(len(pmf))[None, :])
        vals = np.where(idx >= G, self.E[k - 1],
                        post[np.clip(idx, 0, G - 1)])
        return vals @ pmf

    # --- public interface --------------------------------------------------

    def U(self, k: int, w: float) -> float:
        """Pre-observation value at stop k with load w before stop k."""
        w = min(max(float(w), self.W_lo), self.B)
        return float(np.interp(w, self.grid, self._U[k]))

    def post(self, k: int, w: float) -> float:
        """Value just after observing stop k's demand, with load w."""
        w = float(w)
        if w > self.B:
            return float(self.E[k - 1])
        return float(np.interp(max(w, self.W_lo), self.grid, self._post[k]))

    def g_support(self, k: int):
        """(g_min, g_max): extreme lattice values of the planning pmf of g_k."""
        pmf = self._pmf[k]
        g0 = self._g0[k]
        return g0 * self.h, (g0 + len(pmf) - 1) * self.h

    def _range_min(self, k: int, lo: float, hi: float) -> float:
        """Minimum of post[k] over the load interval [lo, hi].

        post[k] is piecewise linear on [W_lo, B] with vertices on the grid
        and equals E[k-1] on (B, inf). The minimum over an interval is
        therefore attained at an endpoint, a grid vertex, or the breach
        plateau.
        """
        post = self._post[k]
        best = math.inf
        if lo <= self.B:
            x0 = max(lo, self.W_lo)
            top = min(hi, self.B)
            best = min(best, self.post(k, x0))
            if top >= x0:
                best = min(best, self.post(k, top))
                j0 = max(0, int(math.ceil((x0 - self.W_lo) / self.h - 1e-9)))
                j1 = min(self.G - 1,
                         int(math.floor((top - self.W_lo) / self.h + 1e-9)))
                if j1 >= j0:
                    best = min(best, float(post[j0:j1 + 1].min()))
        if hi > self.B:
            best = min(best, float(self.E[k - 1]))
        return best

    def innovation(self, k: int, w_prev: float, g: float):
        """Value innovation at stop k.

        Returns (I, L) with I = post(k, w_prev + g) - U(k, w_prev) and
        L = U(k, w_prev) - min over ALL loads of post(k, .) (including the
        breach value E[k-1]). L depends on the past only, and I >= -L holds
        for every realised demand g whatever its law, so the betting
        e-values stay valid when the planning law is misspecified.
        """
        w_prev = float(w_prev)
        U_w = self.U(k, w_prev)
        mn = min(float(np.min(self._post[k])), float(self.E[k - 1]))
        I = self.post(k, w_prev + float(g)) - U_w
        return float(I), float(max(U_w - mn, 0.0))

    def expected_cost(self) -> float:
        """Exact expected cost of the fixed policy from the empty route."""
        return self.U(1, 0.0)


# ============================================================
# Monitor
# ============================================================


class ValueInnovationMonitor:
    """Bank of betting e-processes on normalised value innovations.

    For each fraction f_j a wealth W_j starts at 1. On each update the bet is
    x = (I - eps) / (L + eps) (0 if L + eps <= 0), truncated below at -1, and
    W_j <- W_j * (1 + f_j * x). The monitor's wealth is the mean of the W_j.
    The alarm is raised when the wealth reaches 1/alpha and stays raised until
    ``reset``.
    """

    def __init__(self, alpha=0.05, fractions=(0.05, 0.1, 0.2, 0.4, 0.8)):
        self.alpha = float(alpha)
        self.fractions = np.asarray(fractions, dtype=float)
        self._W = np.ones(len(self.fractions))
        self._alarmed = False

    def update(self, I: float, L: float, eps: float = 0.0) -> bool:
        """Feed one innovation; return True if the monitor is in alarm."""
        denom = L + eps
        x = (I - eps) / denom if denom > 0.0 else 0.0
        x = max(x, -1.0)
        self._W = self._W * (1.0 + self.fractions * x)
        if self.wealth >= 1.0 / self.alpha:
            self._alarmed = True
        return self._alarmed

    def reset(self):
        self._W = np.ones(len(self.fractions))
        self._alarmed = False

    @property
    def wealth(self) -> float:
        return float(self._W.mean())


# ============================================================
# Route-day simulation
# ============================================================


def run_route_day(rv: RouteValue, g_row, monitor=None, eps: float = 0.0) -> dict:
    """Execute the fixed policy on realised demands g_row (length m).

    At every observed stop the innovation (I, L) is computed from the load
    before the stop, then the monitor (if given) is updated. The breach,
    handoff, restock and completion rules follow ``simulate_actions``.
    The monitor is not reset here; callers reset it between days.
    """
    m = rv.m
    W = 0.0
    cost = 0.0
    restocks = 0
    innovations = []
    alarm_stop = None
    action = "complete"
    for k in range(1, m + 1):
        g = float(g_row[k - 1])
        I, L = rv.innovation(k, W, g)
        innovations.append(I)
        if monitor is not None:
            if monitor.update(I, L, eps) and alarm_stop is None:
                alarm_stop = k
        W += g
        if W > rv.B:
            cost += float(rv.E[k - 1])
            action = "emergency"
            break
        if k == m:
            break
        entry = rv.models.get(k)
        if entry is None:
            continue
        iso, F = entry
        chat = float(np.asarray(iso.predict([W]), dtype=float).ravel()[0])
        v_ho = float(rv.H[k - 1])
        v_rs = float(rv.R[k - 1]) + float(F)
        if v_ho < chat and v_ho <= v_rs:
            cost += v_ho
            action = "handoff"
            break
        if v_rs < chat:
            cost += float(rv.R[k - 1])
            restocks += 1
            W = 0.0
    return {
        "cost": cost,
        "action": action,
        "restocks": restocks,
        "innovations": innovations,
        "alarm_stop": alarm_stop,
    }
