"""RARECERT R2 route-day simulator: stochastic travel and service times on top of
the demand model of quantum.rare.RouteSim. No exact DP exists for this model.

Spec: papers/rarecert/THEORY.md, scratchpad spec r2 (section 1). The demand part
(gamma marginals, one-factor copula with rho, cv) is RouteSim itself, imported and
not copied. The route is r = (c_1..c_L) with depot 0; legs l = 0..L are
0 -> c_1, c_l -> c_{l+1}, c_L -> 0, with lengths in km taken from the instance matrix.

Inputs x ~ N(0, I) in R^(3L+4), in this order:
    x[0 : 2L+2]          demand inputs (W_d, W_p, E_d[0..L-1], E_p[0..L-1]), RouteSim order
    x[2L+2]              congestion factor V
    x[2L+3 : 3L+4]       leg noises H_0..H_L

Travel time of leg l (minutes), mean tbar_l:
    T_l = tbar_l exp(sigma_T (sqrt(rho_T) V + sqrt(1-rho_T) H_l) - sigma_T^2 / 2).
Service time at the i-th stop (minutes): S_i = S0 + S1 (d_i + p_i) / 1000 (demands in kg).
Arrival at stop i (1..L): A_i = sum_{l<i} T_l + sum_{j<i} S_j; return R = sum_l T_l + sum_j S_j.

Failure modes, each a scalar g_m(x), failure iff g_m(x) > 0 (columns of modes(x)):
    load_k  (k = 0..L):  L_k - Qc
    late_i  (i = 1..L):  A_i - (Abar_i + slack)
    return:              R - (Rbar + 2 slack)
where Abar_i, Rbar are the values at x = 0. The route fails iff some mode fails.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from quantum.rare import RouteSim, _loads_rows

SPEED_KMH = 40.0
SIGMA_T = 0.25
RHO_T = 0.5
S0 = 10.0
S1 = 6.0
KM_PER_UNIT = 1e-4          # instance matrix units are 0.1 m (scale 10000 of parse_dethloff)
EVAL_CHUNK = 50_000
EXP_CLIP = 300.0


def legs_from_matrix(D, nodes, units_per_km: float = 1e4) -> np.ndarray:
    """Leg lengths in km of the route depot -> nodes[0] -> ... -> nodes[-1] -> depot,
    read from the full instance matrix D (node ids index D directly, depot 0)."""
    D = np.asarray(D, dtype=np.float64)
    path = [0] + [int(i) for i in nodes] + [0]
    legs = np.array([D[a, b] for a, b in zip(path[:-1], path[1:])], dtype=np.float64)
    return legs / units_per_km


class RouteDay:
    """Route-day simulator of one route: demand inputs plus travel and service times.

    Parameters
    ----------
    dbar, pbar : array_like, shape (L,)   means per stop in route order (kg).
    legs_km : array_like, shape (L+1,)    leg lengths in km (see legs_from_matrix).
    cv, rho : demand coefficient of variation and copula loading (RouteSim).
    Qc, slack : load capacity (kg) and lateness slack (minutes). Qc = 0 and
        slack = 0 by default; use with_thresholds to set them on a copy.
    """

    def __init__(self, dbar, pbar, legs_km, cv: float = 0.3, rho: float = 0.6,
                 Qc: float = 0.0, slack: float = 0.0):
        self.demand = RouteSim(dbar, pbar, cv=cv, rho=rho)
        L = self.demand.L
        self.L = L
        legs = np.asarray(legs_km, dtype=np.float64).ravel()
        if legs.shape != (L + 1,):
            raise ValueError(f"legs_km must have length {L + 1}, got {legs.shape}")
        if not (np.all(np.isfinite(legs)) and np.all(legs >= 0)):
            raise ValueError("leg lengths must be finite and non-negative")
        self.legs_km = legs
        self.tbar = 60.0 * legs / SPEED_KMH
        self.n_dem = 2 * L + 2
        self.i_V = self.n_dem
        self.i_H = self.n_dem + 1
        self.dim = self.n_dem + 1 + (L + 1)
        # nominal (all inputs zero): stop arrivals and return time
        x0 = np.zeros((1, self.dim))
        arr0, ret0 = self.arrivals_and_return(x0)
        self.Abar = arr0[0].copy()
        self.Rbar = float(ret0[0])
        self._set_thresholds(float(Qc), float(slack))

    # ------------------------------------------------------------------ thresholds

    def _set_thresholds(self, Qc: float, slack: float) -> None:
        self.Qc = Qc
        self.slack = slack
        # positive per-mode normalisers (sign of every mode is unchanged)
        self.scale = np.concatenate([
            np.full(self.L + 1, max(Qc, 1.0)),
            np.maximum(self.Abar, 1.0),
            [max(self.Rbar, 1.0)],
        ])

    def with_thresholds(self, Qc: float, slack: float) -> "RouteDay":
        """Shallow copy with new Qc and slack; the demand model and nominals are shared."""
        new = copy.copy(self)
        new._set_thresholds(float(Qc), float(slack))
        return new

    # ------------------------------------------------------------------ mode layout

    @property
    def n_modes(self) -> int:
        return 2 * self.L + 2

    @property
    def mode_names(self) -> list[str]:
        return ([f"load_{k}" for k in range(self.L + 1)]
                + [f"late_{i}" for i in range(1, self.L + 1)] + ["return"])

    def load_index(self) -> np.ndarray:
        return np.arange(0, self.L + 1)

    def time_index(self) -> np.ndarray:
        return np.arange(self.L + 1, self.n_modes)

    def is_load(self, m: int) -> bool:
        return 0 <= int(m) <= self.L

    # ------------------------------------------------------------------ simulator

    def _check(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (N, {self.dim}), got {x.shape}")
        return x

    def travel_times(self, x) -> np.ndarray:
        """Leg travel times in minutes, shape (N, L+1)."""
        x = self._check(x)
        V = x[:, self.i_V:self.i_V + 1]
        H = x[:, self.i_H:self.i_H + self.L + 1]
        z = math.sqrt(RHO_T) * V + math.sqrt(1.0 - RHO_T) * H
        # the exponent is clipped at +-EXP_CLIP (about 1200 standard deviations of z):
        # it only guards the ray search of dominating points against overflow to inf/nan
        expo = np.clip(SIGMA_T * z - 0.5 * SIGMA_T * SIGMA_T, -EXP_CLIP, EXP_CLIP)
        return self.tbar[None, :] * np.exp(expo)

    def _all(self, x):
        """Stage loads (N, L+1), arrivals (N, L), return time (N,) for checked x."""
        d, p = self.demand.demands(x[:, :self.n_dem])
        loads = _loads_rows(d, p)
        S = S0 + S1 * (d + p) / 1000.0                       # (N, L) service times
        T = self.travel_times(x)                             # (N, L+1)
        cumT = np.cumsum(T, axis=1)
        cumS = np.cumsum(S, axis=1)
        # A_i (i = 1..L) uses legs 0..i-1 and services of stops 1..i-1 (columns 0..i-2)
        S_before = np.concatenate([np.zeros((x.shape[0], 1)), cumS[:, :self.L - 1]], axis=1)
        arr = cumT[:, :self.L] + S_before
        ret = cumT[:, self.L] + cumS[:, self.L - 1]
        return loads, arr, ret

    def arrivals_and_return(self, x):
        """Arrival times A_1..A_L (N, L) and return time R (N,), in minutes."""
        x = self._check(x)
        _, arr, ret = self._all(x)
        return arr, ret

    def loads(self, x) -> np.ndarray:
        """Stage loads L_0..L_L (N, L+1) in kg, RouteSim's recursion."""
        x = self._check(x)
        d, p = self.demand.demands(x[:, :self.n_dem])
        return _loads_rows(d, p)

    def modes(self, x, chunk: int = EVAL_CHUNK) -> np.ndarray:
        """Raw mode values g_m(x), shape (N, 2L+2); columns as in mode_names."""
        x = self._check(x)
        out = np.empty((x.shape[0], self.n_modes), dtype=np.float64)
        for s in range(0, x.shape[0], chunk):
            xs = x[s:s + chunk]
            loads, arr, ret = self._all(xs)
            blk = out[s:s + chunk]
            blk[:, :self.L + 1] = loads - self.Qc
            blk[:, self.L + 1:2 * self.L + 1] = arr - (self.Abar[None, :] + self.slack)
            blk[:, 2 * self.L + 1] = ret - (self.Rbar + 2.0 * self.slack)
        return out

    def normalised_modes(self, x, chunk: int = EVAL_CHUNK) -> np.ndarray:
        """modes(x) divided by the positive per-mode scale (same signs)."""
        return self.modes(x, chunk=chunk) / self.scale[None, :]

    def fail(self, x, idx=None, chunk: int = EVAL_CHUNK) -> np.ndarray:
        """Route failure indicator (float 0/1) over the modes in idx (default all)."""
        g = self.modes(x, chunk=chunk)
        if idx is not None:
            g = g[:, np.asarray(idx, dtype=int)]
        return (g.max(axis=1) > 0.0).astype(np.float64)
