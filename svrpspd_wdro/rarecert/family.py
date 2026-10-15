"""Marginal-family variants of the route simulator (RARECERT R1, sensitivity study).

FamilyRouteSim changes only the marginal demand law of quantum.rare.RouteSim; the
copula, the stage-load recursion, the threshold search and the IS estimators are
inherited unchanged. family = "gamma" reproduces RouteSim exactly. family =
"lognormal" uses, for every active customer, the lognormal law with the same mean
dbar (pbar) and coefficient of variation cv:

    sigma^2 = log(1 + cv^2),   mu = log(mean) - sigma^2 / 2,
    d = exp(mu + sigma * Phi^{-1}(u)),   u = Phi(sqrt(rho) W + sqrt(1 - rho) E).

The conditional discretised DP of quantum.rare (_conditional_pmfs) is gamma-specific,
so exact_route_prob is not valid for the lognormal family; the sensitivity driver
uses a cross-entropy mixture reference for those rows and flags them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr, ndtri

from quantum.rare import RouteSim, _U_CLIP

FAMILIES = ("gamma", "lognormal")


class FamilyRouteSim(RouteSim):
    """RouteSim with a selectable marginal family (see module docstring)."""

    def __init__(self, dbar, pbar, cv: float = 0.3, rho: float = 0.6, family: str = "gamma"):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
        super().__init__(dbar, pbar, cv=cv, rho=rho)
        self.family = family
        s2 = math.log1p(self.cv * self.cv)
        self.ln_sigma = math.sqrt(s2)
        with np.errstate(divide="ignore"):
            self.ln_mu_d = np.where(self.act_d, np.log(np.where(self.act_d, self.dbar, 1.0)) - 0.5 * s2, 0.0)
            self.ln_mu_p = np.where(self.act_p, np.log(np.where(self.act_p, self.pbar, 1.0)) - 0.5 * s2, 0.0)

    def demands(self, x):
        if self.family == "gamma":
            return super().demands(x)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (N, {self.dim}), got {x.shape}")
        L = self.L
        Wd = x[:, 0:1]
        Wp = x[:, 1:2]
        Ed = x[:, 2:2 + L]
        Ep = x[:, 2 + L:2 + 2 * L]
        ud = np.clip(ndtr(self._sr * Wd + self._s1 * Ed), _U_CLIP, 1.0 - _U_CLIP)
        up = np.clip(ndtr(self._sr * Wp + self._s1 * Ep), _U_CLIP, 1.0 - _U_CLIP)
        d = np.exp(self.ln_mu_d[None, :] + self.ln_sigma * ndtri(ud))
        p = np.exp(self.ln_mu_p[None, :] + self.ln_sigma * ndtri(up))
        d = np.where(self.act_d[None, :], d, 0.0)
        p = np.where(self.act_p[None, :], p, 0.0)
        return d, p
