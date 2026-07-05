"""Scenario generators for SVRPSPD W-DRO.

Generates (N, 2n) demand scenario matrices for empirical distribution F_0.

Conventions:
    - Per-customer independent: each customer j's (d_j, p_j) drawn independently.
    - Within a customer, d_j and p_j are independent (no copula in v1).
    - Marginal mean of d_j = nominal_d[j], marginal mean of p_j = nominal_p[j].
    - Truncation: clip to [0, clip_at] where clip_at defaults to inst.Q (M14).
    - Degenerate corner case: depends on distribution
        * constant            -> always nominal
        * gamma, lognormal    -> nominal iff cv == 0
        * bimodal             -> nominal iff bimodal_spread == 0 AND bimodal_cv == 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.instance import Instance


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""

    distribution: str = "gamma"
    cv: float = 0.2
    seed: int = 42
    clip_at: float | None = None
    bimodal_spread: float = 0.5
    bimodal_cv: float = 0.1

    def __post_init__(self):
        if self.cv < 0:
            raise ValueError(f"CV must be >= 0, got {self.cv}")
        if self.distribution not in ("constant", "gamma", "lognormal", "bimodal"):
            raise ValueError(f"Unknown distribution: {self.distribution!r}")


def generate_scenarios(
    inst: Instance,
    n_scenarios: int,
    config: ScenarioConfig,
) -> np.ndarray:
    """Generate scenario matrix X of shape (N, 2n).

    X[s, 2j]   = d_j realization in scenario s
    X[s, 2j+1] = p_j realization in scenario s
    """
    n = inst.n
    N = n_scenarios
    rng = np.random.default_rng(config.seed)
    clip_at = config.clip_at if config.clip_at is not None else inst.Q

    X = np.empty((N, 2 * n), dtype=np.float64)
    nominal = inst.nominal_xi()

    # ---- degenerate (variance=0) check, distribution-aware ----
    is_degenerate = (
        config.distribution == "constant"
        or (config.distribution in ("gamma", "lognormal") and config.cv == 0.0)
        or (
            config.distribution == "bimodal"
            and config.bimodal_spread == 0.0
            and config.bimodal_cv == 0.0
        )
    )
    if is_degenerate:
        X[:] = nominal
        np.clip(X, 0.0, clip_at, out=X)
        return X

    # ---- per-dimension sampling ----
    for k in range(2 * n):
        mu = nominal[k]
        if mu <= 0:
            X[:, k] = 0.0
            continue

        if config.distribution == "gamma":
            X[:, k] = _sample_gamma(rng, mu, config.cv, N)
        elif config.distribution == "lognormal":
            X[:, k] = _sample_lognormal(rng, mu, config.cv, N)
        elif config.distribution == "bimodal":
            X[:, k] = _sample_bimodal(
                rng, mu, config.bimodal_spread, config.bimodal_cv, N
            )

    np.clip(X, 0.0, clip_at, out=X)
    return X


# ============================================================
# Per-distribution samplers
# ============================================================


def _sample_gamma(
    rng: np.random.Generator, mu: float, cv: float, n: int
) -> np.ndarray:
    """Gamma with mean mu and CV = cv."""
    shape = 1.0 / (cv * cv)
    scale = (cv * cv) * mu
    return rng.gamma(shape=shape, scale=scale, size=n)


def _sample_lognormal(
    rng: np.random.Generator, mu: float, cv: float, n: int
) -> np.ndarray:
    """LogNormal with mean mu and CV = cv."""
    sigma_log2 = np.log(1.0 + cv * cv)
    sigma_log = np.sqrt(sigma_log2)
    mean_log = np.log(mu) - 0.5 * sigma_log2
    return rng.lognormal(mean=mean_log, sigma=sigma_log, size=n)


def _sample_bimodal(
    rng: np.random.Generator,
    mu: float,
    spread: float,
    inner_cv: float,
    n: int,
) -> np.ndarray:
    """50/50 mixture of two Gammas at mu*(1-spread) and mu*(1+spread)."""
    mu_low = max(mu * (1.0 - spread), 1e-6)
    mu_high = mu * (1.0 + spread)
    component = rng.random(n) < 0.5
    samples = np.where(
        component,
        _sample_gamma(rng, mu_low, inner_cv, n),
        _sample_gamma(rng, mu_high, inner_cv, n),
    )
    return samples


# ============================================================
# Diagnostics
# ============================================================


def empirical_moments(X: np.ndarray) -> dict:
    """Compute per-column empirical moments and overall stats."""
    return {
        "shape": X.shape,
        "col_means": X.mean(axis=0),
        "col_stds": X.std(axis=0, ddof=1),
        "col_cvs": np.where(
            X.mean(axis=0) > 0,
            X.std(axis=0, ddof=1) / np.maximum(X.mean(axis=0), 1e-12),
            0.0,
        ),
        "overall_mean": X.mean(),
        "overall_max": X.max(),
        "overall_min": X.min(),
    }