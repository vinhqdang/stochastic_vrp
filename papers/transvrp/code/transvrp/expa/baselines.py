"""Classical baseline ladder for EXPERIMENT A (module M7).

EXPERIMENT_A.md section 6 and M7. ``classical_ladder`` fits the classical
policies on fitting paths; ``eval_classical`` simulates them on evaluation
paths. Neural labels are produced elsewhere.

Labels: none, tuned_tau, baton_eq, dp, otr21 (fit), oracle (eval only).
``baton_1e6`` and ``dp_1e6`` are aliases of ``baton_eq`` and ``dp`` for a
10^6-path fit set, and are accepted only with exactly 10^6 rows.

``otr21`` needs the empirical per-stop statistics (section 3.1) to build its
features. They are passed as ``stats``; without them the label is refused.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baton.costs import (fit_lsm_general, oracle_actions_general,
                           oracle_costs_general, simulate_tau_general,
                           simulate_v2_general, tune_tau_general)
from ..baton.dp_exec import fit_dp
from ..baton.otr21 import _feature_arrays, _stack, fit_lsm_rich
from ..baton.otr2 import _ConstantModel, fit_otr_peak
from .features import RouteStats
from .paths import RouteSpec
from .simulate import simulate_policy

LABELS = ("none", "tuned_tau", "baton_eq", "baton_1e6", "dp", "dp_1e6",
          "otr21", "oracle")
FIT_ALIAS = {"baton_1e6": "baton_eq", "dp_1e6": "dp"}
BIG_PRICE = 1e12
N_BIG = 10 ** 6


@dataclass
class TunedTau:
    models: dict
    tau: float


@dataclass
class RichFit:
    models: dict
    mu: np.ndarray
    sig: np.ndarray


class _RichPredictor:
    """Row-wise adapter so that the OTR-2.1 models run through
    ``simulate_policy``. Features are computed from ``cache.g`` for the rows
    asked for; every feature is a row-wise function of g (section 9, M7)."""

    def __init__(self, model, mu: np.ndarray, sig: np.ndarray):
        self.model = model
        self.mu = mu
        self.sig = sig

    def predict(self, cache, k: int, idx: np.ndarray) -> np.ndarray:
        g = np.asarray(cache.g)[np.asarray(idx, dtype=np.int64)]
        cum, fhat, glast, gmax = _feature_arrays(g, self.mu, self.sig)
        k_idx = k - 1
        if isinstance(self.model, _ConstantModel):
            return np.asarray(self.model.predict(cum[:, k_idx]), dtype=float)
        return np.asarray(self.model.predict(_stack(k_idx, cum, fhat, glast, gmax)),
                          dtype=float)


class _GView:
    """Minimal stand-in for a RouteCache: only ``g`` and ``W`` are read by the
    classical predictors, so no statistics are built."""

    def __init__(self, g: np.ndarray):
        self.g = np.asarray(g, dtype=float)
        self.W = np.cumsum(self.g, axis=1)


def _check_label(lab: str, n_rows: int) -> None:
    if lab not in LABELS:
        raise ValueError(f"unknown label {lab!r}; expected one of {LABELS}")
    if lab in FIT_ALIAS and n_rows != N_BIG:
        raise ValueError(f"label {lab!r} requires a fit set of exactly 10^6 rows")


def classical_ladder(spec: RouteSpec, g_fit: np.ndarray, which: tuple[str, ...],
                     stats: RouteStats | None = None) -> dict[str, object]:
    """Fitted classical models per label (``None`` for oracle)."""
    g_fit = np.asarray(g_fit, dtype=float)
    m = spec.m
    if g_fit.ndim != 2 or g_fit.shape[1] != m:
        raise ValueError("g_fit must have shape (N, m)")
    B, H, E = spec.B, np.asarray(spec.H, float), np.asarray(spec.E, float)
    out: dict[str, object] = {}
    for lab in which:
        _check_label(lab, g_fit.shape[0])
        base = FIT_ALIAS.get(lab, lab)
        if base == "oracle":
            out[lab] = None
        elif base == "none":
            out[lab] = fit_lsm_general(g_fit, B, np.full(m, BIG_PRICE), E)
        elif base == "tuned_tau":
            prob = fit_otr_peak(g_fit, B)
            out[lab] = TunedTau(models=prob,
                                tau=tune_tau_general(g_fit, B, H, E, prob))
        elif base == "baton_eq":
            out[lab] = fit_lsm_general(g_fit, B, H, E)
        elif base == "dp":
            out[lab] = fit_dp(g_fit, B, H, E)
        elif base == "otr21":
            if stats is None:
                raise ValueError("label 'otr21' needs the RouteStats of the fit rows")
            mu = np.asarray(stats.mu_p, float) - np.asarray(stats.mu_d, float)
            sig = np.sqrt(np.asarray(stats.sd_p, float) ** 2
                          + np.asarray(stats.sd_d, float) ** 2)
            out[lab] = RichFit(models=fit_lsm_rich(g_fit, mu, sig, B, H, E),
                               mu=mu, sig=sig)
    return out


def _result(stats: dict) -> dict:
    return {"mean_cost": stats["mean_cost"], "handoff_rate": stats["handoff_rate"],
            "fail_rate": stats["fail_rate"], "costs": stats["costs"]}


def eval_classical(spec: RouteSpec, models: dict[str, object],
                   g_eval: np.ndarray) -> dict[str, dict]:
    """label -> {mean_cost, handoff_rate, fail_rate, costs} on ``g_eval``.

    Billing is always the route's H (section 6: H_bill = H except for none).
    """
    g = np.asarray(g_eval, dtype=float)
    B, H, E = spec.B, np.asarray(spec.H, float), np.asarray(spec.E, float)
    H_big = np.full(spec.m, BIG_PRICE)
    out: dict[str, dict] = {}
    for lab, mdl in models.items():
        base = FIT_ALIAS.get(lab, lab)
        if base == "oracle":
            costs = oracle_costs_general(g, B, H, E)
            act = oracle_actions_general(g, B, H, E)
            out[lab] = {"mean_cost": float(costs.mean()),
                        "handoff_rate": float((act == 1).mean()),
                        "fail_rate": float((act == 2).mean()),
                        "costs": costs}
        elif base == "none":
            st, _ = simulate_v2_general(g, B, H_big, E, mdl, return_actions=True,
                                        H_bill=H)
            out[lab] = _result(st)
        elif base == "tuned_tau":
            st, _ = simulate_tau_general(g, B, H, E, mdl.models, mdl.tau,
                                         return_actions=True)
            out[lab] = _result(st)
        elif base in ("baton_eq", "dp"):
            st, _ = simulate_v2_general(g, B, H, E, mdl, return_actions=True)
            out[lab] = _result(st)
        elif base == "otr21":
            preds = {k: [_RichPredictor(model, mdl.mu, mdl.sig)]
                     for k, model in mdl.models.items()}
            costs, act = simulate_policy(_GView(g), B, H, E, preds, H_bill=H)
            out[lab] = {"mean_cost": float(costs.mean()),
                        "handoff_rate": float((act == 1).mean()),
                        "fail_rate": float((act == 2).mean()),
                        "costs": costs}
        else:
            raise ValueError(f"unknown label {lab!r}")
    return out
