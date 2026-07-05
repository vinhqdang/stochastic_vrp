"""Realistic last-mile cost model for SVRPSPD execution policies.

Models the economics of an urban simultaneous pickup-and-delivery operator
(Grab-style): a contracted planned fleet with fixed day rates, a standby
pool for planned mid-route handoffs, and expensive ad-hoc emergency
vehicles hired at surge prices when a capacity breach actually happens,
with SLA compensation for every customer served late as a consequence.

The key structural feature is that recourse costs are STATE-DEPENDENT:
a handoff or breach at stop k prices the remaining route distance and the
number of downstream customers, so an early emergency is far more
expensive than a late one. A fixed probability threshold cannot express
the resulting decision boundary; the OTR-2.0 optimal-stopping trigger
handles it natively by comparing the continuation-cost estimate to the
handoff price AT THIS STOP.

All monetary quantities are in a common currency unit (think USD/day for
a single vehicle route); defaults are calibrated to public Southeast-Asian
last-mile magnitudes and are swept in the sensitivity study.

Public API
----------
LastMileCosts        -- parameter container with realistic defaults
route_cost_schedules -- per-stop handoff/emergency cost vectors for a route
fit_lsm_general      -- LSM backward induction under per-stop cost schedules
simulate_v2_general  -- vectorized OTR-2.0 execution under cost schedules
simulate_tau_general -- vectorized fixed-threshold execution (v1-style)
tune_tau_general     -- grid-tune tau under the realistic cost model
oracle_costs_general -- clairvoyant lower bound under cost schedules
plan_fixed_cost      -- fleet fixed + variable travel cost of a plan
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .otr2 import _overflow_step, _ConstantModel


# ============================================================
# Parameters
# ============================================================


@dataclass
class LastMileCosts:
    """Cost parameters of an urban last-mile pickup-and-delivery operator.

    Defaults are order-of-magnitude realistic for Southeast-Asian urban
    operations (contracted motorbike/van fleets with gig-economy surge
    hiring) and are subjected to sensitivity analysis in the evaluation.
    """

    # ── planned fleet ────────────────────────────────────────────────
    F_plan: float = 35.0     # $/vehicle-day: contracted driver + vehicle + insurance
    c_km:   float = 0.10     # $/km variable running cost (fuel + maintenance)

    # ── planned handoff (standby pool, scheduled rebalancing) ────────
    F_ho:   float = 12.0     # callout fee for a standby vehicle
    s_ho:   float = 1.2      # standby per-km surcharge multiplier
    c_transfer: float = 3.0  # cargo transfer dwell (driver time + parking)

    # ── emergency recourse (ad-hoc gig vehicle at surge price) ───────
    F_emg:  float = 40.0     # surge callout fee (>> F_ho)
    s_emg:  float = 2.5      # surge per-km multiplier
    p_late: float = 1.5      # $ SLA compensation per downstream customer served late
    p_breach: float = 10.0   # $ churn/goodwill loss at the breached stop itself

    def handoff_cost(self, remaining_km: float) -> float:
        """Planned handoff after stop k: standby vehicle covers the rest."""
        return self.F_ho + self.s_ho * self.c_km * remaining_km + self.c_transfer

    def emergency_cost(self, remaining_km: float, n_downstream: int) -> float:
        """Unplanned breach at stop k: surge vehicle + SLA fallout."""
        return (self.F_emg
                + self.s_emg * self.c_km * remaining_km
                + self.p_late * n_downstream
                + self.p_breach)


# ============================================================
# Per-route cost schedules from real geometry
# ============================================================


def route_cost_schedules(route: list, D: np.ndarray, scale: float,
                         costs: LastMileCosts) -> tuple[np.ndarray, np.ndarray]:
    """Per-stop handoff and emergency cost vectors for one closed route.

    Parameters
    ----------
    route : list of customer indices (depot excluded)
    D     : full distance matrix (integers, Dethloff convention)
    scale : distance scale divisor (Dethloff: 10000)
    costs : LastMileCosts

    Returns
    -------
    H : np.ndarray, shape (m,)
        H[k-1] = cost of a planned handoff after serving stop k. The
        standby vehicle drives the remaining legs k -> k+1 -> ... -> depot.
        (Entry m-1 exists for completeness but is never used online: after
        the last stop the route simply completes.)
    E : np.ndarray, shape (m,)
        E[k-1] = cost of an emergency breach AT stop k: surge vehicle runs
        the remaining route and every downstream customer is served late.
    """
    m = len(route)
    # remaining distance after serving stop k: legs k->k+1->...->m->depot
    rem = np.zeros(m)
    acc = float(D[route[-1], 0]) / scale
    rem[m - 1] = acc
    for i in range(m - 2, -1, -1):
        acc += float(D[route[i], route[i + 1]]) / scale
        rem[i] = acc

    H = np.array([costs.handoff_cost(rem[k]) for k in range(m)])
    E = np.array([costs.emergency_cost(rem[k], m - (k + 1)) for k in range(m)])
    return H, E


def plan_fixed_cost(K: int, total_dist: float, costs: LastMileCosts) -> float:
    """Deterministic part of the plan: contracted fleet + variable travel."""
    return costs.F_plan * K + costs.c_km * total_dist


# ============================================================
# Generalized LSM — per-stop cost schedules
# ============================================================


def fit_lsm_general(g_hist: np.ndarray, B: float,
                    H: np.ndarray, E: np.ndarray) -> dict:
    """Longstaff-Schwartz backward induction under per-stop costs.

    Identical to core.otr2.fit_lsm except that the terminal/stop values
    come from schedules: an emergency at step j costs E[j-1]; a planned
    handoff after step k costs H[k-1]. The flat model is the special case
    H = omegaF * ones, E = Cfail * ones.
    """
    N, m = g_hist.shape
    cum = np.cumsum(g_hist, axis=1)
    ostep = _overflow_step(cum, B)

    future = np.zeros(N)
    breached = ostep <= m
    future[breached] = E[np.clip(ostep[breached] - 1, 0, m - 1)]

    cont_models: dict = {}
    for k in range(m - 1, 0, -1):
        alive = ostep > k
        n_alive = int(alive.sum())
        if n_alive >= 2:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(cum[alive, k - 1], future[alive])
        elif n_alive == 1:
            iso = _ConstantModel(future[alive][0])
        else:
            iso = _ConstantModel(float(E[min(k, m - 1)]))
        cont_models[k] = iso

        pred = iso.predict(cum[:, k - 1])
        stop = alive & (pred > H[k - 1])
        future[stop] = H[k - 1]

    return cont_models


def _simulate_costs_general(g, B, H, E, cont_models, tau=None,
                            prob_models=None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized execution under per-stop cost schedules.

    Two trigger modes:
      cont_models given  -> OTR-2.0 rule: handoff iff chat > H[k-1]
      prob_models + tau  -> v1-style rule: handoff iff p_k(W_k) > tau
    Pass tau=1.0 with prob_models for the no-handoff (reactive) policy.

    Returns
    -------
    (costs, action) : per-scenario cost and action code
                      0 = COMPLETE, 1 = HANDOFF, 2 = EMERGENCY
    """
    N, m = g.shape
    cumW = np.cumsum(g, axis=1)
    costs = np.zeros(N)
    action = np.zeros(N, dtype=np.int8)
    stopped = np.zeros(N, dtype=bool)

    for k_idx in range(m):
        k = k_idx + 1
        active = ~stopped
        if not active.any():
            break
        Wk = cumW[:, k_idx]

        em = active & (Wk > B)
        costs[em] = E[k_idx]
        action[em] = 2
        stopped |= em

        if k == m:
            break

        ho_active = active & ~em
        if not ho_active.any():
            continue
        if cont_models is not None:
            model = cont_models.get(k)
            if model is None:
                continue
            trig = model.predict(Wk[ho_active]) > H[k_idx]
        else:
            model = prob_models.get(k) if prob_models else None
            if model is None:
                continue
            trig = model.predict(Wk[ho_active]) > tau
        ho_idx = np.where(ho_active)[0][trig]
        costs[ho_idx] = H[k_idx]
        action[ho_idx] = 1
        stopped[ho_idx] = True

    return costs, action


def _stats(costs: np.ndarray, action: np.ndarray) -> dict:
    return {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float((action == 1).mean()),
        "fail_rate":     float((action == 2).mean()),
        "complete_rate": float((action == 0).mean()),
    }


def simulate_v2_general(g_test, B, H, E, cont_models) -> dict:
    """OTR-2.0 execution stats under per-stop cost schedules."""
    c, a = _simulate_costs_general(g_test, B, H, E, cont_models)
    return _stats(c, a)


def simulate_tau_general(g_test, B, H, E, prob_models, tau) -> dict:
    """Fixed-threshold (v1-style) execution stats under cost schedules."""
    c, a = _simulate_costs_general(g_test, B, H, E, None,
                                   tau=tau, prob_models=prob_models)
    return _stats(c, a)


def tune_tau_general(g_train, B, H, E, prob_models,
                     tau_grid: np.ndarray | None = None) -> float:
    """Grid-tune a single tau under the realistic cost model.

    This is the strongest form of the v1-style policy: even with the peak
    label fixed and tau tuned against the TRUE realized costs, one scalar
    threshold cannot adapt to per-stop prices — which is what the
    experiments quantify.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.02, 0.98, 49)
    best_tau, best_cost = 1.0, np.inf
    base, _ = _simulate_costs_general(g_train, B, H, E, None,
                                      tau=1.0, prob_models=prob_models)
    if base.mean() < best_cost:
        best_cost, best_tau = float(base.mean()), 1.0
    for tau in tau_grid:
        c, _ = _simulate_costs_general(g_train, B, H, E, None,
                                       tau=float(tau), prob_models=prob_models)
        if c.mean() < best_cost:
            best_cost, best_tau = float(c.mean()), float(tau)
    return best_tau


# ============================================================
# Clairvoyant oracle under cost schedules
# ============================================================


def oracle_costs_general(g: np.ndarray, B: float,
                         H: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Perfect-information lower bound with per-stop prices.

    A clairvoyant sees the whole demand path. If the route never breaches,
    it completes free. If it would breach at step j, the clairvoyant picks
    the cheapest of (i) handing off after any stop k < j, or (ii) simply
    letting the breach happen — min(min_{k<j} H[k-1], E[j-1]). A breach at
    step 1 leaves no decision epoch, so only option (ii) exists.
    """
    N, m = g.shape
    ostep = _overflow_step(np.cumsum(g, axis=1), B)
    costs = np.zeros(N)
    Hmin_before = np.concatenate([[np.inf], np.minimum.accumulate(H[:m - 1])])
    br = ostep <= m
    j = ostep[br]
    costs[br] = np.minimum(Hmin_before[j - 1], E[j - 1])
    return costs
