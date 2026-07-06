"""Online Threshold Reassignment (OTR) for the SVRPSPD -- LOAD-SPACE (Model-A) version.

This replaces the net-increment (g / B) formulation. Three correctness fixes vs that version:

  (1) PEAK label, not END label.  The old fit used  Y = 1{ W_m > B }  (the final running sum),
      so it was blind to interior peaks: a route whose load spiked above Q in the middle and then
      drained below by the end was labelled "no overflow". The online emergency rule, however,
      fires the FIRST time the load exceeds Q -- a peak rule. We now label
          Y = 1{ max_k L_k > Q }
      so the model, the simulator, and the W-DRO Model-A peak all use the SAME overflow event.

  (2) Stochastic departure load.  The old slack  B = Q - L0  used L0 = sum of MEAN deliveries, a
      fixed scalar, while the increments used the realized (random) deliveries -- inconsistent.
      We work directly with the realized Model-A load profile, so the departure load
          L_0 = (realized) total delivery
      varies per scenario, exactly as in the planning model.

  (3) Depot overflow included.  L_0 (= total delivery) is now part of the peak and is checked for
      emergency. The old formulation started at customer 1 and could not see a depot overflow.

COST MODEL (the OTR "two worlds"):
  * HANDOFF   (proactive: predicted P(peak > Q | L_k) > tau)  costs  omegaF   -- one planned spare.
  * EMERGENCY (the route physically overflows, L_k > Q)        costs, by DEFAULT, the same
      peak-based ceil a reactive run would incur, scaled by the chaotic premium:
          ceil((peak - Q) / Q) * Cfail
      This keeps the emergency MAGNITUDE consistent with the reactive ALNS economics (both count
      ceil vehicles); only the premium Cfail/omegaF distinguishes chaotic from planned recourse.
      Pass emergency_ceil=False for the legacy flat-per-event charge (Cfail regardless of size).

Public API (used by run_otr_eval.py): fit_otr, tau_myopic, tune_tau_fast, simulate_fast.
Also exported: load_profile, calibrate_B, calibrate_B_empirical, calibration_rmse.
(The scalar otr_route/simulate and the Gaussian-fallback helpers were removed -- they were on the
old g/B convention and unused on the fast path; re-add in load space if a scarce-data path is needed.)
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


# ============================================================ load profile
def load_profile(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Model-A load profile of ONE route. d, p have shape (N, m): per-scenario delivery and pickup
    of each of the m customers on the route. Returns L of shape (N, m+1):
        L[:, 0]  = total delivery               (depot / departure load, realized & stochastic)
        L[:, k]  = total_d - sum_{j<=k} d_j + sum_{j<=k} p_j   (load after serving customer k)
    """
    total_d = d.sum(axis=1, keepdims=True)                         # (N, 1) realized departure load
    Lmid = total_d - np.cumsum(d, axis=1) + np.cumsum(p, axis=1)   # (N, m) loads after each customer
    return np.concatenate([total_d, Lmid], axis=1)                 # (N, m+1)


# ============================================================ offline fitting
def fit_otr(d_hist: np.ndarray, p_hist: np.ndarray, Q: float) -> dict:
    """Fit per-step conditional PEAK-overflow curves on the realized load profile.

    Returns models : dict[int -> IsotonicRegression] with keys k = 1 .. m-1 (the decision steps
    after serving customer k, before the last). models[k] estimates P( max_j L_j > Q | L_k = ell ),
    monotone non-decreasing in ell. Degenerate labels (a route that never / always overflows in the
    training set) yield a constant fit, which is correct.
    """
    L = load_profile(d_hist, p_hist)                 # (N, m+1)
    Y = (L.max(axis=1) > Q).astype(float)            # PEAK overflow incl. depot  <-- fix (1),(3)
    m = L.shape[1] - 1
    models = {}
    for k in range(1, m):                            # decision steps 1 .. m-1
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(L[:, k], Y)
        models[k] = iso
    return models


# ============================================================ slack / threshold helpers
def calibrate_B(Q: float, L0: float) -> float:
    """Kept for API compatibility: nominal slack B = Q - L0. (Not used by the load-space path.)"""
    return Q - L0


def calibrate_B_empirical(d_hist: np.ndarray, p_hist: np.ndarray, Q: float, alpha: float = 0.10) -> float:
    """Diagnostic: the (1 - alpha) quantile of the realized PEAK load -- a sanity check on Q
    (a well-built route should have this <= Q at the planning risk level)."""
    L = load_profile(d_hist, p_hist)
    return float(np.quantile(L.max(axis=1), 1.0 - alpha))


def tau_myopic(omegaF: float, Cfail: float) -> float:
    """Break-even threshold omegaF / Cfail: hand off when the expected emergency cost p * Cfail
    overtakes the certain handoff cost omegaF. A lower bound on the optimal tau."""
    return omegaF / Cfail


# ============================================================ online simulation (vectorized)
def _simulate(d: np.ndarray, p: np.ndarray, Q: float, tau: float,
              omegaF: float, Cfail: float, models: dict, emergency_ceil: bool = True):
    """Vectorized OTR over all N scenarios. Mirrors the online rule exactly: at each step, an
    emergency check (L_k > Q) first, then -- if not the last step -- a handoff check.
    Returns (costs, action, peak), each shape (N,). action: 0=COMPLETE, 1=HANDOFF, 2=EMERGENCY."""
    N, m = d.shape
    L = load_profile(d, p)                                          # (N, m+1)
    peak = L.max(axis=1)
    emg_cost = (np.ceil(np.clip(peak - Q, 0.0, None) / Q) if emergency_ceil
                else np.ones(N)) * Cfail                            # charged IF this route emergencies

    costs   = np.zeros(N)
    action  = np.zeros(N, dtype=np.int8)
    stopped = np.zeros(N, dtype=bool)

    # depot (k = 0): emergency only -- the route is already dispatched, no proactive handoff here
    em = L[:, 0] > Q
    if em.any():
        costs[em] = emg_cost[em]; action[em] = 2; stopped |= em

    for k in range(1, m + 1):
        if stopped.all():
            break
        Lk = L[:, k]
        em = (~stopped) & (Lk > Q)                                 # 1. physical overflow
        if em.any():
            costs[em] = emg_cost[em]; action[em] = 2; stopped |= em
        if k == m:
            break                                                  # last customer -> rest COMPLETE
        live = ~stopped                                            # 2. proactive handoff
        if live.any() and k in models:
            pr = models[k].predict(Lk[live])
            ho = np.where(live)[0][pr > tau]
            costs[ho] = omegaF; action[ho] = 1; stopped[ho] = True
    return costs, action, peak


def simulate_fast(d_test: np.ndarray, p_test: np.ndarray, Q: float, tau: float,
                  omegaF: float, Cfail: float, models: dict, emergency_ceil: bool = True) -> dict:
    """Run OTR on every test scenario; return summary stats.
    Keys: mean_cost, handoff_rate, fail_rate, complete_rate, emg_vehicles
    (emg_vehicles = mean ceil-vehicles over routes that emergency -- the magnitude of the failures,
    directly comparable to the reactive ALNS E[V_extra])."""
    costs, action, peak = _simulate(d_test, p_test, Q, tau, omegaF, Cfail, models, emergency_ceil)
    ceil_v = np.ceil(np.clip(peak - Q, 0.0, None) / Q)
    return {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float((action == 1).mean()),
        "fail_rate":     float((action == 2).mean()),
        "complete_rate": float((action == 0).mean()),
        "emg_vehicles":  float(np.where(action == 2, ceil_v, 0.0).mean()),
    }


def tune_tau_fast(d_train: np.ndarray, p_train: np.ndarray, Q: float, models: dict,
                  omegaF: float, Cfail: float, tau_grid: np.ndarray | None = None,
                  emergency_ceil: bool = True) -> float:
    """Pick tau with lowest average TRAINING cost (then validate out-of-sample on the test set).
    Warning: with small N and a fine grid this can overfit -- prefer tau_myopic when N < 1000."""
    if tau_grid is None:
        tau_grid = np.linspace(0.02, 0.95, 47)
    best_tau, best = float(tau_grid[0]), np.inf
    for tau in tau_grid:
        costs, _, _ = _simulate(d_train, p_train, Q, float(tau), omegaF, Cfail, models, emergency_ceil)
        c = float(costs.mean())
        if c < best:
            best, best_tau = c, float(tau)
    return best_tau


# ============================================================ calibration diagnostic
def calibration_rmse(d_test: np.ndarray, p_test: np.ndarray, Q: float,
                     models: dict, n_bins: int = 10) -> float:
    """Reliability RMSE across steps/scenarios (predicted P vs realized peak-overflow frequency).
    < 0.05 = well calibrated; nan if no bin has >= 50 samples."""
    L = load_profile(d_test, p_test)
    Y = (L.max(axis=1) > Q).astype(float)
    preds, acts = [], []
    for k in models:
        preds.append(models[k].predict(L[:, k])); acts.append(Y)
    if not preds:
        return float("nan")
    preds = np.concatenate(preds); acts = np.concatenate(acts)
    bins = np.linspace(0.0, 1.0, n_bins + 1); se = []
    for b in range(n_bins):
        msk = (preds >= bins[b]) & (preds < bins[b + 1])
        if msk.sum() > 50:
            se.append((preds[msk].mean() - acts[msk].mean()) ** 2)
    return float(np.sqrt(np.mean(se))) if se else float("nan")