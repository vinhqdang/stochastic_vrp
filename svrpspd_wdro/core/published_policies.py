"""Published rule-based recourse policies, adapted to the handoff setting.

Salavati-Khoshghalb, Gendreau, Jabali and Rei (2019, Transportation
Science 53(5):1334-1353) study rule-based preventive recourse for the
VRPSD: return to the depot when the residual capacity drops below a
threshold. Their three rule families, translated to our mid-route
handoff recourse (residual slack B - W_k plays the role of residual
capacity, a handoff plays the role of the depot return):

    pi1  fixed fraction of capacity:
         hand off after stop k  iff  B - W_k < delta * B
    pi2  next-customer expectation:
         hand off after stop k  iff  B - W_k < eta * E[g_{k+1}]^+
    pi3  remaining-route expectation:
         hand off after stop k  iff  B - W_k < lam * E[sum_{j>k} g_j]^+

Each rule is a position-dependent threshold on W_k, so all three run in
the same vectorized pass used by the other policies; the scalar
coefficient (delta/eta/lam) is grid-tuned on training scenarios against
the true realized costs — the strongest form of each rule.

Public API
----------
pi_thresholds  -- per-position W_k thresholds for a rule and coefficient
simulate_pi    -- vectorized execution under a threshold vector
tune_pi        -- grid-tune the rule coefficient on training data
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def pi_thresholds(kind: str, B: float, g_mean: np.ndarray,
                  coef: float) -> np.ndarray:
    """Threshold vector thr[k-1] for decision epochs k = 1..m-1:
    hand off after stop k iff W_k > thr[k-1]."""
    m = len(g_mean)
    if kind == "pi1":
        ref = np.full(m - 1, B)
    elif kind == "pi2":
        ref = np.maximum(g_mean[1:], _EPS)          # next customer's mean
    elif kind == "pi3":
        rem = np.cumsum(g_mean[::-1])[::-1]          # sum_{j>=k}
        ref = np.maximum(rem[1:], _EPS)              # remaining after stop k
    else:
        raise ValueError(kind)
    return B - coef * ref


def _simulate_costs_pi(g: np.ndarray, B: float, H: np.ndarray,
                       E: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Vectorized execution: physical overflow guard at every stop, then
    the rule trigger W_k > thr[k-1]. Costs from per-stop schedules."""
    N, m = g.shape
    cumW = np.cumsum(g, axis=1)
    costs = np.zeros(N)
    stopped = np.zeros(N, dtype=bool)

    for k_idx in range(m):
        k = k_idx + 1
        active = ~stopped
        if not active.any():
            break
        Wk = cumW[:, k_idx]

        em = active & (Wk > B)
        costs[em] = E[k_idx]
        stopped |= em
        if k == m:
            break

        ho = active & ~em & (Wk > thr[k_idx])
        costs[ho] = H[k_idx]
        stopped |= ho
    return costs


def simulate_pi(g_test: np.ndarray, B: float, H: np.ndarray, E: np.ndarray,
                thr: np.ndarray) -> dict:
    costs = _simulate_costs_pi(g_test, B, H, E, thr)
    # a handoff at stop k costs H[k-1]; an emergency E[k-1]; identify by value
    # only when schedules are strictly ordered — report rates via re-simulation
    # bookkeeping instead:
    N, m = g_test.shape
    cumW = np.cumsum(g_test, axis=1)
    stopped = np.zeros(N, dtype=bool)
    action = np.zeros(N, dtype=np.int8)
    for k_idx in range(m):
        active = ~stopped
        if not active.any():
            break
        Wk = cumW[:, k_idx]
        em = active & (Wk > B)
        action[em] = 2
        stopped |= em
        if k_idx + 1 == m:
            break
        ho = active & ~em & (Wk > thr[k_idx])
        action[ho] = 1
        stopped |= ho
    return {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float((action == 1).mean()),
        "fail_rate":     float((action == 2).mean()),
        "complete_rate": float((action == 0).mean()),
    }


def tune_pi(kind: str, g_train: np.ndarray, B: float, H: np.ndarray,
            E: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Grid-tune the rule coefficient against realized training costs.
    coef=0 disables the rule (pure reactive), which is always in the grid."""
    if grid is None:
        grid = np.concatenate([[0.0], np.geomspace(0.01, 5.0, 40)])
    g_mean = g_train.mean(axis=0)
    best_c, best_cost = 0.0, np.inf
    for c in grid:
        thr = pi_thresholds(kind, B, g_mean, float(c))
        cost = float(_simulate_costs_pi(g_train, B, H, E, thr).mean())
        if cost < best_cost:
            best_cost, best_c = cost, float(c)
    return best_c
