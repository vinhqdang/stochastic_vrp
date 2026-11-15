"""Execution of learned continuation models on test days (module M6).

EXPERIMENT_A.md section 9, M6. ``simulate_policy`` follows
``baton.costs._simulate_costs_general`` line by line: the same emergency check
at every stop, the same handoff trigger ``C_hat > H[k-1]``, and the same
billing with ``H_bill``. The only difference is that the continuation value
C_hat is the mean over a list of predictors, which lets a cross-fitted pair of
networks be deployed as their average (section 2.3). A missing k means no
decision at that stop.
"""
from __future__ import annotations

import numpy as np

from .backward import IsoPredictor, Predictor
from .features import RouteCache


def baton_models(cont_models: dict) -> dict[int, list[Predictor]]:
    """Wrap the output of ``costs.fit_lsm_general`` (sklearn isotonic or
    _ConstantModel per k) as single-predictor lists."""
    return {k: [IsoPredictor(model)] for k, model in cont_models.items()}


def simulate_policy(cache: RouteCache, B: float, H: np.ndarray, E: np.ndarray,
                    models: dict[int, list[Predictor]],
                    H_bill: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-scenario (costs (N,), action (N,) int8) on ``cache.g``.

    Action codes as in BATON: 0 complete, 1 handoff, 2 emergency.
    """
    H = np.asarray(H, dtype=float)
    E = np.asarray(E, dtype=float)
    H_bill = H if H_bill is None else np.asarray(H_bill, dtype=float)
    cumW = cache.W
    N, m = cumW.shape
    if H.shape != (m,) or E.shape != (m,) or H_bill.shape != (m,):
        raise ValueError("H, E and H_bill must have length m")

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
        preds = models.get(k)
        if not preds:
            continue
        rows = np.flatnonzero(ho_active)
        chat = np.mean([p.predict(cache, k, rows) for p in preds], axis=0)
        trig = chat > H[k_idx]
        ho_idx = rows[trig]
        costs[ho_idx] = H_bill[k_idx]
        action[ho_idx] = 1
        stopped[ho_idx] = True

    return costs, action


def summarize(costs: np.ndarray, action: np.ndarray) -> dict:
    """Same keys as ``costs._stats``."""
    return {
        "mean_cost": float(costs.mean()),
        "handoff_rate": float((action == 1).mean()),
        "fail_rate": float((action == 2).mean()),
        "complete_rate": float((action == 0).mean()),
        "costs": costs,
    }
