"""Module D: arc-level Monte Carlo with common random numbers (ground truth)."""

import numpy as np

from .pboi import lam_vector, ordered
from .tsp import tour_stats

_MAX_CHUNK_CELLS = 1 << 24


def _chunk_size(N, chunk):
    return max(1, min(int(chunk), _MAX_CHUNK_CELLS // (N * N)))


def _draw_chunk(N, v, sY, size, seed, chunk_index):
    """Day factors Y (size,) and arc factors eps (size, N, N) for one chunk.

    eps[s, i, j] depends only on (seed, chunk_index, s, i, j), never on a plan.
    """
    rng = np.random.default_rng([int(seed), int(chunk_index)])
    Z = rng.standard_normal(size)
    Y = np.exp(sY * Z - 0.5 * sY * sY)
    if v == 0:
        return Y, None
    eps = rng.gamma(shape=1.0 / (v * v), scale=v * v, size=(size, N, N))
    return Y, eps


def _route_X(eps, C, route, size):
    seq = np.concatenate(([0], np.asarray(route, dtype=np.int64), [0]))
    src = seq[:-1]
    dst = seq[1:]
    c = C[src, dst]
    if eps is None:
        return np.full(size, float(c.sum()))
    return eps[:, src, dst] @ c


def simulate(inst, routes, v, sY, S, seed, chunk=2000):
    """Arc-level simulation of one plan.

    Returns X (S, m) with X_k = sum over arcs of c_ij eps_ij (no day factor)
    and Y (S,) the day factors.
    """
    C = inst.C
    N = C.shape[0]
    cs = _chunk_size(N, chunk)
    m = len(routes)
    X = np.empty((S, m))
    Y = np.empty(S)
    for ci, s0 in enumerate(range(0, S, cs)):
        size = min(cs, S - s0)
        Yc, eps = _draw_chunk(N, v, sY, size, seed, ci)
        Y[s0:s0 + size] = Yc
        for k, r in enumerate(routes):
            X[s0:s0 + size, k] = _route_X(eps, C, r, size)
    return X, Y


def _cvar(x, beta):
    var = np.quantile(x, beta)
    return float(var + np.mean(np.maximum(x - var, 0.0)) / (1.0 - beta))


def _summaries(Z, suffix, m):
    out = {}
    for name in ("max", "range", "gmd", "mad"):
        if name == "mad" and m % 2 == 0:
            continue
        out[f"E_{name}{suffix}"] = float(np.mean(ordered(Z, lam_vector(name, m))))
    out[f"E_var{suffix}"] = float(np.mean(Z.var(axis=1)))
    out[f"CVaR90_range{suffix}"] = _cvar(ordered(Z, lam_vector("range", m)), 0.9)
    out[f"mean_total{suffix}"] = float(Z.sum(axis=1).mean())
    out[f"mean_driver{suffix}"] = Z.mean(axis=0).tolist()
    out[f"sd_driver{suffix}"] = Z.std(axis=0).tolist()
    return out


def evaluate_plans(inst, list_of_routes, v, sY, S=20000, seed=2026, chunk=2000):
    """Evaluate several plans on the same draws (common random numbers).

    Each chunk is drawn once and every plan is evaluated on it. Returns one
    dict per plan. Keys without suffix refer to Phi = Y * X; keys with suffix
    `_X` refer to X alone.
    """
    C = inst.C
    N = C.shape[0]
    cs = _chunk_size(N, chunk)
    P = len(list_of_routes)
    m = len(list_of_routes[0])
    X_all = np.empty((P, S, m))
    Y_all = np.empty(S)
    for ci, s0 in enumerate(range(0, S, cs)):
        size = min(cs, S - s0)
        Yc, eps = _draw_chunk(N, v, sY, size, seed, ci)
        Y_all[s0:s0 + size] = Yc
        for p, plan in enumerate(list_of_routes):
            for k, r in enumerate(plan):
                X_all[p, s0:s0 + size, k] = _route_X(eps, C, r, size)
    results = []
    for p, plan in enumerate(list_of_routes):
        X = X_all[p]
        Phi = X * Y_all[:, None]
        d = {}
        d.update(_summaries(Phi, "", m))
        d.update(_summaries(X, "_X", m))
        d["L_driver"] = [tour_stats(r, C)[0] for r in plan]
        d["W_driver"] = [tour_stats(r, C)[1] for r in plan]
        results.append(d)
    return results


def evaluate_plan(inst, routes, v, sY, S=20000, seed=2026, chunk=2000):
    return evaluate_plans(inst, [routes], v, sY, S=S, seed=seed, chunk=chunk)[0]
