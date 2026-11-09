"""Module D: arc-level Monte Carlo with common random numbers (ground truth)."""

import numpy as np

from .instances import effective_H, noise_matrix
from .pboi import lam_vector, ordered
from .tsp import tour_stats

_MAX_CHUNK_CELLS = 1 << 24

# Arc-noise families (mean 1 per arc, per-arc coefficient of variation v * H_ij).
DISTS = ("gamma", "lognormal", "incident", "regional")
_DIST_DEFAULTS = {
    "gamma": {},
    "lognormal": {},
    "incident": {"p_inc": 0.02, "inc_mult": 2.5},
    "regional": {"n_regions": 4, "region_cv": 0.15},
}


def _chunk_size(N, chunk):
    return max(1, min(int(chunk), _MAX_CHUNK_CELLS // (N * N)))


def _family(dist, dist_params=None):
    """Validated (name, parameter dict) of an arc-noise family, defaults filled in."""
    if dist not in DISTS:
        raise ValueError(f"unknown noise family {dist!r}; use one of {DISTS}")
    params = dict(_DIST_DEFAULTS[dist])
    if dist_params:
        unknown = set(dist_params) - set(params)
        if unknown:
            raise ValueError(f"unknown parameters for {dist!r}: {sorted(unknown)}")
        params.update(dist_params)
    if dist == "incident":
        p = float(params["p_inc"])
        mult = float(params["inc_mult"])
        if not (0.0 <= p <= 1.0) or mult <= 0.0:
            raise ValueError("incident noise needs 0 <= p_inc <= 1 and inc_mult > 0")
        params = {"p_inc": p, "inc_mult": mult}
    if dist == "regional":
        if int(params["n_regions"]) != 4:
            raise ValueError("regional noise splits the nodes by quadrant: n_regions must be 4")
        if float(params["region_cv"]) < 0.0:
            raise ValueError("region_cv must be nonnegative")
        params = {"n_regions": 4, "region_cv": float(params["region_cv"])}
    return dist, params


def region_labels(inst, dist="gamma", dist_params=None):
    """Quadrant label (0..3) of every node for the regional family, else None.

    Quadrant = 2 * (x > median x) + (y > median y) over all nodes of the
    instance; the depot (node 0) takes the label of its own position.
    """
    dist, _ = _family(dist, dist_params)
    if dist != "regional":
        return None
    coords = getattr(inst, "coords", None)
    if coords is None:
        raise ValueError("regional noise needs inst.coords")
    xy = np.asarray(coords, dtype=np.float64)
    mx = np.median(xy[:, 0])
    my = np.median(xy[:, 1])
    return (2 * (xy[:, 0] > mx) + (xy[:, 1] > my)).astype(np.int64)


def _gamma_mean_one(rng, cv2, size, N):
    """Gamma factors with mean 1 and squared cv cv2 (N, N), one draw per (scenario, arc).

    cv2 == 0 gives exactly one.
    """
    cv2 = np.broadcast_to(np.asarray(cv2, dtype=np.float64), (N, N))
    pos = cv2 > 0
    safe = np.where(pos, cv2, 1.0)
    g = rng.gamma(shape=1.0 / safe, scale=safe, size=(size, N, N))
    return np.where(pos[None, :, :], g, 1.0)


def _draw_chunk(N, v, sY, size, seed, chunk_index, H=None, dist="gamma", dist_params=None,
                region=None):
    """Day factors Y (size,) and arc factors eps (size, N, N) for one chunk.

    eps[s, i, j] depends only on (seed, chunk_index, s, i, j), never on a plan.
    With a multiplier H the arc (i, j) has cv v * H_ij. The day factors are drawn
    first from the chunk generator, so Y is the same for every family.

    dist "gamma" (default, the original law): Gamma(shape 1/(vH)^2, scale (vH)^2).
    dist "lognormal": exp(s Z - s^2/2), s^2 = log(1 + (vH)^2).
    dist "incident": G * J / E J with J = inc_mult w.p. p_inc and 1 otherwise,
        G gamma with the cv chosen so that the product has cv vH.
    dist "regional": G * R[s, g(i)] with R lognormal (mean 1, cv region_cv) per
        region and scenario, g(i) the region of the tail node i; `region` holds
        the labels (see region_labels).
    """
    dist, params = _family(dist, dist_params)
    rng = np.random.default_rng([int(seed), int(chunk_index)])
    Z = rng.standard_normal(size)
    Y = np.exp(sY * Z - 0.5 * sY * sY)
    if v == 0:
        return Y, None
    if dist == "gamma":
        if H is None:
            eps = rng.gamma(shape=1.0 / (v * v), scale=v * v, size=(size, N, N))
        else:
            vh = v * np.asarray(H, dtype=np.float64)
            eps = rng.gamma(shape=1.0 / (vh * vh), scale=vh * vh, size=(size, N, N))
        return Y, eps
    cv2 = np.broadcast_to(np.asarray(v * (np.ones((N, N)) if H is None else H),
                                     dtype=np.float64) ** 2, (N, N))
    if dist == "lognormal":
        s2 = np.log1p(cv2)
        s = np.sqrt(s2)
        eps = np.exp(s[None, :, :] * rng.standard_normal((size, N, N)) - 0.5 * s2[None, :, :])
        return Y, eps
    if dist == "incident":
        p = params["p_inc"]
        mult = params["inc_mult"]
        ej = 1.0 - p + p * mult
        cvj2 = p * (1.0 - p) * (mult - 1.0) ** 2 / (ej * ej)
        v02 = (1.0 + cv2) / (1.0 + cvj2) - 1.0
        if np.any(v02 < -1e-12):
            raise ValueError("incident noise: 1 + (vH)^2 < 1 + cv_J^2 on some arc; "
                             "raise v or lower p_inc / inc_mult")
        G = _gamma_mean_one(rng, np.maximum(v02, 0.0), size, N)
        J = np.where(rng.random((size, N, N)) < p, mult, 1.0)
        return Y, G * J / ej
    # regional
    if region is None:
        raise ValueError("regional noise needs the region labels (see region_labels)")
    rcv2 = params["region_cv"] ** 2
    v02 = (1.0 + cv2) / (1.0 + rcv2) - 1.0
    if np.any(v02 < -1e-12):
        raise ValueError("regional noise: 1 + (vH)^2 < 1 + region_cv^2 on some arc; "
                         "raise v or lower region_cv")
    G = _gamma_mean_one(rng, np.maximum(v02, 0.0), size, N)
    n_reg = int(params["n_regions"])
    if rcv2 > 0:
        sR2 = np.log1p(rcv2)
        R = np.exp(np.sqrt(sR2) * rng.standard_normal((size, n_reg)) - 0.5 * sR2)
    else:
        R = np.ones((size, n_reg))
    R_tail = R[:, np.asarray(region, dtype=np.int64)]
    return Y, G * R_tail[:, :, None]


def _route_X(eps, C, route, size):
    seq = np.concatenate(([0], np.asarray(route, dtype=np.int64), [0]))
    src = seq[:-1]
    dst = seq[1:]
    c = C[src, dst]
    if eps is None:
        return np.full(size, float(c.sum()))
    return eps[:, src, dst] @ c


def simulate(inst, routes, v, sY, S, seed, chunk=2000, dist="gamma", dist_params=None):
    """Arc-level simulation of one plan.

    Returns X (S, m) with X_k = sum over arcs of c_ij eps_ij (no day factor)
    and Y (S,) the day factors. dist and dist_params select the arc-noise
    family (see _draw_chunk); the default is the gamma law.
    """
    C = inst.C
    N = C.shape[0]
    H = effective_H(inst)
    dist, params = _family(dist, dist_params)
    region = region_labels(inst, dist, params)
    cs = _chunk_size(N, chunk)
    m = len(routes)
    X = np.empty((S, m))
    Y = np.empty(S)
    for ci, s0 in enumerate(range(0, S, cs)):
        size = min(cs, S - s0)
        Yc, eps = _draw_chunk(N, v, sY, size, seed, ci, H, dist, params, region)
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


def evaluate_plans(inst, list_of_routes, v, sY, S=20000, seed=2026, chunk=2000,
                   dist="gamma", dist_params=None):
    """Evaluate several plans on the same draws (common random numbers).

    Each chunk is drawn once and every plan is evaluated on it. Returns one
    dict per plan. Keys without suffix refer to Phi = Y * X; keys with suffix
    `_X` refer to X alone. dist and dist_params select the arc-noise family;
    for "regional" the region labels come from inst.coords.
    """
    C = inst.C
    N = C.shape[0]
    H = effective_H(inst)
    D = noise_matrix(inst)
    dist, params = _family(dist, dist_params)
    region = region_labels(inst, dist, params)
    cs = _chunk_size(N, chunk)
    P = len(list_of_routes)
    m = len(list_of_routes[0])
    X_all = np.empty((P, S, m))
    Y_all = np.empty(S)
    for ci, s0 in enumerate(range(0, S, cs)):
        size = min(cs, S - s0)
        Yc, eps = _draw_chunk(N, v, sY, size, seed, ci, H, dist, params, region)
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
        d["W_driver"] = [tour_stats(r, C, D)[1] for r in plan]
        results.append(d)
    return results


def evaluate_plan(inst, routes, v, sY, S=20000, seed=2026, chunk=2000, dist="gamma",
                  dist_params=None):
    return evaluate_plans(inst, [routes], v, sY, S=S, seed=seed, chunk=chunk,
                          dist=dist, dist_params=dist_params)[0]
