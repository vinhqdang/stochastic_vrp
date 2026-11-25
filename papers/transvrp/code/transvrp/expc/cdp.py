"""CDP: copula-conditional dynamic program (EXPERIMENT_C.md 1.7, 6, 9 MC3).

Model class: Gaussian copula with gamma marginals, fitted per stop on the training paths.
Delivery and pickup each carry one common factor (Z_d, Z_p, independent of each other);
the normal score of stop j is z_j = sqrt(rho) Z + sqrt(1 - rho) eps_j.

Tables: for every factor-node pair (zeta_d, zeta_p) on a uniform grid, the conditional law
of each stop's increment g_j = p_j - d_j is represented by S scrambled-Sobol draws of eps
mapped through the fitted marginals; THR's backward recursion (thr._continuation, with the
same arithmetic) gives C_k(w | zeta_d, zeta_p) on a uniform load grid.

Decision: C_hat_k = E over the posterior of (Z_d, Z_p) of C_k(W_k | zeta), by 9 x 9
Gauss-Hermite points, bilinear interpolation in the nodes (clamped to [-4, 4]) and linear
interpolation in w; the predictor returns C_hat + tau (tau = 1e-6 E0), as THR does.

Known approximation (stated in the paper): each table treats the factor as known from
stage 1 on, so C_hat under-estimates the continuation cost. It is exact in the rho = 0 limit,
where the tables do not depend on the node and CDP reduces to THR (up to the 256-draw
representation of the conditional laws).
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np
import scipy.stats as st
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import qmc

from ..expa.features import RouteCache, RouteStats
from ..expa.paths import Paths, RouteSpec
from ..expa.simulate import simulate_policy
from ..expb.grid import GridPredictor, lin_interp, tie_tolerance
from ..expb.thr import _grid_bounds

N_NODES = 25            # factor nodes per factor (17 is the logged fallback of 1.7)
N_EPS = 256             # scrambled-Sobol draws of eps per stop and factor
N_GRID = 1001           # load grid points
ZETA_MAX = 4.0
GH_ORDER = 9            # Gauss-Hermite points per factor in the posterior expectation
RHO_MAX = 0.99          # clip of the estimated copula correlation
RHO_EDGE = 1.0          # rho override must lie in [0, RHO_EDGE)
U_CLIP = 1e-12          # clip of uniform variates before quantile maps
BLOCK_ELEMS = 1 << 22   # nodes x grid x draws elements per block (bounds temporaries)


def _phi_inv(u: np.ndarray) -> np.ndarray:
    return st.norm.ppf(np.clip(u, U_CLIP, 1.0 - U_CLIP))


def _gamma_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-stop gamma (shape, scale) by sample moments, and the active mask (mean > 0).
    Inactive (zero-mean) stops have shape = scale = 0 and stay degenerate at 0."""
    if x.shape[0] < 2:
        raise ValueError("at least two paths are needed to fit the marginals")
    mean = x.mean(axis=0)
    var = x.var(axis=0, ddof=1)
    active = mean > 0
    if np.any(active & ~(var > 0)):
        raise ValueError("an active stop has zero sample variance; gamma moments undefined")
    shape = np.zeros(x.shape[1])
    scale = np.zeros(x.shape[1])
    shape[active] = mean[active] ** 2 / var[active]
    scale[active] = var[active] / mean[active]
    return shape, scale, active


def _scores(x: np.ndarray, shape: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normal scores Phi^{-1}(F(x)) of an (n, c) array under c per-column gamma marginals."""
    if x.shape[1] == 0:
        return np.zeros(x.shape)
    return _phi_inv(st.gamma.cdf(x, shape, scale=scale))


def _offdiag_corr_mean(z: np.ndarray) -> float:
    c = z.shape[1]
    if c < 2:
        return 0.0
    r = np.corrcoef(z, rowvar=False)
    return float(r[~np.eye(c, dtype=bool)].mean())


@dataclass(frozen=True)
class CDPModel:
    """Per-stop gamma marginals (shape, scale; 0 for inactive stops) and the factor
    correlations of the delivery and the pickup copula."""
    shape_d: np.ndarray
    scale_d: np.ndarray
    active_d: np.ndarray
    shape_p: np.ndarray
    scale_p: np.ndarray
    active_p: np.ndarray
    rho_d: float
    rho_p: float

    def __post_init__(self) -> None:
        m = self.shape_d.shape[0]
        arrays = (self.scale_d, self.active_d, self.shape_p, self.scale_p, self.active_p)
        if any(a.shape != (m,) for a in arrays):
            raise ValueError("all per-stop arrays must have the same length")
        for rho in (self.rho_d, self.rho_p):
            if not 0.0 <= rho < RHO_EDGE:
                raise ValueError("rho must lie in [0, 1)")


def fit_cdp_model(paths: Paths, rho_override: float | None = None) -> CDPModel:
    """Gamma marginals by moments and rho_d, rho_p by the mean off-diagonal correlation of
    the normal scores over the active stops, clipped to [0, 0.99]. ``rho_override`` replaces
    both (CDP-true in W6 uses 0.6)."""
    shape_d, scale_d, act_d = _gamma_moments(paths.d)
    shape_p, scale_p, act_p = _gamma_moments(paths.p)
    if rho_override is not None:
        if not 0.0 <= float(rho_override) < RHO_EDGE:
            raise ValueError("rho_override must lie in [0, 1)")
        rho_d = rho_p = float(rho_override)
    else:
        zd = _scores(paths.d[:, act_d], shape_d[act_d], scale_d[act_d])
        zp = _scores(paths.p[:, act_p], shape_p[act_p], scale_p[act_p])
        rho_d = float(np.clip(_offdiag_corr_mean(zd), 0.0, RHO_MAX))
        rho_p = float(np.clip(_offdiag_corr_mean(zp), 0.0, RHO_MAX))
    return CDPModel(shape_d=shape_d, scale_d=scale_d, active_d=act_d,
                    shape_p=shape_p, scale_p=scale_p, active_p=act_p,
                    rho_d=rho_d, rho_p=rho_p)


def posterior(z_obs: np.ndarray, rho: float) -> tuple[np.ndarray, float]:
    """Posterior of the factor Z given the scores z_1..z_k of k observed active stops
    (z_obs has shape (n_rows, k)): mean mu = sqrt(rho) sum z / (1 - rho + k rho) per row and
    the common sd s = sqrt((1 - rho) / (1 - rho + k rho)). k = 0 gives (0, 1)."""
    z = np.asarray(z_obs, dtype=float)
    if z.ndim != 2:
        raise ValueError("z_obs must have shape (n_rows, k)")
    if not 0.0 <= rho < RHO_EDGE:
        raise ValueError("rho must lie in [0, 1)")
    k = z.shape[1]
    denom = 1.0 - rho + k * rho
    mu = np.sqrt(rho) * z.sum(axis=1) / denom
    s = float(np.sqrt((1.0 - rho) / denom))
    return mu, s


def _eps_draws(seed: int, m: int, n_eps: int) -> tuple[np.ndarray, np.ndarray]:
    """Scrambled Sobol points in 2m dimensions mapped to normal: the first m columns drive
    the delivery stops, the last m the pickup stops (independent coordinates per stop)."""
    u = qmc.Sobol(d=2 * m, scramble=True, seed=seed).random(n_eps)
    z = _phi_inv(u)
    return np.ascontiguousarray(z[:, :m]), np.ascontiguousarray(z[:, m:])


def _margin_draws(shape: np.ndarray, scale: np.ndarray, active: np.ndarray, rho: float,
                  zeta: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """(n_zeta, S, m) conditional draws of one margin: x = F^{-1}(Phi(sqrt(rho) zeta +
    sqrt(1 - rho) eps)) per stop; zero on inactive stops."""
    arg = np.sqrt(rho) * zeta[:, None, None] + np.sqrt(1.0 - rho) * eps[None, :, :]
    u = np.clip(st.norm.cdf(arg), U_CLIP, 1.0 - U_CLIP)
    a = np.where(active, shape, 1.0)[None, None, :]
    sc = np.where(active, scale, 1.0)[None, None, :]
    x = st.gamma.ppf(u, a, scale=sc)
    return np.where(active[None, None, :], x, 0.0)


def _pair_gaps(model: CDPModel, eps_d: np.ndarray, eps_p: np.ndarray,
               zeta: np.ndarray) -> np.ndarray:
    """(n, n, S, m) with [a, b] = P(zeta_p[b]) - D(zeta_d[a]): the conditional increments."""
    D = _margin_draws(model.shape_d, model.scale_d, model.active_d, model.rho_d, zeta, eps_d)
    P = _margin_draws(model.shape_p, model.scale_p, model.active_p, model.rho_p, zeta, eps_p)
    return P[None, :, :, :] - D[:, None, :, :]


def _pair_continuation(w_grid: np.ndarray, g: np.ndarray, B: float, e_k: float,
                       v_next: np.ndarray, w0: float, h: float) -> np.ndarray:
    """THR's C_k for many node pairs at once: g (P, S) are the next stop's draws per pair,
    v_next (P, n_grid) the next value per pair. Same arithmetic as thr._continuation."""
    n_pairs, n_draws = g.shape
    n_grid = w_grid.size
    out = np.empty((n_pairs, n_grid))
    rows = max(1, BLOCK_ELEMS // (n_grid * n_draws))
    for s in range(0, n_pairs, rows):
        gb = g[s:s + rows]
        vb = v_next[s:s + rows]
        pb = gb.shape[0]
        x = w_grid[None, :, None] + gb[:, None, :]
        pos = (np.maximum(x, w0) - w0) / h
        i = np.minimum(np.floor(pos).astype(np.intp), n_grid - 2)
        frac = pos - i
        flat_i = i.reshape(pb, -1)
        lo = np.take_along_axis(vb, flat_i, axis=1).reshape(x.shape)
        hi = np.take_along_axis(vb, flat_i + 1, axis=1).reshape(x.shape)
        interp = lo + frac * (hi - lo)
        vals = np.where(x > B, e_k, interp)
        out[s:s + pb] = vals.mean(axis=2)
    return out


@dataclass(frozen=True)
class CDPTables:
    """C[k-1, a, b, :] = C_k(. | zeta_d[a], zeta_p[b]) on w_grid (k = 1..m-1)."""
    zeta: np.ndarray
    w_grid: np.ndarray
    C: np.ndarray
    eps_d: np.ndarray
    eps_p: np.ndarray
    tie_tol: float


def build_tables(spec: RouteSpec, model: CDPModel, paths: Paths, n_nodes: int = N_NODES,
                 n_eps: int = N_EPS, n_grid: int = N_GRID, seed: int = 0) -> CDPTables:
    """Backward recursion for every node pair. The load grid is uniform on [w_lo, B] with
    w_lo = thr._grid_bounds on the training paths (THR's data-driven bound, finding S2; the
    bound over the conditional draws of all nodes was removed because it stretched the grid
    over an absurd range). Loads below w_lo take the grid's first value, in the recursion
    and in the predictor, as THR does."""
    m = spec.m
    if m < 2:
        raise ValueError("CDP tables need a route with at least two stops")
    zeta = np.linspace(-ZETA_MAX, ZETA_MAX, n_nodes)
    eps_d, eps_p = _eps_draws(seed, m, n_eps)
    G = _pair_gaps(model, eps_d, eps_p, zeta)
    w_lo = _grid_bounds(paths.g, spec.B, m)
    w_grid = np.linspace(w_lo, spec.B, n_grid)
    w0, h = float(w_grid[0]), float(w_grid[1] - w_grid[0])
    n_pairs = n_nodes * n_nodes
    G2 = G.reshape(n_pairs, n_eps, m)
    C = np.empty((m - 1, n_nodes, n_nodes, n_grid))
    v_next = np.zeros((n_pairs, n_grid))
    for k in range(m - 1, 0, -1):
        ck = _pair_continuation(w_grid, G2[:, :, k], spec.B, float(spec.E[k]), v_next, w0, h)
        C[k - 1] = ck.reshape(n_nodes, n_nodes, n_grid)
        v_next = np.minimum(float(spec.H[k - 1]), ck)
    return CDPTables(zeta=zeta, w_grid=w_grid, C=C, eps_d=eps_d, eps_p=eps_p,
                     tie_tol=tie_tolerance(spec.E0))


def _locate(z: np.ndarray, zeta0: float, dz: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Lower node index and fraction of z (clamped to the node range) on the zeta grid."""
    zc = np.clip(z, zeta0, zeta0 + dz * (n - 1))
    pos = (zc - zeta0) / dz
    i = np.clip(np.floor(pos).astype(np.intp), 0, n - 2)
    return i, pos - i


def recover_increments(cache: RouteCache, idx: np.ndarray, k: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-stop pickup and delivery increments p_1..p_k and d_1..d_k of the rows ``idx``,
    recovered as differences of the cumulative sums P and D in the cache (shape (n, k))."""
    idx = np.asarray(idx, dtype=np.intp)
    p = np.diff(cache.P[idx, :k], axis=1, prepend=0.0)
    d = np.diff(cache.D[idx, :k], axis=1, prepend=0.0)
    return p, d


class CDPPredictor:
    """Predictor for ``simulate_policy``: posterior expectation of the node tables."""

    def __init__(self, model: CDPModel, tables: CDPTables):
        self.model = model
        self.tables = tables
        self.tie_tol = float(tables.tie_tol)
        self._n = int(tables.zeta.size)
        self._zeta0 = float(tables.zeta[0])
        self._dz = float(tables.zeta[1] - tables.zeta[0])
        self._ng = int(tables.w_grid.size)
        self._w0 = float(tables.w_grid[0])
        self._h = float(tables.w_grid[1] - tables.w_grid[0])
        self._flat = tables.C.reshape(tables.C.shape[0], -1)
        self._act_d = np.flatnonzero(model.active_d)
        self._act_p = np.flatnonzero(model.active_p)
        gh_x, gh_w = hermegauss(GH_ORDER)
        self._gh_x = gh_x
        self._gh_w = gh_w / gh_w.sum()

    def _factor_posterior(self, x: np.ndarray, act: np.ndarray, shape: np.ndarray,
                          scale: np.ndarray, rho: float, k: int) -> tuple[np.ndarray, float]:
        cols = act[act < k]
        z = _scores(x[:, cols], shape[cols], scale[cols])
        return posterior(z, rho)

    def _w_index(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower grid index and fraction of each load. The position is clamped to the grid
        (finding S5): loads below the grid take its first value, loads above it its last."""
        pos = np.clip((W - self._w0) / self._h, 0.0, float(self._ng - 1))
        i0 = np.minimum(np.floor(pos).astype(np.intp), self._ng - 2)
        return i0, pos - i0

    def _corner(self, k: int, a: np.ndarray, b: np.ndarray, i0: np.ndarray,
                frac: np.ndarray) -> np.ndarray:
        base = (a * self._n + b) * self._ng + i0
        row = self._flat[k - 1]
        return row[base] * (1.0 - frac) + row[base + 1] * frac

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        if not 1 <= k <= self._flat.shape[0]:
            raise ValueError(f"stage k must be in [1, {self._flat.shape[0]}]")
        idx = np.asarray(idx, dtype=np.intp)
        W = cache.W[idx, k - 1]
        p, d = recover_increments(cache, idx, k)
        mu_d, s_d = self._factor_posterior(d, self._act_d, self.model.shape_d,
                                           self.model.scale_d, self.model.rho_d, k)
        mu_p, s_p = self._factor_posterior(p, self._act_p, self.model.shape_p,
                                           self.model.scale_p, self.model.rho_p, k)
        zd = np.clip(mu_d[:, None] + s_d * self._gh_x[None, :], -ZETA_MAX, ZETA_MAX)
        zp = np.clip(mu_p[:, None] + s_p * self._gh_x[None, :], -ZETA_MAX, ZETA_MAX)
        ad, fd = _locate(zd, self._zeta0, self._dz, self._n)
        bp, fp = _locate(zp, self._zeta0, self._dz, self._n)
        i0, frac = self._w_index(W)
        out = np.zeros(idx.shape[0])
        for i in range(GH_ORDER):
            for j in range(GH_ORDER):
                val = np.zeros(idx.shape[0])
                for da, wa in ((0, 1.0 - fd[:, i]), (1, fd[:, i])):
                    for db, wb in ((0, 1.0 - fp[:, j]), (1, fp[:, j])):
                        val += wa * wb * self._corner(k, ad[:, i] + da, bp[:, j] + db,
                                                      i0, frac)
                out += self._gh_w[i] * self._gh_w[j] * val
        return out + self.tie_tol


def _max_decrease(c: np.ndarray) -> float:
    """Largest decrease of a table along the load grid (0 when nondecreasing)."""
    if c.shape[-1] < 2:
        return 0.0
    return float(max(0.0, -np.min(np.diff(c, axis=-1))))


def fit_cdp(spec: RouteSpec, paths: Paths, N: int, rep: int,
            rho_override: float | None = None, n_nodes: int = N_NODES,
            n_eps: int = N_EPS, n_grid: int = N_GRID, seed: int | None = None) -> dict:
    """CDP on all N training rows. Returns a payload with ``models`` ({k: [predictor]} for
    k = 1..m-1, as fit_thr), ``stats`` (RouteStats), ``diag``, ``budget`` and the fitted
    ``cdp_model`` and ``tables``. The default seed is crc32 of (instance, route, rep)."""
    if paths.p.shape[0] != N:
        raise ValueError(f"expected {N} training paths, got {paths.p.shape[0]}")
    t0 = time.perf_counter()
    stats = RouteStats.fit(paths)
    model = fit_cdp_model(paths, rho_override)
    if seed is None:
        seed = zlib.crc32(f"cdp|{spec.inst}|{spec.route_id}|{int(rep)}".encode("utf-8"))
    t_model = time.perf_counter()
    tables = build_tables(spec, model, paths, n_nodes, n_eps, n_grid, seed)
    t_tables = time.perf_counter()
    predictor = CDPPredictor(model, tables)
    models = {k: [predictor] for k in range(1, spec.m)}
    diag = [{"k": k, "rep": int(rep), "monotone_gap": _max_decrease(tables.C[k - 1])}
            for k in range(1, spec.m)]
    budget = {"unique_paths": int(N), "path_stop_reads": int(N * spec.m),
              "val_reads": 0, "n_grid": int(n_grid), "n_nodes": int(n_nodes),
              "n_eps": int(n_eps), "n_params": 0, "feature_seconds": 0.0,
              "model_seconds": t_model - t0, "table_seconds": t_tables - t_model,
              "table_bytes": int(tables.C.nbytes), "rho_d": model.rho_d,
              "rho_p": model.rho_p, "train_seconds": time.perf_counter() - t0,
              "val_seconds": 0.0, "device": "cpu"}
    return {"models": models, "stats": stats, "diag": diag, "budget": budget,
            "cdp_model": model, "tables": tables}


def node_tables(tables: CDPTables, zeta_d: float, zeta_p: float) -> np.ndarray:
    """(m-1, n_grid) continuation tables at an arbitrary node: bilinear in the nodes
    (clamped), as the predictor interpolates them."""
    zeta = tables.zeta
    n, dz, z0 = zeta.size, float(zeta[1] - zeta[0]), float(zeta[0])
    a, fa = _locate(np.array([zeta_d]), z0, dz, n)
    b, fb = _locate(np.array([zeta_p]), z0, dz, n)
    a, b, fa, fb = int(a[0]), int(b[0]), float(fa[0]), float(fb[0])
    C = tables.C
    return ((1 - fa) * (1 - fb) * C[:, a, b] + fa * (1 - fb) * C[:, a + 1, b]
            + (1 - fa) * fb * C[:, a, b + 1] + fa * fb * C[:, a + 1, b + 1])


def node_check(spec: RouteSpec, model: CDPModel, tables: CDPTables,
               zeta: tuple[float, float] = (0.5, -0.5), n: int = 20000,
               seed: int = 0) -> dict:
    """G-CDP-node (EXPERIMENT_C 6). Simulates n paths from the conditional model at the
    node (fresh eps from numpy, not Sobol), runs the node's own DP policy through
    ``simulate_policy`` and compares the mean cost with the DP value at the initial state
    (the thr_dp_value rule, first-stop draws from the tables' Sobol representation).
    Off-grid nodes use the bilinear table, as deployed."""
    zd, zp = float(zeta[0]), float(zeta[1])
    m = spec.m
    rng = np.random.default_rng(seed)
    eps_d = rng.standard_normal((n, m))
    eps_p = rng.standard_normal((n, m))
    D = _margin_draws(model.shape_d, model.scale_d, model.active_d, model.rho_d,
                      np.array([zd]), eps_d)[0]
    P = _margin_draws(model.shape_p, model.scale_p, model.active_p, model.rho_p,
                      np.array([zp]), eps_p)[0]
    paths = Paths(p=P, d=D)
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    c_node = node_tables(tables, zd, zp)
    pred = GridPredictor(tables.w_grid, c_node, tables.tie_tol)
    models = {k: [pred] for k in range(1, m)}
    costs, _ = simulate_policy(cache, spec.B, spec.H, spec.E, models, H_bill=spec.H)
    mean = float(costs.mean())
    se = float(costs.std(ddof=1) / np.sqrt(n))
    Dn = _margin_draws(model.shape_d, model.scale_d, model.active_d, model.rho_d,
                       np.array([zd]), tables.eps_d)[0]
    Pn = _margin_draws(model.shape_p, model.scale_p, model.active_p, model.rho_p,
                       np.array([zp]), tables.eps_p)[0]
    table = thr_dp_value(Pn[:, 0] - Dn[:, 0], tables.w_grid, c_node[0], float(spec.H[0]),
                         float(spec.E[0]), spec.B)
    diff = mean - table
    tol = 4.0 * se + 1e-3 * float(spec.E0)
    return {"zeta": (zd, zp), "mean": mean, "se": se, "table": table, "diff": diff,
            "tol": tol, "ok": bool(abs(diff) <= tol)}


def thr_dp_value(g1: np.ndarray, w_grid: np.ndarray, c1: np.ndarray, H1: float,
                 E_first: float, B: float) -> float:
    """Same rule as expb.fidelity.thr_dp_value (copied so that this module does not import
    the torch-dependent fidelity module): mean over first-stop increments of
    1{g1 > B} E[0] + 1{g1 <= B} min(H[0], C_1(g1)), C_1 read by linear interpolation."""
    g1 = np.asarray(g1, dtype=float)
    w0, h = float(w_grid[0]), float(w_grid[1] - w_grid[0])
    v1 = np.minimum(H1, lin_interp(g1, w0, h, c1))
    return float(np.mean(np.where(g1 > B, E_first, v1)))


def g_cdp0_within(cdp_costs: np.ndarray, thr_costs: np.ndarray, E0: float) -> bool:
    """G-CDP0 for one route replicate: |mean(CDP - THR)| <= 4 SE + 1e-3 E0, where SE is the
    standard error of the paired per-day difference (the same dev days for both). The
    caller averages the replicates' statistics."""
    diff = np.asarray(cdp_costs, dtype=float) - np.asarray(thr_costs, dtype=float)
    se = float(diff.std(ddof=1) / np.sqrt(diff.size))
    return bool(abs(float(diff.mean())) <= 4.0 * se + 1e-3 * float(E0))
