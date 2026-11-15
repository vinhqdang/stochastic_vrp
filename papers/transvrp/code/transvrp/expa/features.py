"""State features for EXPERIMENT A (module M2). Exact definitions:
EXPERIMENT_A.md section 3.

Stage k (1 <= k <= m-1) is the decision epoch after stop k. Globals (G = 12)
and per-remaining-stop tokens (F = 11, one per j = k+1..m) are built for the
rows ``idx`` of a ``RouteCache``. Everything is vectorized over rows. Feature
values are clipped to [-5, 5] and returned as float32 (the batch functions).

Only columns 1..k of the route are read at stage k, so the stage-k outputs do
not depend on outcomes after stop k (test M2(b)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paths import Paths, RouteSpec

G, F = 12, 11
HISTORY_IDX = (1, 7, 8, 9, 10, 11)     # g2, g8..g12 (0-based)
MLP_DIM = G + 3
CLIP = 5.0
ARMS = ("full", "scalar")


@dataclass
class RouteStats:
    """Empirical per-stop statistics (ddof = 1) of the rows passed to ``fit``."""
    mu_p: np.ndarray
    mu_d: np.ndarray
    sd_p: np.ndarray
    sd_d: np.ndarray

    @staticmethod
    def fit(paths: Paths) -> "RouteStats":
        n = paths.p.shape[0]
        mu_p = paths.p.mean(axis=0)
        mu_d = paths.d.mean(axis=0)
        if n >= 2:
            sd_p = paths.p.std(axis=0, ddof=1)
            sd_d = paths.d.std(axis=0, ddof=1)
        else:
            sd_p = np.zeros_like(mu_p)
            sd_d = np.zeros_like(mu_d)
        return RouteStats(mu_p=mu_p, mu_d=mu_d, sd_p=sd_p, sd_d=sd_d)


class RouteCache:
    """Per-(route, paths, stats) arrays used by the batch builders.

    ``W[:, k-1]`` is the running sum, ``peak`` its running maximum, ``P``/``D``
    the running pickup and delivery sums, and ``zsum_*``/``cnt_*`` the running
    sums of standardized outcomes over stops with a positive fitted SD.
    """

    def __init__(self, spec: RouteSpec, paths: Paths, stats: RouteStats):
        p = np.asarray(paths.p, dtype=np.float64)
        d = np.asarray(paths.d, dtype=np.float64)
        if p.shape != d.shape or p.shape[1] != spec.m:
            raise ValueError("paths must have shape (N, m) matching the spec")
        self.spec = spec
        self.stats = stats
        self.paths = paths
        self.m = spec.m
        self.g = paths.g
        self.W = np.cumsum(self.g, axis=1)
        self.peak = np.maximum.accumulate(self.W, axis=1)
        self.P = np.cumsum(p, axis=1)
        self.D = np.cumsum(d, axis=1)

        mask_p = stats.sd_p > 0
        mask_d = stats.sd_d > 0
        zp = np.where(mask_p, (p - stats.mu_p) / np.where(mask_p, stats.sd_p, 1.0), 0.0)
        zd = np.where(mask_d, (d - stats.mu_d) / np.where(mask_d, stats.sd_d, 1.0), 0.0)
        self.zsum_p = np.cumsum(zp, axis=1)
        self.zsum_d = np.cumsum(zd, axis=1)
        self.cnt_p = np.cumsum(mask_p.astype(np.int64))
        self.cnt_d = np.cumsum(mask_d.astype(np.int64))

        mu_g = stats.mu_p - stats.mu_d
        self.CM = np.concatenate([[0.0], np.cumsum(mu_g)])            # CM[j] = sum_{i<=j}
        self.V = np.concatenate([[0.0], np.cumsum(stats.sd_p ** 2 + stats.sd_d ** 2)])


def _check_stage(cache: RouteCache, k: int) -> None:
    if not 1 <= k <= cache.m - 1:
        raise ValueError(f"stage k must be in [1, m-1] = [1, {cache.m - 1}]")


def _stage_arrays(cache: RouteCache, k: int, idx: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Float64 globals (n, G) and tokens (n, m-k, F), clipped to [-5, 5]."""
    _check_stage(cache, k)
    idx = np.asarray(idx, dtype=np.int64)
    spec, st, m = cache.spec, cache.stats, cache.m
    B, E0 = spec.B, spec.E0
    H, E = spec.H, spec.E
    n = idx.shape[0]

    W = cache.W[idx, k - 1]
    cnt_p = cache.cnt_p[k - 1]
    cnt_d = cache.cnt_d[k - 1]
    zbar_p = cache.zsum_p[idx, k - 1] / cnt_p if cnt_p > 0 else np.zeros(n)
    zbar_d = cache.zsum_d[idx, k - 1] / cnt_d if cnt_d > 0 else np.zeros(n)

    glob = np.empty((n, G), dtype=np.float64)
    glob[:, 0] = W / B
    glob[:, 1] = cache.peak[idx, k - 1] / B
    glob[:, 2] = k / m
    glob[:, 3] = (m - k) / m
    glob[:, 4] = m / 22.0
    glob[:, 5] = H[k - 1] / E0
    glob[:, 6] = E[k - 1] / E0
    glob[:, 7] = cache.P[idx, k - 1] / B
    glob[:, 8] = cache.D[idx, k - 1] / B
    glob[:, 9] = zbar_p
    glob[:, 10] = zbar_d
    glob[:, 11] = cache.g[idx, k - 1] / B

    js = np.arange(k + 1, m + 1)                       # remaining stops, 1-indexed
    nt = m - k
    tok = np.empty((n, nt, F), dtype=np.float64)
    tok[:, :, 0] = (st.mu_p[js - 1] / B)[None, :]
    tok[:, :, 1] = (st.mu_d[js - 1] / B)[None, :]
    tok[:, :, 2] = (st.sd_p[js - 1] / B)[None, :]
    tok[:, :, 3] = (st.sd_d[js - 1] / B)[None, :]
    tok[:, :, 4] = ((js - k) / m)[None, :]
    tok[:, :, 5] = (js / m)[None, :]
    t7 = np.where(js < m, H[np.minimum(js, m) - 1] / E0, 0.0)
    tok[:, :, 6] = t7[None, :]
    tok[:, :, 7] = (E[js - 1] / E0)[None, :]
    tok[:, :, 8] = (js == m).astype(np.float64)[None, :]
    gsum = (cache.CM[js] - cache.CM[k])[None, :]       # sum_{i=k+1..j} mu_g_i
    tok[:, :, 9] = (W[:, None] + gsum) / B
    s_hat = np.sqrt(np.maximum(cache.V[js] - cache.V[k], 0.0))
    denom = np.maximum(s_hat, 1e-6 * B)[None, :]
    tok[:, :, 10] = (B - W[:, None] - gsum) / denom

    np.clip(glob, -CLIP, CLIP, out=glob)
    np.clip(tok, -CLIP, CLIP, out=tok)
    return glob, tok


def stage_batch(cache: RouteCache, k: int, idx: np.ndarray, arm: str
                ) -> tuple[np.ndarray, np.ndarray]:
    """(glob (n, G), tok (n, m-k, F)) as float32. arm 'scalar' zeroes the
    history globals; 'full' keeps them."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    glob, tok = _stage_arrays(cache, k, idx)
    if arm == "scalar":
        glob[:, list(HISTORY_IDX)] = 0.0
    return glob.astype(np.float32), tok.astype(np.float32)


def mlp_batch(cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
    """(n, 15) float32: the 12 globals (no zeroing) and three route-ahead
    summaries: min_j t11 over j = k+1..m, (argmin_j t11 - k)/m, and
    min_{k<j<m} H[j-1]/E0 (0 if no such j). Uses the stored (clipped) t11."""
    glob, tok = _stage_arrays(cache, k, idx)
    n, nt, _ = tok.shape
    t11 = tok[:, :, 10]
    arg = t11.argmin(axis=1)
    mn = t11[np.arange(n), arg]
    s_arg = (arg + 1) / cache.m                        # (j* - k) / m, j* = k + 1 + arg
    if nt >= 2:
        s_h = tok[:, :-1, 6].min(axis=1)
    else:
        s_h = np.zeros(n)
    out = np.concatenate([glob, mn[:, None], s_arg[:, None], s_h[:, None]], axis=1)
    return out.astype(np.float32)
