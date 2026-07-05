"""
Vectorized arc-cost construction for the giant-tour split (HARD / exact).

The benchmark's solve time was dominated NOT by the Bellman DP but by building admissibility:
cantelli_risk() called ~O(n*max_len) times per tour, in Python, recomputed for every
reliability level. This module computes the per-position load mean M(t) and variance Var(t)
for ALL segments of a tour in one vectorized numpy pass (z-independent), so that the
risk at any Cantelli multiplier z is a cheap elementwise reduction reused across the whole
reliability sweep. Distances are precomputed by prefix sums.

Identical results to the slow path -- verified by regression test (see __main__), so it is a
pure speedup, not an approximation.

Load model (matches dethloff_runner route_peaks / rho_route):
  route = tour[i:j] (customers at tour positions i..j-1); depot = node 0.
  start load L0 = sum deliveries; after customer at position t: L(t) = remaining deliveries
  + collected pickups. peak mean M = max(L0, max_t L(t)); per-position variance under
  equicorrelation: Var(t) = (1-rho)*Vind(t) + rho*S(t)^2.
"""

import numpy as np


class TourPrecomp:
    """Holds z-independent per-segment quantities for one giant tour."""
    __slots__ = ("L", "dist", "Mpos", "Vpos", "Mstart", "Vstart", "valid", "max_len")

    def __init__(self, tour, D, dbar, pbar, sig_d, sig_p, rho, max_len):
        tour = list(tour); L = len(tour); self.L = L; self.max_len = max_len
        idx = np.asarray(tour, dtype=int)
        dd = dbar[idx]; pp = pbar[idx]
        sd2 = sig_d[idx] ** 2; sp2 = sig_p[idx] ** 2
        sdv = sig_d[idx]; spv = sig_p[idx]

        # prefix sums over tour positions, length L+1
        Pd = np.concatenate([[0.0], np.cumsum(dd)])
        Pp = np.concatenate([[0.0], np.cumsum(pp)])
        PS2d = np.concatenate([[0.0], np.cumsum(sd2)])
        PS2p = np.concatenate([[0.0], np.cumsum(sp2)])
        PSd = np.concatenate([[0.0], np.cumsum(sdv)])
        PSp = np.concatenate([[0.0], np.cumsum(spv)])

        # edge prefix for distances: E[k] = sum_{s<k} D[tour[s], tour[s+1]]
        if L >= 2:
            edges = D[idx[:-1], idx[1:]]
            E = np.concatenate([[0.0], np.cumsum(edges)])
        else:
            E = np.array([0.0])
        d0 = D[0, idx]            # depot -> each customer
        dN = D[idx, 0]            # each customer -> depot

        # distance matrix dist[i,j] for segment tour[i:j], i<j<=L, j-i<=max_len
        INF = np.inf
        dist = np.full((L + 1, L + 1), INF)
        ii, jj = np.meshgrid(np.arange(L), np.arange(1, L + 1), indexing="ij")
        m = (jj > ii) & (jj - ii <= max_len)
        i_ = ii[m]; j_ = jj[m]
        dval = d0[i_] + (E[j_ - 1] - E[i_]) + dN[j_ - 1]
        dist[i_, j_] = dval
        self.dist = dist

        # per-position arrays indexed [i, j, t], t in [i, j-1]
        # M(i,j,t)   = Pd[j] - Pp[i] + A[t],     A[t]   = Pp[t+1] - Pd[t+1]
        # S(i,j,t)   = PSd[j] - PSp[i] + B[t],   B[t]   = PSp[t+1] - PSd[t+1]
        # Vind(i,j,t)= PS2d[j] - PS2p[i] + C2[t],C2[t]  = PS2p[t+1] - PS2d[t+1]
        A = Pp[1:L + 1] - Pd[1:L + 1]            # length L (t=0..L-1)
        B = PSp[1:L + 1] - PSd[1:L + 1]
        C2 = PS2p[1:L + 1] - PS2d[1:L + 1]

        i_ax = np.arange(L)[:, None, None]
        j_ax = np.arange(L + 1)[None, :, None]
        t_ax = np.arange(L)[None, None, :]
        valid = (i_ax <= t_ax) & (t_ax < j_ax) & (j_ax - i_ax <= max_len) & (j_ax > i_ax)
        self.valid = valid

        M = (Pd[None, :, None] - Pp[:L, None, None]) + A[None, None, :]
        S = (PSd[None, :, None] - PSp[:L, None, None]) + B[None, None, :]
        Vind = (PS2d[None, :, None] - PS2p[:L, None, None]) + C2[None, None, :]
        Var = (1.0 - rho) * Vind + rho * (S ** 2)
        # zero-out invalid so they never win a max; store sqrt-ready
        self.Mpos = np.where(valid, M, -np.inf)
        self.Vpos = np.where(valid, np.clip(Var, 0.0, None), 0.0)

        # start term (t = i-1): L0 mean = Pd[j]-Pd[i]; var = (1-rho)(PS2d[j]-PS2d[i]) + rho(PSd[j]-PSd[i])^2
        Mst = Pd[None, :] - Pd[:L, None]                       # [i, j]
        Vst = ((1.0 - rho) * (PS2d[None, :] - PS2d[:L, None])
               + rho * (PSd[None, :] - PSd[:L, None]) ** 2)
        startvalid = (np.arange(L + 1)[None, :] > np.arange(L)[:, None]) & \
                     (np.arange(L + 1)[None, :] - np.arange(L)[:, None] <= max_len)
        self.Mstart = np.where(startvalid, Mst, -np.inf)
        self.Vstart = np.where(startvalid, np.clip(Vst, 0.0, None), 0.0)

    def risk_matrix(self, z):
        """Cantelli risk[i,j] = max over positions (incl. start) of M + z*sqrt(Var). O(n^2 max_len)."""
        L = self.L
        rp = self.Mpos + z * np.sqrt(self.Vpos)          # [i, j, t]
        rp = np.where(self.valid, rp, -np.inf)
        risk_pos = rp.max(axis=2)                         # [i(<L), j]
        risk_start = self.Mstart + z * np.sqrt(self.Vstart)
        risk = np.maximum(risk_pos, risk_start)           # [i(<L), j]
        out = np.full((L + 1, L + 1), np.inf)
        out[:L, :] = risk
        return out

    def admissible_mask(self, z, Q):
        return self.risk_matrix(z) <= Q + 1e-9


def split_from_matrices(dist, mask, max_len):
    """Exact min-distance split given precomputed dist and admissibility mask. Returns (cost, routes_pos)
    where routes_pos is a list of (i,j) position pairs into the tour."""
    L = dist.shape[0] - 1; INF = np.inf
    dp = np.full(L + 1, INF); dp[0] = 0.0; back = np.full(L + 1, -1, dtype=int)
    for j in range(1, L + 1):
        lo = max(0, j - max_len)
        for i in range(lo, j):
            if mask[i, j] and dp[i] + dist[i, j] < dp[j]:
                dp[j] = dp[i] + dist[i, j]; back[j] = i
    if not np.isfinite(dp[L]):
        return INF, None
    routes = []; j = L
    while j > 0:
        i = back[j]; routes.append((i, j)); j = i
    return float(dp[L]), routes[::-1]


# ----------------------------------------------------------------------------
# Regression test against the slow per-segment path
# ----------------------------------------------------------------------------

def _slow_cantelli(route, dbar, pbar, sig_d, sig_p, z, rho):
    if not route:
        return 0.0
    d = dbar[route]; p = pbar[route]; sd = sig_d[route]; sp = sig_p[route]
    total_d = d.sum()
    M = np.concatenate(([total_d], total_d - np.cumsum(d) + np.cumsum(p)))
    v2 = (sd ** 2).sum()
    Vind = np.concatenate(([v2], v2 - np.cumsum(sd ** 2) + np.cumsum(sp ** 2)))
    s1 = sd.sum()
    S = np.concatenate(([s1], s1 - np.cumsum(sd) + np.cumsum(sp)))
    Vcorr = (1.0 - rho) * Vind + rho * S ** 2
    return float(np.max(M + z * np.sqrt(np.clip(Vcorr, 0.0, None))))


def _run_regression():
    rng = np.random.default_rng(0)
    n = 30; rho = 0.6; max_len = 8
    coords = rng.uniform(0, 100, (n, 2)); coords[0] = [50, 50]
    D = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    dbar = np.concatenate([[0.0], rng.uniform(5, 25, n - 1)])
    pbar = np.concatenate([[0.0], rng.uniform(5, 25, n - 1)])
    sig_d = 0.3 * dbar; sig_p = 0.3 * pbar
    tour = list(rng.permutation(np.arange(1, n)))
    pc = TourPrecomp(tour, D, dbar, pbar, sig_d, sig_p, rho, max_len)

    # 1) risk matrix matches slow cantelli for every segment and several z
    max_risk_err = 0.0
    for z in [0.0, 1.0, 2.5, 4.0]:
        R = pc.risk_matrix(z)
        for i in range(len(tour)):
            for j in range(i + 1, min(i + max_len, len(tour)) + 1):
                seg = tour[i:j]
                slow = _slow_cantelli(seg, dbar, pbar, sig_d, sig_p, z, rho)
                max_risk_err = max(max_risk_err, abs(R[i, j] - slow))
    print(f"[risk] max |fast - slow| over all segments x z = {max_risk_err:.2e}")

    # 2) distances match
    max_dist_err = 0.0
    for i in range(len(tour)):
        for j in range(i + 1, min(i + max_len, len(tour)) + 1):
            seg = tour[i:j]
            d = D[0, seg[0]] + sum(D[seg[k], seg[k + 1]] for k in range(len(seg) - 1)) + D[seg[-1], 0]
            max_dist_err = max(max_dist_err, abs(pc.dist[i, j] - d))
    print(f"[dist] max |fast - slow| over all segments       = {max_dist_err:.2e}")

    # 3) resulting routes identical to the slow split
    Q = float(np.percentile(dbar[1:] + pbar[1:], 90) * 4)
    for z in [1.0, 2.0, 3.0]:
        mask = pc.admissible_mask(z, Q)
        cost, routes = split_from_matrices(pc.dist, mask, max_len)
        # slow
        sig_dd, sig_pp = sig_d, sig_p
        def adm(seg, z=z):
            return _slow_cantelli(list(seg), dbar, pbar, sig_dd, sig_pp, z, rho) <= Q + 1e-9
        Ln = len(tour); INF = np.inf
        dp = [INF] * (Ln + 1); dp[0] = 0.0; bk = [-1] * (Ln + 1)
        for j in range(1, Ln + 1):
            for i in range(max(0, j - max_len), j):
                seg = tour[i:j]
                dd_ = D[0, seg[0]] + sum(D[seg[k], seg[k + 1]] for k in range(len(seg) - 1)) + D[seg[-1], 0]
                if adm(seg) and dp[i] + dd_ < dp[j]:
                    dp[j] = dp[i] + dd_; bk[j] = i
        ok = abs((cost if np.isfinite(cost) else -1) - (dp[Ln] if np.isfinite(dp[Ln]) else -1)) < 1e-6
        print(f"[split z={z}] fast cost={cost:.2f} slow cost={dp[Ln]:.2f}  match={ok}")

    print("\nREGRESSION:", "PASS" if (max_risk_err < 1e-6 and max_dist_err < 1e-6) else "FAIL")


if __name__ == "__main__":
    _run_regression()