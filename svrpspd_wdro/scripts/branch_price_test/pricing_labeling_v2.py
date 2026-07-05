#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRICING-LABELING PROTOTYPE v2  --  N-INDEPENDENT feasibility (upgrade 1)
=======================================================================

v1 showed the tropical-envelope DOMINANCE is N-independent and correct, but the
ENV solve time still grew with N because the empirical-CVaR FEASIBILITY check
was O(N) per label (shared with the naive method).  This upgrade removes that
last O(N) term for the rank-1 factor model.

The structural fact exploited
-----------------------------
Under rank-1 demand q_{i,s} = qbar_i + lam_i F_s, the route peak in scenario s is
    Peak_s = (Dbar + dco F_s) + max_k (b_k + a_k F_s)
           = max_k ( (Dbar+b_k) + (dco+a_k) F_s )  =: g(F_s),
a CONVEX piecewise-linear function of the SCALAR factor F_s.  Hence the empirical
CVaR (mean of the top m = ceil((1-alpha)N) values of {g(F_s)}) is determined by
the FACTOR ORDER STATISTICS, which are PRECOMPUTED ONCE (sorted F + prefix sums).
For a convex g the top-m values lie in the two tails of sorted F; the threshold
is found by binary search and the tail sums are read off the prefix sums in
O(pieces * log N).  No O(N) scan per label.

  exact_cvar(lines):  O(pieces * log N)   -- replaces the O(N log N) sort.

Guard rails
-----------
* SELF-TEST: exact_cvar is checked against the naive sort-based CVaR on thousands
  of random line sets / factor samples / alpha values, including degenerate
  cases (one line, equal slopes, monotone g).  The program ABORTS if any case
  disagrees beyond 1e-6.  Only then does it run the solver.
* END-TO-END: NAIVE and ENV must still match brute force at every N.

If the self-test passes and ENV stays correct with solve time flat in N while
NAIVE grows, the feasibility curse is removed for r=1 and the lead's tractability
claim is demonstrated end-to-end (for low rank).

Delivered after being run in-sandbox.
"""

import numpy as np
import itertools
import time
import bisect as _bisect
import argparse

EPS = 1e-7
FMIN, FMAX = -2.5, 2.5


# ===========================================================================
#  Fast empirical CVaR of a convex-PWL function of a scalar factor (r = 1)
# ===========================================================================

def upper_env(lines):
    """Upper envelope (max) of lines value = B + A*F.
    Returns pieces [(f_left, B, A), ...] with f_left ascending; piece k active
    on [f_left_k, f_left_{k+1})."""
    best = {}
    for (B, A) in lines:
        if (A not in best) or (B > best[A]):
            best[A] = B
    items = sorted(best.items())            # (A, B) ascending in A
    hull = []                               # list of (B, A)
    for A, B in items:
        while hull:
            B2, A2 = hull[-1]
            xn = (B2 - B) / (A - A2)         # crossing of last and new
            if len(hull) >= 2:
                B1, A1 = hull[-2]
                xp = (B1 - B2) / (A2 - A1)
                if xn <= xp:
                    hull.pop()
                    continue
            break
        hull.append((B, A))
    pieces = []
    for i, (B, A) in enumerate(hull):
        if i == 0:
            fl = -np.inf
        else:
            Bp, Ap = hull[i - 1]
            fl = (Bp - B) / (A - Ap)
        pieces.append((fl, B, A))
    return pieces


def g_at(pieces, f):
    return max(B + A * f for (_, B, A) in pieces)


class FStats:
    """Precomputed factor order statistics; provides O(pieces*logN) CVaR."""
    def __init__(self, F, alpha):
        self.Fs = np.sort(np.asarray(F, dtype=float))
        self.N = len(self.Fs)
        self.m = max(1, int(np.ceil((1.0 - alpha) * self.N)))
        self.PF = np.concatenate([[0.0], np.cumsum(self.Fs)])
        self.Fl = self.Fs.tolist()

    def _valley(self, pieces):
        Fs = self.Fs
        lo, hi = 0, self.N - 1
        while hi - lo > 2:
            m1 = lo + (hi - lo) // 3
            m2 = hi - (hi - lo) // 3
            if g_at(pieces, Fs[m1]) < g_at(pieces, Fs[m2]):
                hi = m2
            else:
                lo = m1
        best, bestv = lo, g_at(pieces, Fs[lo])
        for i in range(lo, hi + 1):
            v = g_at(pieces, Fs[i])
            if v < bestv:
                bestv, best = v, i
        return best

    def _count_left_gt(self, ist, tau, pieces):
        # i in [0, ist], g decreasing in i; return size of prefix with g > tau
        lo, hi = 0, ist + 1
        while lo < hi:
            mid = (lo + hi) // 2
            if g_at(pieces, self.Fs[mid]) > tau:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _count_right_gt(self, ist, tau, pieces):
        # i in [ist+1, N-1], g increasing; return (size of suffix with g > tau, first idx)
        N = self.N
        lo, hi = ist + 1, N
        while lo < hi:
            mid = (lo + hi) // 2
            if g_at(pieces, self.Fs[mid]) > tau:
                hi = mid
            else:
                lo = mid + 1
        return N - lo, lo

    def _sum_g(self, a, b, pieces):
        # sum of g(Fs[i]) for i in [a, b), via per-piece affine prefix sums
        if b <= a:
            return 0.0
        total = 0.0
        for k, (fl, B, A) in enumerate(pieces):
            f_next = pieces[k + 1][0] if k + 1 < len(pieces) else np.inf
            lo_i = 0 if fl == -np.inf else _bisect.bisect_left(self.Fl, fl)
            hi_i = self.N if f_next == np.inf else _bisect.bisect_left(self.Fl, f_next)
            lo_i = max(lo_i, a)
            hi_i = min(hi_i, b)
            if hi_i > lo_i:
                total += A * (self.PF[hi_i] - self.PF[lo_i]) + B * (hi_i - lo_i)
        return total

    def cvar(self, lines):
        """Empirical CVaR_alpha of g(F_s) = max_k(B_k + A_k F_s). O(pieces*logN)."""
        pieces = upper_env(lines)
        N, m = self.N, self.m
        ist = self._valley(pieces)
        gmin = g_at(pieces, self.Fs[ist])
        gmax = max(g_at(pieces, self.Fs[0]), g_at(pieces, self.Fs[N - 1]))
        if gmax <= gmin + 1e-15:
            return gmin
        lo, hi = gmin, gmax
        for _ in range(45):
            tau = 0.5 * (lo + hi)
            pL = self._count_left_gt(ist, tau, pieces)
            pR, _ = self._count_right_gt(ist, tau, pieces)
            if pL + pR >= m:
                lo = tau
            else:
                hi = tau
        tau = lo
        pL = self._count_left_gt(ist, tau, pieces)
        pR, _ = self._count_right_gt(ist, tau, pieces)
        cnt = pL + pR
        s = self._sum_g(0, pL, pieces) + self._sum_g(N - pR, N, pieces)
        s += (m - cnt) * tau            # fill remaining tail mass at the threshold
        return s / m


def cvar_naive(values, alpha):
    N = len(values)
    k = max(1, int(np.ceil((1.0 - alpha) * N)))
    return float(np.mean(np.sort(values)[-k:]))


# ---------------------------------------------------------------------------
#  SELF-TEST of exact CVaR vs naive sort
# ---------------------------------------------------------------------------

def self_test(verbose=True):
    rng = np.random.default_rng(12345)
    worst = 0.0
    ncases = 0
    for trial in range(4000):
        N = int(rng.integers(20, 1500))
        alpha = float(rng.choice([0.7, 0.8, 0.9, 0.95]))
        nlines = int(rng.integers(1, 9))
        F = np.clip(rng.normal(0, 1, N), FMIN, FMAX)
        lines = [(float(rng.normal(0, 20)), float(rng.normal(0, 5)))
                 for _ in range(nlines)]
        # occasionally force degenerate slopes
        if rng.random() < 0.2:
            a0 = float(rng.normal(0, 5))
            lines = [(float(rng.normal(0, 20)), a0) for _ in range(nlines)]
        fs = FStats(F, alpha)
        fast = fs.cvar(lines)
        gvals = np.array([g_at(upper_env(lines), f) for f in F])
        slow = cvar_naive(gvals, alpha)
        d = abs(fast - slow)
        worst = max(worst, d)
        ncases += 1
        if d > 1e-6:
            print("SELF-TEST FAIL: N=%d alpha=%.2f nlines=%d  fast=%.8f slow=%.8f d=%.2e"
                  % (N, alpha, nlines, fast, slow, d))
            print("lines:", lines)
            return False
    if verbose:
        print("SELF-TEST PASS: %d cases, max |fast-slow| = %.2e" % (ncases, worst))
    return True


# ===========================================================================
#  Instance + naive + envelope solvers (envelope now uses fast CVaR)
# ===========================================================================

def make_instance(n, seed):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n + 1, 2))
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    dbar = rng.uniform(10, 25, n + 1); dbar[0] = 0.0
    pbar = rng.uniform(10, 25, n + 1); pbar[0] = 0.0
    dload = rng.normal(0, 3.0, n + 1); dload[0] = 0.0
    pload = rng.normal(0, 3.0, n + 1); pload[0] = 0.0
    qbar = pbar - dbar
    lam = pload - dload
    pi = np.zeros(n + 1)
    pi[1:] = 2.0 * D[0, 1:] + rng.uniform(0, 10, n)
    return dict(n=n, D=D, dbar=dbar, pbar=pbar, dco=dload, qbar=qbar,
                lam=lam, pi=pi, feas_pct=55)


def sample_factors(N, seed):
    rng = np.random.default_rng(seed + 777)
    return np.clip(rng.normal(0, 1, N), FMIN, FMAX)


def route_peaks(order, inst, F):
    d = inst['dbar'][order][:, None] + np.outer(inst['dco'][order], F)
    q = inst['qbar'][order][:, None] + np.outer(inst['lam'][order], F)
    Ds = d.sum(axis=0)
    cum = np.cumsum(q, axis=0)
    M = np.vstack([np.zeros((1, len(F))), cum]).max(axis=0)
    return Ds + M


def route_rc(order, inst):
    D, pi = inst['D'], inst['pi']
    path = [0] + list(order) + [0]
    dist = sum(D[path[i], path[i + 1]] for i in range(len(path) - 1))
    return dist - pi[list(order)].sum()


def calibrate_Qeff(inst, alpha, pct, seed):
    F = sample_factors(2000, seed)
    n = inst['n']; custs = list(range(1, n + 1))
    rng = np.random.default_rng(seed + 5)
    vals = []
    for _ in range(400):
        k = rng.integers(1, n + 1)
        order = list(rng.choice(custs, size=k, replace=False))
        vals.append(cvar_naive(route_peaks(order, inst, F), alpha))
    return float(np.percentile(vals, pct))


def brute_force(inst, F, alpha, Qeff):
    n = inst['n']; best_rc, best_route = np.inf, None
    for k in range(1, n + 1):
        for sub in itertools.combinations(range(1, n + 1), k):
            for order in itertools.permutations(sub):
                if cvar_naive(route_peaks(list(order), inst, F), alpha) <= Qeff + EPS:
                    rc = route_rc(list(order), inst)
                    if rc < best_rc - EPS:
                        best_rc, best_route = rc, list(order)
    return best_rc, best_route


def _reconstruct(lab, parent):
    seq, cur = [], lab
    while cur is not None:
        p, w = parent.get(id(cur), (None, None))
        if w is not None:
            seq.append(w)
        cur = p
    return list(reversed(seq))


# ---- naive labeling (per-scenario, O(N)) ----
class LN:
    __slots__ = ('node', 'vis', 'rc', 'D', 'M', 'S')
    def __init__(s, node, vis, rc, D, M, S):
        s.node, s.vis, s.rc, s.D, s.M, s.S = node, vis, rc, D, M, S

def naive_dom(a, b):
    return (a.rc <= b.rc + EPS and np.all(a.D <= b.D + EPS)
            and np.all(a.M <= b.M + EPS) and np.all(a.S <= b.S + EPS))

def naive_solve(inst, F, alpha, Qeff):
    n = inst['n']; pi = inst['pi']; D = inst['D']
    dbar, dco, qbar, lam = inst['dbar'], inst['dco'], inst['qbar'], inst['lam']
    z = np.zeros(len(F))
    start = LN(0, 0, 0.0, z.copy(), z.copy(), z.copy())
    buckets = {(0, 0): [start]}; parent = {id(start): (None, None)}
    best_rc, best_route = np.inf, None
    levels = {0: [(0, 0, start)]}
    for lvl in range(n + 1):
        for (node, vis, lab) in levels.get(lvl, []):
            if vis != 0 and cvar_naive(lab.D + lab.M, alpha) <= Qeff + EPS:
                rc = lab.rc + D[node, 0]
                if rc < best_rc - EPS:
                    best_rc, best_route = rc, _reconstruct(lab, parent)
            for w in range(1, n + 1):
                bit = 1 << (w - 1)
                if vis & bit:
                    continue
                newD = lab.D + dbar[w] + dco[w] * F
                newS = lab.S + qbar[w] + lam[w] * F
                newM = np.maximum(lab.M, newS)
                if cvar_naive(newD + newM, alpha) > Qeff + EPS:
                    continue
                nlab = LN(w, vis | bit, lab.rc + D[node, w] - pi[w], newD, newM, newS)
                parent[id(nlab)] = (lab, w)
                key = (w, vis | bit); bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [nlab]; levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab)); continue
                if any(naive_dom(e, nlab) for e in bucket):
                    continue
                buckets[key] = [e for e in bucket if not naive_dom(nlab, e)] + [nlab]
                levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))
    return best_rc, best_route, sum(len(b) for b in buckets.values())


# ---- envelope labeling (rank-1; fast CVaR feasibility, analytical dominance) ----
class LE:
    __slots__ = ('node', 'vis', 'rc', 'Dbar', 'dco', 'Sbar', 'lamco', 'lines')
    def __init__(s, node, vis, rc, Dbar, dco, Sbar, lamco, lines):
        s.node, s.vis, s.rc = node, vis, rc
        s.Dbar, s.dco, s.Sbar, s.lamco, s.lines = Dbar, dco, Sbar, lamco, lines

def _aff_le(b1, a1, b2, a2):
    return (b1 + a1 * FMIN <= b2 + a2 * FMIN + EPS and
            b1 + a1 * FMAX <= b2 + a2 * FMAX + EPS)

def _env_le(L1, L2):
    cands = [FMIN, FMAX]
    alll = L1 + L2
    for i in range(len(alll)):
        for j in range(i + 1, len(alll)):
            b1, a1 = alll[i]; b2, a2 = alll[j]
            if abs(a1 - a2) > 1e-12:
                f = (b2 - b1) / (a1 - a2)
                if FMIN <= f <= FMAX:
                    cands.append(f)
    for f in cands:
        if max(b + a * f for (b, a) in L1) > max(b + a * f for (b, a) in L2) + EPS:
            return False
    return True

def env_dom(a, b):
    return (a.rc <= b.rc + EPS
            and _aff_le(a.Dbar, a.dco, b.Dbar, b.dco)
            and _aff_le(a.Sbar, a.lamco, b.Sbar, b.lamco)
            and _env_le(a.lines, b.lines))

def env_solve(inst, fs, Qeff):
    n = inst['n']; pi = inst['pi']; D = inst['D']
    dbar, dco, qbar, lam = inst['dbar'], inst['dco'], inst['qbar'], inst['lam']
    start = LE(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [(0.0, 0.0)])
    buckets = {(0, 0): [start]}; parent = {id(start): (None, None)}
    best_rc, best_route = np.inf, None

    def load_lines(lab):
        return [(lab.Dbar + b, lab.dco + a) for (b, a) in lab.lines]

    levels = {0: [(0, 0, start)]}
    for lvl in range(n + 1):
        for (node, vis, lab) in levels.get(lvl, []):
            if vis != 0 and fs.cvar(load_lines(lab)) <= Qeff + EPS:
                rc = lab.rc + D[node, 0]
                if rc < best_rc - EPS:
                    best_rc, best_route = rc, _reconstruct(lab, parent)
            for w in range(1, n + 1):
                bit = 1 << (w - 1)
                if vis & bit:
                    continue
                nb = lab.Sbar + qbar[w]; na = lab.lamco + lam[w]
                nlab = LE(w, vis | bit, lab.rc + D[node, w] - pi[w],
                          lab.Dbar + dbar[w], lab.dco + dco[w], nb, na,
                          lab.lines + [(nb, na)])
                if fs.cvar(load_lines(nlab)) > Qeff + EPS:
                    continue
                parent[id(nlab)] = (lab, w)
                key = (w, vis | bit); bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [nlab]; levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab)); continue
                if any(env_dom(e, nlab) for e in bucket):
                    continue
                buckets[key] = [e for e in bucket if not env_dom(nlab, e)] + [nlab]
                levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))
    return best_rc, best_route, sum(len(b) for b in buckets.values())


# ===========================================================================
#  Driver
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.90)
    args = ap.parse_args()

    print("=" * 80)
    print("PRICING-LABELING v2  --  N-independent feasibility (r=1)")
    print("=" * 80)
    if not self_test():
        print("ABORTING: fast CVaR self-test failed.")
        raise SystemExit(1)

    inst = make_instance(args.n, args.seed)
    alpha = args.alpha
    Qeff = calibrate_Qeff(inst, alpha, inst['feas_pct'], args.seed)
    print("\nseed=%d  n=%d customers  alpha=%.2f  Q_eff=%.2f"
          % (args.seed, args.n, alpha, Qeff))

    Ns = [50, 100, 500, 2000, 10000, 50000]
    Fref = sample_factors(Ns[0], args.seed)
    b_rc, b_route = brute_force(inst, Fref, alpha, Qeff)
    print("correctness anchor @N=%d: brute best rc=%.4f route=%s\n"
          % (Ns[0], b_rc, b_route))

    print("%-7s | %-20s | %-22s | match" % ("N", "NAIVE O(N)", "ENV O(pieces logN)"))
    print("%-7s | %-9s %-10s | %-9s %-11s |" % ("", "labels", "time(s)", "labels", "time(s)"))
    print("-" * 72)
    for N in Ns:
        F = sample_factors(N, args.seed)
        fs = FStats(F, alpha)
        gt_rc = None
        if N <= 2000:
            gt_rc, _ = brute_force(inst, F, alpha, Qeff)
        t0 = time.perf_counter(); n_rc, _, n_lab = naive_solve(inst, F, alpha, Qeff); tn = time.perf_counter() - t0
        t0 = time.perf_counter(); e_rc, _, e_lab = env_solve(inst, fs, Qeff); te = time.perf_counter() - t0
        if gt_rc is not None:
            ok = "OK" if (abs(n_rc - gt_rc) < 1e-4 and abs(e_rc - gt_rc) < 1e-4) else \
                 "MISMATCH n=%.3f e=%.3f gt=%.3f" % (n_rc, e_rc, gt_rc)
        else:
            ok = "OK(n=e)" if abs(n_rc - e_rc) < 1e-4 else "DIFF n=%.3f e=%.3f" % (n_rc, e_rc)
        print("%-7d | %-9d %-10.3f | %-9d %-11.3f | %s" % (N, n_lab, tn, e_lab, te, ok))

    print("\n" + "=" * 80)
    print("READING")
    print("  Self-test PASS => fast CVaR is exact.  ENV must match NAIVE/brute at every N.")
    print("  Target: ENV time FLAT in N (only pieces*logN), NAIVE time grows ~linearly.")
    print("  If ENV is flat and correct: feasibility curse removed for r=1 ->")
    print("    pricing is end-to-end N-independent (label count, dominance, feasibility)")
    print("    -> the Part VII tractability claim is demonstrated; build full B&P next.")
    print("=" * 80)


if __name__ == "__main__":
    main()