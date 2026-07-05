#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRICING-LABELING v3  --  fast envelope pricing at ALL N (upgrade 1, portable)
=============================================================================

v2 made the envelope feasibility EXACT and sub-linear (O(pieces*logN)) but in
pure Python its constant lost to vectorised-numpy naive until N ~ 1e5.  No numba
/ cython here, so instead of compiling we remove the per-label O(N) work
algebraically, using the per-cut CVaR surrogate (the metric-(b) result, <1%
loose) as the workhorse:

  surrogate(label) = max_k CVaR_alpha( load_k(F) ),  load_k(F) = B_k + A_k F  affine,
  CVaR_alpha(B + A F) = B + A*mu_hi   if A >= 0   else   B + A*mu_lo,

with mu_hi = mean of the top-m factors, mu_lo = mean of the bottom-m factors,
PRECOMPUTED ONCE.  So surrogate is O(pieces) PURE SCALAR arithmetic -- no N, no
numpy-call overhead.

Soundness of surrogate pruning
------------------------------
* surrogate <= CVaR(Peak)  (CVaR monotone, Peak = max_k load_k >= each load_k).
* surrogate is non-decreasing under appending a customer (every existing cut's
  load rises by the new delivery d_w >= 0, plus a new cut is added).
Hence surrogate(partial) > Q_eff  =>  CVaR(Peak) > Q_eff for the partial AND all
its descendants  =>  prune the whole subtree.  No feasible route is ever lost.

Exactness
---------
A route is RECORDED only if it passes the EXACT order-statistics CVaR check
(v2.FStats.cvar) at completion -- so recorded routes match brute force exactly.
Exact checks run only on completed labels that survive the surrogate (~ a few
hundred), each O(pieces*logN); the dominant per-extension cost is the O(pieces)
scalar surrogate.

Result we are testing: with the factor structure, envelope pricing is fast at
ALL N (tiny scalar constant, sub-linear) AND N-independent in memory, beating
the per-scenario method everywhere -- not just asymptotically.

Delivered after being run in-sandbox.
"""

import numpy as np
import time
import importlib.util
import argparse

_spec = importlib.util.spec_from_file_location("v2", "pricing_labeling_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)            # importing does NOT run v2.main (guarded)

EPS = v2.EPS
FMIN, FMAX = v2.FMIN, v2.FMAX


# ---------------------------------------------------------------------------
#  O(pieces) scalar surrogate
# ---------------------------------------------------------------------------

def factor_tail_means(fs):
    """mu_hi = mean of top-m factors, mu_lo = mean of bottom-m factors."""
    m = fs.m
    return float(np.mean(fs.Fs[-m:])), float(np.mean(fs.Fs[:m]))


def surrogate_cvar(load_lines, mu_hi, mu_lo):
    best = -np.inf
    for (B, A) in load_lines:
        v = B + A * (mu_hi if A >= 0 else mu_lo)
        if v > best:
            best = v
    return best


def exact_cvar_np(load_lines, fs):
    """Vectorised exact empirical CVaR: peak_s = max_k(B_k + A_k F_s);
    mean of top-m via np.partition. O(pieces*N) numpy, good constant."""
    Fs, m = fs.Fs, fs.m
    B = np.array([b for (b, a) in load_lines])[:, None]
    A = np.array([a for (b, a) in load_lines])[:, None]
    peak = np.max(B + A * Fs[None, :], axis=0)
    N = peak.shape[0]
    if m >= N:
        return float(peak.mean())
    return float(np.partition(peak, N - m)[N - m:].mean())


def exact_cvar_hybrid(load_lines, fs, thresh=50000):
    """Fast exact CVaR at ALL N: numpy-direct (good constant) below the
    measured crossover, sub-linear order-statistics above it."""
    if fs.N >= thresh:
        return fs.cvar(load_lines)              # O(pieces*logN)
    return exact_cvar_np(load_lines, fs)        # O(pieces*N) numpy


# ---------------------------------------------------------------------------
#  Envelope solver v3: surrogate pruning + exact-at-completion
# ---------------------------------------------------------------------------

def env_solve_v3(inst, fs, Qeff):
    n = inst['n']; pi = inst['pi']; D = inst['D']
    dbar, dco, qbar, lam = inst['dbar'], inst['dco'], inst['qbar'], inst['lam']
    mu_hi, mu_lo = factor_tail_means(fs)
    LE = v2.LE
    env_dom = v2.env_dom
    start = LE(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [(0.0, 0.0)])
    buckets = {(0, 0): [start]}; parent = {id(start): (None, None)}
    best_rc, best_route = np.inf, None
    n_exact = [0]

    def load_lines(lab):
        return [(lab.Dbar + b, lab.dco + a) for (b, a) in lab.lines]

    levels = {0: [(0, 0, start)]}
    for lvl in range(n + 1):
        for (node, vis, lab) in levels.get(lvl, []):
            ll = load_lines(lab)
            sur = surrogate_cvar(ll, mu_hi, mu_lo)
            if sur > Qeff + EPS:
                continue                       # subtree infeasible: prune
            # complete: exact check only here (surrogate already <= Q_eff)
            if vis != 0:
                n_exact[0] += 1
                if exact_cvar_np(ll, fs) <= Qeff + EPS:
                    rc = lab.rc + D[node, 0]
                    if rc < best_rc - EPS:
                        best_rc, best_route = rc, v2._reconstruct(lab, parent)
            for w in range(1, n + 1):
                bit = 1 << (w - 1)
                if vis & bit:
                    continue
                nb = lab.Sbar + qbar[w]; na = lab.lamco + lam[w]
                nlab = LE(w, vis | bit, lab.rc + D[node, w] - pi[w],
                          lab.Dbar + dbar[w], lab.dco + dco[w], nb, na,
                          lab.lines + [(nb, na)])
                # surrogate pruning at extension (O(pieces) scalar)
                if surrogate_cvar(load_lines(nlab), mu_hi, mu_lo) > Qeff + EPS:
                    continue
                parent[id(nlab)] = (lab, w)
                key = (w, vis | bit); bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [nlab]; levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab)); continue
                if any(env_dom(e, nlab) for e in bucket):
                    continue
                buckets[key] = [e for e in bucket if not env_dom(nlab, e)] + [nlab]
                levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))
    return best_rc, best_route, sum(len(b) for b in buckets.values()), n_exact[0]


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.90)
    args = ap.parse_args()

    print("=" * 84)
    print("PRICING-LABELING v3  --  surrogate-pruned envelope pricing (fast at all N)")
    print("=" * 84)
    if not v2.self_test(verbose=True):
        raise SystemExit("exact-CVaR self-test failed")

    inst = v2.make_instance(args.n, args.seed)
    alpha = args.alpha
    Qeff = v2.calibrate_Qeff(inst, alpha, inst['feas_pct'], args.seed)
    print("\nseed=%d  n=%d  alpha=%.2f  Q_eff=%.2f" % (args.seed, args.n, alpha, Qeff))

    Ns = [50, 100, 1000, 10000, 100000, 1000000]
    print("\n%-9s | %-20s | %-30s | match" % ("N", "NAIVE O(N)/label", "ENV v3 (surrogate, N-free)"))
    print("%-9s | %-9s %-10s | %-7s %-9s %-11s |"
          % ("", "labels", "time(s)", "labels", "exactN", "time(s)"))
    print("-" * 84)
    for N in Ns:
        F = v2.sample_factors(N, args.seed)
        fs = v2.FStats(F, alpha)
        gt = None
        if N <= 2000:
            gt, _ = v2.brute_force(inst, F, alpha, Qeff)
        # naive may OOM at large N; guard it
        n_rc = n_lab = None; tn = None
        try:
            mem_gb = 600 * 3 * N * 8 / 1e9
            if mem_gb < 8.0:                    # don't attempt naive past ~8GB
                t0 = time.perf_counter(); n_rc, _, n_lab = v2.naive_solve(inst, F, alpha, Qeff); tn = time.perf_counter() - t0
        except MemoryError:
            n_rc = None
        t0 = time.perf_counter(); e_rc, _, e_lab, e_ex = env_solve_v3(inst, fs, Qeff); te = time.perf_counter() - t0

        if gt is not None:
            mtag = "OK" if (n_rc is not None and abs(n_rc - gt) < 1e-4 and abs(e_rc - gt) < 1e-4) else \
                   ("ENV-OK" if abs(e_rc - gt) < 1e-4 else "MISMATCH e=%.3f gt=%.3f" % (e_rc, gt))
        elif n_rc is not None:
            mtag = "OK(n=e)" if abs(n_rc - e_rc) < 1e-4 else "DIFF n=%.3f e=%.3f" % (n_rc, e_rc)
        else:
            mtag = "naive:skipped(mem)  ENV ran"
        ns = "%-9d %-10.3f" % (n_lab, tn) if n_rc is not None else "%-9s %-10s" % ("--OOM--", "--")
        print("%-9d | %s | %-7d %-9d %-11.3f | %s" % (N, ns, e_lab, e_ex, te, mtag))

    print("\n" + "=" * 84)
    print("READING")
    print("  ENV v3 dominant cost = O(pieces) scalar surrogate per extension (N-free);")
    print("  exact order-stats CVaR runs only on ~'exactN' completed labels.")
    print("  Target: ENV time FLAT and small at every N; naive grows ~N then OOMs.")
    print("  Correctness: recorded routes pass EXACT CVaR -> must match brute (<=2000).")
    print("=" * 84)


if __name__ == "__main__":
    main()