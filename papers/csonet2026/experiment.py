"""experiment.py -- algorithms and numerical study for the MWHED paper
(revised version).

Algorithms (all take sites in non-decreasing deadline order, after the
Assumption-1 preprocessing in `preprocess`):
  solve_exact        -- Theorem 3's time-indexed DP (exact optimum)
  solve_fptas        -- Theorem 4's value-scaled FPTAS, implemented as in
                        the proof (value-indexed dual DP); optional
                        `complete=True` adds the post-processing step of
                        Section 5.3 (greedily re-insert rejected sites)
  solve_equal_cost   -- Theorem 5's matroid greedy (union-find), for any
                        number m of identical vehicles (Theorem 6)
  solve_greedy_repair-- weighted Moore-Hodgson-style repair heuristic
                        (no guarantee; unbounded ratio, Proposition 7)
  solve_edd_naive    -- EDD with no reconsideration: every site is
                        dispatched in deadline order, late sites still
                        consume time (Proposition 8's baseline)
  solve_edd_skip     -- EDD with admission control: a site that would
                        miss its deadline is skipped (a stronger, still
                        guarantee-free baseline)

Numerical study (run `python3 experiment.py`; ~2-4 minutes):
  E1  accuracy vs n on the original generator (weights <= 100). NOTE: in
      this regime the FPTAS scaling factor is K = 1 for most settings
      (no rounding at all); the share is recorded and reported.
  E2  FPTAS with scaling ACTIVE: large weights, strongly correlated
      instances, and an adversarial "many sub-K sites" family.
  E3  runtime of the time-indexed exact DP vs the FPTAS as the
      processing-time range grows (weights up to 1e6, scaling active).
  E4  robustness of E1 to heavy-tailed weights and tight deadlines.

Self-contained; imports nothing from the BATON/TEMPO/PARCEL codebases.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np


# --------------------------------------------------------------- helpers
def preprocess(p, d, w):
    """Assumption 1: delete sites with p_i > d_i; return the remaining
    sites sorted by non-decreasing deadline, plus the original indices."""
    keep = [i for i in range(len(p)) if p[i] <= d[i]]
    keep.sort(key=lambda i: (d[i], i))
    return ([p[i] for i in keep], [d[i] for i in keep],
            [w[i] for i in keep], keep)


def edd_feasible(subset, p, d):
    """Is `subset` (indices into deadline-sorted arrays) feasible?"""
    c = 0
    for i in sorted(subset):
        c += p[i]
        if c > d[i]:
            return False
    return True


# ------------------------------------------------------------- algorithms
def solve_exact(p, d, w):
    """Theorem 3: time-indexed DP, O(nP). Returns (value, subset)."""
    n = len(p)
    if n == 0:
        return 0.0, set()
    P = sum(p)
    NEG = -np.inf
    f = np.full(P + 1, NEG)
    f[0] = 0.0
    choice = np.zeros((n, P + 1), dtype=bool)
    ts = np.arange(P + 1)
    for i in range(n):
        pi = p[i]
        cand = f[:P + 1 - pi] + w[i]
        ok = (cand > f[pi:]) & (ts[pi:] <= d[i]) & np.isfinite(f[:P + 1 - pi])
        newf = f.copy()
        newf[pi:] = np.where(ok, cand, f[pi:])
        choice[i, pi:] = ok
        f = newf
    t = int(np.argmax(f))
    best = float(f[t])
    inc = set()
    for i in range(n - 1, -1, -1):
        if choice[i, t]:
            inc.add(i)
            t -= p[i]
    return best, inc


def solve_fptas(p, d, w, eps, complete=False, info=None):
    """Theorem 4: scale weights by K = max(1, eps*max(w)/n), run the
    value-indexed dual DP g(i, v) = min dispatch time reaching scaled
    value v with a feasible subset of sites 1..i. Returns the TRUE
    (unscaled) weight of the returned subset. If `complete`, rejected
    sites are greedily re-inserted (heaviest first) while the set stays
    feasible; this can only increase the returned weight."""
    n = len(p)
    if n == 0:
        return 0.0, set()
    wmax = max(w)
    K = max(1.0, eps * wmax / n)
    wp = [int(math.floor(x / K)) for x in w]
    V = sum(wp)
    INF = np.inf
    g = np.full(V + 1, INF)
    g[0] = 0.0
    choice = np.zeros((n, V + 1), dtype=bool)
    for i in range(n):
        wi = wp[i]
        if wi == 0:
            continue
        cand = g[:V + 1 - wi] + p[i]
        ok = (cand <= d[i]) & (cand < g[wi:])
        newg = g.copy()
        newg[wi:] = np.where(ok, cand, g[wi:])
        choice[i, wi:] = ok
        g = newg
    finite = np.where(np.isfinite(g))[0]
    v = int(finite.max())
    inc = set()
    for i in range(n - 1, -1, -1):
        if choice[i, v]:
            inc.add(i)
            v -= wp[i]
    if complete:
        for i in sorted(set(range(n)) - inc, key=lambda j: -w[j]):
            if edd_feasible(inc | {i}, p, d):
                inc.add(i)
    if info is not None:
        info.update(K=K, scaling_active=(K > 1.0), cells=n * (V + 1))
    return float(sum(w[i] for i in inc)), inc


def solve_edd_naive(p, d, w):
    c = 0
    val = 0.0
    inc = set()
    for i in range(len(p)):
        c += p[i]
        if c <= d[i]:
            val += w[i]
            inc.add(i)
    return val, inc


def solve_edd_skip(p, d, w):
    c = 0
    val = 0.0
    inc = set()
    for i in range(len(p)):
        if c + p[i] <= d[i]:
            c += p[i]
            val += w[i]
            inc.add(i)
    return val, inc


def solve_greedy_repair(p, d, w):
    kept = []
    total = 0
    for i in range(len(p)):
        kept.append(i)
        total += p[i]
        while total > d[i] and kept:
            worst = min(kept, key=lambda j: w[j] / p[j])
            kept.remove(worst)
            total -= p[worst]
    return float(sum(w[i] for i in kept)), set(kept)


def solve_equal_cost(d, w, p, m=1):
    """Theorems 5/6. All dispatch times equal p; m identical vehicles.
    Sort by decreasing weight; give each site the latest free slot at a
    position <= D_i = floor(d_i/p) (m slots per position), found with a
    union-find over positions. Returns (value, subset of indices)."""
    n = len(d)
    D = [min(di // p, n) for di in d]
    parent = list(range(n + 1))
    cap = [m] * (n + 1)

    def find(q):
        root = q
        while parent[root] != root:
            root = parent[root]
        while parent[q] != root:
            parent[q], q = root, parent[q]
        return root

    inc = set()
    val = 0.0
    for i in sorted(range(n), key=lambda j: -w[j]):
        q = find(D[i])
        if q >= 1:
            inc.add(i)
            val += w[i]
            cap[q] -= 1
            if cap[q] == 0:
                parent[q] = q - 1
    return val, inc


# ------------------------------------------------------------- generators
def _sorted(p, d, w):
    idx = sorted(range(len(p)), key=lambda i: d[i])
    return [p[i] for i in idx], [d[i] for i in idx], [w[i] for i in idx]


def random_instance(n, rng, p_max=20, w_max=100, horizon_frac=1.0,
                    weight="uniform"):
    p = [rng.randint(1, p_max) for _ in range(n)]
    if weight == "uniform":
        w = [rng.randint(1, w_max) for _ in range(n)]
    elif weight == "heavy":
        w = [max(1, int(rng.expovariate(1 / 20))) for _ in range(n)]
    else:
        raise ValueError(weight)
    horizon = max(1, int(horizon_frac * sum(p)))
    d = [rng.randint(1, horizon) for _ in range(n)]
    return preprocess(p, d, w)[:3]


def correlated_instance(n, rng, p_max=1000, horizon_frac=0.5):
    """Strongly correlated knapsack-style instance: w_i = 1000*p_i + 500,
    common deadline; the hard family for knapsack-type FPTASs."""
    p = [rng.randint(p_max // 10, p_max) for _ in range(n)]
    w = [1000 * x + 500 for x in p]
    D = max(1, int(horizon_frac * sum(p)))
    d = [D] * n
    return preprocess(p, d, w)[:3]


def stress_instance(n, eps, M=10 ** 6):
    """One dominant site (weight M) plus n-1 sites each just below the
    scaling granule K = eps*M/n, so each rounds to scaled weight 0. All
    sites fit together; the FPTAS (without completion) discards every
    sub-K site, losing nearly eps*M of weight."""
    K = eps * M / n
    small = max(1, int(math.ceil(K)) - 1)
    p = [1] * n
    w = [M] + [small] * (n - 1)
    d = [n] * n
    return preprocess(p, d, w)[:3]


# ------------------------------------------------------------ experiments
def _mean(x):
    return sum(x) / len(x)


E1_SIZES = (10, 15, 20, 25, 30, 50, 75, 100, 200, 400)
E1_TRIALS = 50


def run_accuracy(seed=20260615):
    rng = random.Random(seed)
    out = {}
    for n in E1_SIZES:
        r = {k: [] for k in ("f02", "f01", "rep", "skip", "naive")}
        k1 = {"f02": 0, "f01": 0}
        for _ in range(E1_TRIALS):
            p, d, w = random_instance(n, rng)
            opt, _ = solve_exact(p, d, w)
            for key, eps in (("f02", 0.2), ("f01", 0.1)):
                info = {}
                v, _ = solve_fptas(p, d, w, eps, info=info)
                k1[key] += (not info["scaling_active"])
                r[key].append(v / opt)
            r["rep"].append(solve_greedy_repair(p, d, w)[0] / opt)
            r["skip"].append(solve_edd_skip(p, d, w)[0] / opt)
            r["naive"].append(solve_edd_naive(p, d, w)[0] / opt)
        out[n] = dict(fptas02=_mean(r["f02"]), fptas01=_mean(r["f01"]),
                      repair=_mean(r["rep"]), edd_skip=_mean(r["skip"]),
                      naive=_mean(r["naive"]),
                      fptas01_min=min(r["f01"]), repair_min=min(r["rep"]),
                      naive_min=min(r["naive"]), trials=E1_TRIALS,
                      share_unscaled_eps02=k1["f02"] / E1_TRIALS,
                      share_unscaled_eps01=k1["f01"] / E1_TRIALS)
        print(f"E1 n={n:3d} FPTAS.2={out[n]['fptas02']:.3f} "
              f"FPTAS.1={out[n]['fptas01']:.3f} repair={out[n]['repair']:.3f} "
              f"EDD-skip={out[n]['edd_skip']:.3f} naive={out[n]['naive']:.3f} "
              f"unscaled(.2/.1)={out[n]['share_unscaled_eps02']:.2f}/"
              f"{out[n]['share_unscaled_eps01']:.2f}")
    return out


EPS_GRID = (0.5, 0.3, 0.2, 0.1, 0.05)


def run_scaling_active(seed=20260620, sizes=(50, 100), trials=30):
    """E2: FPTAS accuracy/size/time where scaling is genuinely active."""
    rows = []
    for n in sizes:
        rows += _scaling_active_n(random.Random(seed + n), n, trials)
    return rows


def _scaling_active_n(rng, n, trials):
    regimes = {
        "uniform weights <= 10^6": lambda: random_instance(n, rng, w_max=10 ** 6),
        "strongly correlated": lambda: correlated_instance(n, rng),
    }
    rows = []
    for name, gen in regimes.items():
        insts = [gen() for _ in range(trials)]
        opts = [solve_exact(*x)[0] for x in insts]
        for eps in EPS_GRID:
            rs, rc, Ks, cells, ts = [], [], [], [], []
            sub = 0
            for (p, d, w), opt in zip(insts, opts):
                info = {}
                t0 = time.perf_counter()
                v, _ = solve_fptas(p, d, w, eps, info=info)
                ts.append(1000 * (time.perf_counter() - t0))
                vc, _ = solve_fptas(p, d, w, eps, complete=True)
                rs.append(v / opt)
                sub += v < opt
                rc.append(vc / opt)
                Ks.append(info["K"])
                cells.append(info["cells"])
            rows.append(dict(regime=name, n=n, eps=eps, guarantee=1 - eps,
                             mean_ratio=_mean(rs), min_ratio=min(rs),
                             mean_ratio_completed=_mean(rc),
                             min_ratio_completed=min(rc),
                             share_suboptimal=sub / trials,
                             mean_K=_mean(Ks), mean_cells=_mean(cells),
                             mean_ms=_mean(ts)))
            r = rows[-1]
            print(f"E2 n={n} {name:24s} eps={eps:4.2f} mean={r['mean_ratio']:.5f} "
                  f"min={r['min_ratio']:.5f} (>= {1 - eps:.2f}) "
                  f"completed-min={r['min_ratio_completed']:.5f} "
                  f"K={r['mean_K']:.0f} cells={r['mean_cells']:.0f} "
                  f"{r['mean_ms']:.1f}ms")
    # adversarial family (deterministic)
    for eps in EPS_GRID:
        p, d, w = stress_instance(n, eps)
        opt, _ = solve_exact(p, d, w)
        v, _ = solve_fptas(p, d, w, eps)
        vc, _ = solve_fptas(p, d, w, eps, complete=True)
        rows.append(dict(regime="adversarial (sub-K sites)", n=n, eps=eps,
                         guarantee=1 - eps, mean_ratio=v / opt,
                         min_ratio=v / opt, mean_ratio_completed=vc / opt,
                         min_ratio_completed=vc / opt, share_suboptimal=1.0,
                         mean_K=max(1.0, eps * max(w) / n), mean_cells=0,
                         mean_ms=0.0))
        print(f"E2 n={n} adversarial eps={eps:4.2f} ratio={v / opt:.4f} "
              f"(>= {1 - eps:.2f})  completed={vc / opt:.4f}")
    return rows


def run_runtime(seed=20260616, n=40, eps=0.1, trials=5):
    """E3: exact (time-indexed) vs FPTAS as p_max grows; weights up to
    10^6 so the FPTAS scaling is active (K > 1) throughout."""
    rng = random.Random(seed)
    rows = []
    for p_max in (50, 200, 800, 3200, 12800):
        te, tf, act = [], [], []
        for _ in range(trials):
            p, d, w = random_instance(n, rng, p_max=p_max, w_max=10 ** 6)
            t0 = time.perf_counter()
            solve_exact(p, d, w)
            t1 = time.perf_counter()
            info = {}
            solve_fptas(p, d, w, eps, info=info)
            t2 = time.perf_counter()
            te.append(1000 * (t1 - t0))
            tf.append(1000 * (t2 - t1))
            act.append(info["scaling_active"])
        rows.append(dict(p_max=p_max, exact_ms=_mean(te), fptas_ms=_mean(tf),
                         scaling_active=all(act)))
        print(f"E3 p_max={p_max:6d} exact={rows[-1]['exact_ms']:9.2f} ms "
              f"FPTAS={rows[-1]['fptas_ms']:7.2f} ms active={all(act)}")
    return rows


def run_robustness(seed=20260618, n=30, trials=20):
    """E4: E1's comparison under heavy-tailed weights / tight deadlines."""
    rng = random.Random(seed)
    regimes = [
        ("heavy-tailed weights",
         lambda: random_instance(n, rng, weight="heavy")),
        ("tight deadlines (0.5x horizon)",
         lambda: random_instance(n, rng, horizon_frac=0.5)),
    ]
    rows = []
    for label, gen in regimes:
        r = {k: [] for k in ("f", "rep", "skip", "naive")}
        for _ in range(trials):
            p, d, w = gen()
            opt, _ = solve_exact(p, d, w)
            if opt <= 0:
                continue
            r["f"].append(solve_fptas(p, d, w, 0.1)[0] / opt)
            r["rep"].append(solve_greedy_repair(p, d, w)[0] / opt)
            r["skip"].append(solve_edd_skip(p, d, w)[0] / opt)
            r["naive"].append(solve_edd_naive(p, d, w)[0] / opt)
        rows.append(dict(regime=label, fptas=_mean(r["f"]),
                         repair=_mean(r["rep"]), edd_skip=_mean(r["skip"]),
                         naive=_mean(r["naive"])))
        print(f"E4 {label:32s} FPTAS={rows[-1]['fptas']:.3f} "
              f"repair={rows[-1]['repair']:.3f} "
              f"EDD-skip={rows[-1]['edd_skip']:.3f} "
              f"naive={rows[-1]['naive']:.3f}")
    return rows


def main():
    res = dict(accuracy=run_accuracy())
    print()
    res["scaling_active"] = run_scaling_active()
    print()
    res["runtime"] = run_runtime()
    print()
    res["robustness"] = run_robustness()
    with open("results_illustration.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
