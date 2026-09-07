"""Verify the two claims added to the theory section (Propositions 1(c)
and 4 in THEORY.md), by exhaustive search rather than algebra.

CLAIM A -- the separation is not merely a factor n; it is UNBOUNDED once
saturation is strong enough.

  n receivers, n items, item i is the only item relevant to receiver i,
  broadcast is FREE (billed once), and a receiver pays `lam` for every
  item it receives that is irrelevant to it.

  Broadcasting a set S to everyone yields |S| - lam*|S|*(n-1): each item
  satisfies its one receiver and distracts the other n-1. For
  lam > 1/(n-1) every non-empty S is strictly negative, so the best
  seed set is the EMPTY one, with utility 0. Per-receiver allocation
  sends item i to receiver i alone and gets n with no penalty at all.

  Ratio n/0 = infinity. Proposition 1's factor-n instance is therefore
  the BENIGN case (it is monotone and submodular); the general gap has
  no finite bound.

CLAIM B -- approximation transfers through the resource-allocation DP.

  If each receiver's value curve is known only to within a factor
  (1-eps), the DP over budget splits still returns an allocation worth
  at least (1-eps) of the true optimum. Checked here against brute-force
  enumeration of every allocation, over random instances, with the
  per-receiver curves deliberately corrupted downward by up to eps.

Run: python3 unbounded_check.py
"""

import random
from itertools import combinations, product

# ---------------------------------------------------------------- claim A


def seed_set_best(n, lam):
    """Best broadcast set under free broadcast + irrelevance penalty."""
    best = 0.0                                    # the empty set is feasible
    items = range(n)
    for size in range(n + 1):
        for S in combinations(items, size):
            # each receiver i: +1 if item i in S, -lam for each other item
            u = sum((1.0 if i in S else 0.0) - lam * len([j for j in S if j != i])
                    for i in range(n))
            best = max(best, u)
    return best


def per_receiver_best(n, lam):
    """Best per-receiver allocation, exhaustive over all 2^(n*n) options
    for small n; the diagonal is optimal but we do not assume it."""
    items = list(range(n))
    subsets = [frozenset(S) for size in range(n + 1)
               for S in combinations(items, size)]
    best = 0.0
    for alloc in product(subsets, repeat=n):
        u = sum((1.0 if i in alloc[i] else 0.0)
                - lam * len([j for j in alloc[i] if j != i])
                for i in range(n))
        best = max(best, u)
    return best


def check_claim_a():
    print("CLAIM A: separation is unbounded under strong saturation")
    print(f"{'n':>3} {'lam':>6} {'seed-set OPT':>13} {'per-recv OPT':>13} {'ratio':>8}")
    ok = True
    for n in (2, 3, 4):
        lam = 1.0 / (n - 1) + 0.01            # just past the threshold
        s = seed_set_best(n, lam)
        p = per_receiver_best(n, lam)
        ratio = "inf" if s <= 1e-12 else f"{p / s:.2f}"
        print(f"{n:>3} {lam:>6.3f} {s:>13.4f} {p:>13.4f} {ratio:>8}")
        ok &= (abs(s) < 1e-12 and abs(p - n) < 1e-12)
    print("  -> seed set pinned at 0 while per-receiver attains n:",
          "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- claim B


def knap_curve(costs, vals, B):
    """Exact v(b) for b = 0..B: best modular value at cost <= b."""
    best = [0.0] * (B + 1)
    for c, v in zip(costs, vals):
        for b in range(B, c - 1, -1):
            best[b] = max(best[b], best[b - c] + v)
    # make it monotone in b (it already is, by construction of <=)
    for b in range(1, B + 1):
        best[b] = max(best[b], best[b - 1])
    return best


def dp_allocate(curves, B):
    """max sum_i curve_i(b_i) s.t. sum b_i <= B."""
    acc = [0.0] * (B + 1)
    for cur in curves:
        nxt = [-1e18] * (B + 1)
        for b in range(B + 1):
            for t in range(b + 1):
                cand = acc[b - t] + cur[t]
                if cand > nxt[b]:
                    nxt[b] = cand
        acc = nxt
    return max(acc)


def brute_allocate(n, costs, vals, B):
    """True optimum by enumerating every allocation."""
    m = len(costs)
    subsets = []
    for mask in range(1 << m):
        c = sum(costs[j] for j in range(m) if mask >> j & 1)
        v = sum(vals[j] for j in range(m) if mask >> j & 1)
        subsets.append((c, v))
    best = 0.0
    for alloc in product(subsets, repeat=n):
        if sum(a[0] for a in alloc) <= B:
            best = max(best, sum(a[1] for a in alloc))
    return best


def check_claim_b(trials=400, eps=0.2, seed=11):
    print("\nCLAIM B: (1-eps) curves give a (1-eps) allocation")
    rng = random.Random(seed)
    worst = 1.0
    for _ in range(trials):
        n = rng.randint(2, 3)
        m = rng.randint(2, 4)
        B = rng.randint(2, 8)
        costs = [rng.randint(1, 4) for _ in range(m)]
        vals = [rng.uniform(0, 1) for _ in range(m)]
        true = brute_allocate(n, costs, vals, B)
        if true <= 1e-12:
            continue
        exact = [knap_curve(costs, vals, B) for _ in range(n)]
        # corrupt each curve downward by up to eps, staying achievable
        approx = [[x * (1.0 - rng.uniform(0, eps)) for x in cur] for cur in exact]
        got = dp_allocate(approx, B)
        worst = min(worst, got / true)
    print(f"  worst observed value / OPT over {trials} random instances: "
          f"{worst:.4f}  (guarantee: >= {1 - eps:.2f})")
    ok = worst >= 1 - eps - 1e-9
    print("  ->", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    a = check_claim_a()
    b = check_claim_b()
    print("\nALL CHECKS", "PASS" if (a and b) else "FAIL")
