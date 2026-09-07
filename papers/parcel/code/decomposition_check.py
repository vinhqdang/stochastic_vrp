"""Verify the multi-receiver decomposition and its guarantee.

THE STRUCTURE (Proposition 5 in THEORY.md)
------------------------------------------
The objective is SEPARABLE across receivers -- U = sum_i u_i(S_i) --
and the receivers are coupled only by the single shared budget. So the
problem factors: choose a budget level b_i per receiver, then solve each
receiver independently at its level.

Define, per receiver,
    v_i(b) = max { rel_i(S) - rho_i(c(S)) : c(S) <= b }
and then combine with a resource-allocation DP over levels:
    maximise sum_i v_i(b_i)  subject to  sum_i b_i <= B.

Two things this buys, and both are checked here.

1. EXACTNESS. With the inner problem solved exactly, the level DP is
   exactly optimal -- separability plus one linear constraint is all
   that is needed. Checked against brute force over joint allocations.

2. rho MAY BE ARBITRARY. The saturation penalty enters only through
   v_i(b), evaluated per level, so it need not be convex, monotone in
   any useful sense, or even continuous. A cliff works. This is the
   concrete advantage over the greedy admission price, which needs
   rho' increasing (i.e. rho convex) for its rising-bar argument -- and
   the measured curves are S-shaped, not convex (PROJECT.md 11.1).
   Checked by running the same test with a cliff-shaped penalty.

3. THE APPROXIMATE CASE. With only an alpha-approximation available for
   the inner monotone-submodular-knapsack, the achievable guarantee has
   the regularized (Harshaw-style) form
       U(alg) >= alpha * sum_i rel_i(S_i*) - sum_i rho_i(c(S_i*))
   rather than a multiplicative alpha * OPT. Simulated here by
   degrading the inner oracle to alpha and confirming the bound holds.

Run: python3 decomposition_check.py
"""

import random
from itertools import combinations

E_INV = 1.0 - 1.0 / 2.718281828459045     # 1 - 1/e


def subsets(items):
    return [frozenset(s) for r in range(len(items) + 1)
            for s in combinations(items, r)]


def make_world(n_items, n_recv, rng, penalty="convex"):
    """Coverage-style (submodular) relevance, plus a load penalty."""
    universe = list(range(n_items))
    # Each item covers a random set of "facts"; coverage is monotone
    # submodular, the standard well-behaved case for the inner problem.
    covers = {j: frozenset(rng.sample(universe, rng.randint(1, 3)))
              for j in range(n_items)}
    cost = {j: rng.randint(1, 3) for j in range(n_items)}
    wants = [frozenset(rng.sample(universe, rng.randint(1, 4)))
             for _ in range(n_recv)]

    def rel(i, S):
        got = frozenset().union(*[covers[j] for j in S]) if S else frozenset()
        return float(len(got & wants[i]))

    if penalty == "convex":
        def rho(i, load):
            return 0.12 * load * load / 4.0
    elif penalty == "cliff":
        # Flat, then a sharp drop -- the shape the measurements suggest,
        # and the shape a convexity-dependent rule cannot handle.
        def rho(i, load):
            return 0.0 if load <= 4 else 3.0
    else:
        def rho(i, load):
            return 0.0

    def c(S):
        return sum(cost[j] for j in S)

    return rel, rho, c, cost


def brute_force(n_items, n_recv, rel, rho, c, budget):
    """Exact optimum over joint allocations. Exponential; small n only."""
    subs = subsets(range(n_items))
    best, best_alloc = float("-inf"), None

    def rec(i, spent, chosen, total):
        nonlocal best, best_alloc
        if i == n_recv:
            if total > best:
                best, best_alloc = total, list(chosen)
            return
        for S in subs:
            cs = c(S)
            if spent + cs > budget:
                continue
            chosen.append(S)
            rec(i + 1, spent + cs, chosen,
                total + rel(i, S) - rho(i, cs))
            chosen.pop()

    rec(0, 0, [], 0.0)
    return best, best_alloc


def inner_exact(i, rel, rho, c, subs, level):
    """v_i(level): best net utility for receiver i spending <= level."""
    best, arg = float("-inf"), frozenset()
    for S in subs:
        cs = c(S)
        if cs > level:
            continue
        val = rel(i, S) - rho(i, cs)
        if val > best:
            best, arg = val, S
    return best, arg


def level_dp(n_items, n_recv, rel, rho, c, budget, alpha=1.0):
    """Decomposition: per-receiver curves, then a DP over budget levels.

    `alpha` degrades the inner oracle's RELEVANCE term only, simulating
    an approximate submodular-knapsack solver while the penalty is paid
    in full -- which is what produces the regularized guarantee form.
    """
    subs = subsets(range(n_items))
    curves = []
    for i in range(n_recv):
        row = []
        for b in range(budget + 1):
            best, arg = float("-inf"), frozenset()
            for S in subs:
                cs = c(S)
                if cs > b:
                    continue
                val = alpha * rel(i, S) - rho(i, cs)
                if val > best:
                    best, arg = val, S
            row.append((best, arg))
        curves.append(row)

    # DP over receivers x remaining budget
    NEG = float("-inf")
    dp = [[NEG] * (budget + 1) for _ in range(n_recv + 1)]
    pick = [[None] * (budget + 1) for _ in range(n_recv + 1)]
    for b in range(budget + 1):
        dp[n_recv][b] = 0.0
    for i in range(n_recv - 1, -1, -1):
        for b in range(budget + 1):
            for spend in range(b + 1):
                val, arg = curves[i][spend]
                if val == NEG or dp[i + 1][b - spend] == NEG:
                    continue
                cand = val + dp[i + 1][b - spend]
                if cand > dp[i][b]:
                    dp[i][b] = cand
                    pick[i][b] = (spend, arg)

    alloc, b = [], budget
    for i in range(n_recv):
        spend, arg = pick[i][b]
        alloc.append(arg)
        b -= spend
    # Score the recovered allocation under the TRUE objective
    true = sum(rel(i, S) - rho(i, c(S)) for i, S in enumerate(alloc))
    return dp[0][budget], true, alloc


def main():
    print("1. EXACT inner oracle: does the level DP match brute force?")
    print(f"{'trial':>5s} {'penalty':>8s} {'brute':>8s} {'dp':>8s} {'match':>6s}")
    ok = True
    for t in range(6):
        rng = random.Random(100 + t)
        pen = "convex" if t % 2 == 0 else "cliff"
        n_items, n_recv, budget = 6, 2, 7
        rel, rho, c, _ = make_world(n_items, n_recv, rng, pen)
        bf, _ = brute_force(n_items, n_recv, rel, rho, c, budget)
        _, dp_true, _ = level_dp(n_items, n_recv, rel, rho, c, budget)
        same = abs(bf - dp_true) < 1e-9
        ok &= same
        print(f"{t:5d} {pen:>8s} {bf:8.3f} {dp_true:8.3f} "
              f"{'yes' if same else 'NO':>6s}")
    print(f"   -> decomposition is exact: {ok}\n")

    print("2. APPROXIMATE inner oracle (alpha = 1-1/e): does")
    print("   U(alg) >= alpha*rel(OPT) - rho(OPT) hold?")
    print(f"{'trial':>5s} {'penalty':>8s} {'U(alg)':>8s} {'bound':>8s} {'holds':>6s}")
    held = True
    for t in range(6):
        rng = random.Random(200 + t)
        pen = "convex" if t % 2 == 0 else "cliff"
        n_items, n_recv, budget = 6, 2, 7
        rel, rho, c, _ = make_world(n_items, n_recv, rng, pen)
        _, opt_alloc = brute_force(n_items, n_recv, rel, rho, c, budget)
        bound = (E_INV * sum(rel(i, S) for i, S in enumerate(opt_alloc))
                 - sum(rho(i, c(S)) for i, S in enumerate(opt_alloc)))
        _, alg_true, _ = level_dp(n_items, n_recv, rel, rho, c, budget,
                                  alpha=E_INV)
        h = alg_true >= bound - 1e-9
        held &= h
        print(f"{t:5d} {pen:>8s} {alg_true:8.3f} {bound:8.3f} "
              f"{'yes' if h else 'NO':>6s}")
    print(f"   -> regularized bound holds: {held}")


if __name__ == "__main__":
    main()
