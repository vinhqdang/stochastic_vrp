"""H2 part 1: validation of the spoke-vs-chain gap theorems (seeded, exhaustive).

Claims validated
  T1  W_spoke <= W_chain <= (#individually-feasible sites) * w_max <= n * W_spoke
      and three independent chain solvers agree.
  T2  star metric (delta(i,j)=a_i+a_j): W_chain == W_spoke for all h, w.
  T3  converse: if some pair has delta(i,j) < a_i+a_j, there are h, w with
      W_chain > W_spoke (explicit two-site instance).
  T4  lemma  A_j <= B_j/kappa  (j>=2)  and speed-augmentation theorem
      W_chain(speed 1) <= W_spoke(speed 1/kappa)   [kappa = min_{i!=j} delta/(a_i+a_j)]
  T5  ratio bound  W_chain <= ceil((3 rho - 1)/(2 kappa)) * W_spoke
      and dyadic bound (floor(log2 rho)+1)*ceil(5/(2 kappa)); every J_r class feasible.
  T6  extremal families: cluster (ratio n), geometric ray (ratio n, kappa->1),
      uniform metric (ratio -> 1/kappa), exact brute force.
"""
import itertools
import math
import random
import sys
from h2_common import *

EPS = 1e-9


def rand_instance(rng, n, kind):
    """returns D (n+1 x n+1), plus optional hazard-origin distances."""
    if kind == "euclid":
        pts = [(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(n + 2)]
        D = euclid_metric(pts)
        return [r[:n + 1] for r in D[:n + 1]], [D[n + 1][i] for i in range(n + 1)]
    if kind == "graph":
        D = graph_metric(n + 2, rng)
        return [r[:n + 1] for r in D[:n + 1]], [D[n + 1][i] for i in range(n + 1)]
    if kind == "line":
        pos = [0.0] + [rng.randint(-12, 12) + 0.5 * rng.random() for _ in range(n)]
        pos = [p if abs(p) > 1e-6 else 1.0 for p in pos]
        pos[0] = 0.0
        Hp = rng.uniform(-14, 14)
        D = line_metric(pos)
        return D, [abs(Hp - p) for p in pos]
    if kind == "star":
        a = [0.0] + [rng.randint(1, 9) for _ in range(n)]
        return star_metric(a), [rng.randint(1, 20) for _ in range(n + 1)]
    raise ValueError(kind)


def hazard(rng, D, hd, n, mode):
    if mode == "uniform":
        top = 3 * max(D[0][i] for i in range(1, n + 1)) + 3
        return [0] + [rng.uniform(0, top) for _ in range(n)]
    if mode == "front":                      # h_i = dist(H,i)/v
        v = rng.choice([0.5, 1.0, 1.5, 2.0])
        return [0] + [hd[i] / v for i in range(1, n + 1)]
    if mode == "tight":                      # h_i = chain arrival along a random order
        order = list(range(1, n + 1)); rng.shuffle(order)
        h = [0.0] * (n + 1)
        t = 0.0; cur = 0
        for i in order:
            t += D[cur][i]; cur = i
            h[i] = t * rng.choice([1.0, 1.0, 1.0, 1.05])
        k = rng.randint(1, n)
        for i in order[k:]:
            h[i] = 0.0                       # unreachable tail
        return h
    raise ValueError(mode)


def T1_T2_T4_T5(seed=1, trials=1500):
    rng = random.Random(seed)
    cnt = dict(T1=0, solvers_agree=0, star_eq=0, T4=0, T4_lemma=0, T5=0, T5_dy=0,
               T5_classes=0, strict_gap=0, ratio_max=0.0)
    worst5 = 0.0
    for tr in range(trials):
        n = rng.randint(2, 6)
        kind = rng.choice(["euclid", "graph", "line", "star"])
        D, hd = rand_instance(rng, n, kind)
        h = hazard(rng, D, hd, n, rng.choice(["uniform", "front", "tight"]))
        w = rand_weights(n, rng, equal=rng.random() < 0.4)
        assert is_metric(D), kind
        ws = spoke_opt(D, h, w)
        wc = chain_opt(D, h, w)
        wc2 = chain_opt_ordered_subsets(D, h, w)
        wc3 = chain_pareto_dp(D, h, w)
        assert wc == wc2 == wc3, (wc, wc2, wc3)
        cnt["solvers_agree"] += 1
        # T1
        feas = [i for i in range(1, n + 1) if D[0][i] <= h[i] + EPS]
        wmax = max((w[i] for i in feas), default=0)
        assert ws <= wc, (ws, wc)
        assert wc <= len(feas) * wmax + 0, (wc, feas, wmax)
        if wc > 0:
            assert ws >= wmax and wc <= len(feas) * ws
        cnt["T1"] += 1
        if wc > ws:
            cnt["strict_gap"] += 1
        if ws > 0:
            cnt["ratio_max"] = max(cnt["ratio_max"], wc / ws)
        if kind == "star":
            assert wc == ws, ("star", ws, wc)
            cnt["star_eq"] += 1
        # T4 : kappa, augmentation, lemma
        kap, rho = kappa_rho(D)
        ws_fast = spoke_opt(D, h, w, speed=1.0 / kap)
        assert wc <= ws_fast, ("augmentation", wc, ws_fast, kap)
        cnt["T4"] += 1
        # lemma on a random order with all sites
        order = list(range(1, n + 1)); rng.shuffle(order)
        B = 0.0; cur = 0; Xt = 0.0
        for j, i in enumerate(order):
            B += D[cur][i]; cur = i
            A = 2 * Xt + D[0][i]
            if j >= 1:
                assert A <= B / kap + 1e-7, ("lemma", A, B, kap)
            else:
                assert abs(A - B) < 1e-9
            Xt += D[0][i]
        cnt["T4_lemma"] += 1
        # T5
        if ws > 0:
            b1 = math.ceil((3 * rho - 1) / (2 * kap) - 1e-12)
            b2 = (math.floor(math.log2(rho) + 1e-12) + 1) * math.ceil(5 / (2 * kap) - 1e-12)
            assert wc <= b1 * ws, ("T5", wc, ws, b1)
            assert wc <= b2 * ws, ("T5dy", wc, ws, b2)
            cnt["T5"] += 1; cnt["T5_dy"] += 1
            worst5 = max(worst5, wc / (b1 * ws))
        # direct check of the class construction on the chain-optimal order
        # (tight hazards h = B along an arbitrary order, all sites listed)
    cnt["T5_ratio_to_bound_max"] = worst5
    return cnt


def T5_classes(seed=2, trials=4000):
    """The proof's construction: take ANY order, set h_i = B_i (tightest hazards
    that keep the order chain-feasible); for m=ceil((3 rho-1)/(2 kappa)) every
    residue class J_r must be spoke-feasible (same induced order)."""
    rng = random.Random(seed)
    ok = 0
    for tr in range(trials):
        n = rng.randint(2, 12)
        kind = rng.choice(["euclid", "graph", "line"])
        D, _ = rand_instance(rng, n, kind)
        order = list(range(1, n + 1)); rng.shuffle(order)
        kap, rho = kappa_rho(D)
        m = math.ceil((3 * rho - 1) / (2 * kap) - 1e-12)
        B = []; t = 0.0; cur = 0
        for i in order:
            t += D[cur][i]; cur = i; B.append(t)
        for r in range(m):
            cls = order[r::m]
            Bcls = B[r::m]
            tt = 0.0
            for idx, i in enumerate(cls):
                assert tt + D[0][i] <= Bcls[idx] + 1e-9, ("class infeasible", tr, r, idx)
                tt += 2 * D[0][i]
        ok += 1
    return ok


def T3_converse(seed=3, trials=800):
    rng = random.Random(seed)
    n_nonstar = 0
    for tr in range(trials):
        n = rng.randint(2, 6)
        kind = rng.choice(["euclid", "graph", "line"])
        D, _ = rand_instance(rng, n, kind)
        pair = None
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if D[i][j] < D[0][i] + D[0][j] - 1e-9:
                    pair = (i, j); break
            if pair: break
        if not pair:
            continue
        i, j = pair
        # two-site instance on {0,i,j}
        D2 = [[D[x][y] for y in (0, i, j)] for x in (0, i, j)]
        h = [0, D2[0][1], D2[0][1] + D2[1][2]]
        w = [0, 1, 1]
        assert spoke_opt(D2, h, w) == 1 and chain_opt(D2, h, w) == 2
        n_nonstar += 1
    return n_nonstar


# ------------------------------------------------------------ extremal families
def fam_cluster(n, R=10.0, eps=0.01):
    # n sites at distance R, mutual distance eps (uniform metric): h_j = R+(j-1) eps
    D = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[0][i] = D[i][0] = R
        for j in range(1, n + 1):
            if i != j:
                D[i][j] = eps
    h = [0] + [R + (j - 1) * eps for j in range(1, n + 1)]
    return D, h


def fam_ray(n, r):
    pos = [0.0] + [float(r) ** j for j in range(1, n + 1)]
    D = line_metric(pos)
    h = [0.0] + pos[1:]
    return D, h


def fam_uniform(n, kappa):
    D = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[0][i] = D[i][0] = 1.0
        for j in range(1, n + 1):
            if i != j:
                D[i][j] = 2 * kappa
    h = [0] + [1.0 + (j - 1) * 2 * kappa for j in range(1, n + 1)]
    return D, h


def T6():
    out = []
    # cluster: ratio exactly n
    for n in range(2, 8):
        D, h = fam_cluster(n)
        w = [0] + [1] * n
        ws = spoke_opt(D, h, w); wc = chain_opt(D, h, w)
        kap, rho = kappa_rho(D)
        out.append(("cluster", n, ws, wc, round(kap, 4), round(rho, 3)))
        assert (ws, wc) == (1, n)
    # geometric ray: ratio n for any r>1 ; kappa=(r-1)/(r+1)
    for r in (1.5, 2, 3, 10):
        for n in range(2, 8):
            D, h = fam_ray(n, r)
            w = [0] + [1] * n
            ws = spoke_opt(D, h, w); wc = chain_opt(D, h, w)
            kap, rho = kappa_rho(D)
            assert abs(kap - (r - 1) / (r + 1)) < 1e-9
            assert (ws, wc) == (1, n), (r, n, ws, wc)
            out.append(("ray", r, n, ws, wc, round(kap, 4), round(rho, 1)))
    # uniform metric: ratio n / (floor((n-1) kappa)+1), kappa = 1/q
    for q in (2, 3):
        kappa = 1.0 / q
        for n in range(2, 8):
            D, h = fam_uniform(n, kappa)
            w = [0] + [1] * n
            ws = spoke_opt(D, h, w); wc = chain_opt(D, h, w)
            pred = math.floor((n - 1) * kappa + 1e-9) + 1
            assert wc == n and ws == pred, (q, n, ws, pred)
            out.append(("uniform", q, n, ws, wc))
    return out


def adversarial_search(seed=5, iters=3000):
    """Hill-climb on small Euclidean/graph instances to find the largest
    W_chain/W_spoke observed relative to the proved bound min(n, ceil((3rho-1)/2kappa))."""
    rng = random.Random(seed)
    best = (0, None)
    for it in range(iters):
        n = rng.randint(3, 6)
        kind = rng.choice(["euclid", "graph", "line"])
        D, _ = rand_instance(rng, n, kind)
        h = hazard(rng, D, None, n, "tight") if False else None
        order = list(range(1, n + 1)); rng.shuffle(order)
        h = [0.0] * (n + 1); t = 0.0; cur = 0
        for i in order:
            t += D[cur][i]; cur = i; h[i] = t
        w = [0] + [1] * n
        ws = spoke_opt(D, h, w); wc = chain_opt(D, h, w)
        if ws > 0 and wc / ws > best[0]:
            kap, rho = kappa_rho(D)
            best = (wc / ws, (kind, n, ws, wc, round(kap, 3), round(rho, 2),
                              math.ceil((3 * rho - 1) / (2 * kap))))
    return best


if __name__ == "__main__":
    c = T1_T2_T4_T5(seed=11, trials=1500)
    print("T1/T2/T4/T5 random campaign:", c)
    print("T5 class-construction checks passed:", T5_classes())
    print("T3 converse (non-star pairs, explicit 2-site gap):", T3_converse())
    for row in T6():
        print("T6", row)
    print("adversarial search best ratio:", adversarial_search())
