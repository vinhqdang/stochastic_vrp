#!/usr/bin/env python3
"""ITEM 1: equal-size sites, vehicle-dependent dispatch times p_v.
Three-way check: (A) heap-slots + union-find greedy, (B) counting/Hall criterion
(max weight over all subsets satisfying |{i in S: d_i<=t}| <= C(t) for all t),
(C) exhaustive search over ALL orderings/assignments (mask DP over every
permutation on every vehicle; plus explicit permutation enumeration for n<=5).
Also: matroid exchange axiom checked on all pairs of feasible sets."""
import random, heapq, itertools, sys, bisect

def C_of(t, ps):
    return sum(t // p for p in ps)

# ---------- (A) near-linear algorithm ----------
class UF:
    """union-find over {0..n}; blocks = maximal runs; label[root] = smallest member
    (= largest free slot <= q).  Union by size + path compression => O(alpha(n))."""
    def __init__(s, n):
        s.par = list(range(n + 1)); s.size = [1] * (n + 1); s.label = list(range(n + 1))
    def root(s, x):
        r = x
        while s.par[r] != r: r = s.par[r]
        while s.par[x] != r: s.par[x], x = r, s.par[x]
        return r
    def find(s, x): return s.label[s.root(x)]
    def union_down(s, q):            # merge block of q with block of q-1; keep label of q-1's block
        a, b = s.root(q), s.root(q - 1)
        lab = s.label[b]
        if a == b: return
        if s.size[a] < s.size[b]: a, b = b, a
        s.par[b] = a; s.size[a] += s.size[b]; s.label[a] = lab

def slot_algorithm(ps, ds, ws):
    """returns (value, kept set, assignment {site:(vehicle,position)})"""
    n = len(ds); m = len(ps)
    order_v = sorted(range(m), key=lambda v: ps[v])[:min(m, n)]
    heap = [(ps[v], v, 1) for v in order_v]; heapq.heapify(heap)
    slots = []   # (time, vehicle, position)
    for _ in range(n):
        t, v, j = heapq.heappop(heap)
        slots.append((t, v, j))
        heapq.heappush(heap, ((j + 1) * ps[v], v, j + 1))
    times = [s[0] for s in slots]
    D = [bisect.bisect_right(times, d) for d in ds]          # D_i = #slots (among n earliest) with time <= d_i
    uf = UF(n); kept = []; assign = {}
    for i in sorted(range(n), key=lambda i: -ws[i]):          # stable: ties by index
        if D[i] == 0: continue
        q = uf.find(D[i])
        if q >= 1:
            kept.append(i); assign[i] = slots[q - 1][1:]
            uf.union_down(q)
    return sum(ws[i] for i in kept), kept, assign

def schedule_value(ps, ds, ws, assign):
    """compress to prefix-filled vehicle schedules, EDD per vehicle, recompute on-time weight"""
    per = {}
    for i, (v, j) in assign.items(): per.setdefault(v, []).append((j, i))
    tot = 0
    for v, lst in per.items():
        lst.sort()
        for r, (j, i) in enumerate(lst, 1):
            assert r <= j
            if r * ps[v] <= ds[i]: tot += ws[i]
            else: return None
    return tot

# ---------- (B) Hall / counting criterion ----------
def hall_feasible(S, ps, ds):
    for t in set(ds[i] for i in S):
        if sum(1 for i in S if ds[i] <= t) > C_of(t, ps): return False
    return True

def hall_opt(ps, ds, ws):
    n = len(ds); best = 0
    feas = {}
    for mask in range(1 << n):
        S = [i for i in range(n) if mask >> i & 1]
        if hall_feasible(S, ps, ds):
            feas[mask] = True
            best = max(best, sum(ws[i] for i in S))
    return best, feas

# ---------- (C) exhaustive ----------
def exhaustive_opt(ps, ds, ws):
    n = len(ds); m = len(ps); full = 1 << n
    wm = [0] * full
    for mask in range(1, full):
        low = (mask & -mask).bit_length() - 1
        wm[mask] = wm[mask & (mask - 1)] + ws[low]
    R = [0] + [0] * (full - 1)  # R_0: only empty set usable -> value 0 for any mask (nothing served)
    for v in range(m):
        # F[mask]: can mask be served in SOME order on vehicle v (all permutations, via last-job DP)
        F = [False] * full; F[0] = True
        for mask in range(1, full):
            k = bin(mask).count('1'); comp = k * ps[v]
            for j in range(n):
                if mask >> j & 1 and F[mask ^ (1 << j)] and comp <= ds[j]:
                    F[mask] = True; break
        newR = [0] * full
        for mask in range(full):
            best = R[mask]
            sub = mask
            while True:
                if F[sub]:
                    val = wm[sub] + R[mask ^ sub]
                    if val > best: best = val
                if sub == 0: break
                sub = (sub - 1) & mask
            newR[mask] = best
        R = newR
    return R[full - 1]

def explicit_opt(ps, ds, ws):
    """n<=5: enumerate every vehicle-assignment x every global order; sites on a vehicle served in that order."""
    n = len(ds); m = len(ps); best = 0
    for assign in itertools.product(range(m + 1), repeat=n):   # m = unserved
        for perm in itertools.permutations(range(n)):
            t = [0] * m; tot = 0
            for i in perm:
                v = assign[i]
                if v == m: continue
                t[v] += ps[v]
                if t[v] <= ds[i]: tot += ws[i]
                # late sites still occupy the vehicle in this enumeration (harmless: dominated)
            best = max(best, tot)
    return best

def matroid_check(feas_masks, n):
    fs = list(feas_masks)
    pc = lambda x: bin(x).count('1')
    for A in fs:
        for B in fs:
            if pc(A) < pc(B):
                diff = B & ~A
                if not any((A | (1 << x)) in feas_masks for x in range(n) if diff >> x & 1):
                    return False
    # hereditary
    for A in fs:
        x = A
        while x:
            if (A & ~(x & -x)) not in feas_masks: return False
            x &= x - 1
    return True


# ---------- extension: vehicle availability offsets a_v (slot times a_v + j p_v) ----------
def offsets_test(rng, N=400):
    bad = 0
    for _ in range(N):
        n = rng.randint(1, 7); m = rng.choice([1, 2, 3])
        ps = [rng.randint(1, 6) for _ in range(m)]; av = [rng.randint(0, 6) for _ in range(m)]
        ds = [rng.randint(1, 25) for _ in range(n)]; ws = [rng.randint(1, 20) for _ in range(n)]
        # slot algorithm: n earliest slots a_v + j p_v
        heap = [(av[v] + ps[v], v, 1) for v in range(m)]; heapq.heapify(heap); slots = []
        for _k in range(n):
            t, v, j = heapq.heappop(heap); slots.append(t); heapq.heappush(heap, (av[v] + (j + 1) * ps[v], v, j + 1))
        D = [bisect.bisect_right(slots, d) for d in ds]; uf = UF(n); val = 0
        for i in sorted(range(n), key=lambda i: -ws[i]):
            if D[i] and uf.find(D[i]) >= 1: val += ws[i]; uf.union_down(uf.find(D[i]))
        # exhaustive (all orders, per-vehicle last-job DP), vehicles start at a_v
        full = 1 << n; wm = [0] * full
        for mask in range(1, full):
            low = (mask & -mask).bit_length() - 1; wm[mask] = wm[mask & (mask - 1)] + ws[low]
        R = [0] * full
        for v in range(m):
            F = [False] * full; F[0] = True
            for mask in range(1, full):
                comp = av[v] + bin(mask).count('1') * ps[v]
                for j in range(n):
                    if mask >> j & 1 and F[mask ^ (1 << j)] and comp <= ds[j]: F[mask] = True; break
            newR = [0] * full
            for mask in range(full):
                best = R[mask]; sub = mask
                while True:
                    if F[sub]: best = max(best, wm[sub] + R[mask ^ sub])
                    if sub == 0: break
                    sub = (sub - 1) & mask
                newR[mask] = best
            R = newR
        if R[full - 1] != val: bad += 1
    return N, bad

def main():
    rng = random.Random(20261005)
    N = 1600; mism = 0; stats = {'inst': 0, 'matroid_ok': 0, 'matroid_tested': 0, 'explicit': 0}
    by_m = {1: 0, 2: 0, 3: 0}
    equal_p = 0
    for it in range(N):
        n = rng.randint(1, 8); m = rng.choice([1, 2, 3]); by_m[m] += 1
        if it % 7 == 0:                       # equal-p special case (Theorem 5)
            p = rng.randint(1, 5); ps = [p] * m; equal_p += 1
        else:
            ps = [rng.randint(1, 7) for _ in range(m)]
        dmax = max(2, int(n * max(ps) / m * rng.choice([0.6, 1, 1.5])) + 2)
        ds = [rng.randint(1, dmax) for _ in range(n)]
        ws = [rng.randint(1, rng.choice([3, 10, 100])) for _ in range(n)]
        a, kept, assign = slot_algorithm(ps, ds, ws)
        sv = schedule_value(ps, ds, ws, assign)
        h, feas = hall_opt(ps, ds, ws)
        e = exhaustive_opt(ps, ds, ws)
        ok = (a == h == e) and (sv == a)
        if n <= 5 and it % 5 == 0:
            ex = explicit_opt(ps, ds, ws); stats['explicit'] += 1
            ok = ok and ex == e
        if n <= 6:
            stats['matroid_tested'] += 1
            if matroid_check(feas, n): stats['matroid_ok'] += 1
            else: ok = False
        # also: Hall-feasible sets == exhaustively feasible sets?  (set-level check for n<=6)
        if not ok:
            mism += 1; print("MISMATCH", ps, ds, ws, a, h, e, sv)
        stats['inst'] += 1
    # larger random tests: union-find greedy vs naive greedy with Hall test (O(n^2)), m possibly > n
    big = 0; bigmism = 0
    for it in range(400):
        n = rng.randint(20, 150); m = rng.randint(1, 60)
        ps = [rng.randint(1, 30) for _ in range(m)]
        dmax = rng.randint(5, 400)
        ds = [rng.randint(1, dmax) for _ in range(n)]
        ws = [rng.randint(1, 1000) for _ in range(n)]
        a, kept, assign = slot_algorithm(ps, ds, ws)
        S = []
        for i in sorted(range(n), key=lambda i: -ws[i]):
            if hall_feasible(S + [i], ps, ds): S.append(i)
        b = sum(ws[i] for i in S)
        big += 1
        if a != b or schedule_value(ps, ds, ws, assign) != a:
            bigmism += 1; print("BIG MISMATCH", n, m)
    print(f"small instances: {stats['inst']} (m=1:{by_m[1]}, m=2:{by_m[2]}, m=3:{by_m[3]}; equal-p: {equal_p}); mismatches: {mism}")
    print(f"  explicit permutation/assignment enumeration (n<=5) cross-checked on {stats['explicit']} instances")
    print(f"  matroid axioms (hereditary + augmentation, all feasible-set pairs) verified on {stats['matroid_ok']}/{stats['matroid_tested']} instances (n<=6)")
    print(f"large instances (n in 20..150, m in 1..60): {big}; mismatches vs naive Hall-greedy: {bigmism}")
    N2, b2 = offsets_test(rng); print(f"extension (vehicle start offsets a_v): {N2} instances, mismatches {b2}")
    return b2 + mism + bigmism + (stats['matroid_tested'] - stats['matroid_ok'])

if __name__ == '__main__':
    sys.exit(1 if main() else 0)
