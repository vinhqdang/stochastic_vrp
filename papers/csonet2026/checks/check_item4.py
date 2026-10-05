#!/usr/bin/env python3
"""ITEM 4 checks.
(a) Lemma 'g(i,v) = min total time of a feasible S in {1..i} with scaled value v' + case recursion;
    also shows the literal 'indicator * value' reading is WRONG.
(b) f-recursion with cases (Eq. (1)) vs brute force; literal indicator reading is wrong.
(c) Lemma 1 with ties: S feasible under SOME order of all sites  <=>  EVERY non-decreasing-deadline order of S (S first) is feasible.
(d) union-find with label array (union by size + path compression) vs naive 'largest free slot <= q'.
(e) Theorem 2 reduction incl. odd A and a_i > A/2 (fixed no-instance) vs subset-sum oracle."""
import random, itertools, sys
INF = float('inf')

def feasible_any_order(S, p, d):
    """independent check: exists ANY order of all sites (non-S sites interleaved anywhere) with all of S on time"""
    n = len(p)
    for perm in itertools.permutations(range(n)):
        t = 0; ok = True
        for i in perm:
            t += p[i]
            if i in S and t > d[i]: ok = False; break
        if ok: return True
    return False

def mask_feasible_table(p, d):
    """F[mask]: exists an order of exactly mask on time (DP over last job): independent of EDD lemma"""
    n = len(p); full = 1 << n; tot = [0] * full; F = [False] * full; F[0] = True
    for mask in range(1, full):
        low = (mask & -mask).bit_length() - 1; tot[mask] = tot[mask & (mask - 1)] + p[low]
        for j in range(n):
            if mask >> j & 1 and F[mask ^ (1 << j)] and tot[mask] <= d[j]: F[mask] = True; break
    return F, tot

# ---------------- (a) g lemma ----------------
def g_table_cases(p, d, wp):
    n = len(p); V = sum(wp)
    g = [[INF] * (V + 1) for _ in range(n + 1)]; g[0][0] = 0
    for i in range(1, n + 1):
        for v in range(V + 1):
            take = INF
            if v >= wp[i - 1] and g[i - 1][v - wp[i - 1]] + p[i - 1] <= d[i - 1]:      # cases: +inf if test fails or v < w'_i
                take = g[i - 1][v - wp[i - 1]] + p[i - 1]
            g[i][v] = min(g[i - 1][v], take)
    return g

def g_table_literal(p, d, wp):
    n = len(p); V = sum(wp)
    g = [[INF] * (V + 1) for _ in range(n + 1)]; g[0][0] = 0
    for i in range(1, n + 1):
        for v in range(V + 1):
            vv = v - wp[i - 1]
            prev = g[i - 1][vv] if vv >= 0 else INF
            cond = (prev + p[i - 1] <= d[i - 1])
            g[i][v] = min(g[i - 1][v], (1 if cond else 0) * (prev + p[i - 1]) if prev < INF else 0 if not cond else INF)
    return g

def check_a(rng, N=800):
    bad = 0; lit_bad = 0
    for _ in range(N):
        n = rng.randint(1, 7)
        p = [rng.randint(1, 6) for _ in range(n)]; d = [rng.randint(1, 20) for _ in range(n)]
        order = sorted(range(n), key=lambda i: (d[i], i)); p = [p[i] for i in order]; d = [d[i] for i in order]
        wp = [rng.choice([0, 0, 1, 2, 3, 5]) for _ in range(n)]
        g = g_table_cases(p, d, wp)
        F, tot = mask_feasible_table(p, d)
        V = sum(wp)
        for i in range(0, n + 1):
            for v in range(V + 1):
                best = INF
                for mask in range(1 << i):
                    if F[mask] and sum(wp[j] for j in range(i) if mask >> j & 1) == v: best = min(best, tot[mask])
                if best != g[i][v]: bad += 1
        gl = g_table_literal(p, d, wp)
        if any(gl[n][v] != g[n][v] for v in range(V + 1)): lit_bad += 1
    return N, bad, lit_bad

# ---------------- (b) f recursion ----------------
def f_table_cases(p, d, w):
    n = len(p); P = sum(p); NEG = -INF
    f = [[NEG] * (P + 1) for _ in range(n + 1)]; f[0][0] = 0
    for i in range(1, n + 1):
        for t in range(P + 1):
            if p[i - 1] <= t <= d[i - 1]: f[i][t] = max(f[i - 1][t], f[i - 1][t - p[i - 1]] + w[i - 1])
            else: f[i][t] = f[i - 1][t]
    return f

def f_table_literal(p, d, w):          # Eq. (1) read literally: [t<=d_i] * (f + w), and no -inf clause
    n = len(p); P = sum(p); NEG = -INF
    f = [[NEG] * (P + 1) for _ in range(n + 1)]; f[0][0] = 0
    for i in range(1, n + 1):
        for t in range(P + 1):
            prev = f[i - 1][t - p[i - 1]] if t >= p[i - 1] else NEG
            second = (w[i - 1] + prev) if (t <= d[i - 1] and prev > NEG) else 0   # indicator false -> 0
            f[i][t] = max(f[i - 1][t], second)
    return f

def check_b(rng, N=800):
    bad = lit_bad = 0
    for _ in range(N):
        n = rng.randint(1, 7)
        p = [rng.randint(1, 6) for _ in range(n)]; d = [rng.randint(1, 20) for _ in range(n)]
        order = sorted(range(n), key=lambda i: (d[i], i)); p = [p[i] for i in order]; d = [d[i] for i in order]
        w = [rng.randint(1, 9) for _ in range(n)]
        f = f_table_cases(p, d, w); F, tot = mask_feasible_table(p, d)
        for i in range(n + 1):
            for t in range(sum(p) + 1):
                best = -INF
                for mask in range(1 << i):
                    if F[mask] and tot[mask] == t: best = max(best, sum(w[j] for j in range(i) if mask >> j & 1))
                if best != f[i][t]: bad += 1
        fl = f_table_literal(p, d, w)
        if any(fl[n][t] != f[n][t] for t in range(sum(p) + 1)): lit_bad += 1
    return N, bad, lit_bad

# ---------------- (c) Lemma 1 with ties ----------------
def check_c(rng, N=600):
    bad = 0; subsets = 0; tie_instances = 0
    for _ in range(N):
        n = rng.randint(2, 6)
        p = [rng.randint(1, 4) for _ in range(n)]
        d = [rng.choice([3, 4, 5, 6, 8, 10]) for _ in range(n)]            # many ties
        if len(set(d)) < n: tie_instances += 1
        for mask in range(1, 1 << n):
            S = {i for i in range(n) if mask >> i & 1}; subsets += 1
            anyfeas = feasible_any_order(S, p, d)
            # every non-decreasing-deadline order of S (all tie-breaks), served first
            all_nd = True
            for perm in itertools.permutations(sorted(S)):
                if any(d[perm[k]] > d[perm[k + 1]] for k in range(len(perm) - 1)): continue
                t = 0
                for i in perm:
                    t += p[i]
                    if t > d[i]: all_nd = False
            if anyfeas != all_nd: bad += 1
    return N, subsets, tie_instances, bad

# ---------------- (d) union-find ----------------
class UFLabel:
    def __init__(s, n): s.par = list(range(n + 1)); s.size = [1] * (n + 1); s.label = list(range(n + 1)); s.steps = 0
    def root(s, x):
        r = x
        while s.par[r] != r: r = s.par[r]; s.steps += 1
        while s.par[x] != r: s.par[x], x = r, s.par[x]
        return r
    def find(s, x): return s.label[s.root(x)]
    def union_down(s, q):
        a, b = s.root(q), s.root(q - 1); lab = s.label[b]
        if s.size[a] < s.size[b]: a, b = b, a
        s.par[b] = a; s.size[a] += s.size[b]; s.label[a] = lab

class UFPaperDirection:     # path compression only; always link root(q) under root(q-1)  (no rank)
    def __init__(s, n): s.par = list(range(n + 1)); s.steps = 0
    def find(s, x):
        r = x
        while s.par[r] != r: r = s.par[r]; s.steps += 1
        while s.par[x] != r: s.par[x], x = r, s.par[x]
        return r
    def union_down(s, q): s.par[s.find(q)] = s.find(q - 1)

def check_d(rng, N=1500):
    bad = 0
    for _ in range(N):
        n = rng.randint(1, 60); free = [True] * (n + 1); free[0] = True
        u = UFLabel(n); u2 = UFPaperDirection(n)
        for _ in range(n):
            D = rng.randint(1, n)
            q = D
            while q >= 1 and not free[q]: q -= 1
            if u.find(D) != q or u2.find(D) != q: bad += 1
            if q >= 1: free[q] = False; u.union_down(q); u2.union_down(q)
    # step counts (informal) on a long run
    n = 200000; u = UFLabel(n); u2 = UFPaperDirection(n)
    ds = [rng.randint(1, n) for _ in range(n)]
    for D in ds:
        q = u.find(D)
        q2 = u2.find(D)
        assert q == q2
        if q >= 1: u.union_down(q); u2.union_down(q)
    return N, bad, u.steps / n, u2.steps / n

# ---------------- (e) Theorem 2 reduction ----------------
def reduce_partition(a):
    A = sum(a)
    if A % 2 == 1 or max(a) > A // 2: return ([1], [1], [1], 2)      # fixed no-instance, threshold k=2
    return (list(a), [A // 2] * len(a), list(a), A // 2)

def check_e(rng, N=700):
    bad = 0; odd = big = yes = 0
    for _ in range(N):
        n = rng.randint(1, 9); a = [rng.randint(1, rng.choice([3, 8, 25])) for _ in range(n)]
        if rng.random() < .3: a.append(sum(a) + rng.randint(0, 3))             # forces a_i > A/2 sometimes
        A = sum(a); odd += (A % 2 == 1); big += (max(a) > A / 2)
        reach = {0}
        for x in a: reach |= {r + x for r in reach}
        truth = (A % 2 == 0 and A // 2 in reach); yes += truth
        p, d, w, k = reduce_partition(a)
        assert all(pi <= di for pi, di in zip(p, d))
        F, tot = mask_feasible_table(p, d)
        Wst = max(sum(w[j] for j in range(len(p)) if m >> j & 1) for m in range(1 << len(p)) if F[m])
        if (Wst >= k) != truth: bad += 1
    return N, odd, big, yes, bad

def main():
    rng = random.Random(99)
    N, bad, lit = check_a(rng); print(f"(a) g-lemma+case recursion: {N} instances, all (i,v) cells vs brute force: mismatches {bad}; literal indicator reading wrong on {lit}/{N} instances")
    N, bad2, lit2 = check_b(rng); print(f"(b) f-recursion with cases: {N} instances, all (i,t) cells vs brute force: mismatches {bad2}; literal indicator reading (no -inf clause) wrong on {lit2}/{N}")
    N, S, T, bad3 = check_c(rng); print(f"(c) Lemma 1 with ties: {N} instances ({T} with tied deadlines), {S} subsets; mismatches {bad3}")
    N, bad4, s1, s2 = check_d(rng); print(f"(d) union-find (label array, size+compression) and paper-direction (compression only) vs naive: {N} runs, mismatches {bad4}; avg pointer steps/op for n=2e5 random: label+size {s1:.2f}, compression-only {s2:.2f}")
    N, odd, big, yes, bad5 = check_e(rng); print(f"(e) Theorem 2 reduction: {N} Partition instances ({odd} odd A, {big} with a_i>A/2, {yes} YES); mismatches {bad5}")
    return bad + bad2 + bad3 + bad4 + bad5

if __name__ == '__main__':
    sys.exit(1 if main() else 0)
