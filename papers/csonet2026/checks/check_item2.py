#!/usr/bin/env python3
"""ITEM 2: constant hazard-arrival time H, arrival reading d_i = H + p_i/2, p_i even.
Reduction from PARTITION (a_1..a_n, A=sum even):
   site 0 (dominant): p_0 = 2A+2, w_0 = A+1
   site i (1..n):     p_i = 2 a_i, w_i = a_i
   H = 2A+1 (same for all);  d_i = H + p_i/2   (so w_i = p_i/2 for ALL i).
Claim: W* >= A+1+A/2  <=>  PARTITION yes.
Checks: (1) reduction decision vs subset-sum oracle on random instances (YES and NO);
        (2) W* computed by exhaustive mask DP over ALL orderings (independent of EDD lemma);
        (3) closed-form feasibility criterion  sum_S p - max_S p/2 <= H  vs exhaustive."""
import random, sys, itertools

def exhaustive_Wstar(p, d, w):
    n = len(p); full = 1 << n
    # F[mask]: exists order serving exactly 'mask' first, all on time (last-job DP over all permutations)
    tot = [0] * full; wt = [0] * full
    for mask in range(1, full):
        low = (mask & -mask).bit_length() - 1
        tot[mask] = tot[mask & (mask - 1)] + p[low]; wt[mask] = wt[mask & (mask - 1)] + w[low]
    F = [False] * full; F[0] = True
    for mask in range(1, full):
        for j in range(n):
            if mask >> j & 1 and F[mask ^ (1 << j)] and tot[mask] <= d[j]:
                F[mask] = True; break
    best = max(wt[m] for m in range(full) if F[m])
    return best, F, tot

def partition_oracle(a):
    A = sum(a)
    if A % 2: return False
    reach = {0}
    for x in a: reach |= {r + x for r in reach}
    return A // 2 in reach

def build(a):
    A = sum(a)
    H = 2 * A + 1
    p = [2 * A + 2] + [2 * x for x in a]
    w = [A + 1] + list(a)
    d = [H + pi // 2 for pi in p]
    return p, d, w, H, A

def main():
    rng = random.Random(7)
    yes = no = bad = 0; crit_bad = 0; crit_checked = 0
    N = 600
    for it in range(N):
        n = rng.randint(2, 9)
        mode = it % 3
        if mode == 0:       # planted YES
            a = [rng.randint(1, 12) for _ in range(n)]
            half = sum(a[: n // 2 or 1]); rest = sum(a) - half
            a.append(abs(half - rest) or 1) if half != rest else None
            if sum(a) % 2: a.append(1)       # may break; the oracle decides truth anyway
        elif mode == 1:     # random even-sum
            a = [rng.randint(1, 20) for _ in range(n)]
            if sum(a) % 2: a[0] += 1
        else:               # large element(s) / near-miss NO instances
            a = [rng.randint(1, 6) for _ in range(n)] + [rng.randint(15, 40)]
            if sum(a) % 2: a[-1] += 1
        a = a[:10]
        if sum(a) % 2: a[0] += 1
        truth = partition_oracle(a)
        p, d, w, H, A = build(a)
        assert all(pi % 2 == 0 for pi in p) and all(pi <= di for pi, di in zip(p, d))
        assert len(set(d[i] - p[i] // 2 for i in range(len(p)))) == 1       # constant hazard time
        Wst, F, tot = exhaustive_Wstar(p, d, w)
        decision = Wst >= A + 1 + A // 2
        if decision != truth: bad += 1; print("MISMATCH", a, truth, Wst)
        yes += truth; no += (not truth)
        # (3) closed-form criterion for feasibility of each subset
        nn = len(p)
        for mask in range(1, 1 << nn):
            S = [i for i in range(nn) if mask >> i & 1]
            crit = sum(p[i] for i in S) - max(p[i] for i in S) / 2 <= H
            crit_checked += 1
            if crit != F[mask]: crit_bad += 1
    print(f"PARTITION instances: {N} (YES {yes}, NO {no}); reduction mismatches: {bad}")
    print(f"closed-form feasibility criterion checked on {crit_checked} subsets; mismatches: {crit_bad}")
    # constant weights are polynomial: sanity that Moore-Hodgson equals exhaustive on this very d-structure
    mism = 0; T = 300
    for it in range(T):
        n = rng.randint(2, 9); H = rng.randint(3, 30)
        p = [2 * rng.randint(1, 8) for _ in range(n)]; d = [H + x // 2 for x in p]
        keep = [i for i in range(n) if p[i] <= d[i]]
        p = [p[i] for i in keep]; d = [d[i] for i in keep]
        if not p: continue
        w = [1] * len(p)
        ex, _, _ = exhaustive_Wstar(p, d, w)
        # Moore-Hodgson
        import heapq
        h = []; t = 0
        for i in sorted(range(len(p)), key=lambda i: d[i]):
            heapq.heappush(h, -p[i]); t += p[i]
            if t > d[i]: t += heapq.heappop(h)
        if len(h) != ex: mism += 1
    print(f"constant-weight (unit w) sanity: Moore-Hodgson vs exhaustive on {T} instances with d_i=H+p_i/2: mismatches {mism}")
    return bad + crit_bad + mism

if __name__ == '__main__':
    sys.exit(1 if main() else 0)
