"""Theorem A (rho in the input / narrow band, w = p): verification.

Reduction from SUBSET-SUM-AT-MOST-k:  x_1..x_n >0, target t, cardinality bound k.
Build:  K = 2k*xmax + t + 1,  rho = k + 1/2 (mu = k - 1/2),
  real sites  p = K + x_i,  k dummy sites p = K,  dominant site p = H = (kK+t)/mu,
  w = p, d = rho p.
Claim: W* >= rho*H   iff   some T with |T|<=k has x(T) = t.   (Also W* <= rho*H always.)
For ordinary SUBSET SUM / PARTITION take k = n (and t = A/2).
"""
import random, itertools
from fractions import Fraction as F
from h1_lib import *

random.seed(77)


def build(x, t, k):
    xmax = max(x)
    K = 2 * k * xmax + t + 1
    mu = F(2 * k - 1, 2)
    rho = mu + 1
    H = (k * K + t) / mu
    p = [F(K + xi) for xi in x] + [F(K)] * k + [H]
    return p, rho, H, K


def subset_sum_at_most_k(x, t, k):
    n = len(x)
    for r in range(0, min(k, n) + 1):
        for T in itertools.combinations(range(n), r):
            if sum(x[i] for i in T) == t:
                return True
    return False


tot = yes = no = bad = 0
maxitems = 0
for trial in range(900):
    n = random.randint(1, 8)
    k = random.randint(1, min(n, 6)) if random.random() < 0.6 else n
    k = min(k, 7)
    xr = random.choice([5, 12, 40, 1000])
    x = [random.randint(1, xr) for _ in range(n)]
    if random.random() < 0.5:
        r = random.randint(0, min(k, n))
        t = sum(random.sample(x, r)) if r else 0
        t = max(t, 1)
    else:
        t = random.randint(1, max(1, sum(x)))
    p, rho, H, K = build(x, t, k)
    maxitems = max(maxitems, len(p))
    opt = brute_opt(p, p, rho)         # w = p
    truth = subset_sum_at_most_k(x, t, k)
    claim = (opt >= rho * H)
    tot += 1
    yes += truth; no += (not truth)
    bad += (claim != truth)
    bad += (opt > rho * H)             # upper bound W* <= rho*H
print(f"Theorem A: {tot} random instances (yes={yes}, no={no}), max #sites={maxitems}, violations={bad}")

# Partition-style (k = n, t = A/2) with the same builder
tot = yes = bad = 0
for trial in range(300):
    n = random.randint(2, 7)
    a = [random.randint(1, 30) for _ in range(n)]
    A = sum(a)
    if A % 2:
        continue
    p, rho, H, K = build(a, A // 2, n)
    opt = brute_opt(p, p, rho)
    truth = subset_sum_at_most_k(a, A // 2, n)
    tot += 1; yes += truth
    bad += ((opt >= rho * H) != truth)
print(f"Theorem A (Partition, k=n): {tot} instances (yes={yes}), violations={bad}")

# narrowness of the band: ratio pmax/pmin of the non-dominant sites
x = [random.randint(1, 50) for _ in range(6)]
p, rho, H, K = build(x, sum(x[:3]), 6)
print("example: band ratio (K+xmax)/K =", float(F(K + max(x), K)), " rho =", rho, " H/K =", float(H / K))
