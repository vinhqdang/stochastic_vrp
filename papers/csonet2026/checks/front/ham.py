"""Theorem 24 (strong NP-hardness at Gamma=1): independent check of the reduction from
Hamiltonian path.  Sites at distance R=(n-1)s/(lam-1) from the depot, pairwise s on edges
and 2s on non-edges, h_i=lam*R.  Claim: W_chain = n  <=>  G has a Hamiltonian path."""
import itertools, random, sys
from indep import chain_dp
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
cnt = bad = yes = 0
for it in range(300):
    n = rng.randint(4, 8)
    lam = rng.choice([1.25, 1.5, 2.0, 3.0, 3.5]); 
    if n < lam: continue
    p = rng.choice([0.25, 0.4, 0.6]); s = 1.0
    E = {(i, j): rng.random() < p for i in range(n) for j in range(i + 1, n)}
    adj = lambda i, j: E[(min(i, j), max(i, j))]
    R = (n - 1) * s / (lam - 1)
    N = n + 1; d = [[0.0] * N for _ in range(N)]
    for i in range(1, N):
        d[0][i] = d[i][0] = R
        for j in range(i + 1, N):
            d[i][j] = d[j][i] = s if adj(i - 1, j - 1) else 2 * s
    for i in range(N):                        # triangle inequality
        for j in range(N):
            for k in range(N): assert d[i][j] <= d[i][k] + d[k][j] + 1e-9
    h = {i: lam * R for i in range(N)}; w = [0] + [1] * n
    allb = chain_dp(d, w, h, list(range(1, n + 1)))
    opt = max([bin(S).count("1") for (S, l) in allb] + [0])
    ham = any(all(adj(a, b) for a, b in zip(pm, pm[1:])) for pm in itertools.permutations(range(n)))
    cnt += 1; yes += ham
    if (opt == n) != ham: bad += 1; print("MISMATCH", n, lam, opt, ham)
print("instances", cnt, "(Hamiltonian-path yes: %d)" % yes, "violations", bad)
