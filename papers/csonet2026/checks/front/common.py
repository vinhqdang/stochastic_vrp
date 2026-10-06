"""Common tools for the chained-front problem (depot 0, sites 1..n, h_i = beta + lam*a_i).
Exact solvers (Held-Karp over subsets), instance generators, route evaluation."""
import numpy as np, itertools, math, random

# ---------------------------------------------------------------- metrics
def euclid(P):
    P = np.asarray(P, float)
    return np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))

def graph_metric(W):
    """Floyd-Warshall on symmetric weight matrix (inf = no edge)."""
    D = np.array(W, float)
    n = len(D)
    for k in range(n):
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D

def random_euclid(n, rng, dim=2):
    return euclid(rng.random((n + 1, dim)))

def clustered(n, rng, k=3, spread=0.05, dim=2):
    C = rng.random((k, dim)) * 3
    P = [rng.random(dim)] + [C[rng.integers(k)] + rng.normal(0, spread, dim) for _ in range(n)]
    return euclid(P)

def line(n, rng, geometric=False):
    if geometric:
        x = np.sort(np.exp(rng.random(n) * 3))
    else:
        x = np.sort(rng.random(n) * 3 + 0.05)
    P = np.concatenate([[0.0], x])[:, None]
    return euclid(P)

def two_sided_line(n, rng):
    x = rng.random(n + 1) * 4 - 2
    x[0] = 0
    return euclid(x[:, None])

def ray_like(n, rng, eps=0.1):
    x = np.sort(rng.random(n)) * 3 + 0.1
    P = np.zeros((n + 1, 2)); P[1:, 0] = x; P[1:, 1] = rng.normal(0, eps, n)
    return euclid(P)

def random_tree(n, rng, geometric=True):
    N = n + 1
    W = np.full((N, N), np.inf); np.fill_diagonal(W, 0)
    for v in range(1, N):
        u = rng.integers(0, v)
        w = math.exp(rng.normal(0, 1)) if geometric else rng.random() + 0.1
        W[u, v] = W[v, u] = w
    return graph_metric(W)

def random_graph(n, rng, p=0.4):
    N = n + 1
    W = np.full((N, N), np.inf); np.fill_diagonal(W, 0)
    for v in range(1, N):  # connectivity
        u = rng.integers(0, v); w = rng.random() + 0.1
        W[u, v] = W[v, u] = w
    for u in range(N):
        for v in range(u + 1, N):
            if rng.random() < p:
                w = rng.random() + 0.1
                W[u, v] = W[v, u] = min(W[u, v], w)
    return graph_metric(W)

def star(n, rng, geometric=True):
    N = n + 1
    leg = np.exp(rng.random(n) * 4) if geometric else rng.random(n) + 0.2
    W = np.full((N, N), np.inf); np.fill_diagonal(W, 0)
    for i in range(n):
        W[0, i + 1] = W[i + 1, 0] = leg[i]
    return graph_metric(W)

def spider_branches(n, rng):
    """spine with side branches (caterpillar) -- the geometric-scale tree example"""
    N = n + 1
    W = np.full((N, N), np.inf); np.fill_diagonal(W, 0)
    spine = [0]
    for v in range(1, N):
        if rng.random() < 0.5 or len(spine) == 1:
            u = spine[-1]; w = math.exp(rng.normal(0, .7)); spine.append(v)
        else:
            u = spine[rng.integers(0, len(spine))]; w = 0.1 * math.exp(rng.normal(0, 1))
        W[u, v] = W[v, u] = w
    return graph_metric(W)

def polar_geo(n, rng, spread=5.0, ang=None):
    r = np.exp(rng.random(n) * spread); th = rng.random(n) * 2 * np.pi * (1 if ang is None else ang)
    P = np.zeros((n + 1, 2)); P[1:, 0] = r * np.cos(th); P[1:, 1] = r * np.sin(th)
    return euclid(P)

def thin_wedge(n, rng):
    return polar_geo(n, rng, 5.0, 0.1)

GENERATORS = {
    'polar_geo': polar_geo, 'wedge': thin_wedge,
    'euclid': random_euclid, 'cluster': clustered, 'line': line,
    'line_geo': lambda n, r: line(n, r, True), 'line2': two_sided_line,
    'ray': ray_like, 'tree': random_tree, 'graph': random_graph,
    'star': star, 'star_lin': lambda n, r: star(n, r, False), 'caterp': spider_branches,
}

def check_metric(D, tol=1e-9):
    n = len(D)
    assert np.allclose(D, D.T)
    for k in range(n):
        assert (D <= D[:, k:k+1] + D[k:k+1, :] + tol).all()

# ---------------------------------------------------------------- evaluation
def route_times(D, route):
    """arrival times B_j of the chained route (list of site indices 1..n)."""
    t = []; cur = 0; T = 0.0
    for s in route:
        T += D[cur, s]; t.append(T); cur = s
    return t

def protected_weight(D, w, route, lam, beta=0.0):
    a = D[0]
    t = route_times(D, route)
    return sum(w[s] for s, B in zip(route, t) if B <= beta + lam * a[s] + 1e-12)

def all_protected(D, route, lam, beta=0.0, tol=1e-9):
    a = D[0]
    return all(B <= beta + lam * a[s] + tol for s, B in zip(route, route_times(D, route)))

def excess(D, route):
    a = D[0]
    return [B - a[s] for s, B in zip(route, route_times(D, route))]

# ---------------------------------------------------------------- exact chain
def chain_exact(D, w, lam, beta=0.0, return_route=False, tol=1e-12):
    """max protected weight: Held-Karp over subsets of protected sites.
    best[S][j] = min arrival time at j having visited exactly S (all protected), ending j."""
    n = len(D) - 1
    a = D[0, 1:]; h = beta + lam * a
    D1 = D[1:, 1:]
    INF = float('inf')
    best = np.full((1 << n, n), INF)
    par = {}
    for j in range(n):
        if a[j] <= h[j] + tol:
            best[1 << j, j] = a[j]
    bw = 0.0; bS = 0
    wt = np.array(w[1:], float) if len(w) == n + 1 else np.array(w, float)
    # weight of each mask
    mw = np.zeros(1 << n)
    for S in range(1, 1 << n):
        low = S & -S; j = low.bit_length() - 1
        mw[S] = mw[S ^ low] + wt[j]
    for S in range(1, 1 << n):
        row = best[S]
        if not np.isfinite(row).any():
            continue
        if mw[S] > bw:
            bw = mw[S]; bS = S
        for j in range(n):
            tj = row[j]
            if tj == INF:
                continue
            for k in range(n):
                if S >> k & 1:
                    continue
                T = tj + D1[j, k]
                if T <= h[k] + tol and T < best[S | 1 << k, k]:
                    best[S | 1 << k, k] = T
                    if return_route:
                        par[(S | 1 << k, k)] = j
    if not return_route:
        return bw
    # reconstruct
    if bS == 0:
        return bw, []
    S = bS; j = int(np.argmin(best[S]))
    r = []
    while True:
        r.append(j + 1)
        if S == 1 << j: break
        pj = par[(S, j)]; S ^= 1 << j; j = pj
    return bw, r[::-1]

def chain_bruteforce(D, w, lam, beta=0.0, tol=1e-12):
    """independent check: enumerate all ordered subsets (n<=7)."""
    n = len(D) - 1; best = 0.0
    for k in range(1, n + 1):
        for perm in itertools.permutations(range(1, n + 1), k):
            if all_protected(D, perm, lam, beta, tol):
                best = max(best, sum(w[s] for s in perm))
    return best

# ---------------------------------------------------------------- spoke
def spoke_exact(D, w, lam, beta=0.0, speed=1.0, tol=1e-12):
    """subset DP; arrival A_j = (2*sum_{l<j} a_l + a_j)/speed in EDD order."""
    n = len(D) - 1
    a = D[0, 1:]; h = beta + lam * a
    order = sorted(range(n), key=lambda i: h[i])
    wt = np.array(w[1:], float) if len(w) == n + 1 else np.array(w, float)
    # DP over EDD order: f[load] -- subset enumeration (n small)
    best = 0.0
    for S in range(1, 1 << n):
        load = 0.0; ok = True; tot = 0.0
        for i in order:
            if S >> i & 1:
                if (2 * load + a[i]) / speed > h[i] + tol:
                    ok = False; break
                load += a[i]; tot += wt[i]
        if ok and tot > best: best = tot
    return best

# ---------------------------------------------------------------- exact class oracle
def v_unif_exact(D, w, sites, Phi, tol=1e-12):
    """max weight of a route (from depot) over `sites` with all excess <= Phi.
    g[S][u] = min arrival time; excess monotone so check last only. Returns (weight, route)."""
    sites = list(sites); m = len(sites)
    if m == 0: return 0.0, []
    a = D[0]
    INF = float('inf')
    g = np.full((1 << m, m), INF); par = {}
    for u in range(m):
        g[1 << u, u] = a[sites[u]]
    bw = 0.0; bS = 0; bu = -1
    mw = np.zeros(1 << m)
    for S in range(1, 1 << m):
        low = S & -S; j = low.bit_length() - 1
        mw[S] = mw[S ^ low] + w[sites[j]]
    for S in range(1, 1 << m):
        for u in range(m):
            tu = g[S, u]
            if tu == INF: continue
            if tu - a[sites[u]] > Phi + tol:
                continue
            if mw[S] > bw: bw, bS, bu = mw[S], S, u
            for k in range(m):
                if S >> k & 1: continue
                T = tu + D[sites[u], sites[k]]
                if T < g[S | 1 << k, k]:
                    g[S | 1 << k, k] = T; par[(S | 1 << k, k)] = u
    if bS == 0: return 0.0, []
    S, u = bS, bu; r = []
    while True:
        r.append(sites[u])
        if S == 1 << u: break
        pu = par[(S, u)]; S ^= 1 << u; u = pu
    return bw, r[::-1]
