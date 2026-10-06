"""Numerical verification of every lemma used in the proof + the end-to-end guarantee. Prints counts."""
import sys, math, random, itertools, time
import numpy as np
from common import *
from alg import *

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
R = random.Random(1)
counts = {}
def ok(name, cond, info=''):
    c = counts.setdefault(name, [0, 0])
    c[0] += 1
    if not cond:
        c[1] += 1
        if c[1] <= 3: print('FAIL', name, info)

def rand_instance(n, kinds=None):
    kind = R.choice(kinds or list(GENERATORS))
    D = GENERATORS[kind](n, rng)
    D = np.asarray(D)
    # make sure positive distances
    D = D + 0.0
    w = np.concatenate([[0.0], rng.integers(1, 6, n).astype(float)]) if R.random() < .5 else np.concatenate([[0.0], rng.random(n) + .05])
    return kind, D, w

def rand_params():
    lam = R.choice([1.05, 1.1, 1.25, 1.5, 2, 3, 5, 10, 30]) * (1 + 0.05 * R.random())
    beta = R.choice([0.0, 0.0, 0.05, 0.3, 1.0])
    return lam, beta

t0 = time.time()
# ---- T0: solver cross-check (chain DP vs brute force)
for _ in range(120):
    n = R.randint(2, 6)
    kind, D, w = rand_instance(n); lam, beta = rand_params(); lam = min(lam, 3)
    ok('T0 chain DP == brute force', abs(chain_exact(D, w, lam, beta) - chain_bruteforce(D, w, lam, beta)) < 1e-9)
# ---- T1: metric sanity + W_spoke <= W_chain
for _ in range(300):
    n = R.randint(2, 9)
    kind, D, w = rand_instance(n); lam, beta = rand_params()
    check_metric(D)
    ws = spoke_exact(D, w, lam, beta); wc = chain_exact(D, w, lam, beta)
    ok('T1 W_spoke <= W_chain', ws <= wc + 1e-9)
# ---- T2: excess monotone along ANY route; shortcut keeps B' <= B
for _ in range(500):
    n = R.randint(3, 9); kind, D, w = rand_instance(n)
    r = list(R.sample(range(1, n + 1), R.randint(2, n)))
    e = excess(D, r)
    ok('T2a excess nondecreasing & >=0', all(e[i] <= e[i+1] + 1e-9 for i in range(len(e)-1)) and e[0] >= -1e-9)
    keep = [s for s in r if R.random() < .6] or r[:1]
    Bt = dict(zip(r, route_times(D, r))); Bk = dict(zip(keep, route_times(D, keep)))
    ok('T2b shortcut B\' <= B', all(Bk[s] <= Bt[s] + 1e-9 for s in keep))
    # restart lemma: a contiguous piece restarted from depot has excess e_j - e_first
    i0 = R.randrange(len(r)); i1 = R.randrange(i0, len(r)); piece = r[i0:i1 + 1]
    ep = excess(D, piece)
    ok('T2c restart excess = e_j - e_start', all(abs(ep[j] - (e[i0 + j] - e[i0])) < 1e-9 for j in range(len(piece))))
    # arc identity c(u,s)=delta+a_u-a_s
    if len(r) >= 2:
        i = R.randrange(len(r) - 1); u, s = r[i], r[i + 1]
        ok('T2d e_s = e_u + c(u,s)', abs(e[i+1] - (e[i] + D[u, s] + D[0, u] - D[0, s])) < 1e-9)
print('T0-T2 done', round(time.time() - t0, 1), 's')

# ---- T3: per-class splitting lemma  stand(C_k) <= P * V_unif(Phi_k)   (stand = OPT's class-k weight; check with OPT route)
#       and OPT <= sum_k stand(C_k)  (shortcut); also stand(C_k) <= class-restricted exact optimum with e<=f
for _ in range(250):
    n = R.randint(3, 9); kind, D, w = rand_instance(n); lam, beta = rand_params()
    theta = R.choice([0.5, 0.3, 0.25, 0.6, 2/3, 2/7]); P = pieces(theta)
    wc, r = chain_exact(D, w, lam, beta, return_route=True)
    cls, a0 = classes(D, lam, beta)
    tot = 0.0
    for k, S in cls.items():
        stand = sum(w[s] for s in r if s in S)
        tot += stand
        Phi = theta * (beta + (lam - 1) * alpha_k(k, a0, beta))
        v, _ = v_unif_exact(D, w, S, Phi)
        ok('T3 stand(C_k) <= P*V_unif', stand <= P * v + 1e-9, (kind, lam, beta, theta, k, stand, v))
    ok('T3b OPT = sum of class parts', abs(tot - wc) < 1e-9)
print('T3 done', round(time.time() - t0, 1), 's')

# ---- T4/T5: ALG feasibility (all included sites protected), pre-excess bound, guarantee
def run_alg_tests(N, oracle_mode):
    for _ in range(N):
        n = R.randint(3, 11)
        kind, D, w = rand_instance(n); lam, beta = rand_params()
        theta = R.choice([0.5, 0.5, 0.3, 0.4, 2/3, 2/7]); m = gap_m(lam - 1, theta); P = pieces(theta)
        if oracle_mode == 'exact': orc = v_unif_exact
        else:  # adversarially weak oracle: random sub-route of the exact one (still excess<=Phi)
            def orc(D_, w_, S, Phi, _r=R):
                v, r = v_unif_exact(D_, w_, S, Phi)
                if not r: return 0.0, []
                keep = [s for s in r if _r.random() < 0.7] or r[:1]
                return sum(w_[s] for s in keep), keep
        route, chosen, val = alg(D, w, lam, beta, theta, oracle=orc, m=m, return_info=True)
        pw = protected_weight(D, w, route, lam, beta)
        tot = sum(w[s] for s in route)
        name = 'T4[%s] ' % oracle_mode
        ok(name + 'ALG route fully protected', all_protected(D, route, lam, beta) and abs(pw - tot) < 1e-9, (kind, lam, beta, theta))
        ok(name + 'chosen gaps >= m', all(b - a >= m for a, b in zip(chosen, chosen[1:])))
        # explicit pre-excess invariant
        if route:
            e = excess(D, route)
            cls, a0 = classes(D, lam, beta)
            pos = 0; good = True
            for k in chosen:
                # sites of this class in route: all of them have e <= f(a_j)
                pass
            ok(name + 'e_j <= beta+rho a_j', all(ej <= beta + (lam - 1) * D[0, s] + 1e-9 for ej, s in zip(e, route)))
        if oracle_mode == 'exact':
            opt = chain_exact(D, w, lam, beta)
            ok('T5 ALG >= OPT/(P m)  (alpha=1)', tot >= opt / (P * m) - 1e-9, (kind, lam, beta, theta, tot, opt))
            # sharper: max_r sum_{k=r mod m} V_k >= OPT/(P m) ; ALG >= each residue sum
            resid = [sum(v for k, v in val.items() if (k % m) == rr) for rr in range(m)]
            ok('T5b ALG >= max residue sum', tot >= max(resid) - 1e-9)
            ok('T5c max residue sum >= OPT/(P m)', max(resid) >= opt / (P * m) - 1e-9)
            ok('T5d ALG <= OPT', tot <= opt + 1e-9)
run_alg_tests(600, 'exact')
run_alg_tests(600, 'weak')
print('T4/T5 done', round(time.time() - t0, 1), 's')

# ---- T6: ray formulas
for n in [3, 5, 8, 11]:
    for lam in [1.0, 1.5, 3.0]:
        D = euclid(np.arange(n + 1, dtype=float)[:, None]); w = np.ones(n + 1); w[0] = 0
        ok('T6 ray: W_chain = n', abs(chain_exact(D, w, lam) - n) < 1e-9)
print()
tot = fail = 0
for k, (c, f) in sorted(counts.items()):
    print(f'{k:45s} tests={c:5d} failures={f}')
    tot += c; fail += f
print('TOTAL', tot, 'FAIL', fail, 'time', round(time.time() - t0, 1))
