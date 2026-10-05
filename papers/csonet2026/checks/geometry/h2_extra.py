"""H2 extras: (E1) tightness of the 1/kappa speed factor on the geometric ray,
(E2) FPTAS for the weighted line problem (value-indexed interval DP),
(E3) Camp Fire kappa / rho,  (E4) more seeds for every reduction."""
import math
import random
import itertools
from h2_common import *
import h2_line as L
import h2_affine as AF
import h2_gap as G


def E1():
    rows = []
    for r in (2, 3, 5):
        kappa = (r - 1) / (r + 1)
        for n in (3, 5, 8, 12, 20):
            a = [r ** j for j in range(1, n + 1)]
            # least c with spoke at speed c covering all n sites (in index order)
            need = max((2 * sum(a[:k]) + a[k]) / a[k] for k in range(n))
            rows.append((r, n, round(kappa, 4), round(need, 5), round(1 / kappa, 5),
                         round(need * kappa, 5)))
            assert need <= 1 / kappa + 1e-12          # theorem: 1/kappa suffices
    # brute-force confirmation for n<=6: spoke at speed c just below 'need' cannot cover all n
    for r in (2, 3):
        for n in range(2, 7):
            D, h = G.fam_ray(n, r)
            w = [0] + [1] * n
            kappa = (r - 1) / (r + 1)
            a = [r ** j for j in range(1, n + 1)]
            need = max((2 * sum(a[:k]) + a[k]) / a[k] for k in range(n))
            assert spoke_opt(D, h, w, speed=need * (1 + 1e-9)) == n
            assert spoke_opt(D, h, w, speed=need * (1 - 1e-6)) < n
    return rows


def line_fptas(pos, h, w, eps):
    """value-indexed interval DP with scaled weights: dp[(i,j,side)][v] = min time."""
    left, right = L.line_split(pos)
    nl, nr = len(left), len(right)
    n = nl + nr
    wmax = max(w[1:])
    K = eps * wmax / max(n, 1)
    K = max(K, 1e-12)
    sw = {i: int(w[i] // K) for i in range(1, n + 1)}
    INF = float("inf")
    dp = {(0, 0, 0): {0: 0}}
    best_v, best_true = 0, 0
    trueval = {}   # (key, v) -> some true weight achieving v ... track max true weight too
    dp2 = {(0, 0, 0): {0: (0, 0)}}  # v -> (time, trueweight)
    for tot in range(0, nl + nr + 1):
        for i in range(0, min(tot, nl) + 1):
            j = tot - i
            if j > nr:
                continue
            for side in (0, 1):
                key = (i, j, side)
                if key not in dp2:
                    continue
                cur = 0 if (i == 0 and j == 0) else (-left[i - 1][0] if side == 0 else right[j - 1][0])
                for v, (t, tw) in list(dp2[key].items()):
                    if v > best_v or (v == best_v and tw > best_true):
                        best_v, best_true = v, tw
                    for kind in (0, 1):
                        if kind == 0 and i < nl:
                            dd, s = left[i]; nk = (i + 1, j, 0); t2 = t + abs(cur + dd)
                        elif kind == 1 and j < nr:
                            dd, s = right[j]; nk = (i, j + 1, 1); t2 = t + abs(cur - dd)
                        else:
                            continue
                        ok = t2 <= h[s]
                        v2 = v + (sw[s] if ok else 0)
                        tw2 = tw + (w[s] if ok else 0)
                        slot = dp2.setdefault(nk, {})
                        if v2 not in slot or t2 < slot[v2][0]:
                            slot[v2] = (t2, tw2)
    return best_true


def E2(seed=41, trials=400):
    rng = random.Random(seed)
    worst = 1.0
    for tr in range(trials):
        n = rng.randint(2, 9)
        xs = rng.sample([x for x in range(-14, 15) if x != 0], n)
        pos = [0] + xs
        h = [0] + [rng.randint(abs(pos[i]), 40) for i in range(1, n + 1)]
        w = [0] + [rng.randint(1, 1000) for _ in range(n)]
        opt, _ = L.line_weighted_dp(pos, h, w)
        eps = rng.choice([0.5, 0.2, 0.1])
        got = line_fptas(pos, h, w, eps)
        assert got >= (1 - eps) * opt - 1e-9, (got, opt, eps)
        assert got <= opt
        worst = min(worst, got / opt if opt else 1.0)
    return trials, worst


def E3():
    import math as m
    DEP = (39.5022, -121.5522)
    S = {"Concow": (39.73722, -121.51444), "Paradise": (39.75972, -121.62194),
         "Magalia": (39.833, -121.583), "Yankee Hill": (39.70361, -121.52222)}

    def hav(a, b):
        R = 6371.0
        p1, p2 = m.radians(a[0]), m.radians(b[0])
        dphi = p2 - p1; dl = m.radians(b[1] - a[1])
        x = m.sin(dphi / 2) ** 2 + m.cos(p1) * m.cos(p2) * m.sin(dl / 2) ** 2
        return 2 * R * m.asin(m.sqrt(x))
    names = list(S)
    a = {n: hav(DEP, S[n]) for n in names}
    pairs = {(x, y): hav(S[x], S[y]) for x in names for y in names if x < y}
    kap = min(pairs[(x, y)] / (a[x] + a[y]) for (x, y) in pairs)
    rho = max(a.values()) / min(a.values())
    return {n: round(v, 2) for n, v in a.items()}, {k: round(v, 2) for k, v in pairs.items()}, \
        round(kap, 4), round(rho, 3)


def E4():
    out = {}
    out["gap_random_campaigns"] = [G.T1_T2_T4_T5(seed=s, trials=1000) for s in (101, 102, 103)]
    out["class_checks"] = [G.T5_classes(seed=s, trials=3000) for s in (201, 202)]
    out["converse"] = G.T3_converse(seed=301, trials=800)
    out["line_dp"] = [L.validate_dp(seed=s, trials=1000) for s in (401, 402)]
    out["line_reduction"] = [L.validate_reduction(seed=s, trials=80) for s in (501, 502)]
    out["affine_A2"] = [AF.A2(seed=s, trials=90) for s in (601, 602)]
    out["affine_A3"] = [AF.A3(seed=s, trials=50, nmax=5) for s in (701, 702)]
    return out


if __name__ == "__main__":
    print("E1 ray speed factor (r, n, kappa, needed c, 1/kappa, c*kappa):")
    for row in E1():
        print("   ", row)
    print("E2 FPTAS(line): trials, worst ratio:", E2())
    print("E3 Camp Fire: depot dist (km), pair dist (km), kappa_min, rho:", E3())
    print("E4 repeat-seed campaigns:")
    for k, v in E4().items():
        print("   ", k, v)
