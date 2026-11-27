"""tempo3_hier.py: hierarchical partial pooling of demand evidence.

Day factors: log a_z = mu + eta_z, mu ~ N(0, s_c^2) (city), eta_z ~
N(0, s_z^2) (zone); zones are groups of routes (average-linkage
clustering of routes by mean customer distance). Every pickup of a route
in zone z is Gamma(mean a_z * pbar, cv 0.3). Each vehicle's recourse uses
the library value tables at the quantised posterior mean of its zone's
factor, estimated by:
  fixed    a = 1
  route    the vehicle's own pickups, prior sd sqrt(s_c^2 + s_z^2)
  fullpool all fleet pickups as one factor, same prior
  zonepool pickups of the vehicle's zone only, same prior
  hier     exact posterior of the two-level model on a log grid
  hier_ebH the same with (s_c, s_z) estimated by method of moments
           from H complete past days
  hier_wrong  hierarchical (true hyperparameters) with a WRONG zone
           partition: the route labels are randomly permuted
  oracle   true a_z
Usage (from svrpspd_wdro/): python scripts/tempo3_hier.py [n_inst=10 days=150 workers=4]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
from ev.fleet import FleetPlan, PolicyLibrary, CV, quantize_a, parse_instance  # noqa

SHAPE = 1.0 / CV ** 2
BG = np.linspace(np.log(0.35), np.log(2.8), 161)      # log-factor grid
DB = BG[1] - BG[0]
POL = ["fixed", "route", "fullpool", "zonepool", "hier", "hier_eb30", "hier_eb60", "hier_eb120", "hier_wrong", "oracle"]
EB = (30, 60, 120)
SCEN = [(0.3, 0.0), (0.2, 0.2), (0.1, 0.3), (0.0, 0.3)]
if os.environ.get("TEMPO3_SCEN"):
    SCEN = [tuple(map(float, x.split(":"))) for x in os.environ["TEMPO3_SCEN"].split(",")]


def zones_of(fl, D):
    K = len(fl.routes)
    Z = max(2, int(round(K / 3)))
    M = np.zeros((K, K))
    for i, ri in enumerate(fl.routes):
        for j, rj in enumerate(fl.routes):
            if i < j:
                M[i, j] = M[j, i] = D[np.ix_(ri.customers, rj.customers)].mean()
    lab = fcluster(linkage(squareform(M), "average"), Z, "maxclust")
    return lab - 1


def gauss(x, s):
    return np.exp(-0.5 * (x / s) ** 2) if s > 1e-9 else (np.abs(x) < DB / 2).astype(float)


def conv_gauss(f, s):
    """f on BG convolved with N(0, s^2) (same grid)."""
    if s < 1e-9:
        return f
    n = int(np.ceil(5 * s / DB))
    k = gauss(np.arange(-n, n + 1) * DB, s)
    full = np.convolve(f, k / k.sum(), mode="full")
    return full[n:n + len(f)]


def post_mean(logpost):
    w = np.exp(logpost - logpost.max())
    return float((w @ np.exp(BG)) / w.sum())


def hier_means(Lz, s_c, s_z):
    """Posterior means of a_z for every zone; Lz: (Z, grid) log-lik."""
    Z = Lz.shape[0]
    if s_z < 1e-9:                      # one shared factor
        lp = np.log(gauss(BG, s_c) + 1e-300) + Lz.sum(0)
        m = post_mean(lp)
        return [m] * Z
    if s_c < 1e-9:                      # independent zones
        return [post_mean(np.log(gauss(BG, s_z) + 1e-300) + Lz[z]) for z in range(Z)]
    Mz = []
    for z in range(Z):
        e = np.exp(Lz[z] - Lz[z].max())
        Mz.append(np.log(conv_gauss(e, s_z) + 1e-300) + Lz[z].max())
    Mz = np.array(Mz)
    out = []
    for z in range(Z):
        lmu = np.log(gauss(BG, s_c) + 1e-300) + Mz.sum(0) - Mz[z]
        pmu = np.exp(lmu - lmu.max())
        lp = Lz[z] + np.log(conv_gauss(pmu, s_z) + 1e-300)
        out.append(post_mean(lp))
    return out


def history_estimate(fl, zone, s_c, s_z, H, rng):
    """Method-of-moments estimate of (s_c, s_z) from H complete past days.

    Per day and zone, b_dz = log(sum p / sum pbar) = mu_d + eta_dz + e_dz
    with sampling variance v_z ~ CV^2 sum(pbar^2) / (sum pbar)^2.
    Within-day spread of b_dz estimates s_z^2 + mean v; the variance of
    the daily zone average estimates s_c^2 + (s_z^2 + mean v) / Z.
    """
    Z = zone.max() + 1
    pb = [np.concatenate([fl.routes[r].pbar for r in range(len(fl.routes))
                          if zone[r] == z]) for z in range(Z)]
    pb = [x[x > 0] for x in pb]
    v = np.array([CV ** 2 * (x ** 2).sum() / x.sum() ** 2 for x in pb])
    B = np.zeros((H, Z))
    for d in range(H):
        mu = rng.normal(0, s_c) if s_c > 0 else 0.0
        eta = rng.normal(0, s_z, Z) if s_z > 0 else np.zeros(Z)
        for z in range(Z):
            p = np.exp(mu + eta[z]) * pb[z] * rng.gamma(SHAPE, 1.0, len(pb[z])) / SHAPE
            B[d, z] = np.log(p.sum() / pb[z].sum())
    within = B.var(axis=1, ddof=1).mean()
    sz2 = max(0.0, within - v.mean())
    sc2 = max(0.0, B.mean(axis=1).var(ddof=1) - (sz2 + v.mean()) / Z)
    return float(np.sqrt(sc2)), float(np.sqrt(sz2))


def loglik(p, pbar):
    return -SHAPE * BG - p / (np.exp(BG) * pbar * CV ** 2)


def run_day(fl, lib, zone, s_c, s_z, rng, est, zwrong):
    K = len(fl.routes)
    Z = zone.max() + 1
    mu = rng.normal(0, s_c) if s_c > 0 else 0.0
    eta = rng.normal(0, s_z, Z) if s_z > 0 else np.zeros(Z)
    a_z = np.exp(mu + eta)
    s_tot = np.hypot(s_c, s_z)
    lprior = np.log(gauss(BG, s_tot) + 1e-300)
    base = [(rng.gamma(SHAPE, 1.0, rt.m) / SHAPE, rng.gamma(SHAPE, 1.0, rt.m) / SHAPE)
            for rt in fl.routes]
    out = {}
    for pol in POL:
        cost = 0.0
        se = 0.0
        nd = 0
        W = np.zeros(K)
        alive = [True] * K
        Lr = np.zeros((K, len(BG)))
        for k in range(1, fl.m_max + 1):
            obs = []
            for r, rt in enumerate(fl.routes):
                if not alive[r] or k > rt.m:
                    continue
                p = a_z[zone[r]] * rt.pbar[k - 1] * base[r][0][k - 1]
                d = rt.dbar[k - 1] * base[r][1][k - 1]
                if rt.pbar[k - 1] > 0:
                    Lr[r] += loglik(p, rt.pbar[k - 1])
                W[r] += p - d
                obs.append(r)
            Lz = np.array([Lr[zone == z].sum(0) for z in range(Z)])
            if pol == "hier_wrong":
                Lw = np.array([Lr[zwrong == z].sum(0) for z in range(Z)])
                hw = hier_means(Lw, s_c, s_z)
            if pol == "hier":
                hm = hier_means(Lz, s_c, s_z)
            elif pol.startswith("hier_eb"):
                hm = hier_means(Lz, *est[int(pol[7:])])
            elif pol == "fullpool":
                fm = post_mean(lprior + Lr.sum(0))
            for r in obs:
                rt = fl.routes[r]
                if W[r] > rt.B:
                    cost += rt.E[k - 1]
                    alive[r] = False
                    continue
                if k == rt.m:
                    alive[r] = False
                    continue
                a = {"fixed": 1.0,
                     "oracle": a_z[zone[r]]}.get(pol)
                if pol == "route":
                    a = post_mean(lprior + Lr[r])
                elif pol == "fullpool":
                    a = fm
                elif pol == "zonepool":
                    a = post_mean(lprior + Lz[zone[r]])
                elif pol == "hier_wrong":
                    a = hw[zwrong[r]]
                elif pol.startswith("hier"):
                    a = hm[zone[r]]
                se += (a - a_z[zone[r]]) ** 2
                nd += 1
                rv = lib.get(r, quantize_a(a)).rv
                cont = rv.U(k + 1, W[r])
                rest = rt.R[k - 1] + rv.U(k + 1, 0.0)
                ho = rt.H[k - 1]
                if ho < cont and ho <= rest:
                    cost += ho
                    alive[r] = False
                elif rest < cont:
                    cost += rt.R[k - 1]
                    W[r] = 0.0
        out[pol] = cost
        out[pol + "_se"] = se
        out[pol + "_nd"] = nd
    return out


def run(path, days):  # noqa: C901
    fl = FleetPlan.from_instance(path)
    D = parse_instance(path)[0]
    zone = zones_of(fl, np.asarray(D, float))
    lib = PolicyLibrary(fl, seed=1)
    zwrong = np.random.default_rng(5).permutation(zone)
    rows = []
    for s_c, s_z in SCEN:
        hrng = np.random.default_rng(99)
        est = {H: history_estimate(fl, zone, s_c, s_z, H, hrng) for H in EB}
        rng = np.random.default_rng(11)
        for d in range(days):
            res = run_day(fl, lib, zone, s_c, s_z, rng, est, zwrong)
            rows.append(dict(instance=fl.name, K=len(fl.routes), Z=int(zone.max() + 1),
                             s_c=s_c, s_z=s_z, day=d,
                             **{f"sc_hat{H}": est[H][0] for H in EB},
                             **{f"sz_hat{H}": est[H][1] for H in EB}, **res))
    return rows


if __name__ == "__main__":
    kw = dict(x.split("=") for x in sys.argv[1:])
    n, days, wk = int(kw.get("n_inst", 10)), int(kw.get("days", 150)), int(kw.get("workers", 4))
    fam = kw.get("fam", "Dethloff")
    paths = [p for p in sorted((_W / "data" / fam).glob("*.vrpspd"))
             if (_W / "results" / "plans" / f"{p.stem}.json").exists()][:n]
    with ProcessPoolExecutor(max_workers=wk) as ex:
        res = list(ex.map(run, paths, [days] * len(paths)))
    df = pd.DataFrame([r for rs in res for r in rs])
    (_W / "results" / "tempo3").mkdir(parents=True, exist_ok=True)
    df.to_csv(_W / "results" / "tempo3" / f"hier_{fam}{kw.get('tag', '')}.csv", index=False)
    print(df.groupby(["s_c", "s_z"])[POL].mean().round(2).to_string())
