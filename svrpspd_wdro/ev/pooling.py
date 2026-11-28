"""Geography-based zones and pooled factor learning for the TEMPO experiment.

Replaces the route-clustered zones and the permuted-label misspecification
of ``scripts/tempo3_hier.py`` (see papers/tempo/reviews/internal_review_v3.md).

Model (per day, per scenario)
-----------------------------
* Zones are a geographic partition of the plan's CUSTOMERS, computed from the
  classical MDS embedding of the symmetric distance matrix restricted to the
  plan's customers, then k-means (n_init=10, random_state=0). A route may span
  zones. Z = min(4, max(2, K // 2)).
* Log zone factors b_z = mu + eta_z, mu ~ N(0, s_c^2), eta_z ~ N(0, s_z^2)
  (optionally Student-t_3 scaled to the same sd). Optional within-day drift:
  a random walk on b_z applied after each stop index.
* The true pickup factor of the customer served at stop k is
  exp(b_{z(i)}) at that stop. Pickups p ~ Gamma(mean a pbar, cv_true);
  deliveries d ~ Gamma(mean f_d dbar, cv 0.3), f_d optionally random per day.
* Every estimator assumes gamma pickups with cv 0.3 and a log-normal prior.

Estimators of the zone factors (all from pickups observed by the policy's
own alive vehicles, in stop-index lockstep): route, full, zone, hier (exact
two-level grid posterior, true hyperparameters), hier_eb (hyperparameters
by method of moments on H past days from the same scenario), mle (pooled
ratio, no prior), fixed (a = 1), known (true factors; a reference, not a
bound).

A vehicle's decision factor at stop k is the pbar-weighted mean of the
per-customer factors over its remaining stops k+1..m, quantised to the
policy library grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from ev.fleet import CV, FleetPlan, PolicyLibrary, quantize_a, parse_instance  # noqa: F401

SHAPE = 1.0 / CV ** 2                  # estimator-assumed gamma shape
SHAPE_D = 1.0 / 0.3 ** 2               # delivery gamma shape (cv 0.3)
BG = np.linspace(np.log(0.35), np.log(2.8), 161)   # log-factor grid
DB = BG[1] - BG[0]
ZERO_SD = 1e-6                         # below this a hyper-sd is treated as 0
POL = ["fixed", "route", "full", "zone", "hier", "hier_eb", "mle", "known"]
H_HISTORY = 60
DEL_SD = 0.15
DRIFT_COMMON_SD = 0.03
DRIFT_ZONE_SD = 0.04


@dataclass(frozen=True)
class Scenario:
    name: str
    s_c: float
    s_z: float
    drift: bool = False
    delfactor: bool = False
    heavytail: bool = False
    cv_true: float = CV
    zones_true: str = "same"     # "same": factors on the estimator zones;
                                 # "alt": factors on an independent partition


SCENARIOS = {
    "city": Scenario("city", 0.3, 0.0),
    "mixed": Scenario("mixed", 0.2, 0.2),
    "zone": Scenario("zone", 0.0, 0.3),
    "drift": Scenario("drift", 0.2, 0.2, drift=True),
    "delfactor": Scenario("delfactor", 0.2, 0.2, delfactor=True),
    "heavytail": Scenario("heavytail", 0.2, 0.2, heavytail=True),
    "cvmis": Scenario("cvmis", 0.2, 0.2, cv_true=0.45),
    "wrongzone": Scenario("wrongzone", 0.2, 0.2, zones_true="alt"),
}


# ============================================================
# Grid posterior machinery (copied from scripts/tempo3_hier.py)
# ============================================================


def gauss(x, s):
    """Unnormalised N(0, s^2) kernel on x; a degenerate s gives a delta at
    the grid point nearest to 0."""
    x = np.asarray(x, dtype=float)
    if s < ZERO_SD:
        out = np.zeros(x.shape)
        out.flat[int(np.argmin(np.abs(x)))] = 1.0
        return out
    return np.exp(-0.5 * (x / s) ** 2)


def conv_gauss(f, s):
    """f on BG convolved with N(0, s^2) (same grid, zero padded)."""
    if s < ZERO_SD:
        return f
    n = int(np.ceil(5 * s / DB))
    k = gauss(np.arange(-n, n + 1) * DB, s)
    full = np.convolve(f, k / k.sum(), mode="full")
    return full[n:n + len(f)]


def post_mean(logpost):
    w = np.exp(logpost - logpost.max())
    return float((w @ np.exp(BG)) / w.sum())


def hier_means(Lz, s_c, s_z):
    """Posterior means of a_z for every zone; Lz: (Z, grid) log-likelihood."""
    Z = Lz.shape[0]
    if s_z < ZERO_SD:                   # one shared factor
        lp = np.log(gauss(BG, s_c) + 1e-300) + Lz.sum(0)
        return [post_mean(lp)] * Z
    if s_c < ZERO_SD:                   # independent zones
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


def loglik(p, pbar):
    """Gamma(cv CV) log-likelihood of pickup p on the log-factor grid (up to
    constants): -SHAPE*BG - p / (exp(BG) pbar CV^2). Requires pbar > 0."""
    return -SHAPE * BG - p / (np.exp(BG) * pbar * CV ** 2)


def weighted_factor(pbar_rem, fac_rem):
    """pbar-weighted mean of per-customer factors; unweighted mean when the
    remaining planned pickups sum to zero."""
    w = np.asarray(pbar_rem, dtype=float)
    f = np.asarray(fac_rem, dtype=float)
    s = float(w.sum())
    if s > 0.0:
        return float((w * f).sum() / s)
    return float(f.mean())


# ============================================================
# Zones
# ============================================================


def zone_count(K: int) -> int:
    return int(min(4, max(2, K // 2)))


def customer_zones(D, customers, K: int, Z=None, seed=0) -> np.ndarray:
    """Zone of every customer in ``customers`` (node indices into D).

    Classical MDS of the symmetrised distance matrix restricted to these
    customers (2 dimensions), then k-means with Z = zone_count(K) (or the
    given Z) and random_state = seed. Returns an int array aligned with
    ``customers``, values in range(Z).
    """
    cust = np.asarray(sorted({int(c) for c in customers}))
    if Z is None:
        Z = zone_count(K)
    if len(cust) < Z:
        raise ValueError("fewer customers than zones")
    Dc = np.asarray(D, dtype=float)[np.ix_(cust, cust)]
    Dc = 0.5 * (Dc + Dc.T)
    n = len(cust)
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ (Dc ** 2) @ J
    w, V = np.linalg.eigh(Bm)
    top = np.argsort(w)[::-1][:2]
    coords = V[:, top] * np.sqrt(np.clip(w[top], 0.0, None))
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((n, 2 - coords.shape[1]))])
    lab = KMeans(n_clusters=Z, n_init=10, random_state=seed).fit(coords).labels_
    node_zone = {int(c): int(z) for c, z in zip(cust, lab)}
    return np.array([node_zone[int(c)] for c in customers], dtype=int)


@dataclass
class Structure:
    """Zone structure of one fleet plan.

    K, Z, zone, zc, sk, pb are the estimator's view. Zt, ztrue, zct are the
    zones on which the day's factors are generated (the true partition); they
    equal Z, zone, zc unless the scenario misspecifies the zones.
    """
    K: int
    Z: int
    zone: list          # per route: zone index of each stop (0-based stop)
    zc: np.ndarray      # flattened customer zones (all routes)
    sk: np.ndarray      # flattened stop index (1-based)
    pb: np.ndarray      # flattened planned pickup means
    Zt: int             # number of true zones
    ztrue: list         # per route: true-zone index of each stop
    zct: np.ndarray     # flattened true-zone indices (all routes)


def build_structure(fl: FleetPlan, D, alt=False, Z=None) -> Structure:
    K = len(fl.routes)
    Z = zone_count(K) if Z is None else int(Z)
    allc = [c for rt in fl.routes for c in rt.customers]
    lab = customer_zones(D, allc, K, Z=Z)
    zone, pos = _split_by_route(fl, lab)
    zc = np.concatenate(zone)
    if alt:
        lab_t = customer_zones(D, allc, K, Z=Z + 1, seed=1)
        ztrue, _ = _split_by_route(fl, lab_t)
        Zt = Z + 1
        zct = np.concatenate(ztrue)
    else:
        Zt, ztrue, zct = Z, zone, zc
    sk = np.concatenate([np.arange(1, rt.m + 1) for rt in fl.routes])
    pb = np.concatenate([rt.pbar for rt in fl.routes])
    return Structure(K, Z, zone, zc, sk, pb, Zt, ztrue, zct)


def _split_by_route(fl: FleetPlan, lab):
    out, pos = [], 0
    for rt in fl.routes:
        out.append(lab[pos:pos + rt.m])
        pos += rt.m
    return out, pos


# ============================================================
# Day generation
# ============================================================


def _t3(rng, s, size=None):
    return s * rng.standard_t(3, size=size) / math.sqrt(3.0)


def draw_log_factors(scn: Scenario, Z: int, m_max: int, rng) -> np.ndarray:
    """Log zone factors per stop: array (m_max, Z), row k-1 is stop k."""
    if scn.heavytail:
        mu = float(_t3(rng, scn.s_c))
        eta = _t3(rng, scn.s_z, Z)
    else:
        mu = float(rng.normal(0.0, scn.s_c)) if scn.s_c > 0 else 0.0
        eta = rng.normal(0.0, scn.s_z, Z) if scn.s_z > 0 else np.zeros(Z)
    b0 = mu + eta
    B = np.empty((m_max, Z))
    B[0] = b0
    if scn.drift and m_max > 1:
        c = rng.normal(0.0, DRIFT_COMMON_SD, m_max - 1)
        e = rng.normal(0.0, DRIFT_ZONE_SD, (m_max - 1, Z))
        B[1:] = b0 + np.cumsum(c[:, None] + e, axis=0)
    else:
        B[:] = b0
    return B


@dataclass
class Day:
    p: list          # per route: pickups by stop
    d: list          # per route: deliveries by stop
    dec_known: list  # per route: known decision factor at stop k (index k-1)
    af: list         # per route: true factor per stop
    B: np.ndarray


def draw_day(fl: FleetPlan, st: Structure, scn: Scenario, rng) -> Day:
    B = draw_log_factors(scn, st.Zt, fl.m_max, rng)
    fd = math.exp(float(rng.normal(0.0, DEL_SD))) if scn.delfactor else 1.0
    shape_p = 1.0 / scn.cv_true ** 2
    P, Dl, DK, AF = [], [], [], []
    for r, rt in enumerate(fl.routes):
        m = rt.m
        bp = rng.gamma(shape_p, 1.0, m) / shape_p
        bd = rng.gamma(SHAPE_D, 1.0, m) / SHAPE_D
        af = np.exp(B[np.arange(m), st.ztrue[r]])
        P.append(af * rt.pbar * bp)
        Dl.append(fd * rt.dbar * bd)
        AF.append(af)
        dk = np.ones(m)
        for k in range(1, m):
            dk[k - 1] = weighted_factor(rt.pbar[k:], af[k:])
        DK.append(dk)
    return Day(P, Dl, DK, AF, B)


def history_estimate(zc, sk, pb, Z, scn: Scenario, H, rng, zct=None, Zt=None):
    """Method-of-moments (s_c, s_z) from H complete past days drawn from the
    scenario (including its misspecification). Past factors are generated on
    the true zones zct (Zt of them, default zc and Z); statistics are computed
    on the estimator zones zc. Per day and zone,
    b_dz = log(sum p / sum pbar) has sampling variance
    v_z = CV^2 sum(pbar^2) / (sum pbar)^2 (CV = 0.3, the estimator's
    assumption). Returns (s_c_hat, s_z_hat).
    """
    zc = np.asarray(zc, dtype=int)
    zct = zc if zct is None else np.asarray(zct, dtype=int)
    Zt = Z if Zt is None else Zt
    sk = np.asarray(sk, dtype=int)
    pb = np.asarray(pb, dtype=float)
    zsum = np.bincount(zc, weights=pb, minlength=Z)
    zsq = np.bincount(zc, weights=pb ** 2, minlength=Z)
    keep = zsum > 0
    if keep.sum() < 2:
        return 0.0, 0.0
    v = CV ** 2 * zsq[keep] / zsum[keep] ** 2
    Zk = int(keep.sum())
    m_max = int(sk.max())
    shape = 1.0 / scn.cv_true ** 2
    B = np.zeros((H, Zk))
    for d in range(H):
        L = draw_log_factors(scn, Zt, m_max, rng)
        a = np.exp(L[sk - 1, zct])
        p = a * pb * rng.gamma(shape, 1.0, len(pb)) / shape
        sp = np.bincount(zc, weights=p, minlength=Z)
        B[d] = np.log(sp[keep] / zsum[keep])
    within = B.var(axis=1, ddof=1).mean()
    vbar = float(v.mean())
    sz2 = max(0.0, within - vbar)
    sc2 = max(0.0, B.mean(axis=1).var(ddof=1) - (sz2 + vbar) / Zk)
    return math.sqrt(sc2), math.sqrt(sz2)


# ============================================================
# Policy simulation
# ============================================================


def _estimate_matrix(pol, K, Z, lprior, s_c, s_z, sc_hat, sz_hat,
                     Lc, Lz, Lall, Sp_z, Sb_z, Sp_all, Sb_all):
    """(K, Z) matrix of estimated zone factors for every route's view."""
    if pol == "fixed":
        return np.ones((K, Z))
    if pol == "route":
        v = np.array([post_mean(lprior + Lc[r]) for r in range(K)])
        return np.repeat(v[:, None], Z, axis=1)
    if pol == "full":
        return np.full((K, Z), post_mean(lprior + Lall))
    if pol == "zone":
        v = np.array([post_mean(lprior + Lz[z]) for z in range(Z)])
        return np.repeat(v[None, :], K, axis=0)
    if pol in ("hier", "hier_eb"):
        hs = (s_c, s_z) if pol == "hier" else (sc_hat, sz_hat)
        v = np.array(hier_means(Lz, *hs))
        return np.repeat(v[None, :], K, axis=0)
    if pol == "mle":
        if Sb_all > 0:
            fleet_ratio = Sp_all / Sb_all
        else:
            fleet_ratio = 1.0
        v = np.array([Sp_z[z] / Sb_z[z] if Sb_z[z] > 0 else fleet_ratio
                      for z in range(Z)])
        return np.repeat(v[None, :], K, axis=0)
    raise ValueError(pol)


def simulate_policy(pol, fl: FleetPlan, lib: PolicyLibrary, st: Structure,
                    day: Day, s_c, s_z, sc_hat, sz_hat, trace=None):
    """Run one estimator/decision policy over the day. Returns dict with
    cost, se (sum of squared decision-factor errors against known) and nd.

    If ``trace`` is a dict, per stop index k (1-based) it accumulates
    trace[k] = [sum of squared errors, count], mirroring se and nd. Nothing
    is added for pol == "known". Results do not depend on ``trace``."""
    K, Z, G = st.K, st.Z, len(BG)
    lprior = np.log(gauss(BG, math.hypot(s_c, s_z)) + 1e-300)
    W = np.zeros(K)
    alive = [True] * K
    Lc = np.zeros((K, G))
    Lz = np.zeros((Z, G))
    Lall = np.zeros(G)
    Sp_z, Sb_z = np.zeros(Z), np.zeros(Z)
    Sp_all = Sb_all = 0.0
    cost = se = 0.0
    nd = 0
    for k in range(1, fl.m_max + 1):
        obs = []
        for r, rt in enumerate(fl.routes):
            if not alive[r] or k > rt.m:
                continue
            p = float(day.p[r][k - 1])
            d = float(day.d[r][k - 1])
            b = float(rt.pbar[k - 1])
            if b > 0:
                z = int(st.zone[r][k - 1])
                ll = loglik(p, b)
                Lc[r] += ll
                Lz[z] += ll
                Lall += ll
                Sp_z[z] += p
                Sb_z[z] += b
                Sp_all += p
                Sb_all += b
            W[r] += p - d
            obs.append(r)
        if pol != "known" and obs:
            E = _estimate_matrix(pol, K, Z, lprior, s_c, s_z, sc_hat, sz_hat,
                                 Lc, Lz, Lall, Sp_z, Sb_z, Sp_all, Sb_all)
        for r in obs:
            rt = fl.routes[r]
            if W[r] > rt.B:
                cost += float(rt.E[k - 1])
                alive[r] = False
                continue
            if k == rt.m:
                alive[r] = False
                continue
            dk = float(day.dec_known[r][k - 1])
            if pol == "known":
                a = dk
            else:
                a = weighted_factor(rt.pbar[k:], E[r][st.zone[r][k:]])
                se += (a - dk) ** 2
                if trace is not None:
                    t = trace.setdefault(k, [0.0, 0])
                    t[0] += (a - dk) ** 2
                    t[1] += 1
            nd += 1
            rv = lib.get(r, quantize_a(a)).rv
            cont = rv.U(k + 1, W[r])
            rest = float(rt.R[k - 1]) + rv.U(k + 1, 0.0)
            ho = float(rt.H[k - 1])
            if ho < cont and ho <= rest:
                cost += ho
                alive[r] = False
            elif rest < cont:
                cost += float(rt.R[k - 1])
                W[r] = 0.0
    return {"cost": cost, "se": se, "nd": nd}


def simulate_day(fl, lib, st, day, scn: Scenario, sc_hat, sz_hat, pols=None):
    """Simulate the policies in ``pols`` (default all of POL) on one day.
    Returns {policy: result}; policies not run are absent."""
    out = {}
    for pol in (POL if pols is None else pols):
        res = simulate_policy(pol, fl, lib, st, day, scn.s_c, scn.s_z,
                              sc_hat, sz_hat)
        out[pol] = res
    return out
