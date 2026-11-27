"""Fleet-level execution of BATON route policies under drift, with replanning.

Prototype support for the TEMPO-2 experiments. A plan's routes run
simultaneously; events are processed in stop-clock order (stop 1 of every
route, then stop 2 of every alive route, ...). Each route follows a BATON
policy fit by ``core.costs.fit_lsm_actions`` and valued exactly by
``ev.vi.RouteValue``. A replan at stop index k refits every alive route's
remaining-stop policy under a pickup-scale estimate a_hat, conditioning
on the current load through a pseudo-stop whose demand is the load.

Conventions
-----------
* Planning law: p_i ~ Gamma(mean pbar_i, sd CV*pbar_i) and
  d_i ~ Gamma(mean dbar_i, sd CV*dbar_i), independent, CV = 0.30.
* A drift scenario multiplies pickups by a_p and deliveries by a_d from
  1-based stop index ``onset`` on (on every route).
* ``RoutePolicy.offset = k*`` means the policy was fit after real stop k*;
  its RouteValue lives in suffix coordinates (suffix stop j = real stop
  k* + j), so ``rv.H[j-1]``, ``rv.E[j-1]``, ``rv.R[j-1]`` are the prices of
  real stop k* + j.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from core.costs import (LastMileCosts, route_cost_schedules, restock_schedule,
                        fit_lsm_actions)
from ev.vi import RouteValue, ValueInnovationMonitor

CV = 0.30
N_FIT = 1000
G_DAY0 = 1500
G_REFIT = 1500
RESULTS = Path(__file__).resolve().parent.parent / "results"


# ============================================================
# Dethloff instance parsing (copied from scripts/dethloff_runner.py)
# ============================================================


def _lines_after(txt, tag):
    out, cap = [], False
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if not cap:
            if s.upper().startswith(tag):
                cap = True
            continue
        if any(ch.isalpha() for ch in s):
            break
        out.append(s)
    return out


def _header_int(txt, key):
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            m = re.findall(r"-?\d+", line)
            if m:
                return int(m[-1])
    return None


def _header_float(txt, key):
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            m = re.findall(r"-?\d+\.?\d*", line)
            if m:
                return float(m[-1])
    return None


def parse_instance(path):
    """Return (D, Q, n, scale) for a Dethloff-format instance file."""
    txt = Path(path).read_text(errors="ignore")
    n = _header_int(txt, "DIMENSION")
    Q = _header_float(txt, "CAPACITY")
    toks = []
    for row in _lines_after(txt, "EDGE_WEIGHT_SECTION"):
        toks.extend(row.split())
    vals = [int(float(t)) for t in toks]
    if n is None or len(vals) != n * n:
        raise ValueError("EDGE_WEIGHT_SECTION: need the FULL n x n matrix")
    D = np.array(vals, dtype=np.int64).reshape(n, n)
    return D, Q, n, 10000


# ============================================================
# Fleet
# ============================================================


@dataclass
class RouteData:
    customers: list
    B: float
    H: np.ndarray
    E: np.ndarray
    R: np.ndarray
    pbar: np.ndarray
    dbar: np.ndarray

    @property
    def m(self) -> int:
        return len(self.customers)


class FleetPlan:
    """The non-empty routes of one plan with their slack and cost schedules."""

    def __init__(self, D, Q, scale, dbar, pbar, routes, costs=None, name=""):
        costs = costs or LastMileCosts()
        dbar = np.asarray(dbar, dtype=float)
        pbar = np.asarray(pbar, dtype=float)
        self.name = name
        self.routes: list[RouteData] = []
        for route in routes:
            route = [int(c) for c in route]
            if not route:
                continue
            idx = np.array(route)
            B = float(Q - dbar[idx].sum())
            if B < 0:
                raise ValueError(f"route departs over capacity: B = {B}")
            H, E = route_cost_schedules(route, D, scale, costs)
            R = restock_schedule(route, D, scale, costs)
            self.routes.append(RouteData(route, B, H, E, R,
                                         pbar[idx].copy(), dbar[idx].copy()))
        if not self.routes:
            raise ValueError("fleet has no non-empty routes")
        self.m_max = max(r.m for r in self.routes)

    @classmethod
    def from_instance(cls, path, gate="Det", plans_dir=None, costs=None):
        D, Q, _, scale = parse_instance(path)
        plans_dir = Path(plans_dir) if plans_dir else RESULTS / "plans"
        stem = Path(path).stem
        sol = json.loads((plans_dir / f"{stem}.json").read_text())
        return cls(D, Q, scale, sol["dbar"], sol["pbar"],
                   sol["res"][gate]["plan"], costs, name=stem)


# ============================================================
# Drift scenarios
# ============================================================

# name: (a_p, a_d, onset); onset None means ceil(0.4 * longest route).
SCENARIOS = {
    "null":      (1.0, 1.0, 1),
    "light":     (0.7, 1.0, 1),
    "heavy_del": (1.0, 1.3, 1),
    "surge20":   (1.2, 1.0, 1),
    "surge30":   (1.3, 1.0, 1),
    "surge50":   (1.5, 1.0, 1),
    "late30":    (1.3, 1.0, None),
}


@dataclass(frozen=True)
class Drift:
    name: str
    a_p: float
    a_d: float
    onset: int

    @property
    def is_null(self) -> bool:
        return self.a_p == 1.0 and self.a_d == 1.0


def make_drift(name: str, fleet: FleetPlan) -> Drift:
    a_p, a_d, onset = SCENARIOS[name]
    if onset is None:
        onset = (2 * fleet.m_max + 4) // 5       # ceil(0.4 * m_max), exact
    return Drift(name, float(a_p), float(a_d), int(onset))


def _as_drift(scenario, fleet):
    return make_drift(scenario, fleet) if isinstance(scenario, str) else scenario


def _gamma_block(mean, N, rng):
    out = np.zeros((N, mean.size))
    pos = mean > 0
    if pos.any():
        out[:, pos] = rng.gamma(1.0 / CV ** 2, scale=mean[pos] * CV ** 2,
                                size=(N, int(pos.sum())))
    return out


def _sample_pd(pbar, dbar, N, rng, drift=None):
    """(p, d) arrays of shape (N, m) under the planning law, with drift."""
    p = _gamma_block(pbar, N, rng)
    d = _gamma_block(dbar, N, rng)
    if drift is not None:
        s = drift.onset - 1
        p[:, s:] *= drift.a_p
        d[:, s:] *= drift.a_d
    return p, d


def sample_day(fleet: FleetPlan, scenario, rng):
    """One day for the fleet. Returns (gs, ps): per-route arrays g = p - d
    and p (pickups, needed for estimation). Drift is applied to the same
    base draws, so scenarios share common random numbers."""
    drift = _as_drift(scenario, fleet)
    gs, ps = [], []
    for route in fleet.routes:
        p, d = _sample_pd(route.pbar, route.dbar, 1, rng, drift)
        gs.append(p[0] - d[0])
        ps.append(p[0])
    return gs, ps


# ============================================================
# Policies
# ============================================================


def fit_route_policy(pbar_r, dbar_r, B, H, E, R, seed, N=N_FIT, w0=0.0,
                     G=G_DAY0):
    """Fit a BATON policy on N planning-law days of one route.

    With w0 == 0 this is the plain fit on the route. With w0 != 0 the fit
    is conditioned on a current load w0 (pseudo-stop, see ``_fit_suffix``)
    and the arrays are the remaining stops.

    Returns (models, rv) with models = {k: (iso, F_k)} in stop coordinates
    of the arrays passed in.
    """
    if w0 != 0.0:
        return _fit_suffix(pbar_r, dbar_r, B, H, E, R, w0, seed, N, G,
                           (H[0], E[0], R[0]))
    rng = np.random.default_rng(seed)
    p, d = _sample_pd(np.asarray(pbar_r, float), np.asarray(dbar_r, float),
                      N, rng)
    models = fit_lsm_actions(p - d, B, H, E, R)
    rv = RouteValue(pbar_r, dbar_r, B, H, E, R, models, cv=CV, G=G)
    return models, rv


def _fit_suffix(pbar, dbar, B, H, E, R, W, seed, N, G, pseudo):
    """Policy for remaining stops 1..m given current load W before stop 1.

    Training matrix [W, g_1..g_m]: a first pseudo-stop whose demand is the
    constant load W, so cumulative loads are absolute loads. Its prices
    (``pseudo``) only enter the update at the dropped pseudo-stop, so they
    do not affect any kept model. The decision for real stop j uses the
    fitted stop j+1.
    """
    rng = np.random.default_rng(seed)
    pbar = np.asarray(pbar, float)
    dbar = np.asarray(dbar, float)
    m = len(pbar)
    p, d = _sample_pd(pbar, dbar, N, rng)
    g = np.hstack([np.full((N, 1), float(W)), p - d])
    H0, E0, R0 = pseudo
    Hf = np.concatenate([[H0], np.asarray(H, float)])
    Ef = np.concatenate([[E0], np.asarray(E, float)])
    Rf = np.concatenate([[R0], np.asarray(R, float)])
    fm = fit_lsm_actions(g, B, Hf, Ef, Rf)
    models = {j: fm[j + 1] for j in range(1, m)}
    rv = RouteValue(pbar, dbar, B, H, E, R, models, cv=CV, G=G)
    return models, rv


@dataclass
class RoutePolicy:
    offset: int          # real stop after which the policy was fit (0 = day 0)
    rv: RouteValue       # suffix coordinates: suffix stop j = real stop offset + j
    a: float = 1.0       # pickup scale the policy was fit under


def fit_fleet_day0(fleet: FleetPlan, seed: int, N=N_FIT, G=G_DAY0):
    """Day-0 policies for every route of the fleet (planning law)."""
    out = []
    for r, rt in enumerate(fleet.routes):
        models, rv = fit_route_policy(rt.pbar, rt.dbar, rt.B, rt.H, rt.E, rt.R,
                                      seed + r, N=N, G=G)
        out.append(RoutePolicy(0, rv, 1.0))
    return out


def refit_suffix(route: RouteData, k: int, W: float, a_hat: float, seed,
                 N=N_FIT, G=G_REFIT) -> RoutePolicy:
    """Replan policy for the stops k+1..m of one route after stop k, at
    load W, under a pickup scale a_hat."""
    if not 0 <= k < route.m:
        raise ValueError("no remaining stops to refit")
    pb = float(a_hat) * route.pbar[k:]
    db = route.dbar[k:]
    pseudo = (route.H[max(k - 1, 0)], route.E[max(k - 1, 0)],
              route.R[max(k - 1, 0)])
    _, rv = _fit_suffix(pb, db, route.B, route.H[k:], route.E[k:],
                        route.R[k:], W, seed, N, G, pseudo)
    return RoutePolicy(offset=k, rv=rv, a=float(a_hat))


def estimate_a(p_obs, pbar_obs) -> float:
    """Pickup scale: observed pickups over planned means (customers with
    pbar > 0), floored at 0.3, capped at 3.0; 1.0 if nothing observed."""
    p_obs = np.asarray(p_obs, dtype=float)
    pbar_obs = np.asarray(pbar_obs, dtype=float)
    mask = pbar_obs > 0
    if not mask.any():
        return 1.0
    denom = float(pbar_obs[mask].sum())
    if denom <= 0.0:
        return 1.0
    a = float(p_obs[mask].sum()) / denom
    return min(max(a, 0.3), 3.0)


# ============================================================
# Triggers
# ============================================================


class Trigger:
    """Base trigger. ``observe`` returns True to replan after a stop index.
    ``a_override`` (if not None) replaces the estimated pickup scale."""

    name = "Trigger"
    param = None
    a_override = None

    def reset_day(self):
        pass

    def observe(self, stop_index, obs, extra) -> bool:
        return False

    def after_replan(self):
        pass


class Never(Trigger):
    name = "Never"


class EveryK(Trigger):
    name = "EveryK"

    def __init__(self, k: int):
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.param = self.k

    def observe(self, stop_index, obs, extra):
        return stop_index % self.k == 0


class Continuous(EveryK):
    name = "Continuous"

    def __init__(self):
        super().__init__(1)


class ValueInnovation(Trigger):
    name = "ValueInnovation"

    def __init__(self, alpha=0.05, eps=0.0):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.param = self.alpha
        self.mon = ValueInnovationMonitor(alpha)

    def reset_day(self):
        self.mon.reset()

    def after_replan(self):
        self.mon.reset()

    def observe(self, stop_index, obs, extra):
        fire = False
        for o in obs:
            if self.mon.update(o["I"], o["L"], self.eps):
                fire = True
        return fire


THETA = np.array([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])


def _z_score(o, extra):
    """Standardised demand residual of one observed stop under the CURRENT
    law (pickup means scaled by the current scale a_cur). None if sd = 0."""
    rt = extra["fleet"].routes[o["r"]]
    k = o["k"]
    pb = extra["a_cur"] * rt.pbar[k - 1]
    db = rt.dbar[k - 1]
    sd = CV * math.sqrt(pb * pb + db * db)
    if sd <= 0.0:
        return None
    return (o["g"] - (pb - db)) / sd


class ModelFitLR(Trigger):
    """Mixture likelihood-ratio e-process on the standardised residuals:
    average over THETA of prod exp(theta z - theta^2/2). Alarm at 1/alpha."""

    name = "ModelFitLR"

    def __init__(self, alpha=0.05):
        self.alpha = float(alpha)
        self.param = self.alpha
        self.logW = np.zeros(len(THETA))

    def reset_day(self):
        self.logW = np.zeros(len(THETA))

    def after_replan(self):
        self.logW = np.zeros(len(THETA))

    def observe(self, stop_index, obs, extra):
        for o in obs:
            z = _z_score(o, extra)
            if z is not None:
                self.logW = self.logW + THETA * z - THETA ** 2 / 2.0
        log_wealth = logsumexp(self.logW) - math.log(len(THETA))
        return bool(log_wealth >= -math.log(self.alpha))


class Cusum(Trigger):
    """Two-sided CUSUM on the standardised residuals, reference value 0.25."""

    name = "Cusum"

    def __init__(self, h, k_ref=0.25):
        self.h = float(h)
        self.k_ref = float(k_ref)
        self.param = self.h
        self.sp = 0.0
        self.sm = 0.0

    def reset_day(self):
        self.sp = self.sm = 0.0

    def after_replan(self):
        self.sp = self.sm = 0.0

    def observe(self, stop_index, obs, extra):
        fire = False
        for o in obs:
            z = _z_score(o, extra)
            if z is None:
                continue
            self.sp = max(0.0, self.sp + z - self.k_ref)
            self.sm = max(0.0, self.sm - z - self.k_ref)
            if self.sp > self.h or self.sm > self.h:
                fire = True
        return fire


class CostBased(Trigger):
    """Replan when the predicted saving from refitting under the estimated
    pickup scale exceeds C_rp. The saving is the sum over the evaluated
    alive routes of (current policy cost - refitted policy cost) from the
    current state under a_hat, both valued at grid size G_eval.

    ``max_routes`` caps the number of routes evaluated (the highest-load
    ones), scaling the saving by #remaining / #evaluated. ``cache`` (a dict)
    may be shared across C_rp values for one day: the saving depends only
    on the state, so trajectories that coincide reuse each other's values.
    """

    name = "CostBased"

    def __init__(self, C_rp, max_routes=None, G_eval=800, cache=None):
        self.C_rp = float(C_rp)
        self.param = self.C_rp
        self.max_routes = max_routes
        self.G_eval = int(G_eval)
        self.cache = {} if cache is None else cache

    def observe(self, stop_index, obs, extra):
        a_hat, a_cur = extra["a_hat"], extra["a_cur"]
        if abs(a_hat - a_cur) < 0.05:
            return False
        rem = extra["remaining"]
        if not rem:
            return False
        W = extra["W"]
        evalset = sorted(rem, key=lambda r: -W[r])
        if self.max_routes:
            evalset = evalset[:self.max_routes]
        key = (stop_index, extra["last_replan"], round(a_cur, 9),
               round(a_hat, 9), len(rem),
               tuple((r, round(float(W[r]), 9)) for r in evalset))
        if key not in self.cache:
            self.cache[key] = self._saving(stop_index, evalset, extra)
        saving = self.cache[key] * len(rem) / len(evalset)
        return saving > self.C_rp

    def _saving(self, k, evalset, extra):
        fleet, a_hat = extra["fleet"], extra["a_hat"]
        total = 0.0
        for r in evalset:
            rt = fleet.routes[r]
            W = float(extra["W"][r])
            cur = _current_suffix_cost(rt, extra["policies"][r], k, a_hat,
                                       W, self.G_eval)
            ref = refit_suffix(rt, k, W, a_hat,
                               seed=[extra["base"], k, r],
                               G=self.G_eval).rv.U(1, W)
            total += cur - ref
        return total


def _current_suffix_cost(rt, pol, k, a_hat, W, G):
    """Expected remaining cost from load W after real stop k, of the CURRENT
    policy (models and prices as fit) evaluated under pickup scale a_hat."""
    s = k - pol.offset
    H = pol.rv.H[s:]
    E = pol.rv.E[s:]
    R = pol.rv.R[s:]
    models = {j - s: v for j, v in pol.rv.models.items() if j > s}
    rv = RouteValue(a_hat * rt.pbar[k:], rt.dbar[k:], rt.B, H, E, R, models,
                    cv=CV, G=G)
    return rv.U(1, W)


class Oracle(Trigger):
    """Replans once, at the onset stop index of the true drift, with the
    true pickup factor a_p. Never replans on the null scenario."""

    name = "Oracle"

    def __init__(self, drift: Drift | None):
        self.onset = None if (drift is None or drift.is_null) else drift.onset
        self.a_override = None if self.onset is None else drift.a_p
        self.param = self.onset

    def observe(self, stop_index, obs, extra):
        return self.onset is not None and stop_index == self.onset


# ============================================================
# Fleet-day simulation
# ============================================================


def simulate_fleet_day(fleet: FleetPlan, policies, day, trigger, C_rp, rng):
    """Execute one fleet day in stop-clock order.

    day : (gs, ps) from ``sample_day``.
    policies : list of RoutePolicy, one per fleet route (not modified).
    rng : Generator; one draw seeds all replan refits, and refit seeds are
        then derived from (seed, stop index, route) so that triggers that
        replan at the same stops see identical refit randomness.

    Returns dict: cost (recourse + replan charges), recourse_cost,
    n_replans, breaches, handoffs, restocks, replan_stops.
    """
    gs, ps = day
    n_r = len(fleet.routes)
    base = int(rng.integers(2 ** 31))
    trigger.reset_day()
    pol = list(policies)
    W = np.zeros(n_r)
    alive = [True] * n_r
    recourse = 0.0
    n_rep = 0
    breaches = handoffs = restocks = 0
    replan_stops = []
    obs_p, obs_pbar = [], []
    a_cur = 1.0
    last_replan = 0

    for k in range(1, fleet.m_max + 1):
        if not any(alive):
            break
        obs = []
        for r, rt in enumerate(fleet.routes):
            if not alive[r] or k > rt.m:
                continue
            g = float(gs[r][k - 1])
            p = float(ps[r][k - 1])
            obs_p.append(p)
            obs_pbar.append(rt.pbar[k - 1])
            cur = pol[r]
            j = k - cur.offset
            I, L = cur.rv.innovation(j, W[r], g)
            obs.append({"r": r, "k": k, "g": g, "p": p, "I": I, "L": L})

            W[r] += g
            if W[r] > rt.B:
                recourse += float(rt.E[k - 1])
                breaches += 1
                alive[r] = False
                continue
            if k == rt.m:
                alive[r] = False
                continue
            entry = cur.rv.models.get(j)
            if entry is None:
                continue
            iso, F = entry
            chat = float(np.asarray(iso.predict([W[r]]), dtype=float).ravel()[0])
            v_ho = float(cur.rv.H[j - 1])
            v_rs = float(cur.rv.R[j - 1]) + float(F)
            if v_ho < chat and v_ho <= v_rs:
                recourse += v_ho
                handoffs += 1
                alive[r] = False
            elif v_rs < chat:
                recourse += float(cur.rv.R[j - 1])
                restocks += 1
                W[r] = 0.0

        remaining = [r for r in range(n_r)
                     if alive[r] and k < fleet.routes[r].m]
        if not remaining:
            continue
        a_hat_est = estimate_a(obs_p, obs_pbar)
        extra = {"fleet": fleet, "policies": pol, "W": W.copy(),
                 "remaining": remaining, "a_hat": a_hat_est, "a_cur": a_cur,
                 "last_replan": last_replan, "base": base, "obs_p": obs_p,
                 "obs_pbar": obs_pbar}
        if not trigger.observe(k, obs, extra):
            continue
        override = trigger.a_override
        a_hat = float(override) if override is not None else a_hat_est
        for r in remaining:
            pol[r] = refit_suffix(fleet.routes[r], k, W[r], a_hat,
                                  seed=[base, k, r])
        a_cur = a_hat
        last_replan = k
        n_rep += 1
        replan_stops.append(k)
        trigger.after_replan()

    return {
        "cost": recourse + C_rp * n_rep,
        "recourse_cost": recourse,
        "n_replans": n_rep,
        "breaches": breaches,
        "handoffs": handoffs,
        "restocks": restocks,
        "replan_stops": replan_stops,
    }
