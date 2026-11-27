"""Fleet-level execution of BATON route policies under drift, with replanning.

Prototype support for the TEMPO-2 experiments. A plan's routes run
simultaneously; events are processed in stop-clock order (stop 1 of every
route, then stop 2 of every alive route, ...). Each route follows a BATON
policy fit by ``core.costs.fit_lsm_actions`` and valued exactly by
``ev.vi.RouteValue``.

Policy library. For every route and every pickup scale a in A_GRID a policy
is fit on full-route planning-law days whose pickup means are a * pbar. The
continuation models are Markov in (stop, load), so the library policy for
scale a is used unchanged from any mid-route state: a replan only switches
each alive route to the library entry for the grid value nearest to the
estimated scale. Entries are built lazily and cached (also across days).

Conventions
-----------
* Planning law: p_i ~ Gamma(mean pbar_i, sd CV*pbar_i) and
  d_i ~ Gamma(mean dbar_i, sd CV*dbar_i), independent, CV = 0.30.
* A drift scenario multiplies pickups by a_p and deliveries by a_d from
  1-based stop index ``onset`` on (on every route).
* A replan refits only the pickup scale. A delivery change (heavy_del) is
  therefore never modelled by a replan; this is a stated limitation.
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
G_LIB = 1500      # library entries and day-0 policies
G_CROSS = 800     # cross tables used by CostBased
A_GRID = tuple(round(0.5 + 0.1 * i, 1) for i in range(16))   # 0.5 .. 2.0
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
    p = _gamma_block(np.asarray(pbar, float), N, rng)
    d = _gamma_block(np.asarray(dbar, float), N, rng)
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
# Policy library
# ============================================================


def quantize_a(a: float) -> float:
    """Nearest grid value of A_GRID (ties to the smaller value)."""
    return min(A_GRID, key=lambda g: (abs(g - float(a)), g))


def fit_route_policy(pbar_r, dbar_r, B, H, E, R, seed, N=N_FIT, G=G_LIB):
    """Fit a BATON policy on N planning-law days of one route, with the
    pickup means and deliveries exactly as passed. Returns (models, rv)."""
    rng = np.random.default_rng(seed)
    p, d = _sample_pd(pbar_r, dbar_r, N, rng)
    models = fit_lsm_actions(p - d, B, H, E, R)
    rv = RouteValue(pbar_r, dbar_r, B, H, E, R, models, cv=CV, G=G)
    return models, rv


@dataclass
class RoutePolicy:
    rv: RouteValue      # full-route coordinates (stop k uses rv stop k)
    a: float            # grid pickup scale the policy was fit under


class PolicyLibrary:
    """Per-route policies for every grid scale, built lazily and cached.

    ``get(r, a)`` returns the library policy for route r at the grid value
    nearest to a. ``cross(r, a_cur, a_q)`` is the RouteValue of the policy
    fit at a_cur, valued under pickup scale a_q (G = G_CROSS).
    """

    def __init__(self, fleet: FleetPlan, seed: int, N=N_FIT, G=G_LIB):
        self.fleet = fleet
        self.seed = int(seed)
        self.N = N
        self.G = G
        self._pol = {}
        self._cross = {}

    def seed_for(self, r: int, a: float):
        return [self.seed, int(r), int(round(quantize_a(a) * 10))]

    def get(self, r: int, a: float) -> RoutePolicy:
        a = quantize_a(a)
        key = (int(r), a)
        if key not in self._pol:
            rt = self.fleet.routes[r]
            models, rv = fit_route_policy(a * rt.pbar, rt.dbar, rt.B, rt.H,
                                          rt.E, rt.R, self.seed_for(r, a),
                                          N=self.N, G=self.G)
            self._pol[key] = RoutePolicy(rv, a)
        return self._pol[key]

    def cross(self, r: int, a_cur: float, a_q: float) -> RouteValue:
        a_cur, a_q = quantize_a(a_cur), quantize_a(a_q)
        key = (int(r), a_cur, a_q)
        if key not in self._cross:
            rt = self.fleet.routes[r]
            models = self.get(r, a_cur).rv.models
            self._cross[key] = RouteValue(a_q * rt.pbar, rt.dbar, rt.B, rt.H,
                                          rt.E, rt.R, models, cv=CV,
                                          G=G_CROSS)
        return self._cross[key]


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
    """Replan (switch every alive route to the library policy at a_q) only
    when a_q differs from the current scale and the predicted saving exceeds
    C_rp. Saving at stop k: sum over the remaining routes of
    U_cross(k+1, W) - U_lib(a_q)(k+1, W), i.e. the current policy valued
    under a_q minus the library policy for a_q, from the current loads.

    ``cache`` (a dict) may be shared across C_rp values for one day: the
    saving depends only on the state, so coinciding trajectories reuse it.
    """

    name = "CostBased"

    def __init__(self, C_rp, cache=None):
        self.C_rp = float(C_rp)
        self.param = self.C_rp
        self.cache = {} if cache is None else cache

    def observe(self, stop_index, obs, extra):
        a_q, a_cur = extra["a_q"], extra["a_cur"]
        rem = extra["remaining"]
        if a_q == a_cur or not rem:
            return False
        W = extra["W"]
        key = (stop_index, a_cur, a_q,
               tuple((r, round(float(W[r]), 9)) for r in rem))
        if key not in self.cache:
            self.cache[key] = self._saving(stop_index, rem, extra)
        return self.cache[key] > self.C_rp

    @staticmethod
    def _saving(k, rem, extra):
        lib, a_cur, a_q = extra["lib"], extra["a_cur"], extra["a_q"]
        total = 0.0
        for r in rem:
            W = float(extra["W"][r])
            cur = lib.cross(r, a_cur, a_q).U(k + 1, W)
            ref = lib.get(r, a_q).rv.U(k + 1, W)
            total += cur - ref
        return total


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


def simulate_fleet_day(lib: PolicyLibrary, day, trigger, C_rp):
    """Execute one fleet day in stop-clock order.

    lib : PolicyLibrary of the fleet (day-0 policies are its a = 1 entries).
    day : (gs, ps) from ``sample_day``.

    A replan at stop k (after all routes processed stop k, if the trigger
    fires and some alive route has remaining stops) switches every such
    route to the library policy for quantize_a(a_hat), where a_hat is the
    trigger's a_override or ``estimate_a`` of the pickups observed so far.
    Each replan costs C_rp.

    Returns dict: cost (recourse + replan charges), recourse_cost,
    n_replans, breaches, handoffs, restocks, replan_stops.
    """
    fleet = lib.fleet
    gs, ps = day
    n_r = len(fleet.routes)
    trigger.reset_day()
    pol = [lib.get(r, 1.0) for r in range(n_r)]
    W = np.zeros(n_r)
    alive = [True] * n_r
    recourse = 0.0
    n_rep = 0
    breaches = handoffs = restocks = 0
    replan_stops = []
    obs_p, obs_pbar = [], []
    a_cur = 1.0

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
            I, L = cur.rv.innovation(k, W[r], g)
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
            entry = cur.rv.models.get(k)
            if entry is None:
                continue
            iso, F = entry
            chat = float(np.asarray(iso.predict([W[r]]), dtype=float).ravel()[0])
            v_ho = float(rt.H[k - 1])
            v_rs = float(rt.R[k - 1]) + float(F)
            if v_ho < chat and v_ho <= v_rs:
                recourse += v_ho
                handoffs += 1
                alive[r] = False
            elif v_rs < chat:
                recourse += float(rt.R[k - 1])
                restocks += 1
                W[r] = 0.0

        remaining = [r for r in range(n_r)
                     if alive[r] and k < fleet.routes[r].m]
        if not remaining:
            continue
        a_hat_est = estimate_a(obs_p, obs_pbar)
        override = trigger.a_override
        a_q = quantize_a(override if override is not None else a_hat_est)
        extra = {"fleet": fleet, "lib": lib, "policies": pol,
                 "W": W.copy(), "remaining": remaining, "a_hat": a_hat_est,
                 "a_q": a_q, "a_cur": a_cur, "obs_p": obs_p,
                 "obs_pbar": obs_pbar}
        if not trigger.observe(k, obs, extra):
            continue
        for r in remaining:
            pol[r] = lib.get(r, a_q)
        a_cur = a_q
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
