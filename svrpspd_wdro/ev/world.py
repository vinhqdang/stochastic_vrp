"""world.py — multi-factor stochastic day simulator for one route.

Simulates the execution of one planned route as a chronological event
stream. Five stochastic factors, each with an explicit null model P0
(the law the planner assumed) so the e-process has a defensible H0:

  travel    T_leg = tau_leg * m(clock) * w(rain) * exp(eps),
            eps ~ N(-sig_T^2/2, sig_T^2)  (so E[multiplier noise] = 1);
            m(.) is a known diurnal curve, rain is an exogenous
            covariate visible to the monitor — both belong to the
            conditional null, not to the alternative.
  demand    net increment g_k ~ N(mu_k, sig_k^2) per stop (mu, sig from
            the planning scenariogenerator, cf. dethloff_runner).
  dwell     S_k = a + b*|g_k| + eps, eps ~ N(0, sig_S^2), truncated >= 0.
  accident  zone-wide Poisson counts with rate lam0 per hour, observed
            on each leg over its duration.
  breakdown Bernoulli(p0) per leg.

Drift is injected at a change-point: after clock time t_star the chosen
factor's law shifts (DriftSpec). Null days use DriftSpec(kind="none").

Events are returned in chronological order as dicts with the channel
name, the realized value, the information needed to standardize it
under P0, and predictable context (clock, remaining stops/distance,
running load) available BEFORE the realization — the e-process uses
that context for its sensitivity tilts, which keeps them predictable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── null-model parameters ────────────────────────────────────────────────────

@dataclass
class DayParams:
    """Planning model P0 for one route-day."""
    tau: np.ndarray            # baseline leg times (hours), len m+1 (depot->1..->m->depot ignored past m)
    mu_g: np.ndarray           # mean net demand increment per stop, len m
    sig_g: np.ndarray          # sd of net increment per stop, len m
    B: float                   # residual capacity of the route (paper-1 sense)
    sig_T: float = 0.25        # log-sd of travel-time noise
    dwell_a: float = 0.05      # hours, base service time
    dwell_b: float = 0.002     # hours per unit |g|
    sig_S: float = 0.02        # sd of dwell noise (hours)
    lam0: float = 0.20         # accidents per hour (zone-wide) under H0
    p_break: float = 0.002     # breakdown probability per leg
    t0: float = 8.0            # departure clock time (hours)
    rain_prob: float = 0.25    # chance the day is rainy (exogenous, known to monitor)
    rain_mult: float = 1.30    # travel multiplier when raining
    diurnal: tuple = ((7.0, 9.5, 1.6), (16.5, 19.0, 1.8))  # (start, end, mult) peaks


def diurnal_mult(clock: float, peaks) -> float:
    for s, e, m in peaks:
        if s <= clock % 24.0 <= e:
            return m
    return 1.0


# ── drift specification ──────────────────────────────────────────────────────

@dataclass
class DriftSpec:
    """What goes wrong, and when. kind in {none, traffic, demand,
    accident, dwell, breakdown}. t_star is clock hours; magnitude is
    interpreted per kind (see simulate_route_day)."""
    kind: str = "none"
    t_star: float = 10.0
    magnitude: float = 1.5


# ── simulator ────────────────────────────────────────────────────────────────

def simulate_route_day(p: DayParams, drift: DriftSpec, rng: np.random.Generator):
    """Simulate one day; return (events, meta).

    events: list of dicts, chronological. Every dict has
      channel   'travel' | 'demand' | 'dwell' | 'accident' | 'breakdown'
      z / n / x the realization in the form its bet consumes:
                z = standardized residual under the CONDITIONAL null
                (travel, demand, dwell); n = accident count with 'lam0dt';
                x = breakdown indicator with 'p0'
      clock     realization clock time (hours)
      ctx       predictable context dict (computed BEFORE the draw):
                k (1-based stop index), m, W_prev (running net load
                before this stop), B, rem_frac (remaining-stop fraction),
                drifted (POST-HOC ground-truth flag, for scoring only —
                monitors must not read it)
    meta: dict with clock_end, rain (bool), m.
    """
    m = len(p.mu_g)
    rain = bool(rng.random() < p.rain_prob)
    clock = p.t0
    W = 0.0
    events = []

    def drifted(kind, t):
        return drift.kind == kind and t >= drift.t_star

    for k in range(1, m + 1):
        rem_frac = (m - k + 1) / m
        # ── leg k: travel + accident exposure + breakdown ────────────
        base = p.tau[k - 1] * diurnal_mult(clock, p.diurnal) * \
            (p.rain_mult if rain else 1.0)
        mu_log = np.log(base) - 0.5 * p.sig_T ** 2
        shift = np.log(drift.magnitude) if drifted("traffic", clock) else 0.0
        logT = rng.normal(mu_log + shift, p.sig_T)
        T = float(np.exp(logT))
        events.append(dict(
            channel="travel", z=float((logT - mu_log) / p.sig_T),
            val=T, base=base,
            clock=clock, ctx=dict(k=k, m=m, W_prev=W, B=p.B,
                                  rem_frac=rem_frac,
                                  drifted=drifted("traffic", clock))))

        lam = p.lam0 * (drift.magnitude if drifted("accident", clock) else 1.0)
        n_acc = int(rng.poisson(lam * T))
        events.append(dict(
            channel="accident", n=n_acc, lam0dt=p.lam0 * T,
            clock=clock, ctx=dict(k=k, m=m, W_prev=W, B=p.B,
                                  rem_frac=rem_frac,
                                  drifted=drifted("accident", clock))))

        pb = min(0.9, p.p_break * (drift.magnitude
                                   if drifted("breakdown", clock) else 1.0))
        x_break = int(rng.random() < pb)
        events.append(dict(
            channel="breakdown", x=x_break, p0=p.p_break,
            clock=clock, ctx=dict(k=k, m=m, W_prev=W, B=p.B,
                                  rem_frac=rem_frac,
                                  drifted=drifted("breakdown", clock))))
        clock += T

        # ── stop k: demand then dwell ────────────────────────────────
        gshift = drift.magnitude * p.sig_g[k - 1] \
            if drifted("demand", clock) else 0.0
        g = float(rng.normal(p.mu_g[k - 1] + gshift, p.sig_g[k - 1]))
        events.append(dict(
            channel="demand",
            z=float((g - p.mu_g[k - 1]) / p.sig_g[k - 1]), val=g,
            clock=clock, ctx=dict(k=k, m=m, W_prev=W, B=p.B,
                                  rem_frac=rem_frac,
                                  drifted=drifted("demand", clock))))
        W += g

        mu_S = p.dwell_a + p.dwell_b * abs(g)
        sshift = drift.magnitude * p.sig_S if drifted("dwell", clock) else 0.0
        S = max(0.0, float(rng.normal(mu_S + sshift, p.sig_S)))
        events.append(dict(
            channel="dwell", z=float((S - mu_S) / p.sig_S), val=S,
            clock=clock, ctx=dict(k=k, m=m, W_prev=W, B=p.B,
                                  rem_frac=rem_frac,
                                  drifted=drifted("dwell", clock))))
        clock += S

    return events, dict(clock_end=clock, rain=rain, m=m)


# ── convenience: DayParams from an instance route ────────────────────────────

def params_from_route(route, D, dem, Q, scale, speed_kmh=25.0,
                      cv=0.35, **kw) -> DayParams:
    """Build a DayParams from repo data structures (parse_dethloff
    output + one planned route). Distances D are in 0.1 m units when
    scale=10 (city instances) — converted via scale to km, then to
    hours at speed_kmh."""
    r = np.asarray(route)
    dbar = dem[r, 0].astype(float)
    pbar = dem[r, 1].astype(float)
    seq = np.concatenate([[0], r])
    dist_km = np.array([D[seq[i], seq[i + 1]] for i in range(len(r))],
                       float) / (scale * 1000.0)
    tau = dist_km / speed_kmh
    mu_g = pbar - dbar
    sig_g = cv * np.sqrt(pbar ** 2 + dbar ** 2)
    sig_g = np.maximum(sig_g, 1e-6)
    B = float(Q - dbar.sum())
    return DayParams(tau=tau, mu_g=mu_g, sig_g=sig_g, B=B, **kw)
