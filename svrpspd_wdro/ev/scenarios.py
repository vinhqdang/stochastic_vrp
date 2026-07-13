"""scenarios.py — the comprehensive named scenario grid for TEMPO.

Each entry maps a name to a factory `(t0) -> (drift_or_list, force_rain)`
where t0 is the route's departure clock. Grouped:

  nulls          plain on-model day, and the HOSTILE null: a rainy day
                 that the forecast fully anticipated — a monitor that
                 alarms on forecast rain has broken its conditional
                 null (classical detectors on raw lateness will).
  single-factor  each channel x {mild, severe} x {step onset} plus
                 shape variants: gradual ramp, transient window (the
                 jam that clears), late onset.
  compound       storm (traffic+dwell+accidents, ramped, rainy),
                 rush-crush (demand+dwell), black day (everything).

Magnitudes follow milestone-1 conventions: traffic = log-mean
multiplier, demand/dwell = +sd shift, accident/breakdown = rate ratio.
"""

from __future__ import annotations

from .world import DriftSpec


def _s(kind, mag, dt=2.0, **kw):
    return lambda t0: (DriftSpec(kind=kind, t_star=t0 + dt,
                                 magnitude=mag, **kw), None)


def _multi(specs_kw, rain=None, dt=2.0):
    def make(t0):
        return ([DriftSpec(t_star=t0 + kw.pop("dt", dt), **kw)
                 for kw in [dict(s) for s in specs_kw]], rain)
    return make


SCENARIOS = {
    # ── nulls ────────────────────────────────────────────────────────
    "null":            lambda t0: (DriftSpec(kind="none"), None),
    "null_rain":       lambda t0: (DriftSpec(kind="none"), True),

    # ── single factor, step onset ────────────────────────────────────
    "traffic_mild":    _s("traffic", 1.3),
    "traffic_severe":  _s("traffic", 2.0),
    "demand_mild":     _s("demand", 0.5),
    "demand_severe":   _s("demand", 1.5),
    "dwell_mild":      _s("dwell", 1.5),
    "dwell_severe":    _s("dwell", 4.0),
    "accident_mild":   _s("accident", 3.0),
    "accident_severe": _s("accident", 10.0),
    "breakdown":       _s("breakdown", 50.0),

    # ── shape variants ───────────────────────────────────────────────
    "traffic_ramp":    _s("traffic", 2.0, profile="ramp", ramp_h=2.0),
    "demand_ramp":     _s("demand", 1.2, profile="ramp", ramp_h=2.0),
    "traffic_transient": lambda t0: (DriftSpec(
        kind="traffic", t_star=t0 + 1.5, magnitude=2.2,
        profile="transient", t_clear=t0 + 3.5), None),
    "traffic_late":    _s("traffic", 1.8, dt=4.0),
    "demand_late":     _s("demand", 1.2, dt=4.0),

    # ── compound days ────────────────────────────────────────────────
    "storm": _multi([dict(kind="traffic", magnitude=1.6,
                          profile="ramp", ramp_h=1.0),
                     dict(kind="dwell", magnitude=2.0,
                          profile="ramp", ramp_h=1.0),
                     dict(kind="accident", magnitude=4.0)], rain=True),
    "rush_crush": _multi([dict(kind="demand", magnitude=0.8),
                          dict(kind="dwell", magnitude=2.0)]),
    "black_day": _multi([dict(kind="traffic", magnitude=1.8),
                         dict(kind="demand", magnitude=1.0),
                         dict(kind="dwell", magnitude=2.5),
                         dict(kind="accident", magnitude=6.0),
                         dict(kind="breakdown", magnitude=30.0)]),
}

NULL_NAMES = ("null", "null_rain")
