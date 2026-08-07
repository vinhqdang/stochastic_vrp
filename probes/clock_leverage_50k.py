#!/usr/bin/env python3
"""clock_leverage_50k.py — precondition probe for candidate #9 (KAIROS).

Question: does CLOCK-CONDITIONING have decision leverage at metro scale
(50 km instances) that it lacks at last-mile scale (6 km instances)?
BATON's backward induction conditions continuation values on (stop,
load) only — it is clock-blind. That is harmless if a route's execution
window is short relative to the diurnal congestion cycle, and harmful
if routes cross congestion regimes mid-execution.

Diurnal curve: the MEASURED NYC link-level congestion multipliers
(data_sources/out/nyc_cells_*.csv, median multiplier by hour-of-day,
dry cells), normalized to its daily minimum — i.e. the real shape, not
a synthetic peak model. Base speed 25 km/h, 3 min service per stop.

Metrics per instance (routes from the same solve_fast Det plans the
evaluation uses):
  dur_h      mean route duration departing 08:00 (clock-aware)
  disp%      dispatch-timing leverage: (max-min)/min of route duration
             over departure hours 06:00..18:00 — what choosing WHEN to
             dispatch is worth
  cross%     share of routes whose within-execution multiplier range
             exceeds 20% — mid-route regime crossing
  cont_err%  mean relative error of a CLOCK-BLIND remaining-time
             estimate (multiplier frozen at the current clock) vs the
             clock-aware one, averaged over stops — the bias a
             clock-blind continuation value carries into every
             mid-route recourse comparison

Run: .venv313/bin/python probes/clock_leverage_50k.py
"""

import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "svrpspd_wdro"))
sys.path.insert(0, str(ROOT / "svrpspd_wdro" / "scripts"))

from dethloff_runner import DetGate, parse_dethloff, solve_fast  # noqa: E402

BASE_SPEED = 25.0        # km/h free-flow-ish urban average
SERVICE_H = 0.05         # 3 min per stop
DEP_HOURS = range(6, 19)


def measured_diurnal():
    """Median congestion multiplier by hour (dry), min-normalized."""
    path = ROOT / "data_sources" / "out" / "nyc_cells_2025-01-01_2025-01-07.csv"
    curve = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["wet"] == "False":
                curve[int(row["hour"])] = float(row["p50"])
    m = np.array([curve[h] for h in range(24)])
    return m / m.min()


M = measured_diurnal()


def mult(clock):
    return float(M[int(clock) % 24])


def route_times(route, D, dep):
    """Traverse depot->stops->depot from clock=dep; return (duration,
    multipliers seen, remaining clock-aware time at each stop,
    remaining clock-frozen time at each stop)."""
    seq = [0] + list(route) + [0]
    clock, seen = dep, []
    arrive = []                                  # clock at each stop
    for a, b in zip(seq, seq[1:]):
        m = mult(clock)
        seen.append(m)
        clock += (D[a][b] / BASE_SPEED) * m + (SERVICE_H if b != 0 else 0)
        arrive.append(clock)
    duration = clock - dep

    cont_true, cont_frozen = [], []
    for k in range(1, len(seq) - 1):             # from each customer stop
        c = arrive[k - 1]
        t_true, cl = 0.0, c
        for a, b in zip(seq[k:], seq[k + 1:]):
            t_true += (D[a][b] / BASE_SPEED) * mult(cl) + \
                (SERVICE_H if b != 0 else 0)
            cl = c + t_true
        m0 = mult(c)
        t_frozen = sum((D[a][b] / BASE_SPEED) * m0 +
                       (SERVICE_H if b != 0 else 0)
                       for a, b in zip(seq[k:], seq[k + 1:]))
        cont_true.append(t_true)
        cont_frozen.append(t_frozen)
    return duration, seen, np.array(cont_true), np.array(cont_frozen)


def probe(inst_path):
    D_raw, dem, Q, n, scale = parse_dethloff(inst_path)
    dbar, pbar = dem[:, 0], dem[:, 1]
    gate = DetGate(Q, dbar, pbar)
    routes = [r for r in solve_fast(np.asarray(D_raw), gate, n) if r]
    D = np.asarray(D_raw, dtype=float) / scale       # 0.1 m units -> km
    name = pathlib.Path(inst_path).stem
    durs, disp, crossed, errs = [], [], 0, []
    for r in routes:
        d8, seen, ct, cf = route_times(r, D, 8.0)
        durs.append(d8)
        span = [route_times(r, D, h)[0] for h in DEP_HOURS]
        disp.append((max(span) - min(span)) / min(span))
        if max(seen) / min(seen) > 1.2:
            crossed += 1
        if len(ct):
            errs.append(np.mean(np.abs(cf - ct) / np.maximum(ct, 1e-9)))
    return dict(name=name, K=len(routes),
                dur_h=float(np.mean(durs)),
                disp=100 * float(np.mean(disp)),
                cross=100 * crossed / len(routes),
                cont_err=100 * float(np.mean(errs)))


def main():
    print(f"measured diurnal shape: min=1.00 at "
          f"{int(np.argmin(M))}:00, max={M.max():.2f} at "
          f"{int(np.argmax(M))}:00\n")
    insts = [
        ROOT / "svrpspd_wdro/data/City/HCMC-100-1.vrpspd",
        ROOT / "svrpspd_wdro/data/City/HCMC-400-1.vrpspd",
        ROOT / "svrpspd_wdro/data/City50K/HCMC50K-100-1.vrpspd",
        ROOT / "svrpspd_wdro/data/City50K/HCMC50K-400-1.vrpspd",
        ROOT / "svrpspd_wdro/data/City50K/HANOI50K-400-1.vrpspd",
    ]
    lines = [f"{'instance':<18}{'K':>4}{'dur_h':>8}{'disp%':>8}"
             f"{'cross%':>8}{'cont_err%':>10}"]
    for p in insts:
        r = probe(p)
        lines.append(f"{r['name']:<18}{r['K']:>4}{r['dur_h']:>8.2f}"
                     f"{r['disp']:>8.1f}{r['cross']:>8.1f}"
                     f"{r['cont_err']:>10.1f}")
    out = "\n".join(lines)
    print(out)
    (pathlib.Path(__file__).parent / "clock_leverage_results.txt").write_text(
        __doc__.split("Metrics")[0] + out + "\n")


if __name__ == "__main__":
    main()
