"""eval_matched.py — matched-instance comparison of the trained DGTA-RL
policy against TEMPO-triggered re-optimization and a never-replan
anchor, on held-out instances, under TEMPO's OWN travel-time null model
and drift convention (ev.world / ev.scenarios), not DGTA-RL's native
training-time Gamma process.

Why a different travel-time process at evaluation than at training:
DGTA-RL is trained faithfully to Chen et al. (2025)'s own recipe (Eq. 3
Gamma realization, ev/dgta/env.py), matching how the original paper
trains it. Every OTHER baseline in Section 5 (CUSUM, Page-Hinkley, the
e-detectors, rolling-opt, ...) is likewise built/calibrated on its own
native convention and then evaluated on TEMPO's shared scenario grid
(ev/world.py's lognormal multiplicative travel model + ev/scenarios.py's
DriftSpec). This script gives DGTA-RL the same treatment: train under
its own paper's assumptions, evaluate — like everything else compared
to TEMPO — on TEMPO's matched-instance grid, so the reported numbers
are a genuine apples-to-apples "realized tour cost" comparison and not
an artifact of DGTA-RL being tested on an easier or differently-shaped
distribution than the others.

The drift is the SPATIAL zonal congestion pocket of Section 2 / Figures
3-4 (ev.world.DriftSpec kind="traffic_zone"), TRANSIENT rather than
ever-growing (ev.scenarios' traffic_transient convention: it grows,
then fully clears after a fixed duration) -- not a uniform step
multiplier. This matters for the comparison, not just for fidelity to
the rest of the paper, for two structural reasons: (1) a uniform
slowdown applied to every edge equally cannot be exploited by
resequencing at all (scaling an entire distance matrix by one constant
never changes which tour order is shortest), so a spatial, asymmetric
pocket is the only drift shape a resequencing policy can gain anything
from; (2) an ever-growing (never-clearing) pocket still limits the
achievable gain to reordering among the customers not yet caught by
it, since once a location falls inside a monotonically growing radius
it never escapes -- a TRANSIENT pocket additionally rewards *waiting
out* the jam (deferring the affected customers until after it clears),
which is exactly the value a statistically-triggered, evidence-gated
re-sequencer can capture that a pre-committed static tour cannot.

Three policies, single travel-time channel (DGTA-RL has no demand/
capacity dimension, so this isolates exactly the channel it models):

  never_replan      fixed NN+2opt tour built under EXPECTED (t=0,
                    non-drifted) travel times, executed to completion.
  tempo_resequence  same initial tour; the real TempoMonitor
                    (ev.eprocess.TempoMonitor, alpha=0.05, fed travel
                    events only — every other channel stays silent at
                    logE=0) watches the standardized log-travel-time
                    residual to decide WHEN to fire (a scalar
                    statistic, blind to the pocket's shape, exactly as
                    real). On alarm, the paired replanner
                    (ev.replan.resequence_nn2opt, Section 4.6) re-solves
                    the remaining customers using the CURRENTLY
                    OBSERVABLE pocket extent (a live congestion feed, as
                    Figure 3's "colored by realized congestion" panel
                    depicts) to reprice remaining legs, then the monitor
                    resets (reset-on-alarm, Proposition 4's protocol).
  dgta_rl           the trained DGTA policy's greedy rollout, replaying
                    the model's step_logits against the SAME externally
                    supplied realized-time draws (not DTSPEnv's own
                    Gamma sampler) for a fair per-instance comparison;
                    it has no explicit congestion-zone input, only
                    coordinates and realized arrival times, so any
                    reaction to the pocket must be learned implicitly.

Usage:
    python -m ev.dgta.eval_matched --weights results/dgta/dgta_weights.pt \
        --n-instances 30 --n-days 5 --out results/dgta/matched_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_WDRO = _HERE.parent.parent
sys.path.insert(0, str(_WDRO))

from ev.dgta.model import DGTA                       # noqa: E402
from ev.dgta.env import random_instance, travel_hours  # noqa: E402
from ev.eprocess import TempoMonitor                  # noqa: E402
from ev.replan import resequence_nn2opt               # noqa: E402

SIG_T = 0.25
T0 = 8.0
DAY_HOURS = 8.0
N_TIME = 8
SPEED = 0.35


def zone_radius(clock, t_star, t_clear, radius0, spread):
    """Grows linearly from t_star, then fully clears at t_clear (the
    traffic_transient convention of ev.scenarios) -- not ever-growing."""
    if clock < t_star or clock >= t_clear:
        return -1.0   # no pocket (not yet, or already cleared)
    return radius0 + spread * (clock - t_star)


def in_zone(xy_dest, center, clock, t_star, t_clear, radius0, spread):
    if center is None:
        return False
    r = zone_radius(clock, t_star, t_clear, radius0, spread)
    if r < 0:
        return False
    dx = xy_dest[0] - center[0]
    dy = xy_dest[1] - center[1]
    return (dx * dx + dy * dy) ** 0.5 <= r


def realize_leg(tau, jammed, magnitude, rng):
    """TEMPO's own travel-time null + drift model (ev/world.py, Eq. 1):
    T = tau * exp(eps), eps ~ N(-sig^2/2 + shift, sig^2), with shift =
    log(magnitude) if this leg's destination is currently inside the
    congestion pocket, else 0 (no diurnal multiplier here -- a single,
    isolated travel-time channel is the object of this comparison, and
    synthetic tours run well past one nominal 8-hour day, so a
    time-of-day curve would just add an unrelated confound). Returns
    (T, z), z the standardized residual under P0 (jammed=False)."""
    mu_log = np.log(max(tau, 1e-9)) - 0.5 * SIG_T ** 2
    shift = np.log(magnitude) if jammed else 0.0
    logT = rng.normal(mu_log + shift, SIG_T)
    T = float(np.exp(logT))
    z = float((logT - mu_log) / SIG_T)
    return T, z


def t_idx_of(clock):
    frac = np.clip((clock - T0) / DAY_HOURS, 0.0, 0.999)
    return int(frac * N_TIME)


def run_never_replan(coords, base_hours, order0, rng, pocket):
    center, t_star, t_clear, radius0, spread, magnitude = pocket
    clock = T0
    total = 0.0
    cur = 0
    for nxt in order0:
        jammed = in_zone(coords[nxt], center, clock, t_star, t_clear,
                         radius0, spread)
        T, _ = realize_leg(base_hours[cur, nxt], jammed, magnitude, rng)
        total += T
        clock += T
        cur = nxt
    jammed = in_zone(coords[0], center, clock, t_star, t_clear, radius0,
                     spread)
    T, _ = realize_leg(base_hours[cur, 0], jammed, magnitude, rng)
    total += T
    return total


def run_tempo_resequence(coords, base_hours, order0, n, rng, pocket,
                         alpha=0.05):
    center, t_star, t_clear, radius0, spread, magnitude = pocket
    clock = T0
    total = 0.0
    cur = 0
    order = list(order0)
    monitor = TempoMonitor(alpha=alpha)
    served = 0
    while order:
        nxt = order.pop(0)
        jammed = in_zone(coords[nxt], center, clock, t_star, t_clear,
                         radius0, spread)
        T, z = realize_leg(base_hours[cur, nxt], jammed, magnitude, rng)
        total += T
        clock += T
        served += 1
        cur = nxt
        rem_frac = (n - served) / n
        fired = monitor.update(dict(channel="travel", z=z,
                                    ctx=dict(rem_frac=rem_frac)))
        if fired and order:
            # Reprice remaining legs using the CURRENTLY observable
            # pocket extent (a live congestion feed), not the scalar
            # e-process itself -- the monitor only decides WHEN.
            r_now = zone_radius(clock, t_star, t_clear, radius0, spread)
            adj = base_hours.copy()
            if r_now >= 0:
                for c in order:
                    dx = coords[c][0] - center[0]
                    dy = coords[c][1] - center[1]
                    if (dx * dx + dy * dy) ** 0.5 <= r_now:
                        adj[:, c] *= magnitude
                        adj[c, :] *= magnitude
            order = resequence_nn2opt(cur, order, adj)
            monitor = TempoMonitor(alpha=alpha)   # reset on alarm
    jammed = in_zone(coords[0], center, clock, t_star, t_clear, radius0,
                     spread)
    T, _ = realize_leg(base_hours[cur, 0], jammed, magnitude, rng)
    total += T
    return total


def run_dgta(model, coords, base_hours, rng, pocket, device="cpu"):
    center, t_star, t_clear, radius0, spread, magnitude = pocket
    n = base_hours.shape[0] - 1
    coords_t = torch.tensor(coords[1:], dtype=torch.float32,
                            device=device).unsqueeze(0)   # (1,n,2)
    with torch.no_grad():
        Hbar = model.encode(coords_t)                      # (1,n,T,D)
        clock = T0
        cur = 0
        visited = np.zeros(n + 1, dtype=bool)
        visited[0] = True
        arrival = np.zeros(n + 1, dtype=np.float32)
        total = 0.0
        for _ in range(n):
            ti = t_idx_of(clock)
            cur_node_model = cur - 1 if cur > 0 else 0
            t_idx_t = torch.tensor([min(ti, N_TIME - 1)], device=device)
            arrival_t = torch.tensor(arrival[1:], dtype=torch.float32,
                                     device=device).unsqueeze(0)
            clock_t = torch.tensor([clock], dtype=torch.float32,
                                   device=device)
            visited_t = torch.tensor(visited[1:], device=device).unsqueeze(0)
            logits = model.step_logits(
                Hbar, torch.tensor([cur_node_model], device=device),
                t_idx_t, arrival_t, clock_t, visited_t)
            choice = int(logits.argmax(dim=-1).item())
            nxt = choice + 1
            jammed = in_zone(coords[nxt], center, clock, t_star, t_clear,
                             radius0, spread)
            T, _ = realize_leg(base_hours[cur, nxt], jammed, magnitude, rng)
            total += T
            clock += T
            arrival[nxt] = clock
            visited[nxt] = True
            cur = nxt
        jammed = in_zone(coords[0], center, clock, t_star, t_clear, radius0,
                         spread)
        T, _ = realize_leg(base_hours[cur, 0], jammed, magnitude, rng)
        total += T
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default="results/dgta/dgta_weights.pt")
    ap.add_argument("--n-instances", type=int, default=30)
    ap.add_argument("--n-days", type=int, default=5)
    ap.add_argument("--n-customers", type=int, default=20)
    ap.add_argument("--t-star-offset", type=float, default=2.0)
    ap.add_argument("--clear-duration", type=float, default=6.0)
    ap.add_argument("--magnitude", type=float, default=2.5)
    ap.add_argument("--radius0", type=float, default=0.25)
    ap.add_argument("--spread", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=str,
                    default="results/dgta/matched_eval.json")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = DGTA(n_time=ckpt.get("n_time", N_TIME))
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = args.n_customers

    results = {"never_replan": {"null": [], "traffic_transient": []},
              "tempo_resequence": {"null": [], "traffic_transient": []},
              "dgta_rl": {"null": [], "traffic_transient": []}}

    for i in range(args.n_instances):
        coords_t = random_instance(1, n, device="cpu",
                                   seed=args.seed + i).squeeze(0).numpy()
        d = coords_t[:, None, :] - coords_t[None, :, :]
        dist = np.linalg.norm(d, axis=-1)
        base_hours = travel_hours(torch.tensor(dist)).numpy()
        order0 = resequence_nn2opt(0, list(range(1, n + 1)), base_hours)

        for day in range(args.n_days):
            rng = np.random.default_rng(args.seed * 1000 + i * 100 + day)
            center = rng.uniform(0.0, 1.0, size=2)
            t_star = T0 + args.t_star_offset
            for scenario, mag in [
                ("null", 1.0),
                ("traffic_transient", args.magnitude),
            ]:
                t_clear = t_star + args.clear_duration if mag > 1.0 else -1.0
                pocket = (center, t_star, t_clear, args.radius0,
                         args.spread, mag)
                rng_nr = np.random.default_rng(rng.integers(1 << 31))
                rng_tr = np.random.default_rng(rng.integers(1 << 31))
                rng_dg = np.random.default_rng(rng.integers(1 << 31))
                c_nr = run_never_replan(coords_t, base_hours, order0,
                                        rng_nr, pocket)
                c_tr = run_tempo_resequence(coords_t, base_hours, order0, n,
                                            rng_tr, pocket)
                c_dg = run_dgta(model, coords_t, base_hours, rng_dg, pocket)
                results["never_replan"][scenario].append(c_nr)
                results["tempo_resequence"][scenario].append(c_tr)
                results["dgta_rl"][scenario].append(c_dg)
        print(f"instance {i+1}/{args.n_instances} done", flush=True)

    summary = {}
    for policy, by_scn in results.items():
        summary[policy] = {}
        for scn, vals in by_scn.items():
            arr = np.array(vals)
            summary[policy][scn] = dict(mean=float(arr.mean()),
                                        std=float(arr.std()),
                                        n=int(arr.size))

    never = summary["never_replan"]["traffic_transient"]["mean"]
    for policy in ("tempo_resequence", "dgta_rl"):
        m = summary[policy]["traffic_transient"]["mean"]
        summary[policy]["traffic_transient"]["saving_vs_never_pct"] = \
            100.0 * (never - m) / never

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), results=summary), f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
