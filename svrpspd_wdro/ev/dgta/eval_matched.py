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

Three policies, single travel-time channel (DGTA-RL has no demand/
capacity dimension, so this isolates exactly the channel it models):

  never_replan      fixed NN+2opt tour built under EXPECTED (t=0,
                    non-drifted) travel times, executed to completion.
  tempo_resequence  same initial tour; the real TempoMonitor
                    (ev.eprocess.TempoMonitor, alpha=0.05, fed travel
                    events only — every other channel stays silent at
                    logE=0) watches the standardized log-travel-time
                    residual; on alarm the remaining customers are
                    re-sequenced (ev.replan.resequence_nn2opt) under a
                    distance matrix rescaled by the monitor's own EWMA
                    tilt estimate, and the monitor resets (reset-on-
                    alarm, matching Proposition 4's protocol).
  dgta_rl           the trained DGTA policy's greedy rollout, replaying
                    the model's step_logits against the SAME externally
                    supplied realized-time draws (not DTSPEnv's own
                    Gamma sampler) for a fair per-instance comparison.

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
from ev.world import diurnal_mult, DayParams          # noqa: E402
from ev.eprocess import TempoMonitor                  # noqa: E402
from ev.replan import resequence_nn2opt               # noqa: E402

DIURNAL = DayParams(tau=np.zeros(1), mu_g=np.zeros(1), sig_g=np.ones(1),
                    B=1.0).diurnal
SIG_T = 0.25
T0 = 8.0
DAY_HOURS = 8.0
N_TIME = 8
SPEED = 0.35


def realize_leg(tau, clock, rho, magnitude, rng):
    """TEMPO's own travel-time null + drift model (ev/world.py, Eq. 1):
    T = tau * m(clock) * exp(eps), eps ~ N(-sig^2/2 + rho*log(mag), sig^2).
    Returns (T, z) with z the standardized residual under P0 (rho=0)."""
    m = diurnal_mult(clock, DIURNAL)
    mu_log = np.log(max(tau * m, 1e-9)) - 0.5 * SIG_T ** 2
    shift = rho * np.log(magnitude)
    logT = rng.normal(mu_log + shift, SIG_T)
    T = float(np.exp(logT))
    z = float((logT - mu_log) / SIG_T)
    return T, z


def t_idx_of(clock):
    frac = np.clip((clock - T0) / DAY_HOURS, 0.0, 0.999)
    return int(frac * N_TIME)


def rho_of(clock, t_star, profile="step", ramp_h=2.0):
    if clock < t_star:
        return 0.0
    if profile == "ramp":
        return min(1.0, (clock - t_star) / max(ramp_h, 1e-9))
    return 1.0


def run_never_replan(base_hours, order0, rng, t_star, magnitude, profile):
    clock = T0
    total = 0.0
    cur = 0
    for nxt in order0:
        rho = rho_of(clock, t_star, profile)
        T, _ = realize_leg(base_hours[cur, nxt], clock, rho, magnitude, rng)
        total += T
        clock += T
        cur = nxt
    rho = rho_of(clock, t_star, profile)
    T, _ = realize_leg(base_hours[cur, 0], clock, rho, magnitude, rng)
    total += T
    return total


def run_tempo_resequence(base_hours, order0, n, rng, t_star, magnitude,
                         profile, alpha=0.05):
    clock = T0
    total = 0.0
    cur = 0
    order = list(order0)
    monitor = TempoMonitor(alpha=alpha)
    served = 0
    while order:
        nxt = order.pop(0)
        rho = rho_of(clock, t_star, profile)
        T, z = realize_leg(base_hours[cur, nxt], clock, rho, magnitude, rng)
        total += T
        clock += T
        served += 1
        cur = nxt
        rem_frac = (n - served) / n
        fired = monitor.update(dict(channel="travel", z=z,
                                    ctx=dict(rem_frac=rem_frac)))
        if fired and order:
            tilt = float(np.clip(monitor.ewma["travel"], 0.0,
                                 monitor.theta_max))
            adj = base_hours * np.exp(tilt * SIG_T)
            order = resequence_nn2opt(cur, order, adj)
            monitor = TempoMonitor(alpha=alpha)   # reset on alarm
    rho = rho_of(clock, t_star, profile)
    T, _ = realize_leg(base_hours[cur, 0], clock, rho, magnitude, rng)
    total += T
    return total


def run_dgta(model, coords, base_hours, rng, t_star, magnitude, profile,
            device="cpu"):
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
            rho = rho_of(clock, t_star, profile)
            T, _ = realize_leg(base_hours[cur, nxt], clock, rho, magnitude,
                               rng)
            total += T
            clock += T
            arrival[nxt] = clock
            visited[nxt] = True
            cur = nxt
        rho = rho_of(clock, t_star, profile)
        T, _ = realize_leg(base_hours[cur, 0], clock, rho, magnitude, rng)
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
    ap.add_argument("--magnitude", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=str,
                    default="results/dgta/matched_eval.json")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = DGTA(n_time=ckpt.get("n_time", N_TIME))
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = args.n_customers

    results = {"never_replan": {"null": [], "traffic_severe": []},
              "tempo_resequence": {"null": [], "traffic_severe": []},
              "dgta_rl": {"null": [], "traffic_severe": []}}

    for i in range(args.n_instances):
        coords_t = random_instance(1, n, device="cpu",
                                   seed=args.seed + i).squeeze(0).numpy()
        d = coords_t[:, None, :] - coords_t[None, :, :]
        dist = np.linalg.norm(d, axis=-1)
        base_hours = travel_hours(torch.tensor(dist)).numpy()
        order0 = resequence_nn2opt(0, list(range(1, n + 1)), base_hours)

        for day in range(args.n_days):
            rng = np.random.default_rng(args.seed * 1000 + i * 100 + day)
            for scenario, t_star, mag in [
                ("null", 1e9, 1.0),
                ("traffic_severe", T0 + args.t_star_offset, args.magnitude),
            ]:
                rng_nr = np.random.default_rng(rng.integers(1 << 31))
                rng_tr = np.random.default_rng(rng.integers(1 << 31))
                rng_dg = np.random.default_rng(rng.integers(1 << 31))
                c_nr = run_never_replan(base_hours, order0, rng_nr, t_star,
                                        mag, "step")
                c_tr = run_tempo_resequence(base_hours, order0, n, rng_tr,
                                            t_star, mag, "step")
                c_dg = run_dgta(model, coords_t, base_hours, rng_dg, t_star,
                                mag, "step")
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

    never = summary["never_replan"]["traffic_severe"]["mean"]
    for policy in ("tempo_resequence", "dgta_rl"):
        m = summary[policy]["traffic_severe"]["mean"]
        summary[policy]["traffic_severe"]["saving_vs_never_pct"] = \
            100.0 * (never - m) / never

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), results=summary), f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
