"""Mutation-testbed experiment: calibrate on held-out families, then
certify pools of faithful/mutant candidates on the DISJOINT evaluation
families under three routing policies.

Usage:  python3 run_experiment.py [n_reps]     (default 60)

Outputs results/calibration.json and results/mutation_experiment.json
plus a human-readable results/mutation_experiment.txt.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys
import time

from eclair.calibrate import calibrate
from eclair.certify import Bets, run_certification
from eclair.mutations import make_faithful, make_mutant
from eclair.problems import EVAL_SPECS

ALPHA = 0.05
POOL_K = 6
SEED = 20260807
BUDGET_PROBE_EQUIV = 80          # budget = this many mean-cost probes


def build_pool(spec, rng):
    pool, labels = [], []
    n_faithful = 0
    for _ in range(POOL_K):
        if rng.random() < 0.5:
            pool.append(make_faithful(spec))
            labels.append(True)
            n_faithful += 1
        else:
            m = make_mutant(spec, rng)
            if m is None:                    # rare: fall back to faithful
                pool.append(make_faithful(spec))
                labels.append(True)
                n_faithful += 1
            else:
                pool.append(m)
                labels.append(False)
    return pool, labels


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    rng = random.Random(SEED)
    t0 = time.perf_counter()
    print("calibrating on held-out families ...", flush=True)
    calib = calibrate(rng)
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    for t in ("A", "B", "C"):
        c = calib[t]
        print(f"  tier {t}: p0_hat={c['p0_hat']:.4f} p0_bar={c['p0_bar']:.4f} "
              f"p1_hat={c['p1_hat']:.4f} p1_bet={c['p1_bet']:.4f} "
              f"cost={c['mean_cost']*1000:.1f}ms "
              f"(h0 {c['h0_alarms']}/{c['h0_n']}, h1 {c['h1_alarms']}/{c['h1_n']})",
              flush=True)
    bets = Bets(calib)
    mean_cost = statistics.fmean(bets.cost.values())
    budget = BUDGET_PROBE_EQUIV * mean_cost
    print(f"calibration took {time.perf_counter()-t0:.0f}s; "
          f"per-rep budget = {budget*1000:.0f}ms solver time", flush=True)

    results = {}
    for policy in ("kelly", "round_robin", "cost_blind"):
        prng = random.Random(SEED + 1)          # same pools across policies
        n_faith = n_faith_rej = n_mut = n_mut_rej = 0
        kill_costs, pick_ok, pick_n, abstain = [], 0, 0, 0
        ebh_false = ebh_total = 0
        for rep in range(n_reps):
            spec = EVAL_SPECS[rep % len(EVAL_SPECS)]
            pool, labels = build_pool(spec, prng)
            res = run_certification(pool, bets, policy, budget, ALPHA, prng)
            for st, faithful in zip(res["states"], labels):
                if faithful:
                    n_faith += 1
                    n_faith_rej += st.rejected
                else:
                    n_mut += 1
                    if st.rejected:
                        n_mut_rej += 1
                        kill_costs.append(st.spent)
            for i in res["ebh_rejected"]:
                ebh_total += 1
                ebh_false += labels[i]
            if res["abstained"]:
                abstain += 1
            elif any(labels):
                pick_n += 1
                idx = res["states"].index(res["pick"])
                pick_ok += labels[idx]
        results[policy] = {
            "false_rej": n_faith_rej / max(n_faith, 1), "n_faith": n_faith,
            "detect": n_mut_rej / max(n_mut, 1), "n_mut": n_mut,
            "mean_kill_cost_ms": 1000 * statistics.fmean(kill_costs)
            if kill_costs else None,
            "pick_accuracy": pick_ok / max(pick_n, 1), "n_picks": pick_n,
            "abstain_rate": abstain / n_reps,
            "ebh_fdr": ebh_false / max(ebh_total, 1),
            "ebh_rejections": ebh_total,
        }
        r = results[policy]
        print(f"{policy:<12} false-rej={r['false_rej']:.4f} (n={n_faith})  "
              f"detect={r['detect']:.4f} (n={n_mut})  "
              f"kill-cost={r['mean_kill_cost_ms'] and round(r['mean_kill_cost_ms'])}ms  "
              f"pick-acc={r['pick_accuracy']:.3f}  eBH-FDR={r['ebh_fdr']:.4f}",
              flush=True)

    payload = {"alpha": ALPHA, "pool_k": POOL_K, "n_reps": n_reps,
               "budget_s": budget, "seed": SEED,
               "eval_families": [s.name for s in EVAL_SPECS],
               "results": results}
    (out_dir / "mutation_experiment.json").write_text(
        json.dumps(payload, indent=2))

    lines = [
        "ECLAIR mutation-testbed experiment (real CP-SAT probes)",
        f"alpha={ALPHA} pool_k={POOL_K} reps/policy={n_reps} "
        f"budget={budget*1000:.0f}ms/rep seed={SEED}",
        f"calibration families: knapsack_conflicts, coloring | "
        f"eval families: {', '.join(s.name for s in EVAL_SPECS)}",
        "",
        f"{'policy':<12}{'false-rej':>10}{'detect':>8}{'kill-ms':>9}"
        f"{'pick-acc':>9}{'eBH-FDR':>9}{'abstain':>9}",
    ]
    for pol, r in results.items():
        km = f"{r['mean_kill_cost_ms']:.0f}" if r["mean_kill_cost_ms"] else "-"
        lines.append(f"{pol:<12}{r['false_rej']:>10.4f}{r['detect']:>8.4f}"
                     f"{km:>9}{r['pick_accuracy']:>9.3f}{r['ebh_fdr']:>9.4f}"
                     f"{r['abstain_rate']:>9.3f}")
    lines.append("")
    lines.append(f"total wall time: {time.perf_counter()-t0:.0f}s")
    (out_dir / "mutation_experiment.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
