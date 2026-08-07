"""Probe-tier ablation (PROJECT.md §6): what does each tier contribute?
Kelly routing, alpha=0.05, identical pools per configuration.

Usage: python3 run_tier_ablation.py [n_reps]    (default 100)
Needs results/calibration.json. Writes results/tier_ablation.{json,txt}.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

from eclair.certify import Bets, run_certification
from run_experiment import ALPHA, BUDGET_PROBE_EQUIV, SEED, build_pool
from eclair.problems import EVAL_SPECS

CONFIGS = [("A", "B", "C"), ("A", "B"), ("A", "C"), ("B", "C"),
           ("A",), ("C",)]


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_dir = pathlib.Path(__file__).parent / "results"
    bets = Bets(json.loads((out_dir / "calibration.json").read_text()))
    budget = BUDGET_PROBE_EQUIV * statistics.fmean(bets.cost.values())

    rows = []
    for tiers in CONFIGS:
        prng = random.Random(SEED + 1)
        n_f = f_rej = n_m = m_rej = 0
        kill = []
        for rep in range(n_reps):
            spec = EVAL_SPECS[rep % len(EVAL_SPECS)]
            pool, labels = build_pool(spec, prng)
            res = run_certification(pool, bets, "kelly", budget, ALPHA,
                                    prng, tiers=tiers)
            for st, faithful in zip(res["states"], labels):
                if faithful:
                    n_f += 1
                    f_rej += st.rejected
                else:
                    n_m += 1
                    if st.rejected:
                        m_rej += 1
                        kill.append(st.spent)
        rows.append(dict(tiers="".join(tiers), false_rej=f_rej / n_f,
                         detect=m_rej / n_m,
                         kill_ms=1000 * statistics.fmean(kill) if kill
                         else None))
        r = rows[-1]
        km = f"{r['kill_ms']:.0f}" if r["kill_ms"] else "-"
        print(f"tiers={r['tiers']:<4} false-rej={r['false_rej']:.4f}  "
              f"detect={r['detect']:.4f}  kill-cost={km}ms", flush=True)

    (out_dir / "tier_ablation.json").write_text(json.dumps(
        dict(n_reps=n_reps, alpha=ALPHA, seed=SEED, policy="kelly",
             rows=rows), indent=2))
    lines = ["ECLAIR probe-tier ablation (kelly, alpha=0.05)",
             f"reps/config={n_reps} seed={SEED}", "",
             f"{'tiers':>6}{'false-rej':>11}{'detect':>8}{'kill-ms':>9}"]
    for r in rows:
        km = f"{r['kill_ms']:.0f}" if r["kill_ms"] else "-"
        lines.append(f"{r['tiers']:>6}{r['false_rej']:>11.4f}"
                     f"{r['detect']:>8.4f}{km:>9}")
    (out_dir / "tier_ablation.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
