"""The money plot: empirical false-rejection rate of FAITHFUL
candidates vs the nominal level alpha, plus detection, on the mutation
testbed (Kelly routing, same pools per alpha via a fixed seed).

Usage: python3 run_alpha_sweep.py [n_reps]      (default 100)
Needs results/calibration.json. Writes results/alpha_sweep.{json,txt}.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

from eclair.certify import Bets, run_certification
from run_experiment import BUDGET_PROBE_EQUIV, SEED, build_pool
from eclair.problems import EVAL_SPECS

ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.20]


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_dir = pathlib.Path(__file__).parent / "results"
    bets = Bets(json.loads((out_dir / "calibration.json").read_text()))
    budget = BUDGET_PROBE_EQUIV * statistics.fmean(bets.cost.values())

    rows = []
    for alpha in ALPHAS:
        prng = random.Random(SEED + 1)          # identical pools per alpha
        n_f = f_rej = n_m = m_rej = 0
        for rep in range(n_reps):
            spec = EVAL_SPECS[rep % len(EVAL_SPECS)]
            pool, labels = build_pool(spec, prng)
            res = run_certification(pool, bets, "kelly", budget, alpha, prng)
            for st, faithful in zip(res["states"], labels):
                if faithful:
                    n_f += 1
                    f_rej += st.rejected
                else:
                    n_m += 1
                    m_rej += st.rejected
        rows.append(dict(alpha=alpha, false_rej=f_rej / n_f, n_faith=n_f,
                         detect=m_rej / n_m, n_mut=n_m))
        print(f"alpha={alpha:<5} false-rej={rows[-1]['false_rej']:.4f} "
              f"(n={n_f})  detect={rows[-1]['detect']:.4f} (n={n_m})",
              flush=True)

    (out_dir / "alpha_sweep.json").write_text(json.dumps(
        dict(n_reps=n_reps, seed=SEED, policy="kelly", rows=rows), indent=2))
    lines = ["ECLAIR alpha sweep (mutation testbed, kelly routing)",
             f"reps/alpha={n_reps} seed={SEED}", "",
             f"{'alpha':>6}{'false-rej':>11}{'nominal':>9}{'detect':>8}"]
    for r in rows:
        lines.append(f"{r['alpha']:>6}{r['false_rej']:>11.4f}"
                     f"{r['alpha']:>9}{r['detect']:>8.4f}")
    lines.append("\nvalidity requires false-rej <= alpha on EVERY row;")
    lines.append("conservative calibration should keep it far below.")
    (out_dir / "alpha_sweep.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
