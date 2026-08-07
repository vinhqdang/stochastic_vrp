"""External baselines at matched budget (manuscript Sec. baselines):

  single_shot   accept the first candidate, no verification
  majority      self-consistency: solve all candidates on N_V shared
                mid instances, accept the one agreeing most often with
                the per-instance modal value (no rejection concept)
  fixed_C       valid fixed-sample test: n tier-C probes per candidate
                (n set by the shared budget), reject iff the alarm
                count crosses the exact binomial critical value at
                level alpha under p0_bar
  peek_C        the practitioner's temptation: the SAME binomial test
                applied after every probe at nominal alpha, stopping
                at the first crossing (no correction for optional
                stopping) - the textbook inflation demonstration
  conformal_C   split conformal: per-candidate alarm count in n fixed
                tier-C probes, conformal p-value against a faithful
                calibration sample, reject iff p <= alpha
  eclair        the full framework (Kelly, all tiers), same budget

Usage: python3 run_baselines.py [n_reps]        (default 100)
Writes results/baselines.{json,txt}.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import statistics
import sys

from eclair.certify import Bets, run_certification
from eclair.probes import solve_value, tier_c
from eclair.problems import EVAL_SPECS
from run_experiment import ALPHA, BUDGET_PROBE_EQUIV, SEED, build_pool

N_V = 8              # majority-vote instances
N_CAL = 200          # conformal calibration sample (faithful tier-C runs)


def binom_sf(k, n, p):
    """P(Bin(n,p) >= k)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def critical_k(n, p, alpha):
    """Smallest k with P(Bin(n,p) >= k) <= alpha."""
    for k in range(n + 1):
        if binom_sf(k, n, p) <= alpha:
            return k
    return n + 1


def run_probes_c(cand, pool, n, rng):
    """n tier-C probes; returns list of alarms and total cost."""
    alarms, cost = [], 0.0
    for _ in range(n):
        a, c = tier_c(cand, pool, rng)
        alarms.append(a)
        cost += c
    return alarms, cost


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_dir = pathlib.Path(__file__).parent / "results"
    calib = json.loads((out_dir / "calibration.json").read_text())
    bets = Bets(calib)
    budget = BUDGET_PROBE_EQUIV * statistics.fmean(bets.cost.values())
    # per-candidate tier-C probe count from the same budget
    n_c = max(4, int(budget / 6 / calib["C"]["mean_cost"]))
    k_crit = critical_k(n_c, calib["C"]["p0_bar"], ALPHA)

    # conformal calibration: faithful alarm-counts on held-out families
    from eclair.mutations import make_faithful, make_mutant
    from eclair.problems import CALIB_SPECS
    crng = random.Random(SEED + 7)
    cal_counts = []
    for i in range(N_CAL):
        spec = CALIB_SPECS[i % len(CALIB_SPECS)]
        faith = make_faithful(spec)
        others = [make_faithful(spec) if crng.random() < 0.5
                  else (make_mutant(spec, crng) or make_faithful(spec))
                  for _ in range(5)]
        alarms, _ = run_probes_c(faith, [faith] + others, n_c, crng)
        cal_counts.append(sum(alarms))
    cal_counts.sort()

    def conformal_p(count):
        ge = sum(1 for c in cal_counts if c >= count)
        return (1 + ge) / (N_CAL + 1)

    methods = ["eclair", "single_shot", "majority", "fixed_C", "peek_C",
               "conformal_C"]
    agg = {m: {"n_f": 0, "f_rej": 0, "n_m": 0, "m_rej": 0,
               "picks": 0, "pick_ok": 0} for m in methods}

    for rep in range(n_reps):
        spec = EVAL_SPECS[rep % len(EVAL_SPECS)]
        prng = random.Random(SEED + 100 + rep)
        pool, labels = build_pool(spec, prng)

        def score(m, rejected, pick_idx=None):
            for st_rej, faithful in zip(rejected, labels):
                if faithful:
                    agg[m]["n_f"] += 1
                    agg[m]["f_rej"] += st_rej
                else:
                    agg[m]["n_m"] += 1
                    agg[m]["m_rej"] += st_rej
            if pick_idx is not None and any(labels):
                agg[m]["picks"] += 1
                agg[m]["pick_ok"] += labels[pick_idx]

        # ECLAIR (fresh rng stream mirrors the baselines' independence)
        res = run_certification(pool, bets, "kelly", budget, ALPHA,
                                random.Random(SEED + 200 + rep))
        rej = [st.rejected for st in res["states"]]
        pick = (res["states"].index(res["pick"])
                if res["pick"] is not None else None)
        score("eclair", rej, pick)

        # single shot
        score("single_shot", [False] * len(pool), 0)

        # majority vote on N_V shared instances
        vrng = random.Random(SEED + 300 + rep)
        agree = [0] * len(pool)
        for _ in range(N_V):
            params = spec.gen(vrng, "mid")
            vals = [solve_value(c.build(params)) for c in pool]
            mode = max(set(vals), key=vals.count)
            for i, v in enumerate(vals):
                agree[i] += v == mode
        score("majority", [False] * len(pool),
              max(range(len(pool)), key=lambda i: agree[i]))

        # fixed-sample, peeking, conformal (shared probe streams)
        frng = random.Random(SEED + 400 + rep)
        rej_fix, rej_peek, rej_conf = [], [], []
        for cand in pool:
            alarms, _ = run_probes_c(cand, pool, n_c, frng)
            count = sum(alarms)
            rej_fix.append(count >= k_crit)
            peek = False
            run = 0
            for t, a in enumerate(alarms, 1):
                run += a
                if binom_sf(run, t, calib["C"]["p0_bar"]) <= ALPHA:
                    peek = True
                    break
            rej_peek.append(peek)
            rej_conf.append(conformal_p(count) <= ALPHA)
        surv = [i for i in range(len(pool)) if not rej_fix[i]]
        score("fixed_C", rej_fix, min(surv) if surv else None)
        score("peek_C", rej_peek, None)
        score("conformal_C", rej_conf, None)

    rows = {}
    for m in methods:
        a = agg[m]
        rows[m] = {
            "false_rej": a["f_rej"] / max(a["n_f"], 1),
            "detect": a["m_rej"] / max(a["n_m"], 1),
            "pick_acc": a["pick_ok"] / a["picks"] if a["picks"] else None,
            "n_f": a["n_f"], "n_m": a["n_m"],
        }
        pa = f"{rows[m]['pick_acc']:.3f}" if rows[m]["pick_acc"] is not None else "  -  "
        print(f"{m:<12} false-rej={rows[m]['false_rej']:.4f}  "
              f"detect={rows[m]['detect']:.4f}  pick-acc={pa}", flush=True)

    payload = {"alpha": ALPHA, "n_reps": n_reps, "n_c": n_c,
               "k_crit": k_crit, "seed": SEED, "rows": rows}
    (out_dir / "baselines.json").write_text(json.dumps(payload, indent=2))
    lines = [f"ECLAIR vs external baselines (matched budget, alpha={ALPHA}, "
             f"{n_reps} reps; tier-C fixed-sample n={n_c}, k_crit={k_crit})",
             "",
             f"{'method':<12}{'false-rej':>11}{'detect':>8}{'pick-acc':>10}"]
    for m in methods:
        r = rows[m]
        pa = f"{r['pick_acc']:.3f}" if r["pick_acc"] is not None else "-"
        lines.append(f"{m:<12}{r['false_rej']:>11.4f}{r['detect']:>8.4f}"
                     f"{pa:>10}")
    (out_dir / "baselines.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
