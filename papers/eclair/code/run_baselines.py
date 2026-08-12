"""External baselines, fair design (R1.4): identical candidate pools
across all methods (pool seed depends only on the replication), and a
like-for-like probe diet per comparison.

  eclair        full framework (Kelly, tiers A+B+C)
  eclair_C      the framework restricted to tier C - the like-for-like
                partner for the tier-C-only baselines below
  seq_same / fixed_same
                THE accounting comparison: one round-robin probe
                stream per candidate over A,B,C with recorded
                e-factors; the SAME stream is scored two ways -
                sequential (reject iff the running product ever
                reaches 1/alpha; Ville-valid) and fixed-n (reject iff
                the endpoint product reaches 1/alpha; Markov-valid).
                Identical probes, identical e-factors, identical
                threshold - only the accounting differs.
  single_shot   accept the first candidate, no verification
  majority      self-consistency vote over N_V shared instances
  fixed_C       binomial test on n tier-C probes at level alpha
  peek_C        the same binomial test applied after every probe
                (uncorrected optional stopping)
  conformal_C   split-conformal on the tier-C alarm count

All rates come with Wilson 95% intervals.
Usage: python3 run_baselines.py [n_reps]        (default 300)
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
from eclair.probes import TIER_FNS, solve_value, tier_c
from eclair.problems import EVAL_SPECS
from run_experiment import ALPHA, BUDGET_PROBE_EQUIV, SEED, build_pool, wilson

N_V = 8              # majority-vote instances
N_CAL = 200          # conformal calibration sample


def binom_sf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def critical_k(n, p, alpha):
    for k in range(n + 1):
        if binom_sf(k, n, p) <= alpha:
            return k
    return n + 1


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out_dir = pathlib.Path(__file__).parent / "results"
    calib = json.loads((out_dir / "calibration.json").read_text())
    bets = Bets(calib)
    budget = BUDGET_PROBE_EQUIV * statistics.fmean(bets.cost.values())
    n_stream = max(6, BUDGET_PROBE_EQUIV // 6)      # probes/cand, same-stream
    n_c = max(4, int(budget / 6 / calib["C"]["mean_cost"]))
    k_crit = critical_k(n_c, calib["C"]["p0_bar"], ALPHA)
    log_thr = math.log(1 / ALPHA)

    # conformal calibration sample (faithful, held-out families)
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
        alarms = [tier_c(faith, [faith] + others, crng)[0]
                  for _ in range(n_c)]
        cal_counts.append(sum(alarms))

    def conformal_p(count):
        ge = sum(1 for c in cal_counts if c >= count)
        return (1 + ge) / (N_CAL + 1)

    methods = ["eclair", "eclair_C", "seq_same", "fixed_same",
               "single_shot", "majority", "fixed_C", "peek_C",
               "conformal_C"]
    agg = {m: {"n_f": 0, "f_rej": 0, "n_m": 0, "m_rej": 0,
               "picks": 0, "pick_ok": 0} for m in methods}

    seq_probes, fix_probes = [], []

    def score(m, labels, rejected, pick_idx=None):
        for r, faithful in zip(rejected, labels):
            if faithful:
                agg[m]["n_f"] += 1
                agg[m]["f_rej"] += r
            else:
                agg[m]["n_m"] += 1
                agg[m]["m_rej"] += r
        if pick_idx is not None and any(labels):
            agg[m]["picks"] += 1
            agg[m]["pick_ok"] += labels[pick_idx]

    for rep in range(n_reps):
        spec = EVAL_SPECS[rep % len(EVAL_SPECS)]
        pool, labels = build_pool(spec, random.Random(SEED * 100_000 + rep))

        # ECLAIR full and C-only (own probe streams)
        for m, tiers in (("eclair", ("A", "B", "C")), ("eclair_C", ("C",))):
            res = run_certification(
                pool, bets, "kelly", budget, ALPHA,
                random.Random(SEED + 200 + rep), tiers=tiers)
            rej = [st.rejected for st in res["states"]]
            pick = (res["states"].index(res["pick"])
                    if res["pick"] is not None else None)
            score(m, labels, rej, pick)

        # same-stream sequential vs fixed-n (shared probes + e-factors)
        srng = random.Random(SEED + 500 + rep)
        rej_seq, rej_fix = [], []
        for cand, faithful in zip(pool, labels):
            logE = 0.0
            running_max = -math.inf
            t_cross = None
            for t in range(n_stream):
                tier = ("A", "B", "C")[t % 3]
                try:
                    if tier == "C":
                        alarm, _ = tier_c(cand, pool, srng)
                    else:
                        alarm, _ = TIER_FNS[tier](cand, srng)
                except Exception:
                    logE = math.log(1e18)
                    running_max = logE
                    break
                if tier == "B":
                    if alarm:
                        logE = math.log(1e18)
                        running_max = logE
                        break
                else:
                    logE += bets.log_e(tier, alarm)
                running_max = max(running_max, logE)
                if t_cross is None and running_max >= log_thr:
                    t_cross = t + 1
            rej_seq.append(running_max >= log_thr)
            rej_fix.append(logE >= log_thr)
            if not faithful:
                seq_probes.append(t_cross if t_cross else n_stream)
                fix_probes.append(n_stream)
        score("seq_same", labels, rej_seq)
        score("fixed_same", labels, rej_fix)

        # single shot / majority (no rejection concept)
        score("single_shot", labels, [False] * len(pool), 0)
        vrng = random.Random(SEED + 300 + rep)
        agree = [0] * len(pool)
        for _ in range(N_V):
            params = spec.gen(vrng, "mid")
            vals = [solve_value(c.build(params)) for c in pool]
            mode = max(set(vals), key=vals.count)
            for i, v in enumerate(vals):
                agree[i] += v == mode
        score("majority", labels, [False] * len(pool),
              max(range(len(pool)), key=lambda i: agree[i]))

        # tier-C-only fixed / peeking / conformal (shared stream)
        frng = random.Random(SEED + 400 + rep)
        rj_f, rj_p, rj_c = [], [], []
        for cand in pool:
            alarms = [tier_c(cand, pool, frng)[0] for _ in range(n_c)]
            count = sum(alarms)
            rj_f.append(count >= k_crit)
            peek, run = False, 0
            for t, a in enumerate(alarms, 1):
                run += a
                if binom_sf(run, t, calib["C"]["p0_bar"]) <= ALPHA:
                    peek = True
                    break
            rj_p.append(peek)
            rj_c.append(conformal_p(count) <= ALPHA)
        surv = [i for i in range(len(pool)) if not rj_f[i]]
        score("fixed_C", labels, rj_f, min(surv) if surv else None)
        score("peek_C", labels, rj_p)
        score("conformal_C", labels, rj_c)
        if (rep + 1) % 50 == 0:
            print(f"rep {rep + 1}/{n_reps}", flush=True)

    rows = {}
    for m in methods:
        a = agg[m]
        fr, dt = a["f_rej"] / max(a["n_f"], 1), a["m_rej"] / max(a["n_m"], 1)
        rows[m] = {"false_rej": fr, "false_rej_ci": wilson(a["f_rej"], a["n_f"]),
                   "detect": dt, "detect_ci": wilson(a["m_rej"], a["n_m"]),
                   "pick_acc": a["pick_ok"] / a["picks"] if a["picks"] else None,
                   "n_f": a["n_f"], "n_m": a["n_m"]}
        lo, hi = rows[m]["false_rej_ci"]
        dlo, dhi = rows[m]["detect_ci"]
        pa = (f"{rows[m]['pick_acc']:.3f}"
              if rows[m]["pick_acc"] is not None else "  -  ")
        print(f"{m:<12} false-rej={fr:.4f} [{lo:.3f},{hi:.3f}]  "
              f"detect={dt:.4f} [{dlo:.3f},{dhi:.3f}]  pick={pa}", flush=True)

    payload = {"alpha": ALPHA, "n_reps": n_reps, "n_stream": n_stream,
               "n_c": n_c, "k_crit": k_crit, "seed": SEED, "rows": rows,
               "seq_mean_probes_mut": statistics.fmean(seq_probes),
               "fix_mean_probes_mut": statistics.fmean(fix_probes)}
    print(f"same-stream probes per mutant: seq={payload['seq_mean_probes_mut']:.1f} "
          f"vs fixed={payload['fix_mean_probes_mut']:.1f}")
    (out_dir / "baselines.json").write_text(json.dumps(payload, indent=2))
    lines = [f"ECLAIR vs baselines, fair design (alpha={ALPHA}, "
             f"{n_reps} reps, identical pools; same-stream n={n_stream}; "
             f"tier-C n={n_c}, k_crit={k_crit})", "",
             f"{'method':<12}{'false-rej':>10}{'  95% CI':>16}{'detect':>8}"
             f"{'  95% CI':>16}{'pick':>7}"]
    for m in methods:
        r = rows[m]
        pa = f"{r['pick_acc']:.3f}" if r["pick_acc"] is not None else "-"
        lines.append(
            f"{m:<12}{r['false_rej']:>10.4f}"
            f"  [{r['false_rej_ci'][0]:.3f},{r['false_rej_ci'][1]:.3f}]"
            f"{r['detect']:>8.4f}"
            f"  [{r['detect_ci'][0]:.3f},{r['detect_ci'][1]:.3f}]{pa:>7}")
    (out_dir / "baselines.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
