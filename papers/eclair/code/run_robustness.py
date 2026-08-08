"""Robustness sweeps (manuscript Sec. robustness):

  (a) per-family breakdown of the mutation testbed
  (b) REVERSED calibration split: calibrate on the three former
      evaluation families, evaluate on knapsack+coloring - the
      bidirectional test of cross-family transfer
  (c) pool-size and routing-prior sensitivity

Usage: python3 run_robustness.py [n_reps]     (default 100)
Writes results/robustness.{json,txt}.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

import run_experiment as rx
from eclair.calibrate import calibrate
from eclair.certify import Bets, run_certification
from eclair.problems import CALIB_SPECS, EVAL_SPECS

ALPHA = 0.05
SEED = rx.SEED


def certify_over(specs, bets, budget, n_reps, prng, prior=0.5, pool_k=6):
    stats = {s.name: {"n_f": 0, "f_rej": 0, "n_m": 0, "m_rej": 0}
             for s in specs}
    rx.POOL_K = pool_k
    for rep in range(n_reps):
        spec = specs[rep % len(specs)]
        pool, labels = rx.build_pool(spec, prng)
        res = run_certification(pool, bets, "kelly", budget, ALPHA, prng,
                                prior=prior)
        st_ = stats[spec.name]
        for st, faithful in zip(res["states"], labels):
            if faithful:
                st_["n_f"] += 1
                st_["f_rej"] += st.rejected
            else:
                st_["n_m"] += 1
                st_["m_rej"] += st.rejected
    return stats


def agg(stats):
    n_f = sum(s["n_f"] for s in stats.values())
    f = sum(s["f_rej"] for s in stats.values())
    n_m = sum(s["n_m"] for s in stats.values())
    m = sum(s["m_rej"] for s in stats.values())
    return f / max(n_f, 1), m / max(n_m, 1), n_f, n_m


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_dir = pathlib.Path(__file__).parent / "results"
    calib_fwd = json.loads((out_dir / "calibration.json").read_text())
    bets = Bets(calib_fwd)
    budget = rx.BUDGET_PROBE_EQUIV * statistics.fmean(bets.cost.values())
    out = {}

    # (a) per-family breakdown, forward split
    stats = certify_over(EVAL_SPECS, bets, budget, n_reps,
                         random.Random(SEED + 1))
    out["per_family"] = {
        k: {"false_rej": v["f_rej"] / max(v["n_f"], 1),
            "detect": v["m_rej"] / max(v["n_m"], 1),
            "n_f": v["n_f"], "n_m": v["n_m"]}
        for k, v in stats.items()}
    for k, v in out["per_family"].items():
        print(f"family {k:<18} false-rej={v['false_rej']:.4f} "
              f"detect={v['detect']:.4f}", flush=True)

    # (b) reversed split: calibrate on former eval families
    import eclair.calibrate as cal
    cal.CALIB_SPECS = EVAL_SPECS
    calib_rev = calibrate(random.Random(SEED + 2))
    cal.CALIB_SPECS = CALIB_SPECS
    bets_rev = Bets(calib_rev)
    budget_rev = rx.BUDGET_PROBE_EQUIV * statistics.fmean(
        bets_rev.cost.values())
    stats_rev = certify_over(CALIB_SPECS, bets_rev, budget_rev, n_reps,
                             random.Random(SEED + 3))
    fr, dt, nf, nm = agg(stats_rev)
    out["reversed_split"] = {
        "false_rej": fr, "detect": dt, "n_f": nf, "n_m": nm,
        "calib": {t: {kk: calib_rev[t][kk] for kk in
                      ("p0_hat", "p0_bar", "p1_hat")}
                  for t in ("A", "B", "C")}}
    print(f"reversed split: false-rej={fr:.4f} detect={dt:.4f} "
          f"(n_f={nf}, n_m={nm})", flush=True)

    # (c) pool size and prior sensitivity (forward split)
    out["sensitivity"] = []
    for pool_k, prior in [(4, 0.5), (10, 0.5), (6, 0.3), (6, 0.7)]:
        stats_s = certify_over(EVAL_SPECS, bets, budget, n_reps,
                               random.Random(SEED + 4), prior=prior,
                               pool_k=pool_k)
        fr, dt, nf, nm = agg(stats_s)
        out["sensitivity"].append(
            {"pool_k": pool_k, "prior": prior, "false_rej": fr,
             "detect": dt, "n_f": nf, "n_m": nm})
        print(f"k={pool_k} prior={prior}: false-rej={fr:.4f} "
              f"detect={dt:.4f}", flush=True)
    rx.POOL_K = 6

    (out_dir / "robustness.json").write_text(json.dumps(
        dict(alpha=ALPHA, n_reps=n_reps, seed=SEED, results=out), indent=2))
    (out_dir / "robustness.txt").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
