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
from eclair.probes import SOLVER_PARAMS, estimate_error_rate

ALPHA = 0.05
POOL_K = 6
SEED = 20260807
BUDGET_PROBE_EQUIV = 80          # budget = this many mean-cost probes
CERT_EPS = 0.10                  # eps-certification of the final pick
CERT_PROBE_EQUIV = 70            # cert budget, in mean tier-A costs
ERR_EVAL_N = 400                 # independent err(m) estimate per
                                 # certified candidate (R4.1)
ERR_EVAL_Z = 2.2414              # 97.5% two-sided: two mutant
                                 # statements hold jointly at 95%


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a binomial rate."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def capture_provenance():
    """Environment + source provenance, captured BEFORE the run starts.

    Dirty-tree detection deliberately ignores the run's own untracked
    outputs (results/, staging): what matters is the state of the code
    the interpreter loaded, not artifacts the run is in the middle of
    producing."""
    import hashlib
    import os
    import platform
    import subprocess
    here = pathlib.Path(__file__).parent

    def _git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True,
                                  text=True, cwd=str(here)).stdout.strip()
        except Exception:
            return ""

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "unknown"

    dirty = [l for l in _git("status", "--porcelain", ".").splitlines()
             if l.strip() and "/results/" not in l and "/.staging" not in l]
    src_hashes = {}
    for p in sorted(list(here.glob("eclair/*.py"))
                    + [here / "run_experiment.py"]):
        src_hashes[str(p.relative_to(here))] = hashlib.sha256(
            p.read_bytes()).hexdigest()[:16]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpmpy": _ver("cpmpy"), "ortools": _ver("ortools"),
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(dirty), "git_dirty_files": dirty,
        "source_sha256_16": src_hashes,
        "solver_params": dict(SOLVER_PARAMS),
        "audit_z": ERR_EVAL_Z, "audit_n": ERR_EVAL_N,
        "audit_conf": "97.5% two-sided (all audits)",
        "budget_semantics": "launch threshold; may overshoot by one probe",
    }


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


PUBLICATION_REPS = 100      # runs at/above this size may claim the
                            # canonical filenames (review R5.2)


def write_atomic(path, text):
    """Write via a temporary file + rename so a crashed or killed run
    cannot leave a half-written artifact behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def promote_run(stage_dir, out_dir, files):
    """Publish a whole run at once (review R7.6). Individual atomic
    writes do not give MULTI-FILE consistency: a crash between
    calibration and the mutation results used to leave the canonical
    package mixing two runs. Everything is written to a staging
    directory first, checked for completeness, given a manifest with a
    SHA-256 per file, and only then moved into place."""
    import hashlib
    manifest = {}
    for name in files:
        src = stage_dir / name
        if not src.exists():
            raise RuntimeError(f"run incomplete: {name} missing; "
                               f"canonical artifacts left untouched")
        manifest[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    (stage_dir / "MANIFEST.json").write_text(json.dumps(
        {"files": manifest}, indent=2))
    for name in list(files) + ["MANIFEST.json"]:
        (stage_dir / name).replace(out_dir / name)
    return manifest


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    # Smaller-than-publication runs write to a tagged filename and
    # leave the canonical artifacts untouched.
    tag = "" if n_reps >= PUBLICATION_REPS else f"_smoke{n_reps}"
    stage = out_dir / f".staging{tag}"
    if stage.exists():
        for f in stage.iterdir():
            f.unlink()
    stage.mkdir(parents=True, exist_ok=True)
    if tag:
        print(f"NOTE: n_reps={n_reps} < {PUBLICATION_REPS}; writing "
              f"results/mutation_experiment{tag}.* and leaving the "
              f"canonical artifacts untouched.", flush=True)

    provenance = capture_provenance()          # BEFORE any output exists
    if provenance["git_dirty"]:
        print(f"WARNING: dirty tree ({len(provenance['git_dirty_files'])} "
              f"modified code paths); provenance records this.", flush=True)
    rng = random.Random(SEED)
    t0 = time.perf_counter()
    print("calibrating on held-out families ...", flush=True)
    calib = calibrate(rng)
    write_atomic(stage / f"calibration{tag}.json",
                 json.dumps(calib, indent=2))
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
          f"per-rep budget = {budget*1000:.0f}ms probe time", flush=True)

    results = {}
    for pol_i, policy in enumerate(("kelly", "round_robin", "cost_blind")):
        n_faith = n_faith_rej = n_mut = n_mut_rej = 0
        kill_costs, pick_ok, pick_n, abstain = [], 0, 0, 0
        ebh_false = ebh_total = 0
        cert_n = cert_ok = cert_abstain = 0
        cert_all = cert_ok_all = cert_abstain_all = cert_issued = 0
        kill_costs_screen = []
        certified_log = []
        screen_spend = []
        for rep in range(n_reps):
            spec = EVAL_SPECS[rep % len(EVAL_SPECS)]
            # pools depend ONLY on rep -> truly identical across
            # policies (R1.8); probe randomness is per (policy, rep)
            pool, labels = build_pool(
                spec, random.Random(SEED * 100_000 + rep))
            prng = random.Random(SEED + 1000 * (pol_i + 1) + rep)
            res = run_certification(pool, bets, policy, budget, ALPHA,
                                    prng, eps=CERT_EPS,
                                    cert_budget=CERT_PROBE_EQUIV
                                    * bets.cost["A"])
            screen_spend.append(res["screen_spent"])
            cp = res.get("certified_pick")
            cert_all += 1                       # unconditional denominator
            if cp is None:
                cert_abstain_all += 1
            else:
                cert_issued += 1
                idx_c = res["states"].index(cp)
                cert_ok_all += labels[idx_c]
                # INDEPENDENT measurement of the quantity Theorem 3's
                # null is about, on fresh micro instances (R4.1): a
                # certificate does not prove err < eps, so we measure.
                err, n_e, lo, hi = estimate_error_rate(
                    cp.cand,
                    random.Random(SEED + 77_000 + 1000 * pol_i + rep),
                    n=ERR_EVAL_N, z=ERR_EVAL_Z)
                certified_log.append({
                    "rep": rep, "family": spec.name, "policy": policy,
                    "z": ERR_EVAL_Z, "conf": "97.5% two-sided",
                    "label_faithful": bool(labels[idx_c]),
                    "descriptor": (list(cp.cand.descriptor)
                                   if cp.cand.descriptor else None),
                    "err_hat": err, "err_n": n_e,
                    "err_ci": [lo, hi],
                    "err_below_eps": hi < CERT_EPS,
                    "err_above_eps": lo >= CERT_EPS})
            if any(labels):                     # conditional (availability)
                cert_n += 1
                if cp is None:
                    cert_abstain += 1
                else:
                    cert_ok += labels[res["states"].index(cp)]
            for st, faithful in zip(res["states"], labels):
                if faithful:
                    n_faith += 1
                    n_faith_rej += st.rejected
                else:
                    n_mut += 1
                    if st.rejected:
                        n_mut_rej += 1
                        kill_costs.append(st.spent)          # incl. cert
                        kill_costs_screen.append(st.spent - st.cert_spent)
            for i in res["ebh_rejected"]:
                ebh_total += 1
                ebh_false += labels[i]
            if res["screening_abstained"]:
                abstain += 1
            elif any(labels):
                pick_n += 1
                idx = res["states"].index(res["screening_survivor"])
                pick_ok += labels[idx]
        results[policy] = {
            "false_rej": n_faith_rej / max(n_faith, 1), "n_faith": n_faith,
            "false_rej_ci": wilson(n_faith_rej, n_faith),
            "detect": n_mut_rej / max(n_mut, 1), "n_mut": n_mut,
            "detect_ci": wilson(n_mut_rej, n_mut),
            # conditional on the pool containing a faithful candidate
            "cert_rate": (cert_n - cert_abstain) / max(cert_n, 1),
            "cert_pick_acc": cert_ok / max(cert_n - cert_abstain, 1),
            "cert_abstain": cert_abstain / max(cert_n, 1),
            "n_cert": cert_n,
            # unconditional: every run, including all-mutant pools
            "cert_rate_uncond": cert_issued / max(cert_all, 1),
            "cert_acc_uncond": cert_ok_all / max(cert_issued, 1),
            "n_runs": cert_all, "n_certified": cert_issued,
            "n_certified_faithful": cert_ok_all,
            "n_abstained": cert_abstain_all,
            "mean_kill_cost_screen_ms": 1000 * statistics.fmean(
                kill_costs_screen) if kill_costs_screen else None,
            "mean_kill_cost_ms": 1000 * statistics.fmean(kill_costs)
            if kill_costs else None,
            "pick_accuracy": pick_ok / max(pick_n, 1), "n_picks": pick_n,
            "abstain_rate": abstain / n_reps,
            "ebh_fdr": ebh_false / max(ebh_total, 1),
            "ebh_rejections": ebh_total,
            "mean_screen_spend_ms": 1000 * statistics.fmean(screen_spend),
            "max_screen_spend_ms": 1000 * max(screen_spend),
            "screen_spend_ms_per_run": [round(1000 * x, 3)
                                        for x in screen_spend],
            "certified_log": certified_log,
            "n_cert_err_below_eps": sum(c["err_below_eps"]
                                        for c in certified_log),
            "n_cert_err_above_eps": sum(c["err_above_eps"]
                                        for c in certified_log),
        }
        r = results[policy]
        print(f"{policy:<12} false-rej={r['false_rej']:.4f} (n={n_faith})  "
              f"detect={r['detect']:.4f} (n={n_mut})  "
              f"kill-cost={r['mean_kill_cost_ms'] and round(r['mean_kill_cost_ms'])}ms  "
              f"pick-acc={r['pick_accuracy']:.3f}  eBH-FDR={r['ebh_fdr']:.4f}  "
              f"cert-rate={r['cert_rate']:.3f} cert-acc={r['cert_pick_acc']:.3f}",
              flush=True)

    payload = {"provenance": provenance,
               "alpha": ALPHA, "pool_k": POOL_K, "n_reps": n_reps,
               "budget_s": budget, "seed": SEED,
               "eval_families": [s.name for s in EVAL_SPECS],
               "results": results}
    write_atomic(stage / f"mutation_experiment{tag}.json",
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
    write_atomic(stage / f"mutation_experiment{tag}.txt",
                 "\n".join(lines) + "\n")
    # promote the COMPLETE run in one step (calibration + results
    # together), so the canonical package can never mix two runs
    manifest = promote_run(stage, out_dir,
                           [f"calibration{tag}.json",
                            f"mutation_experiment{tag}.json",
                            f"mutation_experiment{tag}.txt"])
    stage.rmdir()
    print(f"promoted run: {len(manifest)} files + MANIFEST.json")
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
