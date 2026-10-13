#!/usr/bin/env python3
"""Solve and cache planning-gate plans for the region instances (data/Region).

Same planner and gates as the BATON plan cache (dethloff_runner.solve_instance:
Clarke-Wright + 2-opt for n > 90), gates Det and SAA. Plans are written in the
plan-cache JSON layout of run_realistic_eval._save_plans, to a separate
directory results/plans_Region so the 73-instance cache is not mixed with the
region set.

Usage:
    python scripts/solve_region_plans.py [--instances GLOB] [--workers 3]
                                         [--gates Det,SAA] [--force]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dethloff_runner as DR  # noqa: E402

DATA_DIR = ROOT / "data" / "Region"
PLANS_DIR = ROOT / "results" / "plans_Region"


def _payload(sol: dict) -> dict:
    return {
        "name": sol["name"], "n": int(sol["n"]), "Q": float(sol["Q"]),
        "omega_V": float(sol["omega_V"]),
        "dbar": [float(x) for x in sol["dbar"]], "pbar": [float(x) for x in sol["pbar"]],
        "res": {g: {"plan": [[int(c) for c in r] for r in sol["res"][g]["plan"]],
                    "K": int(sol["res"][g]["K"]), "dist": float(sol["res"][g]["dist"]),
                    "time": float(sol["res"][g]["time"])}
                for g in sol["res"]},
    }


def _work(task) -> str:
    path, gates = task
    t0 = time.time()
    sol = DR.solve_instance(str(path), 0.0, DR.NO_IMPROVE, use_prune=True, which=gates)
    out = PLANS_DIR / f"{path.stem}.json"
    out.write_text(json.dumps(_payload(sol)))
    ks = ", ".join(f"{g}: K={sol['res'][g]['K']} dist={sol['res'][g]['dist']:.0f} km "
                   f"({sol['res'][g]['time']:.0f} s)" for g in gates)
    return f"{path.stem} n={sol['n'] - 1}: {ks}  [{time.time() - t0:.0f} s]"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=None, help="fnmatch pattern on instance names")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gates", default="Det,SAA")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    gates = args.gates.split(",")
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(DATA_DIR.glob("*.vrpspd"))
    if args.instances:
        paths = [p for p in paths if fnmatch.fnmatch(p.stem, args.instances)]
    if not args.force:
        paths = [p for p in paths if not (PLANS_DIR / f"{p.stem}.json").exists()]
    # largest first, so the long instances do not finish last
    paths.sort(key=lambda p: -p.stat().st_size)
    print(f"{len(paths)} instances to solve, gates {gates}, workers {args.workers}", flush=True)
    with mp.Pool(args.workers) as pool:
        for i, line in enumerate(pool.imap_unordered(_work, [(p, gates) for p in paths]), 1):
            print(f"[{i}/{len(paths)}] {line}", flush=True)


if __name__ == "__main__":
    main()
