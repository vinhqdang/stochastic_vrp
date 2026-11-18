"""B-DOS gate investigation (CODE_REVIEW_B M1; results in ../DOS_GATE_INVESTIGATION.md).

Dev and train streams only (route_pool 'train' for fitting; DOS validation rows are the
fit's own 20% hold-out). Nothing here reads test or test_b data.

  python dos_gate_investigation.py fit  INST ROUTE_ID STEPS [STEPS ...] --out DIR
      fits DOS at N = 10^5, rep 0, with s_min = s_max = STEPS (0 = the spec default
      ceil(20 n / 8192), about 98 at n = 40,000) and writes DIR/dos_<inst>_<route>.json
  python dos_gate_investigation.py analyze DIR
      pass shares by absolute and relative tolerance and by steps, from the JSON files
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

STEPS = ("0", "100", "400", "1000", "3000")
ABS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
REL = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


def fit(inst: str, route_id: int, steps_list, out: str, N: int = 100_000) -> None:
    from transvrp.expa import sweep
    from transvrp.expa.paths import route_pool
    from transvrp.expb.dos import DOSConfig, fit_dos

    spec = sweep._spec_of(inst, route_id)
    paths = route_pool(inst, "train", 0, N, routes=[route_id])[route_id]
    result = {}
    for steps in steps_list:
        cfg = DOSConfig(lr=1e-3, s_min=steps, s_max=steps) if steps > 0 else DOSConfig(lr=1e-3)
        t0 = time.time()
        _, _, diag, _ = fit_dos(spec, paths, N, 0, cfg)
        rows = []
        for d in diag:
            if d.get("kind") != "stage" or d["n_val"] < 100:
                continue
            obj = next(v for s, v in d["val_loss_per_ckpt"] if s == d["chosen_ckpt"])
            rows.append(dict(k=d["k"], fold=d["fold"], steps=d["steps"], obj=obj,
                             never=d["never_val"], always=float(spec.H[d["k"] - 1] / spec.E0)))
        result[str(steps)] = dict(seconds=time.time() - t0, m=spec.m, rows=rows)
        print(inst, route_id, steps, round(result[str(steps)]["seconds"]), flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"dos_{inst}_{route_id}.json"), "w") as fh:
        json.dump(result, fh)


def analyze(out: str) -> None:
    runs = {os.path.basename(f)[4:-5]: json.load(open(f))
            for f in sorted(glob.glob(os.path.join(out, "dos_*.json")))}
    for k in STEPS:
        gaps, rels, secs = [], [], {}
        for name, d in runs.items():
            if k not in d:
                continue
            rows = d[k]["rows"]
            g = np.array([r["obj"] - min(r["always"], r["never"]) for r in rows])
            span = np.array([abs(r["always"] - r["never"]) for r in rows])
            gaps += list(g)
            rels += list(g / np.maximum(span, 1e-12))
            secs[name] = round(d[k]["seconds"])
        if not gaps:
            continue
        g, r = np.array(gaps), np.array(rels)
        print(f"\nsteps {k}: {g.size} stages; seconds per job {secs}")
        print("  absolute tol  :", "  ".join(f"{t:g}:{(g <= t).mean():.3f}" for t in ABS))
        print("  relative tol  :", "  ".join(f"{t:g}:{(r <= t).mean():.3f}" for t in REL))
        print(f"  95% quantile of the gap: abs {np.quantile(g, 0.95):.3g}, rel {np.quantile(r, 0.95):.3g};"
              f" worst: abs {g.max():.3g}, rel {r.max():.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("inst")
    f.add_argument("route_id", type=int)
    f.add_argument("steps", type=int, nargs="+")
    f.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("out")
    args = ap.parse_args()
    fit(args.inst, args.route_id, args.steps, args.out) if args.cmd == "fit" else analyze(args.out)
