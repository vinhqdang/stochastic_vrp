"""Prediction identity of reused models (EXPERIMENT_C.md 4.3 (b)).

B's FULL and PPO predictors must give bit-identical predictions on W6 dev data at B's fit
commit bf3edab and at C's code. This module uses only modules that exist at bf3edab
(``expa.sweep``, ``expa.paths``, ``expa.features``, ``expb.codecs``), so the same file runs
in a worktree of that commit:

    git worktree add <dir> bf3edab
    git show HEAD:papers/transvrp/code/transvrp/expc/identity_check.py > <scratch>/identity_check.py
    cd <dir>/papers/transvrp/code
    PYTHONPATH=. python <scratch>/identity_check.py predict --stage b_full --inst CON3-0 --route <id> --out <a.npz>
    cd <C checkout>/papers/transvrp/code
    python -m transvrp.expc.identity_check predict --stage b_full --inst CON3-0 --route <id> --out <b.npz>
    python -m transvrp.expc.identity_check compare <a.npz> <b.npz>      (exit 0 only if identical)

``predict_route`` is also used in-process by tests/test_expc_gates.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from transvrp.expa import sweep
from transvrp.expa.features import RouteCache
from transvrp.expa.paths import route_pool

N_ROWS = 1000          # dev rows per route used for the comparison
PAYLOAD_DIR = "results/expB/models"


def predict_route(payload: dict, inst: str, route_id: int, n_rows: int = N_ROWS) -> dict:
    """Predictions of every decision epoch's predictors on the first ``n_rows`` W6 dev
    days of one route: key ``k<epoch>_<predictor index>``, value float64 array."""
    spec = sweep._spec_of(inst, route_id)
    dev = route_pool(inst, "dev")[route_id]
    cache = RouteCache(spec, dev, payload["stats"])
    models = sweep._unpack(payload["models"])     # FULL (arm full) and B arms alike
    rows = np.arange(min(n_rows, cache.W.shape[0]))
    out = {}
    for k in sorted(models):
        for i, pred in enumerate(models[k]):
            out[f"k{k}_{i}"] = np.asarray(pred.predict(cache, k, rows), dtype=np.float64)
    return out


def predict_stage_route(stage_dir: Path, inst: str, route_id: int, rep: int = 0) -> dict:
    """Loads the rep-``rep`` payload of ``route_id`` in ``stage_dir`` and predicts."""
    matches = sorted(Path(stage_dir).glob(f"*__{inst}__r{route_id}__*__rep{rep}.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{len(matches)} payloads for {inst} r{route_id} rep {rep}")
    return predict_route(sweep.load_payload(matches[0]), inst, route_id)


def save(predictions: dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **predictions)
    return out


def compare(a: Path, b: Path) -> tuple[bool, dict]:
    """True when both files hold the same keys and bit-identical arrays."""
    with np.load(a) as za, np.load(b) as zb:
        if sorted(za.files) != sorted(zb.files):
            return False, {"keys": (sorted(za.files), sorted(zb.files))}
        diffs = {k: float(np.max(np.abs(za[k] - zb[k]))) if za[k].size else 0.0
                 for k in za.files}
        same = all(np.array_equal(za[k], zb[k]) for k in za.files)
    return same, diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expc.identity_check")
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("predict")
    run.add_argument("--stage", required=True)
    run.add_argument("--inst", required=True)
    run.add_argument("--route", type=int, required=True)
    run.add_argument("--rep", type=int, default=0)
    run.add_argument("--code", default=".", help="the papers/transvrp/code directory")
    run.add_argument("--payloads", default=None,
                     help="models root (default: CODE/results/expB/models)")
    run.add_argument("--out", required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        same, detail = compare(Path(args.a), Path(args.b))
        print("identical" if same else f"DIFFERENT {detail}")
        return 0 if same else 1
    if args.cmd != "predict":
        ap.error("choose predict or compare")
    root = Path(args.payloads) if args.payloads else Path(args.code) / PAYLOAD_DIR
    preds = predict_stage_route(root / args.stage, args.inst, args.route, args.rep)
    save(preds, Path(args.out))
    print(args.out, file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
