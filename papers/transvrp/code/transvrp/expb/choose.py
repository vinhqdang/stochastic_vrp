"""The section 6.1 configuration-selection rule (EXPERIMENT_B.md, review item M8).

Score of a (config, pass count): mean over the tuning routes of
100 (C_cfg - C_baton_eq) / C_baton_eq on dev, baton_eq fitted on the same rows. The
lowest score wins; candidates within ``TIE_POINTS`` of it are tied, and among the tied
the fewer passes win, then the lower total fit seconds, then the lower config index. A
config with a missing route, a non-finite cost or a non-positive baton_eq cost is excluded
(and recorded). DQN and PPO (``curve=True``) are chosen jointly over (config, pass count)
from the ``<arm>@p<P>`` rows of a 3-pass tuning run; other arms have one point (P = 1).

``write_record`` stores every score and the choice in ``tuning_record.json``;
``load_chosen`` re-derives the choice from the stored scores and refuses a record whose
pairs are not the grid's, so ``stages.CHOSEN_ARM_CFG`` is produced, never typed. Seconds
are the sum over routes of the tuning job's ``fit_seconds`` (a run's passes share one job,
so seconds only separate configs at equal pass counts, where they are comparable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

TIE_POINTS = 0.10
TIE_SLACK = 1e-9            # float noise only; the window itself is TIE_POINTS
PASSES = (1, 2, 3)
BASE_POLICY = "baton_eq"
RECORD_NAME = "tuning_record.json"


def _keyed(df: pd.DataFrame, what: str) -> dict:
    out: dict = {}
    for row in df.itertuples(index=False):
        key = (str(row.inst), int(row.route_id), int(row.rep))
        if key in out:
            raise ValueError(f"duplicate dev row for {key} ({what})")
        out[key] = (float(row.mean_cost), float(row.fit_seconds))
    return out


def _config_points(arm: str, df: pd.DataFrame, expected: set | None, curve: bool):
    """{pass: (score, seconds)} of one config, or None when it is excluded."""
    base = _keyed(df[df["policy"] == BASE_POLICY], BASE_POLICY)
    need = {(i, r, 0) for i, r in expected} if expected is not None else set(base)
    points = {}
    for p in (PASSES if curve else (1,)):
        name = f"{arm}@p{p}" if curve else arm
        cur = _keyed(df[df["policy"] == name], name)
        if not need or not need <= set(cur) or not need <= set(base):
            return None
        gaps, secs = [], 0.0
        for key in sorted(need):
            c, b = cur[key][0], base[key][0]
            if not (math.isfinite(c) and math.isfinite(b) and b > 0):
                return None
            gaps.append(100.0 * (c - b) / b)
            secs += cur[key][1]
        points[p] = (sum(gaps) / len(gaps), secs)
    return points


def _pick(cands: list[tuple], tie: float) -> tuple:
    """cands: (config, passes, score, seconds)."""
    best = min(c[2] for c in cands)
    tied = [c for c in cands if c[2] <= best + tie + TIE_SLACK]
    return min(tied, key=lambda c: (c[1], c[3], c[0]))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_config(arm: str, tune: dict[int, pd.DataFrame], expected_routes=None, *,
                  curve: bool = False, sources: dict | None = None,
                  tie: float = TIE_POINTS) -> dict:
    """Applies the 6.1 rule to ``tune`` ({config index: dev rows of its tuning stage})."""
    expected = None if expected_routes is None else {(str(i), int(r)) for i, r in expected_routes}
    cands, scores, seconds, excluded = [], {}, {}, []
    for idx in sorted(tune):
        points = _config_points(arm, tune[idx], expected, curve)
        if points is None:
            excluded.append(idx)
            continue
        seconds[f"c{idx}"] = points[min(points)][1]
        for p, (score, secs) in points.items():
            scores[f"c{idx}@p{p}"] = score
            cands.append((idx, p, score, secs))
    if not cands:
        raise ValueError(f"no admissible tuning configuration for arm {arm!r} "
                         f"(excluded: {excluded})")
    config, passes, _, _ = _pick(cands, tie)
    return {"config": config, "passes": passes, "curve": bool(curve), "scores": scores,
            "seconds": seconds, "excluded": excluded, "tie": tie,
            "sources": {str(i): _sha(p) for i, p in (sources or {}).items()}}


def _chosen_pairs(arm: str, rec: dict, grid: dict) -> tuple:
    pairs = tuple(grid[arm][rec["config"]])
    return pairs + (("passes", rec["passes"]),) if rec["curve"] else pairs


def write_record(path: Path, records: dict, grid: dict) -> Path:
    arms = {}
    for arm, rec in records.items():
        arms[arm] = dict(rec, pairs=[list(p) for p in _chosen_pairs(arm, rec, grid)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rule": "EXPERIMENT_B 6.1", "arms": arms}, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def _rederive(rec: dict) -> tuple[int, int]:
    cands = []
    for key, score in rec["scores"].items():
        cfg, p = key[1:].split("@p")
        cands.append((int(cfg), int(p), float(score), float(rec["seconds"][f"c{cfg}"])))
    if not cands:
        raise ValueError("record holds no scores")
    best = _pick(cands, float(rec["tie"]))
    return best[0], best[1]


def load_chosen(path: Path, grid: dict) -> dict:
    """{arm: config pairs} from a committed tuning record; {} when there is none yet."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for arm, rec in json.loads(path.read_text(encoding="utf-8"))["arms"].items():
        if arm not in grid or not 0 <= rec["config"] < len(grid[arm]):
            raise ValueError(f"tuning record names config {rec['config']} of {arm!r}, "
                             "which is not in the grid")
        pairs = tuple(tuple(p) for p in rec["pairs"])
        if pairs != _chosen_pairs(arm, rec, grid):
            raise ValueError(f"tuning record pairs for {arm!r} are not the grid's pairs")
        if _rederive(rec) != (rec["config"], rec["passes"]):
            raise ValueError(f"tuning record choice for {arm!r} does not follow from its "
                             "stored scores (rule 6.1)")
        out[arm] = pairs
    return out


def verify_sources(path: Path, results_dir: Path) -> None:
    """Every tuning CSV hashed into the record still has that hash."""
    for arm, rec in json.loads(Path(path).read_text(encoding="utf-8"))["arms"].items():
        for idx, digest in rec["sources"].items():
            src = Path(results_dir) / f"b_tune_{arm}_c{idx}.csv"
            if not src.exists() or _sha(src) != digest:
                raise ValueError(f"{src.name} differs from the file the record was made from")


DOS_TUNE_ROUTES = "subset6"        # EXPERIMENT_C 1.8: the six tuning routes (3 medium, 3 long)


def rows_hashes(tuning_dir: Path) -> dict:
    """sha256 of every row file of one tuning stage (its ``rows`` directory), by file name."""
    rows = Path(tuning_dir) / "rows"
    if not rows.is_dir():
        raise ValueError(f"no rows directory for tuning stage {Path(tuning_dir).name}")
    files = sorted(p for p in rows.iterdir() if p.is_file())
    if not files:
        raise ValueError(f"{rows} holds no row files")
    return {p.name: _sha(p) for p in files}


def verify_rows(path: Path, results_dir: Path) -> None:
    """Every DOS tuning row file hashed into the record still has that hash."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))["arms"]["dos"]
    for cfg, digests in rec.get("rows", {}).items():
        current = rows_hashes(Path(results_dir) / f"b_tune_dos_{cfg}")
        if current != digests:
            raise ValueError(f"row files of DOS tuning {cfg} differ from the record")


def choose_dos(results_dir_b: Path, eligibility: Path, out: Path) -> Path:
    """EXPERIMENT_C 1.8: DOS configuration by the 6.1 rule on the six tuning routes of B's
    dev DOS tuning CSVs (``b_tune_dos_c<i>.csv``). Dev rows only; the record uses the same
    format as ``tuning_record.json`` and is read back by ``stages.load_chosen``."""
    from ..expa import sweep          # noqa: PLC0415 (heavy import)
    from .stages import TUNE_GRID     # noqa: PLC0415
    routes = sweep._route_rows(DOS_TUNE_ROUTES, pd.read_csv(eligibility))
    files = {i: Path(results_dir_b) / f"b_tune_dos_c{i}.csv"
             for i in range(len(TUNE_GRID["dos"]))}
    tune = {i: pd.read_csv(f) for i, f in files.items() if f.exists()}
    rec = choose_config("dos", tune, routes, curve=False,
                        sources={i: files[i] for i in tune})
    rec["rows"] = {f"c{i}": rows_hashes(Path(results_dir_b) / f"b_tune_dos_c{i}") for i in tune}
    return write_record(out, {"dos": rec}, TUNE_GRID)


def main(argv: list[str] | None = None) -> int:
    from ..expa import sweep          # noqa: PLC0415 (heavy import)
    from .stages import ACTIVE_CUTS, TUNE_GRID   # noqa: PLC0415
    ap = argparse.ArgumentParser(prog="python -m transvrp.expb.choose")
    ap.add_argument("--results-dir", type=Path, default=sweep.DEFAULT_RESULTS_B)
    ap.add_argument("--eligibility", type=Path, default=sweep.DEFAULT_RESULTS / "eligibility_dev.csv")
    ap.add_argument("--dos-out", type=Path, default=None,
                    help="write the EXPERIMENT_C 1.8 DOS record to this path and stop")
    args = ap.parse_args(argv)
    if args.dos_out is not None:
        print(choose_dos(args.results_dir, args.eligibility, args.dos_out))
        return 0
    routes = sweep._route_rows(ACTIVE_CUTS.tune_routes, pd.read_csv(args.eligibility))
    records = {}
    for arm, grid in TUNE_GRID.items():
        if len(grid) < 2 or (arm == "dos" and ACTIVE_CUTS.drop_dos):
            continue
        files = {i: args.results_dir / f"b_tune_{arm}_c{i}.csv" for i in range(len(grid))}
        tune = {i: pd.read_csv(f) for i, f in files.items() if f.exists()}
        records[arm] = choose_config(arm, tune, routes, curve=arm in ("dqn", "ppo"),
                                     sources={i: files[i] for i in tune})
    out = write_record(args.results_dir / RECORD_NAME, records, TUNE_GRID)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
