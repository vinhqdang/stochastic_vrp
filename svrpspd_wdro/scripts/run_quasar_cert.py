"""QUASAR cert: simultaneous risk certification of region plans (THEORY_CERT.md).

For every (region family, gate) group the routes of all instances in the group
form one certification family of size d. Per eps in EPS the script picks the
best fixed k in {1, 3, ..., 20} (weights L_j, oracle choice on the true risks),
then per route computes the Q-CERT cost (applications of A, A^dagger) and the
C-CERT sample count. Model L: one application or one route sample costs L_j.

Usage (from svrpspd_wdro/):
    python3 scripts/run_quasar_cert.py [--risks PATH] [--plans DIR] [--out-dir DIR]

Outputs (default results/quasar/): cert_region_routes.csv (one row per route and
eps) and cert_region_summary.md (one row per group and eps).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum.cert import best_k, c_cert_route, q_cert_route  # noqa: E402

ALPHA = 0.10
DELTA = 0.05
EPS = (1e-2, 3e-3, 1e-3, 1e-4)
GATES = ("Det", "SAA")
REGIONS = ("VNN", "GD")
KS = (1,) + tuple(range(3, 21))
OUT_DIR = ROOT / "results" / "quasar"
DEFAULT_RISKS = OUT_DIR / "region_route_risks.json"
DEFAULT_PLANS = ROOT / "results" / "plans_Region"
ROUTES_CSV = "cert_region_routes.csv"
SUMMARY_MD = "cert_region_summary.md"
SE_NOTE = ("p_j are Monte Carlo estimates from 20 000 common scenarios per instance "
           "(standard error up to about 0.0016 at p = 0.05), so gaps below about 3e-3 "
           "are not resolved.")


def load_risks(path: Path) -> dict:
    """Read {instance: {gate: [p_j in plan order]}} and validate the values."""
    data = json.loads(Path(path).read_text())
    for inst, gates in data.items():
        for gate, ps in gates.items():
            arr = np.asarray(ps, dtype=np.float64)
            if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any((arr < 0) | (arr > 1)):
                raise ValueError(f"invalid risks for {inst} {gate}")
    return data


def load_routes(plans: Path, inst: str, gate: str) -> list[list[int]]:
    """Routes of one instance and gate from the plan cache (res[gate]['plan'])."""
    doc = json.loads((Path(plans) / f"{inst}.json").read_text())
    return doc["res"][gate]["plan"]


def region_of(inst: str) -> str:
    return inst.split("-")[0]


def gather(risks: dict, plans: Path) -> list[dict]:
    """One record per route: instance, gate, region, index, L, p."""
    recs = []
    for inst in sorted(risks):
        for gate in sorted(risks[inst]):
            ps = risks[inst][gate]
            routes = load_routes(plans, inst, gate)
            if len(routes) != len(ps):
                raise ValueError(f"{inst} {gate}: {len(ps)} risks but {len(routes)} routes")
            for j, (route, p) in enumerate(zip(routes, ps)):
                recs.append({"instance": inst, "gate": gate, "region": region_of(inst),
                             "index": j, "L": len(route), "p": float(p)})
    return recs


def certify_group(recs: list[dict], eps: float) -> tuple[dict, list[dict]]:
    """Summary dict and per-route rows for one group at one eps."""
    d = len(recs)
    ps = [r["p"] for r in recs]
    Ls = [r["L"] for r in recs]
    k, _ = best_k(ps, ALPHA, eps, DELTA, d, ks=KS, weights=Ls)
    rows, q_costs, c_ns, q_floor, c_floor, q_ideal = [], [], [], [], [], []
    for r in recs:
        q = q_cert_route(r["p"], ALPHA, eps, DELTA, d, k)
        c = c_cert_route(r["p"], ALPHA, eps, DELTA, d)
        q_costs.append(q["cost"])
        q_ideal.append(q["ideal_cost"])
        c_ns.append(c["n"])
        q_floor.append(q["floor"])
        c_floor.append(c["floor"])
        rows.append({"instance": r["instance"], "gate": r["gate"], "region": r["region"],
                     "index": r["index"], "L": r["L"], "p": r["p"], "gap": abs(r["p"] - ALPHA),
                     "eps": eps, "q_rounds": q["rounds"], "q_cost": q["cost"], "q_ideal": q["ideal_cost"],
                     "q_floor": q["floor"],
                     "c_n": c["n"], "c_floor": c["floor"], "certified": c["certified"]})
    L = np.asarray(Ls, dtype=np.float64)
    q_arr, c_arr = np.asarray(q_costs), np.asarray(c_ns, dtype=np.float64)
    qi_arr = np.asarray(q_ideal)
    Q = float(np.sum(L * q_arr))
    C = float(np.sum(L * c_arr))
    Q_ideal = float(np.sum(L * qi_arr))
    # Shared classical sample: one whole-plan sample per instance (cost n_total) yields
    # all its routes; the sample count is set by the hardest route of that instance.
    C_shared = 0.0
    for inst in {r["instance"] for r in recs}:
        idx = [i for i, r in enumerate(recs) if r["instance"] == inst]
        n_total = float(np.sum(L[idx]))
        C_shared += float(np.max(c_arr[idx])) * n_total
    ratio = q_arr / c_arr
    summary = {"region": recs[0]["region"], "gate": recs[0]["gate"], "eps": eps, "d": d, "k": k,
               "Q": Q, "C": C, "C_shared": C_shared, "Q_ideal": Q_ideal, "Q_ideal_over_C": Q_ideal / C, "Q_unw": float(np.sum(q_arr)),
               "C_unw": float(np.sum(c_arr)), "Q_over_C": Q / C, "Q_over_Cs": Q / C_shared,
               "floor_share": float(np.mean(q_floor)), "ratio_median": float(np.median(ratio)),
               "ratio_q90": float(np.quantile(ratio, 0.9)),
               "share_q_lt_c": float(np.mean(q_arr < c_arr)),
               "classical_floor_share": float(np.mean(c_floor))}
    return summary, rows


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-2:
        return f"{x:.3e}"
    return f"{x:.3f}"


def summary_markdown(summaries: list[dict], n_routes: int, n_instances: int, elapsed: float) -> str:
    lines = [
        "# QUASAR cert: simultaneous risk certification of region plans",
        "",
        f"alpha = {ALPHA}, delta = {DELTA}, eps in {list(EPS)}; {n_instances} instances, "
        f"{n_routes} routes. Groups = (region family, gate); d = routes in the group.",
        "",
        f"Note: {SE_NOTE}",
        "",
        "Q = sum_j L_j q_cost_j (applications of A, A^dagger, model L); Q_ideal = sum_j L_j r_t (2 M_t + 1) for the stopping round t alone (no doubling, no earlier rounds), comparable with C; C = sum_j L_j n_j "
        "(classical route samples, model L); C_shared = sum over instances of "
        "max_j n_j * n_total(instance) (one whole-plan sample per instance). Q_unw and C_unw "
        "are the same sums without the L_j weights. k* is the best fixed k for the group and eps "
        "(oracle choice on the true risks). Ratios are per route, q_cost/c_n, and are independent of L_j.",
        f"Run time: {elapsed:.1f} s.",
        "",
        "| region | gate | eps | d | k* | Q | C | C_shared | Q/C | Q/C_shared | Q_ideal | Q_ideal/C | Q_unw | C_unw "
        "| Q floor share | median q/c | 90% q/c | share q<c |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['region']} | {s['gate']} | {s['eps']:g} | {s['d']} | {s['k']} | {_fmt(s['Q'])} | "
            f"{_fmt(s['C'])} | {_fmt(s['C_shared'])} | {_fmt(s['Q_over_C'])} | {_fmt(s['Q_over_Cs'])} | "
            f"{_fmt(s['Q_ideal'])} | {_fmt(s['Q_ideal_over_C'])} | "
            f"{_fmt(s['Q_unw'])} | {_fmt(s['C_unw'])} | {s['floor_share']:.3f} | "
            f"{_fmt(s['ratio_median'])} | {_fmt(s['ratio_q90'])} | {s['share_q_lt_c']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--risks", type=Path, default=DEFAULT_RISKS, help="risk JSON {instance: {gate: [p]}}")
    ap.add_argument("--plans", type=Path, default=DEFAULT_PLANS, help="plan-cache directory")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="output directory")
    args = ap.parse_args(argv)
    t0 = time.time()
    risks = load_risks(args.risks)
    recs = gather(risks, args.plans)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        groups.setdefault((r["region"], r["gate"]), []).append(r)

    summaries, all_rows = [], []
    for region in sorted(groups, key=lambda g: (REGIONS.index(g[0]) if g[0] in REGIONS else 99, g[1])):
        for eps in EPS:
            s, rows = certify_group(groups[region], eps)
            summaries.append(s)
            all_rows.extend(rows)
            print(f"{region[0]} {region[1]} eps={eps:g} d={s['d']} k*={s['k']} "
                  f"Q/C={s['Q_over_C']:.3f} Q/C_shared={s['Q_over_Cs']:.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["instance", "gate", "region", "index", "L", "p", "gap", "eps", "q_rounds", "q_cost", "q_ideal",
              "q_floor", "c_n", "c_floor", "certified"]
    with open(args.out_dir / ROUTES_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    elapsed = time.time() - t0
    md = summary_markdown(summaries, len(recs), len(risks), elapsed)
    (args.out_dir / SUMMARY_MD).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
