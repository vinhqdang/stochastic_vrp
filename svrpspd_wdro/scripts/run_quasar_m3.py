"""QUASAR M3 driver.

Part a (v2, design Revision 2): move statistics on the BATON copula demand
model, common scenarios per instance, and the copula discretisation check.
Part v3 (Revision 3): scenario pass for the end-load lemma and the paired
level-only check (cached per instance under m3a_v3_parts/).
Part v3gh (Revision 3): quadrature check of the copula window probability,
Gauss-Hermite nodes against equal-probability factor bins, against continuous
Monte Carlo on the same scenarios; writes m3a_v3_gh.csv.

Usage (from svrpspd_wdro/):
    python3 scripts/run_quasar_m3.py --part v2 [--rho 0.6] [--N 200000]
        [--every 1] [--instances PATTERN] [--workers 3] [--cap 200] [--force]
    python3 scripts/run_quasar_m3.py --part v2disc [--rho 0.6]
    python3 scripts/run_quasar_m3.py --part v2summary
    python3 scripts/run_quasar_m3.py --part v3 [--workers 3] [--force]
    python3 scripts/run_quasar_m3.py --part v3gh [--workers 4] [--force]
    python3 scripts/run_quasar_m3.py --part v3summary

v2 writes results/quasar/m3a_v2{tag}_moves.csv.gz and m3a_v2{tag}_routes.csv
(tag = "" for rho = 0.6, "_rho0" for rho = 0). Per-instance results are cached
under results/quasar/m3a_v2_parts/ so an interrupted run resumes.
Progress is printed and appended to results/quasar/m3_<part>.log.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # worker processes: one BLAS thread each

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import dethloff_runner as DR  # noqa: E402  (scenario generator of the BATON runs)
from quantum.experiments import (  # noqa: E402
    ALT_DIST_LABELS,
    BOOT_REPS,
    DEFAULT_CV,
    DEFAULT_RHO,
    GATE_BANDS,
    GATE_LEVEL,
    LEVEL_RULES,
    PPRIME_BANDS,
    alt_distance_label,
    band_label,
    chance_copula,
    class_costs_array,
    copula_demand,
    crn_factor_grid,
    end_load_violations,
    gate_distance,
    insertion_candidates,
    insertion_envelopes,
    insertion_peak,
    intra_candidates,
    quantum_queries_array,
    LENGTH_BANDS,
    length_label,
    level_rule,
    load_profile,
    paired_level_check,
    q999_max,
    quantum_queries,
    classical_queries,
    reconstruct_intra,
    route_bootstrap_quantile,
    route_peak,
    rule_of_three_bound,
    signed_costs,
    sum_loads,
    window_stats,
)

PLANS_DIR = ROOT / "results" / "plans"
OUT_DIR = ROOT / "results" / "quasar"
PARTS_ROOT = OUT_DIR / "m3a_v2_parts"
EPS_LIST = (1e-2, 1e-3, 1e-4)
EPS_MAIN = 1e-3
SCEN_SEED = DR.SEED          # scenario seed: the runner's SEED (7), same for all instances
N_SCEN = 200_000
CAP_INS = 200                # insertion candidates per route (sampled)
CAP_INTRA = 200              # intra-route neighbours per route when |r| > ALL_UPTO
ALL_UPTO = 12
STRATA = ((0.0, 0.005, "p<0.005"), (0.005, 0.05, "0.005-0.05"), (0.05, 0.1, "0.05-0.1"),
          (0.1, 0.3, "0.1-0.3"), (0.3, 1.0000001, ">0.3"))
CLASS_NAMES = {1: "insertion", 2: "intra", 3: "alt-position"}
FAMILIES = ("CON", "SCA", "CMT", "city")
DISC_B = (4, 5, 6)
DISC_NODES = 16              # Gauss-Hermite nodes per factor for the discrete copula (design: 16 x 16)
DISC_B6_MAX_L = 8            # b = 6 only for routes up to this length (cost), design rule only
DISC_SEED = 0
RHO0_EVERY = 4               # rho = 0 sensitivity on every 4th instance

_LOG_PATH: Path | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def family_of(name: str) -> str:
    for prefix in ("CON", "SCA", "CMT"):
        if name.startswith(prefix):
            return prefix
    return "city"


def _crc(key) -> int:
    return zlib.crc32(str(key).encode("utf-8")) & 0xFFFFFFFF


def _rng(seed: int, *keys) -> np.random.Generator:
    return np.random.default_rng([int(seed)] + [_crc(k) for k in keys])


def _list_instances(pattern: str | None, every: int = 1) -> list[Path]:
    paths = sorted(PLANS_DIR.glob("*.json"))
    if pattern:
        paths = [p for p in paths if fnmatch.fnmatch(p.stem, pattern)]
    return paths[::max(1, int(every))]


def _tag(rho: float) -> str:
    return "" if abs(rho - DEFAULT_RHO) < 1e-12 else f"_rho{rho:g}"


def _stratum(p: float) -> str:
    for lo, hi, lab in STRATA:
        if lo <= p < hi:
            return lab
    return "NaN"


def draw_scenarios(plan: dict, N: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Common scenarios of one instance, customer-major: (n, N) deliveries and pickups.

    rho = DR.RHO uses dethloff_runner.make_scenarios itself. Other rho values
    call dethloff_runner.sample_demands with the same rng stream order
    (deliveries first, then pickups), so only the copula strength differs.
    """
    dbar = np.asarray(plan["dbar"], dtype=np.float64)
    pbar = np.asarray(plan["pbar"], dtype=np.float64)
    n = dbar.size
    if abs(rho - DR.RHO) < 1e-12:
        d, p = DR.make_scenarios(dbar, pbar, N, DR.CV, DR.DIST, seed)
    else:
        rng = np.random.default_rng(seed)
        d = DR.sample_demands(dbar, n, N, DR.CV, DR.DIST, rng, rho=rho)
        p = DR.sample_demands(pbar, n, N, DR.CV, DR.DIST, rng, rho=rho)
    return np.ascontiguousarray(d.T), np.ascontiguousarray(p.T)


def route_peak_flags(route, dsc, psc, Q: float) -> np.ndarray:
    return route_peak(route, dsc, psc) > Q


# ============================================================
# Part a (v2)
# ============================================================


def _move_row(instance, family, route_id, L, cls, kind, j, pos, pos2, st):
    """One output row of a move, plus its query counts at EPS_MAIN."""
    pn = st["p_new"]
    row = {
        "instance": instance, "family": family, "route_id": route_id, "L": L,
        "cls": cls, "kind": kind, "j": int(j), "pos": int(pos), "pos2": int(pos2),
        "p": st["p"], "p_new": pn, "P_D1": st["P_D1"], "P_Dm1": st["P_Dm1"],
        "var_D": st["var_D"], "delta": st["delta"],
        "rho_q": st["rho_q"], "se_rho_q": st["se_rho_q"],
        "rho_c": st["rho_c"], "se_rho_c": st["se_rho_c"],
        "zero": bool(st["zero"]),
    }
    e = EPS_MAIN
    row["Nq_sa"] = quantum_queries(pn * (1 - pn), e)
    row["Nc_sa"] = classical_queries(pn * (1 - pn), e)
    if cls == 1:
        row["Nq_crn"] = quantum_queries(st["var_D"], e)
        row["Nc_crn"] = classical_queries(st["var_D"], e)
    else:
        row["Nq_crn"], row["Nc_crn"] = signed_costs(st["P_D1"], st["P_Dm1"], e)
    row["Nq_crn_1amp"] = quantum_queries(st["var_D"], e)
    return row


def _work_v2(task: tuple) -> dict:
    """All moves of all unique routes of one plan (rho, N as in the task)."""
    path, rho, N, cap, parts_dir = task
    t0 = time.time()
    plan = json.loads(Path(path).read_text())
    name = plan["name"]
    fam = family_of(name)
    Q = float(plan["Q"])
    dsc, psc = draw_scenarios(plan, N, rho, SCEN_SEED)
    t_scen = time.time() - t0

    # Unique routes over gates; candidates pooled over the gates in which the route occurs.
    info: dict[tuple, dict] = {}
    for gate, g in plan["res"].items():
        rts = g["plan"]
        for ri, r in enumerate(rts):
            if not r:
                continue
            key = tuple(int(c) for c in r)
            rec = info.setdefault(key, {"gates": [], "pool": set()})
            rec["gates"].append(gate)
            for k, o in enumerate(rts):
                if k != ri:
                    rec["pool"].update(int(c) for c in o)
    keys = sorted(info)

    move_rows: list[dict] = []
    route_rows: list[dict] = []
    violations: list[str] = []
    for rid, key in enumerate(keys):
        rec = info[key]
        L = len(key)
        rng = _rng(SCEN_SEED, name, str(key))
        prof = load_profile(key, dsc, psc)
        y = prof.max(axis=0) > Q
        PM, SM = insertion_envelopes(prof)
        p_base = float(y.mean())
        n_pool = len(rec["pool"] - set(key))
        n_cand1 = n_pool * (L + 1)

        # Class 1: insertions (D in {0, 1}).
        cands = insertion_candidates(rec["pool"], key, cap, rng)
        x1s = []
        n1 = 0
        for j, pos in cands:
            pk = insertion_peak(PM, SM, pos, dsc[j], psc[j])
            x = pk > Q
            st = window_stats(y, x)
            if st["P_Dm1"] > 0:
                violations.append(f"{name} {key} j{j} pos{pos}: D<0 on {st['P_Dm1'] * N:.0f} scenarios")
                continue
            n1 += 1
            move_rows.append(_move_row(name, fam, rid, L, 1, "insertion", j, pos, -1, st))
            x1s.append((j, pos, x))
        # Class 3: alternative positions of the same customer (base = pos, new = pos2).
        n3 = 0
        for j, pos, x1 in x1s:
            others = [q for q in range(1, L + 2) if q != pos]
            pos2 = others[int(rng.integers(len(others)))]
            x2 = insertion_peak(PM, SM, pos2, dsc[j], psc[j]) > Q
            st = window_stats(x1, x2)
            n3 += 1
            move_rows.append(_move_row(name, fam, rid, L, 3, "alt-position", j, pos, pos2, st))

        # Class 2: intra-route relocate / swap / 2-opt.
        intra, n_intra_total = intra_candidates(key, CAP_INTRA, rng, all_upto=ALL_UPTO)
        n2 = 0
        for kind, i, k, t in intra:
            x = route_peak_flags(t, dsc, psc, Q)
            st = window_stats(y, x)
            n2 += 1
            move_rows.append(_move_row(name, fam, rid, L, 2, kind, -1, i, k, st))

        # Route-level totals over classes 1 and 2 (base route key): stand-alone
        # estimates of every p' at eps, incremental = base once + increments.
        rows_here = move_rows[-(n1 + n2 + n3):] if (n1 + n2 + n3) else []
        ev = {}
        for eps in EPS_LIST:
            e = f"e{int(round(-math.log10(eps)))}"
            sa_q = sa_c = 0.0
            inc_dq = inc_dc = 0.0
            for r in rows_here:
                if r["cls"] == 3:
                    continue
                pn = r["p_new"]
                sa_q += quantum_queries(pn * (1 - pn), eps)
                sa_c += classical_queries(pn * (1 - pn), eps)
                ef = eps / math.sqrt(2.0)
                if r["cls"] == 1:
                    inc_dq += quantum_queries(r["var_D"], ef)
                    inc_dc += classical_queries(r["var_D"], ef)
                else:
                    q_, c_ = signed_costs(r["P_D1"], r["P_Dm1"], ef)
                    inc_dq += q_
                    inc_dc += c_
            ef = eps / math.sqrt(2.0)
            base_q = quantum_queries(p_base * (1 - p_base), ef)
            base_c = classical_queries(p_base * (1 - p_base), ef)
            ev[f"sa_Nq_{e}"] = sa_q
            ev[f"sa_Nc_{e}"] = sa_c
            ev[f"inc_Nq_{e}"] = base_q + inc_dq
            ev[f"inc_Nc_{e}"] = base_c + inc_dc
        zero_n = sum(1 for r in rows_here if r["zero"])
        route_rows.append({
            "instance": name, "family": fam, "route_id": rid, "L": L,
            "route": " ".join(str(c) for c in key), "gates": ";".join(sorted(set(rec["gates"]))),
            "n_gate_routes": len(rec["gates"]), "p_base": p_base,
            "n_cand1": n_cand1, "n_c1": n1, "n_c2_total": n_intra_total, "n_c2": n2,
            "n_c3": n3, "n_zero": zero_n, **ev,
        })

    if violations:
        raise RuntimeError("insertion lemma violated: " + "; ".join(violations[:5]))
    seconds = time.time() - t0
    out = {"name": name, "family": fam, "rho": rho, "N": N, "cap": cap,
           "moves": pd.DataFrame(move_rows), "routes": pd.DataFrame(route_rows),
           "seconds": seconds, "scen_seconds": t_scen, "n_unique_routes": len(keys),
           "n_gate_routes": sum(len(v["gates"]) for v in info.values())}
    parts_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, parts_dir / f"{name}.pkl")
    return {"name": name, "seconds": seconds, "scen_seconds": t_scen,
            "n_routes": len(keys), "n_moves": len(move_rows)}


def _parts_dir(rho: float, N: int) -> Path:
    return PARTS_ROOT / f"rho{rho:g}_N{N}_seed{SCEN_SEED}"


def run_part_v2(args) -> None:
    paths = _list_instances(args.instances, args.every)
    if not paths:
        raise SystemExit(f"No plan files match {args.instances!r}")
    parts_dir = _parts_dir(args.rho, args.N)
    todo = []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        if args.force or not (parts_dir / f"{name}.pkl").exists():
            todo.append((str(p), args.rho, args.N, args.cap, parts_dir))
    log(f"v2 moves: {len(paths)} instances selected, {len(todo)} to compute; rho={args.rho} "
        f"N={args.N} cap={args.cap} seed={SCEN_SEED} workers={args.workers}")
    t0 = time.time()
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for k, info in enumerate(pool.imap_unordered(_work_v2, todo), 1):
                log(f"  [{k}/{len(todo)}] {info['name']}: {info['n_routes']} unique routes, "
                    f"{info['n_moves']} moves, scenarios {info['scen_seconds']:.1f} s, "
                    f"total {info['seconds']:.1f} s")
    log(f"v2 compute finished in {time.time() - t0:.1f} s; aggregating")
    write_outputs_v2(args.rho, args.N, paths, parts_dir)


def _load_parts(paths, parts_dir):
    moves, routes, meta = [], [], []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        f = parts_dir / f"{name}.pkl"
        if not f.exists():
            continue
        d = pd.read_pickle(f)
        moves.append(d["moves"].assign(_inst_key=name))
        routes.append(d["routes"])
        meta.append({k: d[k] for k in ("name", "family", "N", "rho", "seconds", "scen_seconds",
                                       "n_unique_routes", "n_gate_routes")})
    return (pd.concat(moves, ignore_index=True) if moves else pd.DataFrame(),
            pd.concat(routes, ignore_index=True) if routes else pd.DataFrame(), meta)


def write_outputs_v2(rho: float, N: int, paths, parts_dir: Path) -> None:
    moves, routes, meta = _load_parts(paths, parts_dir)
    tag = _tag(rho)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    moves = moves.drop(columns=["_inst_key"], errors="ignore")
    gz = OUT_DIR / f"m3a_v2{tag}_moves.csv.gz"
    moves.to_csv(gz, index=False, compression="gzip", float_format="%.5g")
    size_mb = gz.stat().st_size / 1e6
    if size_mb > 50:
        sample = moves.sample(frac=0.10, random_state=SCEN_SEED)
        sample.to_csv(OUT_DIR / f"m3a_v2{tag}_moves_sample10.csv.gz", index=False,
                      compression="gzip", float_format="%.5g")
    routes.to_csv(OUT_DIR / f"m3a_v2{tag}_routes.csv", index=False, float_format="%.5g")
    pd.DataFrame(meta).to_csv(OUT_DIR / f"m3a_v2{tag}_instances.csv", index=False)
    log(f"wrote m3a_v2{tag}_moves.csv.gz ({len(moves)} rows, {size_mb:.1f} MB), "
        f"m3a_v2{tag}_routes.csv ({len(routes)} unique routes)")


# ------------------------------------------------------------ discretisation check

def _disc_select(routes: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Two routes per (family, base-p stratum) at random, from the rho = 0.6 run."""
    rng = np.random.default_rng([seed, 1])
    df = routes.copy()
    df["stratum"] = df["p_base"].map(_stratum)
    picks = []
    for fam in FAMILIES:
        for _, _, lab in STRATA:
            pool = df[(df["family"] == fam) & (df["stratum"] == lab)]
            if pool.empty:
                continue
            idx = rng.choice(len(pool), size=min(2, len(pool)), replace=False)
            picks.append(pool.iloc[sorted(int(i) for i in idx)])
    return pd.concat(picks, ignore_index=True) if picks else df.iloc[0:0]


def _work_disc(task: tuple) -> list[dict]:
    path, rows, rho, N, bs = task
    plan = json.loads(Path(path).read_text())
    name = plan["name"]
    fam = family_of(name)
    Q = float(plan["Q"])
    dsc, psc = draw_scenarios(plan, N, rho, SCEN_SEED)
    out = []
    models = {}
    for b in bs:
        for rule in LEVEL_RULES:
            models[(b, rule)] = copula_demand(plan, b, rho, DISC_NODES, DEFAULT_CV, rule)
    for r in rows:
        route = [int(c) for c in r["route"].split()]
        flags = route_peak_flags(route, dsc, psc, Q)
        p_mc = float(flags.mean())
        se = math.sqrt(max(p_mc * (1 - p_mc), 0.0) / N)
        for b, rule in ((b, rule) for b in bs for rule in LEVEL_RULES):
            if b == max(bs) and (rule != "design" or len(route) > DISC_B6_MAX_L):
                continue
            cd = models[(b, rule)]
            p_d = chance_copula(route, cd)
            bias = p_d - p_mc
            out.append({
                "instance": name, "family": fam, "route_id": int(r["route_id"]), "rule": rule,
                "stratum": r["stratum"], "L": len(route), "b": b, "V": 2 ** b,
                "unit": cd.unit, "Q_level": cd.Q_level, "p_disc": p_d, "p_mc": p_mc,
                "mc_se": se, "bias": bias, "bias_lo": bias - 1.96 * se,
                "bias_hi": bias + 1.96 * se, "rel_bias": bias / p_mc if p_mc > 0 else float("nan"),
                "covers0": (bias - 1.96 * se) <= 0.0 <= (bias + 1.96 * se), "n_mc": N,
                "nodes": DISC_NODES,
            })
    return out


def run_part_disc(args) -> None:
    parts_dir = _parts_dir(DEFAULT_RHO, args.N)
    paths = _list_instances(None)
    _, routes, _ = _load_parts(paths, parts_dir)
    if routes.empty:
        raise SystemExit("run --part v2 for rho=0.6 first")
    sel = _disc_select(routes, DISC_SEED)
    log(f"disc: {len(sel)} routes selected (2 per family x stratum where available)")
    by_inst = {}
    for _, r in sel.iterrows():
        by_inst.setdefault(r["instance"], []).append(r)
    tasks = []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        if name in by_inst:
            tasks.append((str(p), by_inst[name], DEFAULT_RHO, args.N, [4, 5, 6]))
    rows: list[dict] = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_work_disc, tasks), 1):
            rows.extend(res)
            log(f"  disc [{k}/{len(tasks)}] {res[0]['instance'] if res else ''} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "m3a_v2_discretisation.csv", index=False, float_format="%.5g")
    log(f"wrote m3a_v2_discretisation.csv ({len(df)} rows)")


def _qs(values: pd.Series) -> list[str]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return ["-"] * 5
    return [f"{x:.4g}" for x in v.quantile([0.5, 0.10, 0.25, 0.75, 0.90]).tolist()]


def _med(values: pd.Series) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return f"{v.median():.4g}" if len(v) else "-"


def _fam_stats(routes: pd.DataFrame, eps_tag: str) -> tuple[float, float, float, float]:
    sq = routes[f"sa_Nq_{eps_tag}"].sum()
    iq = routes[f"inc_Nq_{eps_tag}"].sum()
    sc = routes[f"sa_Nc_{eps_tag}"].sum()
    ic = routes[f"inc_Nc_{eps_tag}"].sum()
    return sq, iq, sc, ic


def summary_v2() -> str:
    mv = pd.read_csv(OUT_DIR / "m3a_v2_moves.csv.gz")
    rt = pd.read_csv(OUT_DIR / "m3a_v2_routes.csv")
    inst = pd.read_csv(OUT_DIR / "m3a_v2_instances.csv")
    mv["stratum"] = mv["p"].map(_stratum)
    has0 = (OUT_DIR / "m3a_v2_rho0_routes.csv").exists()
    if has0:
        mv0 = pd.read_csv(OUT_DIR / "m3a_v2_rho0_moves.csv.gz")
        rt0 = pd.read_csv(OUT_DIR / "m3a_v2_rho0_routes.csv")
        inst0 = pd.read_csv(OUT_DIR / "m3a_v2_rho0_instances.csv")
        mv0["stratum"] = mv0["p"].map(_stratum)
    L: list[str] = []
    n_inst = inst["name"].nunique()
    gate_pairs = int(rt["n_gate_routes"].sum())
    n_cl = mv["cls"].value_counts().to_dict()
    secs = float(inst["seconds"].sum())
    scen = float(inst["scen_seconds"].sum())
    L.append("# QUASAR M3a v2: coherent CRN statistics on the BATON copula model\n")
    L.append(f"Demand model: BATON's one-factor Gaussian copula (RHO = {DEFAULT_RHO}) on gamma "
             f"marginals (cv = {DEFAULT_CV}); deliveries and pickups carry independent factors, "
             "exactly as `dethloff_runner.make_scenarios`. Continuous model, no discretisation.")
    L.append(f"Common scenarios: N = {int(inst['N'].iloc[0])} per instance, seed {SCEN_SEED} "
             "(the runner's SEED), shared by every route and move of that instance.\n")
    L.append(f"Instances: {n_inst}; unique routes: {len(rt)} (from {gate_pairs} gate-route pairs; "
             "identical routes under several gates are evaluated once); moves: "
             f"{len(mv)} (insertion {n_cl.get(1, 0)}, intra-route {n_cl.get(2, 0)}, "
             f"alternative positions {n_cl.get(3, 0)}).")
    L.append(f"Insertion-lemma violations: 0 (a violation aborts the run). Sum of per-instance "
             f"times over all workers: {secs / 60:.1f} min, of which scenario draws "
             f"{scen / 60:.1f} min (3 workers ran in parallel; instances cached before a code "
             "speed-up keep their earlier timings).\n")
    L.append("Caps: insertion candidates are a uniform sample of 200 per route "
             f"({int((rt['n_cand1'] > rt['n_c1']).sum())} of {len(rt)} routes at the cap); "
             f"intra-route neighbours are all for |r| <= {ALL_UPTO} and a uniform sample of "
             f"{CAP_INTRA} otherwise ({int((rt['n_c2'] < rt['n_c2_total']).sum())} routes at the cap); "
             "alternative positions: one random second position per sampled insertion.\n")
    L.append("Definitions: D = 1[F' > Q] - 1[F > Q] over common scenarios; p = P(F > Q) (base), "
             "p' = P(F' > Q) (new route); rho_q = sqrt(Var D)/sqrt(p'(1-p')); rho_c = Var D/(p'(1-p')). "
             "Standard errors by the delta method on the per-scenario influence functions "
             "(stored per move in the CSV). Rows with p' in {0, 1} have NaN ratios and are "
             "excluded from quantiles; the NaN count is in the header of each table.\n")

    # ---- Table A
    L.append("## A. Move class x family x base-p stratum\n")
    L.append("Quantiles of rho_q (median, p10, p25, p75, p90); n = moves with defined rho_q; "
             "frac Var=0 = share of moves with no scenario where D != 0 (over all moves in the cell).\n")
    L.append("| class | family | base p | n | median | p10 | p25 | p75 | p90 | frac Var=0 | NaN |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cls in (1, 2, 3):
        for fam in FAMILIES + ("all",):
            for _, _, lab in STRATA:
                sub = mv[(mv["cls"] == cls) & (mv["stratum"] == lab)]
                if fam != "all":
                    sub = sub[sub["family"] == fam]
                if sub.empty:
                    continue
                nn = int(pd.to_numeric(sub["rho_q"], errors="coerce").notna().sum())
                qs = _qs(sub["rho_q"])
                L.append(f"| {CLASS_NAMES[cls]} | {fam} | {lab} | {nn} | {' | '.join(qs)} | "
                         f"{sub['zero'].mean():.3f} | {len(sub) - nn} |")
    L.append("")
    L.append("Classes: insertion = inter-route insertion (D in {0,1}); intra = relocate, swap "
             "or 2-opt inside the route (D in {-1,0,1}); alt-position = two positions of the "
             "same inserted customer (base = pos, new = pos2; base p is that insertion's p').\n")

    # ---- Table B
    L.append("## B. Incremental feasibility totals (classes 1 and 2; base route once + increments)\n")
    L.append("Stand-alone: each candidate p' estimated at eps. Incremental: base p at eps/sqrt(2), "
             "each increment D at eps/sqrt(2) (signed D: two estimates at half that). "
             "Query counts: canonical AE, 5 x (2M+1) A-applications; classical z^2 var/eps^2, z = 1.96.\n")
    for eps_tag, eps in (("e3", 1e-3), ("e2", 1e-2), ("e4", 1e-4)):
        L.append(f"### eps = {eps:g}\n")
        L.append("| family | routes | stand-alone Nq | incremental Nq | Nq ratio | "
                 "stand-alone Nc | incremental Nc | Nc ratio |\n|---|---|---|---|---|---|---|---|")
        for fam in FAMILIES + ("all",):
            s = rt if fam == "all" else rt[rt["family"] == fam]
            sq, iq, sc, ic = _fam_stats(s, eps_tag)
            L.append(f"| {fam} | {len(s)} | {sq:.4g} | {iq:.4g} | {iq / sq:.4f} | "
                     f"{sc:.4g} | {ic:.4g} | {ic / sc:.4f} |")
        L.append("")

    # ---- Table C
    L.append("## C. Sensitivity: rho = 0 (independent customers) on every "
             f"{RHO0_EVERY}th instance\n")
    if not has0:
        L.append("rho = 0 run not available.\n")
    else:
        names0 = set(inst0["name"])
        m6 = mv[mv["instance"].isin(names0)]
        r6 = rt[rt["instance"].isin(names0)]
        L.append(f"Instances: {len(names0)} ({', '.join(sorted(names0))}). Same seed; rho = 0 draws "
                 "are independent per customer (dethloff_runner.sample_demands with rho = 0).\n")
        L.append("| class | base p | n (rho=0.6) | median rho_q (0.6) | n (rho=0) | "
                 "median rho_q (0) | frac Var=0 (0.6) | frac Var=0 (0) |\n|---|---|---|---|---|---|---|---|")
        for cls in (1, 2, 3):
            for _, _, lab in STRATA:
                a = m6[(m6["cls"] == cls) & (m6["stratum"] == lab)]
                b0 = mv0[(mv0["cls"] == cls) & (mv0["stratum"] == lab)]
                if a.empty and b0.empty:
                    continue
                na = int(pd.to_numeric(a["rho_q"], errors="coerce").notna().sum())
                nb = int(pd.to_numeric(b0["rho_q"], errors="coerce").notna().sum())
                L.append(f"| {CLASS_NAMES[cls]} | {lab} | {na} | {_med(a['rho_q'])} | {nb} | "
                         f"{_med(b0['rho_q'])} | {a['zero'].mean():.3f} | {b0['zero'].mean():.3f} |")
        L.append("")
        L.append("Incremental totals on the same instances (eps = 1e-3):\n")
        L.append("| family | routes | Nq ratio (0.6) | Nq ratio (0) | Nc ratio (0.6) | Nc ratio (0) |\n"
                 "|---|---|---|---|---|---|")
        for fam in FAMILIES + ("all",):
            s6 = r6 if fam == "all" else r6[r6["family"] == fam]
            s0 = rt0 if fam == "all" else rt0[rt0["family"] == fam]
            a = _fam_stats(s6, "e3")
            b0 = _fam_stats(s0, "e3")
            L.append(f"| {fam} | {len(s6)} / {len(s0)} | {a[1] / a[0]:.4f} | {b0[1] / b0[0]:.4f} | "
                     f"{a[3] / a[2]:.4f} | {b0[3] / b0[2]:.4f} |")
        L.append("")

    # ---- Table D
    L.append("## D. Discretisation check (copula discrete model vs continuous copula Monte Carlo)\n")
    if (OUT_DIR / "m3a_v2_discretisation.csv").exists():
        dd = pd.read_csv(OUT_DIR / "m3a_v2_discretisation.csv")
        n_routes_dd = dd[["instance", "route_id"]].drop_duplicates().shape[0]
        L.append(f"Routes: {n_routes_dd} (two per family x "
                 f"base-p stratum where available); {DISC_NODES} Gauss-Hermite nodes per factor "
                 f"({DISC_NODES ** 2} node pairs); MC from the same {int(dd['n_mc'].iloc[0])} "
                 "scenarios. Bias = p_disc - p_mc with MC interval bias +- 1.96 se(p_mc). "
                 "Relative bias only where p_mc >= 0.01. Rules: design (literal) for b = 4, 5, 6 "
                 f"(b = 6 only for routes with L <= {DISC_B6_MAX_L}); nearest (review variant) for "
                 "b = 4, 5. Quadrature convergence on one long route is in table E.\n")
        L.append("| rule | family | b | base-p stratum | checks | median abs bias | max abs bias | "
                 "median abs rel bias | intervals covering 0 |\n|---|---|---|---|---|---|---|---|---|")
        dd["abs_bias"] = dd["bias"].abs()
        dd["abs_rel"] = np.where(dd["p_mc"] >= 0.01, (dd["bias"] / dd["p_mc"]).abs(), np.nan)
        for (rule, fam, b, lab), s in dd.groupby(["rule", "family", "b", "stratum"], sort=True):
            L.append(f"| {rule} | {fam} | {b} | {lab} | {len(s)} | {s['abs_bias'].median():.4g} | "
                     f"{s['abs_bias'].max():.4g} | {_med(s['abs_rel'])} | "
                     f"{int(s['covers0'].sum())} / {len(s)} |")
        L.append("")
        L.append("Overall by rule and b:\n")
        L.append("| rule | b | checks | median abs bias | max abs bias | median abs rel bias (p>=0.01) | "
                 "max abs rel bias (p>=0.01) |\n|---|---|---|---|---|---|---|")
        for (rule, b), s in dd.groupby(["rule", "b"]):
            rel = s["abs_rel"].dropna()
            L.append(f"| {rule} | {b} | {len(s)} | {s['abs_bias'].median():.4g} | {s['abs_bias'].max():.4g} | "
                     f"{rel.median():.4g} | {rel.max():.4g} |")
        L.append("")
    else:
        L.append("Discretisation check not run.\n")
    if (OUT_DIR / "m3a_v2_gh_convergence.csv").exists():
        gc = pd.read_csv(OUT_DIR / "m3a_v2_gh_convergence.csv")
        L.append("## E. Gauss-Hermite quadrature convergence (one long CMT route, design rule)\n")
        L.append("| instance | route | L | b | Gauss-Hermite nodes per factor | p_disc | p_mc | "
                 "p_disc - p_mc | MC se |\n|---|---|---|---|---|---|---|---|---|")
        for r in gc.itertuples():
            L.append(f"| {r.instance} | {r.route_id} | {r.L} | {r.b} | {r.nodes} | {r.p_disc:.5f} | "
                     f"{r.p_mc:.5f} | {r.p_disc - r.p_mc:+.5f} | {r.mc_se:.5f} |")
        L.append("")
    return "\n".join(L)


def write_summary_v2() -> None:
    text = summary_v2()
    (OUT_DIR / "m3a_v2_summary.md").write_text(text, encoding="utf-8")
    log(f"wrote m3a_v2_summary.md ({len(text.splitlines())} lines)")


# ============================================================
# Part v3: interpretation layer (design status after review of revision 2)
# ============================================================
#   v3      scenario pass (same seed and N as v2) for the end-load lemma (every
#           intra move, every route) and the paired level-only check; writes
#           m3a_v3_*.csv; per-instance cache under m3a_v3_parts/
#   v3summary  writes m3a_v3_summary.md from the v2 moves/routes and the v3 CSVs

V3_PARTS = OUT_DIR / "m3a_v3_parts"
V3_PAIRED_B = (4, 5, 6)
PAIRED_PER_CELL = 6            # routes per family x base-p stratum (as many as exist, up to this)
PAIRED_SEED = 1
BIAS_REL_MED = 0.01            # b recommended when median |rel bias| (p >= 0.01) <= 1 %
BIAS_REL_MAX = 0.05            #   and max |rel bias| <= 5 %
DECISION_BAND = (0.02, 0.2)    # decision band of p' (review fix 6)
COST_BAND = (0.02, 0.5)        # band of p' over which cost sums are taken (review fix 2)
V3_EPS_ABS = (1e-3, 1e-4)


def _v3_routes_for_stratified_check(routes: pd.DataFrame, seed: int) -> set[tuple[str, int]]:
    """Up to PAIRED_PER_CELL routes per family x base-p stratum (random, seeded)."""
    rng = np.random.default_rng([seed, PAIRED_SEED])
    df = routes.copy()
    df["stratum"] = df["p_base"].map(_stratum)
    chosen: set[tuple[str, int]] = set()
    for fam in FAMILIES:
        for _, _, lab in STRATA:
            pool = df[(df["family"] == fam) & (df["stratum"] == lab)]
            if pool.empty:
                continue
            k = min(PAIRED_PER_CELL, len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            for i in idx:
                r = pool.iloc[int(i)]
                chosen.add((str(r["instance"]), int(r["route_id"])))
    return chosen


def _work_v3(task: tuple) -> dict:
    path, rrows, irows, paired_ids, N, rho = task
    t0 = time.time()
    plan = json.loads(Path(path).read_text())
    name = plan["name"]
    fam = family_of(name)
    Q = float(plan["Q"])
    dsc, psc = draw_scenarios(plan, N, rho, SCEN_SEED)
    dbar = np.asarray(plan["dbar"], dtype=np.float64)
    pbar = np.asarray(plan["pbar"], dtype=np.float64)
    q_max = q999_max(dbar, pbar, DEFAULT_CV)
    by_route: dict[int, list[dict]] = {}
    for r in irows:
        by_route.setdefault(int(r["route_id"]), []).append(r)

    route_out, move_out, paired_out = [], [], []
    for rr in rrows:
        rid = int(rr["route_id"])
        route = [int(c) for c in str(rr["route"]).split()]
        Ld = sum_loads(route, dsc)
        Lp = sum_loads(route, psc)
        end_ok = (Ld <= Q) & (Lp <= Q)
        over_max = (Ld > Q) | (Lp > Q)
        y = route_peak(route, dsc, psc) > Q
        union = end_ok & y
        n_viol = 0
        n_zero = 0
        max_pd = 0.0
        max_cache_diff = 0.0
        mv_rows = by_route.get(rid, [])
        for m in mv_rows:
            t = reconstruct_intra(route, m["kind"], int(m["pos"]), int(m["pos2"]))
            x = route_peak(t, dsc, psc) > Q
            D = x.astype(np.int8) - y.astype(np.int8)
            p1 = float(np.mean(D == 1))
            pm = float(np.mean(D == -1))
            viol = end_load_violations(D, end_ok)
            zero = not bool(np.any(D != 0))
            cdiff = max(abs(p1 - float(m["P_D1"])), abs(pm - float(m["P_Dm1"])))
            n_viol += viol
            n_zero += int(zero)
            max_pd = max(max_pd, p1 + pm)
            max_cache_diff = max(max_cache_diff, cdiff)
            union = union | (end_ok & x)
            move_out.append({"instance": name, "family": fam, "route_id": rid, "kind": m["kind"],
                             "pos": int(m["pos"]), "pos2": int(m["pos2"]), "L": len(route),
                             "P_D1": p1, "P_Dm1": pm, "pd_ne0": p1 + pm, "zero": zero,
                             "viol": viol, "cache_diff": cdiff})
        route_out.append({
            "instance": name, "family": fam, "route_id": rid, "L": len(route),
            "p_base": float(y.mean()), "P_maxdl": float(over_max.mean()),
            "P_end_ok": float(end_ok.mean()), "n_intra": len(mv_rows), "n_zero": n_zero,
            "B_spec": float(np.mean(end_ok & y)), "B_union": float(np.mean(union)),
            "max_pd_ne0": max_pd, "viol": n_viol, "max_cache_diff": max_cache_diff,
        })
        if (name, rid) in paired_ids:
            for b in V3_PAIRED_B:
                V = 2 ** b
                for rule in LEVEL_RULES:
                    K, unit = level_rule(Q, V, q_max, rule)
                    res = paired_level_check(route, dsc, psc, Q, unit, V, K)
                    paired_out.append({"instance": name, "family": fam, "route_id": rid,
                                       "L": len(route), "p_base": float(y.mean()),
                                       "stratum": _stratum(float(y.mean())), "b": b,
                                       "rule": rule, "K": K, "unit": unit, **res})
    seconds = time.time() - t0
    out = {"name": name, "family": fam, "seconds": seconds, "N": N,
           "routes": route_out, "moves": move_out, "paired": paired_out}
    V3_PARTS.mkdir(parents=True, exist_ok=True)
    with open(V3_PARTS / f"{name}.pkl", "wb") as fh:
        import pickle
        pickle.dump(out, fh)
    return {"name": name, "seconds": seconds, "n_routes": len(route_out),
            "n_moves": len(move_out), "viol": sum(r["viol"] for r in route_out)}


def run_part_v3(args) -> None:
    paths = _list_instances(args.instances, args.every)
    if not paths:
        raise SystemExit(f"No plan files match {args.instances!r}")
    parts_dir = _parts_dir(DEFAULT_RHO, args.N)
    routes = pd.read_csv(OUT_DIR / "m3a_v2_routes.csv")
    moves = pd.read_csv(OUT_DIR / "m3a_v2_moves.csv.gz",
                        usecols=["instance", "route_id", "cls", "kind", "pos", "pos2",
                                 "P_D1", "P_Dm1"])
    intra = moves[moves["cls"] == 2].drop(columns=["cls"])
    # Selections use every route of the v2 cache, so partial runs cache the same samples.
    paired_ids = _v3_routes_for_stratified_check(routes, DISC_SEED)
    names = {json.loads(p.read_text())["name"] for p in paths}
    routes = routes[routes["instance"].isin(names)].reset_index(drop=True)
    intra = intra[intra["instance"].isin(names)]
    log(f"v3: {len(paths)} instances, {len(routes)} routes, {len(intra)} intra moves, "
        f"{len(paired_ids)} routes for the paired level check")
    by_inst_r = {n: g for n, g in routes.groupby("instance")}
    by_inst_i = {n: g for n, g in intra.groupby("instance")}
    todo = []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        if name not in by_inst_r:
            continue
        if args.force or not (V3_PARTS / f"{name}.pkl").exists():
            todo.append((str(p), by_inst_r[name].to_dict("records"),
                         by_inst_i[name].to_dict("records") if name in by_inst_i else [],
                         paired_ids, args.N, DEFAULT_RHO))
    log(f"v3 scenario pass: {len(todo)} instances to compute (N={args.N}, seed={SCEN_SEED})")
    t0 = time.time()
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for k, info in enumerate(pool.imap_unordered(_work_v3, todo), 1):
                log(f"  v3 [{k}/{len(todo)}] {info['name']}: {info['n_routes']} routes, "
                    f"{info['n_moves']} intra moves, violations {info['viol']}, "
                    f"{info['seconds']:.1f} s")
    log(f"v3 compute finished in {time.time() - t0:.1f} s; aggregating")
    write_outputs_v3(paths, parts_dir)


def write_outputs_v3(paths, parts_dir: Path) -> None:
    rs, ms, ps = [], [], []
    names = []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        f = V3_PARTS / f"{name}.pkl"
        if not f.exists():
            continue
        import pickle
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        names.append(name)
        rs += d["routes"]
        ms += d["moves"]
        ps += d["paired"]
    pd.DataFrame(rs).to_csv(OUT_DIR / "m3a_v3_endload_routes.csv", index=False, float_format="%.6g")
    mdf = pd.DataFrame(ms)
    mdf.to_csv(OUT_DIR / "m3a_v3_intra_check.csv.gz", index=False, compression="gzip",
               float_format="%.6g")
    pd.DataFrame(ps).to_csv(OUT_DIR / "m3a_v3_paired.csv", index=False, float_format="%.6g")
    log(f"wrote m3a_v3 CSVs: {len(rs)} routes, {len(mdf)} intra moves, "
        f"{len(ps)} paired rows, {int(mdf['viol'].sum()) if len(mdf) else 0} "
        "end-load violations")



# ============================================================
# Part v3gh: quadrature check (Revision 3)
# ============================================================
#   Gauss-Hermite nodes against equal-probability factor bins (2**f), per copula
#   factor, on the copula window probability of nine routes at b = 5 and b = 6,
#   and against continuous copula Monte Carlo on the same N scenarios. Exact
#   window probabilities (no Monte Carlo inside the quadrature). One JSON per
#   task under m3a_v3_gh_parts/ (resumable); m3a_v3_gh.csv collects them.

GH_SEED = 1                    # route selection seed
GH_B = (5, 6)
GH_NODES = (16, 32, 64, 128)   # Gauss-Hermite nodes per factor
GH_EM_F = (3, 4, 5, 6, 7)      # equal-probability bins per factor: 2**f
GH_TOL = 1e-4                  # rule: max |p(n) - p_ref| over the routes <= GH_TOL
GH_DIR = OUT_DIR / "m3a_v3_gh_parts"
GH_CSV = OUT_DIR / "m3a_v3_gh.csv"
GH_COST_B6 = 7                 # b = 6 pair cost relative to b = 5 (measured)
GH_EMAVG_MAX_B6 = 32       # bin-averaged grids at b = 6 up to 32 bins (64 bins: too slow for all routes)
GH_CFG_ORDER = ("GH16", "GH32", "GH64", "GH128", "EMm8", "EMm16", "EMm32", "EMm64", "EMm128",
                "EMa8", "EMa16", "EMa32", "EMa64", "EMa128")


def _gh_selection(routes: pd.DataFrame, seed: int = GH_SEED) -> list[dict]:
    """Nine routes: the three longest (CMT, city, CON/SCA) and two mixed-length routes in
    each band 3-8, 9-15, 16-25 (seeded; base p in (0.05, 0.95) so the window is informative)."""
    chosen: list[dict] = []
    for fams in (("CMT",), ("city",), ("CON", "SCA")):
        pool = routes[routes["family"].isin(fams)].sort_values(["L", "instance", "route_id"],
                                                               ascending=[False, True, True])
        r = pool.iloc[0]
        chosen.append({"instance": str(r["instance"]), "route_id": int(r["route_id"]),
                       "family": str(r["family"]), "long": True})
    taken = {(c["instance"], c["route_id"]) for c in chosen}
    elig = routes[(routes["p_base"] > 0.05) & (routes["p_base"] < 0.95) & (routes["L"] >= 3)]
    rng = np.random.default_rng([seed, 4])
    for lo, hi in ((3, 8), (9, 15), (16, 25)):
        pool = elig[(elig["L"] >= lo) & (elig["L"] <= hi)].sort_values(["instance", "route_id"])
        pool = pool[[(str(a), int(b)) not in taken for a, b in zip(pool["instance"], pool["route_id"])]]
        for i in rng.choice(len(pool), size=2, replace=False):
            r = pool.iloc[int(i)]
            chosen.append({"instance": str(r["instance"]), "route_id": int(r["route_id"]),
                           "family": str(r["family"]), "long": False})
            taken.add((str(r["instance"]), int(r["route_id"])))
    return chosen


def _gh_configs(b: int) -> list[tuple[str, int]]:
    """(grid, points) for one b: 128 points at b = 5; up to 64 points at b = 6 (a 128-bin
    b = 6 pair grid was too slow for the time available). Grids: "gh" Gauss-Hermite nodes,
    "em" equal-probability bins with midpoint pmfs, "em_avg" equal-probability bins with
    bin-averaged pmfs (Revision 3b)."""
    out = ([("gh", n) for n in GH_NODES] + [("em", 2 ** f) for f in GH_EM_F]
           + [("em_avg", 2 ** f) for f in GH_EM_F])
    if b == 6:
        out = [c for c in out if c[1] <= 64 and (c[0] != "em_avg" or c[1] <= GH_EMAVG_MAX_B6)]
    return out


def _gh_task_file(inst: str, rid: int, b: int, grid: str, points: int) -> Path:
    return GH_DIR / f"{inst}_r{rid}_b{b}_{grid}{points}.json"


def _work_gh(task: tuple) -> dict:
    (plan_path, inst, fam, rid, route, L, long_route, b, grid, points,
     rho, p_mc, mc_se, out_path) = task
    plan = json.loads(Path(plan_path).read_text())
    t0 = time.time()
    cd = copula_demand(plan, b, rho, points, DEFAULT_CV, "nearest", grid)
    p_disc = chance_copula(route, cd)
    row = {"instance": inst, "family": fam, "route_id": int(rid), "L": int(L), "long": bool(long_route),
           "b": int(b), "grid": grid, "points": int(points),
           "f": int(round(math.log2(points))) if grid.startswith("em") else None,
           "p_disc": float(p_disc), "p_mc": float(p_mc), "mc_se": float(mc_se),
           "K": int(cd.Q_level), "unit": float(cd.unit), "seconds": time.time() - t0}
    Path(out_path).write_text(json.dumps(row))
    return row


def run_part_gh(args) -> None:
    routes = pd.read_csv(OUT_DIR / "m3a_v2_routes.csv")
    sel = _gh_selection(routes, GH_SEED)
    GH_DIR.mkdir(parents=True, exist_ok=True)
    N = args.N
    tasks = []
    for item in sel:
        inst, rid = item["instance"], item["route_id"]
        plan_path = PLANS_DIR / f"{inst}.json"
        plan = json.loads(plan_path.read_text())
        route = [int(c) for c in routes[(routes["instance"] == inst) & (routes["route_id"] == rid)]["route"]
                 .iloc[0].split()]
        # The Monte Carlo reference (same N and seed for every grid) is stored in each cached task file;
        # read it from there when a file exists, so that the scenario draws are not repeated.
        cached = [_gh_task_file(inst, rid, b0, g0, p0) for b0 in GH_B for g0, p0 in _gh_configs(b0)]
        cached = [f for f in cached if f.exists()]
        if cached:
            prev = json.loads(cached[0].read_text())
            p_mc, mc_se = float(prev["p_mc"]), float(prev["mc_se"])
        else:
            dsc, psc = draw_scenarios(plan, N, DEFAULT_RHO, SCEN_SEED)
            p_mc = float(route_peak_flags(route, dsc, psc, float(plan["Q"])).mean())
            mc_se = math.sqrt(max(p_mc * (1 - p_mc), 0.0) / N)
        for b in GH_B:
            for grid, points in _gh_configs(b):
                out_path = _gh_task_file(inst, rid, b, grid, points)
                if args.force or not out_path.exists():
                    tasks.append((str(plan_path), inst, item["family"], rid, route, len(route),
                                  item["long"], b, grid, points, DEFAULT_RHO, p_mc, mc_se, str(out_path)))
    tasks.sort(key=lambda t: -(t[9] ** 2) * (GH_COST_B6 if t[7] == 6 else 1))
    log(f"v3gh: {len(sel)} routes, {len(tasks)} tasks to compute (N={N}, workers={args.workers})")
    t0 = time.time()
    if tasks:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for k, row in enumerate(pool.imap_unordered(_work_gh, tasks), 1):
                log(f"  v3gh [{k}/{len(tasks)}] {row['instance']} r{row['route_id']} b{row['b']} "
                    f"{row['grid']}{row['points']}: p = {row['p_disc']:.8f} ({row['seconds']:.1f} s)")
    log(f"v3gh compute finished in {time.time() - t0:.1f} s; aggregating")
    write_outputs_gh(sel)


def write_outputs_gh(sel: list[dict]) -> None:
    rows = []
    for item in sel:
        for b in GH_B:
            for grid, points in _gh_configs(b):
                f = _gh_task_file(item["instance"], item["route_id"], b, grid, points)
                if f.exists():
                    rows.append(json.loads(f.read_text()))
    df = pd.DataFrame(rows)
    df.to_csv(GH_CSV, index=False, float_format="%.10g")
    log(f"wrote m3a_v3_gh.csv: {len(df)} rows")


# ------------------------------------------------------------ v3crn: CRN on the factor grid

# Revision 3b: the window probabilities of the CRN quantities P(D = +1), P(D = -1) on the
# discrete copula model, for a short, a medium and a long route, 6 intra and 6 alternative-
# position moves each (seeded, among moves with P(D != 0) > 1e-3 in the continuous MC),
# under the bin-averaged grids (2^f, f = 3..6) and Gauss-Hermite 64 and 128 nodes. Each factor
# pair is a conditional Monte Carlo with CRN_K draws (see crn_factor_grid); the standard error
# of each grid sum is written next to it. One JSON per (route, grid) under m3a_v3_crn_parts/.

CRN_DIR = OUT_DIR / "m3a_v3_crn_parts"
CRN_CSV = OUT_DIR / "m3a_v3_crn.csv"
CRN_B = 5
CRN_K = 4000                       # scenarios per factor pair
CRN_SEED = 11
CRN_PER_CLASS = 6
CRN_MIN_NONZERO = 1e-3
CRN_GRIDS = (("em_avg", 8), ("em_avg", 16), ("em_avg", 32), ("em_avg", 64), ("gh", 64), ("gh", 128))
CRN_COST = {"em_avg": 1.0, "gh": 1.0}


def _insert_seq(seq: list[int], j: int, pos: int) -> list[int]:
    return seq[: pos - 1] + [int(j)] + seq[pos - 1:]


def _crn_routes(routes: pd.DataFrame) -> list[dict]:
    """Short (smallest L of the nine-route selection), medium (9 <= L <= 15), long (largest L)."""
    sel = _gh_selection(routes, GH_SEED)
    short = min(sel, key=lambda r: routes_L(routes, r))
    long_ = max(sel, key=lambda r: routes_L(routes, r))
    med = [r for r in sel if 9 <= routes_L(routes, r) <= 15]
    out = []
    for role, r in (("short", short), ("medium", med[0] if med else None), ("long", long_)):
        if r is None:
            continue
        row = routes[(routes["instance"] == r["instance"]) & (routes["route_id"] == r["route_id"])].iloc[0]
        out.append({"role": role, "instance": r["instance"], "route_id": int(r["route_id"]),
                    "family": r["family"], "L": int(row["L"]),
                    "route": [int(c) for c in str(row["route"]).split()]})
    return out


def routes_L(routes: pd.DataFrame, r: dict) -> int:
    return int(routes[(routes["instance"] == r["instance"]) & (routes["route_id"] == r["route_id"])]["L"].iloc[0])


def _crn_moves(mv: pd.DataFrame, rt: dict) -> list[dict]:
    """6 intra and 6 alternative-position moves of one route, seeded, with P(D != 0) > CRN_MIN_NONZERO."""
    rows = mv[(mv["instance"] == rt["instance"]) & (mv["route_id"] == rt["route_id"])]
    key = rt["route"]
    out: list[dict] = []
    for cls in (2, 3):
        sub = rows[(rows["cls"] == cls) & ((rows["P_D1"] + rows["P_Dm1"]) > CRN_MIN_NONZERO)]
        sub = sub.sort_values(["j", "pos", "pos2", "kind"]).reset_index(drop=True)
        n_el = len(sub)
        rng = np.random.default_rng([CRN_SEED, rt["route_id"], cls, rt["L"]])
        pick = sorted(rng.choice(n_el, size=min(CRN_PER_CLASS, n_el), replace=False)) if n_el else []
        for i in pick:
            r = sub.iloc[int(i)]
            if cls == 2:
                a = key
                b = reconstruct_intra(key, str(r["kind"]), int(r["pos"]), int(r["pos2"]))
                j = -1
            else:
                j = int(r["j"])
                a = _insert_seq(key, j, int(r["pos"]))
                b = _insert_seq(key, j, int(r["pos2"]))
            out.append({"cls": cls, "kind": str(r["kind"]), "j": j, "pos": int(r["pos"]), "pos2": int(r["pos2"]),
                        "a": [int(c) for c in a], "b": [int(c) for c in b], "n_eligible": n_el,
                        "P_D1_cont": float(r["P_D1"]), "P_Dm1_cont": float(r["P_Dm1"])})
    return out


def _crn_task_file(rt: dict, grid: str, points: int) -> Path:
    return CRN_DIR / f"{rt['instance']}_r{rt['route_id']}_{grid}{points}.json"


def _work_crn(task: tuple) -> dict:
    plan_path, rt, moves, grid, points, out_path = task
    plan = json.loads(Path(plan_path).read_text())
    t0 = time.time()
    cd = copula_demand(plan, CRN_B, DEFAULT_RHO, points, DEFAULT_CV, "nearest", grid)
    p_exact = float(chance_copula(rt["route"], cd))
    pairs = [(rt["route"], rt["route"])] + [(m["a"], m["b"]) for m in moves]
    seed = int(np.random.default_rng([CRN_SEED, rt["route_id"], len(rt["route"])]).integers(2**31))
    res = crn_factor_grid(pairs, cd, CRN_K, seed)
    row = {"role": rt["role"], "instance": rt["instance"], "route_id": rt["route_id"], "L": rt["L"],
           "family": rt["family"], "grid": grid, "points": int(points), "K": int(cd.Q_level),
           "p_base_exact": p_exact, "p_base_mc": float(res["p_a"][0]), "p_base_se": float(np.sqrt(
               max(res["p_a"][0] * (1 - res["p_a"][0]), 0.0) / CRN_K)),
           "moves": [], "seconds": time.time() - t0}
    for k, m in enumerate(moves, start=1):
        row["moves"].append({**{x: m[x] for x in ("cls", "kind", "j", "pos", "pos2", "n_eligible",
                                                  "P_D1_cont", "P_Dm1_cont")},
                             "P_D1": float(res["P_D1"][k]), "se_D1": float(res["se_D1"][k]),
                             "P_Dm1": float(res["P_Dm1"][k]), "se_Dm1": float(res["se_Dm1"][k]),
                             "p_a": float(res["p_a"][k]), "p_b": float(res["p_b"][k])})
    Path(out_path).write_text(json.dumps(row))
    return row


def run_part_crn(args) -> None:
    routes = pd.read_csv(OUT_DIR / "m3a_v2_routes.csv")
    mv = pd.read_csv(OUT_DIR / "m3a_v2_moves.csv.gz")
    CRN_DIR.mkdir(parents=True, exist_ok=True)
    rts = _crn_routes(routes)
    tasks = []
    for rt in rts:
        moves = _crn_moves(mv, rt)
        rt["moves_n"] = len(moves)
        plan_path = PLANS_DIR / f"{rt['instance']}.json"
        for grid, points in CRN_GRIDS:
            out_path = _crn_task_file(rt, grid, points)
            if args.force or not out_path.exists():
                tasks.append((str(plan_path), rt, moves, grid, points, str(out_path)))
        log(f"v3crn route {rt['role']}: {rt['instance']} r{rt['route_id']} L={rt['L']} "
            f"moves={len(moves)} (eligible per class: "
            + ", ".join(str(sorted({m['n_eligible'] for m in moves if m['cls'] == c})) for c in (2, 3)) + ")")
    tasks.sort(key=lambda t: -t[4] ** 2)
    log(f"v3crn: {len(tasks)} tasks (K = {CRN_K} per factor pair, workers = {args.workers})")
    t0 = time.time()
    if tasks:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for k, row in enumerate(pool.imap_unordered(_work_crn, tasks), 1):
                log(f"  v3crn [{k}/{len(tasks)}] {row['instance']} r{row['route_id']} "
                    f"{row['grid']}{row['points']}: p_base exact {row['p_base_exact']:.6f} "
                    f"MC {row['p_base_mc']:.6f} ({row['seconds']:.1f} s)")
    log(f"v3crn compute finished in {time.time() - t0:.1f} s; aggregating")
    write_outputs_crn(rts, mv)


def write_outputs_crn(rts: list[dict], mv: pd.DataFrame) -> None:
    rows = []
    for rt in rts:
        moves = _crn_moves(mv, rt)
        for grid, points in CRN_GRIDS:
            f = _crn_task_file(rt, grid, points)
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            # the pair (route, route) carries no move: its D is 0 and p_base_* is the base tail
            for k, m in enumerate(d["moves"]):
                rows.append({"role": rt["role"], "instance": rt["instance"], "route_id": rt["route_id"],
                             "L": rt["L"], "family": rt["family"], "move_no": k + 1, "cls": m["cls"],
                             "kind": m["kind"], "j": m["j"], "pos": m["pos"], "pos2": m["pos2"],
                             "grid": grid, "points": points, "p_base_exact": d["p_base_exact"],
                             "p_base_mc": d["p_base_mc"], "p_base_se": d["p_base_se"],
                             "P_D1": m["P_D1"], "se_D1": m["se_D1"], "P_Dm1": m["P_Dm1"],
                             "se_Dm1": m["se_Dm1"], "P_D1_cont": m["P_D1_cont"], "P_Dm1_cont": m["P_Dm1_cont"],
                             "p_a": m["p_a"], "p_b": m["p_b"]})
    df = pd.DataFrame(rows)
    df.to_csv(CRN_CSV, index=False, float_format="%.10g")
    log(f"wrote m3a_v3_crn.csv: {len(df)} rows")


# ------------------------------------------------------------ v3 summary

def _fmt(x, nd=3) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.{nd}g}"


def _sci(x) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.3e}"


def _md(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _band_tables(mv: pd.DataFrame, N: int, col: str, bands, title: str,
                 inst_ci: bool = False) -> list[str]:
    """Bands of p' (or of |p' - gate|): median rho_q with a route-bootstrap CI and, when
    inst_ci, an instance-level (two-stage cluster) CI of the same median."""
    L: list[str] = [f"### {title}\n"]
    head = ["class", "family", "band", "n", "median rho_q [95% route-bootstrap CI]", "p10", "p25", "p75", "p90",
            "Var=0 share (in-sample)", "median rho_q upper bound (zeros at 3/N)", "NaN"]
    if inst_ci:
        head.insert(5, "median rho_q [95% instance-level CI]")
    rows = []
    for cls in (1, 2, 3):
        for fam in FAMILIES:
            base = mv[(mv["cls"] == cls) & (mv["family"] == fam)]
            for _, _, lab in bands:
                sub = base[base[col] == lab]
                if sub.empty:
                    continue
                rho = pd.to_numeric(sub["rho_q"], errors="coerce").to_numpy()
                ok = np.isfinite(rho)
                s = sub[ok]
                r = rho[ok]
                n_nan = int((~ok).sum())
                if r.size == 0:
                    rows.append([CLASS_NAMES[cls], fam, lab, 0] + ["-"] * (len(head) - 5) + [n_nan])
                    continue
                pt, lo, hi = route_bootstrap_quantile(r, (s["instance"] + "|" + s["route_id"].astype(str)).to_numpy(),
                                                      0.5, BOOT_REPS, seed=SCEN_SEED)
                q = np.quantile(r, [0.10, 0.25, 0.75, 0.90])
                zero = s["zero"].to_numpy(dtype=bool)
                pn = s["p_new"].to_numpy()
                ub = np.sqrt(np.maximum(s["var_D"].to_numpy(), rule_of_three_bound(N))) / np.sqrt(pn * (1 - pn))
                row = [CLASS_NAMES[cls], fam, lab, r.size, f"{_fmt(pt)} [{_fmt(lo)}, {_fmt(hi)}]"]
                if inst_ci:
                    _, ilo, ihi = route_bootstrap_quantile(r, s["instance"].to_numpy(), 0.5, BOOT_REPS,
                                                           seed=SCEN_SEED)
                    row.append(f"[{_fmt(ilo)}, {_fmt(ihi)}]")
                row += [_fmt(q[0]), _fmt(q[1]), _fmt(q[2]), _fmt(q[3]), f"{zero.mean():.3f}",
                        _fmt(float(np.median(ub))), n_nan]
                rows.append(row)
    L += _md(head, rows)
    L.append("")
    return L


def _lemma_rows(er: pd.DataFrame, im: pd.DataFrame) -> list[str]:
    head = ["family", "routes with intra moves", "intra moves", "observed Var=0 share", "median B_spec",
            "median B_union", "pooled explained share", "median per-route explained share",
            "max P(D != 0) over moves", "median per-route max P(D != 0)", "routes with B_spec < route max",
            "routes with B_union < route max", "D != 0 with an end load > Q"]
    rows = []
    for fam in FAMILIES + ("all",):
        s = er[(er["n_intra"] > 0)] if fam == "all" else er[(er["n_intra"] > 0) & (er["family"] == fam)]
        si = im if fam == "all" else im[im["family"] == fam]
        if s.empty:
            continue
        nmov = int(s["n_intra"].sum())
        zero_share = si["zero"].mean() if len(si) else float("nan")
        pooled = s["P_maxdl"].sum() / s["p_base"].sum() if s["p_base"].sum() > 0 else float("nan")
        pos = s[s["p_base"] > 0]
        per_route = pos["P_maxdl"] / pos["p_base"]
        spec_fail = int((s["max_pd_ne0"] > s["B_spec"] + 1e-12).sum())
        union_fail = int((s["max_pd_ne0"] > s["B_union"] + 1e-12).sum())
        rows.append([fam, len(s), nmov, f"{zero_share:.3f}", f"{s['B_spec'].median():.3g}",
                     f"{s['B_union'].median():.3g}", f"{pooled:.3f}", f"{per_route.median():.3f}",
                     f"{si['pd_ne0'].max():.3g}", f"{s['max_pd_ne0'].median():.3g}", spec_fail, union_fail,
                     int(s["viol"].sum())])
    return _md(head, rows)


def _cost_rows(band: pd.DataFrame, eps_arr: np.ndarray, N: int) -> tuple[list[str], dict, dict]:
    """Sums and per-move medians of query costs by class and family (certified classical cost)."""
    cls_a = band["cls"].to_numpy()
    fam_a = band["family"].to_numpy()
    P1 = band["P_D1"].to_numpy(float)
    Pm1 = band["P_Dm1"].to_numpy(float)
    pn = band["p_new"].to_numpy(float)
    c = class_costs_array(cls_a, P1, Pm1, pn, eps_arr, N)
    ratio = c["Nq"] / c["Nc"]
    rows, res = [], {}
    for cls in (1, 2, 3):
        for fam in FAMILIES:
            m = (cls_a == cls) & (fam_a == fam)
            if not m.any():
                continue
            sq_sa, sq = float(c["Nq_sa"][m].sum()), float(c["Nq"][m].sum())
            sc_sa, sc = float(c["Nc_sa"][m].sum()), float(c["Nc"][m].sum())
            med = float(np.median(ratio[m]))
            rows.append([CLASS_NAMES[cls], fam, int(m.sum()), _sci(sq_sa), _sci(sq), f"{sq / sq_sa:.3f}",
                         _sci(sc_sa), _sci(sc), f"{sc / sc_sa:.3f}", f"{sq / sc:.3g}", f"{med:.3g}"])
            res[(cls, fam)] = {"Nq_Nc": sq / sc, "med": med, "Nc_gain": sc / sc_sa, "Nq_gain": sq / sq_sa,
                               "zero_share": float(band["zero"].to_numpy()[m].mean())}
    return rows, res, c


def _one_sided_rows(band: pd.DataFrame, eps: float, N: int) -> list[list]:
    """Classes 2-3, absolute eps: the zero side of a one-sided move charged a full floor estimate
    (main tables) against the single nonzero-side estimate (M3b convention)."""
    cls_a = band["cls"].to_numpy()
    fam_a = band["family"].to_numpy()
    P1 = band["P_D1"].to_numpy(float)
    Pm1 = band["P_Dm1"].to_numpy(float)
    pn = band["p_new"].to_numpy(float)
    c = class_costs_array(cls_a, P1, Pm1, pn, np.full(cls_a.shape, eps), N)
    one = ((P1 == 0) ^ (Pm1 == 0)) & (cls_a != 1)
    v_nz = np.maximum(np.where(P1 > 0, P1 * (1 - P1), Pm1 * (1 - Pm1)), rule_of_three_bound(N))
    nq_one_single = quantum_queries_array(v_nz, eps)          # one estimate of the nonzero side at eps
    rows = []
    for cls in (2, 3):
        for fam in FAMILIES:
            m = (cls_a == cls) & (fam_a == fam)
            if not m.any():
                continue
            mo = m & one
            nq_main = float(c["Nq"][m].sum())
            nq_single = nq_main - float(c["Nq"][mo].sum()) + float(nq_one_single[mo].sum())
            nc = float(c["Nc"][m].sum())
            rows.append([CLASS_NAMES[cls], fam, int(m.sum()), int(mo.sum()), _sci(nq_main), _sci(nq_single),
                         f"{nq_main / nc:.3g}", f"{nq_single / nc:.3g}"])
    return rows


def _cfg_label(grid: str, points) -> str:
    """GH16 (Gauss-Hermite), EMm8 (equal-probability bins, midpoint pmfs), EMa8 (bin-averaged pmfs)."""
    prefix = {"gh": "GH", "em": "EMm", "em_avg": "EMa"}[grid]
    return f"{prefix}{int(points)}"


def _section_4b(gh: pd.DataFrame) -> list[str]:
    gh = gh.assign(cfg=[_cfg_label(g, p) for g, p in zip(gh["grid"], gh["points"])])
    L: list[str] = []
    A = L.append
    A("### 4b. Quadrature of the copula window probability\n")
    A("Method. The window probability P(F_r > Q) is the exact double sum over factor values "
      "sum_t sum_s w_t w_s P(F > Q | z_t, z_s), computed by the batched exact recursion (no Monte Carlo inside the "
      "quadrature; validated against classical.chance). Three factor grids per copula factor, with the same nine routes "
      "as before (three longest, six of mixed length, seeded; base p in (0.05, 0.95) for the mixed ones):\n"
      "- GH n: Gauss-Hermite nodes, n = 16, 32, 64, 128.\n"
      "- EMm 2^f: equal-probability bins, conditional pmf at the bin midpoint (the Revision 3 grid).\n"
      "- EMa 2^f: equal-probability bins, conditional pmf averaged over each bin, P(l | t) = 2^f integral over bin t of "
      "P(l | z) phi(z) dz, with 32-point Gauss-Legendre in z per bin (outer bins cut at |z| = 9). The mixture over bins "
      "reproduces the marginal level pmf to about 1e-15 (unit test).\n"
      "f = 3..7 at b = 5 and f = 3..5 at b = 6 (EMa 64 at b = 6 was not run for all routes: about four times the cost of "
      "EMa 32 per route, too slow for the time available). Gauss-Hermite and EMm at b = 6 are as before (up to 64).\n"
      "Reference. No grid is the reference. For each route and b the two converging sequences are shown side by side: "
      "the largest Gauss-Hermite grid (GH 128 at b = 5, GH 64 at b = 6) and the largest bin-averaged grid (EMa 128 at "
      "b = 5, EMa 32 at b = 6). Their disagreement is the quadrature uncertainty indicator. Continuous Monte Carlo is "
      "the same N = 200000 common scenarios as the rest of M3a; it contains the level error of Section 4a and is not a "
      "quadrature check.\n")
    cfg5 = ("GH16", "GH32", "GH64", "GH128", "EMa8", "EMa16", "EMa32", "EMa64", "EMa128")
    cfg6 = ("GH16", "GH32", "GH64", "EMa8", "EMa16", "EMa32")
    for b, cfgs, ghmax, emmax in ((5, cfg5, "GH128", "EMa128"), (6, cfg6, "GH64", "EMa32")):
        sub = gh[gh["b"] == b]
        A(f"Table 4b-1, b = {b}: absolute probability of each configuration per route (6 decimals; '-' not computed).\n")
        head = ["route", "family", "L", "p_MC (SE)"] + list(cfgs)
        rows = []
        for (inst_n, rid), s in sorted(sub.groupby(["instance", "route_id"]), key=lambda kv: -int(kv[1].iloc[0]["L"])):
            r0 = s.iloc[0]
            cells = []
            for cfg in cfgs:
                m = s[s["cfg"] == cfg]
                cells.append(f"{float(m['p_disc'].iloc[0]):.6f}" if len(m) else "-")
            rows.append([f"{inst_n} r{int(rid)}", r0["family"], int(r0["L"]),
                         f"{float(r0['p_mc']):.4f} ({float(r0['mc_se']):.4f})"] + cells)
        L += _md(head, rows)
        L.append("")
        A(f"Table 4b-2, b = {b}: the two converging references side by side and their agreement (absolute probability; "
          f"difference = {ghmax} minus {emmax}; EMa increments show the convergence rate of the bin-averaged sequence).\n")
        inc_lab = ([f"EMa128 - EMa64", "EMa64 - EMa32"] if b == 5 else [f"EMa32 - EMa16", "EMa16 - EMa8"])
        head2 = ["route", "L", ghmax, emmax, f"{ghmax} - {emmax}", inc_lab[0], inc_lab[1]]
        rows2 = []
        for (inst_n, rid), s in sorted(sub.groupby(["instance", "route_id"]), key=lambda kv: -int(kv[1].iloc[0]["L"])):
            v = {cfg: (float(s[s["cfg"] == cfg]["p_disc"].iloc[0]) if (s["cfg"] == cfg).any() else float("nan"))
                 for cfg in cfgs}
            if emmax == "EMa128":
                inc = [v["EMa128"] - v["EMa64"], v["EMa64"] - v["EMa32"]]
            else:
                inc = [v["EMa32"] - v["EMa16"], v["EMa16"] - v["EMa8"]]
            rows2.append([f"{inst_n} r{int(rid)}", int(s.iloc[0]["L"]), f"{v[ghmax]:.6f}", f"{v[emmax]:.6f}",
                          f"{v[ghmax] - v[emmax]:+.2e}", f"{inc[0]:+.2e}", f"{inc[1]:+.2e}"])
        L += _md(head2, rows2)
        L.append("")
    L.append("Reading of the two references (b = 5). (i) Short, low-probability routes (CON8-7, SCA8-1, SCA3-1): the "
             "Gauss-Hermite sequence is stable from 32 nodes (GH 64 and GH 128 differ by 2e-6 on SCA3-1 and SCA8-1, and "
             "agree on CON8-7), while the bin-averaged sequence is not yet stable at 128 bins (EMa 128 minus GH 128 is "
             "+1.1e-4 on CON8-7; the EMa increments on SCA3-1 are -3.1e-3 then -9.8e-4). The earlier attribution of the "
             "short-route discrepancy to Gauss-Hermite is withdrawn: the Gauss-Hermite values are the converged ones "
             "there, and the bin-averaged sequence is the open item. (ii) Long routes (L >= 16): Gauss-Hermite 64 is not "
             "converged (GH 64 minus GH 128 is 7.7e-4 on CMT11X and 3.9e-4 on HCMC-100-3), while the bin-averaged grid is "
             "within 1e-4 of EMa 128 from 32 to 64 bins. The two references agree within 1e-5 to 1e-4 on CMT11X, CON3-4, "
             "HCMC-400-1 and HANOI-400-1, and disagree by 2.6e-4 on HCMC-100-3, which neither grid resolves. (iii) Heuristic, "
             "not established by the data: the conditional overflow step has width about sqrt((1 - rho) / (rho L)) "
             f"(0.17 at L = 22, 0.15 at L = 28 for rho = {0.6}), which is narrower than the Gauss-Hermite bulk spacing "
             "pi / sqrt(n) at 64 nodes (0.39), so the long-route Gauss-Hermite error is consistent with the step being "
             "unresolved; the same argument does not explain the short routes, where the step is wider.\n")
    L.append("Recommended bin count for the bin-averaged grid: f = 6 (64 bins) for long routes (L >= 16). On the five long "
             "routes, 32 bins are already within 7.7e-5 of EMa 128 (HCMC-100-3, the largest) and 64 bins within 5.6e-5 "
             "(HCMC-100-3 again), so f = 5 is the smallest grid that passes the 1e-4 tolerance there, and f = 6 lowers the "
             "largest gap; f = 6 is the recommendation for the register. The rule table above reports the smallest bin count "
             "within tolerance on all nine routes, which is not reached on the short low-probability routes (SCA3-1 and "
             "SCA8-1 are 9.8e-4 and 2.6e-4 from EMa 128 at 64 bins and still moving at 128 bins). For those routes "
             "Gauss-Hermite 64 to 128 is the converged rule, and no bin-averaged grid is recommended on the evidence here.\\n")

    # Recommended bin count for the bin-averaged grid, from the data (reference = largest EMa at b = 5)
    sub5 = gh[gh["b"] == 5]
    rt = {}
    for (inst_n, rid), s_ in sub5.groupby(["instance", "route_id"]):
        rt[(inst_n, int(rid))] = {"L": int(s_.iloc[0]["L"]),
                                  **{c: float(s_[s_["cfg"] == c]["p_disc"].iloc[0]) for c in
                                     ("GH128", "EMa8", "EMa16", "EMa32", "EMa64", "EMa128")}}
    short = [v for v in rt.values() if v["L"] <= 8]
    long_ = [v for v in rt.values() if v["L"] >= 16]
    def _dev(group, c):
        return max(abs(v[c] - v["EMa128"]) for v in group) if group else float("nan")
    L.append("Recommendation of the bin count (b = 5, reference EMa 128, tolerance " + f"{GH_TOL:g}" + "):\n")
    head3 = ["bin count 2^f", "max |EMa - EMa128| all 9 routes", "short routes (L <= 8)", "long routes (L >= 16)"]
    rows3 = []
    for f in (3, 4, 5, 6, 7):
        c = f"EMa{2 ** f}"
        rows3.append([f"2^{f} = {2 ** f}", f"{_dev(list(rt.values()), c):.2e}", f"{_dev(short, c):.2e}",
                      f"{_dev(long_, c):.2e}"])
    L += _md(head3, rows3)
    L.append("")
    f_all = next((f for f in (3, 4, 5, 6, 7) if _dev(list(rt.values()), f"EMa{2 ** f}") <= GH_TOL), None)
    gh_gap = max(abs(v["GH128"] - v["EMa128"]) for v in rt.values())
    L.append(f"Largest gap between the two references at b = 5 (GH 128 and EMa 128) over the nine routes: {gh_gap:.2e}. "
             + (f"Within {GH_TOL:g} of EMa 128 on all nine routes, the smallest bin count is 2^{f_all} = {2 ** f_all}, "
                "but 2^7 is the reference itself, so this is not a convergence check: on the short low-probability routes "
                "the 2^6 deviation is 2.6e-4 and EMa 128 is still moving, so no bin count up to 2^6 is shown to meet the "
                "tolerance there."
                if f_all == 7 else
                (f"Smallest bin count within {GH_TOL:g} of EMa 128 on all nine routes: 2^{f_all} = {2 ** f_all}."
                 if f_all is not None else f"No bin count up to 2^6 is within {GH_TOL:g} of EMa 128 on all nine routes."))
             + "\n")
    return L


CRN_TABLE_GRIDS = (("EMa8", "em_avg", 8), ("EMa16", "em_avg", 16), ("EMa32", "em_avg", 32),
                   ("EMa64", "em_avg", 64), ("GH64", "gh", 64), ("GH128", "gh", 128))


def _section_4c(crn: pd.DataFrame) -> list[str]:
    """Factor-grid error on the CRN quantities (conditional Monte Carlo on the discrete model)."""
    L: list[str] = []
    A = L.append
    A("### 4c. Factor-grid error on the CRN quantities P(D = +1), P(D = -1)\n")
    A(f"Method. For three routes (short, medium, long) and 6 intra and 6 alternative-position moves each (seeded, among "
      f"moves with P(D != 0) > {CRN_MIN_NONZERO:g} in the continuous MC), P(D = +1) and P(D = -1) are computed in the "
      "discrete copula model at b = 5 on the grids EMa 8..64 and GH 64 and 128. The exact identity "
      "P(D = +1) = sum_ts w_t w_s P(D = +1 | t, s) holds on the discrete model, but no exact conditional computation "
      "of the joint of two routes is available in reasonable time, so each factor pair (t, s) is a conditional Monte "
      f"Carlo with K = {CRN_K} common draws (customers independent given the pair, pmfs of the discrete model). Both "
      "routes of a pair use the same draws. The standard error of each grid sum is written in Table 4c-2. The per-pair "
      f"budget is K = {CRN_K}, not the 2e5 of the scenario MC, because the grids contain up to 16384 pairs per route. "
      "Per draw, P(D = +1) - P(D = -1) = P(F_b > Q) - P(F_a > Q) exactly, and the tail p_base of each base route is "
      "checked against the exact DP (chance_copula) in Table 4c-1.\n")
    if crn.empty:
        A("(no CRN results)\n")
        return L
    # Table 4c-1: per route and grid, the largest deviations from the two references
    A("Table 4c-1: per route and grid, the largest |deviation| over the 12 moves of P(D = +1) and P(D = -1), against "
      "the two converging references GH 128 and EMa 64 (the largest bin-averaged grid computed for every route); "
      "the base tail is the exact DP minus the conditional MC, in units of its standard error.\n")
    head = ["route", "role", "L", "grid", "max |dP(D=+1)| vs GH128", "max |dP(D=-1)| vs GH128",
            "max |dP(D=+1)| vs EMa64", "max |dP(D=-1)| vs EMa64", "median SE of P(D=+1)",
            "base p exact - MC (SE units)"]
    rows = []
    for (role, inst_n, rid), s in crn.groupby(["role", "instance", "route_id"], sort=False):
        ref = {}
        for g, p in (("gh", 128), ("em_avg", 64)):
            ref[(g, p)] = s[(s["grid"] == g) & (s["points"] == p)].sort_values("move_no")
        L_ = int(s["L"].iloc[0])
        for lab, g, p in CRN_TABLE_GRIDS:
            m = s[(s["grid"] == g) & (s["points"] == p)].sort_values("move_no")
            if m.empty:
                continue
            g128, g64 = ref[("gh", 128)], ref[("em_avg", 64)]
            d1_gh = (m["P_D1"].to_numpy() - g128["P_D1"].to_numpy()) if len(g128) == len(m) else None
            dm_gh = (m["P_Dm1"].to_numpy() - g128["P_Dm1"].to_numpy()) if len(g128) == len(m) else None
            d1_em = (m["P_D1"].to_numpy() - g64["P_D1"].to_numpy()) if len(g64) == len(m) else None
            dm_em = (m["P_Dm1"].to_numpy() - g64["P_Dm1"].to_numpy()) if len(g64) == len(m) else None
            fmt_max = lambda a: f"{np.max(np.abs(a)):.2e}" if a is not None and len(a) else "-"
            pe = float(m["p_base_exact"].iloc[0]); pm = float(m["p_base_mc"].iloc[0]); se = float(m["p_base_se"].iloc[0])
            rows.append([f"{inst_n} r{int(rid)}", role, L_, lab, fmt_max(d1_gh), fmt_max(dm_gh), fmt_max(d1_em),
                         fmt_max(dm_em), f"{np.median(m['se_D1']):.1e}",
                         f"{(pe - pm) / se:+.2f}" if se > 0 else "-"])
    L += _md(head, rows)
    L.append("")
    # Table 4c-2: per move values
    A("Table 4c-2: per move, P(D = +1) and P(D = -1) on each grid (6 decimals; standard error of the largest grid in "
      "brackets; the continuous MC of M3a for the same move in the last column, P(D=+1) / P(D=-1)).\n")
    for role, inst_n, rid in crn[["role", "instance", "route_id"]].drop_duplicates().itertuples(index=False):
        s = crn[(crn["role"] == role) & (crn["instance"] == inst_n) & (crn["route_id"] == rid)]
        A(f"Route {role}: {inst_n} r{int(rid)} (L = {int(s['L'].iloc[0])}).\n")
        head = ["move", "class", "kind"] + [f"+1 {lab}" for lab, _, _ in CRN_TABLE_GRIDS] + [f"-1 {lab}" for lab, _, _ in CRN_TABLE_GRIDS] + ["continuous MC +1 / -1"]
        rows = []
        mv_nums = sorted(s["move_no"].unique())
        for k in mv_nums:
            cells = {}
            for lab, g, p in CRN_TABLE_GRIDS:
                m = s[(s["grid"] == g) & (s["points"] == p) & (s["move_no"] == k)]
                cells[lab] = (float(m["P_D1"].iloc[0]), float(m["P_Dm1"].iloc[0]),
                              float(m["se_D1"].iloc[0]) if len(m) else float("nan")) if len(m) else (float("nan"),) * 3
            base = s[(s["move_no"] == k)].iloc[0]
            pl = [f"{cells[lab][0]:.6f}" for lab, _, _ in CRN_TABLE_GRIDS]
            ml = [f"{cells[lab][1]:.6f}" for lab, _, _ in CRN_TABLE_GRIDS]
            rows.append([int(k), int(base["cls"]), str(base["kind"])] + pl + ml +
                        [f"{float(base['P_D1_cont']):.6f} / {float(base['P_Dm1_cont']):.6f}"])
        L += _md(head, rows)
        L.append("")
    return L


def summary_v3() -> str:
    mv = pd.read_csv(OUT_DIR / "m3a_v2_moves.csv.gz")
    inst = pd.read_csv(OUT_DIR / "m3a_v2_instances.csv")
    er = pd.read_csv(OUT_DIR / "m3a_v3_endload_routes.csv")
    im = pd.read_csv(OUT_DIR / "m3a_v3_intra_check.csv.gz")
    pa = pd.read_csv(OUT_DIR / "m3a_v3_paired.csv")
    gh_raw = pd.read_csv(GH_CSV)
    N = int(inst["N"].iloc[0])
    mv = mv.assign(route_key=mv["instance"] + "|" + mv["route_id"].astype(str))
    mv["pband"] = [band_label(p, PPRIME_BANDS) for p in mv["p_new"]]
    mv["gband"] = [band_label(gate_distance(p), GATE_BANDS) for p in mv["p_new"]]
    L: list[str] = []
    L.append("# QUASAR M3a v3 (revision 3): interpretation layer on the BATON copula model\n")
    L.append("Supersedes the interpretation of m3a_v2_summary.md (its Tables B, D and E are not carried "
             "over; Table A is replaced by Section 1, which bands by p' and by distance to the gate level). "
             "Revision 3 changes: classical certificate floor in Section 3, quadrature study with equal-probability "
             "factor bins in Section 4b, bias by route length and a qualified b rule in Section 4a, "
             "instance-level intervals in Section 1a.\n")
    L.append(f"Scenarios: N = {N} per instance, seed {SCEN_SEED}, rho = {DEFAULT_RHO}, the BATON copula; "
             "moves and routes are the cached v2 results (`m3a_v2_moves.csv.gz`, `m3a_v2_routes.csv`). "
             "Scenarios were recomputed for the end-load and level checks; the recomputed intra-move "
             "P(D = 1) and P(D = -1) agree with the cached values to a maximum absolute difference of "
             f"{im['cache_diff'].max():.1e} (CSV rounding), which confirms that the draws are identical.\n")
    L.append("Definitions: p' = P(F' > Q) (new route), p = P(F > Q) (base), D = 1[F' > Q] - 1[F > Q], "
             "rho_q = sqrt(Var D) / sqrt(p'(1 - p')). Zeros in the sample are upper bounds: with no "
             f"scenario showing D != 0 out of N = {N}, P(D != 0) <= 3/N = {rule_of_three_bound(N):.2g} "
             "(95 %), which gives the plug-in upper bound on rho_q reported next to the medians. "
             "Route-bootstrap intervals resample routes (clusters, with replacement, 1000 replicates); "
             "instance-level intervals resample instances in the same way (two-stage clustering is not "
             "used; within a sampled instance all its routes are kept). The bootstrap median is the lower "
             "weighted median, the table quantiles are linear.\n")

    # ---- Section 1
    L.append("## 1. Decision bands of p' and of the distance to the gate level\n")
    L.append(f"Gate level: per-route failure target {GATE_LEVEL:.2f} (1 - ALPHA of dethloff_runner). Decision band "
             f"p' in [0.02, 0.2]. Distances |p' - {GATE_LEVEL:.2f}| in the second table.\n")
    L += _band_tables(mv, N, "pband", PPRIME_BANDS, "1a. Bands of p'", inst_ci=True)
    L += _band_tables(mv, N, "gband", GATE_BANDS, f"1b. Bands of |p' - {GATE_LEVEL:.2f}|")

    # ---- Section 2
    L.append("## 2. End-load lemma\n")
    L.append("Lemma. For any reordering r' of the same customers, L_0 = sum of deliveries and L_L = sum of "
             "pickups are order-invariant. Let E = {sum d <= Q and sum p <= Q}. Outside E, one end load exceeds Q "
             "for every reordering, so both overflow indicators equal 1 and D = 0. On E, D != 0 forces "
             "max(F_r, F_r') > Q, so P(D != 0) <= P(E and (F_r > Q or F_r' > Q)). The lemma covers every reordering "
             "of the same customers, hence intra-route moves and alternative-position moves (class 3, where the two "
             "routes contain the same customers) alike; the table below checks intra-route moves only. "
             "LEMMA BOUND (valid): B_union = P(E and (F_r > Q or F_r' > Q)), with r' ranging over the sampled "
             "intra neighbours of the route; it bounds P(D != 0) of every such move. DIAGNOSTIC ONLY, NOT A VALID "
             "BOUND: B_spec = P(E and F_r > Q). Explained share = P(max(sum d, sum p) > Q) / P(F_r > Q).\n")
    L += _lemma_rows(er, im)
    L.append("")
    L.append(f"Consistency check (not a test of the lemma, which is an identity): every intra move "
             f"({len(im)} sampled moves over {int(er['n_intra'].gt(0).sum())} routes with intra moves; all "
             f"{len(er)} routes evaluated) was recomputed scenario by scenario. Scenarios with D != 0 and an end "
             f"load above Q: {int(im['viol'].sum())}. Routes with B_union below their largest observed P(D != 0): "
             f"{int((er['max_pd_ne0'] > er['B_union'] + 1e-12).sum())}. The zero count checks that the move "
             "reconstruction and the scenario draws reproduce the cached move code path.\n")
    L.append("B_spec is a diagnostic and is not a valid bound on P(D != 0): it omits scenarios where only the "
             "moved route overflows. The column 'routes with B_spec < route max' counts where it falls short "
             "of the observed largest P(D != 0). B_union is the lemma bound.\n")

    # ---- Section 3
    L.append("## 3. Query costs per class, both paradigms, with certified classical counts\n")
    L.append("Moves with p' in (0.02, 0.5). Stand-alone: p' estimated at eps (quantum: canonical AE, "
             "5 x (2M + 1) applications; classical: max(z^2 p'(1-p') / eps^2, 3 / eps), z = 1.96). CRN: class 1 "
             "one amplitude P(D = 1) with variance max(Var D, 3/N); classes 2-3 quantum two-estimate P(D = 1) and "
             "P(D = -1) at eps/2 each, classical direct signed mean max(z^2 max(Var D, 3/N) / eps^2, 3/eps). "
             "The 3/eps term is the classical certificate floor: without it, a move with no event in the sample "
             "is charged a CLT count (58 samples at eps = 1e-3) that certifies nothing. eps is absolute (1e-3, 1e-4) "
             "or relative, eps_i = 0.1 p'_i per move. Sums are over moves; medians are over moves.\n")
    band = mv[(mv["p_new"] > COST_BAND[0]) & (mv["p_new"] < COST_BAND[1])].copy()
    pn = band["p_new"].to_numpy(float)
    L.append("### 3a. Within-paradigm medians (p' in (0.02, 0.5))\n")
    band["_ub"] = np.sqrt(np.maximum(band["var_D"], rule_of_three_bound(N))) / np.sqrt(pn * (1 - pn))
    rows = []
    for cls in (1, 2, 3):
        for fam in FAMILIES:
            s = band[(band["cls"] == cls) & (band["family"] == fam)]
            if s.empty:
                continue
            rows.append([CLASS_NAMES[cls], fam, len(s), _fmt(s["rho_q"].median()), _fmt(s["rho_c"].median()),
                         _fmt(s["_ub"].median()), f"{s['zero'].mean():.3f}"])
    L += _md(["class", "family", "moves", "median rho_q (in-sample)", "median rho_c (in-sample)",
              "median rho_q upper bound", "Var=0 share"], rows)
    L.append("")
    ins_med = {f: band[(band["cls"] == 1) & (band["family"] == f)]["rho_q"].median() for f in FAMILIES}
    ins_txt = ", ".join(f"{f} {ins_med[f]:.2f}" for f in FAMILIES if pd.notna(ins_med[f]))
    L.append("Reading. rho_q is the ratio of the CRN amplitude standard deviation to the stand-alone one, so the "
             "quantum query count of the CRN difference is about rho_q times the stand-alone count. For insertions "
             f"(class 1, median rho_q by family in the band: {ins_txt}) the value is near 0.9 only on CON/SCA; on CMT and "
             "city a substantial part of the variance is removed by the common draws. Intra-route and alternative-position "
             "moves reorder the same customers, so Var(D) is small: on CON/SCA the medians are 0.13-0.16, and on CMT/city "
             "they are zero in the sample for more than half of the moves. The zero is an in-sample value and is bounded "
             "by the upper-bound column.\n")

    L.append("### 3b. Cost sums (queries) by class and family, certified classical counts\n")
    L.append("Nq/Nq_sa and Nc/Nc_sa: within-paradigm CRN gain (CRN cost over stand-alone cost). Nq/Nc: quantum "
             "CRN over classical CRN (classical direct for classes 2-3) at the same target, on the sums; the last "
             "column is the median over moves of the per-move ratio Nq/Nc. The quantum counts count applications "
             "of A; the classical counts count scenario evaluations. These are different unit costs (see the "
             "reading below).\n")
    eps_variants = (("absolute eps = 1e-3", np.full(band.shape[0], 1e-3)),
                    ("absolute eps = 1e-4", np.full(band.shape[0], 1e-4)),
                    ("relative eps = 0.1 p'", 0.1 * pn))
    head_cost = ["class", "family", "moves", "sum Nq_sa", "sum Nq", "Nq/Nq_sa", "sum Nc_sa", "sum Nc", "Nc/Nc_sa",
                 "Nq/Nc (sums)", "median per-move Nq/Nc"]
    cost_res: dict[str, dict] = {}
    for label, eps_arr in eps_variants:
        rows, res, _ = _cost_rows(band, eps_arr, N)
        cost_res[label] = res
        L.append(f"#### {label}\n")
        L += _md(head_cost, rows)
        L.append("")
    e3 = cost_res["absolute eps = 1e-3"]
    ea = cost_res["absolute eps = 1e-4"]
    er3 = cost_res["relative eps = 0.1 p'"]

    def _r(res, cls, fam, key="Nq_Nc"):
        return f"{res[(cls, fam)][key]:.2g}" if (cls, fam) in res else "-"

    L.append("#### Reading of Section 3b\n")
    L.append(
        "Unit costs and what the ratios mean. Nq counts applications of the amplitude-estimation unitary A. In "
        "the circuit one application of a stand-alone peak oracle is one coherent oracle call with T(L, c, m) "
        "Toffolis (DESIGN_M2.md, Section 'Toffoli count'). A difference oracle evaluates two route predicates on one "
        "scenario register, so one of its applications costs more Toffolis than a stand-alone one; the tables do not "
        "convert counts to Toffolis. Nc counts classical scenario evaluations, each a full route evaluation. A "
        "ratio Nq/Nc below 1 therefore says that the quantum count is smaller; it is not a runtime ratio.\n")
    L.append(
        "Conservativeness of the quantum count. The count uses canonical amplitude estimation with the bound "
        "2 pi sqrt(v)/M + pi^2/M^2, which holds with probability at least 8/pi^2, and R = 5 repetitions with the "
        "median (failure probability about 0.05, so about 95 % confidence). The bound is a worst case over the "
        "position of the phase on the grid. In a check on the exact QPE distribution at the same M, the 81 % quantile "
        "of the actual error was smaller than the bound by factors of 1.6, 2.0 and 4.6 in the three non-aligned cases "
        "checked (a = 0.05 with eps = 1e-2; a = 0.2 with eps = 1e-3; a = 0.2 with eps = 1e-2), and by much more when "
        "the phase falls on the grid. The factor depends on alignment and is not a general constant: the quantum counts "
        "here are conservative by 1.6-4.6x in the three cases checked, and no larger factor is claimed.\n")
    def _range(res, cls_set):
        vals = [res[(c, f)]["Nq_Nc"] for c in cls_set for f in FAMILIES if (c, f) in res]
        return f"{min(vals):.2g} to {max(vals):.2g}" if vals else "-"
    L.append(
        f"Absolute eps = 1e-3 (classical certified). Quantum-to-classical ratios of the sums: insertions "
        f"{_range(e3, (1,))} across families, intra moves {_range(e3, (2,))}, alternative-position moves "
        f"{_range(e3, (3,))}. The city cells, which the CLT count made 7 to 19 times worse in the previous "
        f"table, are now {_r(e3, 3, 'city')} (alternative) and {_r(e3, 2, 'city')} (intra). The city ratio is a ratio of "
        "two constants of the floor regime, not a property of the problem: a move with no event in the sample costs "
        "2 R (2M + 1) = 5130 applications at eps = 1e-3 (R = 5, M = 256, two estimates), against the classical floor "
        "3 / eps = 3000, and 5130 / 3000 = 1.71. At eps = 1e-4 the same count is 20490 against 30000 (0.68). Floor "
        "moves are a large share of city moves (Var(D) = 0 share "
        f"{e3[(2, 'city')]['zero_share']:.2f} of city intra and {e3[(3, 'city')]['zero_share']:.2f} of city alternative "
        "moves), so the city cells mostly measure the floor. The honest statement is comparable cost at 1e-3 (the floor "
        "constants set the ratio) and a lower quantum count at 1e-4 in every class and family of this table, not a quantum "
        "penalty. At eps = 1e-4 the ratios are "
        f"{_range(ea, (1,))} (insertions), {_range(ea, (2,))} (intra) and {_range(ea, (3,))} (alternative).\n")
    L.append(
        f"Within-paradigm CRN gain. Insertions gain little from CRN in the quantum count (Nq/Nq_sa "
        f"{cost_res['absolute eps = 1e-3'][(1, 'CON')]['Nq_gain']:.2f} on CON) and in the classical count "
        f"(Nc/Nc_sa {cost_res['absolute eps = 1e-3'][(1, 'CON')]['Nc_gain']:.2f}). Signed moves gain a lot in both: "
        f"quantum Nq/Nq_sa {_r(e3, 2, 'CON', 'Nq_gain')} and classical Nc/Nc_sa {_r(e3, 2, 'CON', 'Nc_gain')} on CON "
        "intra moves, because the stand-alone variance is much larger than Var(D).\n")
    sig = [er3[(c, f)]["Nq_Nc"] for c in (2, 3) for f in FAMILIES if (c, f) in er3]
    L.append(
        f"Relative precision (eps = 0.1 p'). Here the classical direct estimator is cheaper on signed moves: "
        f"Nq/Nc is {min(sig):.3g} to {max(sig):.3g} over the signed (intra and alternative-position) cells, "
        "so the quantum count is that many times the classical count in these sums (the unit costs differ, see "
        "above). The mechanism: a quantum count grows like sqrt(variance)/eps and a classical count like "
        "variance/eps^2, so the classical estimator wins when eps is large compared with the standard deviation "
        "of the indicators. Here eps_i = 0.1 p'_i is relative to the stand-alone p'_i of each move, not to E[D]. "
        "At eps = 0.1 p' this holds for the signed moves here, while at absolute eps = 1e-4 "
        "(where eps is small against those deviations) the quantum count is the lower one. Insertions are the "
        "exception in the relative block: "
        f"Nq/Nc is {_range(er3, (1,))} there. Recommendation: the absolute blocks (1e-3 and 1e-4) are the main comparison; "
        "the relative block answers a different question (precision relative to p'), and its signed-move ratios favour the "
        "classical direct estimator. Quote it only with that label.\n")

    L.append("### 3c. Convention for one-sided moves (classes 2-3, absolute eps = 1e-3)\n")
    L.append("Reading. The single-estimate convention lowers the quantum count of one-sided moves by the share shown "
             "in the table (the classical direct count is the same). It does not change any conclusion of 3b: the "
             "floor-dominated city cells stay near 1.4 to 1.7 at 1e-3 and the CON/SCA/CMT cells stay below 1 at "
             "1e-3. The main convention is the one used in 3b.\n")
    L.append("Main tables charge a one-sided move (only one of P(D = 1), P(D = -1) is nonzero) two estimates at "
             "eps/2, the zero side at the rule-of-three floor. The M3b convention estimates only the nonzero side, "
             "at eps. Both are shown; the classical direct count is the same in both.\n")
    L += _md(["class", "family", "moves", "one-sided moves", "sum Nq (main convention)",
              "sum Nq (single nonzero-side estimate)", "Nq/Nc (main)", "Nq/Nc (single)"],
             _one_sided_rows(band, 1e-3, N))
    L.append("")

    # ---- Section 4
    L.append("## 4. Discretisation without quadrature error, and the quadrature study\n")
    L.append("### 4a. Paired level-only check (same continuous copula scenarios, nearest-level binning)\n")
    L.append("Bias = P(F_level > K) - P(F > Q) on the same N scenarios (no quadrature anywhere in this "
             "check); paired SE from the scenario-wise difference of the two indicators. Routes: up to "
             f"{PAIRED_PER_CELL} per family x base-p stratum (seeded). Relative bias only where P(F > Q) >= 0.01.\n")
    cnt = pa[pa["b"] == pa["b"].min()].drop_duplicates(["instance", "route_id"])
    ct = pd.crosstab(cnt["family"], cnt["stratum"])
    L.append("Routes per family x stratum (base p): " + "; ".join(
        f"{fam}: " + ", ".join(f"{c} {int(ct.loc[fam, c])}" for c in ct.columns if fam in ct.index and ct.loc[fam, c] > 0)
        for fam in FAMILIES if fam in ct.index) + f". Total routes {len(cnt)}.\n")
    rows = []
    summ_rule: dict[tuple[str, int], dict] = {}
    for rule in LEVEL_RULES:
        for b in V3_PAIRED_B:
            s = pa[(pa["rule"] == rule) & (pa["b"] == b)]
            rel = (s["bias"].abs() / s["p_cont"]).where(s["p_cont"] >= 0.01).dropna()
            summ_rule[(rule, b)] = {"med_rel": float(rel.median()) if len(rel) else float("nan"),
                                    "max_rel": float(rel.max()) if len(rel) else float("nan"),
                                    "n_gt5": int((rel > BIAS_REL_MAX).sum()) if len(rel) else 0,
                                    "gt5_L": sorted(set(int(x) for x in s.loc[rel.index[rel > BIAS_REL_MAX], "L"]))
                                    if len(rel) else []}
            rows.append([rule, b, len(s), _sci(s["bias"].abs().median()), _sci(s["bias"].abs().max()),
                         _sci(s["se"].median()), _fmt(rel.median(), 3), _fmt(rel.max(), 3)])
    L += _md(["rule", "b", "routes", "median abs bias", "max abs bias", "median paired SE",
              "median abs rel bias (p >= 0.01)", "max abs rel bias (p >= 0.01)"], rows)
    L.append("")
    L.append("By family (nearest rule):\n")
    rows = []
    for fam in FAMILIES:
        for b in V3_PAIRED_B:
            s = pa[(pa["rule"] == "nearest") & (pa["b"] == b) & (pa["family"] == fam)]
            if s.empty:
                continue
            rel = (s["bias"].abs() / s["p_cont"]).where(s["p_cont"] >= 0.01).dropna()
            rows.append([fam, b, len(s), _sci(s["bias"].abs().median()), _sci(s["bias"].abs().max()),
                         _fmt(rel.median(), 3), _fmt(rel.max(), 3)])
    L += _md(["family", "b", "routes", "median abs bias", "max abs bias", "median abs rel bias (p >= 0.01)",
              "max abs rel bias (p >= 0.01)"], rows)
    L.append("")
    pa["lbin"] = [length_label(x) for x in pa["L"]]
    L.append("By route length (nearest rule; bands of the route length L, the number of customers on the route):\n")
    rows = []
    for b in V3_PAIRED_B:
        for _, _, lab in LENGTH_BANDS:
            s = pa[(pa["rule"] == "nearest") & (pa["b"] == b) & (pa["lbin"] == lab)]
            if s.empty:
                continue
            rel = (s["bias"].abs() / s["p_cont"]).where(s["p_cont"] >= 0.01).dropna()
            rows.append([b, lab, len(s), _sci(s["bias"].abs().median()), _sci(s["bias"].abs().max()),
                         _fmt(rel.median(), 3) if len(rel) else "-", _fmt(rel.max(), 3) if len(rel) else "-"])
    L += _md(["b", "L band", "routes", "median abs bias", "max abs bias", "median abs rel bias (p >= 0.01)",
              "max abs rel bias (p >= 0.01)"], rows)
    L.append("")
    b_rec = None
    for b in V3_PAIRED_B:
        r = summ_rule[("nearest", b)]
        if np.isfinite(r["med_rel"]) and r["med_rel"] <= BIAS_REL_MED and r["max_rel"] <= BIAS_REL_MAX:
            b_rec = b
            break
    r5 = summ_rule[("nearest", 5)]
    r6 = summ_rule[("nearest", 6)]
    L.append(f"Level rule (a choice, stated as such): median |rel bias| <= {BIAS_REL_MED:.0%} and max |rel bias| "
             f"<= {BIAS_REL_MAX:.0%} (p >= 0.01). Under this rule the recommended b is "
             f"{b_rec if b_rec is not None else 'none of 4, 5, 6'}. The rule is met at b = 6 "
             f"(median {r6['med_rel']:.2%}, max {r6['max_rel']:.2%}). At b = 5 the median meets the rule "
             f"({r5['med_rel']:.2%}) but the maximum does not ({r5['max_rel']:.2%}; {r5['n_gt5']} route(s) above "
             f"5 %, of length L = {', '.join(str(x) for x in r5['gt5_L'])}). The worst relative cases are short routes "
             "with few large demands; the relative bias falls with L, while the median absolute bias is larger on the "
             "16-25 band than on the short routes (see the length table). "
             "At b = 6 the largest absolute bias is "
             f"{pa[(pa['rule'] == 'nearest') & (pa['b'] == 6)]['bias'].abs().max():.2e}, so the level error is "
             "small in relative terms but not below 1e-3 absolute on every route. Cost of b = 6: 6 qubits per "
             "customer and per demand type (12 per customer for delivery and pickup), against 5 (10) at b = 5.\n")

    # ---- Section 4b and 4c
    L.extend(_section_4b(gh_raw))
    crn_raw = pd.read_csv(CRN_CSV) if CRN_CSV.exists() else pd.DataFrame()
    L.extend(_section_4c(crn_raw))

    # ---- Section 5
    L.append("## 5. Alternative-position moves by position distance\n")
    alt = mv[mv["cls"] == 3].copy()
    alt["dlab"] = [alt_distance_label(abs(int(b) - int(a))) for a, b in zip(alt["pos"], alt["pos2"])]
    L.append("Distance |pos2 - pos| (1, 2-3, >3). Median rho_q with route-bootstrap CI; Var=0 share in-sample. "
             "Decision band: p' in [0.02, 0.2].\n")
    L.append("| distance | family | n (all p') | median rho_q [CI] | p10 | p90 | Var=0 share | n (decision band) | median rho_q [CI] (decision band) | Var=0 share (decision band) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for dl in ALT_DIST_LABELS:
        for fam in FAMILIES:
            s = alt[(alt["dlab"] == dl) & (alt["family"] == fam)]
            if s.empty:
                continue
            ok = pd.to_numeric(s["rho_q"], errors="coerce").notna()
            sa = s[ok]
            ra = sa["rho_q"].to_numpy(float)
            pt, lo, hi = route_bootstrap_quantile(ra, sa["route_key"].to_numpy(), 0.5, BOOT_REPS, seed=SCEN_SEED)
            q = np.quantile(ra, [0.10, 0.90])
            db = sa[(sa["p_new"] >= DECISION_BAND[0]) & (sa["p_new"] < DECISION_BAND[1])]
            if len(db):
                rb = db["rho_q"].to_numpy(float)
                pb, lob, hib = route_bootstrap_quantile(rb, db["route_key"].to_numpy(), 0.5, BOOT_REPS, seed=SCEN_SEED)
                dtxt = f"{len(db)} | {_fmt(pb)} [{_fmt(lob)}, {_fmt(hib)}] | {db['zero'].mean():.3f}"
            else:
                dtxt = "0 | - | -"
            L.append(f"| {dl} | {fam} | {len(sa)} | {_fmt(pt)} [{_fmt(lo)}, {_fmt(hi)}] | {_fmt(q[0])} | "
                     f"{_fmt(q[1])} | {sa['zero'].mean():.3f} | {dtxt} |")
    L.append("")
    return "\n".join(L) + "\n"



def write_summary_v3() -> None:
    text = summary_v3()
    (OUT_DIR / "m3a_v3_summary.md").write_text(text, encoding="utf-8")
    log(f"wrote m3a_v3_summary.md ({len(text.splitlines())} lines)")



# ============================================================
# Entry point
# ============================================================


def main(argv: list[str] | None = None) -> None:
    global _LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("v2", "v2disc", "v2summary", "v3", "v3gh", "v3crn", "v3summary"),
                    required=True)
    ap.add_argument("--rho", type=float, default=DEFAULT_RHO, help="copula strength (0.6 = BATON)")
    ap.add_argument("--N", type=int, default=N_SCEN, help="common scenarios per instance")
    ap.add_argument("--every", type=int, default=1, help="use every k-th instance (sorted by name)")
    ap.add_argument("--cap", type=int, default=CAP_INS, help="insertion candidates per route")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--instances", default=None, help="fnmatch pattern on plan names")
    ap.add_argument("--force", action="store_true", help="recompute cached instances")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = OUT_DIR / f"m3_{args.part}.log"
    log(f"command: {' '.join(sys.argv)}")
    if args.part == "v2":
        run_part_v2(args)
    elif args.part == "v2disc":
        run_part_disc(args)
    elif args.part == "v2summary":
        write_summary_v2()
    elif args.part == "v3":
        run_part_v3(args)
    elif args.part == "v3gh":
        run_part_gh(args)
    elif args.part == "v3crn":
        run_part_crn(args)
    elif args.part == "v3summary":
        write_summary_v3()


if __name__ == "__main__":
    main()
