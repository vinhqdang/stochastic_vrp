"""QUASAR M3 driver.

Part a (v2, design Revision 2): move statistics on the BATON copula demand
model, common scenarios per instance, and the copula discretisation check.
Part b: amplitude estimation on the real oracles (unchanged from v1 except the
discretisation rule of instance_demand and the MLAE docstring).

Usage (from svrpspd_wdro/):
    python3 scripts/run_quasar_m3.py --part v2 [--rho 0.6] [--N 200000]
        [--every 1] [--instances PATTERN] [--workers 3] [--cap 200] [--force]
    python3 scripts/run_quasar_m3.py --part v2disc [--rho 0.6]
    python3 scripts/run_quasar_m3.py --part v2summary
    python3 scripts/run_quasar_m3.py --part b|all --b 4 --seed 0

v2 writes results/quasar/m3a_v2{tag}_moves.csv.gz and m3a_v2{tag}_routes.csv
(tag = "" for rho = 0.6, "_rho0" for rho = 0). Per-instance results are cached
under results/quasar/m3a_v2_parts/ so an interrupted run resumes.
Progress is printed and appended to results/quasar/m3_<part>.log.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import multiprocessing as mp
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import dethloff_runner as DR  # noqa: E402  (scenario generator of the BATON runs)
from quantum.classical import chance, insertion_window  # noqa: E402
from quantum.experiments import (  # noqa: E402
    DEFAULT_CV,
    DEFAULT_RHO,
    LEVEL_RULES,
    chance_copula,
    classical_queries,
    copula_demand,
    insertion_candidates,
    insertion_envelopes,
    insertion_peak,
    instance_demand,
    intra_candidates,
    load_profile,
    route_peak,
    mlae_estimate,
    move_stats,
    quantum_queries,
    signed_costs,
    window_stats,
    classical_samples_for_error,
)
from quantum.oracle import (  # noqa: E402
    build_difference_oracle,
    build_peak_oracle,
    estimate_probability,
    exact_probability,
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

# Constants of part b (amplitude estimation on the real oracles).
BAND = (0.01, 0.30)
ZERO_TOL = 1e-12
AE_EPS_TARGETS = (0.05, 0.02, 0.01, 0.005)
AE_SCHEDULES = (2, 3, 4, 5, 6)
AE_SEEDS = tuple(range(10))


def _grid_for(n_qubits: int) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Amplitude-estimation grid by oracle size. The full grid (AE_EPS_TARGETS x
    AE_SCHEDULES) is used up to 13 qubits. Larger statevector circuits take far
    longer per Grover round, so the MLAE schedules and IAE targets are reduced
    (the settings actually run are listed in m3b_summary.md)."""
    if n_qubits <= 13:
        return AE_EPS_TARGETS, AE_SCHEDULES
    if n_qubits <= 16:
        return AE_EPS_TARGETS, (2, 3, 4)
    if n_qubits <= 19:
        return (0.05,), (2,)
    return (0.05,), ()


AE_ALPHA = 0.05
MAX_QUBITS = 22

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


# ============================================================
# Part b
# ============================================================


def _pick_move(plan: dict, src_ri: int, prefix: list[int], dem, Ql: int, seed: int):
    """Insertion move from another route with delta in BAND if one exists (seeded
    random order), otherwise the first with delta > 0."""
    routes = plan["res"]["Det"]["plan"]
    cands = []
    for k, other in enumerate(routes):
        if k == src_ri:
            continue
        for j in other:
            if j in prefix:
                continue
            for pos in range(1, len(prefix) + 2):
                cands.append((int(j), pos))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cands))
    first_pos = None
    for idx in order:
        j, pos = cands[int(idx)]
        s = move_stats(prefix, j, pos, dem, Ql)
        if BAND[0] <= s["delta"] <= BAND[1]:
            return j, pos, s
        if first_pos is None and s["delta"] > ZERO_TOL:
            first_pos = (j, pos, s)
    if first_pos is None:
        raise ValueError("no insertion move with delta > 0 found")
    return first_pos


def _run_ae_on(oracle, meta: dict, exact: float, rows: list[dict],
               eps_list=AE_EPS_TARGETS, schedules=AE_SCHEDULES) -> None:
    a = exact
    v = a * (1.0 - a)
    sd = math.sqrt(max(v, 0.0))
    base = dict(meta, exact_p=a)
    for eps in eps_list:
        for seed in AE_SEEDS:
            t = time.time()
            res = estimate_probability(oracle, epsilon=eps, alpha=AE_ALPHA, method="iae", seed=seed)
            rows.append(_ae_row(base, "iae", f"eps={eps}", seed, res, a, sd, time.time() - t))
        log(f"    {meta['config']} iae eps={eps} done")
    for m in schedules:
        for seed in AE_SEEDS:
            t = time.time()
            res = mlae_estimate(oracle, m, seed=seed)
            rows.append(_ae_row(base, "mlae", f"schedule={m}", seed, res, a, sd, time.time() - t))
        log(f"    {meta['config']} mlae schedule={m} done")


def _ae_row(base: dict, method: str, param: str, seed: int, res: dict, a: float,
            sd: float, wall: float) -> dict:
    err = abs(res["estimate"] - a)
    return dict(
        base, method=method, param=param, seed=seed, estimate=res["estimate"],
        abs_err=err, A_apps=res["num_oracle_queries"],
        A_apps_library=res["num_oracle_queries_library"],
        rounds=" ".join(str(k) for k in res.get("rounds", [])),
        classical_N_same_err=classical_samples_for_error(a, err),
        err_norm=(err * res["num_oracle_queries"] / (math.pi * sd)) if sd > 0 else float("nan"),
        wall_s=wall,
    )


_ONLY: list[str] | None = None
_AGGREGATE = False


def _cached_b(config: str, compute) -> list[dict]:
    """Rows of one configuration, cached as JSON so an interrupted part b resumes.
    With --only SUBSTRING, configurations not containing it are skipped."""
    f = OUT_DIR / "m3b_parts" / (re.sub(r"[^A-Za-z0-9]+", "_", config).strip("_") + ".json")
    if f.exists():
        log(f"  cached: {config}")
        return json.loads(f.read_text())
    if _AGGREGATE or (_ONLY is not None and not any(t in config for t in _ONLY)):
        return []
    out: list[dict] = []
    t = time.time()
    compute(out)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    log(f"  finished {config} in {time.time() - t:.1f} s")
    return out


def run_part_b(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped: list[str] = []
    # Det routes screened for non-trivial window probabilities (0.05 < p < 0.95).
    picks = [("CON8-3", 5, 2), ("CON8-0", 1, 3), ("SCA8-0", 1, 3)]
    for name, ri, L in picks:
        plan = json.loads((PLANS_DIR / f"{name}.json").read_text())
        prefix = [int(c) for c in plan["res"]["Det"]["plan"][ri][:L]]
        log(f"part b: route {name} Det r{ri} prefix {prefix} (L={L})")
        for b in (1, 2):
            dem, unit, Ql = instance_demand(plan, b)
            cfg = f"{name}|r{ri}|L{L}|b{b}|peak"
            p_cls = chance(prefix, dem, Ql)
            oracle = build_peak_oracle(prefix, dem, Ql)
            if oracle.num_qubits > MAX_QUBITS:
                skipped.append(f"{cfg}: {oracle.num_qubits} qubits")
                log(f"  skip {cfg}: {oracle.num_qubits} qubits > {MAX_QUBITS}")
                continue
            exact = exact_probability(oracle)
            if abs(exact - p_cls) > 1e-9:
                raise RuntimeError(f"{cfg}: oracle {exact} != classical {p_cls}")
            log(f"  {cfg}: qubits={oracle.num_qubits} exact p={exact:.6f} (matches classical)")
            meta = dict(config=cfg, instance=name, route_idx=ri, L=L, b=b, kind="peak",
                        n_qubits=oracle.num_qubits, move_j="", move_pos="", move_delta="",
                        p_new="")
            e_list, m_list = _grid_for(oracle.num_qubits)
            rows.extend(_cached_b(meta["config"], lambda out: _run_ae_on(
                oracle, meta, exact, out, eps_list=e_list, schedules=m_list)))
            if b == 1 and L == 2:
                j, pos, s = _pick_move(plan, ri, prefix, dem, Ql, args.seed)
                dcfg = f"{name}|r{ri}|L{L}|b{b}|diff j{j} pos{pos}"
                doracle = build_difference_oracle(prefix, j, pos, dem, Ql)
                if doracle.num_qubits > MAX_QUBITS:
                    skipped.append(f"{dcfg}: {doracle.num_qubits} qubits")
                    log(f"  skip {dcfg}: {doracle.num_qubits} qubits")
                    continue
                dexact = exact_probability(doracle)
                iw = insertion_window(prefix, j, pos, dem, Ql)
                if abs(dexact - iw) > 1e-9 or abs(dexact - s["delta"]) > 1e-9:
                    raise RuntimeError(f"{dcfg}: oracle {dexact}, window {iw}, delta {s['delta']}")
                log(f"  {dcfg}: qubits={doracle.num_qubits} exact D={dexact:.6f} "
                    f"(p'={s['p_new']:.6f}, p={s['p']:.6f})")
                dmeta = dict(config=dcfg, instance=name, route_idx=ri, L=L, b=b, kind="diff",
                             n_qubits=doracle.num_qubits, move_j=j, move_pos=pos,
                             move_delta=s["delta"], p_new=s["p_new"])
                e_list, m_list = _grid_for(doracle.num_qubits)
                rows.extend(_cached_b(dmeta["config"], lambda out: _run_ae_on(
                    doracle, dmeta, dexact, out, eps_list=e_list, schedules=m_list)))

    # Outputs are rebuilt from every cached configuration, so parallel processes
    # that each computed some configurations give the same final files.
    all_rows: list[dict] = []
    for f in sorted((OUT_DIR / "m3b_parts").glob("*.json")):
        all_rows.extend(json.loads(f.read_text()))
    _write_csv_ae(OUT_DIR / "m3b_ae.csv", all_rows)
    summary = _summary_b(all_rows, skipped)
    (OUT_DIR / "m3b_summary.md").write_text(summary, encoding="utf-8")
    log(f"part b wrote m3b_ae.csv ({len(all_rows)} rows) and m3b_summary.md")


def _write_csv_ae(path: Path, rows: list[dict]) -> None:
    _write_csv(path, rows)


def _summary_b(rows: list[dict], skipped: list[str]) -> str:
    df = pd.DataFrame(rows)
    L: list[str] = []
    L.append("# QUASAR M3b: amplitude estimation on the real oracles\n")
    L.append(f"Configurations: {df['config'].nunique() if len(df) else 0}; "
             f"runs: {len(df)}; seeds per setting: {len(AE_SEEDS)}; IAE alpha = {AE_ALPHA}. "
             "A-applications follow quantum.oracle.estimate_probability "
             "(shots x sum(2k+1) over rounds, 1024 shots per round).\n")
    if skipped:
        L.append("Skipped (above the 22-qubit limit): " + "; ".join(skipped) + "\n")
    L.append("Grid by oracle size (the settings actually run are in the table): full grid "
             "(IAE eps 0.05-0.005, MLAE schedules 2-6) up to 13 qubits; MLAE schedules 2-4 at 16 "
             "qubits; IAE eps 0.05 and MLAE schedule 2 at 19 qubits; IAE eps 0.05 only at "
             "22 qubits (no MLAE: a single 22-qubit MLAE schedule-2 run did not finish; IAE at "
             "eps 0.02 at 19 qubits did not finish in 20 minutes). "
             "10 seeds throughout.\n")
    L.append("err_norm = mean abs error x A-applications / (pi sqrt(a(1-a))) "
             "(error bound scaling; roughly constant if the error follows the sqrt(a(1-a)) law). "
             "classical N = 1.96^2 a(1-a)/err^2 for the median error.\n")
    if not len(df):
        return "\n".join(L)
    L.append("| config | kind | qubits | exact a | method | setting | median abs err | "
             "RMS err | median A-apps | median err_norm | median classical N |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (cfg, method, param), s in df.groupby(["config", "method", "param"], sort=False):
        first = s.iloc[0]
        rms = math.sqrt(float((s["abs_err"] ** 2).mean()))
        L.append(
            f"| {cfg} | {first['kind']} | {first['n_qubits']} | {first['exact_p']:.6f} | "
            f"{method} | {param} | {s['abs_err'].median():.3g} | {rms:.3g} | "
            f"{s['A_apps'].median():.0f} | {s['err_norm'].median():.3g} | "
            f"{s['classical_N_same_err'].median():.3g} |"
        )
    L.append("")
    L.append("### Median A-applications per target (IAE), scaling with 1/eps\n")
    iae = df[df["method"] == "iae"]
    if len(iae):
        L.append("| config | eps_target | median A-apps | A-apps x eps |\n|---|---|---|---|")
        for (cfg, param), s in iae.groupby(["config", "param"], sort=False):
            eps = float(param.split("=")[1])
            med = s["A_apps"].median()
            L.append(f"| {cfg} | {eps} | {med:.0f} | {med * eps:.3g} |")
    L.append("")
    return "\n".join(L)



# ============================================================
# Summary (m3a_v2_summary.md)
# ============================================================


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


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
# Entry point
# ============================================================


def main(argv: list[str] | None = None) -> None:
    global _LOG_PATH, _ONLY, _AGGREGATE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("v2", "v2disc", "v2summary", "b"), required=True)
    ap.add_argument("--rho", type=float, default=DEFAULT_RHO, help="copula strength (0.6 = BATON)")
    ap.add_argument("--N", type=int, default=N_SCEN, help="common scenarios per instance")
    ap.add_argument("--every", type=int, default=1, help="use every k-th instance (sorted by name)")
    ap.add_argument("--cap", type=int, default=CAP_INS, help="insertion candidates per route")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--instances", default=None, help="fnmatch pattern on plan names")
    ap.add_argument("--force", action="store_true", help="recompute cached instances")
    ap.add_argument("--seed", type=int, default=0, help="part b seed")
    ap.add_argument("--only", default=None,
                    help="part b: comma-separated texts; compute only configs containing one")
    ap.add_argument("--aggregate", action="store_true",
                    help="part b: write outputs from cached configurations, computing nothing")
    args = ap.parse_args(argv)
    _ONLY = args.only.split(",") if args.only else None
    _AGGREGATE = args.aggregate
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = OUT_DIR / f"m3_{args.part}.log"
    log(f"command: {' '.join(sys.argv)}")
    if args.part == "v2":
        run_part_v2(args)
    elif args.part == "v2disc":
        run_part_disc(args)
    elif args.part == "v2summary":
        write_summary_v2()
    elif args.part == "b":
        run_part_b(args)


if __name__ == "__main__":
    main()
