"""QUASAR M4b task B: repair-step statistics of FIRST-FEASIBLE.

Reference: papers/quasar/THEORY_M4.md section 5. For every plan (gates Det, and
SAA when present) a sample of customers is removed from its route and
re-inserted at every position of the reduced plan. For each candidate the
peak-overflow risk p_j = P(F' > Q) is computed on the common scenarios of the
instance, together with its CRN contrast against the rank-1 candidate. Each
step records k* (certified-feasible index), the stop index s, and the predicted
query counts of FIRST-FEASIBLE (quantum) and of the sequential scan (classical),
with and without CRN.

Mode single (default): each step removes one sampled customer from the otherwise
unchanged plan and re-inserts it.

Mode rr (ruin-and-recreate): per (instance, gate) and round, Q_FRAC of the
customers are removed at once (dethloff_runner.ruin_recreate style), and the
removed customers are re-inserted one by one into the partially destroyed plan,
so the candidate lists come from a partial plan. Each insertion goes to the
cheapest candidate with p <= ALPHA (a new route when there is none). Risks are
evaluated lazily in candidate order (see rr_insert_step).

Usage (from svrpspd_wdro/):
    python3 scripts/run_quasar_m4b.py [--mode single] [--instances PATTERN] [--every K]
        [--workers W] [--N 200000] [--steps 40] [--force]
    python3 scripts/run_quasar_m4b.py --mode rr [--instances PATTERN] [--every K]
        [--workers W] [--N 50000] [--rounds 2] [--force]
    python3 scripts/run_quasar_m4b.py [--mode single|rr] --summary [--instances PATTERN] [--N ..]

Outputs (results/quasar/): single: m4b_steps.csv, m4b_summary.md, log m4b.log,
parts under m4b_parts/. rr: m4b_rr_steps.csv, m4b_rr_summary.md, log m4b_rr.log,
parts under m4b_rr_parts/<key>/. Per-instance parts are cached (keyed by N and
steps in single mode, by N, rounds and seeds in rr), so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # worker processes: one BLAS thread each

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import dethloff_runner as DR  # noqa: E402
import run_quasar_m3 as M3  # noqa: E402  (log, _rng, draw_scenarios, _list_instances)
from quantum.experiments import (  # noqa: E402
    GATE_LEVEL,
    insertion_envelopes,
    insertion_peak,
    load_profile,
)
from quantum.search import (  # noqa: E402
    classical_core_cost,
    classical_scan_cost,
    crn_classical_core_cost,
    crn_quantum_core_cost,
    ideal_schedule,
    kstar,
    quantum_core_cost,
    quantum_search_cost,
    stop_index,
)

ALPHA = GATE_LEVEL            # per-route failure target (0.10)
DELTA = 0.05                  # confidence of the search cost model
EPS = (1e-2, 3e-3, 1e-3)
EPS_TAG = {1e-2: "e2", 3e-3: "e3x", 1e-3: "e3"}
V_WC = 0.25                   # worst-case variance proxy ("wc")
SEED_M4B = 11                 # customer sampling seed
STEPS_MAX = 40                # sampled customers per (instance, gate)
GATES = ("Det", "SAA")
N_SCEN = 200_000
N_RR = 50_000                 # default common scenarios in mode rr
ROUNDS = 2                    # rr: ruin-and-recreate rounds per (instance, gate)
Q_FRAC = 0.2                  # rr: fraction of the customers removed per round
SEED_RR = 13                  # rr: ruin sampling seed
HEAD = 8                      # first p_j stored as a string
DATA_ROOT = ROOT / "data"
DATA_FAMILIES = ("Dethloff", "SalhiNagy", "City", "Region")
OUT_DIR = M3.OUT_DIR
PARTS_ROOT = OUT_DIR / "m4b_parts"
STEPS_CSV = OUT_DIR / "m4b_steps.csv"
SUMMARY_MD = OUT_DIR / "m4b_summary.md"
RR_PARTS_ROOT = OUT_DIR / "m4b_rr_parts"
RR_STEPS_CSV = OUT_DIR / "m4b_rr_steps.csv"
RR_SUMMARY_MD = OUT_DIR / "m4b_rr_summary.md"
FAMILY_ORDER = M3.FAMILIES + ("VNN", "GD")   # region families after the legacy ones

log = M3.log


def family_of(name: str) -> str:
    """Family of a plan name: VNN- and GD- region instances, else M3.family_of."""
    for prefix in ("VNN", "GD"):
        if name.startswith(prefix + "-"):
            return prefix
    return M3.family_of(name)


def distance_path(name: str) -> Path:
    for fam in DATA_FAMILIES:
        p = DATA_ROOT / fam / f"{name}.vrpspd"
        if p.exists():
            return p
    raise FileNotFoundError(f"no distance file for {name} under {DATA_ROOT}")


def distance_matrix(name: str) -> np.ndarray:
    return DR.parse_dethloff(distance_path(name))[0]


# ============================================================
# One repair step (mode single)
# ============================================================


def _candidate_table(plan_routes, c, D, dsc, psc, Q) -> dict | None:
    """Insertion candidates of customer c into the plan with c removed.

    plan_routes: the routes of the ORIGINAL plan (c included, depot = 0).
    Returns None when removing c leaves no route. Otherwise a dict with the
    candidates sorted by (delta, ri, pos) (1-based pos), the risks p and the CRN
    contrasts ap = P(D = 1), am = P(D = -1), varD = Var(D) against the rank-1
    candidate; the reduced routes; the base risk of each reduced route; and the
    rank and risk of the original position (orig_rank = 0, p_orig = NaN when c
    was alone on its route and that position is not a candidate).
    """
    c = int(c)
    orig = [[int(x) for x in r] for r in plan_routes]
    loc = [(i, r.index(c) + 1) for i, r in enumerate(orig) if c in r]
    if len(loc) != 1:
        raise ValueError(f"customer {c} must occur exactly once in the plan, found {len(loc)}")
    i0, p0 = loc[0]

    reduced: list[list[int]] = []
    new_idx: dict[int, int] = {}
    for i, r in enumerate(orig):
        rr = [x for x in r if x != c]
        if rr:
            new_idx[i] = len(reduced)
            reduced.append(rr)
    if not reduced:
        return None

    cands: list[tuple[float, int, int]] = []
    for ri, r in enumerate(reduced):
        for pos in range(1, len(r) + 2):
            u = r[pos - 2] if pos >= 2 else 0
            v = r[pos - 1] if pos <= len(r) else 0
            delta = float(D[u, c] + D[c, v] - D[u, v])
            cands.append((delta, ri, pos))
    cands.sort()
    M = len(cands)
    pos_of = {(ri, pos): j for j, (_, ri, pos) in enumerate(cands)}
    groups: dict[int, list[int]] = {}
    for j, (_, ri, _) in enumerate(cands):
        groups.setdefault(ri, []).append(j)

    # Route of the rank-1 candidate first: its rank-1 flags are the CRN reference x_1.
    ri1 = cands[0][1]
    order = [ri1] + [ri for ri in range(len(reduced)) if ri != ri1]
    p = np.empty(M)
    ap = np.empty(M)
    am = np.empty(M)
    varD = np.empty(M)
    base_risk: dict[int, float] = {}
    x1 = None
    dc, pc = dsc[c], psc[c]
    for ri in order:
        prof = load_profile(reduced[ri], dsc, psc)
        PM, SM = insertion_envelopes(prof)
        base_risk[ri] = float((prof.max(axis=0) > Q).mean())
        for j in groups[ri]:                      # ascending j; j = 0 (rank 1) comes first
            pos = cands[j][2]
            x = insertion_peak(PM, SM, pos, dc, pc) > Q
            if x1 is None:
                if j != 0:
                    raise RuntimeError("rank-1 candidate must be the first one processed")
                x1 = x
            d = x.astype(np.int8) - x1.astype(np.int8)
            p[j] = x.mean()
            ap[j] = np.mean(d == 1)
            am[j] = np.mean(d == -1)
            varD[j] = ap[j] + am[j] - (ap[j] - am[j]) ** 2

    if i0 in new_idx:
        j0 = pos_of[(new_idx[i0], p0)]
        orig_rank, p_orig, dropped = j0 + 1, float(p[j0]), False
    else:
        orig_rank, p_orig, dropped = 0, float("nan"), True

    return {
        "M": M, "cands": cands, "p": p, "ap": ap, "am": am, "varD": varD,
        "reduced": reduced, "base_risk": [base_risk[ri] for ri in range(len(reduced))],
        "orig_rank": orig_rank, "p_orig": p_orig, "dropped": dropped,
    }


def step_record(p, ap, am, varD, M: int) -> dict:
    """Statistics of one repair step from its candidate arrays (index 0 = rank 1).

    p, ap, am, varD hold the candidates j = 1, 2, ... in sorted order: the full
    arrays of length M in mode single, or in mode rr a prefix that covers every
    index read here (see rr_insert_step). Returns p_1, s, p_sorted_head and the
    per-eps keys (suffixes _e2, _e3x, _e3).
    """
    p = np.asarray(p, dtype=np.float64)
    ap = np.asarray(ap, dtype=np.float64)
    am = np.asarray(am, dtype=np.float64)
    vD = np.asarray(varD, dtype=np.float64)
    L = p.size
    p1 = float(p[0])
    v1 = p1 * (1.0 - p1)
    s = stop_index(p, ALPHA)
    sx = s if s >= 1 else M
    rec = {
        "p_1": p1,
        "s": s,
        "p_sorted_head": " ".join(f"{x:.6g}" for x in p[:HEAD]),
    }
    sizes = ideal_schedule(s, M)
    for eps in EPS:
        t = EPS_TAG[eps]
        k = kstar(p, ALPHA, eps)
        kk = k if k >= 1 else M                       # prefix j <= k (or all)
        band_p = p[:k - 1] if k >= 1 else p           # j < k (or all j when k = 0)
        band = int(np.count_nonzero((band_p > ALPHA - eps) & (band_p <= ALPHA + eps)))
        upto = min(M, 2 * k) if k >= 1 else M
        J = np.zeros(L, dtype=bool)
        J[:upto] = True
        J &= p <= 2 * ALPHA
        v_opt = (ALPHA - eps) * (1 - ALPHA + eps)
        vD_opt = 0.0
        if J.any():
            pj = p[J]
            v_opt = max(v_opt, float(np.max(pj * (1 - pj))))
            vD_opt = float(np.max(np.maximum(ap[J] * (1 - ap[J]), am[J] * (1 - am[J]))))
        varD_mean = float(np.mean(vD[:kk]))
        rec.update({
            f"k_{t}": k,
            f"band_{t}": band,
            f"v_opt_{t}": v_opt,
            f"vD_opt_{t}": vD_opt,
            f"varD_mean_{t}": varD_mean,
            f"Nq_full_wc_{t}": quantum_search_cost(sizes, V_WC, eps, DELTA, M),
            f"Nq_full_opt_{t}": quantum_search_cost(sizes, v_opt, eps, DELTA, M),
            f"Nc_full_wc_{t}": classical_scan_cost(s, V_WC, eps, DELTA, M),
            f"Nc_full_opt_{t}": classical_scan_cost(s, v_opt, eps, DELTA, M),
            f"Nq_core_{t}": quantum_core_cost(sx, v_opt, eps),
            f"Nc_core_{t}": classical_core_cost(sx, v_opt, eps),
            f"Nq_crn_core_{t}": crn_quantum_core_cost(sx, vD_opt, v1, eps),
            f"Nc_crn_core_{t}": crn_classical_core_cost(sx, varD_mean, v1, eps),
        })
    return rec


def repair_step(plan_routes, c, D, dsc, psc, Q) -> dict | None:
    """Statistics of one repair step (customer c removed and re-inserted).

    Returns None when c is the only customer of the plan (no reduced route).
    """
    T = _candidate_table(plan_routes, c, D, dsc, psc, Q)
    if T is None:
        return None
    p, ap, am, vD = T["p"], T["ap"], T["am"], T["varD"]
    M = T["M"]
    stats = step_record(p, ap, am, vD, M)
    rec = {
        "M": M,
        "n_routes": len(T["reduced"]),
        "dropped": int(T["dropped"]),
        "p_1": stats["p_1"],
        "s": stats["s"],
        "orig_rank": T["orig_rank"],
        "p_orig": T["p_orig"],
        "feasible_routes": int(sum(b <= ALPHA for b in T["base_risk"])),
        "p_sorted_head": stats["p_sorted_head"],
    }
    rec.update(stats)                                 # keys already present keep their place
    return rec


# ============================================================
# Mode rr: ruin-and-recreate
# ============================================================


def rr_insert_step(routes, c, D, dsc, psc, Q, cache, full=False) -> dict | None:
    """Insert customer c into the current partial plan (mode rr).

    routes: list of routes (depot 0 implicit) WITHOUT c; updated in place. c goes
    to the candidate of rank s (cheapest candidate with p <= ALPHA), or opens a new
    route when s = 0. cache maps a route index to its (PM, SM) envelopes and is
    cleared for the route that changes.

    Candidates are sorted by (delta, ri, pos). Risks are evaluated in that order
    and the evaluation stops after j_stop = min(M, max(2 k_e2, s, HEAD)), where
    k_e2 is the first j with p_j <= ALPHA - 0.01 (s <= k_e2 always). The HEAD term
    keeps p_sorted_head identical to a full evaluation. full=True evaluates all M
    candidates (reference for the tests). Returns the step record, or None when no
    candidate exists (c then opens a route).
    """
    cands: list[tuple[float, int, int]] = []
    for ri, r in enumerate(routes):
        for pos in range(1, len(r) + 2):
            u = r[pos - 2] if pos >= 2 else 0
            v = r[pos - 1] if pos <= len(r) else 0
            cands.append((float(D[u, c] + D[c, v] - D[u, v]), ri, pos))
    cands.sort()
    M = len(cands)
    n_routes = len(routes)
    if M == 0:
        routes.append([c])
        return None

    dc, pc = dsc[c], psc[c]
    p_l: list[float] = []
    ap_l: list[float] = []
    am_l: list[float] = []
    vd_l: list[float] = []
    x1 = None
    s_found = 0
    k_e2 = 0
    limit = M
    head_min = min(M, HEAD)
    j = 0
    while j < limit:
        _, ri, pos = cands[j]
        if ri not in cache:
            cache[ri] = insertion_envelopes(load_profile(routes[ri], dsc, psc))
        PM, SM = cache[ri]
        x = insertion_peak(PM, SM, pos, dc, pc) > Q
        if x1 is None:                               # rank-1 candidate: CRN reference
            x1 = x
        d = x.astype(np.int8) - x1.astype(np.int8)
        pj = float(x.mean())
        apj = float(np.mean(d == 1))
        amj = float(np.mean(d == -1))
        p_l.append(pj)
        ap_l.append(apj)
        am_l.append(amj)
        vd_l.append(apj + amj - (apj - amj) ** 2)
        if s_found == 0 and pj <= ALPHA:
            s_found = j + 1
        if not full and k_e2 == 0 and pj <= ALPHA - EPS[0]:
            k_e2 = j + 1
            limit = min(M, max(2 * k_e2, s_found, head_min))
        j += 1

    stats = step_record(np.array(p_l), np.array(ap_l), np.array(am_l), np.array(vd_l), M)
    s = stats["s"]
    rec = {"M": M, "n_routes": n_routes, "p_1": stats["p_1"], "s": s,
           "p_sorted_head": stats["p_sorted_head"]}
    rec.update(stats)
    rec.update({"n_evaluated": j, "chosen_rank": s, "new_route": int(s == 0)})
    if s >= 1:
        _, ri, pos = cands[s - 1]
        routes[ri].insert(pos - 1, c)
        cache.pop(ri, None)
    else:
        routes.append([c])
    return rec


def rr_round(base, D, dsc, psc, Q, rng, full=False):
    """One ruin-and-recreate round on the plan base (list of routes, depot 0 implicit).

    Ruin: q = max(1, int(Q_FRAC * n)) customers, drawn without replacement by
    rng.choice on the sorted customer list (kept as a Python list), are removed;
    empty routes are dropped and the removed list is shuffled. Recreate: the
    removed customers are inserted in that order by rr_insert_step.
    Returns (final routes, steps) with steps = [(step index, customer, record|None)].
    """
    C = sorted({c for r in base for c in r})
    q = max(1, int(Q_FRAC * len(C)))
    removed = [int(x) for x in rng.choice(C, q, replace=False)]
    rem = set(removed)
    routes = [[c for c in r if c not in rem] for r in base]
    routes = [r for r in routes if r]
    rng.shuffle(removed)
    cache: dict = {}
    steps = []
    for k, c in enumerate(removed):
        steps.append((k, c, rr_insert_step(routes, c, D, dsc, psc, Q, cache, full=full)))
    return routes, steps


# ============================================================
# Worker, parts and outputs
# ============================================================


def _parts_dir(N: int, steps: int) -> Path:
    return PARTS_ROOT / f"N{N}_steps{steps}_seed{DR.SEED}_m4b{SEED_M4B}"


def _rr_parts_dir(N: int, rounds: int) -> Path:
    return RR_PARTS_ROOT / f"N{N}_rounds{rounds}_seed{DR.SEED}_rr{SEED_RR}"


def _work_m4b(task: tuple) -> dict:
    """All repair steps of one plan: gates Det (and SAA when present), shared scenarios."""
    path, N, steps, parts_dir = task
    t0 = time.time()
    plan = json.loads(Path(path).read_text())
    name = plan["name"]
    fam = family_of(name)
    D = distance_matrix(name)
    if D.shape[0] != int(plan["n"]):
        raise ValueError(f"{name}: distance matrix has {D.shape[0]} nodes, plan has n = {plan['n']}")
    Q = float(plan["Q"])
    dsc, psc = M3.draw_scenarios(plan, N, DR.RHO, DR.SEED)

    rows: list[dict] = []
    skipped = 0
    for gate in GATES:
        if gate not in plan["res"]:
            continue
        routes = [[int(c) for c in r] for r in plan["res"][gate]["plan"] if len(r) > 0]
        C = sorted({c for r in routes for c in r})
        rng = M3._rng(SEED_M4B, name, gate)
        take = min(int(steps), len(C))
        for idx in rng.choice(len(C), size=take, replace=False):
            c = C[int(idx)]
            rec = repair_step(routes, c, D, dsc, psc, Q)
            if rec is None:
                skipped += 1
                continue
            rows.append({"instance": name, "family": fam, "gate": gate, "customer": int(c), **rec})
    seconds = time.time() - t0
    out = {"name": name, "family": fam, "N": N, "steps": int(steps), "rows": pd.DataFrame(rows),
           "skipped": skipped, "seconds": seconds}
    parts_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, parts_dir / f"{name}.pkl")
    return {"name": name, "n_steps": len(rows), "skipped": skipped, "seconds": seconds}


def _work_rr(task: tuple) -> dict:
    """All ruin-and-recreate steps of one plan: gates Det (and SAA), ROUNDS rounds each."""
    path, N, rounds, parts_dir = task
    t0 = time.time()
    plan = json.loads(Path(path).read_text())
    name = plan["name"]
    fam = family_of(name)
    D = distance_matrix(name)
    if D.shape[0] != int(plan["n"]):
        raise ValueError(f"{name}: distance matrix has {D.shape[0]} nodes, plan has n = {plan['n']}")
    Q = float(plan["Q"])
    dsc, psc = M3.draw_scenarios(plan, N, DR.RHO, DR.SEED)

    rows: list[dict] = []
    skipped = 0
    for gate in GATES:
        if gate not in plan["res"]:
            continue
        base = [[int(c) for c in r] for r in plan["res"][gate]["plan"] if len(r) > 0]
        for rnd in range(1, int(rounds) + 1):
            rng = M3._rng(SEED_RR, name, gate, rnd)
            _, steps = rr_round(base, D, dsc, psc, Q, rng)
            for k, c, rec in steps:
                if rec is None:
                    skipped += 1
                    continue
                rows.append({"instance": name, "family": fam, "gate": gate, "round": rnd,
                             "step": k, "customer": int(c), **rec})
    seconds = time.time() - t0
    out = {"name": name, "family": fam, "N": N, "rounds": int(rounds), "rows": pd.DataFrame(rows),
           "skipped": skipped, "seconds": seconds}
    parts_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, parts_dir / f"{name}.pkl")
    return {"name": name, "n_steps": len(rows), "skipped": skipped, "seconds": seconds}


def _load_parts(paths, parts_dir: Path) -> pd.DataFrame:
    frames = []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        f = parts_dir / f"{name}.pkl"
        if f.exists():
            frames.append(pd.read_pickle(f)["rows"])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_steps(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STEPS_CSV, index=False, float_format="%.6g")
    log(f"wrote {STEPS_CSV.name} ({len(df)} rows)")


def write_rr_steps(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RR_STEPS_CSV, index=False, float_format="%.6g")
    log(f"wrote {RR_STEPS_CSV.name} ({len(df)} rows)")


def run_part(args) -> None:
    paths = M3._list_instances(args.instances, args.every)
    if not paths:
        raise SystemExit(f"No plan files match {args.instances!r}")
    parts_dir = _parts_dir(args.N, args.steps)
    todo = []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        if args.force or not (parts_dir / f"{name}.pkl").exists():
            todo.append((str(p), args.N, args.steps, parts_dir))
    log(f"m4b: {len(paths)} instances selected, {len(todo)} to compute; N={args.N} "
        f"steps={args.steps} seed={DR.SEED} sampling seed={SEED_M4B} workers={args.workers}")
    t0 = time.time()
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for k, info in enumerate(pool.imap_unordered(_work_m4b, todo), 1):
                log(f"  [{k}/{len(todo)}] {info['name']}: {info['n_steps']} steps, "
                    f"{info['skipped']} skipped, {info['seconds']:.1f} s")
    log(f"m4b compute finished in {time.time() - t0:.1f} s; aggregating")
    write_steps(_load_parts(paths, parts_dir))


def run_rr(args) -> None:
    paths = M3._list_instances(args.instances, args.every)
    if not paths:
        raise SystemExit(f"No plan files match {args.instances!r}")
    parts_dir = _rr_parts_dir(args.N, args.rounds)
    todo = []
    for p in paths:
        name = json.loads(p.read_text())["name"]
        if args.force or not (parts_dir / f"{name}.pkl").exists():
            todo.append((str(p), args.N, args.rounds, parts_dir))
    log(f"m4b rr: {len(paths)} instances selected, {len(todo)} to compute; N={args.N} "
        f"rounds={args.rounds} q_frac={Q_FRAC} seed={DR.SEED} rr seed={SEED_RR} workers={args.workers}")
    t0 = time.time()
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for k, info in enumerate(pool.imap_unordered(_work_rr, todo), 1):
                log(f"  [{k}/{len(todo)}] {info['name']}: {info['n_steps']} steps, "
                    f"{info['skipped']} skipped, {info['seconds']:.1f} s")
    log(f"m4b rr compute finished in {time.time() - t0:.1f} s; aggregating")
    write_rr_steps(_load_parts(paths, parts_dir))


# ------------------------------------------------------------ summary

def _ratio_median(sub: pd.DataFrame, num: str, den: str) -> float:
    return float((sub[num] / sub[den]).median())


def _group_row(sub: pd.DataFrame, t: str) -> dict:
    k = sub[f"k_{t}"]
    row = {
        "steps": len(sub),
        "k0": float((k == 0).mean()),
        "k1": float((k == 1).mean()),
        "k_q50": float(k.quantile(0.5)),
        "k_q90": float(k.quantile(0.9)),
        "k_max": int(k.max()),
        "band_med": float(sub[f"band_{t}"].median()),
        "Nq/Nc full wc": _ratio_median(sub, f"Nq_full_wc_{t}", f"Nc_full_wc_{t}"),
        "Nq/Nc full opt": _ratio_median(sub, f"Nq_full_opt_{t}", f"Nc_full_opt_{t}"),
        "Nq/Nc core": _ratio_median(sub, f"Nq_core_{t}", f"Nc_core_{t}"),
        "Nq/Nc crn core": _ratio_median(sub, f"Nq_crn_core_{t}", f"Nc_crn_core_{t}"),
        "share Nq<Nc full wc": float((sub[f"Nq_full_wc_{t}"] < sub[f"Nc_full_wc_{t}"]).mean()),
        "share Nq<Nc full opt": float((sub[f"Nq_full_opt_{t}"] < sub[f"Nc_full_opt_{t}"]).mean()),
    }
    return row


def _md_table(rows: list[dict], lead: list[str]) -> list[str]:
    cols = lead + [c for c in rows[0] if c not in lead]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.3g}" if abs(v) < 1e4 else f"{v:.3e}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


def summary_markdown(df: pd.DataFrame, rr: bool = False) -> str:
    """Markdown summary per family x gate, with ALL rows, for each eps.

    rr=True adds a table of the share of steps that opened a new route and the
    median chosen rank (0 for new-route steps) per family x gate.
    """
    if df.empty:
        raise ValueError("no steps to summarise")
    families = [f for f in FAMILY_ORDER if f in set(df["family"])]
    gates = [g for g in GATES if g in set(df["gate"])]
    groups = [(f, g) for f in families + ["ALL"] for g in gates + ["ALL"]]

    def sub_of(f, g):
        m = np.ones(len(df), dtype=bool)
        if f != "ALL":
            m &= (df["family"] == f).to_numpy()
        if g != "ALL":
            m &= (df["gate"] == g).to_numpy()
        return df[m]

    title = ("# QUASAR M4b repair-step statistics (task B, mode rr: ruin-and-recreate)" if rr
             else "# QUASAR M4b repair-step statistics (task B)")
    lines = [
        title,
        "",
        f"Steps: {len(df)} over {df['instance'].nunique()} instances; "
        f"alpha = {ALPHA}, DELTA = {DELTA}, V_WC = {V_WC}; "
        "ratios are medians of per-step Nq/Nc; share Nq<Nc = share of steps where the "
        "quantum cost is below the classical one.",
        "",
        "## Stop index s (eps-free)",
        "",
    ]
    s_rows = []
    for f, g in groups:
        sub = sub_of(f, g)
        if sub.empty:
            continue
        s_rows.append({
            "family": f, "gate": g, "steps": len(sub),
            "s=0": float((sub["s"] == 0).mean()),
            "s_q50": float(sub["s"].quantile(0.5)),
            "s_q90": float(sub["s"].quantile(0.9)),
            "s_max": int(sub["s"].max()),
            "M_med": float(sub["M"].median()),
        })
    lines += _md_table(s_rows, ["family", "gate"])
    for eps in EPS:
        t = EPS_TAG[eps]
        lines += ["", f"## eps = {eps:g} (suffix _{t})", ""]
        rows = []
        for f, g in groups:
            sub = sub_of(f, g)
            if sub.empty:
                continue
            rows.append({"family": f, "gate": g, **_group_row(sub, t)})
        lines += _md_table(rows, ["family", "gate"])
    if rr:
        lines += ["", "## Insertions in mode rr (new route share, median chosen rank)", ""]
        rows = []
        for f, g in groups:
            sub = sub_of(f, g)
            if sub.empty:
                continue
            rows.append({
                "family": f, "gate": g, "steps": len(sub),
                "new route share": float(sub["new_route"].mean()),
                "chosen rank median": float(sub["chosen_rank"].median()),
            })
        lines += _md_table(rows, ["family", "gate"])
    return "\n".join(lines) + "\n"


def write_summary(args) -> None:
    paths = M3._list_instances(args.instances, args.every)
    df = _load_parts(paths, _parts_dir(args.N, args.steps))
    if df.empty:
        raise SystemExit("no parts found for the selected instances; run without --summary first")
    write_steps(df)
    SUMMARY_MD.write_text(summary_markdown(df), encoding="utf-8")
    log(f"wrote {SUMMARY_MD.name} ({len(df)} steps)")


def write_rr_summary(args) -> None:
    paths = M3._list_instances(args.instances, args.every)
    df = _load_parts(paths, _rr_parts_dir(args.N, args.rounds))
    if df.empty:
        raise SystemExit("no rr parts found for the selected instances; run without --summary first")
    write_rr_steps(df)
    RR_SUMMARY_MD.write_text(summary_markdown(df, rr=True), encoding="utf-8")
    log(f"wrote {RR_SUMMARY_MD.name} ({len(df)} steps)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("single", "rr"), default="single",
                    help="single: one removed customer per step; rr: ruin-and-recreate rounds")
    ap.add_argument("--instances", default=None, help="fnmatch pattern on plan names")
    ap.add_argument("--every", type=int, default=1, help="use every k-th instance (sorted by name)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--N", type=int, default=None,
                    help=f"common scenarios per instance (default {N_SCEN} single, {N_RR} rr)")
    ap.add_argument("--steps", type=int, default=STEPS_MAX,
                    help="single: sampled customers per (instance, gate); ignored in rr")
    ap.add_argument("--rounds", type=int, default=ROUNDS,
                    help="rr: ruin-and-recreate rounds per (instance, gate); ignored in single")
    ap.add_argument("--force", action="store_true", help="recompute cached instances")
    ap.add_argument("--summary", action="store_true", help="only rebuild the outputs from the parts")
    args = ap.parse_args(argv)
    rr = args.mode == "rr"
    if args.N is None:
        args.N = N_RR if rr else N_SCEN
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    M3._LOG_PATH = OUT_DIR / ("m4b_rr.log" if rr else "m4b.log")
    log(f"command: {' '.join(sys.argv)}")
    if args.summary:
        write_rr_summary(args) if rr else write_summary(args)
    else:
        run_rr(args) if rr else run_part(args)


if __name__ == "__main__":
    main()
