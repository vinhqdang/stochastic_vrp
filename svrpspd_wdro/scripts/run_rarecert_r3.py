"""RARECERT revision round 2 (internal review 2): the extra experiments.

Parts (--part), each a list of independent tasks with a checkpoint, --shard and --reverse:

  passrate   every hub of results/plans_Region (gate SAA) at its planned capacity Q:
             hub overflow probability (hub SMIS, N_hub draws) and route probabilities
             (route SMIS, N_r draws each); pass rates per gamma are made in the summary.
  mhat       every row of results/rarecert/smis_routes.csv (120 routes x 4 levels): the
             computed missed-mode mass m_hat = exp(-(E_other - E_min)) next to the measured
             single-shift error of that row.
  unconf     pickup rescaling and correlated factors: (route, pickup scale s, factor
             correlation r, gamma) variants; reference = SMIS with N_ref draws, cross-check
             = CE mixture; single CE shift as the method under test.
  hubcrude   hub SMIS (N = 1e6) against crude Monte Carlo (N = 1e8) on 8 hubs at the
             calibrated gamma = 1e-3 capacity of results/rarecert/plan_hubs.csv.
  rivals2    the 30 baseline routes at 1e-4 and 1e-6: Glasserman-Li (shifted inner step)
             with 1e6 evaluations, CE mixture with a small pilot, the two-end mixture (the
             two end dominating points only), SMIS re-timed, and the quadrature reference
             on coarser grids (accuracy against the fine reference, wall time).
  coverage2  coverage of the nominal 95% interval at 1e-5 and 1e-6 on balanced routes:
             SMIS (its proposal is deterministic, built once) and the single CE shift
             rebuilt in every replication.
  long       routes with 15 or more stops from the region plans at 1e-4 and 1e-6:
             SMIS, single shift, CE mixture against the quadrature reference.

Usage (from svrpspd_wdro/):
    python3 scripts/run_rarecert_r3.py --part PART [--workers 3] [--shard i/n] [--reverse]
        [--out-dir DIR]
    python3 scripts/run_rarecert_r3.py --part PART --collect DIR1,DIR2,...   # merge checkpoints,
        write PART.csv into --out-dir
Outputs: <out-dir>/<part>.csv (one row per task), <part>_checkpoint.jsonl, <part>.log.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_quasar_rare as rq  # noqa: E402
import run_rarecert_smis as rs  # noqa: E402
import run_rarecert_sensitivity as sens  # noqa: E402
from quantum.rare import (  # noqa: E402
    Mixture, RouteSim, ce_mixture, ce_shift, is_mixture, is_shift, threshold_for)
from rarecert.baselines import exact_reference, exact_route_prob, gl_two_step  # noqa: E402
from rarecert.checkpoint import load, resumable_map  # noqa: E402
from rarecert.plan import HubSim, hub_estimate, route_baselines  # noqa: E402
from rarecert.smis import smis_mixture, stage_scales  # noqa: E402

RES = ROOT / "results" / "rarecert"
PLANS = ROOT / "results" / "plans_Region"
GATE = "SAA"
SEED = 41
CV, RHO = rq.CV, rq.RHO
ETA = 1e-3

N_HUB_Q = 100_000
N_ROUTE_Q = 20_000
N_UNCONF = 200_000
N_UNCONF_REF = 1_000_000
N_HUB_SMIS = 1_000_000
N_HUB_CRUDE = 100_000_000
CRUDE_CHUNK = 200_000
N_RIVAL = 1_000_000
CE_SMALL_PILOT = 2_000
COARSE_GRIDS = ((21, (128, 256)), (21, (256, 512)), (41, (128, 256)), (11, (256, 512)))
COV_REPS = 300
COV_N = 10_000
N_LONG = 1_000_000
LONG_MIN_L = 15
LONG_ROUTES = 12
HUBCRUDE_EXTRA = ("VNN-THAINGUYEN", "VNN-BACGIANG")
HUBCRUDE_SMALLEST = 6


# ============================================================
# Simulators
# ============================================================


class CorrRouteSim(RouteSim):
    """RouteSim whose pickup factor is correlated with the delivery factor:
    W_p' = r W_d + sqrt(1 - r^2) W_p. The inputs stay standard normal, so every IS method
    applies unchanged; only the map from inputs to demands differs."""

    def __init__(self, dbar, pbar, cv=CV, rho=RHO, r=0.0):
        super().__init__(dbar, pbar, cv=cv, rho=rho)
        self.r = float(r)
        self._rc = math.sqrt(1.0 - self.r * self.r)

    def demands(self, x):
        x = np.array(x, dtype=np.float64, copy=True)
        if x.ndim == 1:
            x = x[None, :]
        if self.r != 0.0:
            x[:, 1] = self.r * x[:, 0] + self._rc * x[:, 1]
        return super().demands(x)


def _rng(*parts):
    return np.random.default_rng([SEED] + [int(abs(hash(p)) % (2 ** 31)) if isinstance(p, str) else int(p)
                                           for p in parts])


def _sim_for(rid, means, s=1.0, r=0.0):
    m = means[rid]
    return CorrRouteSim(np.asarray(m["dbar"], float), np.asarray(m["pbar"], float) * s, r=r)


def _hub_plan(path):
    doc = json.loads(Path(path).read_text())
    plan = {"dbar": doc["dbar"], "pbar": doc["pbar"], "routes": doc["res"][GATE]["plan"], "Q": float(doc["Q"])}
    return doc["name"], plan


def two_end_mixture(sim, c, mix, diag, defensive=0.05):
    """The SMIS proposal restricted to the two end modes (delivery block of stage 0,
    pickup block of stage L), reweighted by exp(-(E - E_min))."""
    keep = [i for i, (k, lab) in enumerate(diag["labels"])
            if (k == 0 and str(lab).startswith("d")) or (k == sim.L and str(lab).startswith("p"))]
    if not keep:
        raise RuntimeError("no end mode among the kept SMIS modes")
    E = np.array([diag["E"][i] for i in keep])
    mass = np.exp(-(E - E.min()))
    a = (1.0 - defensive) * mass / mass.sum()
    thetas = np.vstack([mix.thetas[0:1]] + [mix.thetas[1 + i][None, :] for i in keep])
    return Mixture(thetas=thetas, weights=np.concatenate([[defensive], a]),
                   stages=[None] + [diag["labels"][i][0] for i in keep], q=mass, draws=0, skipped=[])


# ============================================================
# Tasks
# ============================================================


def task_passrate(t):
    t0 = time.time()
    name, plan = _hub_plan(t["path"])
    row = {"name": name, "region": name.split("-")[0], "Q": plan["Q"], "error": ""}
    try:
        hub = HubSim(plan, cv=CV, rho=RHO)
        row.update({"n_customers": hub.n, "n_routes": hub.R})
        rng = _rng("passrate", name)
        est = hub_estimate(hub, plan["Q"], t["n_hub"], rng)
        rb = route_baselines(hub, plan["Q"], rng, N_r=t["n_route"])
        row.update({"p_hub": est["p_hub"], "rse_hub": est["rse"], "multi_frac": est["multi_frac"],
                    "route_p": [float(v) for v in rb["route_p_smis"]],
                    "route_rse": [float(v) if math.isfinite(v) else None for v in rb["route_rse"]],
                    "route_L": [len(r) for r in hub.routes]})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.time() - t0
    return row


def task_mhat(t):
    t0 = time.time()
    row = dict(t["row"])
    try:
        sim = _sim_for(t["row"]["route_id"], t["means"])
        c = float(t["row"]["c"])
        mix, diag = smis_mixture(sim, c, eta=ETA, n_arc=0)
        E_d0, E_pL = sens.end_energies(sim, c)
        E_min = diag["E_min"]
        E_other = max(E_d0, E_pL)
        row.update({"E_min": E_min, "E_d0": E_d0, "E_pL": E_pL,
                    "m_hat": math.exp(-(E_other - E_min)) if math.isfinite(E_other) else 0.0,
                    "n_modes": diag["n_modes"], "error": ""})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.time() - t0
    return row


def task_unconf(t):
    t0 = time.time()
    rid, s, r, g = t["route_id"], t["s"], t["r"], t["gamma"]
    row = {"route_id": rid, "region": rid.split("-")[0], "L": t["means"][rid]["L"], "s": s, "r": r,
           "gamma": g, "error": ""}
    try:
        sim = _sim_for(rid, t["means"], s=s, r=r)
        st = stage_scales(sim)
        row["varrho"] = st["ratio"]
        rng = _rng("unconf", rid, int(s * 100), int(r * 100), int(-math.log10(g)))
        c = threshold_for(sim, g, rng)
        row["c"] = c
        mix, diag = smis_mixture(sim, c, eta=ETA, n_arc=0)
        E_d0, E_pL = sens.end_energies(sim, c)
        E_other = max(E_d0, E_pL)
        row.update({"n_modes": diag["n_modes"], "E_min": diag["E_min"],
                    "m_hat": math.exp(-(E_other - diag["E_min"])) if math.isfinite(E_other) else 0.0})
        p_ref, RV_ref, rse_ref, _ = is_mixture(sim, c, mix, t["n_ref"], rng)
        row.update({"p_ref": p_ref, "RV_ref": RV_ref, "rse_ref": rse_ref})
        info: dict = {}
        theta = ce_shift(sim, c, rng, info=info)
        p1, RV1, rse1, _ = is_shift(sim, c, theta, t["n"], rng)
        row.update({"p_single": p1, "RV_single": RV1, "rse_single": rse1})
        cem = ce_mixture(sim, c, rng)
        pc, RVc, rsec, _ = is_mixture(sim, c, cem, t["n"], rng)
        row.update({"p_ce": pc, "RV_ce": RVc, "rse_ce": rsec})
        if p_ref > 0:
            row.update({"e_single": p1 / p_ref - 1.0, "e_ce": pc / p_ref - 1.0})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.time() - t0
    return row


def task_hubcrude(t):
    t0 = time.time()
    name, plan = _hub_plan(PLANS / f"{t['name']}.json")
    row = {"name": name, "c": t["c"], "gamma": t["gamma"], "error": ""}
    try:
        hub = HubSim(plan, cv=CV, rho=RHO)
        row.update({"n_customers": hub.n, "n_routes": hub.R})
        rng = _rng("hubcrude", name)
        est = hub_estimate(hub, t["c"], t["n_smis"], rng)
        row.update({"p_smis": est["p_hub"], "rse_smis": est["rse"], "RV_smis": est["RV"],
                    "smis_seconds": est["build_seconds"] + est["sample_seconds"]})
        hits, n_done, tc = 0, 0, time.time()
        while n_done < t["n_crude"]:
            m = min(CRUDE_CHUNK, t["n_crude"] - n_done)
            hits += int(hub.fail(rng.standard_normal((m, hub.dim)), t["c"]).sum())
            n_done += m
        p_c = hits / n_done
        row.update({"p_crude": p_c, "hits": hits, "n_crude": n_done,
                    "rse_crude": math.sqrt((1 - p_c) / (p_c * n_done)) if hits else math.nan,
                    "crude_seconds": time.time() - tc})
        if hits:
            row["rel"] = row["p_smis"] / p_c - 1.0
            row["z"] = (row["p_smis"] - p_c) / math.hypot(row["rse_smis"] * row["p_smis"], row["rse_crude"] * p_c)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.time() - t0
    return row


def task_rivals2(t):
    t0 = time.time()
    rid, g, c, p_ref = t["route_id"], t["gamma"], t["c"], t["p_ref"]
    row = {"route_id": rid, "gamma": g, "c": c, "p_ref": p_ref, "rho_band": t["rho_band"], "L": t["L"],
           "error": ""}
    try:
        sim = _sim_for(rid, t["means"])
        rng = _rng("rivals2", rid, int(-math.log10(g)))
        tb = time.perf_counter()
        mix, diag = smis_mixture(sim, c, eta=ETA, n_arc=0)
        build = time.perf_counter() - tb
        ts = time.perf_counter()
        p, RV, rse, _ = is_mixture(sim, c, mix, t["n"], rng)
        row.update({"smis_p": p, "smis_RV": RV, "smis_build_s": build, "smis_sample_s": time.perf_counter() - ts})
        tb = time.perf_counter()
        two = two_end_mixture(sim, c, mix, diag)
        p, RV, rse, _ = is_mixture(sim, c, two, t["n"], rng)
        row.update({"two_p": p, "two_RV": RV, "two_s": time.perf_counter() - tb + build,
                    "two_modes": len(two.weights) - 1})
        tb = time.perf_counter()
        p, RV, rse = gl_two_step(sim, c, t["n"], rng, variant="shift")
        row.update({"gl_p": p, "gl_RV": RV, "gl_rse": rse, "gl_s": time.perf_counter() - tb})
        tb = time.perf_counter()
        cem = ce_mixture(sim, c, rng, n_pilot=CE_SMALL_PILOT, n_q=CE_SMALL_PILOT)
        build_ce = time.perf_counter() - tb
        p, RV, rse, _ = is_mixture(sim, c, cem, t["n"], rng)
        row.update({"ce_small_p": p, "ce_small_RV": RV, "ce_small_pilot": cem.draws,
                    "ce_small_build_s": build_ce, "ce_small_s": time.perf_counter() - tb})
        for n_grid, vp in COARSE_GRIDS:
            tag = f"q{n_grid}_{vp[0]}"
            ref = exact_reference(sim, c, n=n_grid, V_pair=vp)
            row.update({f"{tag}_p": ref["p_ext"], f"{tag}_s": ref["wall_s"]})
        for k in ("smis", "two", "gl", "ce_small"):
            if p_ref > 0 and row.get(f"{k}_p") is not None:
                row[f"{k}_rel"] = row[f"{k}_p"] / p_ref - 1.0
        for n_grid, vp in COARSE_GRIDS:
            tag = f"q{n_grid}_{vp[0]}"
            row[f"{tag}_rel"] = row[f"{tag}_p"] / p_ref - 1.0
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.time() - t0
    return row


def task_coverage2(t):
    t0 = time.time()
    rid, g = t["route_id"], t["gamma"]
    rows = []
    base = {"route_id": rid, "L": t["means"][rid]["L"], "gamma": g, "N": t["n"], "reps": t["reps"], "error": ""}
    try:
        sim = _sim_for(rid, t["means"])
        rng = _rng("coverage2", rid, int(-math.log10(g)))
        c = threshold_for(sim, g, rng)
        ref = exact_reference(sim, c)
        p_ref = ref["p_ext"]
        mix, diag = smis_mixture(sim, c, eta=ETA, n_arc=0)

        def smis_est(r):
            p, _, rse, _ = is_mixture(sim, c, mix, t["n"], r)
            return p, rse

        def single_rebuilt(r):
            theta = ce_shift(sim, c, r)
            p, _, rse, _ = is_shift(sim, c, theta, t["n"], r)
            return p, rse

        for method, est in (("smis", smis_est), ("single_rebuilt", single_rebuilt)):
            ps, rses, hits, zeros = [], [], 0, 0
            for _ in range(t["reps"]):
                p, rse = est(rng)
                ps.append(p)
                if p <= 0:
                    zeros += 1
                    continue
                rses.append(rse)
                if math.isfinite(rse) and abs(p - p_ref) <= 1.96 * p * rse:
                    hits += 1
            ps = np.asarray(ps)
            rows.append({**base, "c": c, "p_ref": p_ref, "ref_err_est": ref["err_est"], "method": method,
                         "varrho": diag["ratio"], "coverage": hits / t["reps"],
                         "median_rse": float(np.median(rses)) if rses else math.nan, "n_zero": zeros,
                         "mean_rel": float(ps.mean() / p_ref - 1.0), "median_rel": float(np.median(ps) / p_ref - 1.0),
                         "sd_rel": float(ps.std(ddof=1) / p_ref)})
    except Exception as exc:
        rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})
    for r in rows:
        r["seconds"] = time.time() - t0
    return rows


def task_long(t):
    t0 = time.time()
    row = {"route_id": t["route_id"], "L": len(t["dbar"]), "gamma": t["gamma"], "error": ""}
    try:
        sim = CorrRouteSim(np.asarray(t["dbar"]), np.asarray(t["pbar"]))
        row["varrho"] = stage_scales(sim)["ratio"]
        rng = _rng("long", t["route_id"], int(-math.log10(t["gamma"])))
        c = threshold_for(sim, t["gamma"], rng)
        row["c"] = c
        tb = time.perf_counter()
        mix, diag = smis_mixture(sim, c, eta=ETA, n_arc=0)
        row["smis_build_s"] = time.perf_counter() - tb
        p, RV, rse, _ = is_mixture(sim, c, mix, t["n"], rng)
        row.update({"smis_p": p, "smis_RV": RV, "smis_rse": rse, "n_modes": diag["n_modes"]})
        theta = ce_shift(sim, c, rng)
        p, RV, rse, _ = is_shift(sim, c, theta, t["n"], rng)
        row.update({"single_p": p, "single_RV": RV, "single_rse": rse})
        cem = ce_mixture(sim, c, rng)
        p, RV, rse, _ = is_mixture(sim, c, cem, t["n"], rng)
        row.update({"ce_p": p, "ce_RV": RV, "ce_rse": rse, "ce_pilot": cem.draws})
        ref = exact_reference(sim, c)
        row.update({"p_ref": ref["p_ext"], "ref_err_est": ref["err_est"], "ref_s": ref["wall_s"]})
        for k in ("smis", "single", "ce"):
            row[f"{k}_rel"] = row[f"{k}_p"] / ref["p_ext"] - 1.0
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.time() - t0
    return row


TASK_FN = {"passrate": task_passrate, "mhat": task_mhat, "unconf": task_unconf, "hubcrude": task_hubcrude,
           "rivals2": task_rivals2, "coverage2": task_coverage2, "long": task_long}


def task_key(t):
    return t["key"]


# ============================================================
# Task lists
# ============================================================


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_tasks(part):
    tasks = []
    if part == "passrate":
        for f in sorted(glob.glob(str(PLANS / "*.json"))):
            tasks.append({"key": ["passrate", Path(f).stem], "path": f, "n_hub": N_HUB_Q, "n_route": N_ROUTE_Q})
        return tasks
    if part == "hubcrude":
        rows = [r for r in _csv(RES / "plan_hubs.csv") if math.isclose(float(r["gamma"]), 1e-3) and not r["error"]]
        rows.sort(key=lambda r: int(r["n_customers"]))
        pick = rows[:HUBCRUDE_SMALLEST] + [r for r in rows if r["name"] in HUBCRUDE_EXTRA]
        for r in pick:
            tasks.append({"key": ["hubcrude", r["name"]], "name": r["name"], "c": float(r["c"]), "gamma": 1e-3,
                          "n_smis": N_HUB_SMIS, "n_crude": N_HUB_CRUDE})
        return tasks
    if part == "long":
        cands = []
        for f in sorted(glob.glob(str(PLANS / "*.json"))):
            doc = json.loads(Path(f).read_text())
            db, pb = np.asarray(doc["dbar"]), np.asarray(doc["pbar"])
            for i, r in enumerate(doc["res"][GATE]["plan"]):
                if len(r) >= LONG_MIN_L:
                    cands.append({"route_id": f"{doc['name']}#{i}", "dbar": db[r].tolist(), "pbar": pb[r].tolist()})
        cands.sort(key=lambda d: (len(d["dbar"]), d["route_id"]))
        rng = np.random.default_rng(SEED)
        groups = np.array_split(np.arange(len(cands)), LONG_ROUTES)
        pick = [cands[g[int(rng.integers(len(g)))]] for g in groups if len(g)]
        for d in pick:
            for g in (1e-4, 1e-6):
                tasks.append({"key": ["long", d["route_id"], g], **d, "gamma": g, "n": N_LONG})
        return tasks
    means = rs.route_means()
    if part == "mhat":
        for r in _csv(RES / "smis_routes.csv"):
            if r["error"]:
                continue
            keep = {k: r[k] for k in ("route_id", "instance", "L", "gamma", "c", "ratio", "p_ref", "p_ref_kind",
                                      "p_single", "RV_single", "p_is", "smis0_p")}
            keep["region"] = r["instance"].split("-")[0]
            keep["e_single"] = float(r["p_single"]) / float(r["p_ref"]) - 1.0
            keep["rse_single"] = math.sqrt(float(r["RV_single"]) / 1e6)
            tasks.append({"key": ["mhat", r["route_id"], float(r["gamma"])], "row": keep, "means": means})
        return tasks
    rids = [r["route_id"] for r in sens.quasar_order(sens.RARE_CSV)]
    vnn = [r for r in rids if r.startswith("VNN")]
    gd = [r for r in rids if r.startswith("GD")]
    if part == "unconf":
        for rid in vnn:
            for s in (0.5, 2.0):
                for g in (1e-4, 1e-6):
                    tasks.append({"key": ["unconf", rid, s, 0.0, g], "route_id": rid, "s": s, "r": 0.0, "gamma": g})
        for rid in gd:
            for g in (1e-5, 1e-6):
                tasks.append({"key": ["unconf", rid, 0.5, 0.0, g], "route_id": rid, "s": 0.5, "r": 0.0, "gamma": g})
        for rid in vnn[:20] + gd[:20]:
            for r in (0.5, -0.5):
                tasks.append({"key": ["unconf", rid, 1.0, r, 1e-4], "route_id": rid, "s": 1.0, "r": r, "gamma": 1e-4})
        for t in tasks:
            t.update({"means": means, "n": N_UNCONF, "n_ref": N_UNCONF_REF})
        return tasks
    if part == "rivals2":
        for r in _csv(RES / "baselines_routes.csv"):
            if r["error"]:
                continue
            tasks.append({"key": ["rivals2", r["route_id"], float(r["gamma"])], "route_id": r["route_id"],
                          "gamma": float(r["gamma"]), "c": float(r["c"]), "p_ref": float(r["p_ref"]),
                          "rho_band": r.get("rho_band", ""), "L": int(r["L"]), "means": means, "n": N_RIVAL})
        return tasks
    if part == "coverage2":
        short_vnn = [rid for rid in vnn if means[rid]["L"] <= 8][:10]
        for rid in short_vnn:
            for g in (1e-5, 1e-6):
                tasks.append({"key": ["coverage2", rid, g], "route_id": rid, "gamma": g, "means": means,
                              "n": COV_N, "reps": COV_REPS})
        return tasks
    raise SystemExit(f"unknown part {part!r}")


# ============================================================
# Main
# ============================================================


def _flatten(results):
    out = []
    for r in results:
        out.extend(r if isinstance(r, list) else [r])
    return out


def write_csv(rows, path):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=sorted(TASK_FN))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--collect", default=None, help="comma-separated dirs whose checkpoints are merged")
    ap.add_argument("--limit", type=int, default=None, help="quick test: first N tasks")
    args = ap.parse_args(argv)
    out = Path(args.out_dir or RES / f"r3_{args.part}")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{args.part}_checkpoint.jsonl"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / f"{args.part}.log", mode="a"), logging.StreamHandler()])
    log = logging.getLogger("r3")

    if args.collect:
        done = {}
        for d in args.collect.split(","):
            done.update(load(Path(d) / f"{args.part}_checkpoint.jsonl"))
        rows = _flatten(list(done.values()))
        write_csv(rows, out / f"{args.part}.csv")
        log.info("collected %d results (%d rows) into %s", len(done), len(rows), out / f"{args.part}.csv")
        return 0

    tasks = build_tasks(args.part)
    i, n = (int(v) for v in args.shard.split("/"))
    tasks = [t for k, t in enumerate(tasks) if k % n == i]
    if args.reverse:
        tasks = tasks[::-1]
    if args.limit:
        tasks = tasks[: args.limit]
    log.info("part %s: %d tasks (shard %s, reverse %s), workers %d", args.part, len(tasks), args.shard,
             args.reverse, args.workers)
    fn = TASK_FN[args.part]
    results = []
    t0 = time.time()
    if args.workers <= 1:
        it = resumable_map(fn, tasks, task_key, ckpt)
        for k, res in enumerate(it, 1):
            results.append(res)
            log.info("[%d/%d] done", k, len(tasks))
    else:
        with mp.get_context("fork").Pool(processes=args.workers) as pool:
            for k, res in enumerate(resumable_map(fn, tasks, task_key, ckpt, pool), 1):
                results.append(res)
                log.info("[%d/%d] done", k, len(tasks))
    rows = _flatten(results)
    write_csv(rows, out / f"{args.part}.csv")
    n_err = sum(1 for r in rows if r.get("error"))
    log.info("wrote %s (%d rows, %d errors) in %.1f min", f"{args.part}.csv", len(rows), n_err,
             (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
