"""QUASAR M3b v3 (review fixes): estimator behaviour on the validated rotation model.

Design: papers/quasar/DESIGN_M3.md, Revision 2, "M3b v2 - estimator behaviour"
(v3 review-fix changes are listed in results/quasar/m3b_v3_summary.md, "Deviations").

Step 1  Circuit validation of sin^2((2k+1) theta), k = 0..6. Not rerun: the v2
        output results/quasar/m3b_v2_validation.csv is copied to m3b_v3_validation.csv
        after its maximum deviation is checked.
Step 2  Sample: stratified by move class (cls), stand-alone p' band
        (low < 0.02 <= decision <= 0.2 < high) and family group
        (CON/SCA, CMT, city): up to MOVES_PER_CELL eligible moves per cell,
        all eligible moves where a cell is smaller. Composition is printed.
Step 3  Error metrics per amplitude and setting:
        canonical QAE (m = 3..22) from the EXACT outcome pmf (no sampling);
        MLAE (J = 1..10, S in {32, 64, 128}) and classical Monte Carlo
        (N = 2^4..2^26) from N_SEEDS independent seeds per amplitude.
        Direct signed-mean classical estimator per move (N = 2^4..2^26, N_SEEDS
        seeds): the mean of N draws of D, no two-estimate protocol.
        Results are cached: an amplitude already simulated in the previous run
        (results/quasar/m3b_v3_metrics.csv) is reused exactly, because its
        seeds depend only on its stable slot label (amp<slot>).
Step 4  Log-log slopes of the 90 % quantile (primary) and RMS (secondary), and
        the cost of the difference estimate relative to the stand-alone estimate
        at absolute and relative error targets, by family group.

Eligible moves: every nonzero amplitude has at least HITS_MIN hits; moves with
P(D = +1) = P(D = -1) = 0 are excluded (not resolved by N_SCEN scenarios).

Run from svrpspd_wdro/:  python scripts/run_quasar_m3b.py
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the MLAE grid is a matrix product and the job shares the machine.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum.ae_sim import (  # noqa: E402
    classical_cost,
    classical_estimates,
    cost_at,
    exact_qae_metrics,
    mlae_cost,
    mlae_counts,
    mlae_estimates,
    predicted_cost_ratio,
    qae_cost,
    qae_estimates,
    sample_metrics,
    signed_mean_estimates,
    signed_mean_variance,
    target_scales,
)

OUT = ROOT / "results" / "quasar"
MOVES_GZ = OUT / "m3a_v2_moves.csv.gz"
V2_VALIDATION = OUT / "m3b_v2_validation.csv"
V2_RMS = OUT / "m3b_v2_rms.csv"
V3_METRICS = OUT / "m3b_v3_metrics.csv"  # cache of the previous run, read before it is overwritten

V3_SEED = 20261013
N_SCEN = 200_000
HITS_MIN = 200
N_SEEDS = 4000
QAE_M = list(range(3, 23))
MLAE_S = (32, 64, 128)
MLAE_J = list(range(1, 11))
CLASSICAL_N = [2**e for e in range(4, 27)]
MOVES_PER_CELL = 20
CLASSES = (1, 2, 3)
BAND_NAMES = ("low", "decision", "high")
GROUPS = ("CON/SCA", "CMT", "city")
ABS_TARGETS = (1e-3, 1e-4)
REL_TARGETS = (0.1, 0.01)
TARGETS = [("abs", v) for v in ABS_TARGETS] + [("rel", v) for v in REL_TARGETS]
MLAE_FAMILIES = [f"mlae_S{S}" for S in MLAE_S]
# Families simulated per amplitude record.
RECORD_FAMILIES = ["qae"] + MLAE_FAMILIES + ["classical"]
# Families of the difference-cost table. "classical" is the direct signed-mean
# estimator of the difference (one estimate of D); "classical_2est" is the
# two-estimate protocol (one estimate per nonzero side), kept as a secondary column.
RATIO_FAMILIES = ["qae"] + MLAE_FAMILIES + ["classical", "classical_2est"]
ASYMPTOTIC_LAST = 6
N_WORKERS = 2
EXACT_CHECK_N = 200_000
TINY = 1e-3  # P(D != 0) below this is not resolved by N_SCEN scenarios at HITS_MIN


def _log(msg: str) -> None:
    print(msg, flush=True)


def _cell(cls: int, band: str, group: str) -> str:
    return f"cls{cls}|{band}|{group}"


def _akey(a: float) -> str:
    return f"{float(a):.15e}"


def band_of(p: np.ndarray) -> np.ndarray:
    """low: p < 0.02; decision: 0.02 <= p <= 0.2; high: p > 0.2."""
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.02, "low", np.where(p <= 0.2, "decision", "high"))


def group_of(family) -> str:
    """Family group: CON/SCA (random-generator instances), CMT (CMT instances), city (OSM instances)."""
    f = str(family)
    if f in ("CON", "SCA"):
        return "CON/SCA"
    if f == "CMT":
        return "CMT"
    if f == "city":
        return "city"
    raise ValueError(f"unknown family {f!r}")


# ============================================================
# Step 1: circuit validation (reused from v2)
# ============================================================


def reuse_validation() -> list[dict]:
    with V2_VALIDATION.open() as fh:
        rows = list(csv.DictReader(fh))
    max_dev = max(float(r["abs_diff"]) for r in rows)
    _log(f"  validation (v2 output reused): {len(rows)} rows, max |p_circuit - p_model| = {max_dev:.2e}")
    if max_dev > 1e-12:
        raise RuntimeError("v2 validation does not match the expected 1e-12 level")
    return rows


# ============================================================
# Step 2: population, sample, amplitudes
# ============================================================


def load_moves():
    import pandas as pd

    cols = ["instance", "family", "route_id", "cls", "kind", "j", "pos", "pos2",
            "p_new", "P_D1", "P_Dm1", "rho_q"]
    return pd.read_csv(MOVES_GZ, usecols=cols)


def population_table(df):
    """Per group x class: the whole population, zero and below-resolution moves, and the eligible pool."""
    import pandas as pd

    d = df.copy()
    d["group"] = [group_of(f) for f in d["family"]]
    hp = np.rint(d["P_D1"].to_numpy(float) * N_SCEN)
    hm = np.rint(d["P_Dm1"].to_numpy(float) * N_SCEN)
    hs = np.rint(d["p_new"].to_numpy(float) * N_SCEN)
    zero = (hp == 0) & (hm == 0)
    under = ~zero & ((hs < HITS_MIN) | ((hp > 0) & (hp < HITS_MIN)) | ((hm > 0) & (hm < HITS_MIN)))
    pnz = d["P_D1"].to_numpy(float) + d["P_Dm1"].to_numpy(float)
    d["zero"], d["under"] = zero, under
    d["tiny"] = pnz < TINY
    d["elig"] = ~zero & ~under
    d["pnz"] = pnz
    d["row"] = np.arange(len(d))
    rows = []
    for g in GROUPS:
        for cls in CLASSES:
            s = d[(d["group"] == g) & (d["cls"] == cls)]
            e = s[s["elig"]]
            rows.append({
                "group": g, "cls": cls, "n_moves": int(len(s)),
                "n_zero": int(s["zero"].sum()), "n_under": int(s["under"].sum()),
                "n_eligible": int(len(e)),
                "share_pnz_below_1e-3": float(s["tiny"].mean()) if len(s) else float("nan"),
                "eligible_pnz_min": float(e["pnz"].min()) if len(e) else float("nan"),
                "eligible_pnz_median": float(e["pnz"].median()) if len(e) else float("nan"),
            })
    return rows, d


def select_moves(d, rng: np.random.Generator):
    """Stratified sample from the eligible pool of the prepared frame d (from population_table).

    Returns (selected DataFrame, info dict).
    """
    import pandas as pd

    d = d.copy()
    d["h_sa"] = np.rint(d["p_new"].to_numpy(float) * N_SCEN)
    d["h_p"] = np.rint(d["P_D1"].to_numpy(float) * N_SCEN)
    d["h_m"] = np.rint(d["P_Dm1"].to_numpy(float) * N_SCEN)
    info: dict = {"n_moves": int(len(d))}
    for name, h in (("p_new", d["h_sa"]), ("P_D1", d["h_p"]), ("P_Dm1", d["h_m"])):
        info[f"amp_{name}_zero_hits"] = int((h == 0).sum())
        info[f"amp_{name}_1_to_{HITS_MIN - 1}_hits"] = int(((h > 0) & (h < HITS_MIN)).sum())
        info[f"amp_{name}_ge_{HITS_MIN}_hits"] = int((h >= HITS_MIN).sum())
    d["status"] = np.where(d["zero"], "zero_difference",
                           np.where(d["under"], "under_threshold", "eligible"))
    info["moves_zero_difference"] = int((d["status"] == "zero_difference").sum())
    info["moves_under_threshold"] = int((d["status"] == "under_threshold").sum())
    info["moves_eligible"] = int((d["status"] == "eligible").sum())

    el = d[d["status"] == "eligible"].copy()
    el["band"] = band_of(el["p_new"].to_numpy(float))
    el["group"] = [group_of(f) for f in el["family"]]
    el["cell"] = [_cell(int(c), b, g) for c, b, g in zip(el["cls"], el["band"], el["group"])]
    picks, cells = [], []
    for cls in CLASSES:
        for band in BAND_NAMES:
            for grp in GROUPS:
                pool = el[(el["cls"] == cls) & (el["band"] == band) & (el["group"] == grp)]
                n = min(MOVES_PER_CELL, len(pool))
                if n > 0:
                    idx = np.sort(rng.choice(len(pool), size=n, replace=False))
                    picks.append(pool.iloc[idx])
                fam = pool.iloc[idx]["family"].value_counts().to_dict() if n > 0 else {}
                cells.append({
                    "cell": _cell(cls, band, grp), "cls": cls, "band": band, "group": grp,
                    "pool": int(len(pool)), "sampled": int(n),
                    "shortfall": bool(n < MOVES_PER_CELL),
                    "families": ", ".join(f"{k} {v}" for k, v in sorted(fam.items())) if fam else "-",
                })
    info["cells"] = cells
    sel = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    sel = sel.assign(mode=[_mode(hp, hm) for hp, hm in zip(sel["h_p"], sel["h_m"])])
    return sel, info


def _mode(h_p: float, h_m: float) -> str:
    if h_p > 0 and h_m > 0:
        return "signed"
    return "one-sided+" if h_p > 0 else "one-sided-"


def build_pairs(d, seed: int = V3_SEED, known_slots: dict | None = None):
    """Return (info, records, pairs).

    records: unique amplitudes [{label, slot, a, kind, key}], each simulated once.
    The label is amp<slot>; the slot fixes the seed. known_slots maps an
    amplitude key to the slot it had in the previous run, so cached results
    are reused; new amplitudes get fresh slots above every known slot.
    pairs: one per sampled move: {pid, row, cell, cls, band, group, mode, base,
    a_plus, a_minus, sa (record label), components [(record label, amplitude)],
    rho_q, ids}. components are the nonzero difference sides.
    """
    known_slots = dict(known_slots or {})
    rng = np.random.default_rng(seed)
    sel, info = select_moves(d, rng)
    records: list[dict] = []
    labels: dict[str, str] = {}
    next_slot = max(known_slots.values(), default=-1) + 1

    def rec_for(a: float, kind: str) -> str:
        nonlocal next_slot
        key = _akey(a)
        if key not in labels:
            if key in known_slots:
                slot = known_slots[key]
            else:
                slot = next_slot
                next_slot += 1
            lab = f"amp{slot:04d}"
            labels[key] = lab
            records.append({"label": lab, "slot": slot, "a": float(a), "kind": kind, "key": key})
        return labels[key]

    pairs: list[dict] = []
    for _, r in sel.iterrows():
        comps = []
        if r["h_p"] > 0:
            comps.append((rec_for(r["P_D1"], "difference+"), float(r["P_D1"])))
        if r["h_m"] > 0:
            comps.append((rec_for(r["P_Dm1"], "difference-"), float(r["P_Dm1"])))
        row = int(r["row"])
        pairs.append({
            "pid": f"mv{row:07d}", "row": row,
            "cell": r["cell"], "cls": int(r["cls"]), "band": r["band"], "group": r["group"],
            "mode": r["mode"],
            "base": float(r["p_new"]), "a_plus": float(r["P_D1"]), "a_minus": float(r["P_Dm1"]),
            "sa": rec_for(r["p_new"], "standalone"), "components": comps,
            "rho_q": float(r["rho_q"]),
            "ids": f"{r['instance']} {r['family']} route {int(r['route_id'])} {r['kind']} "
                   f"j={r['j']} pos={r['pos']} pos2={r['pos2']}",
        })
    return info, records, pairs


def composition_text(info) -> str:
    L = ["| cell (cls / band / group) | eligible pool | sampled | families in sample | shortfall |",
         "|---|---|---|---|---|"]
    for c in info["cells"]:
        L.append(f"| {c['cell'].replace('|', ' / ')} | {c['pool']} | {c['sampled']} | {c['families']} | "
                 f"{'yes, all taken' if c['shortfall'] else 'no'} |")
    return "\n".join(L)


# ============================================================
# Cache of the previous run
# ============================================================

_FLOAT_FIELDS = ("a", "mean_abs_err", "median_abs_err", "q90_abs_err", "q95_abs_err", "rms", "bias")
_INT_FIELDS = ("cost_A_apps_or_samples", "n_seeds")


def load_cache(path: Path) -> tuple[dict, dict]:
    """(known_slots, rows_by_label) from the previous metrics file; empty if absent."""
    if not path.exists():
        return {}, {}
    known: dict[str, int] = {}
    rows: dict[str, list[dict]] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            for f in _FLOAT_FIELDS:
                r[f] = float(r[f])
            for f in _INT_FIELDS:
                r[f] = int(r[f])
            lab = r["amplitude"]
            known[_akey(r["a"])] = int(lab[3:])
            rows.setdefault(lab, []).append(r)
    return known, rows


# ============================================================
# Step 3: simulation
# ============================================================


def _row(rec, fam, setting, cost, evaluation, n_seeds, met) -> dict:
    return {
        "amplitude": rec["label"], "kind": rec["kind"], "a": rec["a"],
        "family": fam, "setting": setting, "cost_A_apps_or_samples": int(cost),
        "evaluation": evaluation, "n_seeds": n_seeds,
        "mean_abs_err": met["mean_abs_err"], "median_abs_err": met["median_abs_err"],
        "q90_abs_err": met["q90_abs_err"], "q95_abs_err": met["q95_abs_err"],
        "rms": met["rms"], "bias": met["bias"],
    }


def _record_seed(slot: int) -> np.random.SeedSequence:
    """Same child as np.random.SeedSequence(V3_SEED).spawn(n)[slot] for any n > slot."""
    return np.random.SeedSequence(V3_SEED, spawn_key=(slot,))


def _simulate_record(job):
    """Worker: all record estimators for one amplitude record."""
    rec, slot = job
    seed = _record_seed(slot)
    t0 = time.perf_counter()
    a = rec["a"]
    rows: list[dict] = []
    for m in QAE_M:
        met = exact_qae_metrics(a, m)
        rows.append(_row(rec, "qae", f"m={m}", qae_cost(m), "exact pmf", 0, met))
    s_mlae, s_cls = seed.spawn(2)
    settings = [(J, S) for S in MLAE_S for J in MLAE_J]
    for (J, S), cs in zip(settings, s_mlae.spawn(len(settings))):
        H = mlae_counts(a, J, S, N_SEEDS, np.random.default_rng(cs))
        est = mlae_estimates(H, J, S)
        rows.append(_row(rec, f"mlae_S{S}", f"J={J},S={S}", mlae_cost(J, S), f"sampled {N_SEEDS}",
                         N_SEEDS, sample_metrics(est, a)))
    for N, cs in zip(CLASSICAL_N, s_cls.spawn(len(CLASSICAL_N))):
        est = classical_estimates(a, N, N_SEEDS, np.random.default_rng(cs))
        rows.append(_row(rec, "classical", f"N={N}", classical_cost(N), f"sampled {N_SEEDS}",
                         N_SEEDS, sample_metrics(est, a)))
    return rec["label"], rows, time.perf_counter() - t0


def _direct_job(pr: dict):
    """Worker: direct signed-mean classical estimator of D for one move."""
    t0 = time.perf_counter()
    ap, am = pr["a_plus"], pr["a_minus"]
    truth = ap - am
    children = np.random.SeedSequence(V3_SEED, spawn_key=(1, pr["row"])).spawn(len(CLASSICAL_N))
    rows = []
    for N, cs in zip(CLASSICAL_N, children):
        est = signed_mean_estimates(ap, am, N, N_SEEDS, np.random.default_rng(cs))
        met = sample_metrics(est, truth)
        rows.append({
            "move": pr["pid"], "group": pr["group"], "cls": pr["cls"], "mode": pr["mode"],
            "family": "classical_direct", "setting": f"N={N}", "cost_samples": int(N),
            "n_seeds": N_SEEDS, "truth_E_D": truth, "var_D": signed_mean_variance(ap, am),
            "mean_abs_err": met["mean_abs_err"], "median_abs_err": met["median_abs_err"],
            "q90_abs_err": met["q90_abs_err"], "q95_abs_err": met["q95_abs_err"],
            "rms": met["rms"], "bias": met["bias"],
        })
    return pr["pid"], rows, time.perf_counter() - t0


def _cached_rows(rec, cache_rows: dict) -> list[dict]:
    out = []
    for r in cache_rows[rec["label"]]:
        out.append(dict(r, kind=rec["kind"], a=rec["a"]))
    return out


def reuse_check(records, pairs, cache_rows: dict) -> float:
    """Re-simulate one cached amplitude with the current code and compare with the cache."""
    cached = [r for r in records if r["label"] in cache_rows]
    if not cached:
        return float("nan")
    rec = cached[0]
    _, rows, _ = _simulate_record((rec, rec["slot"]))
    old = {(r["family"], r["setting"]): r for r in cache_rows[rec["label"]]}
    worst = 0.0
    for r in rows:
        o = old[(r["family"], r["setting"])]
        for f in _FLOAT_FIELDS[1:]:
            worst = max(worst, abs(float(r[f]) - float(o[f])))
        if int(r["cost_A_apps_or_samples"]) != int(o["cost_A_apps_or_samples"]):
            raise RuntimeError("reuse check: cost mismatch")
    if worst > 0.0:
        raise RuntimeError(f"reuse check failed for {rec['label']}: max |diff| = {worst:.3e}")
    return worst


def simulate(records, pairs, cache_rows: dict):
    """Simulate new amplitudes and all direct classical moves. Returns (rows, direct_rows, n_new, times)."""
    new = [r for r in records if r["label"] not in cache_rows]
    rows: list[dict] = []
    for r in records:
        if r["label"] in cache_rows:
            rows.extend(_cached_rows(r, cache_rows))
    direct_rows: list[dict] = []
    t0 = time.perf_counter()
    done = 0
    total = len(new) + len(pairs)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for label, r, dt in ex.map(_simulate_record, [(rec, rec["slot"]) for rec in new], chunksize=1):
            rows.extend(r)
            done += 1
            if done % 25 == 0 or done == total:
                _log(f"  {done}/{total} jobs ({label}, {dt:.1f} s); elapsed {time.perf_counter() - t0:.0f} s")
        for pid, r, dt in ex.map(_direct_job, pairs, chunksize=4):
            direct_rows.extend(r)
            done += 1
            if done % 100 == 0 or done == total:
                _log(f"  {done}/{total} jobs ({pid}, {dt:.1f} s); elapsed {time.perf_counter() - t0:.0f} s")
    return rows, direct_rows, len(new), time.perf_counter() - t0


def exact_check() -> list[dict]:
    """Exact QAE mean |error| and q90 against a large sampled estimate."""
    rng = np.random.default_rng(V3_SEED + 1)
    out = []
    for a in (0.005, 0.03, 0.2):
        for m in (6, 10):
            ex = exact_qae_metrics(a, m)
            est = qae_estimates(a, m, EXACT_CHECK_N, rng)
            ae = np.abs(est - a)
            se = float(np.std(ae, ddof=1) / math.sqrt(EXACT_CHECK_N))
            out.append({
                "a": a, "m": m, "n_samples": EXACT_CHECK_N,
                "exact_mean_abs": ex["mean_abs_err"], "sampled_mean_abs": float(np.mean(ae)),
                "z_mean_abs": float((np.mean(ae) - ex["mean_abs_err"]) / se),
                "exact_q90": ex["q90_abs_err"],
                "sampled_q90": float(np.quantile(ae, 0.9, method="inverted_cdf")),
                "exact_rms": ex["rms"], "sampled_rms": float(np.sqrt(np.mean((est - a) ** 2))),
            })
    return out


# ============================================================
# Step 4: analysis
# ============================================================


def curves(rows):
    """{(record label, family): (cost array, q90 array, rms array, mean-abs array)}, cost-ordered."""
    acc: dict = {}
    for r in rows:
        acc.setdefault((r["amplitude"], r["family"]), []).append(
            (r["cost_A_apps_or_samples"], r["q90_abs_err"], r["rms"], r["mean_abs_err"]))
    res = {}
    for key, pts in acc.items():
        pts.sort(key=lambda t: t[0])
        arr = np.array(pts, float)
        res[key] = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    return res


def direct_curves(drows):
    """{move id: (cost, q90, rms, mean-abs)} for the direct signed-mean classical estimator."""
    acc: dict = {}
    for r in drows:
        acc.setdefault(r["move"], []).append(
            (r["cost_samples"], r["q90_abs_err"], r["rms"], r["mean_abs_err"]))
    res = {}
    for key, pts in acc.items():
        pts.sort(key=lambda t: t[0])
        arr = np.array(pts, float)
        res[key] = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    return res


def fitted_slope(cost: np.ndarray, val: np.ndarray) -> float:
    """Log-log slope over the last ASYMPTOTIC_LAST settings (cost order); NaN if any value is zero."""
    c = np.asarray(cost[-ASYMPTOTIC_LAST:], float)
    v = np.asarray(val[-ASYMPTOTIC_LAST:], float)
    if np.any(v <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(c), np.log(v), 1)[0])


def slope_rows(cv, records, pairs):
    """Slopes per amplitude record and family; each record carries the family groups of the moves that use it."""
    groups_of: dict[str, set] = {}
    for pr in pairs:
        for lab in [pr["sa"]] + [lab for lab, _ in pr["components"]]:
            groups_of.setdefault(lab, set()).add(pr["group"])
    rows = []
    for rec in records:
        for fam in RECORD_FAMILIES:
            c, q90, rms, mab = cv[(rec["label"], fam)]
            rows.append({
                "amplitude": rec["label"], "kind": rec["kind"], "a": rec["a"], "family": fam,
                "groups": "|".join(sorted(groups_of.get(rec["label"], set()))),
                "slope_q90": fitted_slope(c, q90), "slope_rms": fitted_slope(c, rms),
                "slope_mean_abs": fitted_slope(c, mab),
            })
    return rows


def ratio_rows(qv, pairs, families=RATIO_FAMILIES, targets=TARGETS):
    """Cost of the difference estimate over the stand-alone cost, at the 90 % quantile target.

    qv: {(record label, family): (cost array, q90 array)} for the record families,
        and {(move id, "classical_direct"): (cost array, q90 array)} for the direct signed-mean estimator.
    family "classical": direct signed-mean estimator of D, one estimate at target T_D.
    family "classical_2est": two-estimate protocol, one estimate per nonzero side at T_D / k
        (k = 1 one-sided, 2 signed).
    Quantum families: one estimate per nonzero side at T_D / k.
    Every row is reported with a status: ok, floor (a target is met at the smallest
    setting, so the cost is only an upper bound) or fail (target not reached).
    """
    rows = []
    for pr in pairs:
        a = pr["base"]
        comps = pr["components"]
        k = len(comps)
        a_sum = sum(amp for _, amp in comps)
        v_sa = a * (1.0 - a)
        v_list = [amp * (1.0 - amp) for _, amp in comps]
        ap, am = pr["a_plus"], pr["a_minus"]
        var_D = signed_mean_variance(ap, am)
        rho_var = math.sqrt(max(var_D, 0.0)) / math.sqrt(v_sa)
        for fam in families:
            if fam == "classical":
                sa_key, method = "classical", "classical"
            elif fam == "classical_2est":
                sa_key, method = "classical", "classical"
            else:
                sa_key, method = fam, "qae" if fam == "qae" else "mlae"
            for kind, val in targets:
                T_sa, T_D = target_scales(a, a_sum, kind, val)
                c_sa, f_sa = cost_at(*qv[(pr["sa"], sa_key)], T_sa)
                if fam == "classical":
                    c_d, f_d = cost_at(*qv[(pr["pid"], "classical_direct")], T_D)
                    c_comp, floors = [c_d], [f_sa, f_d]
                    pred = predicted_cost_ratio(method, v_sa, [var_D], T_sa, T_D)
                    n_est = 1
                else:
                    c_comp, floors = [], [f_sa]
                    for lab, _ in comps:
                        c, f = cost_at(*qv[(lab, sa_key if fam == "classical_2est" else fam)], T_D / k)
                        c_comp.append(c)
                        floors.append(f)
                    pred = predicted_cost_ratio(method, v_sa, v_list, T_sa, T_D)
                    n_est = k
                if not (math.isfinite(c_sa) and all(math.isfinite(c) for c in c_comp)):
                    status = "fail"
                elif any(floors):
                    status = "floor"
                else:
                    status = "ok"
                total = sum(c_comp) if all(math.isfinite(c) for c in c_comp) else float("nan")
                ratio = total / c_sa if status == "ok" else float("nan")
                rows.append({
                    "cell": pr["cell"], "cls": pr["cls"], "band": pr["band"], "group": pr["group"],
                    "mode": pr["mode"], "pid": pr["pid"], "ids": pr["ids"],
                    "base_p": a, "P_D1": ap, "P_Dm1": am, "a_sum": a_sum, "var_D": var_D,
                    "rho_q_var": rho_var, "rho_q_file": pr["rho_q"],
                    "family": fam, "n_estimates": n_est, "target_type": kind, "target_value": val,
                    "T_sa": T_sa, "T_D": T_D,
                    "cost_standalone": c_sa,
                    "cost_diff_components": ";".join(f"{c:.6g}" for c in c_comp),
                    "cost_diff_total": total,
                    "status": status,
                    "ratio": ratio, "pred": pred,
                    "ratio_over_pred": ratio / pred if status == "ok" and pred > 0 else float("nan"),
                })
    return rows


def _finite(v):
    return [x for x in v if x is not None and math.isfinite(x)]


def _med(v):
    v = _finite(v)
    return float(np.median(v)) if v else float("nan")


def _p(v, q):
    v = _finite(v)
    return float(np.quantile(v, q)) if v else float("nan")


def _split_cell(cell: str):
    cls, band, group = cell.split("|")
    return int(cls[3:]), band, group


def ratio_summary(rr):
    """Per (cell, family, target): counts and medians over the same ok rows."""
    out = []
    keys = sorted({(r["cell"], r["family"], r["target_type"], r["target_value"]) for r in rr},
                  key=lambda k: (_split_cell(k[0]), RATIO_FAMILIES.index(k[1]), k[2], -k[3]))
    for cell, fam, tt, tv in keys:
        g = [r for r in rr if r["cell"] == cell and r["family"] == fam
             and r["target_type"] == tt and r["target_value"] == tv]
        ok = [r for r in g if r["status"] == "ok"]
        rat = [r["ratio"] for r in ok]
        prd = [r["pred"] for r in ok]
        rop = [r["ratio_over_pred"] for r in ok]
        cls, band, group = _split_cell(cell)
        out.append({
            "cell": cell, "cls": cls, "band": band, "group": group,
            "family": fam, "target_type": tt, "target_value": tv,
            "mode_mix": f"{sum(r['mode'] == 'signed' for r in g)} signed / {sum(r['mode'] != 'signed' for r in g)} one-sided",
            "n_total": len(g), "n_ok": len(ok),
            "n_floor": sum(r["status"] == "floor" for r in g),
            "n_fail": sum(r["status"] == "fail" for r in g),
            "median_ratio": _med(rat), "p10_ratio": _p(rat, 0.1), "p90_ratio": _p(rat, 0.9),
            "median_pred": _med(prd), "median_ratio_over_pred": _med(rop),
            "p10_ratio_over_pred": _p(rop, 0.1), "p90_ratio_over_pred": _p(rop, 0.9),
        })
    return out


def group_summary(rr):
    """Per (group, family, target) over all classes and bands: counts and ratio medians over ok rows."""
    out = []
    for g in GROUPS:
        for fam in RATIO_FAMILIES:
            for tt, tv in TARGETS:
                sub = [r for r in rr if r["group"] == g and r["family"] == fam
                       and r["target_type"] == tt and r["target_value"] == tv]
                ok = [r for r in sub if r["status"] == "ok"]
                out.append({
                    "group": g, "family": fam, "target_type": tt, "target_value": tv,
                    "n_total": len(sub), "n_ok": len(ok),
                    "n_floor": sum(r["status"] == "floor" for r in sub),
                    "n_fail": sum(r["status"] == "fail" for r in sub),
                    "median_ratio": _med([r["ratio"] for r in ok]),
                    "p10_ratio": _p([r["ratio"] for r in ok], 0.1),
                    "p90_ratio": _p([r["ratio"] for r in ok], 0.9),
                })
    return out


def v2_carryover():
    """S = 1 and S = 8 (and S = 64) at the largest J of the v2 run, from its stored q90 errors (not rerun)."""
    import pandas as pd

    rms = pd.read_csv(V2_RMS, usecols=["a", "kind", "family", "setting", "q90_abs_err"])
    rows = []
    for fam in ("mlae_S1", "mlae_S8", "mlae_S64"):
        sub = rms[rms["family"] == fam]
        last = sub[sub["setting"].str.startswith("J=10,")].set_index(["a", "kind"])["q90_abs_err"]
        best = sub.groupby(["a", "kind"])["q90_abs_err"].min()
        rows.append({
            "family": fam, "n_amplitudes": int(last.size),
            "median_q90_at_J10": float(np.median(last)),
            "frac_J10_q90_le_1e-3": float(np.mean(last <= 1e-3)),
            "frac_any_J_q90_le_1e-3": float(np.mean(best <= 1e-3)),
            "frac_any_J_q90_le_1e-4": float(np.mean(best <= 1e-4)),
        })
    return rows


def make_figure(cv, show, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "classical": ("grey", "o", "classical (N samples)"),
        "qae": ("tab:blue", "s", "canonical QAE (exact pmf)"),
        "mlae_S64": ("tab:red", "D", "MLAE S = 64"),
    }
    fig, axes = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, title) in zip(axes, show):
        for fam, (col, mk, name) in styles.items():
            c, q90, _, _ = cv[(lab, fam)]
            m = q90 > 0
            ax.loglog(c[m], q90[m], marker=mk, color=col, ms=4, lw=1.2, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("A-applications (samples for classical)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("90 % quantile of |a_hat - a|")
    axes[-1].legend(fontsize=8)
    fig.suptitle("QUASAR M3b v3 (review fixes): 90 % quantile error versus cost (4000 seeds; QAE exact)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return "nan"
        return f"{float(x):.{nd}g}"
    return str(x)


def _pop_lines(pop):
    L = ["| group | class | moves in M3a | zero difference | side below 200 hits | eligible | share with P(D != 0) < 1e-3 | eligible: min P(D != 0) | eligible: median P(D != 0) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in pop:
        L.append(f"| {r['group']} | {r['cls']} | {r['n_moves']} | {r['n_zero']} | {r['n_under']} | {r['n_eligible']} | "
                 f"{r['share_pnz_below_1e-3']:.3f} | {fmt(r['eligible_pnz_min'])} | {fmt(r['eligible_pnz_median'])} |")
    return L


CLAIMS_TEXT: list[str] = [
    "Read with the table above. Every cost ratio divides a difference estimate by a stand-alone estimate of the same paradigm; "
    "ratios are not quantum-versus-classical comparisons. The sample covers P(D != 0) >= 1e-3 only.",
    "",
    "**Supported (eligible sample, medians over ok rows, n_ok in the table):**",
    "",
    "- CRN lowers the quantum cost of the difference at absolute 1e-3 in all three groups. QAE: 0.65 (CON/SCA, n = 180), "
    "0.46 (CMT, n = 153), 0.29 (city, n = 112). MLAE S = 64: 0.60, 0.45, 0.38. The same direction holds at 1e-4 for QAE "
    "(0.61, 0.45, 0.32).",
    "- At absolute 1e-4 MLAE S = 32 fails for most moves (111, 102 and 77 of the 180, 153 and 112 moves; Section 5a), so its medians "
    "are not quoted. MLAE S = 64 fails for 24, 18 and 14 moves.",
    "- Relative targets (T_D = r P(D != 0), so the target shrinks with the difference): the difference costs more than the stand-alone "
    "estimate in all three groups. QAE 3.2 to 5.5, MLAE S = 64 1.4 to 4.1, direct classical 8 to 20 (r = 0.1 and 0.01). "
    "Supported for the sampled moves only.",
    "- Scaling: the quantum slope of the 90 % quantile is -1 and the classical slope -0.5, in every group (Section 3, group table).",
    "- The sqrt(v) law for the quantum difference cost holds on class 1: ratio / rho_q for MLAE S = 64 and 128 lies between 0.94 "
    "and 1.11 across bands and absolute targets. QAE has alignment scatter (0.72 to 1.29). Classes 2 and 3 are not tested against rho_q.",
    "- The direct classical estimator agrees with its Var(D) / T^2 prediction (ratio / rho_c 0.97 to 1.04). This is a consistency check, "
    "not evidence about the quantum law.",
    "",
    "**Not supported by M3b:**",
    "",
    "- The regime where P(D != 0) is tiny (below 1e-3). Such moves are zero-difference or have a side below 200 hits and are excluded. "
    "In the M3a file they are 38 to 95 % of the classes 2 and 3 moves of every group (table above) and 3 to 7 % of class 1. "
    "M3b makes no claim about the floor regime or about the Var(D) = 0 share.",
    "- City classes 2 and 3. Eligible city moves are 974 (class 2) and 1076 (class 3), with median P(D != 0) of 0.0023. The class 2 "
    "low and decision cells are empty, the class 3 decision cell has 12 moves, and the class 2 and 3 high cells have 20 each. "
    "Only the high-band city numbers for classes 2 and 3 have usable n.",
    "- Any claim from the cells with n_ok below 10 (listed in Section 5b, e.g. MLAE S = 32 at 1e-4 on cls1/high/CMT, "
    "cls1/high/city, cls2/high/CMT, cls2/high/city, with 0 ok rows). Their medians are shown only for completeness.",
    "- A quantum-versus-classical cost comparison, which M3b does not compute; the classical columns here measure the classical "
    "difference against the classical stand-alone, not against the quantum estimator.",
]


def write_summary(path, meta, val_rows, info, pop, pairs, records, slopes, rsum, gsum, carry, exact_rows) -> None:
    L: list[str] = []
    A = L.append
    A("# QUASAR M3b v3 (review fixes): estimator behaviour on the validated rotation model")
    A("")
    A(f"Produced by scripts/run_quasar_m3b.py. Root seed SeedSequence({V3_SEED}); {N_SEEDS} independent "
      "seeds per sampled setting; canonical QAE is computed from its exact outcome pmf. Each amplitude's seed is fixed by its "
      "slot (amp<slot>), so amplitudes simulated in the previous run were reused exactly.")
    A("")
    A("## 1. Circuit validation (reused from v2, not rerun)")
    A("")
    A("The four real oracles of v2 (three peak, one difference; 19 to 22 qubits) were simulated once with an exact statevector "
      "and compared with sin^2((2k+1) theta), k = 0..6. The rows are copied unchanged to m3b_v3_validation.csv.")
    A("")
    mx = max(float(r["abs_diff"]) for r in val_rows)
    A(f"Maximum |p_circuit - p_model| over the four oracles and k = 0..6: **{mx:.2e}**.")
    A("")
    A("## 2. Population, sample and exclusions")
    A("")
    A(f"Source: {MOVES_GZ.name}, {info['n_moves']} moves, 2e5 scenarios each (hits = amplitude x 2e5). "
      f"Eligible: every nonzero amplitude has at least {HITS_MIN} hits, and the move is not zero-difference. "
      f"Zero-difference moves: {info['moves_zero_difference']} excluded; moves with a nonzero amplitude below {HITS_MIN} hits: "
      f"{info['moves_under_threshold']} excluded; eligible: {info['moves_eligible']}.")
    A("")
    A("Family groups: CON/SCA (random-generator instances, families CON and SCA), CMT (the CMT instances), city (the OSM instances).")
    A("")
    A("Population by group and class (the whole M3a file; the last three columns show what the eligible sample can and cannot speak to):")
    A("")
    L.extend(_pop_lines(pop))
    A("")
    A("Stratified sample: class x stand-alone p' band x family group, up to 20 moves per cell (drawn at random from the eligible pool). "
      "Cells with fewer eligible moves than 20 contribute all of them and are marked.")
    A("")
    L.append(composition_text(info))
    A("")
    n_one = sum(1 for p in pairs if p["mode"] != "signed")
    n_short = sum(1 for c in info["cells"] if c["shortfall"])
    A(f"Sampled moves: {len(pairs)} ({n_one} one-sided, {len(pairs) - n_one} signed); cells with a shortfall: {n_short} of "
      f"{len(info['cells'])}. Unique amplitudes in the sample: {len(records)}, of which {meta['amplitudes reused']} reused from the "
      f"previous run and {meta['amplitudes simulated']} simulated now.")
    A("")
    A("## 3. Error metrics and fitted slopes")
    A("")
    A("Primary metric: the 90 % quantile of |a_hat - a|. Slopes are log-log fits over the last six settings of each family in "
      "cost order (QAE m = 17..22; MLAE J = 5..10 for each S; classical N = 2^21..2^26). Medians and [min, max] over the "
      "amplitude records of each kind (all groups).")
    A("")
    A("| kind | family | slope, 90 % quantile | slope, mean abs error | slope, RMS (secondary) |")
    A("|---|---|---|---|---|")
    for kind in ("standalone", "difference+", "difference-"):
        for fam in RECORD_FAMILIES:
            cells = []
            for key in ("slope_q90", "slope_mean_abs", "slope_rms"):
                v = [x[key] for x in slopes if x["kind"] == kind and x["family"] == fam]
                v = _finite(v)
                cells.append(f"{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}] (n={len(v)})" if v else "nan")
            A(f"| {kind} | {fam} | {cells[0]} | {cells[1]} | {cells[2]} |")
    A("")
    A("Slopes by family group (90 % quantile; records used by moves of that group, median [min, max], n records). "
      "A record used by moves of two groups counts in both.")
    A("")
    A("| group | kind | qae | mlae_S64 | classical |")
    A("|---|---|---|---|---|")
    for g in GROUPS:
        for kind in ("standalone", "difference"):
            cells = []
            for fam in ("qae", "mlae_S64", "classical"):
                v = [x["slope_q90"] for x in slopes if x["family"] == fam and g in x["groups"].split("|")
                     and (x["kind"] == "standalone" if kind == "standalone" else x["kind"].startswith("difference"))]
                v = _finite(v)
                cells.append(f"{np.median(v):.2f} [{min(v):.2f}, {max(v):.2f}] (n={len(v)})" if v else "-")
            A(f"| {g} | {kind} | " + " | ".join(cells) + " |")
    A("")
    A("**Why the QAE 90 % quantile slope is read as a median over records.** For large N the exact QAE error is "
      "|a_hat - a| = 2 pi sqrt(a(1-a)) |delta| / N (1 + o(1)), where delta is the offset of the outcome from N phi, and the outcome law of delta "
      "depends only on f_m = frac(2^m theta / pi): q90 = 2 pi sqrt(a(1-a)) g(f_m) / N with g bounded. At fixed a the factor g(f_m) "
      "moves with m, so the slope of one amplitude over six values of m is alignment noise. The population slope is -1; the median "
      "over records is the statistic to read, and the per-record range is shown to make that noise visible.")
    A("")
    A("**Why the RMS is a secondary column.** For canonical QAE the tails are P(|delta| > x) of order 1/(pi^2 x), so "
      "P(|a_hat - a| > e) is of order 1/(N e) for 1/N << e << 1. The mean absolute error then behaves as (log N)/N (up to the alignment "
      "factor). The mean square, E(a_hat - a)^2 = integral of 2 e P(|a_hat - a| > e) de, is of order 1/N with a constant c(a) = "
      "2 sqrt(a(1-a))/pi: the integrand is about 2 c(a)/N for every e in (1/N, O(1)), so the mass is spread over error levels and "
      "carried by outcomes of probability about 1/N. The RMS therefore scales as N^(-1/2) and is dominated by these heavy tails. A "
      "sample of 4000 seeds almost never contains them, so the sampled RMS of QAE under-estimates it; the exact RMS is reported as a "
      "secondary column for that reason.")
    A("")
    A("## 4. Exact QAE metrics versus a large sampled estimate")
    A("")
    A(f"Exact mean absolute error and q90 from the pmf, against {EXACT_CHECK_N} sampled QAE runs (z = (sampled - exact) / standard error):")
    A("")
    A("| a | m | exact mean abs | sampled mean abs | z | exact q90 | sampled q90 | exact RMS | sampled RMS |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in exact_rows:
        A(f"| {r['a']:g} | {r['m']} | {fmt(r['exact_mean_abs'], 4)} | {fmt(r['sampled_mean_abs'], 4)} | {r['z_mean_abs']:.2f} | "
          f"{fmt(r['exact_q90'], 4)} | {fmt(r['sampled_q90'], 4)} | {fmt(r['exact_rms'], 4)} | {fmt(r['sampled_rms'], 4)} |")
    A("")
    A("## 5. Cost of the difference estimate relative to the stand-alone estimate")
    A("")
    A("Targets on the 90 % quantile of |a_hat - a|: absolute 1e-3 and 1e-4 (T_sa = T_D = eps); relative 0.1 and 0.01 "
      "(T_sa = r p', T_D = r (a+ + a-), with a+ + a- = P(D != 0)). Cost ratio = cost(difference) / cost(stand-alone).")
    A("")
    A("Families of the difference:")
    A("")
    A("- qae, mlae_S32/64/128: one estimate of the nonzero side for one-sided moves (target T_D); two estimates of P(D = +1) and "
      "P(D = -1) at T_D / 2 each for signed moves, cost summed. Prediction: sqrt(v_D)/T_D over sqrt(v_sa)/T_sa for one-sided moves; "
      "2 (sqrt v+ + sqrt v-)/T_D over sqrt(v_sa)/T_sa for signed moves.")
    A("- **classical** (primary classical column): the direct signed-mean estimator of D, one sample mean of N draws of D in {-1, 0, +1} "
      "(multinomial counts), target T_D on the 90 % quantile. No two-estimate protocol. Prediction Var(D)/T_D^2 over v_sa/T_sa^2, "
      "Var(D) = a+ + a- - (a+ - a-)^2; for one-sided moves Var(D) = a+(1 - a+).")
    A("- **classical_2est** (secondary): the two-estimate protocol of the previous v3 run (one Monte Carlo estimate per nonzero side, each at T_D / k). "
      "For one-sided moves it equals the single estimate and agrees with classical in law; for signed moves it costs about 4x the direct "
      "estimator at equal variance and is reported only to show that protocol effect.")
    A("- Status: ok (cost measured), floor (a target is met at the smallest setting, so the cost is an upper bound and the row is not used), "
      "fail (target not reached within the grid). Medians and the ratio/prediction spread use the ok rows only, and the same rows for all three columns.")
    A("- QAE cost-to-target inherits the alignment factor g(f_m) of Section 3 (cost resolved in factors of 2 over m), so QAE ratios carry "
      "multiplicative scatter.")
    A("")
    A("### 5a. Summary by family group (all classes and bands; ok rows; median ratio [p10, p90] and n_ok / n_floor / n_fail)")
    A("")
    A("| group | target | " + " | ".join(RATIO_FAMILIES) + " |")
    A("|---|---|" + "---|" * len(RATIO_FAMILIES))
    for g in GROUPS:
        for tt, tv in TARGETS:
            lab = f"{tt} {tv:g}"
            cols = []
            for fam in RATIO_FAMILIES:
                x = [r for r in gsum if r["group"] == g and r["family"] == fam and r["target_type"] == tt and r["target_value"] == tv][0]
                cols.append(f"{fmt(x['median_ratio'])} [{fmt(x['p10_ratio'])}, {fmt(x['p90_ratio'])}]; "
                            f"{x['n_ok']} / {x['n_floor']} / {x['n_fail']}")
            A(f"| {g} | {lab} | " + " | ".join(cols) + " |")
    A("")
    A("### 5b. By cell (class x p' band x family group)")
    A("")
    A("Each cell: median ratio / median prediction; median (ratio / prediction) [p10, p90]; n_ok / n_floor / n_fail. "
      "Cells with no eligible move are omitted; cells with n_ok < 10 must not be quoted.")
    A("")
    cells_sorted = sorted({r["cell"] for r in rsum}, key=lambda c: (_split_cell(c)[0], BAND_NAMES.index(_split_cell(c)[1]),
                                                                    GROUPS.index(_split_cell(c)[2])))
    for tt, tv in TARGETS:
        lab = f"absolute {tv:g}" if tt == "abs" else f"relative {tv:g} (x amplitude)"
        A(f"#### Target: {lab}")
        A("")
        A("| cell | " + " | ".join(RATIO_FAMILIES) + " |")
        A("|---|" + "---|" * len(RATIO_FAMILIES))
        for cell in cells_sorted:
            cols = []
            for fam in RATIO_FAMILIES:
                x = [r for r in rsum if r["cell"] == cell and r["family"] == fam
                     and r["target_type"] == tt and r["target_value"] == tv]
                if not x:
                    cols.append("-")
                    continue
                r = x[0]
                cols.append(
                    f"{fmt(r['median_ratio'])} / {fmt(r['median_pred'])}; "
                    f"{fmt(r['median_ratio_over_pred'])} [{fmt(r['p10_ratio_over_pred'])}, {fmt(r['p90_ratio_over_pred'])}]; "
                    f"{r['n_ok']} / {r['n_floor']} / {r['n_fail']}")
            A(f"| {cell.replace('|', ' / ')} | " + " | ".join(cols) + " |")
        A("")
    A("Every row is in m3b_v3_ratio.csv with its status; per-cell counts in m3b_v3_ratio_summary.csv.")
    A("")
    A("### 5c. Class 1 against the rho prediction (absolute targets)")
    A("")
    A("For class 1 (insertion; P(D = -1) = 0) the one-sided prediction is rho_q = sqrt(Var D) / sqrt(p'(1-p')) for the quantum "
      "estimators, and rho_c = rho_q^2 for the classical estimators. The lines below divide the median ratio by the family's own "
      "prediction: ratio / rho_q for the quantum families, ratio / rho_c (rho_c = rho_q^2) for the classical families. Range of the "
      "per-cell medians over the three p' bands.")
    A("")
    for fam in RATIO_FAMILIES:
        pname = "rho_c" if fam.startswith("classical") else "rho_q"
        parts = []
        for tv in ABS_TARGETS:
            meds = [r["median_ratio_over_pred"] for r in rsum if r["cls"] == 1
                    and r["family"] == fam and r["target_type"] == "abs" and r["target_value"] == tv]
            fin = _finite(meds)
            parts.append(f"target {tv:g}: {min(fin):.2f} to {max(fin):.2f}" if fin else f"target {tv:g}: none")
        A(f"- {fam} (ratio / {pname}): " + "; ".join(parts))
    A("")
    A("**Consistency check, not a test, for the classical columns.** For a mean of i.i.d. draws the cost to reach a target scales as "
      "Var/T^2 (CLT, 90 % quantile); the classical prediction is that law, so the classical ratio / prediction is a consistency check "
      "that only the integer grid of N (factor-2 steps, log-log interpolation) and the discreteness of the 90 % quantile can move. It is "
      "tautological for a binomial and says nothing about the quantum law. Only the quantum columns test the sqrt(v) law.")
    A("")
    A("## 6. Claims by family group")
    A("")
    A("Cost ratio = difference / stand-alone, medians over ok rows; n_ok in brackets.")
    A("")
    L.extend(_claims_lines(pop, gsum, info))
    A("")
    L.extend(CLAIMS_TEXT)
    A("")
    A("## 7. MLAE with S = 1 and S = 8: not rerun (from v2)")
    A("")
    s1, s8 = carry[0], carry[1]
    A("v2 included S = 1 and S = 8. v3 keeps S in {32, 64, 128} only and does not rerun S = 1 or S = 8. The table is computed from "
      "m3b_v2_rms.csv (2000 seeds, the v2 run) over all v2 amplitude records (stand-alone and difference). "
      f"S = 1 does not converge: its median 90 % error at J = 10 is {s1['median_q90_at_J10']:.2g}, and only "
      f"{s1['frac_any_J_q90_le_1e-3']:.0%} of records reach 1e-3 at any J and {s1['frac_any_J_q90_le_1e-4']:.0%} reach 1e-4. "
      f"S = 8 is too slow for the 1e-4 target: {s8['frac_J10_q90_le_1e-3']:.0%} of records reach 1e-3 at J = 10 but only "
      f"{s8['frac_any_J_q90_le_1e-4']:.0%} reach 1e-4 at any J (S = 64: {carry[2]['frac_any_J_q90_le_1e-4']:.0%}). "
      "Neither S = 1 nor S = 8 is used in the cost comparison.")
    A("")
    A("| family (v2) | records | median q90 at J = 10 | fraction with q90 <= 1e-3 at J = 10 | fraction with q90 <= 1e-3 at some J | fraction with q90 <= 1e-4 at some J |")
    A("|---|---|---|---|---|---|")
    for c in carry:
        A(f"| {c['family']} | {c['n_amplitudes']} | {fmt(c['median_q90_at_J10'])} | {c['frac_J10_q90_le_1e-3']:.2f} | "
          f"{c['frac_any_J_q90_le_1e-3']:.2f} | {c['frac_any_J_q90_le_1e-4']:.2f} |")
    A("")
    A("## 8. Run")
    A("")
    for k, v in meta.items():
        if k in ("amplitudes reused", "amplitudes simulated"):
            continue
        A(f"- {k}: {v}")
    A("")
    A("## 9. Deviations from the brief")
    A("")
    A("- Stratification: class x p' band x family group (27 cells), up to 20 moves per cell (the previous v3 run had 9 cells of 30). "
      "Cells with fewer eligible moves than 20 contribute all of them; the composition is in Section 2.")
    A("- Classical column: the direct signed-mean estimator of D (one estimate, cost z^2 Var(D)/eps^2 in its simulated form). "
      "The two-estimate protocol of the previous v3 run is kept as the secondary column classical_2est.")
    A("- Relabelling: the classical ratio of class 1 is divided by rho_c = rho_q^2 and labelled so; the classical ratio / prediction "
      "is a consistency check (Section 5c).")
    A("- Amplitude reuse: each amplitude's seed is fixed by its slot (amp<slot>). The previous run's metrics file was read as a cache "
      "and each reused amplitude was re-simulated once before the write, and matched the cached rows exactly (Section 8). "
      "Only amplitudes absent from the cache were simulated, with the same N_SEEDS = 4000.")
    A("- Labels of amplitudes are now their slots (amp<slot>), not their order in the sample; the previous run used the order, so labels differ for some amplitudes.")
    A("- The 200-hit rule and the exclusion of zero-difference moves are unchanged. Moves with P(D != 0) < 1e-3 are not in the sample, "
      "so the regime where P(D != 0) is tiny is not covered by M3b (Section 6).")
    A("- Relative targets were not specified in the brief beyond the example. The definitions are T_sa = r p' and T_D = r (a+ + a-).")
    A("- Grids were not reduced: QAE m = 3..22, MLAE J = 1..10 with S in {32, 64, 128}, classical N = 2^4..2^26. Sampled estimators use "
      f"{N_SEEDS} seeds; QAE is exact and has no seeds.")
    A("- Step 1 was reused from v2 and not rerun, as the brief allows.")
    A("")
    path.write_text("\n".join(L) + "\n")


def _claims_lines(pop, gsum, info) -> list[str]:
    """Claims by family group: the numbers behind each claim (interpretation is written in the summary text)."""
    def g_rows(g, fam, tt, tv):
        return [r for r in gsum if r["group"] == g and r["family"] == fam
                and r["target_type"] == tt and r["target_value"] == tv][0]

    L = ["| group | sampled moves | population P(D != 0) < 1e-3 (classes 1 / 2 / 3) | target | qae median ratio (n_ok) | mlae_S64 median ratio (n_ok) | classical direct median ratio (n_ok) | classical_2est median ratio (n_ok) |",
         "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        ns = sum(c["sampled"] for c in info["cells"] if c["group"] == g)
        tiny = " / ".join(f"{next(r for r in pop if r['group'] == g and r['cls'] == c)['share_pnz_below_1e-3']:.2f}"
                          for c in CLASSES)
        for tt, tv in TARGETS:
            cols = []
            for fam in ("qae", "mlae_S64", "classical", "classical_2est"):
                x = g_rows(g, fam, tt, tv)
                cols.append(f"{fmt(x['median_ratio'])} ({x['n_ok']})")
            L.append(f"| {g} | {ns} | {tiny} | {tt} {tv:g} | " + " | ".join(cols) + " |")
    return L


def main() -> None:
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    _log("Step 1: circuit validation (v2 output reused)")
    val_rows = reuse_validation()
    write_csv(OUT / "m3b_v3_validation.csv", val_rows)

    _log("Step 2: population, sample and cache")
    t1 = time.perf_counter()
    df = load_moves()
    pop, d = population_table(df)
    known, cache_rows = load_cache(V3_METRICS)
    info, records, pairs = build_pairs(d, known_slots=known)
    n_reuse = sum(1 for r in records if r["label"] in cache_rows)
    _log(f"  {info['moves_eligible']} eligible moves, {len(pairs)} sampled, {len(records)} unique amplitudes; "
         f"{n_reuse} in the cache, {len(records) - n_reuse} to simulate ({time.perf_counter() - t1:.1f} s)")
    _log("  composition:")
    for c in info["cells"]:
        _log(f"    {c['cell']:<22} pool {c['pool']:>7}  sampled {c['sampled']:>3}  {c['families']}")
    write_csv(OUT / "m3b_v3_sampled_moves.csv", [
        {k: v for k, v in p.items() if k != "components"} | {
            "components": ";".join(f"{lab}:{amp:.6g}" for lab, amp in p["components"])} for p in pairs])
    write_csv(OUT / "m3b_v3_population.csv", pop)

    _log("Step 3: exact QAE check")
    exact_rows = exact_check()
    write_csv(OUT / "m3b_v3_exact_check.csv", exact_rows)

    _log("Step 4: simulation (QAE exact, MLAE and classical 4000 seeds; direct signed-mean classical per move)")
    if n_reuse:
        worst = reuse_check(records, pairs, cache_rows)
        _log(f"  reuse check: one cached amplitude re-simulated, max |diff| vs cache = {worst:.1e}")
    rows, direct_rows, n_new, t_sim = simulate(records, pairs, cache_rows)
    rows.sort(key=lambda r: (r["amplitude"], RECORD_FAMILIES.index(r["family"])))
    write_csv(OUT / "m3b_v3_metrics.csv", rows)
    write_csv(OUT / "m3b_v3_direct_classical.csv", direct_rows)
    _log(f"  simulation: {len(rows)} record rows, {len(direct_rows)} direct rows, {n_new} new amplitudes in {t_sim:.1f} s")

    _log("Step 5: analysis")
    cv = curves(rows)
    dc = direct_curves(direct_rows)
    slopes = slope_rows(cv, records, pairs)
    write_csv(OUT / "m3b_v3_slopes.csv", slopes)
    qv = {k: (v[0], v[1]) for k, v in cv.items()}
    qv.update({(k, "classical_direct"): (v[0], v[1]) for k, v in dc.items()})
    rr = ratio_rows(qv, pairs)
    write_csv(OUT / "m3b_v3_ratio.csv", rr)
    rsum = ratio_summary(rr)
    write_csv(OUT / "m3b_v3_ratio_summary.csv", rsum)
    gsum = group_summary(rr)
    write_csv(OUT / "m3b_v3_group_summary.csv", gsum)
    carry = v2_carryover()
    write_csv(OUT / "m3b_v3_v2_carryover.csv", carry)

    sa_recs = {p["sa"]: p["base"] for p in pairs}
    show = []
    for t in (0.005, 0.03, 0.2):
        lab = min(sa_recs, key=lambda k: abs(math.log(sa_recs[k]) - math.log(t)))
        show.append((lab, f"stand-alone a = {sa_recs[lab]:.3g}"))
    make_figure(cv, show, OUT / "m3b_v3_error.png")

    meta = {
        "validation": "reused from v2 (no circuits rerun)",
        "amplitudes reused": n_reuse,
        "amplitudes simulated": n_new,
        "simulation time": f"{t_sim:.0f} s with {N_WORKERS} workers",
        "total time": f"{time.perf_counter() - t_all:.0f} s",
        "QAE": f"exact pmf, m = {QAE_M[0]}..{QAE_M[-1]}",
        "MLAE": f"J = {MLAE_J[0]}..{MLAE_J[-1]}, S in {list(MLAE_S)}, {N_SEEDS} seeds, grid 20001 points + golden-section",
        "classical": f"N = 2^{int(math.log2(CLASSICAL_N[0]))}..2^{int(math.log2(CLASSICAL_N[-1]))}, {N_SEEDS} seeds",
        "direct signed-mean classical": f"per move, same N grid and {N_SEEDS} seeds (multinomial draws of D)",
        "seeds": f"root SeedSequence({V3_SEED}); child per amplitude slot, per move (spawn key (1, row))",
        "python": sys.version.split()[0],
    }
    write_summary(OUT / "m3b_v3_summary.md", meta, val_rows, info, pop, pairs, records, slopes, rsum, gsum, carry, exact_rows)
    _log(f"done in {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
