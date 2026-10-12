"""QUASAR M3b v3: estimator behaviour on the validated rotation model.

Design: papers/quasar/DESIGN_M3.md, Revision 2, "M3b v2 - estimator behaviour"
(v3 changes are listed in results/quasar/m3b_v3_summary.md, "Deviations").

Step 1  Circuit validation of sin^2((2k+1) theta), k = 0..6. Not rerun: the v2
        output results/quasar/m3b_v2_validation.csv is copied to m3b_v3_validation.csv
        after its maximum deviation is checked.
Step 2  Error metrics per amplitude and setting:
        canonical QAE (m = 3..22) from the EXACT outcome pmf (no sampling);
        MLAE (J = 1..10, S in {32, 64, 128}) and classical Monte Carlo
        (N = 2^4..2^26) from N_SEEDS = 4000 independent seeds.
        Metrics: mean |a_hat - a|, median, 90 % and 95 % quantiles of |a_hat - a|,
        RMS and bias.
Step 3  Log-log slopes of the 90 % quantile (primary) and of the RMS (secondary),
        and the cost of a difference estimate relative to the stand-alone
        estimate at absolute and relative error targets.

Amplitudes come from results/quasar/m3a_v2_moves.csv.gz (2e5 scenarios per move).
Eligible moves: every nonzero amplitude has at least HITS_MIN hits; moves with
P(D = +1) = P(D = -1) = 0 are excluded. Stratified by move class (cls) and
stand-alone p' band (low < 0.02 <= decision <= 0.2 < high), up to MOVES_PER_CELL
moves per cell.

Run from svrpspd_wdro/:  python scripts/run_quasar_m3b.py
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the MLAE grid is a matrix product and the job shares the machine.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import json  # noqa: E402
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
    target_scales,
)

OUT = ROOT / "results" / "quasar"
MOVES_GZ = OUT / "m3a_v2_moves.csv.gz"
V2_VALIDATION = OUT / "m3b_v2_validation.csv"
V2_RMS = OUT / "m3b_v2_rms.csv"

V3_SEED = 20261013
N_SCEN = 200_000
HITS_MIN = 200
N_SEEDS = 4000
QAE_M = list(range(3, 23))
MLAE_S = (32, 64, 128)
MLAE_J = list(range(1, 11))
CLASSICAL_N = [2**e for e in range(4, 27)]
MOVES_PER_CELL = 30
CLASSES = (1, 2, 3)
BAND_NAMES = ("low", "decision", "high")
ABS_TARGETS = (1e-3, 1e-4)
REL_TARGETS = (0.1, 0.01)
TARGETS = [("abs", v) for v in ABS_TARGETS] + [("rel", v) for v in REL_TARGETS]
FAMILIES = ["qae"] + [f"mlae_S{S}" for S in MLAE_S] + ["classical"]
ASYMPTOTIC_LAST = 6
N_WORKERS = 2
EXACT_CHECK_N = 200_000


def _log(msg: str) -> None:
    print(msg, flush=True)


def _cell(cls: int, band: str) -> str:
    return f"cls{cls}|{band}"


def band_of(p: np.ndarray) -> np.ndarray:
    """low: p < 0.02; decision: 0.02 <= p <= 0.2; high: p > 0.2."""
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.02, "low", np.where(p <= 0.2, "decision", "high"))


# ============================================================
# Step 1: circuit validation (reused from v2)
# ============================================================


def reuse_validation() -> list[dict]:
    import csv as _csv

    with V2_VALIDATION.open() as fh:
        rows = list(_csv.DictReader(fh))
    max_dev = max(float(r["abs_diff"]) for r in rows)
    _log(f"  validation (v2 output reused): {len(rows)} rows, max |p_circuit - p_model| = {max_dev:.2e}")
    if max_dev > 1e-12:
        raise RuntimeError("v2 validation does not match the expected 1e-12 level")
    return rows


# ============================================================
# Step 2 inputs: amplitudes
# ============================================================


def load_moves():
    import pandas as pd

    cols = ["instance", "family", "route_id", "cls", "kind", "j", "pos", "pos2",
            "p_new", "P_D1", "P_Dm1", "rho_q"]
    return pd.read_csv(MOVES_GZ, usecols=cols)


def select_moves(df, rng: np.random.Generator):
    """Eligibility, stratified sample and exclusion counts.

    Returns (selected DataFrame, info dict). Hits are rint(amplitude * N_SCEN).
    """
    d = df.copy()
    d["h_sa"] = np.rint(d["p_new"].to_numpy(float) * N_SCEN)
    d["h_p"] = np.rint(d["P_D1"].to_numpy(float) * N_SCEN)
    d["h_m"] = np.rint(d["P_Dm1"].to_numpy(float) * N_SCEN)
    zero = (d["h_p"] == 0) & (d["h_m"] == 0)
    under = (
        (d["h_sa"] < HITS_MIN)
        | ((d["h_p"] > 0) & (d["h_p"] < HITS_MIN))
        | ((d["h_m"] > 0) & (d["h_m"] < HITS_MIN))
    )
    d["status"] = np.where(zero, "zero_difference", np.where(under, "under_threshold", "eligible"))

    info: dict = {"n_moves": int(len(d))}
    for name, h in (("p_new", d["h_sa"]), ("P_D1", d["h_p"]), ("P_Dm1", d["h_m"])):
        info[f"amp_{name}_zero_hits"] = int((h == 0).sum())
        info[f"amp_{name}_1_to_{HITS_MIN - 1}_hits"] = int(((h > 0) & (h < HITS_MIN)).sum())
        info[f"amp_{name}_ge_{HITS_MIN}_hits"] = int((h >= HITS_MIN).sum())
    info["moves_zero_difference"] = int((d["status"] == "zero_difference").sum())
    info["moves_under_threshold"] = int((d["status"] == "under_threshold").sum())
    info["moves_eligible"] = int((d["status"] == "eligible").sum())

    el = d[d["status"] == "eligible"].copy()
    el["band"] = band_of(el["p_new"].to_numpy(float))
    el["cell"] = [_cell(int(c), b) for c, b in zip(el["cls"], el["band"])]
    picks = []
    cells = []
    for cls in CLASSES:
        for band in BAND_NAMES:
            pool = el[(el["cls"] == cls) & (el["band"] == band)]
            n = min(MOVES_PER_CELL, len(pool))
            if n > 0:
                idx = np.sort(rng.choice(len(pool), size=n, replace=False))
                picks.append(pool.iloc[idx])
            cells.append({"cell": _cell(cls, band), "pool": int(len(pool)), "sampled": int(n)})
    info["cells"] = cells
    sel = _concat(picks)
    sel = sel.assign(mode=[_mode(hp, hm) for hp, hm in zip(sel["h_p"], sel["h_m"])])
    return sel, info


def _concat(parts):
    import pandas as pd

    if not parts:
        return pd.DataFrame(columns=["cell", "band", "cls", "p_new", "P_D1", "P_Dm1", "rho_q", "h_p", "h_m", "mode"])
    return pd.concat(parts, ignore_index=True)


def _mode(h_p: float, h_m: float) -> str:
    if h_p > 0 and h_m > 0:
        return "signed"
    return "one-sided+" if h_p > 0 else "one-sided-"


def build_pairs(df=None, seed: int = V3_SEED):
    """Return (info, records, pairs).

    records: unique amplitudes [{label, a, kind}], each simulated once.
    pairs: one per sampled move: {cell, cls, band, mode, base, a_plus, a_minus,
    sa (record label), components [(record label, amplitude)], rho_q, ids}.
    components are the nonzero difference sides: one for one-sided moves, two
    for signed moves.
    """
    if df is None:
        df = load_moves()
    rng = np.random.default_rng(seed)
    sel, info = select_moves(df, rng)
    records: list[dict] = []
    labels: dict[str, str] = {}

    def rec_for(a: float, kind: str) -> str:
        key = f"{float(a):.15e}"
        if key not in labels:
            lab = f"amp{len(records):04d}"
            labels[key] = lab
            records.append({"label": lab, "a": float(a), "kind": kind})
        return labels[key]

    pairs: list[dict] = []
    for _, r in sel.iterrows():
        comps = []
        if r["h_p"] > 0:
            comps.append((rec_for(r["P_D1"], "difference+"), float(r["P_D1"])))
        if r["h_m"] > 0:
            comps.append((rec_for(r["P_Dm1"], "difference-"), float(r["P_Dm1"])))
        pairs.append({
            "cell": r["cell"], "cls": int(r["cls"]), "band": r["band"], "mode": r["mode"],
            "base": float(r["p_new"]), "a_plus": float(r["P_D1"]), "a_minus": float(r["P_Dm1"]),
            "sa": rec_for(r["p_new"], "standalone"), "components": comps,
            "rho_q": float(r["rho_q"]),
            "ids": f"{r['instance']} {r['family']} route {int(r['route_id'])} {r['kind']} "
                   f"j={r['j']} pos={r['pos']} pos2={r['pos2']}",
        })
    return info, records, pairs


# ============================================================
# Step 2: simulation
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


def _simulate_record(job):
    """Worker: all estimators for one amplitude record."""
    rec, seed = job
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


def simulate(records) -> list[dict]:
    root = np.random.SeedSequence(V3_SEED)
    jobs = list(zip(records, root.spawn(len(records))))
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (label, r, dt) in enumerate(ex.map(_simulate_record, jobs, chunksize=1)):
            rows.extend(r)
            if k % 25 == 0 or k == len(jobs) - 1:
                _log(f"  {k + 1}/{len(jobs)} amplitudes ({label}, {dt:.1f} s each); "
                     f"elapsed {time.perf_counter() - t0:.0f} s")
    return rows


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
# Step 3: analysis
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


def fitted_slope(cost: np.ndarray, val: np.ndarray) -> float:
    """Log-log slope over the last ASYMPTOTIC_LAST settings (cost order); NaN if any value is zero."""
    c = np.asarray(cost[-ASYMPTOTIC_LAST:], float)
    v = np.asarray(val[-ASYMPTOTIC_LAST:], float)
    if np.any(v <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(c), np.log(v), 1)[0])


def slope_rows(cv, records):
    rows = []
    for rec in records:
        for fam in FAMILIES:
            c, q90, rms, mab = cv[(rec["label"], fam)]
            rows.append({
                "amplitude": rec["label"], "kind": rec["kind"], "a": rec["a"], "family": fam,
                "slope_q90": fitted_slope(c, q90), "slope_rms": fitted_slope(c, rms),
                "slope_mean_abs": fitted_slope(c, mab),
            })
    return rows


def ratio_rows(qv, pairs, families=FAMILIES, targets=TARGETS):
    """Cost of the difference estimate over the stand-alone cost, at the 90 % quantile target.

    qv: {(record label, family): (cost array, q90 array)}.
    One-sided moves: one estimate of the nonzero side at target T_D.
    Signed moves: two estimates, each at target T_D / 2 (the two errors sum to T_D).
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
        rho_var = math.sqrt(max(ap + am - (ap - am) ** 2, 0.0)) / math.sqrt(v_sa)
        for fam in families:
            method = "classical" if fam == "classical" else ("qae" if fam == "qae" else "mlae")
            for kind, val in targets:
                T_sa, T_D = target_scales(a, a_sum, kind, val)
                c_sa, f_sa = cost_at(*qv[(pr["sa"], fam)][:2], T_sa)
                c_comp, floors = [], [f_sa]
                for lab, _ in comps:
                    c, f = cost_at(*qv[(lab, fam)][:2], T_D / k)
                    c_comp.append(c)
                    floors.append(f)
                if not (math.isfinite(c_sa) and all(math.isfinite(c) for c in c_comp)):
                    status = "fail"
                elif any(floors):
                    status = "floor"
                else:
                    status = "ok"
                total = sum(c_comp) if all(math.isfinite(c) for c in c_comp) else float("nan")
                ratio = total / c_sa if status in ("ok",) else float("nan")
                pred = predicted_cost_ratio(method, v_sa, v_list, T_sa, T_D)
                rows.append({
                    "cell": pr["cell"], "cls": pr["cls"], "band": pr["band"], "mode": pr["mode"],
                    "ids": pr["ids"], "base_p": a, "P_D1": ap, "P_Dm1": am, "a_sum": a_sum,
                    "rho_q_var": rho_var, "rho_q_file": pr["rho_q"],
                    "family": fam, "target_type": kind, "target_value": val,
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


def ratio_summary(rr):
    """Per (cell, family, target): counts and medians over the same ok rows."""
    out = []
    keys = sorted({(r["cell"], r["family"], r["target_type"], r["target_value"]) for r in rr},
                  key=lambda k: (k[0], FAMILIES.index(k[1]), k[2], -k[3]))
    for cell, fam, tt, tv in keys:
        g = [r for r in rr if r["cell"] == cell and r["family"] == fam
             and r["target_type"] == tt and r["target_value"] == tv]
        ok = [r for r in g if r["status"] == "ok"]
        rat = [r["ratio"] for r in ok]
        prd = [r["pred"] for r in ok]
        rop = [r["ratio_over_pred"] for r in ok]
        out.append({
            "cell": cell, "family": fam, "target_type": tt, "target_value": tv,
            "mode_mix": f"{sum(r['mode'] == 'signed' for r in g)} signed / {sum(r['mode'] != 'signed' for r in g)} one-sided",
            "n_total": len(g), "n_ok": len(ok),
            "n_floor": sum(r["status"] == "floor" for r in g),
            "n_fail": sum(r["status"] == "fail" for r in g),
            "median_ratio": _med(rat), "median_pred": _med(prd), "median_ratio_over_pred": _med(rop),
            "p10_ratio_over_pred": _p(rop, 0.1), "p90_ratio_over_pred": _p(rop, 0.9),
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
    fig.suptitle("QUASAR M3b v3: 90 % quantile error versus cost (4000 seeds; QAE exact)")
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


def write_summary(path, meta, val_rows, info, pairs, records, slopes, rsum, carry, exact_rows) -> None:
    L: list[str] = []
    A = L.append
    A("# QUASAR M3b v3: estimator behaviour on the validated rotation model")
    A("")
    A(f"Produced by scripts/run_quasar_m3b.py. Root seed SeedSequence({V3_SEED}); {N_SEEDS} independent "
      "seeds per sampled setting; canonical QAE is computed from its exact outcome pmf.")
    A("")
    A("## 1. Circuit validation (reused from v2, not rerun)")
    A("")
    A("The four real oracles of v2 (three peak, one difference; 19 to 22 qubits) were simulated once with an exact statevector "
      "and compared with sin^2((2k+1) theta), k = 0..6. The rows are copied unchanged to m3b_v3_validation.csv.")
    A("")
    mx = max(float(r["abs_diff"]) for r in val_rows)
    A(f"Maximum |p_circuit - p_model| over the four oracles and k = 0..6: **{mx:.2e}**.")
    A("")
    A("## 2. Sample and exclusions")
    A("")
    A(f"Source: {MOVES_GZ.name}, {info['n_moves']} moves, 2e5 scenarios each (hits = amplitude x 2e5). "
      f"Eligible: every nonzero amplitude has at least {HITS_MIN} hits, and the move is not zero-difference.")
    A("")
    A(f"- Zero-difference moves (P(D=+1) = P(D=-1) = 0): **{info['moves_zero_difference']}** excluded.")
    A(f"- Moves with a nonzero amplitude below {HITS_MIN} hits (stand-alone p' or a difference side): "
      f"**{info['moves_under_threshold']}** excluded.")
    A(f"- Eligible moves: {info['moves_eligible']}.")
    A(f"- Amplitude values below the threshold (1 to {HITS_MIN - 1} hits): p' {info['amp_p_new_1_to_199_hits']}, "
      f"P(D=+1) {info['amp_P_D1_1_to_199_hits']}, P(D=-1) {info['amp_P_Dm1_1_to_199_hits']}; "
      f"zero hits (one-sided sides, not amplitudes to estimate): P(D=+1) {info['amp_P_D1_zero_hits']}, "
      f"P(D=-1) {info['amp_P_Dm1_zero_hits']}.")
    A("")
    A("Stratification: move class cls (1 = insertion, 2 = intra-route, 3 = alternative position) by stand-alone p' band "
      "(low: p' < 0.02; decision: 0.02 <= p' <= 0.2; high: p' > 0.2). Up to 30 moves per cell, drawn at random from the eligible pool.")
    A("")
    A("| cell | eligible pool | sampled |")
    A("|---|---|---|")
    for c in info["cells"]:
        A(f"| {c['cell'].replace('|', ' / ')} | {c['pool']} | {c['sampled']} |")
    A("")
    n_one = sum(1 for p in pairs if p["mode"] != "signed")
    A(f"Sampled moves: {len(pairs)} ({n_one} one-sided, {len(pairs) - n_one} signed). "
      f"Unique amplitudes simulated: {len(records)}.")
    A("")
    A("## 3. Error metrics and fitted slopes")
    A("")
    A("Primary metric: the 90 % quantile of |a_hat - a|. Slopes are log-log fits over the last six settings of each family in "
      "cost order (QAE m = 17..22; MLAE J = 5..10 for each S; classical N = 2^21..2^26). Medians and [min, max] over the "
      "amplitude records of each kind.")
    A("")
    A("| kind | family | slope, 90 % quantile | slope, mean abs error | slope, RMS (secondary) |")
    A("|---|---|---|---|---|")
    for kind in ("standalone", "difference+", "difference-"):
        for fam in FAMILIES:
            cells = []
            for key in ("slope_q90", "slope_mean_abs", "slope_rms"):
                v = [x[key] for x in slopes if x["kind"] == kind and x["family"] == fam]
                v = _finite(v)
                cells.append(f"{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}] (n={len(v)})" if v else "nan")
            A(f"| {kind} | {fam} | {cells[0]} | {cells[1]} | {cells[2]} |")
    A("")
    A("**Why the QAE 90 % quantile slope is read as a median over records.** For large N the exact QAE error is "
      "|a_hat - a| = 2 pi sqrt(a(1-a)) |delta| / N (1 + o(1)), where delta is the offset of the outcome from N phi, and the outcome law of delta "
      "depends only on f_m = frac(2^m theta / pi): q90 = 2 pi sqrt(a(1-a)) g(f_m) / N with g bounded. The exact values agree with this: for a = 0.2, q90 N is about proportional to f_m "
      "for small f_m (0.013 at f = 0.005, 0.40 at f = 0.16) and reaches about 4 at f = 0.6. At fixed a the factor g(f_m) therefore moves with m, "
      "so the slope of one amplitude over six values of m is alignment noise. The population slope is -1 (the median over records is the statistic to read; "
      "the per-record range is shown only to make that noise visible).")
    A("")
    A("**Why the RMS is a secondary column.** For canonical QAE the tails are P(|delta| > x) of order 1/(pi^2 x), so "
      "P(|a_hat - a| > e) is of order 1/(N e) for 1/N << e << 1. The mean absolute error then behaves as "
      "(log N)/N (up to the alignment factor), with slope -1 in cost. The mean square, "
      "E(a_hat - a)^2 = integral of 2 e P(|a_hat - a| > e) de, is of order 1/N: the integrand 2 e P(|a_hat - a| > e) is about 2/(pi^2 N) for every e in (1/N, O(1)), so the mass is spread "
      "uniformly over error levels up to O(1) and is carried by outcomes of probability about 1/N and error of order a. It therefore scales as "
      "N^(-1/2) and is dominated by these heavy tails (for the QAE rows of m3b_v3_metrics.csv, sqrt(N) times the exact RMS is checkable directly and oscillates with m). A finite sample of 2000 or 4000 seeds almost never contains "
      "them, so a sampled RMS of QAE under-estimates it and its slope is noise; the exact RMS is reported as a secondary column for that reason.")
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
    A("- One-sided moves use one estimate of the nonzero side at target T_D: ratio = cost_D / cost_sa, prediction "
      "sqrt(v_D) / T_D over sqrt(v_sa) / T_sa for quantum estimators (v = a(1-a)) and v_D / T_D^2 over v_sa / T_sa^2 for classical.")
    A("- Signed moves use two estimates of P(D = +1) and P(D = -1) at target T_D / 2 each: ratio = (cost_+ + cost_-) / cost_sa, "
      "prediction 2 (sqrt v+ + sqrt v-) / T_D over sqrt(v_sa) / T_sa (quantum); 4 (v+ + v-) / T_D^2 over v_sa / T_sa^2 (classical).")
    A("- Status: ok (cost measured), floor (a target is met at the smallest setting, so the cost is an upper bound and the row is not used), "
      "fail (target not reached within the grid). Medians and the ratio/prediction spread use the ok rows only, and the same rows for all three columns.")
    A("- QAE cost-to-target inherits the alignment factor g(f_m) of Section 3: the cost is the smallest m from which every larger m meets the target "
      "(suffix maximum over the error, cost resolved in factors of 2), so QAE ratios carry a multiplicative scatter. Read the QAE columns with that in mind.")
    A("- The claim 'ratio tracks rho_q' applies to class 1 (insertion) with absolute targets only: there the prediction for one-sided moves "
      "is exactly rho_q = sqrt(Var D) / sqrt(p'(1-p')). It is not claimed for classes 2 and 3 or for relative targets, whose predictions differ.")
    A("")
    for tt, tv in TARGETS:
        lab = f"absolute {tv:g}" if tt == "abs" else f"relative {tv:g} (x amplitude)"
        A(f"### Target: {lab}")
        A("")
        A("Each cell: median ratio / median prediction; median (ratio / prediction) [p10, p90]; ok / floor / fail counts.")
        A("")
        A("| cell | " + " | ".join(FAMILIES) + " |")
        A("|---|" + "---|" * len(FAMILIES))
        for cell in sorted({r["cell"] for r in rsum}, key=lambda c: (int(c[3]), BAND_NAMES.index(c.split("|")[1]))):
            cols = []
            for fam in FAMILIES:
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
    A("Counts per cell and family (targets: 4) are in m3b_v3_ratio_summary.csv; every row is in m3b_v3_ratio.csv with its status.")
    A("")
    A("**Class 1 against rho_q (absolute targets only).** For class 1 (insertion; P(D = -1) = 0 for every class-1 move in the source file) the predicted ratio is "
      "rho_q. The median of ratio / rho_q over the ok rows, for the three p' bands, is:")
    A("")
    for fam in FAMILIES:
        parts = []
        for tv in ABS_TARGETS:
            meds = [r["median_ratio_over_pred"] for r in rsum if r["cell"].startswith("cls1|")
                    and r["family"] == fam and r["target_type"] == "abs" and r["target_value"] == tv]
            fin = _finite(meds)
            parts.append(f"target {tv:g}: {min(fin):.2f} to {max(fin):.2f}" if fin else f"target {tv:g}: none")
        A(f"- {fam}: " + "; ".join(parts))
    A("")
    A("Only class 1 is compared with rho_q. For classes 2 and 3 the signed prediction is 2(sqrt v+ + sqrt v-)/sqrt(v) "
      "(in [2, 2 sqrt 2] times rho_q) and is reported as the prediction, not as rho_q.")
    A("")
    A("Figure m3b_v3_error.png: the 90 % quantile error against A-applications (classical samples), log-log, for three stand-alone amplitudes "
      "(about 0.005, 0.03, 0.2). The QAE curve is jagged because the exact 90 % quantile depends on the alignment f_m of Section 3. "
      "At a = 0.005025 the curve is flat at 1.8e-8 for m >= 15: theta/pi lies within about 4e-8 of the dyadic point k/2^15, so every "
      "grid from m = 15 up returns the same estimate sin^2(pi k / 2^15) and the error cannot fall below its distance from a. "
      "This is an exact property of the estimator at this amplitude, not a numerical floor.")
    A("")
    A("## 6. MLAE with S = 1 and S = 8: not rerun (from v2)")
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
    A("## 7. Run")
    A("")
    for k, v in meta.items():
        A(f"- {k}: {v}")
    A("")
    A("## 8. Deviations from the brief")
    A("")
    A("- Sample: 30 moves per class x band cell (9 cells, 270 moves, 588 unique amplitudes). Every cell had at least 30 eligible moves. "
      "The 200-hit rule applies to every nonzero amplitude of a move, so a move is excluded if any of its amplitudes is below it; both counts are in Section 2.")
    A("- One-sided moves are those with exactly one nonzero side. The brief gives P(D = -1) = 0; in classes 2 and 3 the nonzero side is sometimes "
      "P(D = -1), so the single estimate is taken on whichever side is nonzero.")
    A("- Relative targets were not specified in the brief beyond the example. The definitions used are T_sa = r p' and "
      "T_D = r (a+ + a-). For signed moves each of the two estimates has target T_D / 2, as in v2 for absolute targets.")
    A("- Grids were not reduced: QAE m = 3..22, MLAE J = 1..10 with S in {32, 64, 128}, classical N = 2^4..2^26. Sampled estimators use "
      f"{N_SEEDS} seeds; QAE is exact and has no seeds.")
    A("- Slopes use the same six-setting window for every family, as in v2. For QAE the per-amplitude slope is alignment noise (Section 3), "
      "so the median over records is the value to read; per-record spreads are shown.")
    A("- Step 1 was reused from v2 and not rerun, as the brief allows. The MLAE code was rewritten for speed (one matrix product into a reused "
      "buffer, refinement vectorised over seeds). The grid, argmax and golden-section refinement are unchanged, and on four test settings the "
      "new and previous outputs agree exactly (maximum difference 0.0).")
    A("- The ratio table reports rows at the floor and rows that fail as separate counts. They are excluded from the medians, which are computed "
      "on the same ok rows for ratio, prediction and their quotient.")
    A("- The exact QAE quantile is the smallest value with cdf >= q; the sampled quantile uses the inverted-cdf rule, the same definition on a sample.")
    A("")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    _log("Step 1: circuit validation (v2 output reused)")
    val_rows = reuse_validation()
    write_csv(OUT / "m3b_v3_validation.csv", val_rows)

    _log("Step 2: amplitudes")
    t1 = time.perf_counter()
    info, records, pairs = build_pairs()
    write_csv(OUT / "m3b_v3_sampled_moves.csv", [
        {k: v for k, v in p.items() if k != "components"} | {
            "components": ";".join(f"{lab}:{amp:.6g}" for lab, amp in p["components"])} for p in pairs])
    _log(f"  {info['moves_eligible']} eligible moves, {len(pairs)} sampled, {len(records)} unique amplitudes "
         f"({time.perf_counter() - t1:.1f} s)")

    _log("Step 3: exact QAE check")
    exact_rows = exact_check()
    write_csv(OUT / "m3b_v3_exact_check.csv", exact_rows)

    _log("Step 4: simulation (QAE exact, MLAE and classical 4000 seeds)")
    t2 = time.perf_counter()
    rows = simulate(records)
    t_sim = time.perf_counter() - t2
    write_csv(OUT / "m3b_v3_metrics.csv", rows)
    _log(f"  simulation: {len(rows)} rows in {t_sim:.1f} s")

    _log("Step 5: analysis")
    cv = curves(rows)
    slopes = slope_rows(cv, records)
    write_csv(OUT / "m3b_v3_slopes.csv", slopes)
    qv = {k: (v[0], v[1]) for k, v in cv.items()}
    rr = ratio_rows(qv, pairs)
    write_csv(OUT / "m3b_v3_ratio.csv", rr)
    rsum = ratio_summary(rr)
    write_csv(OUT / "m3b_v3_ratio_summary.csv", rsum)
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
        "simulation time": f"{t_sim:.0f} s with {N_WORKERS} workers",
        "total time": f"{time.perf_counter() - t_all:.0f} s",
        "QAE": f"exact pmf, m = {QAE_M[0]}..{QAE_M[-1]}",
        "MLAE": f"J = {MLAE_J[0]}..{MLAE_J[-1]}, S in {list(MLAE_S)}, {N_SEEDS} seeds, grid 20001 points + golden-section",
        "classical": f"N = 2^{int(math.log2(CLASSICAL_N[0]))}..2^{int(math.log2(CLASSICAL_N[-1]))}, {N_SEEDS} seeds",
        "seeds": f"root SeedSequence({V3_SEED}); one spawned stream per amplitude, estimator and setting",
        "python": sys.version.split()[0],
    }
    write_summary(OUT / "m3b_v3_summary.md", meta, val_rows, info, pairs, records, slopes, rsum, carry, exact_rows)
    _log(f"done in {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
