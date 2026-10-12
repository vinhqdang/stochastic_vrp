"""QUASAR M3b v2: estimator behaviour on the validated rotation model.

Design: papers/quasar/DESIGN_M3.md, Revision 2, "M3b v2 - estimator behaviour".

Step 1  Validate sin^2((2k+1) theta) for k = 0..6 on real small peak and
        difference circuits (statevector, <= 22 qubits).
Step 2  Simulate canonical QAE (m = 3..12), MLAE (J = 1..10, S in {1, 8, 64})
        and classical Monte Carlo (N = 2^4..2^22) on the rotation model, with
        2000 independent seeds per setting.
Step 3  Report RMS error against A-applications, fitted log-log slopes, and
        the cost ratio of difference and stand-alone estimators at equal RMS
        against rho_q.

Amplitudes. If results/quasar/m3a_v2_moves.csv.gz exists (columns cls, p_new,
P_D1, P_Dm1, ...), 30 random moves per move class (cls) with 0.001 < p_new < 0.5
are used: stand-alone a = p_new, difference a+ = P_D1, a- = P_Dm1. Otherwise a
fixed illustrative grid is used: stand-alone a in {0.001, 0.005, 0.02, 0.05,
0.1, 0.3}, difference a+ = a rho^2 with rho in {0.1, 0.3, 0.5} and a- = 0.

Run from svrpspd_wdro/:  python scripts/run_quasar_m3b.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import gamma

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum.ae_sim import estimates, grover_validation  # noqa: E402
from quantum.classical import DiscreteDemand, peak_distribution  # noqa: E402
from quantum.oracle import (  # noqa: E402
    build_difference_oracle,
    build_peak_oracle,
    exact_probability,
)

OUT = ROOT / "results" / "quasar"
PLANS = ROOT / "results" / "plans"
MOVES_GZ = OUT / "m3a_v2_moves.csv.gz"

ROOT_SEED = 20261012
N_SEEDS = 2000
GAMMA_CV = 0.30
K_VALID = range(7)
QAE_M = list(range(3, 23))  # spec 3..12; extended, see summary
MLAE_SETTINGS = [(J, S) for J in range(1, 11) for S in (1, 8, 64)]
CLASSICAL_N = [2**e for e in range(4, 27)]  # spec 2^4..2^22; extended, see summary
FALLBACK_A = [0.001, 0.005, 0.02, 0.05, 0.1, 0.3]
FALLBACK_RHO = [0.1, 0.3, 0.5]
TARGETS = [1e-3, 1e-4]
FAMILIES = ["qae", "mlae_S1", "mlae_S8", "mlae_S64", "classical"]
ASYMPTOTIC_LAST = 6  # fit slopes over the last six settings of each family
MOVES_PER_CLASS = 30
N_WORKERS = 3


def _log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================
# Step 1: circuit validation
# ============================================================


def _gamma_levels(mean: float, V: int, u: float) -> np.ndarray:
    """Level pmf of a gamma demand (cv GAMMA_CV) on V levels of width u.

    Level v covers [(v - 1/2) u, (v + 1/2) u); level 0 takes all mass below u/2
    and the top level all mass above (V - 3/2) u. Zero means give a point mass.
    """
    if mean <= 0.0:
        p = np.zeros(V)
        p[0] = 1.0
        return p
    shape = 1.0 / (GAMMA_CV * GAMMA_CV)
    scale = mean * GAMMA_CV * GAMMA_CV
    bounds = np.concatenate(([-np.inf], (np.arange(1, V) - 0.5) * u, [np.inf]))
    return np.diff(gamma.cdf(bounds, shape, scale=scale))


def small_demand(plan: dict, customers, b: int) -> DiscreteDemand:
    """Independent-marginal demand model for the listed customers (validation only).

    Built from the plan's dbar/pbar. The unit is set so that the largest mean
    plus three coefficients of variation sits at the top level. Customers not
    listed keep a point mass at zero.
    """
    V = 1 << b
    mu_d = np.asarray(plan["dbar"], dtype=float)
    mu_p = np.asarray(plan["pbar"], dtype=float)
    n = mu_d.size
    u = max((1.0 + 3.0 * GAMMA_CV) * max(mu_d[c], mu_p[c]) for c in customers) / (V - 1)
    pd = np.zeros((n, V))
    pp = np.zeros((n, V))
    pd[:, 0] = 1.0
    pp[:, 0] = 1.0
    for c in customers:
        pd[c] = _gamma_levels(mu_d[c], V, u)
        pp[c] = _gamma_levels(mu_p[c], V, u)
    return DiscreteDemand(pmf_d=pd, pmf_p=pp, unit=float(u))


def median_threshold(seq, dem: DiscreteDemand) -> tuple[int, float]:
    """Smallest level q with P(F <= q) >= 1/2 for the base route, and P(F > q)."""
    cdf = np.cumsum(peak_distribution(seq, dem))
    q = int(np.searchsorted(cdf, 0.5, side="left"))
    return q, float(1.0 - cdf[q])


def validation_cases():
    """(kind, plan name, route index, prefix length, b, diff spec or None)."""
    return [
        ("peak", "CON8-3", 0, 4, 1, None),
        ("peak", "CON8-0", 0, 2, 2, None),
        ("peak", "SCA8-0", 0, 2, 2, None),
        ("difference", "CON8-3", 0, 2, 1, {"j_index": 2, "pos": 2}),
    ]


def run_validation() -> list[dict]:
    rows: list[dict] = []
    for kind, name, ridx, L, b, dspec in validation_cases():
        plan = json.loads((PLANS / f"{name}.json").read_text())
        route = plan["res"]["Det"]["plan"][ridx]
        seq = [int(c) for c in route[:L]]
        if dspec is None:
            cust = seq
        else:
            j = int(route[dspec["j_index"]])
            cust = seq + [j]
        dem = small_demand(plan, cust, b)
        q, p_thr = median_threshold(seq, dem)
        if dspec is None:
            oracle = build_peak_oracle(seq, dem, q)
        else:
            oracle = build_difference_oracle(seq, j, dspec["pos"], dem, q)
        a_exact = exact_probability(oracle, method="enumeration")
        res = grover_validation(oracle, ks=K_VALID)
        max_diff = max(abs(pc - pm) for _, pc, pm in res)
        _log(
            f"  {kind:10s} {name} route[{ridx}] L={L} b={b} qubits={oracle.num_qubits} "
            f"Q_level={q} a={a_exact:.6f} max|diff|={max_diff:.2e}"
        )
        for k, pc, pm in res:
            rows.append(
                {
                    "kind": kind,
                    "plan": name,
                    "route_index": ridx,
                    "customers": " ".join(str(c) for c in cust),
                    "b": b,
                    "n_qubits": oracle.num_qubits,
                    "threshold_level": q,
                    "P_base_gt_threshold": p_thr,
                    "a_exact": a_exact,
                    "k": k,
                    "p_circuit": pc,
                    "p_model": pm,
                    "abs_diff": abs(pc - pm),
                }
            )
    return rows


# ============================================================
# Step 2: amplitudes
# ============================================================


def _amp_key(a: float) -> str:
    return f"{a:.15e}"


def build_pairs():
    """Return (source, records, pairs).

    records: unique amplitudes [{label, a, kind}] simulated once each.
    pairs: one per estimation target: {group, sa, dp, dm, base, a_plus,
    a_minus, ids} where sa/dp/dm are record labels (dm is the a- record).
    """
    records: list[dict] = []
    labels: dict[str, str] = {}

    def rec_for(a: float, kind: str) -> str:
        key = _amp_key(a)
        if key not in labels:
            lab = f"amp{len(records):04d}"
            labels[key] = lab
            records.append({"label": lab, "a": float(a), "kind": kind})
        return labels[key]

    pairs: list[dict] = []
    if MOVES_GZ.exists():
        import pandas as pd

        cols = ["instance", "family", "route_id", "cls", "kind", "j", "pos", "pos2",
                "p", "p_new", "P_D1", "P_Dm1", "var_D", "rho_q"]
        df = pd.read_csv(MOVES_GZ, usecols=cols)
        sel = df[(df["p_new"] > 0.001) & (df["p_new"] < 0.5)]
        rng = np.random.default_rng(ROOT_SEED)
        chosen = []
        for cls in sorted(sel["cls"].unique()):
            g = sel[sel["cls"] == cls]
            pick = rng.choice(len(g), size=min(MOVES_PER_CLASS, len(g)), replace=False)
            chosen.append(g.iloc[np.sort(pick)])
        sample = pd.concat(chosen)
        sample.to_csv(OUT / "m3b_v2_moves_sample.csv", index=False)
        for _, r in sample.iterrows():
            base = float(r["p_new"])
            ap = float(r["P_D1"])
            am = float(r["P_Dm1"])
            pairs.append({
                "group": f"cls{int(r['cls'])}",
                "sa": rec_for(base, "standalone"),
                "dp": rec_for(ap, "difference"),
                "dm": rec_for(am, "difference-minus"),
                "base": base, "a_plus": ap, "a_minus": am,
                "rho_file": float(r["rho_q"]),
                "ids": f"{r['instance']} {r['family']} route {int(r['route_id'])} {r['kind']} j={r['j']} pos={r['pos']} pos2={r['pos2']}",
            })
        source = (f"{MOVES_GZ.name}: {MOVES_PER_CLASS} random moves per class (cls) with "
                  f"0.001 < p_new < 0.5; {len(sample)} moves sampled")
    else:
        for a in FALLBACK_A:
            pairs_sa = rec_for(a, "standalone")
            for rho in FALLBACK_RHO:
                pairs.append({
                    "group": f"rho{rho:g}",
                    "sa": pairs_sa,
                    "dp": rec_for(a * rho * rho, "difference"),
                    "dm": rec_for(0.0, "difference-minus"),
                    "base": a, "a_plus": a * rho * rho, "a_minus": 0.0,
                    "rho_file": None, "ids": f"fallback a={a:g} rho={rho:g}",
                })
        source = "fallback grid (results/quasar/m3a_v2_moves.csv.gz absent)"
    return source, records, pairs


def _simulate_record(job):
    """Worker: all three estimators for one amplitude record."""
    rec, aseed = job
    t0 = time.perf_counter()
    method_seeds = aseed.spawn(3)
    a = rec["a"]
    rows: list[dict] = []
    for method, settings, mseed in (
        ("qae", QAE_M, method_seeds[0]),
        ("mlae", MLAE_SETTINGS, method_seeds[1]),
        ("classical", CLASSICAL_N, method_seeds[2]),
    ):
        for setting_seed, setting in zip(mseed.spawn(len(settings)), settings):
            est, cost = estimates(a, method, setting, N_SEEDS, np.random.default_rng(setting_seed))
            err = est - a
            if method == "mlae":
                J, S = setting
                fam = f"mlae_S{S}"
                sname = f"J={J},S={S}"
            elif method == "qae":
                fam = "qae"
                sname = f"m={setting}"
            else:
                fam = "classical"
                sname = f"N={setting}"
            ae = np.abs(err)
            rows.append({
                "amplitude": rec["label"],
                "kind": rec["kind"],
                "a": a,
                "family": fam,
                "setting": sname,
                "cost_A_apps_or_samples": int(cost),
                "rms": float(np.sqrt(np.mean(err * err))),
                "bias": float(np.mean(err)),
                "median_abs_err": float(np.median(ae)),
                "q90_abs_err": float(np.quantile(ae, 0.9)),
                "n_seeds": N_SEEDS,
            })
    return rec["label"], rows, time.perf_counter() - t0


def simulate(records) -> list[dict]:
    root = np.random.SeedSequence(ROOT_SEED)
    seeds = root.spawn(len(records))
    jobs = list(zip(records, seeds))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for label, r, dt in ex.map(_simulate_record, jobs):
            rows.extend(r)
            _log(f"  {label} a={next(x['a'] for x in records if x['label'] == label):.3e} done in {dt:.1f} s")
    return rows


# ============================================================
# Step 3: analysis
# ============================================================


def curves(rows):
    """{(amplitude label, family): (cost array, rms array)} with settings in cost order."""
    out: dict = {}
    for r in rows:
        key = (r["amplitude"], r["family"])
        out.setdefault(key, []).append((r["cost_A_apps_or_samples"], r["rms"]))
    res = {}
    for key, pts in out.items():
        pts.sort(key=lambda t: t[0])
        res[key] = (np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float))
    return res


def cost_at(cost: np.ndarray, rms: np.ndarray, eps: float):
    """Cost to reach RMS eps: the smallest cost from which every larger setting also has RMS <= eps.

    The suffix maximum is used (not the running minimum), so a lucky dip at one
    setting does not count; for the canonical QAE the RMS depends on the phase
    alignment of 2^m theta/pi and is not monotone in m. Log-log interpolation
    between the last failing and the first passing setting. Returns (cost, floor):
    floor is True if eps is met at the smallest setting (cost is then a floor).
    Returns (nan, False) if no suffix passes.
    """
    suf = np.maximum.accumulate(rms[::-1])[::-1]
    ok = np.nonzero(suf <= eps)[0]
    if ok.size == 0:
        return float("nan"), False
    i = int(ok[0])
    if i == 0:
        return float(cost[0]), True
    if suf[i] <= 0.0:
        return float(cost[i]), False
    t = (math.log(eps) - math.log(suf[i - 1])) / (math.log(suf[i]) - math.log(suf[i - 1]))
    lc = math.log(cost[i - 1]) + t * (math.log(cost[i]) - math.log(cost[i - 1]))
    return float(math.exp(lc)), False


def q90_curves(rows):
    """{(amplitude, family): (cost array, q90 absolute error array)}."""
    out: dict = {}
    for r in rows:
        out.setdefault((r["amplitude"], r["family"]), []).append((r["cost_A_apps_or_samples"], r["q90_abs_err"]))
    res = {}
    for key, pts in out.items():
        pts.sort(key=lambda t: t[0])
        res[key] = (np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float))
    return res


def fitted_slope(cost: np.ndarray, rms: np.ndarray) -> float:
    """Log-log slope over the last ASYMPTOTIC_LAST settings (NaN if any RMS is zero)."""
    c = cost[-ASYMPTOTIC_LAST:]
    r = rms[-ASYMPTOTIC_LAST:]
    if np.any(r <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(c), np.log(r), 1)[0])


def slope_rows(cv, qv, records, pairs):
    group_of = {}
    for pr in pairs:
        for role in ("sa", "dp", "dm"):
            group_of.setdefault(pr[role], (pr["group"], role))
    rows = []
    for rec in records:
        if rec["label"] not in group_of:
            continue
        grp, role = group_of[rec["label"]]
        for fam in FAMILIES:
            c, r = cv[(rec["label"], fam)]
            cq, q = qv[(rec["label"], fam)]
            rows.append({"group": grp, "role": role, "amplitude": rec["label"], "a": rec["a"],
                         "family": fam, "slope_rms": fitted_slope(c, r), "slope_q90": fitted_slope(cq, q)})
    return rows


def ratio_rows(cv, pairs):
    """Cost ratios (difference over stand-alone) at equal RMS, per estimation target."""
    out = []
    for pr in pairs:
        a, ap, am = pr["base"], pr["a_plus"], pr["a_minus"]
        vb = a * (1.0 - a)
        vp = ap * (1.0 - ap)
        vm = am * (1.0 - am)
        varD = ap + am - (ap - am) ** 2
        rho_q_var = math.sqrt(max(varD, 0.0)) / math.sqrt(vb) if vb > 0 else float("nan")
        for fam in FAMILIES:
            cs, rs = cv[(pr["sa"], fam)]
            cd, rd = cv[(pr["dp"], fam)]
            cm, rm = cv[(pr["dm"], fam)]
            if fam == "classical":
                pred_single = vp / vb
                pred_spec = 4.0 * (vp + vm) / vb
            else:
                pred_single = math.sqrt(vp) / math.sqrt(vb)
                pred_spec = 2.0 * (math.sqrt(vp) + math.sqrt(vm)) / math.sqrt(vb)
            for eps in TARGETS:
                c_sa, f_sa = cost_at(cs, rs, eps)
                c_single, f_single = cost_at(cd, rd, eps)
                c_p, f_p = cost_at(cd, rd, eps / 2)
                c_m, _ = cost_at(cm, rm, eps / 2)
                c_spec = c_p + c_m
                out.append({
                    "group": pr["group"], "ids": pr["ids"], "base_p": a, "P_D1": ap, "P_Dm1": am,
                    "family": fam, "target_rms": eps,
                    "cost_standalone": c_sa, "cost_diff_single": c_single,
                    "cost_diff_spec": c_spec,
                    "ratio_single": c_single / c_sa, "ratio_spec": c_spec / c_sa,
                    "pred_single": pred_single, "pred_spec": pred_spec,
                    "rho_q_var": rho_q_var, "rho_q_file": pr["rho_file"] if pr["rho_file"] is not None else "",
                    "signed": am > 0.0,
                    "at_floor": bool(f_sa or f_single or f_p),
                })
    return out


def _median(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _spread(vals):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return "-"
    return f"{np.quantile(v, 0.1):.3g} - {np.quantile(v, 0.9):.3g}"


def make_figure(cv, show, path: Path) -> None:
    """show: list of (record label, title)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "classical": ("grey", "o", "classical N"),
        "qae": ("tab:blue", "s", "QAE m"),
        "mlae_S1": ("tab:orange", ".", "MLAE S=1"),
        "mlae_S8": ("tab:green", "^", "MLAE S=8"),
        "mlae_S64": ("tab:red", "D", "MLAE S=64"),
    }
    fig, axes = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, title) in zip(axes, show):
        for fam in FAMILIES:
            c, r = cv[(lab, fam)]
            m = r > 0
            col, mk, name = styles[fam]
            ax.loglog(c[m], r[m], marker=mk, color=col, ls=":" if fam == "mlae_S1" else "-", ms=4, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("A-applications (samples for classical)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("RMS error")
    axes[-1].legend(fontsize=8)
    fig.suptitle("QUASAR M3b v2: RMS error vs cost on the rotation model (2000 seeds)")
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
        if not np.isfinite(x):
            return "nan"
        return f"{float(x):.{nd}g}"
    return str(x)


def write_summary(path, meta, val_rows, source, records, pairs, slopes, ratios) -> None:
    lines: list[str] = []
    A = lines.append
    A("# QUASAR M3b v2: estimator behaviour on the validated rotation model")
    A("")
    A(f"Generated by scripts/run_quasar_m3b.py. Seeds: root SeedSequence({ROOT_SEED}); "
      f"{N_SEEDS} independent replicates per setting; one spawned stream per amplitude and estimator.")
    A("")
    A("## Step 1: circuit validation (exact statevector, k = 0..6)")
    A("")
    A("| kind | plan / route | prefix L | b | customers | qubits | Q_level | P(base > Q_level) | a (enumeration) | max abs diff over k |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    seen = []
    for r in val_rows:
        key = (r["kind"], r["plan"], r["route_index"], r["b"], r["customers"])
        if key in seen:
            continue
        seen.append(key)
        grp = [x for x in val_rows if (x["kind"], x["plan"], x["route_index"], x["b"], x["customers"]) == key]
        L = len(r["customers"].split()) if r["kind"] == "peak" else len(r["customers"].split()) - 1
        A(f"| {r['kind']} | {r['plan']} route {r['route_index']} | {L} | {r['b']} | {r['customers']} | "
          f"{r['n_qubits']} | {r['threshold_level']} | {fmt(r['P_base_gt_threshold'])} | "
          f"{fmt(r['a_exact'], 6)} | {max(x['abs_diff'] for x in grp):.2e} |")
    overall = max(r["abs_diff"] for r in val_rows)
    A("")
    A(f"Maximum |p_circuit - p_model| over all oracles and k = 0..6: **{overall:.2e}**.")
    A("")
    A("Thresholds: the plan Q lies far above the peak of these prefixes (p = 0), so each oracle uses the discrete median of its base route's "
      "peak (smallest q with P(F <= q) >= 1/2). The demand marginals are independent gamma (cv 0.3) built from the plan's dbar/pbar, "
      "for circuit validation only; the M3a statistics use the copula model.")
    A("")
    A("## Amplitudes")
    A("")
    A(f"Source: {source}.")
    A(f"Unique amplitudes simulated: {len(records)}; estimation targets (pairs): {len(pairs)}.")
    A("")
    groups = sorted({pr["group"] for pr in pairs})
    A("| group | targets | median p' | median P(D=1) | median P(D=-1) | median rho_q (file) | median rho_q (recomputed) | share with P(D=-1) > 0 |")
    A("|---|---|---|---|---|---|---|---|")
    for g in groups:
        pg = [pr for pr in pairs if pr["group"] == g]
        rq_f = [pr["rho_file"] for pr in pg if pr["rho_file"] is not None]
        rq_r = [r for r in ratios if r["group"] == g and r["family"] == "qae" and r["target_rms"] == TARGETS[0]]
        rq_r = [r["rho_q_var"] for r in rq_r]
        A(f"| {g} | {len(pg)} | {fmt(_median([pr['base'] for pr in pg]))} | {fmt(_median([pr['a_plus'] for pr in pg]))} | "
          f"{fmt(_median([pr['a_minus'] for pr in pg]))} | {fmt(_median(rq_f)) if rq_f else '-'} | {fmt(_median(rq_r))} | "
          f"{sum(1 for pr in pg if pr['a_minus'] > 0) / len(pg):.2f} |")
    A("")
    A("## Fitted log-log slopes (last six settings per family; expected -1 quantum, -1/2 classical)")
    A("")
    A("Median (min, max) over the targets of each group; stand-alone records (sa) and difference records (dp). "
      "The last column is the slope of the 90% quantile of |a_hat - a| for the stand-alone records.")
    A("")
    A("| group | family | sa slope (RMS) | dp slope (RMS) | sa slope (90% abs error) |")
    A("|---|---|---|---|---|")
    for g in groups:
        for fam in FAMILIES:
            cells = []
            for role in ("sa", "dp"):
                v = [x["slope_rms"] for x in slopes if x["group"] == g and x["role"] == role and x["family"] == fam]
                v = [x for x in v if np.isfinite(x)]
                cells.append(f"{np.median(v):.3f} ({min(v):.3f}, {max(v):.3f})" if v else "nan")
            v = [x["slope_q90"] for x in slopes if x["group"] == g and x["role"] == "sa" and x["family"] == fam]
            v = [x for x in v if np.isfinite(x)]
            cells.append(f"{np.median(v):.3f} ({min(v):.3f}, {max(v):.3f})" if v else "nan")
            A(f"| {g} | {fam} | {cells[0]} | {cells[1]} | {cells[2]} |")
    A("")
    A("## Difference versus stand-alone cost at equal RMS")
    A("")
    A("cost_standalone: cost to reach RMS eps for p'. ratio_single: cost to reach RMS eps for one estimate of P(D=1) divided by cost_standalone "
      "(the relevant comparison when D is nonnegative, P(D=-1) = 0). ratio_spec: two estimates, P(D=1) and P(D=-1), each at RMS eps/2, "
      "divided by cost_standalone. Predicted single ratio: sqrt(P(D=1)(1-P(D=1))) / sqrt(p'(1-p')) for quantum families and the same squared "
      "for classical. Predicted spec ratio: 2 (sqrt v+ + sqrt v-) / sqrt(p'(1-p')) for quantum and 4 (v+ + v-)/(p'(1-p')) for classical, "
      "with v = P(1-P). rho_q (file) and rho_q (recomputed) are sqrt(Var D)/sqrt(p'(1-p')) with Var D = P(D=1) + P(D=-1) - (P(D=1)-P(D=-1))^2. "
      "Rows where a target is already met at the smallest setting are excluded (at_floor).")
    A("")
    A("Medians over valid rows (at_floor = False). n = valid rows. For the single ratio only rows with P(D=-1) = 0 are used (n_single).")
    A("")
    A("| group | family | target RMS | n | n_single | median ratio_single | median pred_single | ratio_single / pred p10-p90 | median ratio_spec | median pred_spec | ratio_spec / pred p10-p90 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        for fam in FAMILIES:
            for eps in TARGETS:
                rr = [r for r in ratios if r["group"] == g and r["family"] == fam and r["target_rms"] == eps and not r["at_floor"]]
                single = [r for r in rr if not r["signed"]]
                if not rr:
                    A(f"| {g} | {fam} | {eps:g} | 0 | 0 | - | - | - | - | - | - |")
                    continue
                x1 = [r["ratio_single"] / r["pred_single"] for r in single if r["pred_single"] > 0]
                x2 = [r["ratio_spec"] / r["pred_spec"] for r in rr if r["pred_spec"] > 0]
                A(f"| {g} | {fam} | {eps:g} | {len(rr)} | {len(single)} | "
                  f"{fmt(_median([r['ratio_single'] for r in single]))} | {fmt(_median([r['pred_single'] for r in single]))} | "
                  f"{_spread(x1)} | "
                  f"{fmt(_median([r['ratio_spec'] for r in rr]))} | {fmt(_median([r['pred_spec'] for r in rr]))} | "
                  f"{_spread(x2)} |")
    A("")
    A("Full per-target table: m3b_v2_ratio.csv. Sampled moves: m3b_v2_moves_sample.csv.")
    A("")
    A("## Notes on definitions and ranges")
    A("")
    A("- Settings were extended beyond the design grid where the design grid cannot reach the targets. "
      "Canonical QAE uses m = 3..22 (design 3..12). Its RMS is dominated by outcomes far from the peak, "
      "whose total probability is about sin^2(pi frac(2^m theta/pi)) / N, so the RMS depends on the alignment of 2^m theta/pi "
      "and is not monotone in m (it oscillates with m, and is about 0.01 at m = 12 for a = 0.07). "
      "Classical samples run to N = 2^26 (design 2^22): at a = 0.07 the target RMS 1e-4 needs N near 6.5e6.")
    A("- cost_at uses the suffix maximum of RMS over cost: the cost is the smallest cost from which every larger setting also meets the target. "
      "This avoids crediting a lucky dip (the QAE RMS is not monotone).")
    A("- The 90% absolute-error column reports the high-probability error that the canonical QAE bound actually controls; it is provided beside the RMS.")
    A("- Resolution of the QAE cost ratios: the canonical QAE cost moves in factors of two (cost = 2 (2^m - 1) + 1), so a cost-to-target is only resolved to about a factor of two, and the alignment effect above adds scatter. The QAE ratio/prediction spread columns should be read with this in mind; the classical and MLAE (S = 64) families have fine cost resolution and are the sharp test of the sqrt(variance) law.")
    A("")
    A("## Run")
    A("")
    for k, v in meta.items():
        A(f"- {k}: {v}")
    A("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    _log("Step 1: circuit validation")
    t1 = time.perf_counter()
    val_rows = run_validation()
    t_val = time.perf_counter() - t1
    write_csv(OUT / "m3b_v2_validation.csv", val_rows)
    _log(f"  validation: {len(val_rows)} rows in {t_val:.1f} s")

    _log("Step 2: amplitudes and simulations")
    source, records, pairs = build_pairs()
    _log(f"  source: {source}; {len(records)} unique amplitudes; {len(pairs)} targets")
    t2 = time.perf_counter()
    rms_rows = simulate(records)
    t_sim = time.perf_counter() - t2
    write_csv(OUT / "m3b_v2_rms.csv", rms_rows)
    _log(f"  simulation: {len(rms_rows)} rows in {t_sim:.1f} s")

    _log("Step 3: analysis")
    cv = curves(rms_rows)
    qv = q90_curves(rms_rows)
    slopes = slope_rows(cv, qv, records, pairs)
    ratios = ratio_rows(cv, pairs)
    write_csv(OUT / "m3b_v2_ratio.csv", ratios)
    write_csv(OUT / "m3b_v2_slopes.csv", slopes)
    # figure: three stand-alone amplitudes spanning the range
    targets = [0.005, 0.03, 0.2] if MOVES_GZ.exists() else [0.001, 0.02, 0.1]
    sa_recs = {pr["sa"]: pr["base"] for pr in pairs}
    show = []
    for t in targets:
        lab = min(sa_recs, key=lambda k: abs(sa_recs[k] - t))
        show.append((lab, f"stand-alone a = {sa_recs[lab]:.3g}"))
    make_figure(cv, show, OUT / "m3b_v2_rms.png")
    meta = {
        "validation time": f"{t_val:.1f} s",
        "simulation time": f"{t_sim:.1f} s (workers {N_WORKERS})",
        "total time": f"{time.perf_counter() - t_all:.1f} s",
        "python": sys.version.split()[0],
        "seeds per setting": N_SEEDS,
        "classical N": "2^4 .. 2^26",
        "QAE m": "3 .. 22",
        "MLAE": "J = 1 .. 10, S in {1, 8, 64}, grid 20001 points + golden-section refinement",
    }
    write_summary(OUT / "m3b_v2_summary.md", meta, val_rows, source, records, pairs, slopes, ratios)
    _log(f"done in {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
