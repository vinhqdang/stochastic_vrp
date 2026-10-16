"""Summaries and figures of the round-2 experiments (run_rarecert_r3.py).

Reads the checkpoints of every shard directory given by --roots (default: results/rarecert/r3_local
and results/rarecert/r3_colab/*), merges them per part, and writes
  results/rarecert/r3_summary.md      numbers for the manuscript, one section per part
  results/rarecert/r3_<part>.csv      merged rows
  papers/rarecert/manuscript/figures/rarecert_mhat_all.pdf   (m_hat diagnostic on all rows)
  papers/rarecert/manuscript/figures/rarecert_unconf.pdf     (pickup rescaling, both regions)

Usage (from svrpspd_wdro/):  python3 scripts/analyze_rarecert_r3.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results" / "rarecert"
FIG = ROOT.parent / "papers" / "rarecert" / "manuscript" / "figures"
GAMMAS = (1e-3, 1e-4, 1e-5, 1e-6)
PARTS = ("passrate", "mhat", "unconf", "rivals2", "coverage2", "hubcrude", "long", "r2crude")


def merged(part: str) -> list[dict]:
    files = glob.glob(str(RES / "r3_local" / "*" / part / f"{part}_checkpoint.jsonl"))
    files += glob.glob(str(RES / "r3_colab" / "*" / "s*" / part / f"{part}_checkpoint.jsonl"))
    files += glob.glob(str(RES / "r3_colab" / "*" / "*" / "s*" / part / f"{part}_checkpoint.jsonl"))
    done = {}
    for f in files:
        for line in open(f):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[json.dumps(doc["key"])] = doc["res"]
    rows = []
    for r in done.values():
        rows.extend(r if isinstance(r, list) else [r])
    return [r for r in rows if not r.get("error")]


def fl(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def med(v):
    a = np.array([x for x in v if math.isfinite(x)])
    return float(np.median(a)) if a.size else math.nan


def q(v, p):
    a = np.array([x for x in v if math.isfinite(x)])
    return float(np.quantile(a, p)) if a.size else math.nan


def g3(x):
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.3g}"


def pct(x, d=2):
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:.{d}f}%"


def write_csv(rows, path):
    if not rows:
        return
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


# ------------------------------------------------------------------ passrate


def sec_passrate(L, rows):
    L.append("## Plan-level result: overflow probability at the planned capacity Q\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    n_expected = 148
    L.append(f"{len(rows)} of {n_expected} hubs done. Hub SMIS N = 1e5 and route SMIS N = 2e4 per route at c = Q "
             "(results/plans_Region, SAA plans).\n")
    L.append("| region | hubs | median P_hub(Q) | q10 | q90 | P_region(Q) = 1 - prod(1 - P_h) |")
    L.append("|---|---|---|---|---|---|")
    for reg in ("VNN", "GD", "all"):
        sub = [r for r in rows if reg == "all" or r["region"] == reg]
        p = [fl(r["p_hub"]) for r in sub]
        pr = 1.0 - float(np.exp(np.sum(np.log1p(-np.minimum([x for x in p if math.isfinite(x)], 1 - 1e-16)))))
        L.append(f"| {reg} | {len(sub)} | {g3(med(p))} | {g3(q(p, .1))} | {g3(q(p, .9))} | {g3(pr)} |")
    L.append("")
    L.append("Share of hubs with P_hub(Q) <= gamma, and of routes with P_route(Q) <= gamma (route SMIS, "
             "tolerance: estimate <= gamma):\n")
    L.append("| gamma | VNN hubs | GD hubs | all hubs | VNN routes | GD routes | all routes |")
    L.append("|---|---|---|---|---|---|---|")
    for g in GAMMAS:
        cells = []
        for kind in ("hub", "route"):
            for reg in ("VNN", "GD", "all"):
                sub = [r for r in rows if reg == "all" or r["region"] == reg]
                if kind == "hub":
                    v = [fl(r["p_hub"]) <= g for r in sub if math.isfinite(fl(r["p_hub"]))]
                else:
                    v = [fl(x) <= g for r in sub for x in r["route_p"] if math.isfinite(fl(x))]
                cells.append(f"{np.mean(v) * 100:.1f}% ({sum(v)}/{len(v)})" if v else "n/a")
        L.append(f"| {g:g} | {' | '.join(cells)} |")
    L.append("")
    rp = [fl(x) for r in rows for x in r["route_p"]]
    L.append(f"Routes: {len(rp)}; median P_route(Q) {g3(med(rp))}, q90 {g3(q(rp, .9))}, share with P_route(Q) = 0: "
             f"{np.mean([x == 0 for x in rp if math.isfinite(x)]) * 100:.1f}%.\n")


# ------------------------------------------------------------------ mhat


def sec_mhat(L, rows):
    L.append("## Missed-mode mass on all 480 route-gamma rows\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    for r in rows:
        r["m_hat"], r["e_single"], r["rse_single"] = fl(r["m_hat"]), fl(r["e_single"]), fl(r["rse_single"])
        r["gamma"] = fl(r["gamma"])
        r["ratio"] = fl(r["ratio"])
    L.append(f"{len(rows)} rows. Single shift error e = p_single / p_ref - 1.\n")
    bins = [(0, 1e-6), (1e-6, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 2.0)]
    L.append("| m_hat bin | rows | region VNN / GD | median e | median abs e | max abs e | share e < 0 | "
             "median abs e / m_hat | median 2 rse |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["m_hat"] < hi or (hi == 2.0 and r["m_hat"] >= lo)]
        if not sub:
            L.append(f"| [{lo:g}, {hi:g}) | 0 | | | | | | | |")
            continue
        e = [r["e_single"] for r in sub]
        ae = [abs(x) for x in e]
        ratio = [abs(r["e_single"]) / r["m_hat"] for r in sub if r["m_hat"] > 0]
        nv = sum(1 for r in sub if r["region"] == "VNN")
        L.append(f"| [{lo:g}, {hi:g}) | {len(sub)} | {nv} / {len(sub) - nv} | {pct(med(e))} | {pct(med(ae))} | "
                 f"{pct(max(ae))} | {np.mean([x < 0 for x in e]):.2f} | {g3(med(ratio))} | "
                 f"{pct(2 * med([r['rse_single'] for r in sub]))} |")
    L.append("")
    L.append("By level (gamma) for rows with m_hat >= 1e-3:\n")
    L.append("| gamma | rows | median e | median abs e | median abs e / m_hat |")
    L.append("|---|---|---|---|---|")
    for g in GAMMAS:
        sub = [r for r in rows if math.isclose(r["gamma"], g, rel_tol=1e-6) and r["m_hat"] >= 1e-3]
        if sub:
            L.append(f"| {g:g} | {len(sub)} | {pct(med([r['e_single'] for r in sub]))} | "
                     f"{pct(med([abs(r['e_single']) for r in sub]))} | "
                     f"{g3(med([abs(r['e_single']) / r['m_hat'] for r in sub]))} |")
    L.append("")
    # figure
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for reg, mk, col in (("VNN", "o", "k"), ("GD", "s", "0.55")):
        sub = [r for r in rows if r["region"] == reg and math.isfinite(r["m_hat"]) and math.isfinite(r["e_single"])]
        x = np.maximum([r["m_hat"] for r in sub], 1e-8)
        y = np.maximum([abs(r["e_single"]) for r in sub], 1e-5)
        ax.scatter(x, y, s=9, marker=mk, facecolors="none" if reg == "VNN" else col, edgecolors=col, linewidths=0.5,
                   label={"VNN": "Northern Vietnam", "GD": "Guangdong"}[reg])
    g = np.logspace(-8, 0, 40)
    ax.plot(g, g, color="0.4", lw=0.8, ls="--", label=r"$|{\rm error}|=\hat m$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e-9, 2.0)
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel(r"missed-mode mass $\hat m$ (left edge: $\hat m\leq10^{-8}$)")
    ax.set_ylabel("|single-shift error|")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "rarecert_mhat_all.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ unconf


def sec_unconf(L, rows):
    L.append("## Pickup rescaling and correlated factors\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    for r in rows:
        for k in ("s", "r", "gamma", "varrho", "m_hat", "e_single", "e_ce", "rse_single", "rse_ref", "RV_single"):
            r[k] = fl(r.get(k))
    L.append(f"{len(rows)} rows. Reference = SMIS with 1e6 draws; e = p_method / p_ref - 1.\n")
    L.append("| region | pickup scale s | factor corr r | gamma | rows | varrho median | m_hat median | "
             "single e median | min | max | share e < 0 | share abs e > 2 rse | CE mixture abs e median |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(r["region"], r["s"], r["r"], r["gamma"]) for r in rows}, key=lambda k: (k[0], k[2], k[1], -k[3]))
    for reg, s, rr, g in keys:
        sub = [r for r in rows if (r["region"], r["s"], r["r"], r["gamma"]) == (reg, s, rr, g)]
        e = [r["e_single"] for r in sub]
        big = [abs(r["e_single"]) > 2 * r["rse_single"] for r in sub if math.isfinite(r["e_single"])]
        L.append(f"| {reg} | {s:g} | {rr:g} | {g:g} | {len(sub)} | {g3(med([r['varrho'] for r in sub]))} | "
                 f"{g3(med([r['m_hat'] for r in sub]))} | {pct(med(e))} | {pct(min(e))} | {pct(max(e))} | "
                 f"{np.mean([x < 0 for x in e if math.isfinite(x)]):.2f} | {np.mean(big):.2f} | "
                 f"{pct(med([abs(fl(r['e_ce'])) for r in sub]))} |")
    L.append("")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for reg, mk, col in (("VNN", "o", "k"), ("GD", "s", "0.55")):
        sub = [r for r in rows if r["region"] == reg and r["r"] == 0 and math.isfinite(r["e_single"])]
        ax.scatter([r["varrho"] for r in sub], [100 * r["e_single"] for r in sub], s=11, marker=mk,
                   facecolors="none" if reg == "VNN" else col, edgecolors=col, linewidths=0.6,
                   label={"VNN": "Northern Vietnam (pickups x0.5, x2)", "GD": "Guangdong (pickups x0.5)"}[reg])
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0.75, color="0.4", ls=":", lw=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"end-mode ratio $\varrho$ of the rescaled route")
    ax.set_ylabel(r"single-shift error (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "rarecert_unconf.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ hubcrude


def sec_hubcrude(L, rows):
    L.append("## Hub SMIS against crude Monte Carlo (gamma = 1e-3 calibrated capacity)\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    base = {}
    p = RES / "baselines_hubs.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            base[r["name"]] = r
    L.append("| hub | customers | routes | SMIS (N=1e6) | rse | crude | rse | z | SMIS/crude - 1 | "
             "exact quadrature | err est | SMIS/exact - 1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    z, rel = [], []
    for r in sorted(rows, key=lambda r: fl(r["n_customers"])):
        b = base.get(r["name"], {})
        pe = fl(b.get("p_ref"))
        L.append(f"| {r['name']} | {r['n_customers']} | {r['n_routes']} | {g3(fl(r['p_smis']))} | "
                 f"{pct(fl(r['rse_smis']))} | {g3(fl(r['p_crude']))} | {pct(fl(r['rse_crude']))} | "
                 f"{fl(r.get('z')):.2f} | {pct(fl(r.get('rel')))} | {g3(pe)} | {pct(fl(b.get('err_est')) / pe) if pe == pe else 'n/a'} | "
                 f"{pct(fl(r['p_smis']) / pe - 1) if pe == pe else 'n/a'} |")
        z.append(fl(r.get("z")))
        rel.append(fl(r.get("rel")))
    z = [x for x in z if math.isfinite(x)]
    rel = [x for x in rel if math.isfinite(x)]
    if len(z) > 1:
        t = np.mean(z) / (np.std(z, ddof=1) / math.sqrt(len(z)))
        L.append(f"\nMean z {np.mean(z):.2f} over {len(z)} hubs (t = {t:.2f}); mean relative difference "
                 f"{pct(np.mean(rel))}, median {pct(np.median(rel))}.\n")


# ------------------------------------------------------------------ rivals2


def sec_rivals2(L, rows):
    L.append("## Budget-fair rivals and quadrature grids\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    for r in rows:
        r["gamma"] = fl(r["gamma"])
    bal = lambda r: r["route_id"].startswith("VNN")  # noqa: E731
    L.append(f"{len(rows)} route-gamma rows (N = 1e6 simulator evaluations for every sampling method). "
             "Times are wall times on a loaded shared machine (indicative).\n")
    L.append("| gamma | group | rows | SMIS e med / max | two-end e med / max | GL shifted e med / max | "
             "CE small-pilot e med / max | SMIS RV | two-end RV | GL RV | CE small RV | CE small pilot | "
             "GL signed e median | SMIS signed e median |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in (1e-4, 1e-6):
        for name, f in (("balanced (VNN)", bal), ("other (GD)", lambda r: not bal(r)), ("all", lambda r: True)):
            sub = [r for r in rows if math.isclose(r["gamma"], g, rel_tol=1e-6) and f(r)]
            if not sub:
                continue

            def ae(k):
                v = [abs(fl(r.get(k))) for r in sub]
                return f"{pct(med(v))} / {pct(max([x for x in v if math.isfinite(x)] or [math.nan]))}"
            L.append(f"| {g:g} | {name} | {len(sub)} | {ae('smis_rel')} | {ae('two_rel')} | {ae('gl_rel')} | "
                     f"{ae('ce_small_rel')} | {g3(med([fl(r['smis_RV']) for r in sub]))} | "
                     f"{g3(med([fl(r['two_RV']) for r in sub]))} | {g3(med([fl(r['gl_RV']) for r in sub]))} | "
                     f"{g3(med([fl(r['ce_small_RV']) for r in sub]))} | "
                     f"{g3(med([fl(r['ce_small_pilot']) for r in sub]))} | "
                     f"{pct(med([fl(r['gl_rel']) for r in sub]))} | {pct(med([fl(r['smis_rel']) for r in sub]))} |")
    L.append("")
    L.append("Quadrature reference on coarser grids (error against the fine reference p_ref, wall time per "
             "evaluation):\n")
    L.append("| grid (nodes, V pair) | rows | abs e median | q90 | max | wall s median | q90 |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, label in (("q41_128", "41, (128,256)"), ("q21_256", "21, (256,512)"), ("q21_128", "21, (128,256)"),
                       ("q11_256", "11, (256,512)")):
        e = [abs(fl(r.get(f"{tag}_rel"))) for r in rows]
        t = [fl(r.get(f"{tag}_s")) for r in rows]
        L.append(f"| {label} | {sum(math.isfinite(x) for x in e)} | {pct(med(e))} | {pct(q(e, .9))} | "
                 f"{pct(max([x for x in e if math.isfinite(x)] or [math.nan]))} | {g3(med(t))} | {g3(q(t, .9))} |")
    L.append("\nReference grid for comparison: 41 nodes, V = (256, 512); median wall time in baselines_summary.md.\n")
    L.append("Wall time to reach relative precision 0.01 at 95% confidence (build + 1.96^2 RV / 0.01^2 draws at "
             "the measured per-draw time; medians over rows):\n")
    for name, kb, ks, kRV, nd in (("SMIS", "smis_build_s", "smis_sample_s", "smis_RV", 1),
                                  ("CE small pilot", "ce_small_build_s", "ce_small_s", "ce_small_RV", 1)):
        ts = []
        for r in rows:
            RV = fl(r.get(kRV))
            build, tot = fl(r.get(kb)), fl(r.get(ks))
            per = (fl(r.get("smis_sample_s")) / 1e6) if name == "SMIS" else (max(tot - build, 0.0) / 1e6)
            if math.isfinite(RV) and math.isfinite(build) and math.isfinite(per):
                ts.append(build + 1.96 ** 2 * RV / 0.01 ** 2 * per)
        L.append(f"- {name}: median {g3(med(ts))} s (q90 {g3(q(ts, .9))} s)")
    L.append("")


# ------------------------------------------------------------------ coverage2


def sec_coverage2(L, rows):
    L.append("## Coverage at rare levels on balanced routes\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    for r in rows:
        r["gamma"] = fl(r["gamma"])
    L.append("300 replications of N = 1e4 draws per route; SMIS proposal built once, single shift rebuilt in "
             "every replication. Reference: quadrature (41 nodes, V = 256/512).\n")
    L.append("| gamma | method | routes | coverage pooled | min | max | median mean error | median sd |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g in (1e-5, 1e-6):
        for m in ("smis", "single_rebuilt"):
            sub = [r for r in rows if math.isclose(r["gamma"], g, rel_tol=1e-6) and r["method"] == m]
            if not sub:
                continue
            c = [fl(r["coverage"]) for r in sub]
            L.append(f"| {g:g} | {m} | {len(sub)} | {np.mean(c):.3f} | {min(c):.3f} | {max(c):.3f} | "
                     f"{pct(med([fl(r['mean_rel']) for r in sub]))} | {pct(med([fl(r['sd_rel']) for r in sub]))} |")
    L.append("")


# ------------------------------------------------------------------ long


def sec_long(L, rows):
    L.append("## Routes with 15 or more stops (crude Monte Carlo reference)\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    for r in rows:
        r["gamma"] = fl(r["gamma"])
    L.append("| gamma | rows | L range | crude draws | ref rse median | SMIS e median / max (abs) | "
             "single e median / max | CE e median / max | SMIS RV | single RV | CE RV |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in (1e-3, 1e-4):
        sub = [r for r in rows if math.isclose(r["gamma"], g, rel_tol=1e-6)]
        if not sub:
            continue

        def ae(k):
            v = [abs(fl(r.get(k))) for r in sub]
            return f"{pct(med(v))} / {pct(max([x for x in v if math.isfinite(x)] or [math.nan]))}"
        Ls = [int(r["L"]) for r in sub]
        L.append(f"| {g:g} | {len(sub)} | {min(Ls)}-{max(Ls)} | {int(fl(sub[0]['n_crude'])):,} | "
                 f"{pct(med([fl(r['rse_ref']) for r in sub]))} | {ae('smis_rel')} | {ae('single_rel')} | "
                 f"{ae('ce_rel')} | {g3(med([fl(r['smis_RV']) for r in sub]))} | "
                 f"{g3(med([fl(r['single_RV']) for r in sub]))} | {g3(med([fl(r['ce_RV']) for r in sub]))} |")
    L.append("")


def sec_r2crude(L, rows):
    L.append("## Route-day simulator: crude Monte Carlo at gamma = 1e-4 and 1e-5\n")
    if not rows:
        L.append("No rows yet.\n")
        return
    L.append("| route | L | gamma | class | crude draws | crude p | rse | MDIS e | CE mixture e | single e |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (-fl(r["gamma"]), r["route_id"])):
        L.append(f"| {r['route_id']} | {r['L']} | {fl(r['gamma']):g} | {r['cls']} | {int(fl(r['n_crude'])):,} | "
                 f"{g3(fl(r['p_crude']))} | {pct(fl(r['rse_crude']))} | {pct(fl(r['rel_mdis']))} | "
                 f"{pct(fl(r['rel_ce']))} | {pct(fl(r['rel_single']))} |")
    for g in (1e-4, 1e-5):
        sub = [r for r in rows if math.isclose(fl(r["gamma"]), g, rel_tol=1e-6)]
        if sub:
            L.append(f"\ngamma {g:g}: {len(sub)} routes, MDIS abs e median {pct(med([abs(fl(r['rel_mdis'])) for r in sub]))}, "
                     f"max {pct(max(abs(fl(r['rel_mdis'])) for r in sub))}; median crude rse {pct(med([fl(r['rse_crude']) for r in sub]))}.")
    L.append("")


SECTIONS = {"r2crude": sec_r2crude, "passrate": sec_passrate, "mhat": sec_mhat, "unconf": sec_unconf, "rivals2": sec_rivals2,
            "coverage2": sec_coverage2, "hubcrude": sec_hubcrude, "long": sec_long}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    L = ["# RARECERT round 2: extra experiments\n"]
    for part in PARTS:
        rows = merged(part)
        write_csv(rows, RES / f"r3_{part}.csv")
        SECTIONS[part](L, rows)
    (RES / "r3_summary.md").write_text("\n".join(L))
    print("wrote", RES / "r3_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
