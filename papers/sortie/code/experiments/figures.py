"""Publication figures for the SORTIE paper, built from existing results only.

Reads results/*.csv and results/plans/**/*.json. Figures 1-3 re-simulate the
Cost, Det and SORTIE-0 plans of X641-n30-b0 with sortie.sim.simulate, using the
same seed and sample size as the E3 out-of-sample evaluation (the simulated
range reproduces the stored E_range exactly). Nothing is re-optimised.

Writes figures/<stem>.pdf and figures/<stem>.png (200 dpi) under papers/sortie/.

Run:  cd papers/sortie/code && python experiments/figures.py
"""

import csv
import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PAPER_DIR = os.path.dirname(CODE_DIR)
for _p in (CODE_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as cm  # noqa: E402  (sets thread limits before numpy is imported)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from sortie.instances import effective_H, noise_matrix  # noqa: E402
from sortie.pboi import lam_vector  # noqa: E402
from sortie.sim import simulate  # noqa: E402

RES = cm.RESULTS
FIG = os.path.join(PAPER_DIR, "figures")

METHOD_COLORS = {"det": "#c0392b", "saa": "#7f8c8d", "sortie0": "#2471a3",
                 "sortie1": "#17a589", "cost": "#8e44ad"}
METHOD_LABELS = {"det": "Det", "saa": "SAA", "sortie0": "SORTIE-0",
                 "sortie1": "SORTIE-1", "cost": "Cost"}
MEASURE_LABELS = {"max": "Max", "range": "Range", "gmd": "GMD", "mad": "MAD", "var": "Var"}
MEASURE_COLORS = {"max": "#0072b2", "range": "#d55e00", "gmd": "#009e73", "mad": "#cc79a7"}
ROUTE_COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#666666"]
FULL_W, HALF_W = 6.5, 3.2
EXAMPLE = (30, 0)  # representative instance (median SORTIE-0 vs Det gain, range, B=1.10) for figures 1-3 and 7
M_EXAMPLE = 5


# ----------------------------------------------------------------------------
# style and I/O helpers
# ----------------------------------------------------------------------------

def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 200,
        "figure.dpi": 100,
    })


_CACHE = {}


def rows(name):
    """All rows of results/<name>.csv (cached)."""
    if name not in _CACHE:
        with open(os.path.join(RES, f"{name}.csv"), newline="", encoding="utf-8") as fh:
            _CACHE[name] = list(csv.DictReader(fh))
    return _CACHE[name]


def _b_value(text):
    """Numeric budget factor of a CSV cell, or None for cost rows (nan)."""
    if text is None or text == "" or text.lower() == "nan":
        return None
    return round(float(text), 2)


def find(name, **kw):
    """Rows of results/<name>.csv whose fields equal all keyword filters.

    The keyword B is compared numerically; B=None matches cost rows.
    """
    out = []
    for r in rows(name):
        ok = True
        for k, v in kw.items():
            if k == "B":
                want = None if v is None else round(float(v), 2)
                if _b_value(r["B"]) != want:
                    ok = False
                    break
            elif r[k] != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def one(name, **kw):
    hits = find(name, **kw)
    if len(hits) != 1:
        raise RuntimeError(f"{name}: expected one row for {kw}, found {len(hits)}")
    return hits[0]


def plan_json(row):
    with open(os.path.join(RES, row["plan_file"]), encoding="utf-8") as fh:
        return json.load(fh)


def parse_nb(name):
    m = re.match(r"X641-n(\d+)-b(\d+)", name)
    return int(m.group(1)), int(m.group(2))


def e3_instances():
    names = {r["instance"] for r in rows("e3_main") if r["method"] != "cost"}
    return sorted(names, key=parse_nb)


def improvement(det_value, alt_value):
    """Percent improvement of the alternative over Det (positive = lower cost)."""
    return 100.0 * (det_value - alt_value) / det_value


def save(fig, stem):
    paths = []
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIG, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.03)
        paths.append(path)
    plt.close(fig)
    return paths


def simulated_workloads(inst, routes, n, b):
    """Realised driver workloads Phi = Y * X (S x m) with the E3 evaluation draws."""
    X, Y = simulate(inst, routes, cm.V_DEF, cm.SY, cm.S_OOS, cm.inst_seed(n, b))
    return X * Y[:, None]


def example_plans():
    """Cost, Det and SORTIE-0 range plans (B=1.10) of the example instance."""
    n, b = EXAMPLE
    name = cm.instance_name(n, b)
    inst = cm.load_instance(n, b)
    det = one("e3_main", instance=name, measure="range", B=1.10, method="det")
    s0 = one("e3_main", instance=name, measure="range", B=1.10, method="sortie0")
    plans = [
        dict(key="cost", label="Cost", routes=cm.load_seed(name)["routes"], row=None),
        dict(key="det", label="Det", routes=plan_json(det)["routes"], row=det),
        dict(key="sortie0", label="SORTIE-0", routes=plan_json(s0)["routes"], row=s0),
    ]
    for p in plans:
        L, W = cm.route_stats(p["routes"], inst.C, noise_matrix(inst))
        p["L"] = np.asarray(L, dtype=np.float64)
        p["W"] = np.asarray(W, dtype=np.float64)
        if p["row"] is not None:
            stored = np.asarray(plan_json(p["row"])["L"], dtype=np.float64)
            if not np.allclose(stored, p["L"]):
                print(f"[figures] warning: stored L differs from recomputed L for {p['key']}")
    return inst, plans


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def fig_routes():
    """Route maps of the Cost, Det and SORTIE-0 range plans (n=30, b=0, B=1.10)."""
    inst, plans = example_plans()
    C = inst.coords
    lo, hi = C.min(axis=0), C.max(axis=0)
    pad = 0.05 * float((hi - lo).max())
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.9), sharex=True, sharey=True)
    for idx, (ax, p) in enumerate(zip(axes, plans)):
        handles = []
        for k, route in enumerate(p["routes"]):
            color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
            xy = C[[0] + list(route) + [0]]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2, zorder=2)
            handles.append(Line2D([], [], color=color, lw=1.6,
                                  label=f"$R_{{{k + 1}}}$: $L_{{{k + 1}}}$ = {p['L'][k]:.0f}"))
        ax.scatter(C[1:, 0], C[1:, 1], s=7, c="black", linewidths=0, zorder=3)
        ax.scatter(C[0, 0], C[0, 1], marker="s", s=36, c="black", zorder=4)
        handles.append(Line2D([], [], marker="s", ls="", color="black", ms=4.5, label="Depot"))
        handles.append(Line2D([], [], marker="o", ls="", color="black", ms=3, label="Customer"))
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad + 0.16 * float(hi[1] - lo[1]))
        ax.set_aspect("equal")
        ax.set_xlabel("x coordinate")
        if idx == 0:
            ax.set_ylabel("y coordinate")
        ax.tick_params(labelsize=7)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                  fontsize=6.5, frameon=False, handlelength=1.4, columnspacing=0.8)
        ax.text(0.03, 0.97, f"({'abc'[idx]}) {p['label']}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return save(fig, "fig_routes")


def fig_workloads():
    """Violins of simulated per-driver workloads Phi_k for the three example plans."""
    n, b = EXAMPLE
    inst, plans = example_plans()
    data = []
    for p in plans:
        Phi = simulated_workloads(inst, p["routes"], n, b)
        order = np.argsort(Phi.mean(axis=0))
        Phi = Phi[:, order]
        L_sorted = p["L"][order]
        E_range = float(np.mean(Phi.max(axis=1) - Phi.min(axis=1)))
        L_range = float(p["L"].max() - p["L"].min())
        data.append((p, Phi, L_sorted, E_range, L_range))
        if p["row"] is not None:
            ref = float(p["row"]["oos_obj"])
            print(f"[figures] fig_workloads {p['key']}: E range {E_range:.3f} "
                  f"(E3 stored {ref:.3f})")
    ymax = max(float(np.quantile(Phi, 0.995)) for _, Phi, _, _, _ in data)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.9), sharey=True)
    for idx, (ax, (p, Phi, L_sorted, E_range, L_range)) in enumerate(zip(axes, data)):
        m = Phi.shape[1]
        pos = np.arange(1, m + 1)
        color = METHOD_COLORS[p["key"]]
        parts = ax.violinplot([Phi[:, k] for k in range(m)], positions=pos, widths=0.75,
                              showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.8)
        ax.scatter(pos, L_sorted, marker="D", s=13, color="black", zorder=4)
        ax.set_xticks(pos)
        ax.set_xlabel("Driver $k$ (sorted by mean)")
        ax.set_ylim(0, 1.32 * ymax)
        if idx == 0:
            ax.set_ylabel(r"Realised workload $\Phi_k = Y\,X_k$ (distance units)")
        ax.text(0.03, 0.97,
                f"E range = {E_range:.0f}\nDet range of $L$ = {L_range:.0f}\n"
                f"\u25C6 = $L_k$",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        ax.text(0.97, 0.03, f"({'abc'[idx]}) {p['label']}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return save(fig, "fig_workloads")


def fig_rangedist():
    """Histograms of the realised range for the Det and SORTIE-0 plans of fig_routes."""
    n, b = EXAMPLE
    inst, plans = example_plans()
    sel = [p for p in plans if p["key"] in ("det", "sortie0")]
    realised = {}
    for p in sel:
        Phi = simulated_workloads(inst, p["routes"], n, b)
        realised[p["key"]] = Phi.max(axis=1) - Phi.min(axis=1)
    pooled = np.concatenate(list(realised.values()))
    lo, hi = np.quantile(pooled, 0.002), np.quantile(pooled, 0.998)
    bins = np.linspace(lo, hi, 60)
    grid = np.linspace(lo, hi, 300)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.7))
    ytop = 1.8 * max(float(gaussian_kde(rr)(grid).max()) for rr in realised.values())
    for p in sel:
        color = METHOD_COLORS[p["key"]]
        rr = realised[p["key"]]
        ax.hist(rr, bins=bins, density=True, color=color, alpha=0.3, lw=0)
        ax.plot(grid, gaussian_kde(rr)(grid), color=color, lw=1.3)
        # reference lines stop below the legend
        ax.plot([float(p["L"].max() - p["L"].min())] * 2, [0, 0.6 * ytop],
                color=color, ls="--", lw=1.1)
        ax.plot([float(rr.mean())] * 2, [0, 0.6 * ytop], color=color, ls="-", lw=1.1)
        print(f"[figures] fig_rangedist {p['key']}: E range {rr.mean():.1f}, "
              f"range(L) {p['L'].max() - p['L'].min():.1f}")
    handles = [Patch(color=METHOD_COLORS[p["key"]], alpha=0.5,
                     label=f"{METHOD_LABELS[p['key']]} realised range") for p in sel]
    handles += [Line2D([], [], color="black", ls="--", lw=1.1, label="range of $L$ (deterministic)"),
                Line2D([], [], color="black", ls="-", lw=1.1, label="expected range $\\mathbb{E}$")]
    ax.set_ylim(0, ytop)
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7)
    ax.set_xlabel(r"Realised range $\max_k \Phi_k - \min_k \Phi_k$ (distance units)")
    ax.set_ylabel("Density")
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return save(fig, "fig_rangedist")


def fig_illusion_bound():
    """Out-of-sample gap of Det plans against the Proposition 4 bound."""
    lam1 = {}
    points = {}
    for r in rows("e3_main"):
        if r["method"] != "det" or r["measure"] not in MEASURE_COLORS:
            continue
        measure = r["measure"]
        m = int(r["m"])
        lam1[measure] = float(np.abs(lam_vector(measure, m)).sum())
        pj = plan_json(r)
        bound = lam1[measure] * float(r["v"]) * float(np.sqrt(np.sum(pj["W"])))
        gap = float(r["oos_obj_X"]) - float(r["exante"])
        points.setdefault(measure, ([], []))
        points[measure][0].append(bound)
        points[measure][1].append(gap)
    print("[figures] fig_illusion_bound ||lambda||_1:",
          {k: round(v, 3) for k, v in sorted(lam1.items())})
    fig, ax = plt.subplots(figsize=(HALF_W, 3.0))
    all_x, all_y = [], []
    for measure in ("max", "range", "gmd", "mad"):
        if measure not in points:
            continue
        x, y = points[measure]
        all_x += x
        all_y += y
        ax.scatter(x, y, s=10, color=MEASURE_COLORS[measure], alpha=0.75, linewidths=0,
                   label=MEASURE_LABELS[measure])
        above = int(np.sum(np.asarray(y) > np.asarray(x) + 1e-6))
        print(f"[figures] fig_illusion_bound {measure}: n={len(x)} above_bound={above}")
    top = 1.05 * max(max(all_x), max(all_y))
    bottom = min(0.0, 1.05 * min(all_y))
    ax.plot([0, top], [0, top], color="black", ls="--", lw=0.9, label="$y = x$")
    ax.set_xlim(0, top)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Prop. 4 bound $\|\lambda\|_1\, v\,\sqrt{\sum_k W_k}$ (distance units)")
    ax.set_ylabel(r"Gap $\mathbb{E}\,\ell_\lambda(X) - \ell_\lambda(L)$ (distance units)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    return save(fig, "fig_illusion_bound")


def fig_convergence():
    """Anytime curves of the in-model value (E7), relative to each run's final value."""
    T = np.geomspace(0.1, 120.0, 400)
    methods = ["det", "saa", "sortie0", "sortie1"]
    ns = (50, 75, 100)
    curves = {}
    for r in rows("e7_scale"):
        if r["method"] not in methods:
            continue
        pj = plan_json(r)
        h = np.asarray(pj["history"], dtype=np.float64)
        idx = np.clip(np.searchsorted(h[:, 0], T, side="right") - 1, 0, None)
        final = float(pj["value"])
        rel = 100.0 * (h[idx, 1] - final) / abs(final)
        curves.setdefault((int(r["n"]), r["method"]), []).append(rel)
    top = 1.1 * max(float(np.max(v)) for v in curves.values())
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.7), sharey=True)
    for idx, (ax, n) in enumerate(zip(axes, ns)):
        for method in methods:
            arr = np.asarray(curves[(n, method)])
            med = np.median(arr, axis=0)
            q1, q3 = np.percentile(arr, [25, 75], axis=0)
            ax.fill_between(T, q1, q3, color=METHOD_COLORS[method], alpha=0.18, lw=0)
            ax.plot(T, med, color=METHOD_COLORS[method], lw=1.4, label=METHOD_LABELS[method])
        ax.set_xscale("log")
        ax.set_xlim(0.1, 120)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.minorticks_off()
        ax.set_yscale("symlog", linthresh=10, linscale=0.6)
        ax.set_ylim(0, top)
        ax.set_yticks([0, 10, 50, 100, 200, 400])
        ax.set_yticklabels(["0", "10", "50", "100", "200", "400"])
        ax.set_xlabel("Time (s, log scale)")
        if idx == 0:
            ax.set_ylabel("Gap to final value (%)")
        ax.text(0.97, 0.97, f"({'abc'[idx]}) $n$ = {n}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, "fig_convergence")


def fig_e7b_seeds():
    """Seed-to-seed spread of the out-of-sample value, normalised by mean Det."""
    recs = rows("e7b_seeds")
    methods = ["det", "saa", "sortie0"]
    measures = ["range", "gmd", "max"]
    instances = sorted({r["instance"] for r in recs}, key=lambda s: int(s.rsplit("-b", 1)[1]))
    offsets = {"det": -0.22, "saa": 0.0, "sortie0": 0.22}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.9), sharey=True)
    for idx, (ax, measure) in enumerate(zip(axes, measures)):
        for ii, inst in enumerate(instances):
            det_vals = [float(r["oos_obj"]) for r in recs
                        if r["instance"] == inst and r["measure"] == measure and r["method"] == "det"]
            det_mean = float(np.mean(det_vals))
            for method in methods:
                vals = np.array([float(r["oos_obj"]) / det_mean for r in recs
                                 if r["instance"] == inst and r["measure"] == measure
                                 and r["method"] == method])
                x = ii + offsets[method]
                ax.scatter(np.full(len(vals), x), vals, s=13, color=METHOD_COLORS[method],
                           alpha=0.7, linewidths=0, zorder=3)
                ax.scatter([x], [vals.mean()], marker="D", s=34, color=METHOD_COLORS[method],
                           edgecolor="black", linewidths=0.7, zorder=4)
        ax.axhline(1.0, color="grey", ls=":", lw=0.8, zorder=1)
        ax.set_xticks(range(len(instances)))
        ax.set_xticklabels([s.rsplit("-", 1)[1] for s in instances])
        ax.set_xlim(-0.6, len(instances) - 0.4)
        ax.set_xlabel("Instance X641-n100 (block)")
        if idx == 0:
            ax.set_ylabel("Out-of-sample value / mean Det")
        ax.text(0.03, 0.97, f"({'abc'[idx]}) {MEASURE_LABELS[measure]}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", color=METHOD_COLORS[m], ms=4,
                      label=METHOD_LABELS[m]) for m in methods]
    handles.append(Line2D([], [], marker="D", ls="", mfc="white", mec="black", ms=5,
                          label="Mean over seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, "fig_e7b_seeds")


def fig_congestion():
    """Heterogeneous noise multiplier on X641-n30-b0 with the E8 range plans overlaid."""
    n, b = EXAMPLE
    inst = cm.load_instance(n, b, cong=True)
    name = cm.instance_name(n, b, cong=True)
    H = effective_H(inst)
    if H is None:
        raise RuntimeError("congestion instance has no heterogeneous multiplier")
    N = H.shape[0]
    off = ~np.eye(N, dtype=bool)
    node_H = np.array([H[i, off[i]].mean() for i in range(N)])
    vmin, vmax = float(H[off].min()), float(H[off].max())
    C = inst.coords
    panels = [("det", "Det"), ("sortie0", "SORTIE-0")]
    span = float((C.max(axis=0) - C.min(axis=0)).max())
    pad = 0.05 * span
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.5), sharex=True, sharey=True)
    sc = None
    for idx, (ax, (method, label)) in enumerate(zip(axes, panels)):
        row = one("e8_congestion", instance=name, measure="range", B=1.10, method=method)
        routes = plan_json(row)["routes"]
        L, _ = cm.route_stats(routes, inst.C, noise_matrix(inst))
        handles = []
        for k, route in enumerate(routes):
            color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
            xy = C[[0] + list(route) + [0]]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2, zorder=2)
            handles.append(Line2D([], [], color=color, lw=1.6,
                                  label=f"$R_{{{k + 1}}}$: $L_{{{k + 1}}}$ = {L[k]:.0f}"))
        sc = ax.scatter(C[1:, 0], C[1:, 1], c=node_H[1:], cmap="viridis", vmin=vmin, vmax=vmax,
                        s=26, edgecolor="black", linewidths=0.4, zorder=4)
        ax.scatter(C[0, 0], C[0, 1], marker="s", s=60, c=[node_H[0]], cmap="viridis",
                   vmin=vmin, vmax=vmax, edgecolor="black", linewidths=0.6, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x coordinate")
        if idx == 0:
            ax.set_ylabel("y coordinate")
        ax.tick_params(labelsize=7)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                  fontsize=6.5, frameon=False, handlelength=1.4, columnspacing=0.8)
        ax.set_xlim(C[:, 0].min() - pad, C[:, 0].max() + pad)
        ax.set_ylim(C[:, 1].min() - pad, C[:, 1].max() + pad + 0.16 * span)
        ax.text(0.03, 0.97,
                f"({'ab'[idx]}) {label}\nE range = {float(row['oos_obj']):.0f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.915, 0.22, 0.018, 0.56])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"Mean arc noise multiplier $\bar H_i$ (node $i$)")
    return save(fig, "fig_congestion")


def fig_heatmap_e3():
    """Per-instance improvement of SORTIE-0 over Det across measures and budgets (E3)."""
    insts = e3_instances()
    cols = [(m, B) for m in ("max", "range", "gmd", "mad", "var") for B in (1.05, 1.10)]
    M = np.zeros((len(insts), len(cols)))
    for i, inst in enumerate(insts):
        for j, (m, B) in enumerate(cols):
            d = one("e3_main", instance=inst, measure=m, B=B, method="det")
            s = one("e3_main", instance=inst, measure=m, B=B, method="sortie0")
            M[i, j] = improvement(float(d["oos_obj"]), float(s["oos_obj"]))
    vm = float(np.max(np.abs(M)))
    print(f"[figures] fig_heatmap_e3: share of cells with SORTIE-0 better = "
          f"{np.mean(M > 0):.2f}, median = {np.median(M):.2f}%")
    fig, ax = plt.subplots(figsize=(FULL_W, 5.4))
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vm, vmax=vm),
                   aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            colour = "white" if abs(M[i, j]) > 0.6 * vm else "black"
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.3, color=colour)
    for k in (4, 9, 14, 19):
        ax.axhline(k + 0.5, color="white", lw=1.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{MEASURE_LABELS[m]}\n{B:.2f}" for m, B in cols], fontsize=7)
    ax.set_yticks(range(len(insts)))
    ax.set_yticklabels([f"n{parse_nb(s)[0]}-b{parse_nb(s)[1]}" for s in insts], fontsize=6.5)
    ax.set_xlabel("Measure and budget factor $B$")
    ax.set_ylabel("Instance ($n$, block $b$)")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Improvement of SORTIE-0 over Det (%)")
    fig.tight_layout()
    return save(fig, "fig_heatmap_e3")


def fig_e8_vs_e3():
    """Paired SORTIE-0 vs Det improvements without (E3) and with (E8) congestion, B=1.10."""
    markers = {"range": "o", "gmd": "s", "max": "^"}
    xs = {m: [] for m in markers}
    ys = {m: [] for m in markers}
    for inst in e3_instances():
        for m in markers:
            d3 = one("e3_main", instance=inst, measure=m, B=1.10, method="det")
            s3 = one("e3_main", instance=inst, measure=m, B=1.10, method="sortie0")
            d8 = one("e8_congestion", instance=inst + "-cong", measure=m, B=1.10, method="det")
            s8 = one("e8_congestion", instance=inst + "-cong", measure=m, B=1.10, method="sortie0")
            xs[m].append(improvement(float(d3["oos_obj"]), float(s3["oos_obj"])))
            ys[m].append(improvement(float(d8["oos_obj"]), float(s8["oos_obj"])))
    lim = 1.1 * float(np.max(np.abs(np.concatenate([np.asarray(xs[m] + ys[m]) for m in markers]))))
    fig, ax = plt.subplots(figsize=(HALF_W, 3.2))
    for m, mk in markers.items():
        ax.scatter(xs[m], ys[m], marker=mk, s=17, color="#444444", alpha=0.75, linewidths=0,
                   label=MEASURE_LABELS[m])
    grid = np.linspace(-lim, lim, 400)
    ax.plot(grid, grid, color="black", ls="--", lw=0.8, label="$y = x$")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xscale("symlog", linthresh=2, linscale=0.6)
    ax.set_yscale("symlog", linthresh=2, linscale=0.6)
    ticks = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    ax.minorticks_off()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Improvement over Det, no congestion (E3, %)")
    ax.set_ylabel("Improvement over Det, congestion (E8, %)")
    ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return save(fig, "fig_e8_vs_e3")


def fig_cost_fairness():
    """Cost ratio against out-of-sample range relative to the Cost plan (E3, measure range)."""
    cost_range = {r["instance"]: float(r["E_range"]) for r in find("e3_main", method="cost")}
    markers = {1.05: "o", 1.10: "s"}
    fig, ax = plt.subplots(figsize=(HALF_W, 3.5))
    for r in find("e3_main", measure="range"):
        if r["method"] == "cost":
            continue
        x = float(r["cost_ratio"])
        y = float(r["oos_obj"]) / cost_range[r["instance"]]
        B = _b_value(r["B"])
        ax.scatter(x, y, marker=markers[B], s=15, color=METHOD_COLORS[r["method"]], alpha=0.8,
                   linewidths=0)
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvline(1.0, color="grey", ls=":", lw=0.8)
    method_handles = [Line2D([], [], marker="o", ls="", color=METHOD_COLORS[m], ms=4.5,
                             label=METHOD_LABELS[m]) for m in ("det", "saa", "sortie0", "sortie1")]
    b_handles = [Line2D([], [], marker=markers[B], ls="", color="#444444", mfc="#444444", ms=4.5,
                        label=f"$B$ = {B:.2f}") for B in (1.05, 1.10)]
    ax.legend(handles=method_handles + b_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=7)
    ax.set_xlabel(r"Cost ratio $\sum_k L_k / C^\star$")
    ax.set_ylabel("Out-of-sample range / Cost-plan range")
    fig.tight_layout()
    return save(fig, "fig_cost_fairness")


FIGURES = [
    fig_routes,
    fig_workloads,
    fig_rangedist,
    fig_illusion_bound,
    fig_convergence,
    fig_e7b_seeds,
    fig_congestion,
    fig_heatmap_e3,
    fig_e8_vs_e3,
    fig_cost_fairness,
]


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    produced, failed = [], []
    for fn in FIGURES:
        try:
            produced += fn()
            print(f"[figures] ok {fn.__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"[figures] FAILED {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            plt.close("all")
    print("[figures] produced:")
    for path in produced:
        print("  " + path)
    if failed:
        print("[figures] failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
