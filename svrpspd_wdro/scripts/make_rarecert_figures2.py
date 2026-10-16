"""RARECERT round 2 figures built from the code and the result files.

fig_geometry      rarecert_geometry.pdf: one Vietnamese route (two end modes of similar energy) in the
                  plane spanned by the two end-mode dominating points; failure set, normal rings, and
                  200 draws from the single shift and from the SMIS mixture.
fig_cost          rarecert_cost_vs_level.pdf: samples for relative precision 0.1 and 0.01 against gamma
                  for crude Monte Carlo, SMIS (measured relative variance) and the CE rival; and the
                  wall time at 1% from the idle-machine timing.
fig_quantum       rarecert_quantum_calls.pdf: oracle calls against relative precision, classical and
                  quantum (explicit constants), at gamma = 1e-4 and 1e-6, with the gate time a logical
                  gate would need for quantum parity.
fig_oracle        rarecert_oracle_resources.pdf: qubits, controlled-NOT count and depth of the real
                  peak-load oracle (quantum.oracle) against route length and level bits.
fig_hubs          rarecert_hubs.pdf: overflow probability of the hubs at the planned capacity, the
                  capacity buffer needed against gamma, and the concentration of risk across routes.

Usage (from svrpspd_wdro/):  python3 scripts/make_rarecert_figures2.py [geometry|cost|quantum|oracle|hubs ...]
"""

from __future__ import annotations

import csv
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
import matplotlib.ticker  # noqa: E402

RES = ROOT / "results" / "rarecert"
FIG = ROOT.parent / "papers" / "rarecert" / "manuscript" / "figures"
BLUE, ORANGE, GREY, RED, GREEN = "#1f5fa8", "#d9730d", "0.45", "#b2182b", "#2a7f4f"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150})


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ geometry


def fig_geometry():
    import run_rarecert_r3 as r3
    from quantum.rare import RouteSim  # noqa: F401
    from rarecert.smis import block_direction, dominating_point, smis_mixture

    rid = "VNN-BACNINH#6"
    means = r3.rs.route_means()
    sim = r3._sim_for(rid, means)
    rows = [r for r in _read(RES / "smis_routes.csv") if r["route_id"] == rid and r["gamma"] == "0.0001"]
    c = float(rows[0]["c"])
    mix, diag = smis_mixture(sim, c, eta=1e-3, n_arc=0)
    xd, Ed = dominating_point(sim, c, 0, block_direction(sim, 0, "d"))
    xp, Ep = dominating_point(sim, c, sim.L, block_direction(sim, sim.L, "p"))
    ed, ep = xd / np.linalg.norm(xd), xp / np.linalg.norm(xp)

    def embed(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return a[..., None] * ed + b[..., None] * ep

    g = np.linspace(-3.5, 8.0, 161)
    A, B = np.meshgrid(g, g)
    F = sim.peak(embed(A, B).reshape(-1, sim.dim)).reshape(A.shape)

    rng = np.random.default_rng(5)
    n = 200

    def draw(thetas, weights, n):
        k = rng.choice(len(weights), size=n, p=weights / weights.sum())
        x = rng.standard_normal((n, sim.dim)) + thetas[k]
        return x, k

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.35), sharex=True, sharey=True)
    titles = ["single shift (cross-entropy at the dominant end)", "SMIS: mixture at the computed end modes"]
    # single shift: at the lower-energy end mode
    theta_single = (xp if Ep <= Ed else xd)[None, :]
    props = [(theta_single, np.array([1.0])), (mix.thetas, mix.weights)]
    for ax, ttl, (th, w) in zip(axes, titles, props):
        ax.contourf(A, B, (F > c).astype(float), levels=[0.5, 1.5], colors=["#f4d9d6"], alpha=0.9)
        ax.contour(A, B, F, levels=[c], colors=[RED], linewidths=1.1)
        for rr in (1, 2, 3, 4):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, ls=(0, (2, 3)), lw=0.5, color=GREY))
        x, k = draw(th, w, n)
        a, b = x @ ed, x @ ep
        fail = sim.peak(x) > c
        ax.scatter(a[~fail], b[~fail], s=5, color="0.6", lw=0, zorder=3)
        ax.scatter(a[fail], b[fail], s=7, color=BLUE, lw=0, zorder=4)
        ax.plot([np.linalg.norm(xd)], [0], marker="*", ms=9, color=ORANGE, mec="k", mew=0.4, zorder=6)
        ax.plot([0], [np.linalg.norm(xp)], marker="*", ms=9, color=ORANGE, mec="k", mew=0.4, zorder=6)
        ax.plot([0], [0], marker="+", ms=6, color="k", zorder=6)
        ax.set_title(ttl, fontsize=7.5)
        ax.set_xlabel(r"coordinate along the delivery-end point $x_d/|x_d|$")
        ax.set_xlim(-3.5, 8.0)
        ax.set_ylim(-3.5, 8.0)
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"coordinate along the pickup-end point $x_p/|x_p|$")
    axes[0].annotate(r"$x_d$: $E=%.1f$" % Ed, (np.linalg.norm(xd), 0), (np.linalg.norm(xd) - 0.2, -1.8), fontsize=7,
                     ha="center")
    axes[0].annotate(r"$x_p$: $E=%.1f$" % Ep, (0, np.linalg.norm(xp)), (3.4, np.linalg.norm(xp) + 0.5), fontsize=7)
    fig.text(0.5, 0.005, "blue: draws that overflow   grey: draws that do not   red line: $F=c$ in this slice through the origin   "
             "stars: the two end-mode dominating points   dotted: normal rings of radius 1 to 4", ha="center", fontsize=6.3)
    n_ok = [int((sim.peak(draw(th, w, 4000)[0]) > c).sum()) for th, w in props]
    axes[0].text(0.03, 0.97, f"share of draws that overflow: {n_ok[0] / 4000:.2f}", transform=axes[0].transAxes,
                 va="top", fontsize=6.5)
    axes[1].text(0.03, 0.97, f"share of draws that overflow: {n_ok[1] / 4000:.2f}", transform=axes[1].transAxes,
                 va="top", fontsize=6.5)
    fig.suptitle(f"{rid}, gamma = 1e-4, c = {c:,.0f}, $\\varrho$ = {float(rows[0]['ratio']):.2f}", fontsize=7.5, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / "rarecert_geometry.pdf")
    plt.close(fig)
    return {"route": rid, "Ed": Ed, "Ep": Ep, "share_single": n_ok[0] / 4000, "share_mix": n_ok[1] / 4000}


# ------------------------------------------------------------------ cost vs level


def fig_cost():
    rows = _read(RES / "smis_routes.csv")
    gam = [1e-3, 1e-4, 1e-5, 1e-6]
    RV_s, RV_ce = [], []
    for g in gam:
        sub = [r for r in rows if math.isclose(float(r["gamma"]), g, rel_tol=1e-6) and not r["error"]]
        RV_s.append(np.median([float(r["smis0_RV"]) for r in sub]))
        RV_ce.append(np.median([float(r["RV_is"]) for r in sub]))
    gam_ext = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
    # RV grows about 0.5 per decade (measured: 3.8 -> 5.9 over three decades); extrapolate linearly in log gamma
    slope = (RV_s[-1] - RV_s[0]) / 3.0
    RV_ext = RV_s[0] + slope * (-np.log10(gam_ext) - 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    ax = axes[0]
    for eps, ls in ((0.1, "-"), (0.01, "--")):
        crude = 1.96 ** 2 * (1 - gam_ext) / (gam_ext * eps ** 2)
        smis = 1.96 ** 2 * RV_ext / eps ** 2
        ax.plot(gam_ext, crude, ls, color=GREY, lw=1.1, label=f"crude Monte Carlo, $\\varepsilon={eps:g}$" if eps == 0.1 else None)
        ax.plot(gam_ext, smis, ls, color=BLUE, lw=1.3, label=f"SMIS (measured RV), $\\varepsilon={eps:g}$" if eps == 0.1 else None)
    ax.plot(gam_ext, 1.96 ** 2 * (1 - gam_ext) / (gam_ext * 0.01 ** 2), "--", color=GREY, lw=1.1)
    ax.plot(gam_ext, 1.96 ** 2 * RV_ext / 0.01 ** 2, "--", color=BLUE, lw=1.3)
    ax.axvspan(1e-6, 1e-3, color="#eef3fa", zorder=0)
    ax.set_ylim(1e2, 1e14)
    ax.text(3e-5, 3e2, "levels tested", ha="center", fontsize=7, color=BLUE)
    ax.text(3e-8, 3e7, "solid: $\\varepsilon=0.1$\ndashed: $\\varepsilon=0.01$", fontsize=7, ha="center")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"service level $\gamma$")
    ax.set_ylabel("samples for 95% confidence")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax = axes[1]
    t = _read(RES / "timing" / "timing.csv")
    names = [("smis_t", "SMIS", BLUE), ("single_t", "single\nCE shift", "0.55"), ("gl_t", "Glasserman-\nLi", "0.55"),
             ("ce_t", "CE\nmixture", "0.55"), ("quad_mid_t", "quadrature\n(128,256)", ORANGE),
             ("quad_fine_t", "quadrature\n(256,512)", ORANGE)]
    for j, (k, lab, col) in enumerate(names):
        v = [float(r[k]) for r in t]
        ax.scatter([j] * len(v), v, s=12, color=col, zorder=3, lw=0)
        ax.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color="k", lw=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names], fontsize=6.5)
    ax.set_ylabel("seconds to 1% (one core, idle)")
    ax.text(0.02, 0.97, r"$\gamma=10^{-4}$, six routes; bar: median", transform=ax.transAxes, va="top", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "rarecert_cost_vs_level.pdf")
    plt.close(fig)
    return {"RV_s": RV_s, "slope_per_decade": slope}


# ------------------------------------------------------------------ quantum


def _oracle_gate_count(L=8, bits=8):
    """Gate counts of one application of the real peak-load oracle (quantum.oracle)."""
    from quantum.classical import DiscreteDemand
    from quantum.oracle import build_peak_oracle, resource_counts
    V = 2 ** bits
    k = np.arange(V)
    pmf = np.exp(-0.5 * ((k - V / 8) / (V / 24)) ** 2)
    pmf /= pmf.sum()
    dem = DiscreteDemand(pmf_d=np.tile(pmf, (L, 1)), pmf_p=np.tile(pmf, (L, 1)), unit=1.0)
    orc = build_peak_oracle(list(range(L)), dem, Q_level=V * L // 5)
    return resource_counts(orc)


def fig_oracle():
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4))
    Ls = [2, 3, 4, 5, 6, 8, 10, 12]
    for bits, col in ((3, BLUE), (5, ORANGE), (8, GREEN)):
        q, cc, dp = [], [], []
        for L in Ls:
            rc = _oracle_gate_count(L, bits)
            q.append(rc["qubits"]), cc.append(rc["ccx"]), dp.append(rc["depth"])
        out[bits] = (q, cc, dp)
        for ax, y in zip(axes, (q, cc, dp)):
            ax.plot(Ls, y, "-o", ms=3, color=col, lw=1.1, label=f"{bits} bits per demand ($V={2 ** bits}$)")
    for ax, lab in zip(axes, ("qubits", "Toffoli gates", "circuit depth")):
        ax.set_xlabel("stops $L$")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=6.3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "rarecert_oracle_resources.pdf")
    plt.close(fig)
    return out


def fig_quantum():
    sys.path.insert(0, str(ROOT))
    from quantum.rare import aqe_cost, aqe_is_cost, best_aqe, c_crude, c_is
    rows = _read(RES / "smis_routes.csv")
    eps = np.logspace(-2.7, -0.5, 12)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    gam_list = (1e-4, 1e-6)
    for gi, g in enumerate(gam_list):
        sub = [r for r in rows if math.isclose(float(r["gamma"]), g, rel_tol=1e-6) and not r["error"]]
        RV = float(np.median([float(r["smis0_RV"]) for r in sub]))
        p = g
        cc = np.array([c_crude(p, e) for e in eps])
        ci = np.array([c_is(RV, e) for e in eps])
        qc = np.array([best_aqe(lambda k, d: aqe_cost(p, e, d, k))["ideal"] for e in eps])
        qi = np.array([best_aqe(lambda k, d: aqe_is_cost(RV, p, e, d, k))["ideal"] for e in eps])
        ax = axes[gi]
        ax.plot(eps, cc, color=GREY, lw=1.1, label="crude Monte Carlo")
        ax.plot(eps, ci, color=BLUE, lw=1.6, label="classical IS (SMIS), no pilot")
        ax.plot(eps, qc, "--", color=ORANGE, lw=1.1, label="QAE, crude oracle")
        ax.plot(eps, qi, "--", color=RED, lw=1.3, label="QAE, IS oracle (idealised)")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel("relative precision $\\varepsilon$")
        ax.set_ylabel("simulator or oracle calls")
        ax.set_title(f"$\\gamma={g:g}$, RV$={RV:.1f}$", fontsize=7.5)
        ax.invert_xaxis() if False else None
        if gi == 0:
            ax.legend(fontsize=6, loc="upper right")
    # break-even logical-gate time
    ax = axes[2]
    rc = _oracle_gate_count(8, 8)
    gates = float(rc["ccx"])          # Toffoli gates per oracle call (one route, 8 stops, 8 bits per demand)
    t_sim = 1.2e-5
    sub = [r for r in rows if math.isclose(float(r["gamma"]), 1e-4, rel_tol=1e-6) and not r["error"]]
    RV = float(np.median([float(r["smis0_RV"]) for r in sub]))
    qi = np.array([best_aqe(lambda k, d: aqe_is_cost(RV, 1e-4, e, d, k))["ideal"] for e in eps])
    ci = np.array([c_is(RV, e) for e in eps])
    tb = ci * t_sim / (qi * gates)
    tb_fast = ci * 1e-7 / (qi * gates)
    ax.plot(eps, tb, color=RED, lw=1.3, label="vs measured numpy draw ($1.2\\times10^{-5}$ s)")
    ax.plot(eps, tb_fast, "--", color=RED, lw=1.1, label="vs assumed compiled draw ($10^{-7}$ s)")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("relative precision $\\varepsilon$")
    ax.set_ylabel("time per Toffoli for parity (s)")
    ax.set_title("Toffoli time at which QAE ties classical IS", fontsize=7)
    ax.axhline(1e-6, color="0.3", lw=0.7, ls=":")
    ax.text(eps[-1], 1.4e-6, "1 $\\mu$s (reference)", fontsize=6.3, ha="right")
    ax.set_ylim(1e-12, 1e-5)
    ax.legend(fontsize=5.8, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    fig.savefig(FIG / "rarecert_quantum_calls.pdf")
    plt.close(fig)
    return {"gates_per_call": gates, "RV": RV, "tbreak_range": (float(tb.min()), float(tb.max()))}


# ------------------------------------------------------------------ hubs


def fig_hubs():
    pr = _read(RES / "r3_passrate.csv")
    hubs = _read(RES / "plan_hubs.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    ax = axes[0]
    for reg, col, lab in (("VNN", BLUE, "Northern Vietnam"), ("GD", ORANGE, "Guangdong")):
        v = [float(r["p_hub"]) for r in pr if r["region"] == reg]
        ax.hist(v, bins=np.linspace(0.0, 0.2, 21), color=col, alpha=0.75, label=f"{lab} ({len(v)})")
    ax.set_xlabel("overflow probability of a hub at $Q$")
    ax.set_ylabel("hubs")
    ax.legend(fontsize=6.5)
    ax = axes[1]
    for reg, col in (("VNN", BLUE), ("GD", ORANGE)):
        for g, mk in ((1e-3, "o"), (1e-5, "s")):
            sub = [r for r in hubs if r["region"] == reg and math.isclose(float(r["gamma"]), g, rel_tol=1e-6) and not r["error"]]
            ax.scatter([float(r["n_customers"]) for r in sub], [100 * float(r["b_final"]) for r in sub], s=9, marker=mk,
                       facecolors="none" if mk == "o" else col, edgecolors=col, lw=0.7)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([30, 100, 300])
    ax.set_xticklabels(["30", "100", "300"])
    ax.set_xlabel("customers of the hub")
    ax.set_ylabel("capacity buffer $c/Q-1$ (%)")
    ax.set_title("open: $\\gamma=10^{-3}$, filled: $10^{-5}$", fontsize=7)
    ax = axes[2]
    att = _read(RES / "plan_attribution.csv")
    shares = []
    import collections
    by = collections.defaultdict(list)
    for r in att:
        if math.isclose(float(r["gamma"]), 1e-3, rel_tol=1e-6):
            by[r["name"]].append(float(r["p_route_hubdraws"]))
    cum = []
    for name, p in by.items():
        p = np.sort(np.array(p))[::-1]
        cum.append(np.cumsum(p) / p.sum())
    K = 12
    M = np.array([np.pad(c, (0, max(0, K - len(c))), constant_values=1.0)[:K] for c in cum])
    xs = np.arange(1, K + 1)
    ax.plot(xs, np.median(M, axis=0), color=BLUE, lw=1.4)
    ax.fill_between(xs, np.quantile(M, 0.1, axis=0), np.quantile(M, 0.9, axis=0), color="#c9d9ee", lw=0)
    ax.set_xlabel("riskiest routes of a hub, in order")
    ax.set_ylabel("cumulative share of route risk")
    ax.text(0.97, 0.05, "median and 10-90% band\n$\\gamma=10^{-3}$, 66 hubs", transform=ax.transAxes, ha="right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "rarecert_hubs.pdf")
    plt.close(fig)
    return {"n_hubs": len(by)}


def main(argv=None) -> int:
    which = (argv or sys.argv[1:]) or ["geometry", "cost", "oracle", "quantum", "hubs"]
    fns = {"geometry": fig_geometry, "cost": fig_cost, "oracle": fig_oracle, "quantum": fig_quantum, "hubs": fig_hubs}
    FIG.mkdir(parents=True, exist_ok=True)
    for w in which:
        print(w, fns[w]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
