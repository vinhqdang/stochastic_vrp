#!/usr/bin/env python3
"""
make_figures.py — Explanatory visualizations for BATON on real city maps.

Outputs (results/figures/):
  fig1_city_maps.png     2x2 street maps (Hanoi, New York, Paris, Shanghai)
                         with the depot, customers and ALNS routes drawn on
                         the real drive network.
  fig2_how_it_works.png  The algorithm explainer: (A) onboard-load fan over
                         route stops with the capacity line; (B) the BATON
                         decision rule on one demand-spike day — estimated
                         cost-to-continue vs the per-stop handoff price, with
                         each policy's trigger stop; (C) resulting expected
                         execution cost per policy on 2,000 test days.
  fig3_map_replay.png    The same spike day replayed on the Hanoi map:
                         where the reactive policy breaches vs where BATON
                         hands off to a standby vehicle.

Colors follow the validated reference palette (fixed categorical order);
status red is reserved for the capacity breach.
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from dethloff_runner import (parse_dethloff, sample_demands, solve_fast,
                             DetGate, InflationGate, CV, DIST, SEED)
from core.otr_endpoint import fit_otr
from core.otr2 import fit_otr_peak, calibrate_B_empirical_peak
from core.costs import (LastMileCosts, route_cost_schedules, fit_lsm_general,
                        simulate_v2_general, simulate_tau_general,
                        tune_tau_general, oracle_costs_general)
from core.published_policies import pi_thresholds, simulate_pi, tune_pi

# ── palette (reference instance, light mode) ─────────────────────────────────
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
ROUTE_COLORS = [C["blue"], C["aqua"], C["yellow"], C["green"], C["violet"],
                C["red"]]
STATUS_CRITICAL = "#c62f2e"
ROAD   = "#e3e2dd"
INK    = "#1a1a19"
MUTED  = "#8a897f"

FIG_DIR = _WDRO / "results" / "figures"
CITY_DIR = _WDRO / "data" / "City"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load_graph(city):
    import osmnx as ox
    from make_city_instances import CITIES
    lat, lon, radius = CITIES[city]
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(CITY_DIR / "_cache")
    G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
    return ox.truncate.largest_component(G, strongly=True)


def _street_path(G, u, v):
    """Lat/lon polyline of the shortest street path between two OSM nodes."""
    import networkx as nx
    nodes = nx.shortest_path(G, u, v, weight="length")
    return [(G.nodes[w]["x"], G.nodes[w]["y"]) for w in nodes]


def _instance(city):
    name = f"{city.upper()}-100-1"
    D, dem, Q, n, scale = parse_dethloff(str(CITY_DIR / f"{name}.vrpspd"))
    coords = json.loads((CITY_DIR / f"{name}.coords.json").read_text())["nodes"]
    return name, D, dem, Q, n, scale, coords


def _plan(D, dem, Q, n, robust=False):
    dbar = dem[:, 0].astype(float)
    pbar = dem[:, 1].astype(float)
    # robust=True plans with the Gounaris-style inflation gate: routes then
    # depart with real slack (B = Q - L0 > 0), so capacity breaches happen
    # MID-ROUTE and the execution-policy story is visible. Deterministic
    # CW plans pack routes to Q and push all risk to the first stop.
    gate = InflationGate(Q, dbar, pbar) if robust else DetGate(Q, dbar, pbar)
    return solve_fast(D, gate, n), dbar, pbar


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — four city maps with routes
# ═══════════════════════════════════════════════════════════════════════════════

def fig1(cities=("hanoi", "nyc", "paris", "shanghai")):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, city in zip(axes.flat, cities):
        G = _load_graph(city)
        name, D, dem, Q, n, scale, coords = _instance(city)
        plan, dbar, pbar = _plan(D, dem, Q, n)
        osm_ids = [c[0] for c in coords]
        lat = {i: coords[i][1] for i in range(n)}
        lon = {i: coords[i][2] for i in range(n)}

        for u, v, data in G.edges(data=True):
            ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                    [G.nodes[u]["y"], G.nodes[v]["y"]],
                    color=ROAD, lw=0.35, zorder=1)

        for ri, route in enumerate(plan):
            col = ROUTE_COLORS[ri] if ri < len(ROUTE_COLORS) else MUTED
            seq = [0] + route + [0]
            for a, b in zip(seq[:-1], seq[1:]):
                try:
                    pts = _street_path(G, osm_ids[a], osm_ids[b])
                    ax.plot(*zip(*pts), color=col, lw=1.4, zorder=3,
                            solid_capstyle="round")
                except Exception:
                    ax.plot([lon[a], lon[b]], [lat[a], lat[b]],
                            color=col, lw=1.0, ls=":", zorder=3)
            ax.scatter([lon[c] for c in route], [lat[c] for c in route],
                       s=11, color=col, zorder=4, edgecolors="white",
                       linewidths=0.4)

        ax.scatter([lon[0]], [lat[0]], marker="*", s=260, color=INK,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.set_title(f"{city.upper()} — {len(plan)} routes, "
                     f"{n - 1} customers", fontsize=11, color=INK)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(list(lat.values())))))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("SVRPSPD instances on real street networks — "
                 "depot (★), customers and planned routes",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_city_maps.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_city_maps.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures 2 & 3 — the explainer on one Hanoi route
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_route_and_scenarios(city="hanoi"):
    """Pick a didactic route: meaningful overflow risk (3-30% of days) AND
    mid-route breaches (median breach stop >= 3), so the policies have both
    something to prevent and room to act. Deterministic CW plans pack routes
    to capacity (all risk at stop 1); a strongly inflated gate removes risk
    entirely; so we sweep the inflation level for the sweet spot."""
    name, D, dem, Q, n, scale, coords = _instance(city)
    dbar = dem[:, 0].astype(float)
    pbar = dem[:, 1].astype(float)

    rng_seed = SEED + 1
    rng = np.random.default_rng(rng_seed)
    dsc_tr = sample_demands(dbar, n, 1000, CV, DIST, rng)
    psc_tr = sample_demands(pbar, n, 1000, CV, DIST, rng)
    rng2 = np.random.default_rng(rng_seed + 99_991)
    dsc_te = sample_demands(dbar, n, 2000, CV, DIST, rng2)
    psc_te = sample_demands(pbar, n, 2000, CV, DIST, rng2)

    best, best_key = None, (-1.0, -1.0)
    best_plan, best_rate = None, 0.0
    for alpha in (0.10, 0.08, 0.12, 0.06, 0.15, 0.04):
        plan = solve_fast(D, InflationGate(Q, dbar, pbar, alpha=alpha), n)
        for route in plan:
            if len(route) < 6:
                continue
            r = np.array(route)
            B = float(Q - dbar[r].sum())
            if B <= 0:
                continue
            g = psc_te[:, r] - dsc_te[:, r]
            cum = np.cumsum(g, 1)
            br = cum.max(1) > B
            rate = float(br.mean())
            if not (0.03 <= rate <= 0.30):
                continue
            med_ostep = float(np.median((cum > B).argmax(1)[br] + 1))
            if med_ostep < 3:
                continue
            key = (med_ostep, rate)
            if key > best_key:
                best, best_key = route, key
                best_plan, best_rate = plan, rate
        if best is not None:
            break
    if best is None:                       # last resort: highest-rate route
        plan = solve_fast(D, InflationGate(Q, dbar, pbar, alpha=0.10), n)
        best = max((rt for rt in plan if len(rt) >= 6),
                   key=lambda rt: float(
                       (np.cumsum(psc_te[:, rt] - dsc_te[:, rt], 1).max(1)
                        > Q - dbar[np.array(rt)].sum()).mean()))
        best_plan, best_rate = plan, 0.0
    return (name, D, dem, Q, n, scale, coords, best_plan, dbar, pbar,
            dsc_tr, psc_tr, dsc_te, psc_te, best, best_rate)


def fig23(city="hanoi"):
    (name, D, dem, Q, n, scale, coords, plan, dbar, pbar,
     dsc_tr, psc_tr, dsc_te, psc_te, route, ovr) = _pick_route_and_scenarios(city)
    r = np.array(route)
    m = len(r)
    L0 = float(dbar[r].sum())
    B = float(Q - L0)
    g_tr = psc_tr[:, r] - dsc_tr[:, r]
    g_te = psc_te[:, r] - dsc_te[:, r]
    costs = LastMileCosts()
    H, E = route_cost_schedules(route, D, scale, costs)

    # policies
    v1_models = fit_otr(g_tr, B)
    fb_models = fit_otr_peak(g_tr, B)
    tau_v1 = tune_tau_general(g_tr, B, H, E, v1_models)
    cm = fit_lsm_general(g_tr, B, H, E)
    c3 = tune_pi("pi3", g_tr, B, H, E)
    thr3 = pi_thresholds("pi3", B, g_tr.mean(0), c3)

    # spike scenario: a breaching test day with a late peak AND the widest
    # gap between BATON's trigger and the breach — the anticipation the
    # figure exists to show
    cum_te = np.cumsum(g_te, axis=1)
    breach = cum_te.max(1) > B
    ostep = np.where(breach, (cum_te > B).argmax(1) + 1, 0)
    cand = np.where(breach & (ostep >= max(3, m // 2)))[0]

    def _trigger_on(day):
        w = cum_te[day]
        for k in range(1, m):
            if w[k - 1] > B:
                return None                          # breached before acting
            if cm[k].predict(np.array([w[k - 1]]))[0] > H[k - 1]:
                return k
        return None

    s, best_gap = None, -1
    for day in cand[:200]:
        kk = _trigger_on(int(day))
        if kk is not None and int(ostep[day]) - kk > best_gap:
            s, best_gap = int(day), int(ostep[day]) - kk
    if s is None:
        s = int(cand[0]) if len(cand) else int(np.argmax(cum_te.max(1)))
    spike = cum_te[s]                                # W_k of the spike day
    o = int(ostep[s]) if breach[s] else m

    # trigger stops on the spike day
    def first_trigger(pred):
        for k in range(1, m):
            if spike[k - 1] > B:
                return None                          # breached before acting
            if pred(k, spike[k - 1]):
                return k
        return None

    k_v2 = first_trigger(lambda k, w: cm[k].predict(np.array([w]))[0] > H[k - 1])
    k_v1 = first_trigger(lambda k, w: v1_models[k].predict(np.array([w]))[0] > tau_v1)
    k_p3 = first_trigger(lambda k, w: w > thr3[k - 1])

    # ── figure 2 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})

    axA, axB, axC = axes
    stops = np.arange(m + 1)
    fan = L0 + cum_te[:200]
    for i in range(200):
        axA.plot(stops, np.concatenate([[L0], fan[i]]), color=C["blue"],
                 alpha=0.05, lw=0.8, zorder=1)
    axA.plot(stops, np.concatenate([[L0], L0 + np.median(cum_te, 0)]),
             color=C["blue"], lw=2, zorder=3, label="median day")
    axA.plot(stops, np.concatenate([[L0], L0 + spike]), color=C["yellow"],
             lw=2.2, zorder=4, label="demand-spike day")
    axA.axhline(Q, color=STATUS_CRITICAL, ls="--", lw=1.4, zorder=2)
    axA.text(0.1, Q + 2, "vehicle capacity Q", color=STATUS_CRITICAL,
             fontsize=8.5)
    if o <= m:
        axA.scatter([o], [L0 + spike[o - 1]], marker="X", s=110,
                    color=STATUS_CRITICAL, zorder=6)
        axA.annotate("breach if nobody acts", (o, L0 + spike[o - 1]),
                     textcoords="offset points", xytext=(-70, 10),
                     fontsize=8.5, color=STATUS_CRITICAL)
    axA.set_xlabel("stop along route"); axA.set_ylabel("onboard load (kg)")
    axA.set_ylim(bottom=0, top=max(Q * 1.15, float(fan.max()) + 5))
    axA.set_title("A — Onboard load is uncertain: 200 simulated days",
                  fontsize=10, loc="left")
    axA.legend(frameon=False, fontsize=8.5, loc="lower left")

    # panel B: cost-to-continue vs price-to-hand-off along the spike day
    k_max = min(o, m - 1)                    # decisions end at the breach
    ks = np.arange(1, k_max + 1)
    chat = np.array([cm[k].predict(np.array([spike[k - 1]]))[0] for k in ks])
    axB.plot(ks, chat, color=C["blue"], lw=2,
             label=r"BATON estimate $\hat C_k$: cost of continuing")
    axB.plot(ks, H[:k_max], color=MUTED, lw=1.6, ls="-",
             label=r"handoff price $H_k$ at this stop")
    if k_v2:
        axB.scatter([k_v2], [chat[k_v2 - 1]], s=90, color=C["green"], zorder=5)
        axB.annotate("BATON hands off:\ncontinuing now costs more",
                     (k_v2, chat[k_v2 - 1]), textcoords="offset points",
                     xytext=(8, -34), fontsize=8.5, color=C["green"])
    if k_p3 and k_p3 != k_v2:
        axB.axvline(k_p3, color=C["violet"], lw=1.2, ls=":")
        axB.text(k_p3 + 0.1, axB.get_ylim()[1] * 0.55, "π3 rule",
                 rotation=90, fontsize=8, color=C["violet"])
    if k_v1 and k_v1 != k_v2:
        axB.axvline(k_v1, color=C["aqua"], lw=1.2, ls=":")
        axB.text(k_v1 + 0.1, axB.get_ylim()[1] * 0.8, "endpoint threshold",
                 rotation=90, fontsize=8, color=C["aqua"])
    if o <= m:
        axB.axvline(o, color=STATUS_CRITICAL, lw=1.2, ls="--")
        axB.text(o + 0.1, axB.get_ylim()[1] * 0.25, "breach",
                 rotation=90, fontsize=8, color=STATUS_CRITICAL)
    axB.set_xlabel("stop along route"); axB.set_ylabel("cost ($)")
    axB.set_title("B — The decision rule on the spike day", fontsize=10,
                  loc="left")
    axB.legend(frameon=False, fontsize=8.5, loc="upper right")

    # panel C: expected execution cost per policy on the test days
    pol_costs = {
        "reactive": simulate_tau_general(g_te, B, H, E, fb_models, tau=1.0),
        "tuned threshold": simulate_tau_general(g_te, B, H, E, v1_models, tau=tau_v1),
        "π3 rule":  simulate_pi(g_te, B, H, E, thr3),
        "BATON":  simulate_v2_general(g_te, B, H, E, cm),
    }
    orc = oracle_costs_general(g_te, B, H, E)
    labels = list(pol_costs) + ["oracle"]
    vals = [pol_costs[k]["mean_cost"] for k in pol_costs] + [float(orc.mean())]
    bar_cols = [MUTED, C["aqua"], C["violet"], C["blue"], INK]
    bars = axC.bar(labels, vals, color=bar_cols, width=0.62)
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8.5, color=INK)
    axC.set_ylabel("expected execution cost ($/day)")
    axC.set_title("C — Result over 2,000 test days", fontsize=10, loc="left")
    axC.tick_params(axis="x", labelrotation=20)

    fig.suptitle(f"How BATON works — route with {m} stops, {name} "
                 f"(peak-overflow rate {ovr * 100:.0f}%)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_how_it_works.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_how_it_works.png", flush=True)

    # ── figure 3: map replay ────────────────────────────────────────────────
    G = _load_graph(city)
    osm_ids = [c[0] for c in coords]
    lat = {i: coords[i][1] for i in range(n)}
    lon = {i: coords[i][2] for i in range(n)}

    fig, ax = plt.subplots(figsize=(9, 9))
    for u, v, data in G.edges(data=True):
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color=ROAD, lw=0.4, zorder=1)

    seq = [0] + route + [0]
    hand = k_v2 if k_v2 else m
    for i, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
        served_leg = i < hand           # legs the original truck still drives
        col = C["blue"] if served_leg else C["aqua"]
        ls = "-" if served_leg else "--"
        try:
            pts = _street_path(G, osm_ids[a], osm_ids[b])
            ax.plot(*zip(*pts), color=col, lw=2.2 if served_leg else 1.8,
                    ls=ls, zorder=3, solid_capstyle="round")
        except Exception:
            pass
    xs = [lon[c] for c in route]; ys = [lat[c] for c in route]
    ax.scatter(xs, ys, s=34, color=C["blue"], zorder=4,
               edgecolors="white", linewidths=0.6)
    for i, cnode in enumerate(route, 1):
        ax.annotate(str(i), (lon[cnode], lat[cnode]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7.5,
                    color=INK)
    ax.scatter([lon[0]], [lat[0]], marker="*", s=380, color=INK, zorder=6,
               edgecolors="white", linewidths=1.0)
    if k_v2:
        hn = route[k_v2 - 1]
        ax.scatter([lon[hn]], [lat[hn]], s=200, facecolors="none",
                   edgecolors=C["green"], linewidths=2.4, zorder=6)
        ax.annotate("BATON hands off here —\nstandby vehicle finishes "
                    "the dashed stops", (lon[hn], lat[hn]),
                    textcoords="offset points", xytext=(12, 12), fontsize=9,
                    color=C["green"])
    if o <= m:
        bn = route[o - 1]
        ax.scatter([lon[bn]], [lat[bn]], marker="X", s=180,
                   color=STATUS_CRITICAL, zorder=6)
        ax.annotate("reactive policy: vehicle\noverflows here instead",
                    (lon[bn], lat[bn]), textcoords="offset points",
                    xytext=(12, -26), fontsize=9, color=STATUS_CRITICAL)

    handles = [
        Line2D([], [], color=C["blue"], lw=2.2, label="original vehicle"),
        Line2D([], [], color=C["aqua"], lw=1.8, ls="--",
               label="standby vehicle after handoff"),
        Line2D([], [], marker="*", color=INK, lw=0, markersize=14,
               label="depot"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")
    pad = 0.004
    ax.set_xlim(min(xs + [lon[0]]) - pad, max(xs + [lon[0]]) + pad)
    ax.set_ylim(min(ys + [lat[0]]) - pad, max(ys + [lat[0]]) + pad)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(ys))))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("The demand-spike day replayed on the Hanoi street network",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_map_replay.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_map_replay.png", flush=True)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        fig1()
    if which in ("all", "23"):
        fig23()


# ═══════════════════════════════════════════════════════════════════════════════
# Figures 4 & 5 — results charts (dot plots)
# ═══════════════════════════════════════════════════════════════════════════════

def fig45():
    import pandas as pd
    GATES = ["Det", "SAA", "WDRO", "Gounaris", "Cui", "MDRO"]
    GATE_DISP = {"Det": "Deterministic", "SAA": "SAA-CVaR", "WDRO": "W-DRO",
                 "Gounaris": "Robust (Gounaris)", "Cui": "Robust (B-S budget)",
                 "MDRO": "Moment-DRO"}
    POL = [("fb_tau", "tuned threshold", MUTED, "o"),
           ("restock", "restock rule", C["violet"], "o"),
           ("v2_lsm", "BATON-HO", C["aqua"], "o"),
           ("v2_act", "BATON", C["blue"], "D"),
           ("oracle", "oracle (handoff-only)", INK, "*")]

    d = pd.read_csv(_WDRO / "results" / "results_grand_dethloff.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ys = np.arange(len(GATES))[::-1]
    for y, g in zip(ys, GATES):
        s = d[d.Plan == g]
        ax.axhline(y, color=ROAD, lw=0.8, zorder=1)
        for lbl, name, col, mk in POL:
            v = s[f"{lbl}_saving"].mean()
            ax.scatter([v], [y], s=150 if mk == "*" else (90 if mk == "D" else 55),
                       color=col, marker=mk, zorder=4 if lbl == "v2_act" else 3,
                       edgecolors="white", linewidths=0.8)
            if lbl == "v2_act":
                ax.annotate(f"{v:.0f}%", (v, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9,
                            color=C["blue"], fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([GATE_DISP[g] for g in GATES])
    ax.set_xlabel("expected-recourse saving vs reactive policy (%)")
    ax.set_xlim(-12, 60)
    handles = [plt.Line2D([], [], marker=mk, color=col, lw=0,
                          ms=11 if mk == "*" else (8 if mk == "D" else 7),
                          label=nm) for _, nm, col, mk in POL]
    ax.legend(handles=handles, frameon=False, fontsize=8.5,
              loc="upper left", bbox_to_anchor=(0.0, -0.12), ncols=5)
    ax.set_title("Execution-policy savings by planning gate — 40 Dethloff "
                 "instances, three-class fleet costs", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_gate_dots.png", dpi=200, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_gate_dots.png")

    # fig 5: cost sensitivity
    import glob as _glob
    CS = [("baseline", None),
          ("cheap emergencies (F$_{emg}$=25)", "F_emg_25"),
          ("dear emergencies (F$_{emg}$=60)", "F_emg_60"),
          ("cheap standby (F$_{sb}$=10)", "F_standby_10"),
          ("dear standby (F$_{sb}$=35)", "F_standby_35"),
          ("low SLA price (p$_{late}$=0.5)", "p_late_0_5"),
          ("high SLA price (p$_{late}$=3)", "p_late_3_0"),
          ("mild surge (s$_{emg}$=1.5)", "s_emg_1_5"),
          ("heavy surge (s$_{emg}$=4)", "s_emg_4_0")]
    base = d[d.Plan.isin(["Det", "SAA"])]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ys = np.arange(len(CS))[::-1]
    POL5 = [p for p in POL if p[0] != "v2_lsm"]
    for y, (disp, tag) in zip(ys, CS):
        s = base if tag is None else pd.read_csv(
            _WDRO / "results" / f"results_costsens_{tag}.csv")
        ax.axhline(y, color=ROAD, lw=0.8, zorder=1)
        for lbl, name, col, mk in POL5:
            v = s[f"{lbl}_saving"].mean()
            ax.scatter([v], [y], s=150 if mk == "*" else (90 if mk == "D" else 55),
                       color=col, marker=mk, zorder=4 if lbl == "v2_act" else 3,
                       edgecolors="white", linewidths=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in CS], fontsize=9)
    ax.set_xlabel("expected-recourse saving vs reactive policy (%)")
    handles = [plt.Line2D([], [], marker=mk, color=col, lw=0,
                          ms=11 if mk == "*" else (8 if mk == "D" else 7),
                          label=nm) for _, nm, col, mk in POL5]
    ax.legend(handles=handles, frameon=False, fontsize=8.5,
              loc="upper left", bbox_to_anchor=(0.0, -0.12), ncols=4)
    ax.set_title("Robustness of the policy ranking to fleet economics "
                 "(one factor at a time)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_costsens.png", dpi=200, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_costsens.png")
