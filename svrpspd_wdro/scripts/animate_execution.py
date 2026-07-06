#!/usr/bin/env python3
"""
animate_execution.py — Animate the execution of a route assignment under a
realized demand scenario, on the real street network.

Takes (i) an ASSIGNMENT — the routes an algorithm produced, (ii) DATA — a
demand scenario with the actual delivery/pickup realized at every customer
(drawn from the same generator as the evaluation, selectable by index), and
(iii) an execution POLICY, then simulates the day and renders it:

  * vehicle 1 drives stop to stop along true OSM street paths, the load
    gauge updates with each realized delivery (-) and pickup (+);
  * if the policy triggers, vehicle 1 parks and a STANDBY vehicle takes
    over the remaining stops (dashed path, its own colour);
  * if nobody acts and capacity breaches, the stop flashes red and an
    EMERGENCY vehicle finishes the route.

Modes:
  policy=v2|v1|reactive     single animation
  policy=compare            reactive vs BATON side by side, same scenario

Usage:
    python scripts/animate_execution.py [instance=data/City/HANOI-100-1.vrpspd]
        [scenario=0] [policy=compare] [fps=7] [out=results/figures]

Scenario indices enumerate breaching test days of the chosen route
(scenario=0 is the first day whose demand path would breach capacity, so
there is something to see; use scenario=-1..-9 for calm days).
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from dethloff_runner import parse_dethloff
from core.otr_endpoint import fit_otr
from core.otr2 import fit_otr_peak
from core.costs import (LastMileCosts, route_cost_schedules,
                        fit_lsm_general, tune_tau_general)
from make_figures import (_load_graph, _street_path, _pick_route_and_scenarios,
                          C, ROAD, INK, MUTED, STATUS_CRITICAL)

VEH1   = C["blue"]
VEH2   = C["aqua"]      # standby
VEH3   = STATUS_CRITICAL  # emergency
FIG_DIR = _WDRO / "results" / "figures"


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation of one day under one policy -> event timeline
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_day(policy, m, B, Q, L0, d_real, p_real, H, E,
                 cm=None, v1_models=None, tau=None):
    """Step through the day. Returns dict with:
       switch_stop  stop AFTER which a handoff happens (None if never)
       breach_stop  stop AT which capacity physically breaches (None)
       loads        onboard load after each stop, len m+1 (index 0 = depart)
       cost         realized execution cost of the day
    """
    W = 0.0
    loads = [L0]
    switch = breach = None
    cost = 0.0
    for k in range(1, m + 1):
        W += p_real[k - 1] - d_real[k - 1]
        loads.append(L0 + W)
        if W > B and breach is None and switch is None:
            breach = k
            cost = float(E[k - 1])
            break
        if k == m or switch is not None or breach is not None:
            continue
        trig = False
        if policy == "v2" and cm is not None:
            trig = cm[k].predict(np.array([W]))[0] > H[k - 1]
        elif policy == "v1" and v1_models is not None:
            trig = v1_models[k].predict(np.array([W]))[0] > tau
        if trig:
            switch = k
            cost = float(H[k - 1])
    if breach is None and switch is None:
        pass                                        # clean completion, cost 0
    if switch is not None:
        # remaining stops are served by the standby vehicle; track its load
        Ws = 0.0
        for k in range(switch + 1, m + 1):
            Ws += p_real[k - 1] - d_real[k - 1]
            loads[k] = None                          # veh1 no longer loaded
        loads = loads[:switch + 1]
    return {"switch_stop": switch, "breach_stop": breach,
            "loads": loads, "cost": cost}


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry: frames along real street paths
# ═══════════════════════════════════════════════════════════════════════════════

def _resample(pts, n):
    """n points evenly spaced by arc length along a lon/lat polyline."""
    P = np.asarray(pts, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


def build_frames(G, osm_ids, route, sim, frames_per_leg=5, hold=2):
    """Timeline of frames: (x, y, vehicle_id, stop_just_served, event).
    vehicle_id: 1 original, 2 standby, 3 emergency."""
    seq = [0] + route + [0]
    switch, breach = sim["switch_stop"], sim["breach_stop"]
    frames = []
    veh = 1
    for i, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
        k_arrive = i + 1 if i + 1 <= len(route) else None   # stop index served
        try:
            pts = _resample(_street_path(G, osm_ids[a], osm_ids[b]),
                            frames_per_leg)
        except Exception:
            pts = np.array([[0, 0]] * frames_per_leg)
        for j, (x, y) in enumerate(pts):
            frames.append(dict(x=x, y=y, veh=veh, served=None, event=None))
        if k_arrive is not None and k_arrive <= len(route):
            frames.append(dict(x=pts[-1][0], y=pts[-1][1], veh=veh,
                               served=k_arrive, event=None))
            for _ in range(hold):
                frames.append(dict(x=pts[-1][0], y=pts[-1][1], veh=veh,
                                   served=None, event=None))
        if breach is not None and k_arrive == breach:
            for _ in range(4):
                frames.append(dict(x=pts[-1][0], y=pts[-1][1], veh=3,
                                   served=None, event="breach"))
            veh = 3                                    # emergency finishes
        if switch is not None and k_arrive == switch:
            for _ in range(4):
                frames.append(dict(x=pts[-1][0], y=pts[-1][1], veh=2,
                                   served=None, event="handoff"))
            veh = 2                                    # standby takes over
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_static(ax, G, lonr, latr, xs, ys, lon0, lat0):
    for u, v in G.edges():
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        if lonr[0] <= x1 <= lonr[1] and latr[0] <= y1 <= latr[1]:
            ax.plot([x1, G.nodes[v]["x"]], [y1, G.nodes[v]["y"]],
                    color=ROAD, lw=0.5, zorder=1)
    ax.scatter(xs, ys, s=46, facecolors="white", edgecolors=MUTED,
               linewidths=1.2, zorder=4)
    ax.scatter([lon0], [lat0], marker="*", s=380, color=INK, zorder=5,
               edgecolors="white", linewidths=1.0)
    ax.set_xlim(*lonr); ax.set_ylim(*latr)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(ys))))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def animate(instance_stem="HANOI-100-1", policy="compare", scen_rank=0,
            fps=7, out_dir=FIG_DIR):
    (name, D, dem, Q, n, scale, coords, plan, dbar, pbar,
     dsc_tr, psc_tr, dsc_te, psc_te, route, ovr) = _pick_route_and_scenarios(
        instance_stem.split("-")[0].lower())
    r = np.array(route)
    m = len(r)
    L0 = float(dbar[r].sum())
    B = float(Q - L0)
    g_tr = psc_tr[:, r] - dsc_tr[:, r]
    costs = LastMileCosts()
    H, E = route_cost_schedules(route, D, scale, costs)
    cm = fit_lsm_general(g_tr, B, H, E)
    v1_models = fit_otr(g_tr, B)
    tau = tune_tau_general(g_tr, B, H, E, v1_models)

    # choose the scenario: rank among breaching days (or calm days if <0)
    g_te = psc_te[:, r] - dsc_te[:, r]
    cum = np.cumsum(g_te, axis=1)
    breach_days = np.where(cum.max(1) > B)[0]
    calm_days = np.where(cum.max(1) <= B)[0]
    s = int(breach_days[scen_rank]) if scen_rank >= 0 else int(calm_days[scen_rank])
    d_real = dsc_te[s][r]
    p_real = psc_te[s][r]

    G = _load_graph(instance_stem.split("-")[0].lower())
    osm_ids = [c[0] for c in coords]
    lat = {i: coords[i][1] for i in range(n)}
    lon = {i: coords[i][2] for i in range(n)}
    xs = [lon[c] for c in route]; ys = [lat[c] for c in route]
    pad = 0.004
    lonr = (min(xs + [lon[0]]) - pad, max(xs + [lon[0]]) + pad)
    latr = (min(ys + [lat[0]]) - pad, max(ys + [lat[0]]) + pad)

    policies = ["reactive", "v2"] if policy == "compare" else [policy]
    sims, frame_sets = [], []
    for p in policies:
        sim = simulate_day(p, m, B, Q, L0, d_real, p_real, H, E,
                           cm=cm, v1_models=v1_models, tau=tau)
        sims.append(sim)
        frame_sets.append(build_frames(G, osm_ids, route, sim))
    n_frames = max(len(f) for f in frame_sets)
    for f in frame_sets:                              # pad to equal length
        f.extend([f[-1]] * (n_frames - len(f)))

    ncol = len(policies)
    fig, axes = plt.subplots(1, ncol, figsize=(8 * ncol, 9.5))
    axes = np.atleast_1d(axes)
    veh_dots, load_bars, load_txts, served_sc, ev_txts = [], [], [], [], []
    TITLES = {"reactive": "Reactive (no policy)",
              "v1": "endpoint threshold (v1)",
              "v2": "BATON (optimal stopping)"}
    trails = []          # per panel: {veh_id: Line2D with accumulated path}
    trail_axes = []
    for ax, p, sim in zip(axes, policies, sims):
        _draw_static(ax, G, lonr, latr, xs, ys, lon[0], lat[0])
        ax.set_title(f"{TITLES[p]} — realized cost ${sim['cost']:.1f}",
                     fontsize=12, color=INK)
        d, = ax.plot([], [], marker="o", ms=13, color=VEH1, zorder=8,
                     mec="white", mew=1.2)
        veh_dots.append(d)
        trails.append({})
        trail_axes.append(ax)
        served_sc.append(ax.scatter([], [], s=46, color=VEH1, zorder=6,
                                    edgecolors="white", linewidths=0.8))
        # load gauge (inset, bottom-left)
        gx = ax.inset_axes([0.03, 0.03, 0.05, 0.34])
        gx.set_xlim(0, 1); gx.set_ylim(0, Q * 1.25)
        gx.axhline(Q, color=STATUS_CRITICAL, ls="--", lw=1.2)
        gx.set_xticks([]); gx.set_yticks([0, int(Q)])
        gx.tick_params(labelsize=7)
        bar = gx.bar([0.5], [L0], width=0.9, color=VEH1)[0]
        load_bars.append((gx, bar))
        load_txts.append(gx.text(0.5, Q * 1.18, f"{L0:.0f} kg", ha="center",
                                 fontsize=7.5, color=INK))
        ev_txts.append(ax.text(0.03, 0.99, "", transform=ax.transAxes,
                               fontsize=10.5, va="top", color=INK))

    served_pts = [[] for _ in policies]
    load_now = [L0 for _ in policies]

    def update(fi):
        artists = []
        for pi, frames in enumerate(frame_sets):
            fr = frames[fi]
            col = {1: VEH1, 2: VEH2, 3: VEH3}[fr["veh"]]
            veh_dots[pi].set_data([fr["x"]], [fr["y"]])
            veh_dots[pi].set_color(col)
            # breadcrumb trail: the roads this vehicle has actually driven
            tr = trails[pi].get(fr["veh"])
            if tr is None:
                tr, = trail_axes[pi].plot([], [], color=col, lw=2.6,
                                          alpha=0.85, zorder=5,
                                          solid_capstyle="round")
                trails[pi][fr["veh"]] = tr
            tx, ty = tr.get_data()
            tr.set_data(np.append(tx, fr["x"]), np.append(ty, fr["y"]))
            artists.append(tr)
            if fr["served"]:
                k = fr["served"]
                served_pts[pi].append((xs[k - 1], ys[k - 1]))
                served_sc[pi].set_offsets(np.array(served_pts[pi]))
                load_now[pi] = load_now[pi] - d_real[k - 1] + p_real[k - 1]
                gx, bar = load_bars[pi]
                bar.set_height(max(load_now[pi], 0))
                over = load_now[pi] > Q
                bar.set_color(STATUS_CRITICAL if over else
                              {1: VEH1, 2: VEH2, 3: VEH3}[fr["veh"]])
                load_txts[pi].set_text(f"{load_now[pi]:.0f} kg")
                ev_txts[pi].set_text(f"stop {k}/{m}: "
                                     f"-{d_real[k-1]:.1f} kg delivered, "
                                     f"+{p_real[k-1]:.1f} kg picked up")
            if fr["event"] == "handoff":
                ev_txts[pi].set_text("HANDOFF: standby vehicle takes the "
                                     "remaining stops")
                ev_txts[pi].set_color(VEH2)
                gx, bar = load_bars[pi]
                load_now[pi] = 0.0  # veh1 unloaded; gauge follows new vehicle
                bar.set_height(0)
            if fr["event"] == "breach":
                ev_txts[pi].set_text("CAPACITY BREACH — emergency vehicle "
                                     "called at surge price")
                ev_txts[pi].set_color(STATUS_CRITICAL)
            artists += [veh_dots[pi], served_sc[pi], ev_txts[pi]]
        return artists

    legend = [Line2D([], [], marker="o", color=VEH1, lw=0, ms=10,
                     label="vehicle 1 (planned)"),
              Line2D([], [], marker="o", color=VEH2, lw=0, ms=10,
                     label="standby vehicle (after handoff)"),
              Line2D([], [], marker="o", color=VEH3, lw=0, ms=10,
                     label="emergency vehicle (after breach)")]
    axes[-1].legend(handles=legend, frameon=False, fontsize=9,
                    loc="lower right")
    day_kind = "demand-spike day" if scen_rank >= 0 else "ordinary day"
    fig.suptitle(f"{name} — one {day_kind}, route with {m} stops "
                 f"(same realized demands in every panel)", fontsize=13)
    fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out = Path(out_dir) / f"anim_{name}_{policy}_s{scen_rank}.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"wrote {out}  ({n_frames} frames)")
    return out


def animate_fleet(instance_stem="HANOI-200-1", scen_rank=0, fps=7,
                  out_dir=FIG_DIR, alpha_gate=0.10, still=False):
    """Whole-plan animation: every vehicle of the plan drives its route
    simultaneously under BATON, with breadcrumb trails, per-route colors,
    handoff markers (standby vehicles in the route's color, dashed trail)
    and breach markers. One shared demand scenario for the whole fleet.
    With still=True, replay every frame onto one axes and save a static
    print-quality PNG (full trails + all event markers) instead of a GIF."""
    from dethloff_runner import (parse_dethloff, sample_demands, solve_fast,
                                 InflationGate, CV, DIST, SEED)
    from make_figures import ROUTE_COLORS
    import json as _json

    city = instance_stem.split("-")[0].lower()
    inst = _WDRO / "data" / "City" / f"{instance_stem}.vrpspd"
    D, dem, Q, n, scale = parse_dethloff(str(inst))
    coords = _json.loads((inst.parent / f"{instance_stem}.coords.json")
                         .read_text())["nodes"]
    dbar = dem[:, 0].astype(float)
    pbar = dem[:, 1].astype(float)
    plan = [r for r in solve_fast(D, InflationGate(Q, dbar, pbar,
                                                   alpha=alpha_gate), n) if r]

    seed = SEED + 5
    rng = np.random.default_rng(seed)
    dsc_tr = sample_demands(dbar, n, 1000, CV, DIST, rng)
    psc_tr = sample_demands(pbar, n, 1000, CV, DIST, rng)
    rng2 = np.random.default_rng(seed + 99_991)
    dsc_te = sample_demands(dbar, n, 500, CV, DIST, rng2)
    psc_te = sample_demands(pbar, n, 500, CV, DIST, rng2)
    costs = LastMileCosts()

    # fit BATON per route; pick the day with the most "at-risk" routes
    fits, risk = [], np.zeros(500)
    for route in plan:
        r = np.array(route)
        B = float(Q - dbar[r].sum())
        g_tr = psc_tr[:, r] - dsc_tr[:, r]
        H, E = route_cost_schedules(route, D, scale, costs)
        cm = fit_lsm_general(g_tr, B, H, E)
        fits.append((route, B, H, E, cm))
        g_te = psc_te[:, r] - dsc_te[:, r]
        risk += (np.cumsum(g_te, 1).max(1) > B).astype(float)
    day = int(np.argsort(risk)[-(1 + scen_rank)])

    G = _load_graph(city)
    osm_ids = [c[0] for c in coords]
    lat = [c[1] for c in coords]
    lon = [c[2] for c in coords]

    frame_sets, sims = [], []
    for route, B, H, E, cm in fits:
        r = np.array(route)
        L0 = float(dbar[r].sum())
        d_real = dsc_te[day][r]
        p_real = psc_te[day][r]
        sim = simulate_day("v2", len(r), B, Q, L0, d_real, p_real, H, E, cm=cm)
        sims.append(sim)
        frame_sets.append(build_frames(G, osm_ids, route, sim,
                                       frames_per_leg=4, hold=1))
    n_frames = max(len(f) for f in frame_sets)
    for f in frame_sets:
        f.extend([f[-1]] * (n_frames - len(f)))

    fig, ax = plt.subplots(figsize=(14, 14))
    all_c = [c for route in plan for c in route]
    xs = [lon[c] for c in all_c]; ys = [lat[c] for c in all_c]
    pad = 0.003
    lonr = (min(xs + [lon[0]]) - pad, max(xs + [lon[0]]) + pad)
    latr = (min(ys + [lat[0]]) - pad, max(ys + [lat[0]]) + pad)
    _draw_static(ax, G, lonr, latr, xs, ys, lon[0], lat[0])

    n_ho = sum(1 for s in sims if s["switch_stop"])
    n_br = sum(1 for s in sims if s["breach_stop"])
    fleet_cost = sum(s["cost"] for s in sims)
    ax.set_title(f"{instance_stem} — {len(plan)} vehicles under BATON, one "
                 f"high-demand day: {n_ho} handoffs, {n_br} breaches, "
                 f"fleet recourse ${fleet_cost:.0f}", fontsize=14, color=INK)

    dots, trails = [], []
    for ri in range(len(plan)):
        col = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
        d, = ax.plot([], [], marker="o", ms=11, color=col, zorder=8,
                     mec="white", mew=1.1)
        dots.append(d)
        trails.append({})

    def update(fi):
        artists = []
        for ri, frames in enumerate(frame_sets):
            fr = frames[fi]
            base = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
            col = base if fr["veh"] == 1 else \
                (VEH3 if fr["veh"] == 3 else base)
            dots[ri].set_data([fr["x"]], [fr["y"]])
            dots[ri].set_color(col)
            tr = trails[ri].get(fr["veh"])
            if tr is None:
                ls = "-" if fr["veh"] == 1 else "--"
                tcol = base if fr["veh"] != 3 else VEH3
                tr, = ax.plot([], [], color=tcol, lw=2.0, ls=ls,
                              alpha=0.8, zorder=5, solid_capstyle="round")
                trails[ri][fr["veh"]] = tr
            tx, ty = tr.get_data()
            tr.set_data(np.append(tx, fr["x"]), np.append(ty, fr["y"]))
            if fr["event"] == "handoff":
                ax.scatter([fr["x"]], [fr["y"]], s=170, facecolors="none",
                           edgecolors=base, linewidths=2.2, zorder=7)
            if fr["event"] == "breach":
                ax.scatter([fr["x"]], [fr["y"]], marker="X", s=150,
                           color=VEH3, zorder=7)
            artists += [dots[ri], tr]
        return artists

    legend = [Line2D([], [], color=INK, lw=2, label="planned vehicle (trail)"),
              Line2D([], [], color=INK, lw=2, ls="--",
                     label="standby vehicle after handoff (dashed trail)"),
              Line2D([], [], marker="o", color="white", markerfacecolor="none",
                     markeredgecolor=INK, lw=0, ms=11, label="handoff point"),
              Line2D([], [], marker="X", color=VEH3, lw=0, ms=11,
                     label="capacity breach"),
              Line2D([], [], marker="*", color=INK, lw=0, ms=15,
                     label="depot")]
    ax.legend(handles=legend, frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()

    if still:
        for fi in range(n_frames):
            update(fi)
        for d in dots:            # vehicles end parked at the depot
            d.set_visible(False)
        out = Path(out_dir) / f"fig6_fleet_{instance_stem}.png"
        fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}  (static, {len(plan)} vehicles)")
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out = Path(out_dir) / f"anim_{instance_stem}_fleet_s{scen_rank}.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"wrote {out}  ({n_frames} frames, {len(plan)} vehicles)")
    return out


def main():
    kw = dict(instance_stem="HANOI-100-1", policy="compare", scen_rank=0,
              fps=7)
    still = False
    for a in sys.argv[1:]:
        if   a.startswith("instance="): kw["instance_stem"] = Path(a[9:]).stem
        elif a.startswith("policy="):   kw["policy"] = a[7:]
        elif a.startswith("scenario="): kw["scen_rank"] = int(a[9:])
        elif a.startswith("fps="):      kw["fps"] = int(a[4:])
        elif a == "still":              still = True
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if kw["policy"] == "fleet":
        animate_fleet(instance_stem=kw["instance_stem"],
                      scen_rank=kw["scen_rank"], fps=kw["fps"], still=still)
    else:
        animate(**kw)


if __name__ == "__main__":
    main()
