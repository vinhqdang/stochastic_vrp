#!/usr/bin/env python3
"""ev_animate_fleet.py — the TEMPO fleet animation on HANOI-200-1.

Thirteen vehicles drive their planned routes simultaneously on the real
street network while the day's stochastic processes run visibly:

  * the network breathes with the clock (diurnal rush-hour tint) and a
    SPATIAL jam pocket opens at 09:00 and spreads — street segments
    inside the growing zone turn red, the rest of the city stays on
    its forecast;
  * accident events flash where they happen; vehicles crawl inside the
    jam; every stop realizes its demand and dwell;
  * one pooled TEMPO e-process monitors the WHOLE plan (evidence from
    all vehicles feeds one martingale — the dispatcher watches the
    plan, not a vehicle); its log e-value climbs to the alarm;
  * a raw-residuals strip shows the per-event standardized noise the
    monitor is betting on — the stochastic process itself;
  * at the alarm every vehicle's remaining stops are re-sequenced
    (nearest-neighbour stand-in for the warm-started ALNS replan) and
    trails switch to dashed violet.

Usage (from svrpspd_wdro/): python scripts/ev_animate_fleet.py [fps=8]
Output: results/figures/anim_tempo_fleet.gif
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

_SCRIPTS = Path(__file__).resolve().parent
_WDRO = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from make_figures import (C, INK, MUTED, STREET, STATUS_CRITICAL,   # noqa: E402
                          ROUTE_COLORS, FIG_DIR, _load_graph, _street_path)
from animate_execution import _resample                             # noqa: E402
from dethloff_runner import parse_dethloff                          # noqa: E402
from ev.world import (DayParams, DriftSpec, simulate_route_day,     # noqa: E402
                      params_from_route, diurnal_mult)
from ev.eprocess import MasterEProcess                              # noqa: E402

NAME = "HANOI-200-1"
ALPHA = 0.05
T_STAR = 9.0
JAM_MAG = 2.2
DT = 2.5 / 60.0          # frame step: 2.5 simulated minutes
CH_COLOR = {"travel": C["blue"], "demand": C["aqua"], "dwell": C["yellow"],
            "accident": C["violet"], "breakdown": STATUS_CRITICAL}


def _nn(remaining, cur_xy, xy):
    rem = list(remaining)
    seq = []
    cx, cy = cur_xy
    while rem:
        nxt = min(rem, key=lambda c: (xy[c][0] - cx) ** 2
                  + (xy[c][1] - cy) ** 2)
        seq.append(nxt)
        cx, cy = xy[nxt]
        rem.remove(nxt)
    return seq


def main(fps=8):
    doc = json.loads((_WDRO / "results" / "plans" / f"{NAME}.json")
                     .read_text())
    D, dem, Q, n, scale = parse_dethloff(
        str(_WDRO / "data" / "City" / f"{NAME}.vrpspd"))
    coords = json.loads((_WDRO / "data" / "City" / f"{NAME}.coords.json")
                        .read_text())["nodes"]
    routes = [r for r in doc["res"]["Det"]["plan"] if r]
    G = _load_graph("hanoi")
    osm = [c[0] for c in coords]
    xy = [(c[2], c[1]) for c in coords]          # (lon, lat)

    # jam pocket centred on a customer in the mid-west of the map
    all_c = [c for r in routes for c in r]
    lons = np.array([xy[c][0] for c in all_c])
    lats = np.array([xy[c][1] for c in all_c])
    cen = (float(np.quantile(lons, 0.50)), float(np.quantile(lats, 0.50)))
    drift = DriftSpec(kind="traffic_zone", t_star=T_STAR,
                      magnitude=JAM_MAG, center=cen,
                      radius0=0.012, spread=0.015)

    # simulate every route-day under the SHARED zone drift
    sims = []
    for vi, route in enumerate(routes):
        p = params_from_route(route, D, dem, Q, scale,
                              stop_xy=np.array([xy[c] for c in route]))
        p.tau = p.tau * (5.5 / max(p.tau.sum(), 1e-9))   # a working day
        events, _ = simulate_route_day(
            p, drift, np.random.default_rng(4200 + vi))
        sims.append((route, p, events))

    # pooled TEMPO: all events across the fleet, in clock order
    pooled = sorted(
        ((ev["clock"], vi, ev) for vi, (_, _, evs) in enumerate(sims)
         for ev in evs), key=lambda t: t[0])
    # breakdown runs as a PARALLEL e-process with a union-bound alpha
    # split (alpha/2 each): its rare-event stake would otherwise bleed
    # ~n_vehicles x n_legs premiums out of the pooled monitor
    # (PROJECT.md §9); no breakdown realizes on this day, so the
    # displayed gauge is the continuous-channel master.
    mon = MasterEProcess(alpha=ALPHA / 2)
    mon_brk = MasterEProcess(alpha=ALPHA / 2)
    e_clock, e_log, alarm_clock = [], [], None
    resid = []                                   # (clock, z, channel)
    acc_flash = []                               # (clock, vi)
    for ck, vi, ev in pooled:
        if ev["channel"] == "breakdown":
            if mon_brk.update(ev) and alarm_clock is None:
                alarm_clock = ck
            continue
        fired = mon.update(ev)
        e_clock.append(ck)
        e_log.append(mon.logE)
        if "z" in ev:
            resid.append((ck, ev["z"], ev["channel"]))
        elif ev["channel"] == "accident" and ev["n"] >= 1:
            resid.append((ck, (ev["n"] - ev["lam0dt"])
                          / np.sqrt(max(ev["lam0dt"], 1e-9)), "accident"))
            acc_flash.append((ck, vi))
        if fired and alarm_clock is None:
            alarm_clock = ck
    e_clock = np.array(e_clock)
    e_log = np.array(e_log)
    print(f"pooled alarm at clock {alarm_clock} "
          f"({len(pooled)} events, {len(routes)} vehicles, "
          f"final logE {mon.logE:.1f})", flush=True)

    # per-vehicle motion timeline: (t0, t1, pts) segments + stop holds;
    # after the alarm, remaining stops are NN-resequenced
    def leg_pts(a, b):
        try:
            return _resample(_street_path(G, osm[a], osm[b]), 24)
        except Exception:
            return np.linspace([xy[a][0], xy[a][1]],
                               [xy[b][0], xy[b][1]], 24)

    timelines = []                               # per vehicle
    for vi, (route, p, events) in enumerate(sims):
        T_real = [e["val"] for e in events if e["channel"] == "travel"]
        S_real = [e["val"] for e in events if e["channel"] == "dwell"]
        segs, clock, cur = [], p.t0, 0
        replanned = False
        k = 0
        stops = list(route)
        while k < len(stops):
            nxt = stops[k]
            if alarm_clock and clock >= alarm_clock and not replanned:
                stops = stops[:k] + _nn(stops[k:], xy[cur], xy)
                nxt = stops[k]
                replanned = True
            if k < len(T_real):
                T = T_real[k] if not replanned else \
                    p.tau[min(k, len(p.tau) - 1)] * \
                    diurnal_mult(clock, p.diurnal) * \
                    (JAM_MAG if drift.in_zone(xy[nxt], clock) else 1.0)
                S = S_real[min(k, len(S_real) - 1)]
            else:
                T, S = 0.15, 0.05
            segs.append((clock, clock + T, leg_pts(cur, nxt),
                         replanned))
            clock += T + S
            cur = nxt
            k += 1
        timelines.append(segs)
    t_end = min(max(s[-1][1] for s in timelines), 15.0)

    def pos_at(vi, t):
        segs = timelines[vi]
        if t < segs[0][0]:
            return (*segs[0][2][0], "wait", False)
        for t0, t1, pts, rp in segs:
            if t <= t1:
                if t < t0:
                    return (*pts[0], "dwell", rp)
                u = (t - t0) / max(t1 - t0, 1e-9)
                i = min(int(u * (len(pts) - 1)), len(pts) - 1)
                return (*pts[i], "move", rp)
        return (*segs[-1][2][-1], "done", segs[-1][3])

    # figure
    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.62, 1.0],
                          height_ratios=[1.15, 0.85, 0.72],
                          wspace=0.14, hspace=0.42)
    axM = fig.add_subplot(gs[:, 0])
    axE = fig.add_subplot(gs[0, 1])
    axZ = fig.add_subplot(gs[1, 1])
    axL = fig.add_subplot(gs[2, 1])

    # map: streets as a LineCollection with per-segment colors
    edges = list(G.edges())
    segs_xy = np.array([[[G.nodes[u]["x"], G.nodes[u]["y"]],
                         [G.nodes[v]["x"], G.nodes[v]["y"]]]
                        for u, v in edges])
    mids = segs_xy.mean(axis=1)
    streets = LineCollection(segs_xy, linewidths=0.55, zorder=1)
    axM.add_collection(streets)
    zone = Circle(cen, 0.0, fill=False, ls="--", lw=1.6,
                  ec=STATUS_CRITICAL, zorder=2)
    axM.add_patch(zone)
    axM.scatter(lons, lats, s=13, facecolors="white", edgecolors=MUTED,
                linewidths=0.6, zorder=4)
    axM.scatter([xy[0][0]], [xy[0][1]], marker="*", s=340, color=INK,
                zorder=6, edgecolors="white", linewidths=1.0)
    pad = 0.004
    axM.set_xlim(lons.min() - pad, lons.max() + pad)
    axM.set_ylim(lats.min() - pad, lats.max() + pad)
    axM.set_aspect(1.0 / np.cos(np.deg2rad(lats.mean())))
    axM.set_xticks([]); axM.set_yticks([])
    for s_ in axM.spines.values():
        s_.set_visible(False)
    axM.set_title(f"{NAME} — {len(routes)} vehicles; a jam pocket opens "
                  "at 09:00 and spreads", fontsize=12.5, loc="left")
    clock_txt = axM.text(0.02, 0.975, "", transform=axM.transAxes,
                         fontsize=16, family="monospace", va="top",
                         color=INK)
    traf_txt = axM.text(0.02, 0.94, "", transform=axM.transAxes,
                        fontsize=10.5, va="top", color=INK)
    banner = axM.text(0.02, 0.015, "", transform=axM.transAxes,
                      fontsize=11.5, va="bottom", color=INK)

    vdots, vtrails_solid, vtrails_replan = [], [], []
    trail_hist = []
    for vi in range(len(routes)):
        col = ROUTE_COLORS[vi % len(ROUTE_COLORS)]
        d, = axM.plot([], [], marker="o", ms=9.5, color=col, zorder=8,
                      mec="white", mew=1.0)
        s, = axM.plot([], [], color=col, lw=1.7, alpha=0.75, zorder=3,
                      solid_capstyle="round")
        r, = axM.plot([], [], color=C["violet"], lw=1.7, ls="--",
                      alpha=0.85, zorder=3)
        vdots.append(d); vtrails_solid.append(s); vtrails_replan.append(r)
        trail_hist.append({"s": [[], []], "r": [[], []]})
    flashes = axM.scatter([], [], marker="X", s=150,
                          color=STATUS_CRITICAL, zorder=9)

    # TEMPO gauge (pooled)
    thr = np.log(2 / ALPHA)      # alpha/2: breakdown holds the other half
    axE.set_xlim(8.0, t_end)
    axE.set_ylim(min(e_log.min(), -2) - 0.5, 12.0)   # clip: the crossing
    e_log = np.minimum(e_log, 11.5)                  # is the story, not
    axE.text(t_end - 0.1, 11.0, "clipped", ha="right",  # the stratosphere
             fontsize=7.5, color=MUTED)
    axE.axhline(thr, color=STATUS_CRITICAL, ls="--", lw=1.3)
    axE.text(8.05, thr + 0.3, r"alarm $\log(1/\alpha)$",
             color=STATUS_CRITICAL, fontsize=8.5)
    if alarm_clock:
        axE.axvline(T_STAR, color=INK, ls=":", lw=1.1)
    eline, = axE.plot([], [], color=INK, lw=2.0)
    emark, = axE.plot([], [], marker="X", ms=12, color=STATUS_CRITICAL,
                      ls="none", zorder=6)
    axE.set_title("TEMPO — one e-process for the whole fleet",
                  fontsize=10.5, loc="left")
    axE.set_ylabel(r"$\log E_t$")
    axE.tick_params(labelsize=8)

    # raw stochastic-process strip: standardized residuals per event
    axZ.set_xlim(8.0, t_end)
    axZ.set_ylim(-4, 6)
    axZ.axhline(0, color=MUTED, lw=0.9)
    axZ.axvline(T_STAR, color=INK, ls=":", lw=1.1)
    zscat = axZ.scatter([], [], s=9, alpha=0.6)
    axZ.set_title("the raw noise TEMPO bets on — standardized residuals\n"
                  "(blue travel, teal demand, yellow dwell, violet accident)",
                  fontsize=9.5, loc="left")
    axZ.set_ylabel("z")
    axZ.tick_params(labelsize=8)

    # fleet progress
    axL.set_xlim(8.0, t_end)
    axL.set_ylim(0, sum(len(r) for r in routes) * 1.05)
    served_line, = axL.plot([], [], color=C["green"], lw=2.0)
    axL.axvline(T_STAR, color=INK, ls=":", lw=1.1)
    axL.set_title("customers served", fontsize=10.5, loc="left")
    axL.set_xlabel("clock (h)")
    axL.tick_params(labelsize=8)
    stop_times = sorted(t1 for segs in timelines for _, t1, _, _ in segs)

    # per-road stochastic congestion: every edge carries (i) a static
    # severity (some roads are just slower), (ii) its own jam response
    # exponent, (iii) an AR(1) log-noise state with ~temporal
    # persistence — so no two roads ever share a traffic level, and a
    # road's level this minute predicts its level the next.
    rng_f = np.random.default_rng(77)
    sev = np.exp(rng_f.normal(0.0, 0.15, len(mids)))
    jam_w = rng_f.uniform(0.70, 1.35, len(mids))
    field = {"x": rng_f.normal(0.0, 0.15, len(mids))}
    RHO = 0.90

    base = np.array(matplotlib.colors.to_rgb(STREET))
    amber = np.array(matplotlib.colors.to_rgb("#eda100"))
    red = np.array(matplotlib.colors.to_rgb("#c62f2e"))

    def blend(x):
        x = np.clip(x, 0.0, 1.0)[:, None]
        c1 = base + (amber - base) * np.minimum(x, .5) * 2 * .6
        c2 = (base + (amber - base) * .6) + \
            (red - (base + (amber - base) * .6)) * np.maximum(x - .5, 0) * 2
        return np.where(x < .5, c1, c2)

    ts = np.arange(8.0, t_end, DT)

    def update(fi):
        t = ts[fi]
        fc = diurnal_mult(t, DayParams(np.zeros(1), np.zeros(1),
                                       np.ones(1), 1.0).diurnal)
        r_t = drift.zone_radius(t) if t >= T_STAR else 0.0
        inz = (np.hypot(mids[:, 0] - cen[0], mids[:, 1] - cen[1]) <= r_t)
        if fi % 2 == 0:                       # traffic states persist ~5 min
            field["x"] = RHO * field["x"] + np.sqrt(1 - RHO ** 2) * \
                rng_f.normal(0.0, 0.15, len(mids))
        mult = fc * sev * np.exp(field["x"])
        mult[inz] *= JAM_MAG ** jam_w[inz]
        streets.set_color(blend((mult - 1.0) / 1.5))
        zone.set_radius(r_t)
        hh, mm = int(t) % 24, int((t % 1) * 60)
        clock_txt.set_text(f"{hh:02d}:{mm:02d}")
        amax = fc * (JAM_MAG if r_t > 0 else 1.0)
        traf_txt.set_text(
            f"forecast {fc:.1f}x — jam zone {amax:.1f}x"
            + ("  ← UNEXPECTED" if r_t > 0 else "   (none)"))
        traf_txt.set_color(STATUS_CRITICAL if r_t > 0 else INK)

        fl_x, fl_y = [], []
        for ck, vi in acc_flash:
            if 0 <= t - ck <= 4 * DT:
                px, py, _, _ = pos_at(vi, ck)
                fl_x.append(px); fl_y.append(py)
        flashes.set_offsets(np.c_[fl_x, fl_y] if fl_x else
                            np.empty((0, 2)))

        for vi in range(len(routes)):
            px, py, mode, rp = pos_at(vi, t)
            vdots[vi].set_data([px], [py])
            key = "r" if rp else "s"
            trail_hist[vi][key][0].append(px)
            trail_hist[vi][key][1].append(py)
            vtrails_solid[vi].set_data(*trail_hist[vi]["s"])
            vtrails_replan[vi].set_data(*trail_hist[vi]["r"])

        mask = e_clock <= t
        eline.set_data(e_clock[mask], e_log[mask])
        if alarm_clock and t >= alarm_clock:
            emark.set_data([alarm_clock],
                           [e_log[np.searchsorted(e_clock, alarm_clock)]])
        pts = [(ck, z, CH_COLOR.get(ch, MUTED))
               for ck, z, ch in resid if ck <= t]
        if pts:
            zscat.set_offsets(np.array([(a, b) for a, b, _ in pts]))
            zscat.set_color([c for _, _, c in pts])
        nsrv = np.searchsorted(stop_times, t)
        served_line.set_data([8.0, t] if nsrv == 0 else
                             np.array(stop_times[:nsrv]).repeat(1),
                             np.arange(1, nsrv + 1) if nsrv else [0, 0])
        if t < T_STAR:
            banner.set_text("morning peak as forecast — no evidence, "
                            "no alarm")
            banner.set_color(INK)
        elif alarm_clock is None or t < alarm_clock:
            banner.set_text("jam pocket spreading — evidence pooling "
                            "across the fleet…")
            banner.set_color("#b97e0a")
        else:
            banner.set_text("TEMPO alarm → fleet replanned (dashed); "
                            "false-alarm risk ≤ 5% for the whole day")
            banner.set_color(STATUS_CRITICAL)
        return []

    legend = [Line2D([], [], color=MUTED, lw=2, label="planned trail"),
              Line2D([], [], color=C["violet"], lw=2, ls="--",
                     label="after TEMPO replan"),
              Line2D([], [], marker="X", color=STATUS_CRITICAL, lw=0,
                     ms=10, label="accident"),
              Line2D([], [], color=STATUS_CRITICAL, lw=1.6, ls="--",
                     label="jam zone (spreading)"),
              Line2D([], [], marker="*", color=INK, lw=0, ms=13,
                     label="depot")]
    axM.legend(handles=legend, frameon=False, fontsize=8,
               loc="upper right")
    fig.suptitle("TEMPO on a fleet — thirteen vehicles, one anytime-valid "
                 "monitor: the network breathes, a jam pocket spreads, "
                 "evidence pools, the plan is rescued",
                 fontsize=12.5)

    anim = FuncAnimation(fig, update, frames=len(ts), blit=False)
    out = FIG_DIR / "anim_tempo_fleet.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=95)
    plt.close(fig)
    print(f"wrote {out}  ({len(ts)} frames)", flush=True)


if __name__ == "__main__":
    fps = 8
    for a in sys.argv[1:]:
        if a.startswith("fps="):
            fps = int(a[4:])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main(fps=fps)
