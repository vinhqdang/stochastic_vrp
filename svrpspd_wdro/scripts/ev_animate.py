#!/usr/bin/env python3
"""ev_animate.py — animated GIF of TEMPO working a traffic-jam day on
the real Hanoi street network, side by side with what BATON sees.

Layout: big map panel (vehicle drives real street paths; trail blue
while the day is on-model, amber once the unannounced jam starts,
violet dashed after TEMPO's replan) + two live gauges:

  top-right     TEMPO's log E_t climbing to the alarm threshold
  bottom-right  BATON's view — the onboard load vs the boundary B,
                which stays calm all day (a jam does not touch the
                load), so BATON never acts. Not a flaw: out of scope.

After the alarm the remaining stops are re-sequenced (nearest-neighbour
from the alarm stop — a stand-in for the warm-started ALNS replan) and
the vehicle follows the new sequence.

INTERNAL visual (BATON under review; not for the TEMPO manuscript).

Usage (from svrpspd_wdro/): python scripts/ev_animate.py [fps=8]
Output: results/figures/anim_tempo_vs_baton.gif
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

_SCRIPTS = Path(__file__).resolve().parent
_WDRO = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from make_figures import (C, INK, MUTED, STREET, STATUS_CRITICAL,  # noqa: E402
                          FIG_DIR, CITY_DIR, _load_graph, _street_path)
from animate_execution import _resample                            # noqa: E402
from ev_figures import (_setup_didactic, _pick_traffic_day,        # noqa: E402
                        _run_tempo, ALPHA)

AMBER = C["yellow"]


def _nn_reorder(remaining, start_node, lon, lat):
    """Nearest-neighbour re-sequencing of the remaining stops — the
    animation's stand-in for the warm-started ALNS replan."""
    rem = list(remaining)
    seq = []
    cur = start_node
    while rem:
        nxt = min(rem, key=lambda c: (lon[c] - lon[cur]) ** 2
                  + (lat[c] - lat[cur]) ** 2)
        seq.append(nxt)
        rem.remove(nxt)
        cur = nxt
    return seq


def main(fps=8):
    p, route, m, H, cm = _setup_didactic()
    picked = _pick_traffic_day(p, cm, H)
    assert picked, "no didactic traffic day"
    _, events, g_real, alarm = picked
    logE, _, _ = _run_tempo(events)
    k_alarm = alarm // 5 + 1
    k_drift = next(i for i, e in enumerate(events)
                   if e["ctx"]["drifted"]) // 5 + 1
    W = np.cumsum(g_real)

    coords = json.loads((CITY_DIR / "HANOI-100-1.coords.json")
                        .read_text())["nodes"]
    G = _load_graph("hanoi")
    osm = [c[0] for c in coords]
    lon = [c[2] for c in coords]
    lat = [c[1] for c in coords]

    # visit order: planned until the alarm stop, then NN-replanned
    pre = list(route[:k_alarm])
    post = _nn_reorder(route[k_alarm:], pre[-1], lon, lat)
    visit = pre + post

    # realized leg/dwell durations from the simulated day drive the clock
    from ev.world import diurnal_mult
    T_real = [e["val"] for e in events if e["channel"] == "travel"]
    S_real = [e["val"] for e in events if e["channel"] == "dwell"]
    t_star = next(e["clock"] for e in events if e["ctx"]["drifted"])
    jam_mult = 2.0                          # the drift magnitude used

    # frames along real street paths; slower frames inside the jam;
    # every frame knows the wall clock and the actual vs forecast
    # network multiplier so the MAP ITSELF changes with time
    frames = []
    seq = [0] + visit
    clock = p.t0
    for i, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
        k = i + 1
        npts = 6 if k < k_drift else 9      # jammed legs crawl
        try:
            pts = _resample(_street_path(G, osm[a], osm[b]), npts)
        except Exception:
            pts = np.linspace([lon[a], lat[a]], [lon[b], lat[b]], npts)
        phase = ("plan" if k < k_drift else
                 "jam" if k <= k_alarm else "replan")
        T = T_real[min(k - 1, len(T_real) - 1)]
        S = S_real[min(k - 1, len(S_real) - 1)]
        for j, (x, y) in enumerate(pts):
            tj = clock + T * (j + 1) / npts
            fc = diurnal_mult(tj, p.diurnal)
            ac = fc * (jam_mult if tj >= t_star else 1.0)
            frames.append(dict(x=x, y=y, k=k, phase=phase, clock=tj,
                               fmult=fc, amult=ac,
                               arrive=(j == npts - 1)))
        clock += T + S
        frames += [frames[-1]] * 2          # hold at the stop

    # figure
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0],
                          height_ratios=[1.0, 1.0], wspace=0.16,
                          hspace=0.28)
    axM = fig.add_subplot(gs[:, 0])
    axE = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1])

    segs = [[(G.nodes[u]["x"], G.nodes[u]["y"]),
             (G.nodes[v]["x"], G.nodes[v]["y"])] for u, v in G.edges()]
    streets = LineCollection(segs, colors=STREET, linewidths=0.55,
                             zorder=1)
    axM.add_collection(streets)
    xs = [lon[c] for c in visit] + [lon[0]]
    ys = [lat[c] for c in visit] + [lat[0]]
    pad = 0.004
    axM.set_xlim(min(xs) - pad, max(xs) + pad)
    axM.set_ylim(min(ys) - pad, max(ys) + pad)
    axM.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(ys))))
    axM.set_xticks([]); axM.set_yticks([])
    for s_ in axM.spines.values():
        s_.set_visible(False)
    axM.scatter([lon[c] for c in route], [lat[c] for c in route],
                s=42, facecolors="white", edgecolors=MUTED,
                linewidths=1.1, zorder=4)
    axM.scatter([lon[0]], [lat[0]], marker="*", s=360, color=INK,
                zorder=5, edgecolors="white", linewidths=1.0)
    axM.set_title("HANOI-100-1 — the network breathes with the clock",
                  fontsize=12, color=INK, loc="left")

    trail = {"plan": axM.plot([], [], color=C["blue"], lw=2.4,
                              zorder=3, solid_capstyle="round")[0],
             "jam": axM.plot([], [], color=AMBER, lw=2.8, zorder=3,
                             solid_capstyle="round")[0],
             "replan": axM.plot([], [], color=C["violet"], lw=2.4,
                                ls="--", zorder=3)[0]}
    veh, = axM.plot([], [], marker="o", ms=13, color=C["blue"],
                    mec="white", mew=1.2, zorder=8)
    banner = axM.text(0.02, 0.02, "", transform=axM.transAxes,
                      fontsize=11.5, color=INK, va="bottom")
    clock_txt = axM.text(0.02, 0.975, "", transform=axM.transAxes,
                         fontsize=15, color=INK, va="top",
                         family="monospace")
    traf_txt = axM.text(0.02, 0.935, "", transform=axM.transAxes,
                        fontsize=10.5, color=INK, va="top")

    def street_color(actual):
        """Grey (free) -> amber (busy) -> red (jammed)."""
        x = min(max((actual - 1.0) / 1.5, 0.0), 1.0)
        base = np.array([0.753, 0.749, 0.714])       # STREET
        amber = np.array([0.929, 0.631, 0.000])
        red = np.array([0.776, 0.184, 0.180])
        return tuple(base + (amber - base) * min(x, .5) * 2 * .55
                     if x < .5 else
                     base * 0 + amber + (red - amber) * (x - .5) * 2) \
            if x > 0 else tuple(base)

    # TEMPO gauge
    thr = np.log(1 / ALPHA)
    n_ev = len(logE)
    axE.set_xlim(0, n_ev)
    axE.set_ylim(min(logE.min(), -1) - 0.5, max(logE.max(), thr) + 1.5)
    axE.axhline(thr, color=STATUS_CRITICAL, ls="--", lw=1.4)
    axE.text(n_ev * 0.02, thr + 0.25, r"alarm threshold $\log(1/\alpha)$",
             color=STATUS_CRITICAL, fontsize=9)
    eline, = axE.plot([], [], color=INK, lw=2.2)
    emark, = axE.plot([], [], marker="X", ms=13, color=STATUS_CRITICAL,
                      ls="none", zorder=6)
    axE.set_title("TEMPO — evidence the day is off-model", fontsize=11,
                  loc="left")
    axE.set_ylabel(r"$\log E_t$")
    axE.set_xlabel("events (5 per stop)")

    # BATON gauge
    axB.set_xlim(0.5, m + 0.5)
    lo = min(W.min(), 0) - 2
    hi = max(W.max(), p.B) + 3
    axB.set_ylim(lo, hi)
    axB.axhline(p.B, color=STATUS_CRITICAL, ls="--", lw=1.4)
    axB.text(m * 0.55, p.B + 0.02 * (hi - lo), "recourse boundary $B$",
             color=STATUS_CRITICAL, fontsize=9)
    wline, = axB.plot([], [], color=C["aqua"], lw=2.2)
    axB.set_title("BATON — the load it watches stays normal:\n"
                  "a jam is outside its state (by design)",
                  fontsize=11, loc="left")
    axB.set_ylabel("onboard net load $W_k$")
    axB.set_xlabel("stop along route")

    fig.suptitle("TEMPO vs BATON, one bad day — BATON prices per-route "
                 "recourse from the load; TEMPO tests the whole plan "
                 "and calls the replan", fontsize=12.5)

    trail_pts = {"plan": [[], []], "jam": [[], []], "replan": [[], []]}
    state = {"k_done": 0}

    def update(fi):
        fr = frames[fi]
        streets.set_color([street_color(fr["amult"])])
        hh = int(fr["clock"]) % 24
        mm = int((fr["clock"] % 1) * 60)
        clock_txt.set_text(f"{hh:02d}:{mm:02d}")
        gap = fr["amult"] / max(fr["fmult"], 1e-9)
        traf_txt.set_text(
            f"traffic {fr['amult']:.1f}x — forecast {fr['fmult']:.1f}x"
            + ("  (as expected)" if gap < 1.05 else "  ← UNEXPECTED"))
        traf_txt.set_color(INK if gap < 1.05 else STATUS_CRITICAL)
        trail_pts[fr["phase"]][0].append(fr["x"])
        trail_pts[fr["phase"]][1].append(fr["y"])
        for ph, ln in trail.items():
            ln.set_data(*trail_pts[ph])
        veh.set_data([fr["x"]], [fr["y"]])
        veh.set_color(C["blue"] if fr["phase"] == "plan" else
                      AMBER if fr["phase"] == "jam" else C["violet"])
        if fr["arrive"]:
            state["k_done"] = fr["k"]
        kd = state["k_done"]
        ne = min(kd * 5, n_ev)
        eline.set_data(np.arange(ne), logE[:ne])
        if alarm is not None and ne > alarm:
            emark.set_data([alarm], [logE[alarm]])
        wline.set_data(np.arange(1, min(kd, m) + 1), W[:min(kd, m)])
        if fr["phase"] == "plan":
            banner.set_text("on-model driving — plan trusted")
            banner.set_color(INK)
        elif fr["phase"] == "jam" and (alarm is None or kd * 5 <= alarm):
            banner.set_text("jam running — evidence accumulating…")
            banner.set_color(AMBER)
        else:
            banner.set_text("TEMPO alarm → rest of day re-planned (BATON idle)")
            banner.set_color(STATUS_CRITICAL)
        return [veh, eline, emark, wline, banner, clock_txt, traf_txt,
                streets, *trail.values()]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = FIG_DIR / "anim_tempo_vs_baton.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames, alarm at stop {k_alarm}, "
          f"jam from stop {k_drift})", flush=True)


if __name__ == "__main__":
    fps = 8
    for a in sys.argv[1:]:
        if a.startswith("fps="):
            fps = int(a[4:])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main(fps=fps)
