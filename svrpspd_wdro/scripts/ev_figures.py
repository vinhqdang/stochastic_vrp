#!/usr/bin/env python3
"""ev_figures.py — explanatory visualizations for TEMPO (paper 2), in
the same visual language as the BATON figures. INTERNAL comparison
figures: BATON is under review, so these do NOT go into the TEMPO
manuscript — they exist so we can see how the two relate.

Outputs (results/figures/):
  tempo_fig1_explainer.png   how TEMPO works: e-process trajectories
                             over many simulated days (null vs drift),
                             one day's per-channel evidence, and the
                             physical signal behind it.
  tempo_fig2_vs_baton.png    TEMPO vs BATON on the same two days of the
                             same Hanoi route: a demand-surge day (both
                             react) and a traffic-jam day (BATON is
                             structurally blind — its state is the
                             onboard load — while TEMPO fires).
  tempo_fig3_map.png         the traffic-jam day on the Hanoi street
                             network: where drift starts, how far the
                             route runs on a stale plan, where TEMPO
                             calls the replan.

Usage (from svrpspd_wdro/): python scripts/ev_figures.py [1|2|3|all]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPTS = Path(__file__).resolve().parent
_WDRO = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from make_figures import (C, INK, MUTED, STREET, STATUS_CRITICAL,   # noqa: E402
                          FIG_DIR, CITY_DIR, _load_graph, _street_path,
                          _pick_route_and_scenarios)
from dethloff_runner import parse_dethloff                          # noqa: E402
from core.costs import (LastMileCosts, route_cost_schedules,        # noqa: E402
                        fit_lsm_general)
from ev.world import DriftSpec, simulate_route_day, params_from_route  # noqa: E402
from ev.eprocess import MasterEProcess                              # noqa: E402

ALPHA = 0.05
NAME = "HANOI-100-1"
STOP_CH = ("travel", "accident", "breakdown", "demand", "dwell")  # per-stop order


# ── shared setup: route, planning model, BATON fit ───────────────────────────

def _setup():
    doc = json.loads((_WDRO / "results" / "plans" / f"{NAME}.json")
                     .read_text())
    D, dem, Q, n, scale = parse_dethloff(str(CITY_DIR / f"{NAME}.vrpspd"))
    route = max(doc["res"]["Det"]["plan"], key=len)
    p = params_from_route(route, D, dem, Q, scale)
    p.tau = p.tau * (5.0 / max(p.tau.sum(), 1e-9))   # a ~5 h driving day
    m = len(route)

    rng = np.random.default_rng(20260712)
    g_tr = rng.normal(p.mu_g, p.sig_g, size=(1000, m))
    H, E = route_cost_schedules(route, D, scale, LastMileCosts())
    cm = fit_lsm_general(g_tr, p.B, H, E)
    return p, route, D, scale, m, H, E, cm


def _baton_trigger(g_real, B, H, cm):
    """Mirror of the fig2/fig23 first_trigger: BATON hands off at the
    first stop where the fitted continuation cost exceeds the handoff
    price; returns None if it breaches first or never triggers."""
    W = np.cumsum(g_real)
    m = len(g_real)
    for k in range(1, m):
        if W[k - 1] > B:
            return None
        if cm[k].predict(np.array([W[k - 1]]))[0] > H[k - 1]:
            return k
    return None


def _run_tempo(events):
    """Feed events; return per-event logE array, per-channel cumulative
    log traces, and the first alarm event index (None if silent)."""
    mon = MasterEProcess(alpha=ALPHA)
    logE, alarm = [], None
    by_ch = {c: [] for c in mon.CHANNELS}
    for i, ev in enumerate(events):
        fired = mon.update(ev)
        logE.append(mon.logE)
        for c in mon.CHANNELS:
            by_ch[c].append(mon.log_by_channel[c])
        if fired and alarm is None:
            alarm = i
    return np.array(logE), {c: np.array(v) for c, v in by_ch.items()}, alarm


def _day(p, kind, mag, seed, t_off=1.0):
    drift = DriftSpec(kind=kind, t_star=p.t0 + t_off, magnitude=mag)
    events, meta = simulate_route_day(p, drift,
                                      np.random.default_rng(seed))
    return events, meta, drift


def _g_real(events, p):
    z = [e["z"] for e in events if e["channel"] == "demand"]
    return np.array(z) * p.sig_g + p.mu_g


def _first_drift_idx(events):
    return next((i for i, e in enumerate(events) if e["ctx"]["drifted"]),
                None)


CH_COLOR = {"travel": C["blue"], "demand": C["aqua"], "dwell": C["yellow"],
            "accident": C["violet"], "breakdown": STATUS_CRITICAL}


# ── figure 1: how TEMPO works ────────────────────────────────────────────────

def fig1():
    p, route, D, scale, m, H, E, cm = _setup()
    thr = np.log(1 / ALPHA)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    axA, axB, axC = axes

    # A: spaghetti of logE over many days, null vs traffic drift
    n_days = 60
    first_cross = []
    for s in range(n_days):
        ev, _, _ = _day(p, "none", 0.0, 9000 + s)
        logE, _, _ = _run_tempo(ev)
        axA.plot(logE, color=MUTED, alpha=0.25, lw=0.8, zorder=1)
    for s in range(n_days):
        ev, _, _ = _day(p, "traffic", 1.8, 5000 + s)
        logE, _, alarm = _run_tempo(ev)
        axA.plot(logE, color=C["blue"], alpha=0.30, lw=0.9, zorder=2)
        if alarm is not None:
            first_cross.append(alarm)
    fd = _first_drift_idx(_day(p, "traffic", 1.8, 5000)[0])
    axA.axhline(thr, color=STATUS_CRITICAL, ls="--", lw=1.4)
    axA.text(1, thr + 0.25, r"alarm: $\log(1/\alpha)$",
             color=STATUS_CRITICAL, fontsize=8.5)
    axA.axvline(fd, color=INK, ls=":", lw=1.2)
    axA.text(fd + 1, axA.get_ylim()[0] + 0.5, "traffic drift starts",
             fontsize=8.5, color=INK, rotation=90, va="bottom")
    axA.set_xlabel("event along the day")
    axA.set_ylabel(r"$\log E_t$")
    axA.set_title("A — e-process over 60 null days (grey)\n"
                  "and 60 traffic-drift days (blue)", fontsize=10,
                  loc="left")

    # B: one drift day, per-channel cumulative evidence
    ev, _, _ = _day(p, "traffic", 1.8, 5003)
    logE, by_ch, alarm = _run_tempo(ev)
    for c, tr in by_ch.items():
        axB.plot(tr, color=CH_COLOR[c], lw=1.6, label=c)
    axB.plot(logE, color=INK, lw=2.2, label=r"master $\log E_t$")
    axB.axhline(thr, color=STATUS_CRITICAL, ls="--", lw=1.2)
    if alarm is not None:
        axB.scatter([alarm], [logE[alarm]], s=110, color=STATUS_CRITICAL,
                    zorder=6, marker="X")
        axB.annotate("alarm — replan", (alarm, logE[alarm]),
                     textcoords="offset points", xytext=(-70, 8),
                     fontsize=9, color=STATUS_CRITICAL)
    axB.axvline(_first_drift_idx(ev), color=INK, ls=":", lw=1.2)
    axB.set_xlabel("event along the day")
    axB.set_title("B — one drift day: evidence by channel", fontsize=10,
                  loc="left")
    axB.legend(frameon=False, fontsize=8, loc="upper left")

    # C: the physical signal behind panel B
    zs = [e["z"] for e in ev if e["channel"] == "travel"]
    lateness = np.cumsum((np.exp(np.array(zs) * p.sig_T) - 1.0)
                         * p.tau * 60.0)
    axC.plot(np.arange(1, m + 1), lateness, color=C["blue"], lw=2.0)
    axC.axhline(0, color=MUTED, lw=1.0)
    kd = (_first_drift_idx(ev)) // len(STOP_CH) + 1
    axC.axvline(kd, color=INK, ls=":", lw=1.2)
    if alarm is not None:
        axC.axvline(alarm // len(STOP_CH) + 1, color=STATUS_CRITICAL,
                    ls="--", lw=1.4)
        axC.text(alarm // len(STOP_CH) + 1.3, lateness.max() * 0.15,
                 "TEMPO alarm", rotation=90, fontsize=8.5,
                 color=STATUS_CRITICAL)
    axC.set_xlabel("stop along route")
    axC.set_ylabel("cumulative lateness vs forecast (min)")
    axC.set_title("C — what the dispatcher experiences", fontsize=10,
                  loc="left")

    fig.suptitle(f"How TEMPO works — {NAME}, longest route "
                 f"({m} stops), traffic drift 1.8x after 1 h",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tempo_fig1_explainer.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote tempo_fig1_explainer.png", flush=True)


# ── figure 2: TEMPO vs BATON, same route, two kinds of bad day ──────────────

def _pick_demand_day(p, cm, H, seeds=range(7000, 7400)):
    """A didactic demand-surge day: BATON hands off AND TEMPO fires,
    both strictly after the change-point, on a day whose load path
    genuinely threatens the boundary."""
    best = None
    for s in seeds:
        ev, _, _ = _day(p, "demand", 1.2, s)
        g = _g_real(ev, p)
        kb = _baton_trigger(g, p.B, H, cm)
        logE, _, alarm = _run_tempo(ev)
        fd = _first_drift_idx(ev)
        if fd is None or alarm is None or kb is None:
            continue
        fd_stop = fd // 5 + 1
        k_tempo = alarm // 5 + 1
        if kb > fd_stop and k_tempo >= fd_stop \
                and np.cumsum(g).max() > p.B:
            gap = abs(k_tempo - kb)
            if best is None or gap < best[0]:
                best = (gap, s, ev, g, kb, alarm)
    return best


def _pick_traffic_day(p, cm, H, seeds=range(8000, 8200)):
    """A traffic-jam day: TEMPO fires, BATON never triggers (and the
    load path never breaches, so there is nothing for BATON to see)."""
    for s in seeds:
        ev, _, _ = _day(p, "traffic", 2.0, s)
        g = _g_real(ev, p)
        kb = _baton_trigger(g, p.B, H, cm)
        logE, _, alarm = _run_tempo(ev)
        fd = _first_drift_idx(ev)
        if kb is None and alarm is not None and fd and alarm >= fd \
                and np.cumsum(g).max() <= p.B:
            return s, ev, g, alarm
    return None


def _setup_didactic():
    """The BATON-paper didactic route: inflation-gate plan with real
    slack B and mid-route breach risk (same picker as fig2/fig3 of
    paper 1), so BATON's handoff is a mid-route decision rather than a
    stop-1 formality."""
    (name, D, dem, Q, n, scale, coords, plan, dbar, pbar,
     dsc_tr, psc_tr, dsc_te, psc_te, route, ovr) = \
        _pick_route_and_scenarios("hanoi")
    r = np.array(route)
    g_tr = psc_tr[:, r] - dsc_tr[:, r]
    p = params_from_route(route, D, dem, Q, scale)
    p.B = float(Q - dbar[r].sum())
    p.mu_g = g_tr.mean(0)
    p.sig_g = np.maximum(g_tr.std(0), 1e-6)
    p.tau = p.tau * (5.0 / max(p.tau.sum(), 1e-9))
    H, E = route_cost_schedules(route, D, scale, LastMileCosts())
    cm = fit_lsm_general(g_tr, p.B, H, E)
    return p, route, len(route), H, cm


def fig2():
    p, route, m, H, cm = _setup_didactic()
    thr = np.log(1 / ALPHA)
    stops = np.arange(1, m + 1)

    dem = _pick_demand_day(p, cm, H)
    tra = _pick_traffic_day(p, cm, H)
    assert dem and tra, "no didactic day found — widen seed scan"
    _, sd, ev_d, g_d, kb_d, al_d = dem
    st, ev_t, g_t, al_t = tra

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    (axA, axB), (axC, axD) = axes

    # top-left: demand day, load path + BATON
    W = np.cumsum(g_d)
    axA.plot(stops, W, color=C["blue"], lw=2.0, label="net load $W_k$")
    axA.axhline(p.B, color=STATUS_CRITICAL, ls="--", lw=1.4)
    axA.text(m * 0.72, p.B + 0.02 * (W.max() - W.min()),
             "recourse boundary $B$", color=STATUS_CRITICAL, fontsize=8.5)
    axA.axvline(kb_d, color=C["aqua"], lw=2.0)
    axA.text(kb_d + 0.3, W.max() * 0.98, "BATON hands off", rotation=90,
             fontsize=9, color=C["aqua"], va="top")
    kbrk = next((k for k in range(1, m + 1) if W[k - 1] > p.B), None)
    if kbrk:
        axA.scatter([kbrk], [W[kbrk - 1]], marker="X", s=130,
                    color=STATUS_CRITICAL, zorder=6)
        axA.annotate("breach if nobody acts", (kbrk, W[kbrk - 1]),
                     textcoords="offset points", xytext=(8, -14),
                     fontsize=8.5, color=STATUS_CRITICAL)
    fd_stop = _first_drift_idx(ev_d) // 5 + 1
    axA.axvline(fd_stop, color=INK, ls=":", lw=1.2)
    axA.set_title("demand-surge day — the load path\n"
                  "(the signal BATON watches)", fontsize=10, loc="left")
    axA.set_ylabel("onboard net load")
    axA.legend(frameon=False, fontsize=8)

    # top-right: traffic day, lateness (BATON has no state for this)
    zs = [e["z"] for e in ev_t if e["channel"] == "travel"]
    late = np.cumsum((np.exp(np.array(zs) * p.sig_T) - 1.0) * p.tau * 60.0)
    axB.plot(stops, late, color=C["blue"], lw=2.0)
    axB.axhline(0, color=MUTED, lw=1.0)
    fd_stop_t = _first_drift_idx(ev_t) // 5 + 1
    axB.axvline(fd_stop_t, color=INK, ls=":", lw=1.2)
    axB.set_title("traffic-jam day — cumulative lateness\n"
                  "(invisible to BATON: load stays normal)", fontsize=10,
                  loc="left")
    axB.set_ylabel("lateness vs forecast (min)")

    # bottom row: TEMPO's log E on both days
    for ax, ev, alarm, kb in ((axC, ev_d, al_d, kb_d),
                              (axD, ev_t, al_t, None)):
        logE, by_ch, _ = _run_tempo(ev)
        x = np.arange(len(logE)) / 5.0 + 1.0      # event -> stop axis
        ax.plot(x, logE, color=INK, lw=2.0, label=r"$\log E_t$")
        ax.axhline(thr, color=STATUS_CRITICAL, ls="--", lw=1.2)
        if alarm is not None:
            ax.scatter([x[alarm]], [logE[alarm]], s=110, marker="X",
                       color=STATUS_CRITICAL, zorder=6)
            ax.text(x[alarm] + 0.3, logE[alarm] * 0.8, "TEMPO alarm",
                    fontsize=9, color=STATUS_CRITICAL)
        if kb is not None:
            ax.axvline(kb, color=C["aqua"], lw=2.0, alpha=0.7)
        ax.axvline(_first_drift_idx(ev) / 5.0 + 1.0, color=INK,
                   ls=":", lw=1.2)
        ax.set_xlabel("stop along route")
        ax.set_ylabel(r"$\log E_t$")
    axC.set_title("TEMPO on the demand day — both react", fontsize=10,
                  loc="left")
    axD.set_title("TEMPO on the traffic day — only TEMPO fires",
                  fontsize=10, loc="left")

    fig.suptitle(f"TEMPO vs BATON on the same route (HANOI-100-1, {m} stops) — "
                 "complementary, not competing:\nBATON prices per-route "
                 "recourse from the load; TEMPO tests the whole planning "
                 "model and calls the replan", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tempo_fig2_vs_baton.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote tempo_fig2_vs_baton.png (demand seed {sd}, "
          f"traffic seed {st})", flush=True)


# ── figure 3: the traffic day on the Hanoi map ───────────────────────────────

def fig3():
    p, route, D, scale, m, H, E, cm = _setup()
    tra = _pick_traffic_day(p, cm, H)
    assert tra, "no didactic traffic day found"
    _, ev_t, _, alarm = tra
    k_alarm = alarm // 5 + 1
    k_drift = _first_drift_idx(ev_t) // 5 + 1

    coords = json.loads((CITY_DIR / f"{NAME}.coords.json")
                        .read_text())["nodes"]
    G = _load_graph(NAME.split("-")[0].lower())
    osm = [c[0] for c in coords]
    lat = [c[1] for c in coords]
    lon = [c[2] for c in coords]

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    for u, v in G.edges():
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color=STREET, lw=0.5, zorder=1)

    seq = [0] + list(route) + [0]
    for i, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
        k = i + 1                                     # stop being approached
        if k <= k_drift:
            col, ls, lw = C["blue"], "-", 2.2         # on-model
        elif k <= k_alarm:
            col, ls, lw = C["yellow"], "-", 2.6       # drift, undetected
        else:
            col, ls, lw = MUTED, ":", 1.8             # stale plan post-alarm
        try:
            pts = _street_path(G, osm[a], osm[b])
            ax.plot(*zip(*pts), color=col, ls=ls, lw=lw, zorder=3,
                    solid_capstyle="round")
        except Exception:
            ax.plot([lon[a], lon[b]], [lat[a], lat[b]], color=col,
                    ls=":", lw=1.2, zorder=3)
    for i, c in enumerate(route):
        ax.scatter([lon[c]], [lat[c]], s=22, zorder=4,
                   color=(C["blue"] if i + 1 <= k_drift else
                          C["yellow"] if i + 1 <= k_alarm else MUTED),
                   edgecolors="white", linewidths=0.5)

    an = route[k_alarm - 1]
    ax.scatter([lon[an]], [lat[an]], s=260, facecolors="none",
               edgecolors=STATUS_CRITICAL, linewidths=2.6, zorder=6)
    ax.annotate("TEMPO alarm — evidence of a 2x jam\n"
                "crosses $1/\\alpha$: replan the rest of the day",
                (lon[an], lat[an]), textcoords="offset points",
                xytext=(14, 10), fontsize=9.5, color=STATUS_CRITICAL)
    dn = route[k_drift - 1]
    ax.annotate("jam begins (unannounced)", (lon[dn], lat[dn]),
                textcoords="offset points", xytext=(10, -16),
                fontsize=9.5, color=INK)
    ax.scatter([lon[0]], [lat[0]], marker="*", s=380, color=INK,
               zorder=6, edgecolors="white", linewidths=1.0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=C["blue"], lw=2.2,
               label="on-model driving (plan trusted)"),
        Line2D([], [], color=C["yellow"], lw=2.6,
               label="drift running, not yet detectable"),
        Line2D([], [], color=MUTED, lw=1.8, ls=":",
               label="stale plan TEMPO refuses to continue"),
        Line2D([], [], marker="*", color=INK, lw=0, markersize=14,
               label="depot")], frameon=False, fontsize=9,
        loc="lower left")

    xs = [lon[c] for c in route] + [lon[0]]
    ys = [lat[c] for c in route] + [lat[0]]
    pad = 0.004
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(ys))))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("The traffic-jam day on the Hanoi street network — "
                 "when TEMPO calls the replan", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tempo_fig3_map.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote tempo_fig3_map.png", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if which in ("all", "1"):
        fig1()
    if which in ("all", "2"):
        fig2()
    if which in ("all", "3"):
        fig3()
