"""RARECERT R1 task B figures (plain matplotlib, grayscale-safe).

(a) rarecert_bias_varrho.pdf: signed single-shift relative bias p_single/p_ref - 1 against
    the closed-form end-mode ratio varrho, for every row of results/rarecert/smis_routes.csv
    (N = 1e6 single-shift draws, rse from RV_single) and of sensitivity_routes.csv (N from
    its rse_single column). Vertical lines at varrho = 1/4 and 3/4; each point carries a
    bar of +-2 rse.
(b) rarecert_bias_mhat.pdf: absolute single-shift relative bias against the exact-energy
    mode mass m_hat of the weaker end mode (sensitivity_routes.csv only, the sole file that
    records m_hat), log-log, with the identity line.
(c) rarecert_mode_energies.pdf: norm |x*| = sqrt(2 E) of the dominating point of every
    stage block (delivery block of stage k, pickup block of stage k) for one Northern Vietnam
    and one Guangdong route at gamma = 1e-4 (threshold c from smis_routes.csv), against the
    asymptotic c / lambda_k of the same block (lambda_k = lam_d[k] or lam_p[k] from
    rarecert.smis.stage_scales).

Usage (from svrpspd_wdro/):  python3 scripts/make_rarecert_figures.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

RES = ROOT / "results" / "rarecert"
SMIS_CSV = RES / "smis_routes.csv"
SENS_CSV = RES / "sensitivity_routes.csv"
FIG_DIR = ROOT.parent / "papers" / "rarecert" / "manuscript" / "figures"
N_SMIS_SINGLE = 1_000_000          # single-shift draws in the QUASAR table (run_quasar_rare.N_TRUTH)
GAMMA_C = 1e-4
VARRHO_LINES = (0.25, 0.75)  # varrho thresholds drawn as vertical lines

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
MARKERS = {"base": "o", "rho": "s", "cv": "^", "pickup": "D", "family": "v"}
GRAY = {"smis": "0.55", "sens": "0.0"}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def bias_rows():
    """(varrho, signed bias, rse, source, factor) for every usable row of both CSVs."""
    out = []
    for r in _read(SMIS_CSV):
        p, pr = _f(r["p_single"]), _f(r["p_ref"])
        rv = _f(r["RV_single"])
        if not (math.isfinite(p) and math.isfinite(pr) and pr > 0 and math.isfinite(rv)):
            continue
        out.append((_f(r["ratio"]), p / pr - 1.0, math.sqrt(rv / N_SMIS_SINGLE), "smis", "gamma"))
    for r in _read(SENS_CSV):
        if r["error"]:
            continue
        e = _f(r["e_single"])
        rse = _f(r["rse_single"])
        if not (math.isfinite(e) and math.isfinite(rse)):
            continue
        out.append((_f(r["varrho"]), e, rse, "sens", r["factor"]))
    return [o for o in out if math.isfinite(o[0])]


def fig_bias_varrho():
    rows = bias_rows()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axvline(VARRHO_LINES[0], color="0.4", ls="--", lw=0.8)
    ax.axvline(VARRHO_LINES[1], color="0.4", ls=":", lw=0.8)
    for src in ("smis", "sens"):
        sub = [r for r in rows if r[3] == src]
        if not sub:
            continue
        x = np.array([r[0] for r in sub])
        y = np.array([r[1] for r in sub])
        e = np.array([2.0 * r[2] for r in sub])
        if src == "smis":
            ax.errorbar(x, y, yerr=e, fmt="o", ms=2.5, color=GRAY["smis"], ecolor="0.75", elinewidth=0.5,
                        capsize=0, label="QUASAR table (N = 1e6)")
        else:
            for fac, mk in MARKERS.items():
                s2 = [r for r in sub if r[4] == fac]
                if not s2:
                    continue
                ax.errorbar([r[0] for r in s2], [r[1] for r in s2], yerr=[2.0 * r[2] for r in s2], fmt=mk,
                            ms=3.5, mfc="none", color="k", ecolor="0.3", elinewidth=0.6, capsize=0,
                            label=f"sensitivity: {fac}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"end-mode ratio $\varrho$")
    ax.set_ylabel("single-shift bias $p_{\\rm single}/p_{\\rm ref}-1$")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "rarecert_bias_varrho.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out, len(rows)


def fig_bias_mhat():
    pts = []
    for r in _read(SENS_CSV):
        if r["error"]:
            continue
        m, e = _f(r["m_hat"]), _f(r["e_single"])
        if math.isfinite(m) and m > 0 and math.isfinite(e) and abs(e) > 0:
            pts.append((m, abs(e), r["factor"]))
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for fac, mk in MARKERS.items():
        s2 = [p for p in pts if p[2] == fac]
        if s2:
            ax.scatter([p[0] for p in s2], [p[1] for p in s2], marker=mk, s=14, facecolors="none",
                       edgecolors="k", linewidths=0.6, label=fac)
    if pts:
        lo = min(min(p[0] for p in pts), min(p[1] for p in pts))
        hi = max(max(p[0] for p in pts), max(p[1] for p in pts))
        g = np.logspace(math.log10(lo) - 0.2, math.log10(hi) + 0.2, 50)
        ax.plot(g, g, color="0.45", lw=0.8, ls="--", label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"exact-energy mode mass $\hat m=\exp(-(E_{\rm other}-E_{\min}))$")
    ax.set_ylabel("|single-shift bias|")
    if pts:
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no rows with finite m_hat > 0", transform=ax.transAxes, ha="center", fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "rarecert_bias_mhat.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out, len(pts)


def _pick_route(rows, prefix: str):
    cands = [r for r in rows if r["route_id"].startswith(prefix) and math.isclose(_f(r["gamma"]), GAMMA_C,
                                                                                 rel_tol=1e-9)
             and _f(r["L"]) >= 6]
    return cands[0] if cands else None


def fig_mode_energies():
    import run_rarecert_smis as rs
    from quantum.rare import RouteSim
    from rarecert.smis import Unreachable, block_direction, dominating_point, stage_scales

    rows = _read(SMIS_CSV)
    means = rs.route_means()
    picks = [("VNN", "Northern Vietnam"), ("GD", "Guangdong")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
    summary = []
    for ax, (prefix, title) in zip(axes, picks):
        r = _pick_route(rows, prefix + "-")
        if r is None:
            ax.set_title(f"{title}: no route")
            continue
        rid = r["route_id"]
        m = means[rid]
        c = _f(r["c"])
        sim = RouteSim(np.asarray(m["dbar"], dtype=float), np.asarray(m["pbar"], dtype=float), cv=0.3, rho=0.6)
        st = stage_scales(sim)
        ks = np.arange(sim.L + 1)
        for block, mk, lam in (("d", "o", st["lam_d"]), ("p", "s", st["lam_p"])):
            norms, asym, kk = [], [], []
            for k in ks:
                has = (k < sim.L and sim.act_d[k:].any()) if block == "d" else (k > 0 and sim.act_p[:k].any())
                if not has:
                    continue
                try:
                    x, _ = dominating_point(sim, c, int(k), block_direction(sim, int(k), block))
                except Unreachable:
                    continue
                kk.append(k)
                norms.append(float(np.linalg.norm(x)))
                asym.append(c / lam[k])
            if kk:
                ax.plot(kk, norms, mk, mfc="none" if block == "d" else "k", color="k", ms=4,
                        label=f"{'delivery' if block == 'd' else 'pickup'} block: $|x^*|$")
                ax.plot(kk, asym, "-" if block == "d" else ":", color="0.4", lw=0.9,
                        label=f"{'delivery' if block == 'd' else 'pickup'} block: $c/\\lambda_k$")
        ax.set_yscale("log")
        ax.set_xlabel("stage $k$")
        ax.set_title(f"{title}, {rid}, $L={sim.L}$", fontsize=8)
        ax.set_ylabel(r"norm $|x^*|=\sqrt{2E}$")
        ax.legend(fontsize=6, frameon=False)
        summary.append((rid, sim.L, c))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "rarecert_mode_energies.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out, summary


def main() -> int:
    a, na = fig_bias_varrho()
    print(f"wrote {a} ({na} points)")
    b, nb = fig_bias_mhat()
    print(f"wrote {b} ({nb} points)")
    c, info = fig_mode_energies()
    print(f"wrote {c} routes {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
