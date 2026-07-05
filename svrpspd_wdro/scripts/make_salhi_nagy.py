#!/usr/bin/env python3
"""
make_salhi_nagy.py — Derive the Salhi–Nagy (1999) VRPSPD benchmark from CMT.

The classical medium-to-large VRPSPD benchmark of Salhi and Nagy (1999)
is constructed deterministically from the CMT instances of Christofides,
Mingozzi and Toth (1979): for customer i with coordinates (x_i, y_i) and
CVRP demand q_i, the split ratio is

    r_i = min(x_i / y_i, y_i / x_i)                (r_i in (0, 1])

and the X-instances set  delivery_i = r_i * q_i,  pickup_i = (1 - r_i) * q_i,
while the Y-instances swap the two. We generate X/Y versions of the CMT
instances WITHOUT route-duration limits (CMT1-5, 11, 12; 50-199 customers),
the subset used throughout the VRPSPD literature.

Instance files are fetched from CVRPLIB (galgos.inf.puc-rio.br/cvrplib) and
written in the Dethloff .vrpspd layout consumed by dethloff_runner:
distances are Euclidean, stored as round(d * 10^4) so the pipeline's fixed
10^4 divisor returns the original coordinate units.

Usage:
    python scripts/make_salhi_nagy.py [out=data/SalhiNagy]
"""

import sys
import math
import urllib.request
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS))

from make_city_instances import write_vrpspd

BASE = "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download/instance/{}"
# CVRPLIB download ids: CMT1..CMT14 = 99..112. VRPSPD literature uses the
# duration-unconstrained subset:
CMT_IDS = {"CMT1": 99, "CMT2": 100, "CMT3": 101, "CMT4": 102, "CMT5": 103,
           "CMT11": 109, "CMT12": 110}

DIST_SCALE = 10_000    # stored int = round(euclid * 10^4); pipeline divides by 10^4


def _fetch(cmt: str) -> str:
    url = BASE.format(CMT_IDS[cmt])
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode()


def _parse_cvrp(txt: str):
    """Minimal TSPLIB parser for EUC_2D CVRP files."""
    lines = [l.strip() for l in txt.splitlines()]
    n = Q = None
    coords, demands = {}, {}
    section = None
    for l in lines:
        u = l.upper()
        if u.startswith("DIMENSION"):
            n = int(l.split(":")[-1])
        elif u.startswith("CAPACITY"):
            Q = float(l.split(":")[-1])
        elif u.startswith("NODE_COORD_SECTION"):
            section = "coord"; continue
        elif u.startswith("DEMAND_SECTION"):
            section = "demand"; continue
        elif u.startswith(("DEPOT_SECTION", "EOF")):
            section = None; continue
        if section and l and l[0].isdigit():
            t = l.split()
            if section == "coord":
                coords[int(t[0])] = (float(t[1]), float(t[2]))
            else:
                demands[int(t[0])] = float(t[1])
    assert n and Q and len(coords) == n, "parse failure"
    xy = np.array([coords[i + 1] for i in range(n)])       # node 1 = depot
    q  = np.array([demands.get(i + 1, 0.0) for i in range(n)])
    return xy, q, Q, n


def _euclid_matrix(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.round(np.hypot(diff[..., 0], diff[..., 1]) * DIST_SCALE).astype(np.int64)


def _split(xy: np.ndarray, q: np.ndarray):
    """Salhi–Nagy ratio split. Depot row keeps zero demand."""
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.minimum(x / np.where(y == 0, np.nan, y),
                       y / np.where(x == 0, np.nan, x))
    r = np.nan_to_num(r, nan=0.5)          # degenerate coordinates -> even split
    r = np.clip(r, 0.0, 1.0)
    deliv = r * q
    pick  = q - deliv
    return deliv, pick


def main():
    out = _WDRO / "data" / "SalhiNagy"
    for arg in sys.argv[1:]:
        if arg.startswith("out="):
            out = Path(arg[4:])
    out.mkdir(parents=True, exist_ok=True)

    for cmt in CMT_IDS:
        txt = _fetch(cmt)
        xy, q, Q, n = _parse_cvrp(txt)
        D = _euclid_matrix(xy)
        deliv, pick = _split(xy, q)
        for tag in ("X", "Y"):
            d, p = (deliv, pick) if tag == "X" else (pick, deliv)
            dem = np.column_stack([d[1:], p[1:]])          # customers only
            name = f"{cmt}{tag}"
            write_vrpspd(out / f"{name}.vrpspd", name, D, dem, Q,
                         f"Salhi-Nagy 1999 {tag}-split of {cmt} "
                         f"(r=min(x/y,y/x)), n_cust={n - 1}")
            print(f"  wrote {name}.vrpspd  n_cust={n - 1}  Q={Q:.0f}  "
                  f"deliv_sum={d.sum():.0f}  pick_sum={p.sum():.0f}")


if __name__ == "__main__":
    main()
