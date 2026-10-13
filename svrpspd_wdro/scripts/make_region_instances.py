#!/usr/bin/env python3
"""
make_region_instances.py — Region-scale SVRPSPD instances (one per depot area).

Builds instances that serve a whole region, one instance per depot area:
  VNN  Northern Vietnam (Bac Bo, the 25 provinces of the 2008 boundaries)
  GD   Guangdong (CHN-ADM2 units inside the "Guangzhou Province" ADM1 polygon)

Population comes from Kontur Population 2023-11-01 (H3 res-8 hexagons); area
boundaries from geoBoundaries gbOpen; road distances from the public OSRM
table service (OpenStreetMap). Output is the Dethloff .vrpspd layout, so
`dethloff_runner.parse_dethloff` and every QUASAR script consume it unchanged.

Pipeline per area (steps 1-5 need no network; `--dry-run` stops after 5):
  1. P_a = population of the area's hexes; k_a = max(N_MIN, round(P_a / ppp)).
  2. Population-weighted k-means on hex centres (planar km), point = weighted
     centroid, pop = cluster sum; clusters with pop < MIN_CLUSTER_POP dropped.
  3. Clusters with pop > SPLIT = 2*ppp are re-clustered once into
     ceil(pop/ppp) parts.
  4. If the area still has more than N_MAX points, the area is split: its hexes
     are partitioned by population-weighted k-means into ceil(n/N_MAX) sub-areas
     (named {NAME}-1, {NAME}-2, ...), each planned again as its own instance.
  5. Depot = population-weighted medoid over the points (haversine); it is an
     extra node 0 co-located with that point, which stays a customer.
  6. OSRM road-distance table over [depot] + points, snap filter, null filter,
     symmetrisation, demands. Writes .vrpspd, .coords.json, manifest.csv, README.

Usage (run from svrpspd_wdro/):
    python scripts/make_region_instances.py --region VNN|GD|all
        [--areas GLOB] [--pop-per-point X] [--force] [--dry-run] [--readme]

Dry-run needs no network and writes nothing. A normal run downloads road
distances from the public OSRM server (cached under data/Region/_raw/osrm_cache).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import hashlib
import json
import math
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_WDRO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from make_city_instances import SCALE, write_vrpspd  # noqa: E402  (shared format)

# ---------------------------------------------------------------- constants
R_EARTH_KM      = 6371.0
N_MIN           = 10
SPLIT_FACTOR    = 2.0            # SPLIT = SPLIT_FACTOR * ppp
N_MAX           = 600            # max service points per instance
MIN_CLUSTER_POP = 100.0
KMEANS_N_INIT   = 4
SNAP_MAX_M      = 5000.0
MAX_AREA_SPLIT_TRIES = 6

OSRM_URL        = "https://router.project-osrm.org/table/v1/driving/"
BLOCK           = 100            # max sources / destinations per request
REQUEST_GAP_S   = 1.0
RETRY_BACKOFF_S = (2, 4, 8, 16)
HTTP_TIMEOUT_S  = 120

PICKUP_SPREAD   = (0.5, 1.5)     # pickup = pop * PICK_RATE * KG * U(0.5, 1.5)

DATA_DIR  = _WDRO / "data" / "Region"
RAW_DIR   = DATA_DIR / "_raw"
CACHE_DIR = RAW_DIR / "osrm_cache"

VN_PROVINCES = [
    "Hà Nội", "Hải Phòng", "Quảng Ninh", "Hải Dương", "Hưng Yên", "Bắc Ninh",
    "Vĩnh Phúc", "Thái Bình", "Hà Nam", "Nam Định", "Ninh Bình", "Hà Giang",
    "Cao Bằng", "Bắc Kạn", "Tuyên Quang", "Lào Cai", "Yên Bái", "Thái Nguyên",
    "Lạng Sơn", "Bắc Giang", "Phú Thọ", "Điện Biên", "Lai Châu", "Sơn La",
    "Hòa Bình",
]

# Per-region demand and capacity configuration. Every number carries its
# source string; "ASSUMPTION" marks modelling choices, not measured data.
REGION_CFG = {
    "VNN": {
        "label": "Northern Vietnam",
        "gz": "kontur_VN.gpkg.gz", "gpkg": "kontur_VN.gpkg",
        "deliv": 0.0874,     # parcels / person / day
        "pick": 0.0874,
        "kg": 1.0,           # kg / parcel
        "Q": 15000.0,        # kg
        "ppp": 15000.0,      # people per service point
        "src": {
            "deliv": "MIC year-end 2024: 3.2e9 items / 100.3e6 people / 365 days",
            "pick": "MIC year-end 2024 national flows, balanced with delivery "
                    "(national rate; no regional split published)",
            "kg": "ASSUMPTION: 1.0 kg per parcel (no weight data used)",
            "Q": "ASSUMPTION: 15 t truck class",
            "ppp": "design choice (service-point granularity)",
        },
    },
    "GD": {
        "label": "Guangdong",
        "gz": "kontur_CN.gpkg.gz", "gpkg": "kontur_CN.gpkg",
        "deliv": 0.3396,
        "pick": 0.913,
        "kg": 1.0,
        "Q": 30000.0,
        "ppp": 4000.0,
        "src": {
            "deliv": "China 2024: 1745e8 parcels / 1408e6 people / 365 days, used as a "
                     "national per-capita proxy for parcels received",
            "pick": "Guangdong 2024 origin volume 425.9e8 parcels / 127.8e6 people / 365 days",
            "kg": "ASSUMPTION: 1.0 kg per parcel (no weight data used)",
            "Q": "ASSUMPTION: 30 t truck class",
            "ppp": "design choice (service-point granularity)",
        },
    },
}

MANIFEST_COLS = ["name", "region", "area", "parent_area", "n_customers", "pop_area",
                 "pop_served", "pop_dropped", "pop_per_point_eff", "Q",
                 "sum_delivery", "max_delivery", "n_requests", "D_max_km",
                 "D_mean_km", "seconds"]


# ---------------------------------------------------------------- data classes
@dataclass
class Area:
    name: str            # display name (boundary shapeName, cleaned, or sub-area name)
    geom: object         # shapely geometry used for hex assignment (None for sub-areas)


@dataclass
class Cluster:
    members: np.ndarray  # indices into the area's hex arrays
    pop: float
    xy: np.ndarray       # weighted centroid, planar km


@dataclass
class Plan:
    area: str
    pop_area: float
    n_hex: int
    k_a: int             # k from step 1
    ppp: float
    lat: np.ndarray      # points (K,)
    lon: np.ndarray
    pop: np.ndarray      # points (K,)
    depot: int           # index into points of the medoid
    pop_small: float     # population in clusters dropped for < MIN_CLUSTER_POP


@dataclass
class Block:
    src: list
    dst: list
    coords: list         # coordinate indices sent in the URL
    diag: bool


# ---------------------------------------------------------------- names
def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d").replace("Đ", "D")


def slug(name: str) -> str:
    """ASCII upper-case slug: 'Hà Nội\\t' -> 'HANOI', 'Điện Biên' -> 'DIENBIEN'."""
    return re.sub(r"[^A-Za-z0-9]", "", ascii_fold(name)).upper()


def area_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) % 2**31


def instance_names(region: str, areas: list[Area]) -> list[str]:
    used, out = set(), []
    for i, a in enumerate(areas):
        base = f"{region}-{slug(a.name) or f'AREA{i:03d}'}"
        name, k = base, 2
        while name in used:
            name = f"{base}-{k}"
            k += 1
        used.add(name)
        out.append(name)
    return out


# ---------------------------------------------------------------- geometry
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def project(lat, lon, lat0, lon0):
    """Local planar km: x = R cos(lat0) dlon, y = R dlat (radians)."""
    x = R_EARTH_KM * math.cos(math.radians(lat0)) * np.radians(np.asarray(lon) - lon0)
    y = R_EARTH_KM * np.radians(np.asarray(lat) - lat0)
    return np.column_stack([x, y])


def unproject(xy, lat0, lon0):
    lat = lat0 + np.degrees(xy[:, 1] / R_EARTH_KM)
    lon = lon0 + np.degrees(xy[:, 0] / (R_EARTH_KM * math.cos(math.radians(lat0))))
    return lat, lon


def weighted_medoid(lat, lon, pop) -> int:
    """Index i minimising sum_j pop_j * haversine(i, j)."""
    lat, lon, pop = map(np.asarray, (lat, lon, pop))
    H = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return int(np.argmin(H @ pop))


# ---------------------------------------------------------------- areas
def _load_geojson(name: str):
    from shapely import make_valid
    from shapely.geometry import shape
    data = json.loads((RAW_DIR / name).read_text(encoding="utf-8"))
    out = []
    for f in data["features"]:
        g = shape(f["geometry"])
        if not g.is_valid:
            g = make_valid(g)
        out.append((f["properties"], g))
    return out


def _norm(s) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


def load_areas(region: str) -> list[Area]:
    if region == "VNN":
        found = {}
        for props, g in _load_geojson("VNM-ADM1.geojson"):
            found.setdefault(_norm(props.get("shapeName")), g)
        areas = []
        for prov in VN_PROVINCES:
            key = _norm(prov)
            if key not in found:
                raise RuntimeError(f"VNN: province not found in VNM-ADM1: {prov!r}")
            areas.append(Area(name=key, geom=found[key]))
        if len(areas) != 25:
            raise RuntimeError(f"VNN: expected 25 provinces, got {len(areas)}")
        return areas
    if region == "GD":
        adm1 = [g for p, g in _load_geojson("CHN-ADM1.geojson")
                if _norm(p.get("shapeName")).startswith("Guangzhou")]
        if len(adm1) != 1:
            raise RuntimeError(f"GD: expected one 'Guangzhou*' ADM1, got {len(adm1)}")
        gd = adm1[0]
        areas = []
        for props, g in _load_geojson("CHN-ADM2.geojson"):
            if not gd.covers(g.representative_point()):
                continue
            nm = _norm(props.get("shapeName")) or f"Unnamed ADM2 {props.get('shapeID')}"
            areas.append(Area(name=nm, geom=g))
        if not 80 <= len(areas) <= 110:
            raise RuntimeError(f"GD: {len(areas)} ADM2 units, expected 80..110")
        return areas
    raise ValueError(region)


def assign_hexes(lat, lon, areas: list[Area]):
    """Owner index per hex (-1 = outside every area) and the count of hexes
    that matched two areas (should be 0)."""
    import shapely
    owner = np.full(len(lat), -1, dtype=np.int64)
    multi = 0
    for i, a in enumerate(areas):
        x0, y0, x1, y1 = a.geom.bounds
        cand = np.flatnonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
        if len(cand) == 0:
            continue
        inside = cand[shapely.contains_xy(a.geom, lon[cand], lat[cand])]
        multi += int(np.count_nonzero(owner[inside] >= 0))
        owner[inside[owner[inside] < 0]] = i
    return owner, multi


def ensure_gpkg(region: str) -> Path:
    cfg = REGION_CFG[region]
    out, gz = RAW_DIR / cfg["gpkg"], RAW_DIR / cfg["gz"]
    if out.exists():
        return out
    print(f"[{region}] decompressing {gz.name} -> {out.name} ...", flush=True)
    tmp = out.with_suffix(".gpkg.part")
    with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    tmp.replace(out)
    return out


def load_hexes(region: str):
    """(lat, lon, pop) for every Kontur hexagon with population > 0."""
    import h3
    gpkg = ensure_gpkg(region)
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT h3, population FROM population WHERE population > 0").fetchall()
    finally:
        con.close()
    n = len(rows)
    lat = np.empty(n)
    lon = np.empty(n)
    pop = np.empty(n)
    for i, (h, p) in enumerate(rows):
        la, lo = h3.cell_to_latlng(h)
        lat[i], lon[i], pop[i] = la, lo, p
    return lat, lon, pop


# ---------------------------------------------------------------- steps 1-5
def _kmeans_pass(xy, w, k, seed) -> list[Cluster]:
    from sklearn.cluster import KMeans
    n = len(w)
    k = min(k, n)
    if k <= 1:
        labels = np.zeros(n, dtype=np.int64)
    else:
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed)
        labels = km.fit(xy, sample_weight=w).labels_
    out = []
    for lab in np.unique(labels):
        mem = np.flatnonzero(labels == lab)
        out.append(Cluster(members=mem, pop=float(w[mem].sum()),
                           xy=np.average(xy[mem], axis=0, weights=w[mem])))
    return out


def _split_oversize(clusters, xy, w, area, ppp, split) -> list[Cluster]:
    out = []
    for ci, c in enumerate(clusters):
        if c.pop <= split or len(c.members) < 2:
            out.append(c)
            continue
        sub_k = min(math.ceil(c.pop / ppp), len(c.members))
        seed = area_seed(f"{area}#{ci}")
        for s in _kmeans_pass(xy[c.members], w[c.members], sub_k, seed):
            out.append(Cluster(members=c.members[s.members], pop=s.pop, xy=s.xy))
    return out


def plan_area(lat, lon, pop, area: str, ppp: float) -> Plan:
    """Steps 1-5 (planning) for one area or sub-area. Pure: no network.
    The returned plan may hold more than N_MAX points; the caller splits it."""
    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    xy = project(lat, lon, lat0, lon0)
    w = np.asarray(pop, dtype=float)
    P = float(w.sum())
    k_a = max(N_MIN, int(round(P / ppp)))
    clusters = _kmeans_pass(xy, w, k_a, area_seed(area))
    clusters = [c for c in clusters if c.pop >= MIN_CLUSTER_POP]
    clusters = _split_oversize(clusters, xy, w, area, ppp, SPLIT_FACTOR * ppp)
    clusters = [c for c in clusters if c.pop >= MIN_CLUSTER_POP]
    if not clusters:
        raise RuntimeError(f"{area}: no clusters survive")
    cxy = np.vstack([c.xy for c in clusters])
    cpop = np.array([c.pop for c in clusters])
    plat, plon = unproject(cxy, lat0, lon0)
    plat, plon = np.round(plat, 5), np.round(plon, 5)   # the OSRM URL precision
    dep = weighted_medoid(plat, plon, cpop)
    return Plan(area=area, pop_area=P, n_hex=len(w), k_a=k_a, ppp=ppp,
                lat=plat, lon=plon, pop=cpop, depot=dep,
                pop_small=P - float(cpop.sum()))


def split_area(lat, lon, pop, area: str, ppp: float, n_points: int):
    """Step 4: partition an area whose plan has more than N_MAX points into
    ceil(n/N_MAX) population-weighted k-means sub-areas, each planned on its own.
    If a sub-area still exceeds N_MAX the partition is retried with one more part."""
    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    xy = project(lat, lon, lat0, lon0)
    w = np.asarray(pop, dtype=float)
    s = math.ceil(n_points / N_MAX)
    for _ in range(MAX_AREA_SPLIT_TRIES):
        parts = []
        clusters = _kmeans_pass(xy, w, s, area_seed(f"{area}/split{s}"))
        ok = True
        for j, c in enumerate(clusters, 1):
            sub_name = f"{area}-{j}"
            sub = plan_area(lat[c.members], lon[c.members], w[c.members], sub_name, ppp)
            if len(sub.pop) > N_MAX:
                ok = False
                break
            parts.append((sub_name, sub))
        if ok:
            return parts
        s += 1
    raise RuntimeError(f"{area}: could not split below N_MAX in {MAX_AREA_SPLIT_TRIES} tries")


# ---------------------------------------------------------------- OSRM
def osrm_block_plan(n: int) -> list[Block]:
    """Every ordered pair (i, j) of the n coordinates is covered exactly once.
    Diagonal blocks send one chunk (<=100 coordinates); off-diagonal blocks
    send source chunk then destination chunk (<=200 coordinates)."""
    chunks = [list(range(s, min(s + BLOCK, n))) for s in range(0, n, BLOCK)]
    plan = []
    for I in chunks:
        for J in chunks:
            if I is J:
                plan.append(Block(src=I, dst=I, coords=I, diag=True))
            else:
                plan.append(Block(src=I, dst=J, coords=I + J, diag=False))
    return plan


def block_url(coords, b: Block) -> str:
    pts = ";".join(f"{coords[i][1]:.5f},{coords[i][0]:.5f}" for i in b.coords)
    if b.diag:
        return f"{OSRM_URL}{pts}?annotations=distance"
    ns = len(b.src)
    srcs = ";".join(str(k) for k in range(ns))
    dsts = ";".join(str(ns + k) for k in range(len(b.dst)))
    return f"{OSRM_URL}{pts}?annotations=distance&sources={srcs}&destinations={dsts}"


_LAST_NET = [0.0]


def _osrm_get(url: str) -> dict:
    """Cached GET of one OSRM table response. Network calls are spaced by
    REQUEST_GAP_S; 429/5xx/network errors retry with back-off 2, 4, 8, 16 s."""
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    last_err = None
    for back in (0,) + RETRY_BACKOFF_S:
        if back:
            time.sleep(back)
        gap = REQUEST_GAP_S - (time.time() - _LAST_NET[0])
        if gap > 0:
            time.sleep(gap)
        _LAST_NET[0] = time.time()
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 or e.code >= 500:
                continue
            raise RuntimeError(f"OSRM HTTP {e.code}: {url[:120]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            last_err = e
            continue
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM code {data.get('code')!r}: {url[:120]}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    raise RuntimeError(f"OSRM request failed after retries: {last_err}")


def build_matrix(coords, get_json=None):
    """Directed road distances (m) over coords (n x 2 lat/lon).
    Returns (M with NaN for unroutable pairs, snap distances (m) per coord,
    number of blocks)."""
    get = get_json if get_json is not None else _osrm_get
    n = len(coords)
    M = np.full((n, n), np.nan)
    snap = np.full(n, np.nan)
    blocks = osrm_block_plan(n)
    for b in blocks:
        resp = get(block_url(coords, b))
        for r, s in enumerate(b.src):
            if np.isnan(snap[s]):
                snap[s] = float(resp["sources"][r]["distance"])
        for c, d in enumerate(b.dst):
            if np.isnan(snap[d]):
                snap[d] = float(resp["destinations"][c]["distance"])
        for r, s in enumerate(b.src):
            row = resp["distances"][r]
            for c, d in enumerate(b.dst):
                v = row[c]
                M[s, d] = np.nan if v is None else float(v)
    np.fill_diagonal(M, 0.0)
    return M, snap, len(blocks)


def select_nodes(M, snap, coords, pops, snap_max=SNAP_MAX_M):
    """Snap filter, depot fallback, null filter on the matrix from build_matrix.
    Node 0 = depot copy; nodes 1.. = points (pops[0] == 0)."""
    M = np.asarray(M)
    n = len(snap)
    ok = np.isfinite(snap) & (snap <= snap_max)
    dropped_snap = [k for k in range(1, n) if not ok[k]]
    survivors = [k for k in range(1, n) if ok[k]]
    if not survivors:
        raise RuntimeError("no point survives the snap filter")
    dep, replaced = 0, False
    if not ok[0]:
        sv = np.array(survivors)
        dep = int(sv[weighted_medoid(coords[sv, 0], coords[sv, 1], pops[sv])])
        replaced = True
    cust = list(survivors)
    dropped_null = []
    first = True
    while True:
        idx = [dep] + cust
        sub = M[np.ix_(idx, idx)]
        nanm = np.isnan(sub)
        if not nanm.any():
            break
        if first:
            bad = [cust[p - 1] for p in range(1, len(idx)) if nanm[0, p] or nanm[p, 0]]
            bad_set = set(bad)
            cust = [c for c in cust if c not in bad_set]
            dropped_null += bad
            first = False
            continue
        cnt = nanm.sum(axis=0) + nanm.sum(axis=1)
        p = int(np.argmax(cnt[1:])) + 1
        dropped_null.append(cust.pop(p - 1))
    if not cust:
        raise RuntimeError("no customer survives the null filter")
    return {"dep": dep, "cust": cust, "dropped_snap": dropped_snap,
            "dropped_null": dropped_null, "replaced_depot": replaced}


# ---------------------------------------------------------------- demand
def make_demands(name: str, pops, cfg: dict) -> np.ndarray:
    """(n, 2) [delivery, pickup] kg:
      delivery = pop * DELIV_RATE * KG
      pickup   = pop * PICK_RATE * KG * U(0.5, 1.5)   (rng seeded from the name)
    both floored at 1.0 kg and rounded to 0.1 kg. Asserts max(d, p) <= Q/2."""
    rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
    pops = np.asarray(pops, dtype=float)
    d_raw = pops * cfg["deliv"] * cfg["kg"]
    p_raw = pops * cfg["pick"] * cfg["kg"] * rng.uniform(*PICKUP_SPREAD, len(pops))
    deliv = np.round(np.maximum(d_raw, 1.0), 1)
    pick = np.round(np.maximum(p_raw, 1.0), 1)
    dem = np.column_stack([deliv, pick])
    if len(dem) and dem.max() > cfg["Q"] / 2.0:
        raise RuntimeError(f"{name}: max demand {dem.max():.1f} kg > Q/2 = {cfg['Q'] / 2:.0f}")
    return dem


# ---------------------------------------------------------------- outputs
def process_area(region: str, area: Area, name: str, plan: Plan, parent: str) -> dict:
    t0 = time.time()
    cfg = REGION_CFG[region]
    coords = np.vstack([[plan.lat[plan.depot], plan.lon[plan.depot]],
                        np.column_stack([plan.lat, plan.lon])])
    pops = np.r_[0.0, plan.pop]
    M, snap, n_req = build_matrix(coords)
    sel = select_nodes(M, snap, coords, pops)
    idx = [sel["dep"]] + sel["cust"]
    D = M[np.ix_(idx, idx)]
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    if not np.isfinite(D).all():
        raise RuntimeError(f"{name}: non-finite distances after filtering")
    node_coords = coords[idx]
    node_pops = pops[idx].copy()
    node_pops[0] = 0.0
    dem = make_demands(name, node_pops[1:], cfg)
    Dint = np.round(D * SCALE).astype(np.int64)

    served = float(node_pops[1:].sum())
    n_cust = len(idx) - 1
    origin = f" (sub-area of {ascii_fold(parent)})" if parent else ""
    comment = (f"Kontur Population 2023-11-01 (CC BY 4.0); geoBoundaries gbOpen "
               f"(CC BY 4.0); OSRM road distances on OpenStreetMap (ODbL). "
               f"Region {region}, area {ascii_fold(area.name)}{origin}, area population "
               f"{plan.pop_area:.0f}, served {served:.0f} in {n_cust} customers, "
               f"depot = population-weighted medoid")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_vrpspd(DATA_DIR / f"{name}.vrpspd", name, Dint, dem, cfg["Q"], comment)
    sidecar = {"region": region, "area": area.name, "parent_area": parent,
               "depot": [float(node_coords[0, 0]), float(node_coords[0, 1])],
               "nodes": [[float(la), float(lo), float(p)]
                         for (la, lo), p in zip(node_coords, node_pops)]}
    (DATA_DIR / f"{name}.coords.json").write_text(json.dumps(sidecar), encoding="utf-8")

    off = D[~np.eye(len(D), dtype=bool)] if len(D) > 1 else np.zeros(1)
    row = {
        "name": name, "region": region, "area": area.name, "parent_area": parent,
        "n_customers": n_cust, "pop_area": round(plan.pop_area, 2),
        "pop_served": round(served, 2),
        "pop_dropped": round(plan.pop_area - served, 2),
        "pop_per_point_eff": round(plan.ppp, 2), "Q": cfg["Q"],
        "sum_delivery": round(float(dem[:, 0].sum()), 1),
        "max_delivery": round(float(dem[:, 0].max()), 1),
        "n_requests": n_req,
        "D_max_km": round(float(D.max()) / 1000.0, 3),
        "D_mean_km": round(float(off.mean()) / 1000.0, 3),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  wrote {name}: customers={n_cust} served={served:.0f} "
          f"dropped_snap={len(sel['dropped_snap'])} "
          f"dropped_null={len(sel['dropped_null'])} "
          f"replaced_depot={sel['replaced_depot']} requests={n_req}", flush=True)
    return row


def read_manifest() -> dict:
    p = DATA_DIR / "manifest.csv"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8", newline="") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def write_manifest(rows: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in MANIFEST_COLS})


def write_readme(rows: dict) -> None:
    totals = {}
    for r in rows.values():
        t = totals.setdefault(r["region"], {"inst": 0, "cust": 0, "pop": 0.0,
                                            "served": 0.0, "dropped": 0.0,
                                            "split": set()})
        t["inst"] += 1
        t["cust"] += int(r["n_customers"])
        t["pop"] += float(r["pop_area"])
        t["served"] += float(r["pop_served"])
        t["dropped"] += float(r["pop_dropped"])
        if r.get("parent_area"):
            t["split"].add(r["parent_area"])
    table = ["| Region | Instances | Customers | Area population | Served | Dropped | Parent areas split |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for reg in sorted(totals):
        t = totals[reg]
        table.append(f"| {reg} ({REGION_CFG[reg]['label']}) | {t['inst']} | {t['cust']} | "
                     f"{t['pop']:,.0f} | {t['served']:,.0f} | {t['dropped']:,.0f} | "
                     f"{len(t['split'])} |")
    if not totals:
        table.append("| (none generated yet) | | | | | | |")

    param_rows = ["| Region | Parameter | Value | Source / status |", "|---|---|---:|---|"]
    labels = {"deliv": "DELIV_RATE (parcels/person/day)", "pick": "PICK_RATE (parcels/person/day)",
              "kg": "KG per parcel", "Q": "Q (kg)", "ppp": "POP_PER_POINT"}
    for reg, cfg in REGION_CFG.items():
        for key in ("deliv", "pick", "kg", "Q", "ppp"):
            param_rows.append(f"| {reg} | {labels[key]} | {cfg[key]:g} | {cfg['src'][key]} |")
    text = f"""# Region-scale SVRPSPD instances

Generated by `scripts/make_region_instances.py` (one `.vrpspd` per depot area or
sub-area, Dethloff layout, readable by `dethloff_runner.parse_dethloff`). Raw
inputs and the OSRM response cache live in `_raw/` and are git-ignored.

Regions: `VNN` = Northern Vietnam (the 25 provinces of the 2008 boundaries,
the 2025 merger is not used); `GD` = Guangdong, taken as the geoBoundaries
ADM1 unit labelled "Guangzhou Province" with its CHN-ADM2 units as areas.

## Sources and licences

- Population: Kontur Population 2023-11-01, H3 resolution-8 hexagons, CC BY 4.0.
- Boundaries: geoBoundaries gbOpen (simplified), CC BY 4.0.
- Road distances: OSRM public table service over OpenStreetMap, ODbL.

## Demand and capacity parameters

Numbers marked ASSUMPTION are modelling choices, not measured data.

{chr(10).join(param_rows)}

Demand per service point: delivery = pop x DELIV_RATE x KG; pickup =
pop x PICK_RATE x KG x U(0.5, 1.5); both floored at 1.0 kg and rounded to 0.1 kg.
Asserted: max(delivery, pickup) <= Q / 2.

## Model assumptions

- Service points: population-weighted k-means of hex centres, k = max(10,
  round(P / POP_PER_POINT)); clusters under 100 people dropped; clusters above
  2 x POP_PER_POINT split once into ceil(pop / POP_PER_POINT) parts.
- Area splitting: an area still above 600 service points is partitioned by
  population-weighted k-means into ceil(n / 600) sub-areas, named `NAME-1`,
  `NAME-2`, ...; each sub-area is its own instance with its own depot. The
  manifest column `parent_area` records the parent.
- Depot: an extra node 0 at the population-weighted medoid point; that point
  stays a customer.
- Road distances: directed OSRM table; points whose snap distance exceeds 5 km
  dropped; unroutable points dropped; then symmetrised 0.5 (D + D^T) and written
  in 0.1 m units (`SCALE = 10`, as in `make_city_instances.py`).

## Regeneration

    cd svrpspd_wdro
    python scripts/make_region_instances.py --region all --dry-run   # no network
    python scripts/make_region_instances.py --region all            # OSRM download
    python scripts/make_region_instances.py --readme                # rewrite this file

## Region totals

{chr(10).join(table)}
"""
    (DATA_DIR / "README.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- driver
def _matches(pattern: str, area: Area, name: str) -> bool:
    p = pattern.lower()
    return any(fnmatch.fnmatchcase(s.lower(), p)
               for s in (area.name, slug(area.name), name))


def make_jobs(region: str, area: Area, name: str, lat, lon, pop, ppp: float):
    """Planning for one depot area: [(Area, instance name, Plan, parent)]."""
    plan = plan_area(lat, lon, pop, area.name, ppp)
    if len(plan.pop) <= N_MAX:
        return [(area, name, plan, "")], False
    parts = split_area(lat, lon, pop, area.name, ppp, len(plan.pop))
    jobs = []
    for j, (sub_name, sub) in enumerate(parts, 1):
        jobs.append((Area(name=sub_name, geom=None), f"{name}-{j}", sub, area.name))
    return jobs, True


def run_region(region: str, args):
    cfg = REGION_CFG[region]
    ppp = args.pop_per_point if args.pop_per_point else cfg["ppp"]
    print(f"[{region}] loading areas and hexes ...", flush=True)
    areas = load_areas(region)
    names = instance_names(region, areas)
    lat, lon, pop = load_hexes(region)
    owner, multi = assign_hexes(lat, lon, areas)
    outside = owner < 0
    print(f"[{region}] {len(areas)} areas, {len(pop)} hexes with population; "
          f"outside all areas: {int(outside.sum())} hexes, "
          f"{pop[outside].sum():,.0f} people; hexes in two areas: {multi}", flush=True)

    existing = read_manifest()
    rows, planned = [], []
    for i, (area, name) in enumerate(zip(areas, names)):
        if args.areas and not _matches(args.areas, area, name):
            continue
        sel = owner == i
        if not sel.any():
            print(f"  {name}: no hexes, skipped", flush=True)
            continue
        jobs, was_split = make_jobs(region, area, name, lat[sel], lon[sel], pop[sel], ppp)
        for job_area, job_name, plan, parent in jobs:
            if args.dry_run:
                planned.append((job_name, plan, parent))
                print(f"  {job_name:<26} hexes={plan.n_hex:>6} P={plan.pop_area:>13,.0f} "
                      f"k_a={plan.k_a:>5} ppp={plan.ppp:>8,.0f} "
                      f"points={len(plan.pop):>4} "
                      f"split={'yes' if was_split else 'no '} "
                      f"served_pre={plan.pop.sum():>13,.0f}", flush=True)
                continue
            out = DATA_DIR / f"{job_name}.vrpspd"
            if out.exists() and not args.force and job_name in existing:
                print(f"  {job_name}: exists, skipped (use --force)", flush=True)
                rows.append(existing[job_name])
                continue
            rows.append(process_area(region, job_area, job_name, plan, parent))
    if args.dry_run:
        return planned, outside, pop, multi, sorted({p for _, _, p in planned if p})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", choices=["VNN", "GD", "all"], default=None)
    ap.add_argument("--areas", default=None,
                    help="glob on area name, slug or instance name (case-insensitive)")
    ap.add_argument("--pop-per-point", type=float, default=None,
                    help="override POP_PER_POINT for every region (default: per region)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--readme", action="store_true")
    args = ap.parse_args(argv)

    if args.region is None:
        if args.readme:
            write_readme(read_manifest())
            print(f"rewrote {DATA_DIR / 'README.md'}")
            return 0
        ap.error("--region is required (VNN, GD or all)")

    regions = list(REGION_CFG) if args.region == "all" else [args.region]
    if args.dry_run:
        grand = {"inst": 0, "cust": 0, "pop": 0.0, "split": 0}
        for reg in regions:
            planned, outside, pop, multi, split_parents = run_region(reg, args)
            n_pts = sum(len(p.pop) for _, p, _ in planned)
            p_tot = sum(p.pop_area for _, p, _ in planned)
            print(f"[{reg}] DRY-RUN TOTALS: instances={len(planned)} "
                  f"customers(pre-OSRM)={n_pts} area_population={p_tot:,.0f} "
                  f"served_pre={sum(p.pop.sum() for _, p, _ in planned):,.0f}")
            print(f"[{reg}] parent areas split ({len(split_parents)}): "
                  f"{', '.join(split_parents) or 'none'}")
            print(f"[{reg}] hexes outside all areas: {int(outside.sum())} "
                  f"({pop[outside].sum():,.0f} people)")
            grand["inst"] += len(planned)
            grand["cust"] += n_pts
            grand["pop"] += p_tot
            grand["split"] += len(split_parents)
        print(f"DRY-RUN GRAND TOTAL: instances={grand['inst']} "
              f"customers(pre-OSRM)={grand['cust']} population={grand['pop']:,.0f} "
              f"parent_areas_split={grand['split']}")
        return 0

    rows_all = read_manifest()
    for reg in regions:
        for r in run_region(reg, args):
            rows_all[r["name"]] = r
    write_manifest(rows_all)
    write_readme(rows_all)
    print(f"manifest and README updated: {len(rows_all)} instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
