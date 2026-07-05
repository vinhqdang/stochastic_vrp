#!/usr/bin/env python3
"""
make_city_instances.py — Generate large SVRPSPD instances from real city maps.

The Dethloff benchmark tops out at 50 customers. This script builds
100-400-customer instances on the REAL drive networks of Ho Chi Minh City
and Hanoi (OpenStreetMap): customers are sampled at actual street
intersections, the depot sits near the city centre, and the distance
matrix contains true shortest-path road distances (metres) computed on the
directed drive graph.

Demands follow the Dethloff convention (delivery + pickup per customer):
delivery volumes are gamma-distributed around a parcel-van scale and each
customer returns a pickup that is a uniform fraction of its delivery —
the simultaneous-service structure that makes mid-route capacity peaks
(and therefore OTR-style execution policies) non-trivial.

Output format is exactly the Dethloff .vrpspd layout (DIMENSION, CAPACITY,
EDGE_WEIGHT_SECTION full matrix, PICKUP_AND_DELIVERY_SECTION), so
`dethloff_runner.parse_dethloff` and every evaluation script consume the
new instances unchanged.

Usage:
    python scripts/make_city_instances.py [sizes=100,200,400] [seeds=1]
                                          [cities=hcmc,hanoi] [out=data/City]

Requires network access to OpenStreetMap (osmnx). The downloaded graphs
are cached in data/City/_cache via osmnx's own cache.
"""

import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent

CITIES = {
    # name: (lat, lon of depot ~ central logistics area, radius m)
    "hcmc":  (10.7769, 106.7009, 6000),   # Ho Chi Minh City, District 1 centre
    "hanoi": (21.0285, 105.8542, 6000),   # Hanoi, Hoan Kiem centre
}

# demand model (kg): parcel-van operations
DELIV_MEAN  = 8.0     # mean delivery weight per customer
DELIV_CV    = 0.6     # spread of mean weights ACROSS customers
PICKUP_FRAC = (0.2, 0.8)   # per-customer pickup = U(a,b) x delivery
Q_VEHICLE   = 150.0   # vehicle capacity (kg) -> ~ 12-18 stops per route
# distances are written in 0.1 m units so that the pipeline's fixed
# divisor of 10^4 (dethloff_runner.parse_dethloff) yields kilometres:
# metres * 10 / 10^4 = km
SCALE       = 10


def _load_graph(city: str):
    import osmnx as ox
    lat, lon, radius = CITIES[city]
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(_WDRO / "data" / "City" / "_cache")
    G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
    G = ox.truncate.largest_component(G, strongly=True)
    return G, (lat, lon)


def _distance_matrix(G, node_ids: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path road distances (m) between the chosen nodes,
    computed with scipy's Dijkstra on the full graph adjacency."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    nodes = list(G.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for u, v, data in G.edges(data=True):
        rows.append(idx[u]); cols.append(idx[v])
        vals.append(float(data.get("length", 1.0)))
    A = coo_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()

    src = np.array([idx[v] for v in node_ids])
    dist = dijkstra(A, directed=True, indices=src)
    D = dist[:, src]
    if not np.isfinite(D).all():
        raise RuntimeError("disconnected node pair despite SCC restriction")
    return np.round(D * SCALE).astype(np.int64)


def _sample_nodes(G, depot_latlon, n_cust: int, rng) -> np.ndarray:
    """Depot = street node nearest the city-centre point; customers are a
    uniform sample (without replacement) of the remaining street nodes."""
    import osmnx as ox
    depot = ox.distance.nearest_nodes(G, X=depot_latlon[1], Y=depot_latlon[0])
    others = np.array([v for v in G.nodes if v != depot])
    cust = rng.choice(others, size=n_cust, replace=False)
    return np.concatenate([[depot], cust])


def _demands(n_cust: int, rng) -> np.ndarray:
    """(n_cust, 2) array: [:,0]=delivery, [:,1]=pickup (kg, >= 1)."""
    k = 1.0 / (DELIV_CV ** 2)
    deliv = rng.gamma(k, DELIV_MEAN / k, n_cust)
    frac = rng.uniform(*PICKUP_FRAC, n_cust)
    pick = deliv * frac
    return np.column_stack([np.maximum(deliv, 1.0),
                            np.maximum(pick, 1.0)]).round(1)


def write_vrpspd(path: Path, name: str, D: np.ndarray, dem: np.ndarray,
                 Q: float, comment: str) -> None:
    n = D.shape[0]
    lines = [
        f"NAME : {name}",
        f"COMMENT : {comment}",
        "TYPE : VRPSPD",
        f"DIMENSION : {n}",
        f"CAPACITY : {Q:.0f}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    for i in range(n):
        lines.append(" ".join(str(int(x)) for x in D[i]))
    lines.append("PICKUP_AND_DELIVERY_SECTION")
    lines.append("1 0 0")                       # depot: no demand
    for i in range(1, n):
        d, p = dem[i - 1]
        lines.append(f"{i + 1} {d:.1f} {p:.1f}")
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n")


def main():
    sizes  = [100, 200, 400]
    seeds  = [1]
    cities = list(CITIES)
    out    = _WDRO / "data" / "City"

    for arg in sys.argv[1:]:
        if   arg.startswith("sizes="):  sizes  = [int(x) for x in arg[6:].split(",")]
        elif arg.startswith("seeds="):  seeds  = [int(x) for x in arg[6:].split(",")]
        elif arg.startswith("cities="): cities = arg[7:].split(",")
        elif arg.startswith("out="):    out    = Path(arg[4:])

    out.mkdir(parents=True, exist_ok=True)
    for city in cities:
        print(f"[{city}] downloading drive network ...", flush=True)
        t0 = time.time()
        G, centre = _load_graph(city)
        print(f"[{city}] {len(G.nodes):,} nodes, {len(G.edges):,} edges "
              f"({time.time()-t0:.0f}s)")

        for n_cust in sizes:
            for seed in seeds:
                rng = np.random.default_rng(10_000 * n_cust + seed)
                node_ids = _sample_nodes(G, centre, n_cust, rng)
                t0 = time.time()
                D = _distance_matrix(G, node_ids)
                dem = _demands(n_cust, rng)
                name = f"{city.upper()}-{n_cust}-{seed}"
                write_vrpspd(out / f"{name}.vrpspd", name, D, dem, Q_VEHICLE,
                             f"OSM drive network, depot {centre}, seed {seed}")
                print(f"  wrote {name}.vrpspd  (n={n_cust}, "
                      f"D in [{D[D>0].min()}, {D.max()}] m, "
                      f"{time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
