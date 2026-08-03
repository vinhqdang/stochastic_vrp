"""Build empirical congestion-multiplier distributions from real NYC link data.

Turns the NYC DOT Traffic Speeds feed into the object a stochastic or robust
routing model actually needs: for each (road segment, hour-of-day, weather
state), an empirical distribution over the **congestion multiplier**

    r = observed travel time / free-flow travel time

Working in multipliers rather than raw seconds is the step that makes the
estimate portable. Raw NYC seconds mean nothing on a Hanoi edge; a distribution
over "this class of road runs 1.8x its free-flow time at 17:00 in the rain"
transfers to any network where free-flow time is known, which is exactly what
OSM gives us. It also removes link length as a nuisance dimension, so
observations from short and long links pool.

Deliberately requires **no OSM/Overpass access**: free-flow speed is estimated
per link from that link's own observed distribution (a high percentile), and
link length comes from the feed's own `link_points` polyline via haversine. So
this runs even when Overpass is unreachable, which it was when this was written.
Map-matching to OSM edges is a separate, later step needed only to *transfer*
these distributions to other cities.

Sources, both credential-free (see `data_sources/README.md`):
  - NYC DOT Traffic Speeds, Socrata `i4gi-tjb9`
  - Open-Meteo historical archive (hourly precipitation)

Run:
    python3 data_sources/build_nyc_link_distributions.py --start 2025-01-01 \
        --end 2025-01-31 --out data_sources/out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

SOCRATA = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"
WX = ("https://archive-api.open-meteo.com/v1/archive"
      "?latitude=40.7128&longitude=-74.0060&start_date={s}&end_date={e}"
      "&hourly=precipitation,snowfall&timezone=America%2FNew_York")
PAGE = 50000


def fetch_json(url: str, cache: str | None = None):
    if cache and os.path.exists(cache) and os.path.getsize(cache) > 0:
        return json.load(open(cache))
    with urllib.request.urlopen(url, timeout=180) as r:
        data = json.load(r)
    if cache:
        json.dump(data, open(cache, "w"))
    return data


def pull_speeds(start: str, end: str, cachedir: str) -> pd.DataFrame:
    """Page the Socrata feed over a date range."""
    frames, offset = [], 0
    where = (f"data_as_of >= '{start}T00:00:00' AND data_as_of < '{end}T23:59:59'")
    while True:
        q = {
            "$select": "link_id,speed,travel_time,data_as_of,link_points,borough",
            "$where": where,
            "$order": "data_as_of",
            "$limit": PAGE,
            "$offset": offset,
        }
        url = SOCRATA + "?" + urllib.parse.urlencode(q)
        cache = os.path.join(cachedir, f"dot_{start}_{end}_{offset}.json")
        rows = fetch_json(url, cache)
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        print(f"    page offset {offset:>8}: {len(rows):>6} rows")
        if len(rows) < PAGE:
            break
        offset += PAGE
    if not frames:
        raise SystemExit("no rows returned; check the date range")
    return pd.concat(frames, ignore_index=True)


def polyline_length_mi(link_points: str) -> float:
    """Haversine length of a 'lat,lon lat,lon ...' polyline, in miles."""
    pts = []
    for tok in str(link_points).replace(",", " ").split():
        try:
            pts.append(float(tok))
        except ValueError:
            return np.nan
    if len(pts) < 4:
        return np.nan
    lat, lon = pts[0::2], pts[1::2]
    # Some polylines carry an odd token count (a truncated trailing pair), which
    # leaves the two lists unequal; drop the dangling coordinate.
    n = min(len(lat), len(lon))
    lat, lon = lat[:n], lon[:n]
    if n < 2:
        return np.nan
    tot = 0.0
    for i in range(n - 1):
        p1, p2 = math.radians(lat[i]), math.radians(lat[i + 1])
        dphi, dlam = p2 - p1, math.radians(lon[i + 1] - lon[i])
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
        tot += 3958.7613 * 2 * math.asin(min(1.0, math.sqrt(a)))
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-01-07")
    ap.add_argument("--out", default="data_sources/out")
    ap.add_argument("--cache", default=os.environ.get("PROBE_CACHE", "/tmp/nycdot"))
    ap.add_argument("--freeflow-pct", type=float, default=90.0,
                    help="percentile of a link's own speeds taken as free flow")
    ap.add_argument("--min-obs", type=int, default=200,
                    help="observations a link needs before it is kept")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    print("=" * 76)
    print(f"NYC link congestion multipliers  {a.start} .. {a.end}")
    print("=" * 76)

    print("  pulling NYC DOT speeds")
    df = pull_speeds(a.start, a.end, a.cache)
    print(f"  raw rows: {len(df):,}")

    for c in ("speed", "travel_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["data_as_of"], errors="coerce")

    # The feed genuinely contains garbage: min(data_as_of) in the full table is
    # 1930-12-09, and zero/negative speeds and travel times appear. State the
    # exclusion rule in any write-up.
    n0 = len(df)
    df = df.dropna(subset=["speed", "travel_time", "ts"])
    df = df[(df.speed > 0) & (df.travel_time > 0)]
    df = df[(df.ts >= a.start) & (df.ts <= f"{a.end}T23:59:59")]
    df = df[df.speed < 100]
    print(f"  after cleaning: {len(df):,} ({100*len(df)/max(n0,1):.1f}% kept)")

    # ---- per-link geometry and free-flow reference ------------------------
    geo = df.groupby("link_id").link_points.first().apply(polyline_length_mi)
    ff = df.groupby("link_id").speed.quantile(a.freeflow_pct / 100.0)
    cnt = df.groupby("link_id").size()
    keep = cnt[cnt >= a.min_obs].index.intersection(
        geo[(geo > 0.05) & geo.notna()].index)
    df = df[df.link_id.isin(keep)].copy()
    print(f"  links kept: {len(keep)} of {cnt.size} "
          f"(>= {a.min_obs} obs and usable geometry)")

    df["len_mi"] = df.link_id.map(geo)
    df["ff_mph"] = df.link_id.map(ff)
    # free-flow seconds from geometry and the link's own uncongested speed
    df["ff_sec"] = df.len_mi / df.ff_mph * 3600.0
    df["r"] = df.travel_time / df.ff_sec
    # A multiplier below 1 just means the observation beat the 90th-percentile
    # reference; keep a little slack but drop the physically absurd tail.
    df = df[(df.r > 0.3) & (df.r < 20.0)]

    # ---- weather join -----------------------------------------------------
    w = fetch_json(WX.format(s=a.start, e=a.end),
                   os.path.join(a.cache, f"wx_{a.start}_{a.end}.json"))["hourly"]
    wx = pd.DataFrame({"hr_ts": pd.to_datetime(w["time"]),
                       "precip": [x or 0 for x in w["precipitation"]],
                       "snow": [x or 0 for x in w["snowfall"]]})
    df["hr_ts"] = df.ts.dt.floor("h")
    df = df.merge(wx, on="hr_ts", how="left")
    df["wet"] = (df.precip.fillna(0) > 0.1) | (df.snow.fillna(0) > 0.05)
    df["hour"] = df.ts.dt.hour

    print(f"\n  usable observations: {len(df):,}   wet share "
          f"{100*df.wet.mean():.1f}%")
    print(f"  congestion multiplier r: median {df.r.median():.3f}, "
          f"p90 {df.r.quantile(.9):.3f}, p99 {df.r.quantile(.99):.3f}")

    # ---- the deliverable: empirical distributions per cell ----------------
    cell = df.groupby(["hour", "wet"]).r.agg(
        n="size", mean="mean", sd="std", p50="median",
        p90=lambda s: s.quantile(0.90), p99=lambda s: s.quantile(0.99))
    cell["cv"] = cell["sd"] / cell["mean"]
    print("\n--- congestion multiplier by hour x weather ---")
    print(f"  {'hr':>3} {'wet':>5} {'n':>8} {'mean':>7} {'sd':>6} {'p90':>6} {'cv':>6}")
    for (h, wet), row in cell.iterrows():
        if h % 3 == 0:
            print(f"  {h:>3} {str(wet):>5} {int(row['n']):>8} {row['mean']:>7.3f} "
                  f"{row['sd']:>6.3f} {row['p90']:>6.3f} {row['cv']:>6.3f}")

    dry = cell.xs(False, level="wet"); wetc = cell.xs(True, level="wet")
    common = dry.index.intersection(wetc.index)
    if len(common):
        dm = (wetc.loc[common, "mean"] - dry.loc[common, "mean"]).mean()
        dcv = (wetc.loc[common, "cv"] - dry.loc[common, "cv"]).mean()
        print(f"\n  wet - dry, averaged over hours:  mean multiplier {dm:+.4f}, "
              f"CV {dcv:+.4f}")
        print("  A positive mean shift with a ~zero or negative CV shift is the")
        print("  same pattern the trip-level probe found: weather moves the")
        print("  centre, not the spread.")

    outp = os.path.join(a.out, f"nyc_multipliers_{a.start}_{a.end}.parquet")
    df[["link_id", "ts", "hour", "wet", "borough", "len_mi", "ff_mph",
        "travel_time", "r"]].to_parquet(outp, index=False)
    cell.to_csv(os.path.join(a.out, f"nyc_cells_{a.start}_{a.end}.csv"))
    print(f"\n  wrote {outp}")
    print(f"  wrote {os.path.join(a.out, f'nyc_cells_{a.start}_{a.end}.csv')}")
    print("=" * 76)


if __name__ == "__main__":
    main()
