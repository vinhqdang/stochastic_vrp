"""Feasibility probe: real NYC travel times joined to real hourly weather.

Establishes, against live public endpoints and with no credentials, that we can
build **empirical travel-time distributions from measured data** rather than
synthetic noise — and quantifies how much of the variation weather actually
explains once time-of-day is controlled for.

Two sources, both verified reachable headlessly from this container:

  1. **NYC TLC trip records** — yellow-taxi monthly parquet on CloudFront.
     No account, no API key. ~50-70 MB/month, ~3.5 M trips/month.
     https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
     Gives per-trip pickup/dropoff timestamps, trip distance, and origin /
     destination **taxi-zone IDs** (`PULocationID` / `DOLocationID`, 263
     zones), *not* lat/lon — so trips give zone-pair OD travel times and can
     never be matched to individual OSM edges. Raw coordinates survive only in
     the 2009 and 2010 files; from 2011-01 onward every month is zone-only
     (the belief that lat/lon persists to mid-2016 refers to the retired CSVs,
     not to what CloudFront serves now). For link-level travel time use the
     NYC DOT Traffic Speeds feed instead — see `data_sources/README.md`.

  2. **Open-Meteo historical archive** — hourly reanalysis, no API key.
     https://archive-api.open-meteo.com/v1/archive
     Temperature, precipitation, snowfall, wind at any lat/lon, so the same
     call works for every city in `svrpspd_wdro/data/City/`.

What the probe reports, and why each piece matters for an ambiguity set:

  * diurnal speed profile — the deterministic seasonality to remove first;
  * count of (origin zone, destination zone, hour) cells with enough
    observations to form a genuine empirical distribution;
  * within-cell coefficient of variation — the dispersion a robust model
    would actually hedge against;
  * the weather effect on speed **after** removing hour-of-week means, which
    is the only version of that effect worth reporting: raw correlations are
    confounded by the fact that it rains at all hours but traffic is not
    uniform across them;
  * whether weather inflates *dispersion* or merely shifts the *mean* —
    these imply different models, and the answer here is not the obvious one.

Run:  python3 data_sources/probe_nyc_traffic_weather.py [--month 2025-01]
Needs: pyarrow, pandas, numpy, scipy, requests-free (uses urllib).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

TLC_URL = ("https://d37ci6vzurychx.cloudfront.net/trip-data/"
           "yellow_tripdata_{month}.parquet")
WX_URL = ("https://archive-api.open-meteo.com/v1/archive"
          "?latitude={lat}&longitude={lon}"
          "&start_date={start}&end_date={end}"
          "&hourly=temperature_2m,precipitation,snowfall,wind_speed_10m"
          "&timezone={tz}")

# Manhattan; TLC zone centroids would be finer but the weather field is
# essentially uniform at city scale for this purpose.
NYC = dict(lat=40.7128, lon=-74.0060, tz="America%2FNew_York")


def fetch(url: str, path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    urllib.request.urlretrieve(url, path)
    return path


def load_trips(month: str, cache: str) -> pd.DataFrame:
    """Clean per-trip durations and implied speeds for one month."""
    path = fetch(TLC_URL.format(month=month), os.path.join(cache, f"tlc_{month}.parquet"))
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
            "PULocationID", "DOLocationID"]
    d = pq.read_table(path, columns=cols).to_pandas()
    d = d.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    d["dur"] = (d.tpep_dropoff_datetime - d.tpep_pickup_datetime).dt.total_seconds() / 60.0
    # Filters: drop zero/absurd durations, zero distances, and speeds that are
    # physically implausible for Manhattan surface streets. These are standard
    # for the TLC feed, which contains meter errors and clock glitches.
    d = d[(d.dur > 0.5) & (d.dur < 180) & (d.trip_distance > 0.1)].copy()
    d["mph"] = d.trip_distance / (d.dur / 60.0)
    d = d[(d.mph > 0.5) & (d.mph < 80)].copy()
    d["ts"] = d.tpep_pickup_datetime.dt.floor("h")
    start, end = f"{month}-01", f"{month}-01"
    lo = pd.Timestamp(start)
    hi = lo + pd.offsets.MonthBegin(1)
    return d[(d.ts >= lo) & (d.ts < hi)]


def load_weather(month: str, cache: str) -> pd.DataFrame:
    lo = pd.Timestamp(f"{month}-01")
    hi = lo + pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)
    url = WX_URL.format(start=lo.date(), end=hi.date(), **NYC)
    path = fetch(url, os.path.join(cache, f"wx_{month}.json"))
    h = json.load(open(path))["hourly"]
    return pd.DataFrame({
        "ts": pd.to_datetime(h["time"]),
        "precip": [x or 0 for x in h["precipitation"]],
        "snow": [x or 0 for x in h["snowfall"]],
        "temp": [x if x is not None else np.nan for x in h["temperature_2m"]],
        "wind": [x or 0 for x in h["wind_speed_10m"]],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2025-01")
    ap.add_argument("--cache", default=os.environ.get("PROBE_CACHE", "/tmp/wrnf_probe"))
    ap.add_argument("--min-cell", type=int, default=30,
                    help="observations needed before an (OD,hour) cell counts")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    print("=" * 74)
    print(f"NYC traffic x weather feasibility probe — {args.month}")
    print("=" * 74)

    d = load_trips(args.month, args.cache)
    print(f"clean trips: {len(d):,}")

    # ---- 1. diurnal profile: the seasonality to remove before modelling ----
    d["hour"] = d.tpep_pickup_datetime.dt.hour
    prof = d.groupby("hour").mph.agg(["mean", "std", "size"]).round(2)
    print("\n--- mean speed by hour of day (mph) ---")
    print(f"  fastest hour {prof['mean'].idxmax():02d}:00 at {prof['mean'].max():.1f}"
          f"   slowest hour {prof['mean'].idxmin():02d}:00 at {prof['mean'].min():.1f}"
          f"   ratio {prof['mean'].max()/prof['mean'].min():.2f}x")

    # ---- 2. do we have enough data for genuine empirical distributions? ----
    cell = d.groupby(["PULocationID", "DOLocationID", "hour"]).dur.agg(["count", "mean", "std"])
    rich = cell[cell["count"] >= args.min_cell]
    cv = (rich["std"] / rich["mean"]).dropna()
    print(f"\n--- (origin, destination, hour) cells ---")
    print(f"  cells with >={args.min_cell} obs : {len(rich):,}")
    print(f"  cells with >=100 obs      : {(cell['count'] >= 100).sum():,}")
    print(f"  within-cell CV of duration: median {cv.median():.3f}, "
          f"IQR [{cv.quantile(.25):.3f}, {cv.quantile(.75):.3f}]")
    print("  => real dispersion at fixed OD and fixed hour; this is the")
    print("     uncertainty a robust/stochastic model has to price.")

    # ---- 3. weather, controlling for hour-of-week ----
    hr = d.groupby("ts").agg(mph=("mph", "mean"), n=("mph", "size")).reset_index()
    hr = hr[hr.n >= 200]
    m = hr.merge(load_weather(args.month, args.cache), on="ts")
    m["hour"], m["dow"] = m.ts.dt.hour, m.ts.dt.dayofweek
    # Removing the hour-of-week mean is the key step: raw correlations
    # understate the effect because time-of-day dominates the variance.
    m["resid"] = m.mph - m.groupby(["hour", "dow"]).mph.transform("mean")

    print(f"\n--- weather vs speed ({len(m)} hours with >=200 trips) ---")
    print(f"  {'channel':8} {'raw corr':>10} {'residual corr':>15}")
    for v in ["precip", "snow", "wind"]:
        print(f"  {v:8} {m.mph.corr(m[v]):>+10.3f} {m.resid.corr(m[v]):>+15.3f}")
    print("  (residual = after removing hour-of-week means; the raw column is")
    print("   confounded and understates precipitation)")

    wet, dry = m[m.precip > 0.1], m[m.precip <= 0.01]
    t, p = stats.ttest_ind(wet.resid, dry.resid, equal_var=False)
    print(f"\n  wet hours (n={len(wet)}) vs dry (n={len(dry)}):")
    print(f"    slowdown vs same hour-of-week: "
          f"{wet.resid.mean() - dry.resid.mean():+.3f} mph  "
          f"(Welch t={t:.2f}, p={p:.3g})")
    sn = m[m.snow > 0.05]
    if len(sn) > 3:
        t2, p2 = stats.ttest_ind(sn.resid, dry.resid, equal_var=False)
        print(f"    snow hours (n={len(sn)}): "
              f"{sn.resid.mean() - dry.resid.mean():+.3f} mph "
              f"(Welch t={t2:.2f}, p={p2:.3g})")

    # ---- 4. mean shift or variance inflation? different models follow ----
    print(f"\n  dispersion of residual speed: wet {wet.resid.std():.3f} "
          f"vs dry {dry.resid.std():.3f}")
    if wet.resid.std() <= dry.resid.std():
        print("  => weather shifts the MEAN but does not inflate the spread.")
        print("     Argues for treating weather as CONTEXT conditioning the")
        print("     distribution, not as an extra uncertainty dimension that")
        print("     widens an ambiguity set.")
    else:
        print("  => weather inflates dispersion as well as shifting the mean.")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
