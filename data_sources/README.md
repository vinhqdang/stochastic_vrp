# Real data sources — verified, credential-free

Feasibility probes for grounding uncertainty models in **measured** data
instead of synthetic noise. Not tied to any one paper: papers 1–3 are frozen
and paper 4 was abandoned, so this exists to serve whatever comes next.

Every row in the first table was fetched successfully from this container,
headlessly, with no account and no API key. Failures are listed too — a short
verified list beats a long speculative one.

## Verified working

| # | Source | What it measures | Coverage | Licence |
|---|---|---|---|---|
| **1** | **NYC DOT Traffic Speeds** (Socrata `i4gi-tjb9`) | **per-link `speed` (mph) *and* `travel_time` (s)**, plus `link_points` WGS84 polyline → **map-matchable to OSM edges** | NYC, 125 active links, 2017-07 → live, ~110 M rows, 1–5 min resolution | **none stated** (`license: None`) — NYC ToU only |
| **2** | **METR-LA / PEMS-BAY** (Zenodo `10.5281/zenodo.5146275`) | loop-detector speeds (mph), 5-min | LA 207 sensors (2012-03→06); Bay Area 325 sensors (from 2017-01) | **CC BY 4.0** — cleanest licence here |
| **3** | **HCMC segment velocities** (Kaggle `thanhnguyen2612`) | segment velocity (km/h) + LOS A–F per half-hour, keyed by **real OSM node IDs** | **Ho Chi Minh City**, 2020-07→2021-04, 122 days, 90,938 obs / 10,027 segments, 8.4 MB | **CC0** — fully redistributable |
| **4** | **NYC TLC trip records** (CloudFront parquet) | per-trip durations, distance, **OD taxi-zone IDs** | NYC, 2009 → 2026-01 monthly; yellow ~3.5 M trips/month, ~50–70 MB | none stated (accuracy disclaimer only) |
| **5** | **Open-Meteo archive** | hourly temperature, precipitation, snowfall, wind at **any lat/lon** | global, so all five `City/` instances | CC BY 4.0, attribution required (ERA5 underneath) |
| **6** | **Madrid histórico** | `vmed` mean speed (km/h), 15-min | Madrid 2013→2026, but speed only on 366 M-30 ring points | CC BY 4.0 |
| **7** | **Berlin Verkehrsdetektion** | hourly speed + flow **by vehicle class** | Berlin, 2015→2024, 292 detectors | dl-de/by-2.0 |
| **8** | **UCI Porto taxi trajectories** | 1.7 M trips, 15-s GPS polylines → derivable link times | Porto, 2013-07→2014-06, 534 MB | CC BY 4.0 |

Verified endpoints:

```
https://data.cityofnewyork.us/resource/i4gi-tjb9.json?$limit=50000&$select=link_id,speed,travel_time,data_as_of,link_points
https://zenodo.org/api/records/5146275/files/METR-LA.csv/content
https://www.kaggle.com/api/v1/datasets/download/thanhnguyen2612/traffic-flow-data-in-ho-chi-minh-city-viet-nam
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
https://archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..&hourly=temperature_2m,precipitation,snowfall,wind_speed_10m
```

### Correction: when TLC actually has coordinates

An earlier version of this file said recent TLC data is zone-level and older
years had coordinates. That was too vague, and the widely repeated version of
it is wrong. Probing parquet footers by HTTP range across the archive:

- `2009-*` → `Start_Lon, Start_Lat, End_Lon, End_Lat`
- `2010-*` → `pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude`
- **`2011-01` onward → `PULocationID` / `DOLocationID` only.** Confirmed for
  2011–2016 (including **2016-06**), 2025 and 2026-01. Green is zone-only even
  for 2014-01.

The common belief that lat/lon persists through mid-2016 is true of the
**retired original CSVs**, not of what CloudFront serves today. So
map-matchable raw coordinates exist only for **2009–2010**; everything since is
263 taxi zones, i.e. zone-pair OD travel times, never link-level. **Source #1
is the answer to link-level travel time, not TLC.**

### Licence caution for a data-availability statement

COAP requires one. Note the asymmetry: NYC DOT and TLC state **no licence**
(usage is by NYC terms of use and universal practice, not an explicit grant),
whereas METR-LA/PEMS-BAY is CC BY 4.0 and the HCMC set is CC0. If a processed
extract needs to be committed to a public repo, prefer #2 and #3, and cite #1
by URL rather than redistributing bulk rows.

## What the NYC probe establishes

`python3 data_sources/probe_nyc_traffic_weather.py --month 2025-01`

On January 2025 (3,355,999 clean trips, 734 hours):

- **Strong diurnal congestion.** 19.1 mph at 05:00 against 9.8 mph at 17:00 —
  a 1.94× swing. Deterministic seasonality; remove it before calling anything
  uncertainty.
- **Enough data for genuine empirical distributions.** 24,837
  (origin, destination, hour) cells with ≥30 observations; 8,183 with ≥100. At
  fixed OD and fixed hour, duration still has median coefficient of variation
  0.301 (IQR 0.233–0.392) — real, irreducible dispersion.
- **Weather matters, and time-of-day hides it.** Raw precipitation–speed
  correlation is only −0.079; after removing hour-of-week means it roughly
  doubles to −0.195. Wet hours run **1.04 mph slower** than the same
  hour-of-week (Welch t = −5.15, p = 6.9×10⁻⁶); snow 0.90 mph slower
  (t = −4.99, p = 1.7×10⁻⁵).
- **But weather shifts the mean without inflating the spread.** Residual
  dispersion is 1.191 mph wet versus 1.303 dry — slightly *lower*. Weather
  behaves like **context conditioning the distribution**, not an extra
  uncertainty dimension that widens an ambiguity set. Different model.
- **Wind is not a clean channel.** Residual correlation is *positive* (+0.191).
  Confounded (clear cold nights, lower volume in storms). Do not present it as
  a congestion driver without a mechanism.

## Recommended pipeline to an empirical ambiguity-set centre

Using #1 as primary and #2 as the licence-clean secondary:

1. Page `i4gi-tjb9` at 50k rows/call selecting `link_id, speed, travel_time,
   data_as_of, link_points`; store parquet, not JSON.
2. **Map-match once.** Decode `link_points` to a LineString and match to OSMnx
   edges by Hausdorff distance + bearing agreement. Only 125 links — verify by
   hand and commit the link→edge map as JSON. That kills the matching-error
   objection.
3. **Normalise to a unit-free congestion multiplier**
   `r = travel_time / (length_osm / speed_freeflow)`, free-flow being the
   per-link 90th-percentile speed. This is the step that lets a distribution
   estimated on NYC transfer to Hanoi/Paris/Shanghai edges of the same OSM
   `highway` class — raw NYC seconds do not transfer.
4. Bucket by (`highway` class, hour-of-day, weekday/weekend) and form
   $\hat P_N = \frac1N\sum_i \delta_{r_i}$ per bucket. Set the radius by the
   $\varepsilon_N \propto N^{-1/2}$ rate, or better, calibrate by held-out-month
   cross-validation.
5. **Clean aggressively and state the rule.** Drop `speed <= 0`,
   `travel_time <= 0`, and junk timestamps — `min(data_as_of)` is **1930-12-09**,
   so the table genuinely contains garbage.
6. **For HCMC use #3 to validate, not as the primary sample.** It is CC0 and
   keyed by genuine OSM node IDs (two nodes were checked against the live OSM
   API and matched to 7 decimals), so it joins onto what
   `make_city_instances.py` already builds — but it is COVID-era, 122 days,
   ~9 observations per segment, with velocities quantised (modal value
   1 km/h).

For **Paris there is no honest open path** to a travel-time distribution; the
transferred-multiplier step above is what keeps Paris in the instance set
without inventing data.

## Failed verification — do not plan around these

| Candidate | Outcome |
|---|---|
| **Caltrans PeMS** | still login-gated; accounts need approval |
| **UTD19** | signup form, link emailed; `utd19.ethz.ch/opendata.html` → 404 |
| **LargeST** | downloads unauthenticated but **7.56 GB**, and it is flow not speed, CC BY-**NC** |
| **Uber Movement** | **dead** — decommissioned, no mirror found on GitHub/Zenodo/Dataverse; would have had HCMC + Paris |
| **Paris / Île-de-France** | flow and occupancy only; the sole speed table is 60 city-wide monthly means |
| **Barcelona** | ordinal congestion 0–6, no km/h; bulk download captcha-walled |
| **Hamburg** | categorical only; archive zips are 85-byte stubs |
| **Netherlands NDW** | live snapshot fine, but no fetchable archive (host unresolvable / 401) |
| **Vietnam government portals** | nothing — HCMC portal has no transport topic; Hanoi and national portals unresolvable |
| **IEEE DataPort GRACTRANET** | real HCMC/Hanoi Waze travel times but subscription-gated |
| **Grab-Posisi** | email-gated, and Singapore/Jakarta only |
| **TfL LCAP** | data fetches (200) but **licence unresolved** through this proxy — verify before use |
| **HERE / TomTom / Google / Mapbox** | **all four forbid what we need.** Google bars storing/resharing durations; HERE §6.4(b) explicitly bars exposing content under open-source/open-data licences, i.e. a public repo; TomTom bars derived databases and forces 60-day deletion; Mapbox bars redistribution and bulk queries. No academic tier at HERE or TomTom |
