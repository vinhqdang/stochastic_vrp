# Real data sources — verified, credential-free

Feasibility probes for grounding uncertainty models in **measured** data
instead of synthetic noise. Not tied to any one paper: papers 1–3 are frozen
and paper 4 was abandoned, so this exists to serve whatever comes next.

Everything here was verified reachable from this container, headlessly, with
no account and no API key.

## Verified endpoints

| Source | What it gives | Access | Size |
|---|---|---|---|
| **NYC TLC trip records** — `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet` | per-trip pickup/dropoff timestamps, trip distance, origin/destination **taxi-zone IDs** | HTTP 200, no credentials | 48–70 MB/month, ~3.5 M trips |
| **Open-Meteo archive** — `https://archive-api.open-meteo.com/v1/archive` | hourly temperature, precipitation, snowfall, wind at any lat/lon | HTTP 200, **no API key** | trivial (JSON) |

Open-Meteo takes arbitrary coordinates, so the same call covers every city in
`svrpspd_wdro/data/City/` — Ho Chi Minh City, Hanoi, New York, Paris,
Shanghai — not just NYC.

### Important limitation, stated up front

Recent TLC data is **zone-level**, not lat/lon: `PULocationID` /
`DOLocationID` over 263 taxi zones. Older years had coordinates; current ones
do not. So these records support **zone-pair OD travel-time distributions**,
and cannot be map-matched to individual OSM edges. Any claim about per-link
travel times would need a different source (loop detectors, probe speeds).

## What the probe establishes

`python3 data_sources/probe_nyc_traffic_weather.py --month 2025-01`

Results on January 2025 (3,355,999 clean trips, 734 hours):

- **Strong diurnal congestion.** Mean speed 19.1 mph at 05:00 against 9.8 mph
  at 17:00 — a 1.94× swing. This is deterministic seasonality and must be
  removed before anything is called uncertainty.
- **Enough data for genuine empirical distributions.** 24,837
  (origin, destination, hour) cells have ≥30 observations; 8,183 have ≥100.
  Within a fixed OD pair and fixed hour, duration still has a median
  coefficient of variation of 0.301 (IQR 0.233–0.392) — real, irreducible
  dispersion, which is exactly what a stochastic or robust model prices.
- **Weather matters, and time-of-day hides it.** The raw correlation between
  precipitation and speed is only −0.079; after removing hour-of-week means it
  roughly doubles to −0.195. Wet hours run **1.04 mph slower** than the same
  hour-of-week average (Welch t = −5.15, p = 6.9×10⁻⁶); snow hours 0.90 mph
  slower (t = −4.99, p = 1.7×10⁻⁵). Reporting the raw correlation would have
  understated the effect by half.
- **But weather shifts the mean without inflating the spread.** Residual speed
  dispersion is 1.191 mph in wet hours versus 1.303 dry — slightly *lower*.
  This is the modelling-relevant finding: weather behaves like **context that
  conditions the distribution**, not like an extra uncertainty dimension that
  widens an ambiguity set. It argues for a contextual/conditional formulation
  over simply adding a channel.
- **Wind is not a clean channel.** Residual correlation is *positive* (+0.191):
  windy hours are faster. Almost certainly confounded (clear cold nights, lower
  traffic volume during storms). Do not present wind as a congestion driver
  without identifying the mechanism.

## Licensing

- **TLC**: NYC open data, published by the Taxi & Limousine Commission for
  public use. Derived statistics and figures are publishable; check the current
  terms page before redistributing bulk raw extracts.
- **Open-Meteo**: free for non-commercial use, CC-BY-4.0 on the data, attribution
  required. Underlying reanalysis is ERA5 (Copernicus).

Confirm both before a manuscript's data-availability statement — COAP requires
one.

## Not verified / rejected

Candidates worth knowing about but not established here: PeMS (registration),
METR-LA / PEMS-BAY (mirror-dependent), UTD19 (terms), Uber Movement (retired —
it had HCMC and Paris travel times, which would have matched our instance set).
A separate survey of these was in progress; extend this table as they are
confirmed rather than assuming any of them work.
