# City instances at 50 km radius — metro-scale rerun (2026-08-07)

Post-submission experiment (does NOT touch the frozen 6 km instances,
plans, or any paper table): `make_city_instances.py` gained a
`radius=` override; 50 km instances carry a `50K` name suffix and live
in `data/City50K/` with their own plan cache (`plans_City50K`), so the
BATON-era artifacts stay byte-identical.

Generation: `python scripts/make_city_instances.py cities=hcmc,hanoi
sizes=100,200,400 seeds=1 radius=50000` (Overpass needs the raised
600 s per-query timeout at this scale; HCMC's 50 km drive graph is
298,003 nodes / 693,006 edges vs ~20k nodes at 6 km).
Evaluation: `python scripts/run_realistic_eval.py dir=data/City50K
policies=Det workers=2 out=results_city50k_eval` (needs Python >= 3.10).

## HCMC, Det gate, realistic cost model — 6 km vs 50 km

Means across instances (6 km: 5 instances incl. extra seeds at n=100;
50 km: 3 instances, seed 1; saving % of the recourse term vs reactive):

| radius | travel km | v1 | fb | BATON-HO | BATON | restock | DP-50k | oracle |
|---|---|---|---|---|---|---|---|---|
| 6 km | 201 | 5.8 | 6.3 | 6.6 | 6.6 | 0.2 | 8.8 | 33.5 |
| **50 km** | **1097** | 8.5 | 9.4 | **10.5** | **10.6** | −0.2 | 12.0 | 39.4 |

Takeaways:

1. **Travel scales 5.4x** (201 -> 1097 km) at identical customer
   counts and capacities — vehicle counts barely move (capacity-bound),
   each route just gets ~5x longer.
2. **Every policy's recourse saving rises ~1.5x** at metro scale.
   Longer routes mean more mid-route demand exposure and more option
   value per handoff decision — consistent with the sensitivity
   finding in RESULTS_OTR2.md §4 ("v2's edge grows with route
   length").
3. **The policy ranking is unchanged**: BATON >= BATON-HO > peak+tau
   fallback > endpoint v1 > equal-data DP, with DP-50k above all
   fitted policies and the clairvoyant oracle far above everything.
   External validity of the 6 km conclusions extends to metro scale.
4. **Depot-restock stays worthless in the Vietnamese cities even at
   50 km** (-0.2%): the depot is central but routes now range tens of
   km away, so BATON's deployment selection correctly keeps the
   handoff-only action set (v2_act == v2_lsm on 2 of 3 instances).
5. BATON-HO captures **87% of the near-exact DP-50k saving** at 50 km
   vs 75% at 6 km — the backward-induction estimator gets relatively
   stronger exactly where the problem gets harder.

Caveat: 50 km discs around HCMC/Hanoi include peri-urban and rural
road network; a 100 km-diameter parcel operation with 150 kg vans is a
different operational regime (closer to regional distribution than
last-mile). The instances are kept as a stress test of the policies'
scale behaviour, not as a realistic single-depot last-mile scenario.

Hanoi 50 km instances: generation in progress (the Overpass download
is the bottleneck); rerun the two commands above to extend the table
when they land.
