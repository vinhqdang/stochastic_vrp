# SVRPSPD pipeline — planning gates + execution policies

Stochastic Vehicle Routing with Simultaneous Pickup and Delivery:
ALNS planning under six feasibility gates, and online execution policies
(headlined by **OTR-2.0**, a peak-aware optimal-stopping handoff rule)
evaluated under a realistic three-class fleet cost model.

## Module map

```
core/
  otr.py                OTR 1.0 (endpoint-label isotonic curves + tau threshold)
  otr2.py               OTR-2.0: peak labels, Longstaff-Schwartz continuation
                        models, cost-comparison trigger, oracle, validation
  otr21.py              OTR-2.1 (experimental): feature-enriched continuation
                        models (factor posterior, recency, spike detector)
  costs.py              Realistic last-mile cost model: planned / standby /
                        emergency vehicle classes, per-stop price schedules,
                        generalized LSM + simulators, clairvoyant oracle
  dp_exec.py            Plug-in dynamic program (exact-method benchmark for
                        the execution stage; quantile-binned backward induction)
  published_policies.py Salavati-Khoshghalb et al. (2019) rule-based recourse
                        (pi1/pi2/pi3), adapted to handoff recourse
  alns_wdro.py, wdro_*.py, filter.py, cache.py, ...   W-DRO planning internals

scripts/
  dethloff_runner.py    Instance parsing, scenario generation, ALNS + the six
                        gates: Det, SAA, WDRO, GNRS (Gounaris 2013 inflation),
                        BSIM (Bertsimas-Sim 2004 budget), MDRO (Ghosal et al.
                        2024 moment ambiguity); solve_fast for n>90
  run_realistic_eval.py MAIN EVALUATION: gates x 11 execution policies
                        (reactive, v1, myopic, fallback, pi1-3, OTR-2.0,
                        DP equal-data, DP-50k, oracle) under core/costs.py
  run_otr2_eval.py      Flat two-price evaluation + synthetic spec benchmark
  run_otr2_sensitivity.py  5-factor sensitivity sweeps with Wilcoxon tests
  run_mip_cert.py       Exact-MIP planning certification (HiGHS or Gurobi WLS)
  make_city_instances.py   Real-map instances from OSM drive networks
                        (HCMC, Hanoi, NYC, Paris, Shanghai; road distances)
  make_salhi_nagy.py    Salhi-Nagy 1999 CMT-X/Y benchmark from CVRPLIB
  make_figures.py       Paper/report figures (city maps, decision explainer)
  animate_execution.py  Animated trip replay: assignment + demand scenario ->
                        GIF (handoff dispatches standby vehicle, breach calls
                        emergency vehicle); compare mode = side-by-side
  validate_otr21.py     Paired v2 vs v2.1 validation over real routes

data/    Dethloff (40x50 cust), SalhiNagy (14x50-199), City (9x100-400)
tests/   pytest suite (~180 tests)
results/ CSVs consumed by ../paper/make_tables.py; figures/
```

## Reproducing the headline tables

```bash
# grand comparison grid (Dethloff, 6 gates x 11 policies)
python scripts/run_realistic_eval.py policies=Det,SAA,WDRO,GNRS,BSIM,MDRO \
       workers=3 out=results_grand_dethloff

# large-scale benchmarks
python scripts/run_realistic_eval.py dir=data/SalhiNagy policies=Det out=results_salhinagy_eval
python scripts/run_realistic_eval.py dir=data/City      policies=Det out=results_city_eval

# synthetic spec benchmark + sensitivity
python scripts/run_otr2_eval.py synthetic
python scripts/run_otr2_sensitivity.py

# exact-solver certification (HiGHS by default; solver=gurobi with a WLS licence)
python scripts/run_mip_cert.py dethloff tlim=300
python scripts/run_mip_cert.py small

# visuals
python scripts/make_figures.py all
python scripts/animate_execution.py policy=compare scenario=0

# tests
python -m pytest tests/ -q
```

Solved plans are cached per instance in `results/plans/*.json` (all gates
merged), so evaluation reruns skip ALNS. Results roll up in
`../RESULTS_OTR2.md`; the manuscript tables regenerate from the CSVs via
`../paper/make_tables.py`.
