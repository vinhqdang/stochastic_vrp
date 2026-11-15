# transvrp code (self-contained)

Nothing here imports `svrpspd_wdro/`. Run from this directory:

```bash
pip install numpy scipy scikit-learn pandas pytest
python -m pytest tests -q            # parity with the original engine
python -m transvrp.benchmark CON3-0  # none / BATON / oracle on one instance
```

| path | content |
|---|---|
| `transvrp/baton/` | verbatim copies of BATON `otr2.py`, `costs.py` (see `PROVENANCE.md`; do not edit) |
| `transvrp/data.py` | Dethloff parser, correlated-gamma scenario sampler, plan loader, per-route problem builder |
| `transvrp/benchmark.py` | baseline ladder for the two-action {continue, hand off} problem |
| `data/Dethloff/` | the 40 Dethloff instances (copied) |
| `plans/` | cached 'Det'-gate plans for CON3-* and SCA3-* (copied), so no ALNS is run |
| `tests/test_parity.py` | golden numbers from the original engine |

Neural policies (direction A) go in new modules under `transvrp/`.

## Experiment A (`transvrp/expa/`, spec: `../EXPERIMENT_A.md`)

| module | spec | content |
|---|---|---|
| `paths.py` | M1 | chunked seeded path pools (train/dev/test streams), RouteSpec |
| `features.py` | M2 | online-observable features: FULL, SCALAR, MLP summaries |
| `backward.py` | M3 | backward-in-k driver with 2-fold cross-fitting |
| `model.py`, `train.py` | M4, M5 | set encoder (~7.7e4 params) and MLP, per-stage training |
| `simulate.py` | M6 | policy simulator (parity with BATON's simulator) |
| `baselines.py` | M7 | none / tuned tau / BATON / plug-in DP / OTR-2.1 / oracle |
| `eligibility.py` | M8 | dev-stream route eligibility and strata |
| `sweep.py` | M9 | fit/eval runner, smoke config, frozen-commit gate for the test stream |
| `analysis.py` | M10 | instance-level decision rules, Holm, TOST |
| `boost.py` | M11 | gradient-boosting arm (WIN branch only) |

```bash
cd papers/transvrp/code
pip install -r requirements-expA.txt        # torch needs the CPU index URL
python -m pytest tests -q                    # fast tests
python -m pytest tests -q -m "slow or not slow"   # all 89 tests, ~4 min
```
