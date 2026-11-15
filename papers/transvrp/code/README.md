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
