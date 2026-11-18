# TRANSVRP — status

- **Working title:** open (placeholder `TRANSVRP`). Transformer / LLM methods
  for stochastic VRP (SVRPSPD) optimisation.
- **Venue:** not chosen.
- **State:** **ACTIVE — experiment A primary result in (2026-11-18).** Full sweep
  fitted at commit `c37ad7c` on a local CPU machine (32.1 h, no cuts), dev sanity
  passed, test stream opened once at `FROZEN COMMIT: c37ad7c…` (see `RESEARCH_LOG.md`).
  **Primary decision: WIN** — at 10^5 paths FULL lowers recourse cost by 0.82%
  [CI95 +0.65, +0.99] versus BATON at equal paths (n = 20 instances; 19/20 better).
  H2 better (gain is information), H3 equivalent, ARCH equivalent (say "neural
  regressor"); crossover N = 500, FULL worse at N = 100. Report:
  `code/results/expA/analysis_test/summary.md`. One analysis-only patch (curve grid)
  was pre-registered by hash before the test stream opened. Per-day `.npz` rows stay
  local (sha256 manifest committed). Boost arm M11 (test, frozen 9f2a29b): **equivalent** (FULL vs BOOST +0.25%,
  CI90 [+0.19, +0.32] inside ±0.70%), so the claim is about the state, not networks:
  trees on the same state suffice. Experiment A complete. **Next:** plan the
  manuscript with the author.
  `PROJECT.md` holds the reading of the seed paper, the literature map, and candidate
  directions.
- **Freeze policy:** open for edits.
- **Seed paper:** Liu et al., OR-Transformer, arXiv 2609.01933v2 (full text
  read). Local copy in `refs_local/`.
- **Separation from papers under review:** BATON (C&OR, r2) and TEMPO (TS) are
  frozen. This paper may *read* BATON's engine as a baseline (as TEMPO does) but
  must not edit `papers/baton/`, `papers/tempo/`, or `svrpspd_wdro/core/`/`ev/`
  behaviour. **Decided 2026-11-15 (author): self-contained.** The needed BATON
  files, instances and plans were copied into `code/` (provenance in
  `code/transvrp/baton/PROVENANCE.md`); nothing imports `svrpspd_wdro/`.
- **Blocking:** (1) pick direction (PROJECT.md section 4); (2) ~~separation decision~~ resolved: self-contained;
  (3) read the weight-space programming blog and logs in full (found; only summary-read so far).
