# TRANSVRP — status

- **Working title:** open (placeholder `TRANSVRP`). Transformer / LLM methods
  for stochastic VRP (SVRPSPD) optimisation.
- **Venue:** not chosen.
- **State:** **ACTIVE — study phase (opened 2026-11-15).** No manuscript, no
  code, no experiments yet. `PROJECT.md` holds the reading of the seed paper,
  the literature map, and candidate directions with go/no-go tests.
- **Freeze policy:** open for edits.
- **Seed paper:** Liu et al., OR-Transformer, arXiv 2609.01933v2 (full text
  read). Local copy in `refs_local/`.
- **Separation from papers under review:** BATON (C&OR, r2) and TEMPO (TS) are
  frozen. This paper may *read* BATON's engine as a baseline (as TEMPO does) but
  must not edit `papers/baton/`, `papers/tempo/`, or `svrpspd_wdro/core/`/`ev/`
  behaviour. New code goes in a new subpackage `svrpspd_wdro/neural/` with
  tests, or inside this directory if the author decides on full separation
  (**open decision**, see PROJECT.md section 6).
- **Blocking:** (1) pick direction (PROJECT.md section 4); (2) separation decision;
  (3) find the primary source for weight-space programming before using it.
