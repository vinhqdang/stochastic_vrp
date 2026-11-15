# TRANSVRP — status

- **Working title:** open (placeholder `TRANSVRP`). Transformer / LLM methods
  for stochastic VRP (SVRPSPD) optimisation.
- **Venue:** not chosen.
- **State:** **ACTIVE — study phase (opened 2026-11-15).** No manuscript, no
  experiments yet; baseline code copied and parity-tested. `PROJECT.md` holds the reading of the seed paper,
  the literature map, and candidate directions with go/no-go tests.
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
