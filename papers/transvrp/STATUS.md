# TRANSVRP — status

- **Working title:** open (placeholder `TRANSVRP`). Transformer / LLM methods
  for stochastic VRP (SVRPSPD) optimisation.
- **Venue:** not chosen.
- **State:** **ACTIVE — study phase (opened 2026-11-15).** No manuscript, no
  experiment results yet. `EXPERIMENT_A.md` (revision 2, reviewed in `REVIEW_A.md`) is
  implemented as modules M0-M11 in `code/transvrp/expa/` (175 tests pass after code review `CODE_REVIEW_A.md` and two fix batches); **nothing has
  been run beyond smoke/unit tests, the test stream is untouched by any neural code,
  and the pilot has not been run.** Remaining: memory at 10^6 is reduced but not measured; stop rules 1 and 4 are
  process checks, not code; `RESEARCH_LOG.md` does not exist, so the test-stream
  gate (now in the data layer) refuses all test evaluation until a frozen commit hash
  is recorded there. Pass `--chunk-dir` or rely on the .gitignore rule for the chunk
  cache. Next: re-review of the fixes, then the pilot on the dev stream. `PROJECT.md` holds the reading of the seed paper,
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
