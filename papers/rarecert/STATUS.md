# RARECERT — status

- **Working title:** Certifying rare overflow risk of regional pickup-and-
  delivery plans by stage-mixture importance sampling.
- **Origin:** spin-off of QUASAR (paper 7) direction A (2026-10-14): the
  classical rival built to test quantum amplitude estimation turned out to
  be the stronger, positive result.
- **Venue candidates:** INFORMS Journal on Computing; Transportation
  Science; EJOR. Decide after the novelty check (PROJECT.md §6).
- **State:** ACTIVE — planning. Open for edits.
- **Sharing:** uses the shared engine (instances, scenario generator, plan
  caches incl. the region set `data/Region`, `results/plans_Region`) and
  `svrpspd_wdro/quantum/rare.py` (classical parts) — to be moved to a
  classical subpackage before submission. Overlap with BATON (planning
  gates, online recourse) and TEMPO (monitoring) is none at the level of
  claims: this paper evaluates a given plan's tail risk.

## Milestones
- [ ] R0 novelty check (IS for unions of rare events; rare risk in routing)
- [ ] R1 theory: structure of the overflow event, efficiency of the
      stage-mixture estimator, certification procedure
- [ ] R2 richer simulator where the exact DP does not apply
- [ ] R3 experiments on the region set (148 instances) and benchmarks
- [ ] R4 manuscript
