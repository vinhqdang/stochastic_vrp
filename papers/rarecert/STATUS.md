# RARECERT — status

- **Working title:** Certifying rare overflow risk of regional pickup-and-
  delivery plans by stage-mixture importance sampling.
- **Origin:** spin-off of QUASAR (paper 7) direction A (2026-10-14): the
  classical rival built to test quantum amplitude estimation turned out to
  be the stronger, positive result.
- **Venue candidates:** INFORMS Journal on Computing; Transportation
  Science; EJOR. Decide after the novelty check (PROJECT.md §6).
- **State:** ACTIVE — draft 3 of the manuscript complete (2026-10-16): both internal reviews addressed (plan-level pass rate at the planned capacity, missed-mode diagnostic on all rows, pickup rescaling and correlated factors, budget-fair rivals, idle-machine timing, hub and route-day checks against crude Monte Carlo, coverage at rare levels, longer routes); proofs remain sketches. Next: third internal review, then full proofs or a decision to submit with sketches labelled. Open for edits.
- **Sharing:** uses the shared engine (instances, scenario generator, plan
  caches incl. the region set `data/Region`, `results/plans_Region`) and
  `svrpspd_wdro/quantum/rare.py` (classical parts) — to be moved to a
  classical subpackage before submission. Overlap with BATON (planning
  gates, online recourse) and TEMPO (monitoring) is none at the level of
  claims: this paper evaluates a given plan's tail risk.

## Milestones
- [x] R0 novelty check (IS for unions of rare events; rare risk in routing)
- [~] R1 theory: draft revised after review (THEORY.md); SMIS measured on 120 routes; full proofs pending
- [x] R2 richer simulator where the exact DP does not apply (route day: lognormal travel with a shared factor, load-dependent service, time windows; MDIS on 60 routes: error vs 10⁷-draw crude MC 0.76% median / 1.77% q90 at γ = 1e-3, MDIS–CE agreement 0.2–0.24%, RV 4.3–6.1, single shift off by up to 98%, CE costs 1 750–1 830× more samples at ε_rel = 0.1 and 18–19× at 0.01)
- [x] R3 experiments: 66 calibrated hubs at 1e-3 and 1e-5 (buffers 35–43% / 64–76% median by region), exact conditional program and Glasserman–Li / conditional MC / subset simulation on 30 routes, exact hub program on 6 hubs, sensitivity (rho, cv, pickup scaling, lognormal) and CI coverage on 30 routes
- [~] R4 manuscript: draft 3 (manuscript/rarecert.tex, 25 pages), no pending sections; round-2 experiments in svrpspd_wdro/results/rarecert/r3_summary.md
