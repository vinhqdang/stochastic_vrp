# RARECERT — status

- **Working title:** Certifying rare overflow risk of regional pickup-and-
  delivery plans by stage-mixture importance sampling.
- **Origin:** spin-off of QUASAR (paper 7) direction A (2026-10-14): the
  classical rival built to test quantum amplitude estimation turned out to
  be the stronger, positive result.
- **Venue:** SUBMITTED to INFORMS Journal on Computing (October 2026), area
  Simulation, Stochastic Models, & Stochastic Optimization; single-blind;
  submission files in `submission_ijoc/` (INFORMS template, cover letter,
  `rarecert_ijoc.pdf`, 28 pages = ~25 main incl. references + appendix as
  supplement). Decision pending.
- **State:** **UNDER REVIEW at INFORMS J. Computing — FROZEN.** Do not edit `manuscript/`, `submission_ijoc/` or the results it cites until the decision arrives (revisions only on the editor's request). Draft 3 content: both internal reviews addressed; proofs remain sketches (stated as such). Known weaknesses recorded at submission: proofs are sketches, 30-route GD sensitivity, 8 long routes, no independent hub check at 1e-5.
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
