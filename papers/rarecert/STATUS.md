# RARECERT — status

- **Working title:** Certifying rare overflow risk of regional pickup-and-
  delivery plans by stage-mixture importance sampling.
- **Origin:** spin-off of QUASAR (paper 7) direction A (2026-10-14): the
  classical rival built to test quantum amplitude estimation turned out to
  be the stronger, positive result.
- **Venue history:** INFORMS Journal on Computing (submitted October 2026,
  JOC-2026-10-OA-2667) — **REJECTED 2026-11-13** by the area editor without review
  (model too narrow for IJOC; the analytical importance-sampling part praised;
  editor suggested *Probability in the Engineering and Informational Sciences*).
  Decision and response plan: `reviews/ijoc_decision_2026-11-13.md`. The IJOC files
  (`submission_ijoc/`, `manuscript/`) are kept unchanged as the record of that
  submission.
- **Venue now:** *Methodology and Computing in Applied Probability* (MCAP,
  Springer; subscription/hybrid, no fee without the open-access option). The editor's
  suggestion, PEIS, was dropped the same day: it has been gold open access since 2024
  (article processing charge) and the author has no funds for that. Files in
  `submission_mcap/` (`rarecert_mcap.tex`, `cover_letter.tex`, figures, generated
  table inputs); Springer-style abstract (230 words), keywords, MSC codes and a
  Declarations section.
- **State:** **SUBMITTED to MCAP on 2026-11-13 — UNDER REVIEW, FROZEN.** Submitted
  version = commit `68f1c6b`, package `submission_mcap/`
  (`rarecert_mcap.pdf`, 19 pages, Springer Nature template; standard non-open-access
  model). Do not edit `submission_mcap/`, the scripts it cites
  (`run_rarecert_peis_*.py`, `rarecert/covering.py`) or `results/rarecert/peis/` until
  the decision arrives; revisions only on the editor's request. Known weaknesses at
  submission: efficiency theorem is for asymptotic (not computed) dominating points; with
  mass weights the covering condition holds for only 28% of balanced Vietnamese routes;
  figures carry Type 3 fonts.
- **Rewrite as a probability paper (2026-11-13; first aimed at PEIS):** rewritten as a probability paper with complete
  proofs (log-asymptotics with explicit error terms, sharp single-shift threshold
  ϱ = 1/4, new silent-underestimation proposition, mixture efficiency under a covering
  condition with constant Ψ computed by `svrpspd_wdro/rarecert/covering.py`).
  New computations: `svrpspd_wdro/scripts/run_rarecert_peis_conditions.py`
  (conditions on all 5 920 routes) and `run_rarecert_peis_bias.py` (end-mode
  shares vs measured bias); outputs in `svrpspd_wdro/results/rarecert/peis/`.
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
- [x] R5 MCAP submission (2026-11-13, commit `68f1c6b`)
- [~] R4 manuscript: draft 3 (manuscript/rarecert.tex, 25 pages), no pending sections; round-2 experiments in svrpspd_wdro/results/rarecert/r3_summary.md
