# TRANSVRP — research log (experiment A)

Every tuning decision, deviation from `EXPERIMENT_A.md`, and every run that
touches data is logged here with date (GMT+7) and reason, **before** results are
looked at where possible. Short commit hashes only in ordinary entries.

**Test-stream gate.** `transvrp/expa/paths.py::verify_test_gate` opens the test
stream only for a full 40-character commit hash that appears in this file and
equals the checked-out commit on a clean tree. No such hash is recorded yet: the
test stream is closed. The freeze entry, when made, will be the only place a full
hash appears, under a heading "FROZEN COMMIT".

## Log

### 2026-11-15 — specification and review
- `EXPERIMENT_A.md` revision 2 written after review `REVIEW_A.md` (12 findings,
  10 accepted, 2 partly). Margin delta = 0.7%, unit of analysis = instance (n = 20).
- Disclosure (spec section 12.1): while re-measuring BATON refit noise, one
  BATON-only pass ran on the 20,000 test days. No neural model existed; no number
  from that pass is used. The paper must state this.
- Erratum (spec section 12.2): acceptance test M3(d) cannot hold for alive rows;
  the leakage guarantee is fit/apply disjointness (M3(b)).

### 2026-11-15 to 2026-11-16 — implementation and code review
- Modules M0-M11 implemented (commits 7ff6af4, 269b8b6, 368e61e).
- Code review `CODE_REVIEW_A.md`: 5 majors (test gaps for cross-fit, SCALAR
  columns, 80% statistics; gate only at CLI; analysis swallowed missing
  contrasts). Fixed in 1e73246 and e0979b4 (175 tests).
- Re-review `CODE_REVIEW_A2.md` (9ba6ae9): no blocker for the dev pilot; P5
  projection never computable, gate gaps under adversarial use, minors. Fix batch C
  follows before the pilot.

### Environment for the pilot
- 4 CPU cores, 15 GB RAM, about 27 GB free disk (fixed session allowance),
  Python 3.13, torch 2.15 CPU. The chunk cache goes outside the repository.

### 2026-11-16 — pilot started on the DEV stream (commit e5fb8da)
- Fix batch C committed (207 tests green). Pilot = spec section 7.3, stage `pilot`,
  CON3-0 (4 routes), dev stream only; test stream stays closed.
- Order: eligibility table on the 20 instances (dev) -> fit (4 workers) -> eval dev ->
  merge -> 10^6 timing pass (FULL, 10^5 nets) -> pilot report with P1-P6.
- Chunk cache outside the repository (session scratch); results under
  `code/results/expA/` (chunks and models git-ignored).
- P2 (rerun reproducibility) is not run in this first pass; it will be reported as
  missing rather than assumed.
- The P5 extrapolation rule was fixed in code before any pilot timing existed
  (plan_cells docstring); MLP cost is a prior (0.2 x FULL), not measured.

### 2026-11-16 — pilot outcome (dev stream only)
- Run 12:05-13:06 GMT+7: eligibility (80 routes, 62 eligible), fit 73 jobs in 39 min
  on 4 workers, eval + merge 1 min, 10^6 timing pass 20.5 min (stop updates 19.5 min,
  peak RSS 1.8 GB).
- Two bugs found by the pilot and fixed before the report was accepted (7d9b3b7):
  (1) the report step could not load models pickled under `python -m` (class recorded
  as `__mp_main__.Job`); fixed by `load_payload` plus a root-cause change, existing
  pickles reused unchanged; (2) the P5 projection added wall MINUTES to the 3 h term
  and labelled the sum hours (printed 149.6 days). Corrected: 14,353 job-minutes /
  60 / 4 + 3 h = **62.8 h = 2.62 days -> CPU** (spec platform rule: <= 4 days).
- P2 rerun: cell full / CON3-0 / route 0 / N = 10^4 / rep 0 refitted at 7d9b3b7 (main
  pilot models were fitted at e5fb8da; the changes between them touch loading,
  reporting and job filtering, not fitting) — max |diff| = 0 on mean_cost, handoff_rate,
  fail_rate. Rerun table: `code/results/expA/pilot/p2_rerun_*.csv`.
- Pilot report `code/results/expA/pilot_report.md`: P1-P6 all PASS. P3: SCALAR /
  BATON-eq = 1.00017 (tolerance 2%). P6: warm-start chosen on 7.6% of stages, fold
  disagreement 0.47%. Monotonicity: 0% of SCALAR curves decrease. No lr/S_k iteration
  was needed, so nothing in the design was tuned on the pilot.
- The P5 projection rests on 2 measured FULL cells and a linear a + bN extrapolation
  to 10^6 (one route); treat it as order-of-magnitude.
- Pilot cost numbers are not read as evidence for any hypothesis (one instance; design
  frozen).
