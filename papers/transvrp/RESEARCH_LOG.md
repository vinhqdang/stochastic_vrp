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

### 2026-11-16 — local machine for the full sweep; Windows portability fix (before any fit)
- Machine (author's laptop): Intel i9-13950HX (24 cores / 32 threads), 64 GB RAM,
  Windows 11 Pro, Anaconda Python 3.13.11, torch 2.6.0+cu124 **run on CPU**, numpy
  2.2.6, scipy 1.15.3, scikit-learn 1.8.0, pandas 2.3.3. These differ from the
  pinned pilot versions (`requirements-expA.txt`); every model of the sweep is fitted
  in this one environment, and no pilot model is reused.
- Platform: CPU, as the platform rule of spec 7.2 requires (projection 2.62 days ≤ 4).
  The author asked about the GPU; it was kept off to preserve the pre-registered
  platform and bit-identical reruns (P2). Workers: `--jobs 12` instead of 4 (more
  cores; worker count changes wall time only, not any computed value).
- The code did not run on Windows: `measure.py` imported the POSIX-only `resource`
  module, text files were read/written in the cp1252 default encoding (breaking the
  research-log gate read and `summary.md`), and concurrent workers creating the same
  chunk could hit `PermissionError` on `os.replace`. Fix (value-neutral: no seed,
  array, model or cost changes): Windows peak-working-set fallback for peak RSS,
  explicit UTF-8 on text I/O, `_publish_chunk` (drop tmp if another worker already
  published the deterministic chunk, else retry). Implemented with mutation-checked
  tests, independently reviewed (no blocker; residual rare lock risks covered by
  resumable fitting, run under a restart loop). 235 tests pass on this machine.

### 2026-11-16 14:56 GMT+7 — full sweep started at commit c37ad7c
- Fit stages `main` then `big`, `--jobs 12`, chunk cache outside the repository
  (C:/work/transvrp_chunks), eligibility `results/expA/eligibility_dev.csv` (frozen on
  dev in the pilot). Machine as in the previous entry. Restart-on-crash loop (fitting is
  resumable). No code change until the test evaluation is done.

### 2026-11-17/18 — fit complete (commit c37ad7c)
- `main`: 2,604/2,604 models, 2026-11-16 14:56 -> 2026-11-17 19:07 GMT+7 (28.2 h, 12
  workers, one try, no restart). `big`: 36/36, 19:07 -> 23:03 (3.9 h). Total 32.1 h,
  well under the 7-day ceiling; no cut of section 7.2 applied. Chunk cache 32+ GB.
- An unrelated job of the author briefly took ~24 GB RAM on 2026-11-17 ~10:15; the
  sweep was not affected (no failed job).

### 2026-11-18 07:14 GMT+7 — dev evaluation and sanity (dev only; nothing changed because of it)
- eval + merge (main, big) on dev at c37ad7c. Coverage 20 instances / 62 routes /
  3 reps; every row fitted at c37ad7c on a clean tree.
- 0 rows below the oracle for every policy. 7 cells above 1.005 x none, all at
  N = 100 (5 tuned_tau, 2 nn_full): small-data overfitting, plausible. All costs finite.
- **Analysis bug found (dev):** `analysis.curve()` built its grid from every N of
  nn_full; with the big stage combined, nn_full has N = 10^6 (descriptive 12-route
  subset, contrast S10) but baton_eq has no 10^6 rows, so `write_summary` raised
  "no rows for policy 'baton_eq' at N = 1000000". Spec section 8 item 5 defines the
  curve on the ladder grid only. Fix: grid = N where both nn_full and baton_eq have
  rows; any N other than 10^6 present for only one of them raises (guard). Tests
  added and mutation-checked; independent review: acceptable, guard added on its
  advice. Fitting, evaluation, contrasts, Holm and the primary are unchanged.
- **Procedure (author's decision, deviation from handoff "fix then refit"):** the
  models were fitted at c37ad7c and the fitting code is unchanged, so no refit. The
  patch is pre-registered here BEFORE the test stream is opened:
  `papers/transvrp/patches/analysis_curve_grid.patch`,
  sha256 8cef74f5fc11de6544b321326d822c313a42ab1ce280d431898d4fed4db3a183.
  It is stashed, test eval + merge run at the frozen commit on a clean tree (gate
  enforced), then the identical patch is restored and only the analysis step runs
  with it. The patch is committed afterwards; its hash can be checked against the
  commit diff of analysis.py and the test.
- Dev analysis with the patch: completes (output in results/expA/analysis_dev).
- Disclosure: the dev analysis CLI printed its dev branch (WIN, crossover N = 500 on
  dev). It is a dev sanity output only and changed nothing.

FROZEN COMMIT: c37ad7c4685714b0991b5c4b0f857ee00c0ddb55

### 2026-11-18 08:48 GMT+7 — test evaluation at the frozen commit; primary decision WIN
- Test eval + merge (main, big) at c37ad7c4685714b0991b5c4b0f857ee00c0ddb55 on a clean
  tree, gate enforced (`--frozen-commit`): 07:15 to 08:47 (1 h 32 min, 12 workers).
  main: 2604 models evaluated, 0 skipped, 6324 rows; big: 36 models, 72 rows.
- The pre-registered analysis patch was then restored from the stash; the restored
  diff of `papers/transvrp/code` hashes to 8cef74f5…a183, identical to the value
  logged above before the test stream was opened. Analysis ran with it at 08:48.
- **Primary (section 1.4): WIN.** FULL vs BATON-eq at N = 10^5: mean Δ̄ = +0.823%,
  CI95 [+0.65, +0.99], s = 0.365, exact one-sided Wilcoxon p = 1.9e-06, 19/20
  instances better (δ = 0.70%). All four WIN conditions hold.
- Attribution (1.5): H2 FULL vs SCALAR @ 10^5 better (+0.83%, 20/20; the gain is
  information); H3 SCALAR @ 10^5 vs DP @ 10^6 equivalent; ARCH FULL vs MLP @ 10^5
  equivalent by the δ margin (+0.31%, CI95 [+0.21, +0.40], 20/20 better), so the paper
  says "neural regressor" only; M11 (boost) pending, run next because of WIN.
- Secondary family (Holm S1–S11) as in `results/expA/analysis_test/summary.md`.
  Notable: S1 (N = 100) FULL worse than BATON-eq (−1.34%, Holm p = 0.043); S10, S11
  FULL @ 10^5 better than BATON and DP at 10^6 paths. Data-size curve crosses at
  N = 500 (same crossover as the dev sanity output).
- Nothing else unexpected. Test suite with the patch: 237 passed.
- Results storage (author's decision): CSVs, merged tables and analysis outputs are
  committed; the 5353 per-day `.npz` arrays (1.2 GB, variance item 2(b)) stay local
  and are git-ignored; their sha256 list is committed as
  `code/results/expA/npz_manifest.sha256`.

### 2026-11-18 09:10 GMT+7 — boost arm M11 (WIN branch) started at commit 9f2a29b
- Same machine, CPU, 12 workers, same chunk cache. `sweep fit --stage boost`
  (HistGradientBoostingRegressor on the FULL globals, 10^5, reps 0-2, eligible routes),
  then dev eval + merge for sanity, then a second `FROZEN COMMIT` line for this arm,
  then test eval + merge with `--frozen-commit`.
- M11 analysis procedure, fixed now before any boost result: the analysis CLI has no
  boost input; M11 is computed when the merged table holds policy `boost`. So
  `main.test.csv` and `boost.test.csv` are concatenated row-wise (no code change)
  into `main_boost.test.csv`, and the analysis is run on it into
  `results/expA/analysis_test_m11`. The primary and S1-S11 there must equal
  `analysis_test` exactly (checked); only the M11 line is new. M11 has its own
  alpha = 0.05, outside Holm (section 1.5).
- Boost fit 09:11 to 09:23 (186/186 models, 0 skipped); dev eval + merge 09:23 to
  09:25. Dev sanity: 20 instances / 62 routes / 3 reps, every row fitted at 9f2a29b,
  0 rows below the oracle, 0 above 1.005 x none, all finite. No dev contrast with
  nn_full was computed.

FROZEN COMMIT: 9f2a29bff90081251a60302d0af5e9f0258377fd

### 2026-11-18 09:31 GMT+7 — boost arm M11 test result: equivalent
- Test eval + merge (boost) at frozen commit 9f2a29b, gate enforced: 09:25 to 09:29,
  186 models, 186 rows. Concatenation and analysis as fixed above, into
  `results/expA/analysis_test_m11`. Its summary equals `analysis_test` line for line
  except the M11 line, and curve.csv is byte-identical (checked).
- **M11 (FULL vs BOOST @ 10^5, alpha 0.05, outside Holm): equivalent.** Δ̄ = +0.254%,
  median +0.189%, CI95 [+0.17, +0.34], CI90 [+0.19, +0.32], 20/20 instances FULL
  cheaper, two-sided p = 1.9e-06. The CI90 lies inside ±δ (0.70%), so by section 1.5
  the paper's claim is about the state (information), not about neural networks:
  "trees on the same state suffice". FULL is consistently but immaterially cheaper.
- Experiment A is complete. Next: plan the manuscript with the author.

### 2026-11-18 19:37 GMT+7 — experiment B adopted (spec revision 3)
- Reason: BATON is under review and cannot be cited; the paper needs comparisons with
  published 2023–2026 algorithms. In the paper A's `baton_eq` is presented as
  Longstaff–Schwartz isotonic regression on (k, W). Literature: `LIT_2025_2026.md`
  (DOIs checked in Crossref); methods of Zhu et al. (Networks 2026) and Kadyrov et al.
  (ORP 2025) read from the PDFs: `METHODS_ALP_PPO.md` (PDFs local, git-ignored).
- Spec `EXPERIMENT_B.md` revision 3 after two independent reviews (`REVIEW_B.md`,
  `REVIEW_B2.md`, both "accept with required changes", all items addressed).
  Arms: THR, ALP-Z, RLSM, DOS, DQN (GPU), PPO; fresh stream test_b; A's test streams are
  never read by B.
- The author asked to implement every arm and keep the "good" ones; told that arm
  selection after test_b is forbidden. Arms leave the family only by the dev gates of
  section 7, decided before the FAMILY line; every arm in the family is reported.
- The author signed off all design thresholds T1–T15 of section 0 as written.
- Platform: DQN on RTX 5000 Ada (torch 2.6.0+cu124, CUDA checked), other arms CPU,
  12 workers.

### 2026-11-18 21:34 GMT+7 — experiment B implementation progress (M12–M18)
- Source verification of DQN, RLSM, DOS and the PPO code done (`VERIFY_CITATIONS.md`,
  spec revision 3.1). DQN code is `moda707/vrpvcsd_of` (the `vrp-sdo` link in the draft
  was the 2026 paper's instances). Material differences from the sources are listed in
  spec 3.0/12: the DQN source trains 3 to 5 million trials with an MLP on hand-built
  features (ours at most 240k transitions per pass with the set encoder); the PPO code has
  no value net (the paper does; the spec follows the paper); DOS source trains 3,000+d
  steps per date (ours at most 400); RLSM source regularises by K, not ridge.
  Reporting rule: the paper states the budget ratio plainly.
- Implemented with tests and mutation checks (`code/MUTATION_B.md`): M12 gate and test_b
  stream (reviewed independently: `CODE_REVIEW_B_M12_M13.md`; the review's required
  changes were applied), M13 THR, M14 ALP-Z, M15 RLSM, M16 DOS, M17 padded set encoder,
  M18 DQN. The edit to A's gate fixtures only changes the log line form to the anchored
  `FROZEN COMMIT: <hash>`; A's gate accepts a subset of what it accepted before. Full
  suite: 396 passed.
- Measured on the RTX 5000 Ada: one DQN update (batch 256, m = 17) takes 22.8 ms against
  the 5 ms prior of spec 9.1. The pilot (spec 9.3) replaces all priors; if the
  projection exceeds the trigger the ordered cuts apply, and launch-overhead
  optimisation of the DQN update (no change of the algorithm, equality tested) is
  allowed before the freeze.
- Not yet implemented: M19 PPO, M20 codecs/registry, M21 sweep integration (records
  `libraries` in payloads, family/model/library columns), M22 fidelity, M23 analysis_b.
