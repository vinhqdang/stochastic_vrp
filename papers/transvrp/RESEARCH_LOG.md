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

### 2026-11-18 22:18 GMT+7 — experiment B: M19 PPO, M20 codecs/registry, DQN update speed-up
- M19 PPO and M20 codecs/registry implemented with tests and mutation checks
  (`code/MUTATION_B.md`). The DQN update (M18) was optimised before any fit without
  changing the algorithm (host syncs removed, packed copies, fused Adam, CUDA graph):
  17.9 ms to 2.15 ms per update; CPU path bitwise equal to the old path, GPU parameter
  difference about 1e-6 relative (fused Adam), tested with a kept reference path.
- Full suite: 461 passed in the first run; one test-isolation failure (a `reload` in
  test_expb_padded left a second class object for later isinstance checks) fixed in the
  test, not in the code.
- Open for M21: top-level `payload["libraries"]` for the gate; `dos.init_net` ignores
  `cfg.width` (defect, fix with a test); per-pass checkpoint selection for PPO (spec 6.3).

### 2026-11-18 23:06 GMT+7 — experiment B: M21 sweep integration
- B stages, per-pass curves for DQN/PPO, test_b row columns (family_sha256, model_sha256,
  library versions), top-level payload `libraries`, `manifest` CLI, DOS width fix.
  `b_smoke` fit + eval + merge on dev yields exactly the ten policies. A's stage job
  lists are unchanged (snapshot test). Full suite: 492 passed. test_b was never read.
- Note for cut (i) (reuse of A's FULL/ladder models): A's payloads do not record
  `libraries`, so gate item 8 refuses reuse stages; cut (i) cannot apply unless a
  provenance note for A's models is added first. With the faster DQN update it may not be
  needed; the pilot decides.

### 2026-11-18 23:58 GMT+7 — experiment B: M22 fidelity, M23 analysis_b, DOS diag; spec 3.2
- M22 `expb/fidelity.py` (dev gates, FAMILY derivation, `fidelity_dev.md` machine-parsed
  back by the gate) and M23 `expb/analysis_b.py` (refuses mismatched family hash,
  missing or extra arms, non-test_b rows, coverage != 20) implemented with tests and
  mutation checks. DOS now records the never-stop validation objective (and objectives
  for constant and carry stages), which gate B-DOS reads; without it DOS would have been
  dropped for a defect of ours. Fitted models are bitwise unchanged (digest test).
- Spec revision 3.2 (before any pilot): "G2 route-reps" defined as rep 0 of every
  eligible route (62 route-reps). No threshold of section 0 changed.
- Full suite: 582 passed. All of M12–M23 are implemented; no fit, pilot or dev run of B
  has been made, and test_b has never been read.
- Next: independent review of the whole B code, then the pilot (spec 9.3).

### 2026-11-19 05:45 GMT+7 — experiment B: whole-code review and fixes; spec revision 3.4
- Independent whole-code review (`CODE_REVIEW_B.md`, accept with required changes): 2 critical,
  8 major. No path to test_b without the gate, no future-information leak, no hidden
  handicap on THR, ALP-Z, RLSM, DQN or PPO were found. All findings fixed (mutation-checked):
  results directory of B stages (expB), manifest CLI, one GPU process, per-stage cut switches
  and `dos:cut` spelling, gate-side analysis cleanliness check, committed tuning record and
  the section 6.1 selection rule, `derive_family` for cut arms, per-route B-THR statistic,
  shared PPO predictor (evaluation about 11x faster), additional analysis reports.
- **Decisions by the author (2026-11-19), made before any pilot or fit; they only decide
  which arm is dropped for a defect of ours and never compare with FULL:**
  (1) T4: 3 SE becomes 4 SE (a correct THR was falsely dropped about 15% of the time at 3 SE,
  about 0.4% at 4 SE). (2) B-DOS: absolute tolerance 1e-6 replaced by relative
  0.02 x |always-stop - never-stop|, 400 steps per stage (the absolute tolerance is unreachable
  with a sigmoid output: best 94.9% of stages against the 95% rule). Evidence:
  `DOS_GATE_INVESTIGATION.md` (dev data only).
- Decision by the lead: cut (i) (reuse of A's models) struck; it cannot pass gate item 8 and
  is not needed with the faster DQN update. Cuts renumbered (i) drop DOS, (ii) 6 tuning
  routes, (iii) reps 3 to 2, (iv) P = 1.
- Spec is now revision 3.4 (changelog inside). Full suite: 682 passed. No pilot, tuning, fit
  or evaluation of B has been run; test_b has never been read.

### 2026-11-19 08:55 GMT+7 — experiment B pilot (spec 9.3), dev only, HEAD dda41a8
- Run: stage `b_pilot` (CON3-0, 4 eligible routes; every arm at N = 10^4, reps 0 and 1; timing
  cells at N = 10^5, route 3), fit 05:55–08:05, dev eval to 08:06. 70 models, 86 dev rows. No
  code or default changed; test_b not read.
- **Q1** done (682 passed). **Q2** pass: one cell per arm (thr, alpz, rlsm, dos, ppo, nn_full;
  route 0, N = 10^4, rep 0) refit in a separate directory gives bit-identical dev costs.
  The DQN rerun difference was not measured (not required to be bitwise).
- **Q3** pass: refit of FULL and ladder on 6 routes (CON3-0 routes 0, 1, 3; SCA3-0 routes
  0, 1, 3) at N = 10^5, rep 0, equals A's `main.csv` dev costs exactly (difference 0.0 on
  all 24 rows).
- **Q4** no violations: C_oracle <= C <= 1.005 C_none on every pilot cell. Observation
  (diagnosis, not a gate): PPO at N = 10^4 equals the never-handoff cost on 2 of 8 cells
  (no learning in one pass); tuning (spec 6) decides, nothing is changed here.
- **Q5** measured, route 3 (m = 17), N = 10^5, one pass, contention from other jobs included:
  DQN 53 s fit (76 s wall alone including start-up); PPO 2672 s (0.74 h, prior 1.15 h);
  DOS 134 s; THR 126 s; RLSM 3 s; ALP-Z 0.3 s. GPU concurrency: two simultaneous DQN
  processes took 198 s wall against 76 s for one, i.e. 0.77x throughput, below the 1.5x
  needed, so one GPU process is kept (as in 9.1).
- Projection with measured times, worst-case route times applied to all 186 jobs, P = 3 for
  DQN and PPO, sequential sum: FULL refit 10.0 h (A's measurement), ladder 0.2 h, THR + ALP-Z
  + RLSM + DOS 1.1 h, DQN main 9.4 h, PPO main 37.6 h (11 workers), tuning about 13.7 h (PPO
  8.9 h scaled by 0.74/1.15, DQN 4.2 h, RLSM and DOS 0.6 h), independence world 0.8 h,
  evaluation 4 h: **about 77 h <= 134.4 h. The trigger does not apply; no cut is made**
  (`ACTIVE_CUTS` unchanged). The DQN and tuning rows are estimates built from one timing cell
  and are checked again during tuning.
- **Q6** coverage: all six arms plus FULL and the ladder produced rows.

### 2026-11-19 17:50 GMT+7 — experiment B: tuning run times, cuts (i)–(iii) applied (compute only)
- Tuning on dev, 12 subset routes, rep 0, no cut yet: THR, ALP-Z, RLSM c0–c5, DOS c0–c1 and
  DQN c0–c3 finished (13:32 for DQN); PPO c0 finished 17:40 after 7.4 h wall. Measured on
  the 12 jobs (12 concurrent CPU workers): PPO 3-pass job 6.59 h mean (5.36–7.34 h), i.e.
  about 2.2 h per pass, three times the pilot's 0.74 h (a single timing cell with little
  contention); DQN 3-pass job 270 s mean. The PPO c1 run (started 17:40) was stopped at
  17:41 and discarded. **No PPO or DQN tuning score had been read when the following was
  decided; the decision uses run times only.**
- Projection with these measurements (sequential sum, P = 3, 3 replicates, 12-route tuning,
  no cuts): tuning done 12.1 h + PPO c1–c3 22.2 h; main: FULL 10.0, ladder 0.2, cheap arms
  1.1, DQN 186 x 270 s = 14.0, PPO 186 x 6.59 h / 12 = 102.1; independence world 0.8;
  evaluation 4: **about 166 h > 134.4 h**. Trigger (spec 9) applies; cuts in the order of 9.2:
  (i) drop DOS (-0.6 h of main; its tuning is sunk and stays on disk, not used): 165.9 h;
  (ii) tune on 6 routes for the unfinished PPO configs c1–c3 (6 jobs each, run concurrently,
  about 9.9 h): 153.6 h; (iii) replicates 3 to 2 for every arm including FULL and the
  ladder: main 126 h -> 85 h: **about 112 h <= 134.4 h**. Cut (iv) (P = 1) is not needed
  and is not applied. The scoring routes of every already tuned arm (THR, ALP-Z, RLSM, DQN,
  PPO c0) are restricted to the same 6 routes by `choose.py` (the rows already contain them;
  nothing is re-run).
- Action: `stages.ACTIVE_CUTS = Cuts(drop_dos=True, tune_routes="subset6", reps=(0, 1))`;
  FAMILY line will carry `dropped=[dos:cut]`; two tests that pinned the default are updated.
  test_b has never been read.

### 2026-11-20 03:30 GMT+7 — experiment B: tuning finished, configurations chosen by the 6.1 rule
- Tuning on dev only (rep 0, N = 10^5, the 6 routes of cut (ii) for scoring), all runs under
  the cut set logged on 2026-11-19 17:50, code `11ad4d8` for PPO c1–c3 (PPO c0, DQN, RLSM,
  THR, ALP-Z and DOS were fitted on the 12 subset routes before the cut; their scores are
  taken on the 6 routes). Wall: PPO c1–c3 ran concurrently 18:24 to 03:13 (about 8.8 h).
  DOS tuning (c0, c1) is sunk by cut (i) and is not used.
- Selection by `expb/choose.py` (rule 6.1, tie window 0.10 points, fewer passes, lower
  seconds, lower index), record `results/expB/tuning_record.json` (every score, fit seconds,
  hashes of the tuning CSVs). Scores are mean 100 (C_cfg − C_baton_eq) / C_baton_eq; negative
  means cheaper than BATON-eq on the tuning routes:
  - DQN: config c1 (lr 1e-4, target period 32,000 transitions), **P = 1** (c1@p1 −0.857;
    c3@p3 −0.838 is within 0.10, so the smaller pass count wins).
  - PPO: config c1 (lr 1e-4, gamma 1.0), **P = 3** (c1@p3 +0.245; c1@p2 +0.381, the drop
    from pass 2 to 3 is 0.14 < 0.5, so PPO is not labelled budget-limited by the 6.3 rule).
    PPO is dearer than BATON-eq on the tuning routes for every config (+0.24 to +3.7).
  - RLSM: config c2 (K = 100, ridge 1e-6) (−0.522; c4 and c5 are within 0.10 and slower).
  - THR and ALP-Z have no grid. DOS is cut.
- The DQN selection is noisy: scores of one config differ by up to 0.28 points between
  passes. This is reported as such; nothing is re-run.
- test_b has never been read. Next: the dev gates and the main fit (spec 5.3 and 7).
