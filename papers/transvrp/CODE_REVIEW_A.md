# Code review of EXPERIMENT A implementation (expa/ modules and tests)

Scope: `code/transvrp/expa/*` (paths, features, backward, model, train, simulate, baselines, eligibility, sweep, analysis, boost) and `code/tests/test_expa_*.py`, against `EXPERIMENT_A.md` sections 1-9 and 12. Reference code (`baton/`, `data.py`) treated as correct. Review date 2026-11-15. No code, test or git state was changed; scratch copies for mutation checks lived in a temporary directory outside the repository. The test stream was never evaluated.

## Verdict: READY WITH FIXES

The estimator itself is correct in everything I could check. Step indexing is consistent with BATON (`cum[:, k-1]`, `H[k-1]`, `E[tau-1]`). The simulator charges costs exactly as `_simulate_costs_general` does (emergency `E[k_idx]`, billing `H_bill`). Features read only columns 1..k. Training streams are deterministic and disjoint. The CLI also works end to end with `python -m ... --jobs 2` (smoke stage, fit, eval and merge, 31 s).

The real weaknesses are in the safety net. Three plausible bugs in the leakage-critical parts pass every test (confirmed by mutation). The test-stream gate is a convenience check and not a barrier. The analysis silently shrinks the Holm family when data are missing. None of these changes a result today. They are exactly the defects that would change a conclusion without anyone noticing during a multi-day run. Fix findings 1-5 before the pilot; the rest can follow.

Test suite: `python3 -m pytest tests -q` gives **89 passed in 222 s**. This is the whole suite. `pytest.ini` registers the `slow` marker but nothing deselects it, so the 3 slow tests run by default.

## Findings, ranked

### 1. MAJOR (confirmed by mutation): no test detects a broken cross-fit
`backward.py:159-166` (stop update) and `tests/test_expa_backward.py::test_leakage_log_is_disjoint_and_validation_is_unseen`.

The spec's central guarantee is that no row's stopping decision comes from a predictor fitted on that row. The test checks only the *log* (`fit_rows`, `applied_rows`), and `ls_backward` writes that log itself with fixed labels (`_log_entry(log, k, "A", Aa, Ba, dis)`). The log is therefore a statement about intent, not about what `predict` was called on. Nothing else pins the behaviour: `iso_fit_fn` with folds is only exercised by the dead-row and characterisation tests.

Mutation (scratch copy A): replace the stop update by
```
future[Aa[netA.predict(cache,k,Aa) > h]] = h      # A rows decided by net A itself
future[Ba[netB.predict(cache,k,Ba) > h]] = h
```
Result: `tests/test_expa_backward.py` and `tests/test_expa_train.py` give **13 passed**. The cross-fit is silently gone and the log still says disjoint.

Fix: add a test with a spy predictor. Let `fit_fn` return an object that records the row indices passed to each `predict` call, and the rows it was fitted on. Assert that for each k the spy of fold A is never called on A's fit rows during the stop update (only on B rows and validation rows). Also assert that the final `future` values on A rows equal what `net_B` implies. Additionally, build `log` from the actual `predict` call arguments (record inside a thin wrapper) instead of from the variables.

### 2. MAJOR (confirmed by mutation): SCALAR arm could leak the running peak and no test would notice
`features.py:25` (`HISTORY_IDX`) and `tests/test_expa_features.py::test_scalar_ignores_history_full_does_not`.

H2 and S5/S6/S8 rest on "SCALAR sees only (k, W_k)". g2 (peak) is a history feature. The test (a) uses two toy rows with equal W_1 *and equal peak* (they differ only in a p/d split that cancels), so it cannot see a peak leak; (b) asserts `gs[:, list(HISTORY_IDX)] == 0` using the very constant under test.

Mutation (scratch copy B): `HISTORY_IDX = (7, 8, 9, 10, 11)` (peak no longer zeroed). Result: `test_expa_features.py` gives **7 passed**. Under this mutation SCALAR has strictly more information than BATON and H2 would be biased towards "no information".

Fix: use a hard-coded literal list of the six zeroed columns in the test (not the module constant). Add a case at stage k >= 2 with equal W_k but different running max (for example W = (3, 1) against (1, 1), up to the clip), and assert scalar outputs are bit-identical while FULL differs. A stronger property test: SCALAR output is a function of (k, W_k) only, so sample random rows, group by rounded W_k, and compare.

### 3. MAJOR (confirmed by mutation): fitting statistics may include validation rows without any test noticing
`train.py:287-288`, `boost.py:81-82`; the only test is `test_route_stats_use_only_rows_passed`, which tests `RouteStats.fit`, not the choice of rows in `fit_neural`.

Spec 3.1 requires empirical means and SDs from the fitting rows (80%). Mutation (scratch copy C): `stats = RouteStats.fit(paths)` (all N rows). Result: `test_expa_train.py` gives **8 passed**. The effect on cost would be tiny (20% more rows in the moments), but it breaks "validation rows are kept out of fitting" and stats parity between arms.

Fix: in `test_expa_train.py` assert `np.array_equal(stats.mu_p, RouteStats.fit(Paths(p[fit_all], d[fit_all])).mu_p)` and that it differs from stats of all N rows. Same for `fit_boost`.

### 4. MAJOR (confirmed): the test-stream gate is a CLI convenience, not a barrier
`sweep.py:246-264, 433-438`; `paths.py:100-136`; `sweep.py:375-404`.

Bypasses, from easiest:
1. `route_pool(inst, "test")` returns the 20,000 test days with no check (verified: `(20000, 9)` on CON3-0). Any script can fit, tune and evaluate on it.
2. `run_eval_job(model_path, "test", ctx)` has no gate; only `run_eval` calls `check_test_gate`. Calling it directly evaluates test rows.
3. The CLI flag `--research-log` takes any file. I wrote the current HEAD hash (`368e61e4...`) into a scratch file and `check_test_gate("main","test",HEAD,scratch_log,HEAD)` passed with nothing recorded in the repo's `RESEARCH_LOG.md`. Also `run_eval(commit=...)` lets the caller assert the commit.
4. `current_commit()` reads only `HEAD`. A dirty working tree (edited code or eligibility, uncommitted) passes as "the frozen commit".
5. The frozen hash is not compared with each model's `fit_commit`. Models fitted at an earlier or later commit are evaluated on test without complaint.
6. `papers/transvrp/RESEARCH_LOG.md` does not exist yet, so the gate refuses everything until it is created. Create it before the pilot, since the spec's logging duties (tuning notes, deviations, 12.1 disclosure) go there.

None of this matters if the author is disciplined, but the spec 5 text says the gate is the protection against accidental test use, and accidents go through (1) and (2).

Fix: remove `--research-log` and `commit=` from the CLI/public signature (hard-wire the default path). Require a clean tree, using `git status --porcelain` read through the `.git` index timestamps or an explicit `git` call. Check that every model pickle in the stage has `fit_commit == frozen_commit` (or record and compare a hash of `expa/` sources). Move the test-purpose check into `route_pool`/`run_eval_job` behind a module-level token set only by a passed gate (still soft, but accidents stop).

### 5. MAJOR (confirmed by reading): analysis shrinks the Holm family silently when rows are missing
`analysis.py:197-200` (`except ValueError: return None`), `analysis.py:303-308`, `analysis.py:320-321`.

`contrast` swallows every `ValueError`, including "do not cover the same routes and replicates" and "duplicate rows". Such errors indicate a broken or incomplete job grid, not an absent contrast. `write_summary` then computes Holm over `present` only. If one SCALAR job is missing, S6 and S8 vanish and Holm uses m = 9 or 10 instead of 11. That is less conservative with no message. A missing primary row prints "Not available" instead of failing loudly. There is also no assertion that n = 20 instances, 62 routes and 3 replicates are present, and `WIN_MIN_COUNT = 13` is absolute (not scaled if n differs).

Fix: let coverage errors propagate. Treat only "policy absent from the table by design" as None (S7, S10, S11 when no big stage; M11 outside WIN). In `write_summary`, assert the expected instance and route counts. Compute Holm with m fixed at 11 (a missing contrast contributes p = 1) and list missing contrasts in `summary.md`.

### 6. MINOR/MAJOR: pre-registered outputs and checks that are not implemented
- Cluster bootstrap (section 1.2: 10,000 draws, seed 20261115) is absent. `grep bootstrap expa/` gives nothing.
- Section 4.2 diagnostic (fraction of (route, k) on a 101-point W grid where SCALAR's C-hat decreases by more than 0.01 E0) is absent.
- Variance reporting (b), the median per-day paired SE, is declared "not in this table" (`analysis.py:402`). The per-day arrays are in the `.npz` files but nothing reads them.
- Pilot criteria P3, P3', P4 (C_oracle <= C_pi <= 1.005 C_none), P6 (warm-start share < 50%), and stop rules 2, 3 and 7 are not computed by any code. The diag (`chosen_ckpt`, `disagree_rate`, `val_mse_per_ckpt`) is collected but never summarised.
- The pilot's 10^6 timing pass (data generation plus stop update with the 10^5 nets for route 3) has no job. The stage plan only covers the N = 10^5 FULL training cell.
- Mean GC and rank-biserial are reported; gap closed for rows with `none`/`oracle` missing (neural-only N cells) is NaN, which is acceptable.

Fix: add `pilot_report.py` (or an `analysis` subcommand) that computes P3, P4, P6 and stop rules 2, 3 and 7 from the merged dev CSV and the diag, and a bootstrap function with the specified seed. Implement the monotonicity diagnostic in the same place.

### 7. MINOR (suspicion, not measured): performance and memory at N = 10^6
- `sweep.py:333`, `paths.py:112-125`. `route_pool` regenerates all 10 chunks for every (route, rep) `big` job, because `lru_cache(maxsize=1)` holds only the last chunk and jobs are sorted `(inst, rep, route_id, N, ...)`. For 186 jobs that is about 1,860 chunk draws, about 3.1 h at the spec's 6 s per chunk, against the spec's 600 chunks (about 1 h). Likewise the N <= 10^5 jobs following a 10^6 job redraw chunk 0. Fix: group jobs by (inst, rep) and fit all routes of that (inst, rep) from one pass over the chunks, or keep a per-(inst, rep) route-column cache.
- `RouteCache` at N = 10^6, m = 22 holds about 13 float64 (N, m) arrays (about 2.3 GB) plus constructor temporaries; `stage_batch` at k = 1 builds float64 tokens (4e5 x 21 x 11 x 8 B = 0.74 GB), a clipped copy and a float32 copy. A conservative guess is 4-5 GB per worker, 16-20 GB for 4 workers. Fix: build tokens in float32 in chunks, drop `p`, `d`, `zp`, `zd` after the cache is built, or run the 10^6 stage with 2 workers.
- `backward.py:155-158` predicts every alive row twice more only to log the disagreement rate. At 10^6 rows this roughly triples stop-update forward cost. Fix: compute the disagreement from the already available `pA_on_B`/`pB_on_A` on a fixed subsample of 20,000 rows.
- `route_pool` for 10^6 builds all routes of the instance, with lists plus concatenation (transient 2x). Pass the single route needed.

### 8. MINOR: cross-fit leakage is first-order only; checkpoint selection falls back to training MSE
- Row i in fold A is decided at k by net_k^B. Net_k^B was fitted on B rows whose targets contain decisions of nets_{j>k}^A, fitted on A rows including row i's outcomes, and warm starts carry the same chain. This is second-order, small, and accepted by the spec (section 12.2 reads the guarantee as fit/apply disjointness). The P3' flag (SCALAR cheaper than BATON-eq by more than 1%) is the right check; make sure the pilot report computes it (finding 6).
- `train.py:143-163`: when no validation row is alive at stage k (likely for late k at N = 100 and 500, where validation has 20 and 100 rows), `select_on` becomes "train" and the lowest-training-MSE checkpoint (usually the last) is chosen, so there is no early stopping. The spec says only that validation MSE selects the checkpoint. Record `select_on` per stage in the pilot report and say so for N = 100 and 500 (already partly stated in section 10 for N = 100).

### 9. MINOR: smaller items
- `boost.py:62-63`: no `E <= E0` guard (only `fit_neural` checks). Harmless since boost does not scale, but cheap to add.
- `analysis.py:83`: `scipy.stats.wilcoxon(method="exact")` ignores ties; fine for continuous instance means, but a degenerate constructed vector (many equal Δ_i) would give a non-exact p. Document it.
- `paths.py:64-67`: the lru cache key contains `rep`, so dev/test chunks are redrawn when `rep` differs, although rep is ignored. Pass a constant for non-train purposes.
- `test_std` and `test` use the same seed (`base + 99991`) with different n. The delivery matrices share their leading rows (row-major fill), so `test_std` deliveries are the first 2,000 test rows. It is only used for parity tests; just do not use it for any tuning.
- `sweep.py:177-178`: `ARM_POLICY["boost"]` and the extra "boost" stage are not in the M9 list, but they are documented in the code. Fine.
- `sweep.py` registers `smoke`, `pilot`, `main` and `big` as separate model/row directories, and `analysis._combine` expects 10^6 classical rows to agree between `main` and `big`. This is tested (`test_big_stage_disagreeing_with_main_is_refused`), good.

## Checks that passed (no defect found)

- Step indexing: iso driver matches `fit_lsm_general` on CON3-0 (`atol 1e-9`, every k); `tau` uses `_overflow_step`; initial `future = E[tau-1]`; stops only on `alive`.
- Few-alive handling: `>=20` rule, carry of the warm net, constant fallback (mean target, or `E[min(k, m-1)]` if empty), cold flag set when the warm object is not a net (so a later stage after a constant gets a cold start, which is sensible).
- Sigmoid/E0: targets `y/E0` in [0, 1]; `E, H <= E0` asserted in `fit_neural`; predictions `E0 * u`.
- Simulator: parity test with `_simulate_costs_general` including the `none` case and `H_bill`; the oracle lower bound is asserted per path.
- Features: hand-computed toy tests (stage 1 and 2, MLP summaries), the no-peek test over all k and arms, float32 and clip checks. `HISTORY_IDX` equals g2, g8..g12.
- Seeds: train `[base, 1, rep, c]` (list), dev `[base, 3]`, test and test_std `base + 99991` (int). No collisions among the 20 `stable_seed` values; arm is excluded from the crc32 seed key (tested).
- Fairness: BATON, tuned tau, OTR-2.1, DP (10^6 only), FULL, SCALAR, MLP and BOOST all take the same `route_pool("train", rep, N)` prefix at each (route, rep, N), and the same dev/test days (rep ignored), so the comparisons are paired.
- Analysis: one-sided exact Wilcoxon at 0.025 for the primary rule, two-sided Holm at 0.05 for family rules, TOST-style CI90 equivalence only, ordering WIN, LOSE, TIE_STRONG, TIE_WEAK reachable and exhaustive, all branch boundary tests from M10 present and passing, no unreachable branch found. Instance-level aggregation: replicate mean, then route, then instance (checked on constructed data).
- Determinism: P2-style rerun gives bit-identical predictions (`test_same_key_gives_identical_predictions`); multi-process spawn works (smoke via `python -m ... --jobs 2`, CLI pickles load).

## Mutation check summary

| mutation (scratch copy) | tests run | result |
|---|---|---|
| A rows decided by net A (backward.py stop update) | backward + train, 13 tests | 13 passed (undetected) |
| `HISTORY_IDX` without g2 (peak) | features, 7 tests | 7 passed (undetected) |
| stats fitted on all N rows (train.py) | train, 8 tests | 8 passed (undetected) |

Plausible further mutations not run but reasoned: shifting `H[k-1]` to `H[k]` in the driver or `E[k_idx]` to `E[k_idx-1]` in the simulator would be caught (LS parity and simulator parity tests compare against BATON exactly); swapping `"greater"`/`"less"` in `go_no_go` would be caught by the boundary tests.
