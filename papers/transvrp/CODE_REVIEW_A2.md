# Re-review of EXPERIMENT A code after fix batches 1e73246 and e0979b4

Scope: `git diff 88b92f4 e0979b4 -- papers/transvrp .gitignore` against `CODE_REVIEW_A.md` and `EXPERIMENT_A.md` (sections 1.2, 4.2, 5, 7, 8, 10). Review date 2026-11-16. No repository file other than this one was written; scratch copies for comparisons and mutations were kept outside the repository. The test stream was never evaluated, and no pilot, sweep or large-scale run was started.

## Verdict: READY WITH FIXES

All five majors of the first review are closed, and my own mutations of the old defects are now caught. The gate fails closed in every ordinary failure mode I tried. No blocker for a pilot on the dev stream. Three things should be fixed before or during the pilot because the pilot report depends on them: the P5 projection can never be computed (finding N1), `RESEARCH_LOG.md` still does not exist, and memory reported as "cache plus stop" does not include the stop updates (N3). A few gate gaps matter before the real test run, not before the pilot.

Test suite: `python3 -m pytest tests -q` from `papers/transvrp/code` gives **175 passed in 339 s** (run while other jobs shared the 4 cores).

## Status of the findings of CODE_REVIEW_A.md

| # | finding | status | evidence |
|---|---|---|---|
| 1 | no test detects a broken cross-fit | RESOLVED | New `test_expa_backward.py::test_cross_fit_decisions_never_use_a_predictor_fitted_on_the_row` (spy predictors, records per-call rows). My mutation (A rows decided by net A, `backward.py:176`) now fails that test (1 failed). The log is still written from variables, not from the call arguments, but the spy covers the behaviour. |
| 2 | SCALAR could leak the peak | RESOLVED | `test_expa_features.py::test_scalar_blind_to_running_peak_at_stage_2`, `..._zeroes_the_spec_history_columns_and_nothing_else`, `..._function_of_k_and_W_only`. Mutation `HISTORY_IDX` without g2: fails on the literal-columns test. |
| 3 | fit statistics may include validation rows | RESOLVED | `test_expa_train.py::test_fit_neural_statistics_are_fitted_on_the_80_percent_rows_only`; my mutation `RouteStats.fit(paths)` (`train.py:280`) fails it. A matching boost test exists (`test_expa_boost.py`). |
| 4.1 | `route_pool(..., "test")` unguarded | RESOLVED (public API) | `paths.py:171-172` calls `verify_test_gate` before anything is generated; verified with `frozen=None`, wrong case, no log. Residual: private `_chunk_arrays("CON3-0","test",0,0,None)` still returns the 20,000 test days (I ran it); see N5. |
| 4.2 | `run_eval_job` unguarded | RESOLVED | `sweep.py:353-354` verifies before the skip check; `test_run_eval_job_refuses_test_without_the_gate`. |
| 4.3 | `--research-log` / `commit=` redirect the gate | RESOLVED | Both removed from CLI and `run_eval`; `test_research_log_cannot_be_redirected_from_the_cli`. `RESEARCH_LOG` and `current_commit` are module globals that only a test can patch. |
| 4.4 | dirty tree passes as frozen commit | PARTLY | `tree_is_clean` (`paths.py:88-98`) uses `git status --porcelain --untracked-files=no -- .` under the code directory. Tracked edits are refused (scratch repo: refused). Not covered: untracked new `.py` files (PASS in my probe), `git update-index --assume-unchanged` edits (PASS), ignored files, and anything outside `code/`. See N5. |
| 4.5 | model `fit_commit` not compared | RESOLVED | `sweep.py:361-363`; test `..._refuses_models_not_fitted_at_the_frozen_commit`; mutation `if False:` is caught. Caveat: `fit_commit` is HEAD at fit time even if the tree was dirty then (N5). |
| 4.6 | `RESEARCH_LOG.md` missing | NOT RESOLVED | `papers/transvrp/RESEARCH_LOG.md` does not exist (checked). The gate therefore refuses all test evaluation, which is safe. The log is also where the spec requires tuning notes, deviations and the 12.1 disclosure; create it before the pilot's first tuning iteration. |
| 5 | analysis shrinks Holm family silently | RESOLVED | `analysis.py` `check_coverage` (instances, routes, replicates, per policy), `contrast` no longer swallows errors, `write_summary` runs `check_coverage` on PRIMARY and all 11 family tests, `FAMILY_SIZE = 11` asserted, `assert len(adj) == FAMILY_SIZE`. Tests in `test_expa_analysis.py` and `test_expa_analysis_b.py`. S7/S10/S11 use the all-routes classical 10^6 rows of the main stage (spec 5), so strict 62-route coverage is correct. |
| 6a | cluster bootstrap missing | RESOLVED (code) | `analysis.py` `cluster_bootstrap`: 10,000 draws, seed 20261115, one `integers` call, wired into `write_summary`. Test-quality gap: mutating the CI95 percentiles from [2.5, 97.5] to [5, 95] survives all 16 tests in `test_expa_analysis_b.py` (N6). |
| 6b | section 4.2 monotonicity diagnostic missing | PARTLY | `w_grid_curve`, `monotonicity_curves`, `monotonicity_diagnostic` exist with unit tests, but nothing calls them: not in `write_summary`, not in `pilot.py`, no CLI. Needs a driver before the main analysis (N4). |
| 6c | variance (b) missing | PARTLY | `variance_b_median` is wired into `write_summary`, but `analysis.py` passes `eval_set="dev"` fixed (see N2) and labels the value "%-points of route cost" although it is in cost units. |
| 6d | pilot P3, P3', P4, P6, rules 2, 3, 7 | RESOLVED | `pilot.py` implements them; boundary tests in `test_expa_pilot.py`. Mutations of P3's tolerance and rule 7's threshold are caught. P5 is only half done (N1). |
| 6e | 10^6 timing pass | RESOLVED (code), memory reading flawed | `sweep.run_timing` and the `timing` subcommand; `test_expa_timing.py`. See N3. |
| 6f | gap closed NaN | n/a | accepted in the first review. |
| 7a | `route_pool` regenerates chunks for every big job | RESOLVED | on-disk chunk cache, `routes=` restriction, job order already (inst, rep, route). Cached read 0.03 s against 5.8 s generation (CON3-0, N = 1500). |
| 7b | memory at 10^6 | RESOLVED, unmeasured | `RouteCache` drops p/d references, zsum in float32, `stage_batch`/`mlp_batch` build in 65,536-row blocks. Not measured at 10^6 (not allowed here); the timing stage is the way to measure it. |
| 7c | disagreement doubles forward cost | RESOLVED | `backward.py` `_disagreement`, fixed 20,000-row evenly spaced subsample. |
| 7d | route_pool builds all routes | RESOLVED | `routes=[...]` used in `run_fit_job`, `run_eval_job`, `run_timing`. |
| 8 | first-order cross-fit leakage; select_on fallback | RESOLVED | P3' flag implemented; `diag_table` carries `select_on`, P6 reports `stages_select_on_train`. |
| 9a | boost lacks E <= E0 guard | RESOLVED | `boost.py:75-76`. |
| 9b | Wilcoxon exact with ties | RESOLVED (documented) | `_wilcoxon_p` docstring. |
| 9c | rep in dev/test cache key | RESOLVED | `paths.py:174-175` fixes rep = 0 for non-train purposes. I checked dev pools for rep 0 and rep 2 are bit-identical, and that this equals what the old code produced (the old dev stream also ignored rep), so earlier parity assumptions hold. |
| 9d | test_std shares seed with test | RESOLVED (documented) | stated in the `paths.py` docstring. |
| 9e/9f | boost stage, big/main combine | n/a | no defect. |

## Checks on the new code

**Gate (a).** Probe in a scratch git repository holding a copy of the new code:

| situation | result |
|---|---|
| clean tree, HEAD equals logged hash | PASS |
| 12-character prefix of the hash | refused (full-hash regex) |
| upper-case hash | refused |
| tracked file edited | refused |
| git not on PATH | refused (fails closed) |
| HEAD moved after logging | refused |
| detached HEAD at the logged hash | PASS (correct) |
| untracked new `.py` under `code/` | PASS (gap) |
| tracked edit hidden with `update-index --assume-unchanged` | PASS (gap, adversarial) |

Log substring check: the hash may appear anywhere in the log (a comment would do). `current_commit` handles loose refs and packed-refs but not worktrees (`commondir`); in a worktree it returns "unknown" and refuses, which is safe. Because the log lives outside `code/`, it can be committed after the frozen code commit without making the tree dirty; that works as intended.

**Chunk cache (b).** Cached and uncached pools are bit-identical (train, rep 1, N = 1500, all routes, and the one-route read; dev pools for rep 0 and 2). Prefix property holds (N = 700 equals the first 700 rows of N = 1500). The key is (instance, purpose, rep, chunk, n, crc32 of the seed repr); N is not needed because a request takes a prefix of the fixed chunk. Writes use a per-pid temporary file and `os.replace`, so a reader never sees a partial file, and two workers writing the same chunk write identical bytes. Only weakness: the key has no version of the generator or plan files, so a changed `gen_scenarios` or plan would be served stale (N7). Disk size is not bounded (N7).

**Float32 and blockwise features (d).** Compared old (88b92f4) and new code on CON3-0 route 3, dev rows, stages 1, 5, 10, 16, FULL, SCALAR, MLP. SCALAR: bit-identical in every case. FULL and MLP: token features identical; the global vector differs in about 4% of entries by at most 2.4e-7 (one float32 unit), only in the mean-standardised outcome columns at stages where zsum is float32-rounded before the division. Not material for results (no results exist yet) and still deterministic, but note that FULL/MLP are not bit-identical to the pre-change code. Block size does not change the output (tested; I also found no ordering dependence by reading `stage_batch`).

**Analysis (e).** Holm over 11, strict coverage, cluster bootstrap (instances resampled, routes kept together), exact Wilcoxon, classification boundaries: no defect found beyond N2 and N6. Monotonicity helper follows spec 4.2 on a W grid from min W_k to B with the other features held at fitted means, unit = (route, k) curve. Fine, but unwired (N4).

**Pilot (f).** P1 to P4 and P6, P3', stop rules 2, 3, 5, 6, 7 match the spec text, with documented readings: P3 uses a relative SD (root mean square of route-level BATON replicate SDs over the mean route cost), which is more permissive than the SD of the pooled mean; P6 pools trained stages including the cold stage. Stop rule 6 is evaluated on the uncut projection. The pilot plan has the spec's cells: 4 CON3-0 routes (m = 9, 12, 12, 17), 73 jobs, one full-size 10^5 cell. P5 incomplete (N1).

## New findings, ranked

**N1. MAJOR (confirmed): P5 projection can never be computed from the pilot.** `pilot.py` `projection_hours` needs a measured minute value for all ten plan cells (`full@100/500/2000/10^4/10^5`, `scalar@2000/10^5`, `mlp@10^5`, `full@10^6`, `scalar@10^6`); the pilot measures only `full@1000/10^4/10^5`, `scalar@1000/10^4` and the 10^6 pass. I ran it with those cells: `hours: None, missing: [full@100, full@500, full@2000, scalar@2000, scalar@100000, mlp@100000, scalar@1000000]`. So `platform_decision` and stop rule 6 never fire, and spec 7.3 P5 ("the table is recomputed, the platform rule and cut 1 are applied") is not met by code. Also `full@10^6` takes only generation plus stop updates, not the net fit. Fix: extrapolate per-job minutes in N from the measured cells (fit seconds against N on the pilot cells, scalar = full scaled by the pilot ratio, mlp = 0.2 times full as in 7.2), state the rule, and keep "measured" and "extrapolated" cells separate in the report. Until then, P5 must be judged by hand.

**N2. MAJOR for the final analysis, not for the pilot (confirmed): variance (b) is hard-wired to the dev stream.** `analysis.py` `write_summary` calls `variance_b_median(..., eval_set="dev")` although its docstring says "same stage's eval set". On the test summary it would read dev arrays (or fail). Fix: pass the eval set through from the CLI (`--eval`). Also relabel: the value is in cost units, not "%-points".

**N3. MINOR (confirmed): `peak_cache_stop_mb` does not cover the stop updates.** `sweep.py` `run_timing` wraps only `RouteCache(...)` in `traced_peak`; `simulate_policy` runs in a separate lap outside it, so the largest allocation (stage batches at k = 1) is only visible in `process_peak_rss_mb`. Fix: put the stop updates inside the traced block, or rename the field. I measured tracemalloc overhead on generation as nil (6.24 s against 6.17 s), so tracing the whole block is affordable.

**N4. MINOR (confirmed): monotonicity diagnostic has no driver.** Add a call in the analysis CLI for SCALAR at 10^5 on the models directory (spec 4.2), or a subcommand; otherwise the main report will omit a pre-registered check.

**N5. MINOR (gate gaps; confirmed by probe, adversarial only).** (i) untracked new files and `assume-unchanged` hide code changes from `tree_is_clean`; use `git status --porcelain --untracked-files=normal` limited to `code/` excluding `results/`, and `git ls-files -v | grep '^h'` for the assume-unchanged flag; (ii) `_chunk_arrays` and `_generate_chunk` accept purpose "test" with no gate; move the check into `_stream` or `_generate_chunk`; (iii) a model fitted with a dirty tree is stamped with HEAD; record `tree_is_clean()` at fit time and refuse models fitted dirty in the test gate. None matters for the dev pilot.

**N6. MINOR (confirmed by mutation): two test gaps.** (a) CI95 and CI90 percentile limits of `cluster_bootstrap` are not tested (swap survives 16 tests). (b) `tree_is_clean`'s dirty detection on a real git repository is not tested: replacing the return by `out.returncode == 0` passes all gate tests; only "outside a repository" is exercised. Fix: a test that builds a tiny repository in `tmp_path`, edits a tracked file, and asserts `False`; and a bootstrap test with a known two-point distribution and exact percentiles.

**N7. MINOR (suspected, not measured at scale).** Cache grows without bound: about 80 MB per 10^5-row chunk for 51-column Dethloff instances, so 10 chunks times 3 replicates times 20 instances is on the order of 50 GB when the main stage fits classical 10^6 models for every replicate; this machine has 27 GB free. Concurrent workers can also generate the same chunk twice (correct but wasteful). Delete chunks of finished (instance, rep) pairs, or put the cache on a larger volume; add a generator version to the key.

**N8. MINOR (confirmed by reading).** `stop_rule_6` uses the uncut projection (spec: after the platform rule and all cuts); P3's SD reading is the permissive one. Both are documented in `pilot.py`; keep the choice in the pilot report.

## Mutations of my own on the new code (scratch copy)

| mutation | tests run | result |
|---|---|---|
| gate: `tree_is_clean` returns True | sweep gate tests | caught (`test_tree_check_fails_closed_outside_a_repository`) |
| gate: `tree_is_clean` ignores git's stdout | sweep gate tests | **survives** (8 passed) |
| gate: model `fit_commit` check disabled | sweep gate tests | caught |
| chunk file name drops the replicate | `test_expa_chunks.py` | caught (`test_chunk_file_key_contains_all_five_parts`) |
| bootstrap CI95 percentiles [5, 95] | `test_expa_analysis_b.py` | **survives** (16 passed) |
| pilot P3 tolerance without the 3 SD term | `test_expa_pilot.py` | caught |
| pilot stop rule 7 threshold changed | `test_expa_pilot.py` | caught |
| old defects: cross-fit by own net; SCALAR without peak zeroing; stats on all rows | backward, features, train tests | all three caught |

## Readiness for the pilot (spec 7.3, dev stream)

Yes, with these steps first: (1) create `papers/transvrp/RESEARCH_LOG.md` and log the tuning and deviations from the start; (2) accept that P5 will be judged by hand until N1 is fixed (the timing JSON and per-job `fit_seconds` are recorded either way); (3) run `sweep timing` after the `full@10^5` route-3 job and read `process_peak_rss_mb` rather than `peak_cache_stop_mb` (N3); (4) keep the chunk cache off the repository (the `.gitignore` rule covers `results/**/chunks/`, and `--chunk-dir` is available). No blocker found: the pilot plan (73 jobs, four CON3-0 routes) and the P1-P4, P6 and stop-rule checkers work on the dev stream, and the gate keeps the test stream closed.
