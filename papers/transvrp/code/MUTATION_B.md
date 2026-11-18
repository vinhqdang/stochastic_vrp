# MUTATION_B: mutation checks for M12 (test_b gate) and M13 (THR, grid)

Procedure: one single-line mutation at a time is applied to the source, the targeted
tests are run with `pytest -x`, and the original bytes are restored in a `finally`
block (restoration verified by grep afterwards). "killed" = at least one test fails.

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| G1 | expa/paths.py | `if frozen not in recorded:` -> substring test on the raw log text | killed | test_expb_gate::test_a_substring_mention_no_longer_unlocks_a |
| G2 | expa/paths.py | `if gate.stream != purpose:` -> `if False:` (stream mismatch accepted) | killed | test_expb_gate::test_test_token_does_not_unlock_test_b_chunks |
| G3 | expa/paths.py | `[base, TEST_B_SEED_TAG]` -> `base + 99_991` (reuse test seed) | killed | test_expb_gate::test_test_b_size_and_seed_are_distinct_from_test |
| G4 | expb/gatecheck.py | `norm`: CRLF replacement removed | killed | test_expb_gate::test_crlf_only_change_is_accepted |
| G5 | expb/gatecheck.py | clean check: `status not in ("??","!!")` -> `if False` (tracked change ignored) | killed | test_expb_gate::test_dirty_tracked_code_refused |
| G6 | expb/gatecheck.py | `excl = excluded_for_stage(stage)` -> `[]` (EXCL dropped) | killed | test_expb_gate::test_repaired_other_arm_is_accepted_for_thr |
| G7 | expb/gatecheck.py | gate item 5 second equality (fit vs HEAD code) -> `if False` | killed | test_expb_gate::test_pre_manifest_change_of_own_arm_refused |
| G8 | expb/gatecheck.py | pin check `len(commits) != 1` -> `< 1` (several pinning commits allowed) | **survived, then killed** (see below) | test_expb_gate::test_family_line_edited_in_second_commit_refused |
| G9 | expb/gatecheck.py | model file hash comparison -> `if False` | killed | test_expb_gate::test_model_hash_differs_from_manifest_refused |
| G10 | expb/gatecheck.py | library versions comparison -> `if False` | killed | test_expb_gate::test_library_version_mismatch_refused |
| G11 | expb/gatecheck.py | FAMILY list vs derivation block comparison -> `if False` | killed | test_expb_gate::test_family_list_differing_from_block_refused |
| G12 | expa/sweep.py | stage test_stream check -> `if False` (A stage accepts test_b) | killed | test_expb_gate::test_a_stage_refuses_test_b_and_b_stage_refuses_test |
| G13 | expb/gatecheck.py | `PIN_PATTERN` anchor `^(...)` removed (mid-line mention pins) | killed | test_expb_gate::test_later_prose_mention_of_the_pin_phrase_is_not_a_second_pin |
| G14 | expb/gatecheck.py | model directory vs manifest exact-name check `on_disk != pinned` -> `if False` | killed | test_expb_gate::test_stray_pkl_not_in_manifest_is_refused |
| G15 | expb/gatecheck.py | library payload comparison `recorded != expected` -> `if False` | killed | test_expb_gate::test_library_mismatch_in_a_later_payload_is_refused (also test_library_version_mismatch_refused) |
| G16 | expb/gatecheck.py | index-flag check `line[0] != INDEX_FLAG_TAG` -> `if False` (skip-worktree allowed) | killed | test_expb_gate::test_skip_worktree_flag_on_tracked_code_is_refused |
| G17 | expb/gatecheck.py | status pathspec without top-level `code/*.py` | killed | test_expb_gate::test_untracked_top_level_code_module_is_refused |
| G18 | expb/gatecheck.py | `git log -m` -> `git log` (merges not diffed) | killed | test_expb_gate::test_merge_commit_editing_the_pinned_line_is_refused |
| G19 | expb/gatecheck.py | gate item 3 `len(hits) != 1` -> `< 1` (stage in two FROZEN lines allowed) | killed | test_expb_gate::test_stage_in_two_frozen_lines_refused |
| G20 | expb/gatecheck.py | gate item 4 ancestor check `returncode != 0` -> `if False` | killed | test_expb_gate::test_fit_commit_not_ancestor_refused |
| G21 | expb/gatecheck.py | gate item 4 fit blob: FAMILY-line check off | killed | test_expb_gate::test_fit_commit_log_holding_family_line_refused |
| G22 | expb/gatecheck.py | gate item 4 fit blob: FROZEN-stage check off | killed | test_expb_gate::test_fit_commit_log_holding_family_line_refused |
| G23 | expb/gatecheck.py | manifests hash comparison (item 6, first half) -> `if False` | killed | test_expb_gate::test_manifest_changed_after_manifest_commit_refused |
| G24 | expb/gatecheck.py | fidelity hash comparison (item 7, first half) -> `if False` | killed | test_expb_gate::test_fidelity_hash_mismatch_refused |
| G25 | expb/gatecheck.py | default-deny: `code_files` restricted to `/expb/` (allow-list) | killed | test_expb_gate::test_pre_manifest_change_of_pilot_refused_for_every_stage (also test_allow_list_mutant_guard_every_code_file_is_hashed) |
| G26 | expb/gatecheck.py | `text_hash` reads the working-tree file instead of the blob | killed | test_expb_gate::test_text_hash_reads_the_committed_blob_not_the_working_copy |
| G27 | expa/sweep.py | run_eval_job `gated = eval_set in TEST_STREAMS` -> `== "test"` | killed | test_expb_gate::test_run_eval_job_routes_test_b_through_its_gate |
| G28 | expa/sweep.py | run_eval_job stage/stream check removed (`check_stream_for_stage` -> `pass`) | killed | test_expb_gate::test_run_eval_job_refuses_a_stage_on_another_stream |
| G29 | expa/sweep.py | run_eval_job model-of-another-stage check (`owner is not None and owner != ctx.stage`) -> `if False` | killed | test_expb_gate::test_run_eval_job_refuses_a_model_of_another_stage |
| G30 | expa/sweep.py | `check_stream_for_stage` stream equality -> `if False` | killed | test_expb_gate::test_a_stage_refuses_test_b_and_b_stage_refuses_test (also test_run_eval_job_refuses_a_stage_on_another_stream) |
| G31 | expb/gatecheck.py | bytecode scan: `.pyc`/`.pyo` filter -> `if True` | killed | test_expb_gate::test_sourceless_pyc_outside_pycache_refused |
| G32 | expb/gatecheck.py | orphan bytecode: tracked-source test -> `if False` | killed | test_expb_gate::test_orphan_pycache_file_refused |
| T1 | expb/thr.py | breach cost `E[k]` -> `E[k-1]` | killed | test_expb_thr::test_two_stop_route_matches_hand_computation |
| T2 | expb/thr.py | `V_m = 0` -> `V_m = H[m-1]` | **survived, then killed** (see below) | test_expb_thr::test_two_stop_route_matches_hand_computation |
| T3 | expb/grid.py | lower clip at w_lo removed | killed | test_expb_thr::test_predict_is_linear_interpolation_and_clips_below_the_grid |
| T4 | expb/grid.py | tie tolerance sign flipped (`+ tie_tol` -> `- tie_tol`) | killed | test_expb_thr::test_tie_tolerance_is_added_to_the_prediction |
| T5 | expb/thr.py | threshold test `>=` -> `>` (tie direction) | killed | test_expb_thr::test_thresholds_use_the_tie_direction_of_t1 |
| T6 | expb/thr.py | permutation without replacement -> with replacement | killed | test_expb_thr::test_permutation_preserves_column_marginals_exactly |
| T7 | expb/thr.py | `V_k = min(H[k-1], C_k)` -> `min(H[k], C_k)` | killed | test_expb_thr::test_two_stop_route_matches_hand_computation |
| T8 | expb/thr.py | breach test `x > B` -> `x >= B` | killed | test_expb_thr::test_two_stop_route_matches_hand_computation |

## Survivors and how they were closed

- **G8 (survived first run).** The test only exercised item 2 indirectly: the edited
  FAMILY line was refused by the family-list check, so the "exactly one pinning commit"
  rule was never the reason for refusal. The test now also calls
  `gatecheck.check_pin_commit` directly and expects "touched the pinned lines". Re-run:
  killed.
- **T2 (survived first run).** The toy routes had H[m-1] = 0, so `V_m = H[m-1]` is
  numerically identical to `V_m = 0` on those inputs (an equivalent mutant). The toy
  routes now set the unused final H to a nonzero value (0.7 and 0.6); the policy never
  reads H[m-1], but the DP does under the mutant. Re-run: killed.

## Limits of the check

- Each mutation is a single-line change, checked with the tests named above only (a
  kill shows that some test detects the mutant, not that the mutant is the only
  detectable fault).
- Not mutated (no equivalent single-line change exists in the stated scope): the
  library-version payload reader (`sweep.load_payload` call), the `family-line` report.
- Rows G13 to G32 were run in review round 2 (CODE_REVIEW_B_M12_M13). Each row names the
  test(s) that the selection `pytest -x -k` ran for that mutant; a kill means at least one
  selected test failed, and the original bytes were restored and verified after each run.

# MUTATION_B: mutation checks for M14 (ALP-Z), M15 (RLSM), M16 (DOS)

Same procedure as above: one single-line mutation at a time, the module's test file run
with `pytest -x`, original bytes restored in a `finally` block and verified byte-for-byte
afterwards. "killed" = at least one test fails.

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| A1 | expb/alp.py | `linprog(-c, ...)` -> `linprog(c, ...)` (maximise -> minimise) | killed | test_expb_alp::test_lp_initial_objective_plus_breach_constant_is_the_dp_initial_value |
| A2 | expb/alp.py | handoff bounds `ub = H[k-1]` -> `ub = inf` (handoff rows dropped) | killed | test_expb_alp::test_lp_uniform_objective_reproduces_the_dp_at_every_lattice_point |
| A3 | expb/alp.py | next-stage index `max(i+r, 0)` -> `max(i, 0)` in the continue rows | killed | test_expb_alp::test_lp_initial_objective_plus_breach_constant_is_the_dp_initial_value |
| A4 | expb/alp.py | breach test `J > n-1` -> `J >= n-1` in the lattice DP | killed | test_expb_alp::test_lattice_dp_matches_the_loop_reference |
| A5 | expb/alp.py | deployed tie `GridPredictor(..., tie_tolerance(E0))` -> `(..., 0.0)` (strict L > H) | **survived, then killed** (see below) | test_expb_alp::test_deploy_predictor_hands_off_at_the_tie_on_the_lattice |
| A6 | expb/alp.py | breach cost in the deployed table `E[k]` -> `E[k-1]` | killed | test_expb_alp::test_deployed_rule_equals_thr_on_a_lattice_aligned_toy |
| R1 | expb/rlsm.py | LeakyReLU `slope*z` -> `0*z` (ReLU) | killed | test_expb_rlsm::test_leaky_relu_uses_the_slope_on_the_negative_branch |
| R2 | expb/rlsm.py | intercept column `ones` -> `zeros` in the design | killed | test_expb_rlsm::test_k1_slope1_zero_ridge_is_ordinary_least_squares_on_a_constructed_target |
| R3 | expb/rlsm.py | ridge `lam = ridge*n` -> `0*ridge*n` (penalty ignored) | killed | test_expb_rlsm::test_ridge_penalty_is_lambda_times_n_fit |
| R4 | expb/rlsm.py | augmented system `sqrt(lam) I` -> `lam I` | killed | test_expb_rlsm::test_ridge_lstsq_matches_the_closed_form_and_uses_one_solve |
| R5 | expb/rlsm.py | random layer `standard_normal` -> `uniform(-1, 1)` | killed | test_expb_rlsm::test_draw_layer_is_standard_normal |
| R6 | expb/rlsm.py | seed key without the fold label | killed | test_expb_rlsm::test_draw_layer_is_seeded_and_differs_by_stage_and_fold |
| R7 | expb/rlsm.py | prediction clip to [0, E0] removed | killed | test_expb_rlsm::test_predictions_are_clipped_to_zero_and_E0 |
| R8 | expb/rlsm.py | fallback test `n < min_rows` -> `n <= min_rows` | killed | test_expb_rlsm::test_exactly_min_rows_is_fitted_not_constant |
| R9 | expb/rlsm.py | constant fallback value `mean(y)` -> `E[0]` | killed | test_expb_rlsm::test_too_few_rows_give_the_constant_fallback |
| D1 | expb/dos.py | loss `F*Hs + (1-F)*u` -> `F*u + (1-F)*Hs` (H and y swapped) | killed | test_expb_dos::test_loss_is_the_stop_or_continue_cost_on_the_E0_scale |
| D2 | expb/dos.py | surrogate sign `H + E0(pi-1/2)` -> `H - E0(pi-1/2)` | killed | test_expb_dos::test_surrogate_is_H_plus_E0_times_pi_minus_half |
| D3 | expb/dos.py | warm start `deepcopy(warm.net)` -> fresh `DOSNet()` | killed | test_expb_dos::test_warm_start_copies_the_previous_stage_network |
| D4 | expb/dos.py | warm test `isinstance(warm, DOSPredictor)` -> `warm is not None` | killed | test_expb_dos::test_cold_start_when_the_warm_predictor_is_constant |
| D5 | expb/dos.py | batch norm replaced by `Identity` | killed | test_expb_dos::test_network_shapes_range_and_architecture |
| D6 | expb/dos.py | surrogate without centring `(pi - 0.5)` -> `(pi)` | killed | test_expb_dos::test_surrogate_is_H_plus_E0_times_pi_minus_half |
| D7 | expb/dos.py | checkpoint score on validation `score = vl` -> `score = tl` (train selection) | **survived, then killed** (see below) | test_expb_dos::test_checkpoint_is_the_earliest_minimum_of_the_validation_loss_not_the_training_loss |
| D8 | expb/dos.py | step floor `max(s_min, raw)` -> `max(1, raw)` | killed | test_expb_dos::test_stage_steps_follow_the_formula_and_its_floor_and_cap |

## Survivors and how they were closed

- **A5 (survived first run).** The tie test built its own `GridPredictor`, so the tie
  argument inside `fit_alpz` was never exercised. The deployment step was moved into
  `alp.deploy_predictor`, which `fit_alpz` calls, and a test now hands a vertex through
  `deploy_predictor` and `simulate_policy` on the lattice. Re-run: killed.
- **D7 (survived first run).** The checkpoint test asserted the choice equals the
  validation argmin, but on its single seed the training and validation argmins coincided,
  so the mutant passed. The test now runs eight seeds and asserts (i) the choice equals the
  validation argmin on each and (ii) at least one seed where the two argmins differ, so a
  training-loss selection cannot pass. Re-run: killed.

## Limits of the check

- Single-line mutants only, checked against the named tests (a kill shows a test detects
  the mutant, not that it is the only detectable fault).
- Not mutated: the LP solver call itself (its status check is covered by the status-0
  assertions), the deterministic-seed plumbing of `torch` (covered by the reproducibility
  test only), and the budget bookkeeping.

# MUTATION_B: mutation checks for M17 (padded tables, masked SetNet) and M18 (DQN)

Same procedure as above: one single-line mutation at a time, the module's targeted test
file run with `pytest -x`, original bytes restored in a `finally` block and verified
byte-for-byte afterwards (sha1 before and after). "killed" = at least one test fails.

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| P1 | expb/padded.py | token mean mask `masked_fill(pad..., 0.0)` -> `pad.unsqueeze(-1) & False` (pads enter the mean) | killed | test_expb_padded::test_pad_token_values_do_not_change_the_output |
| P2 | expb/padded.py | encoder mask polarity `src_key_padding_mask=keep_pad` -> `~keep_pad` | killed | test_expb_padded::test_masked_net_equals_setnet_on_unpadded_batches |
| P3 | expb/padded.py | encoder layers run unmasked `layer(h, src_key_padding_mask=keep_pad)` -> `layer(h)` | killed | test_expb_padded::test_pad_token_values_do_not_change_the_output |
| P4 | expb/padded.py | token mean divides by the padded length `n_real` -> `pad.shape[1]` | killed | test_expb_padded::test_masked_mean_is_the_mean_over_real_tokens_only |
| P5 | expb/padded.py | pad mask off by one `slots >= n_real` -> `slots > n_real` | killed | test_expb_padded::test_table_shapes_and_pad_mask_by_stage |
| P6 | expb/padded.py | stage range check `< 1` -> `< 0` | killed | test_expb_padded::test_out_of_range_stage_is_refused |
| Q1 | expb/dqn.py | double-Q: online argmax replaced by target test `q_on > h` -> `q_tg > h` (single network) | killed | test_expb_dqn::test_double_q_target_on_a_two_state_toy |
| Q2 | expb/dqn.py | terminal mask dropped `c + (1 - done) * v` -> `c + v` | killed | test_expb_dqn::test_double_q_target_on_a_two_state_toy |
| Q3 | expb/dqn.py | behaviour hands off: `always_continue` returns `np.zeros` | killed | test_expb_dqn::test_transitions_match_the_simulated_continue_path_row_by_row |
| Q4 | expb/dqn.py | handoff rows keep going and store transitions `act = act[cont]` -> `act = act` | killed | test_expb_dqn::test_handoff_behaviour_stores_no_transition_and_stops_the_path |
| Q5 | expb/dqn.py | breach cost index `E[k]` -> `E[k - 1]` | killed | test_expb_dqn::test_transitions_match_the_simulated_continue_path_row_by_row |
| Q6 | expb/dqn.py | learn-start gate `buf.size < learn_start` -> `False` | killed | test_expb_dqn::test_no_update_before_learn_start_and_the_buffer_is_capped |
| Q7 | expb/dqn.py | one update per 32 -> per 16 stored transitions (`since -= per_update // 2`) | killed | test_expb_dqn::test_update_count_is_one_per_per_update_transitions |
| Q8 | expb/dqn.py | target period doubled in the copy schedule (`next_target += 2 * T`) | killed | test_expb_dqn::test_target_copies_happen_every_target_period_in_transitions |

## Survivors and how they were closed

- No mutant survived its first run.
- Test-suite changes made while building the checks (not survivors): `make_setup` uses a
  non-constant E (`linspace(1, 0.5, m)`), so `E[k]` -> `E[k-1]` is not an equivalent mutant;
  a handoff-behaviour test (`test_handoff_behaviour_stores_no_transition_and_stops_the_path`)
  was added so that Q4 is detectable at all.
- Implementation change made while timing (not a mutant): the cuBLAS determinism variable
  is set at import of `expb/dqn.py` instead of inside `train_dqn_record`, because it has to
  be set before CUDA initialises; the in-function setting failed on the first CUDA run.

## Limits of the check

- Single-line mutants only, each checked against the named tests (a kill shows that a test
  detects the mutant, not that it is the only detectable fault).
- Not mutated: the `pooled` head (covered by the equivalence test), checkpoint hashing,
  the `RouteStats` split, and the budget bookkeeping. The checkpoint tie rule (`<` ->
  `<=`) was not mutated: no test forces an exact tie; the earliest-tie property is asserted
  only on the random toy and holds there, so a tie-breaking mutant could survive.
