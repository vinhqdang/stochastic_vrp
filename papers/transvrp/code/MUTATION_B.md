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

## M18 fast update path (launch-overhead optimisation, pre-freeze)

Same procedure: one single-line mutation of the new fast path in `expb/dqn.py` at a time,
`pytest tests/test_expb_dqn.py -x`, original bytes restored in a `finally` block, sha1 of
the file verified equal before and after (restored: True). The equality tests compare the
fast path with `_update_reference` (the old update, verbatim) in `tests/test_expb_dqn.py`:
CPU bitwise (60 updates incl. a target copy; and a full `train_dqn_record` run, identical
checkpoint hashes), GPU CUDA-graph vs reference relative L2 parameter error <= 1e-5 after
200 updates at lr 1e-4 (measured about 1e-6; fused Adam is the only numerical difference;
graph replay equals the same update run eagerly bitwise).

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| F1 | expb/dqn.py | next stage `clamp(ks + 1, max=m - 1)` -> `clamp(ks, ...)` | killed | test_expb_dqn::test_fast_update_equals_reference_bitwise_on_cpu |
| F2 | expb/dqn.py | packed batch swaps cost and done columns | killed | test_expb_dqn::test_one_stage_route_converges_to_the_breach_cost_within_two_percent |
| F3 | expb/dqn.py | fused gather stage index `cat([ks, k_next]) - 1` -> no `- 1` | killed | test_expb_dqn::test_always_continue_visits_every_stage_with_alive_rows |
| F4 | expb/dqn.py | cached pad table `>=` -> `>` | killed | test_expb_dqn::test_fast_update_equals_reference_bitwise_on_cpu |
| F5 | expb/dqn.py | graph capture restore `p.copy_(q)` -> `p.add_(0.0)` (warm-up leaks into training) | killed | test_expb_dqn::test_graph_update_matches_reference_on_gpu_within_1e5 |
| F6 | expb/dqn.py | graph capture does not zero optimiser state `v.zero_()` -> `v.add_(0.0)` | killed | test_expb_dqn::test_graph_update_matches_reference_on_gpu_within_1e5 |

Not mutated: the pinned-buffer ring synchronisation (a race, not deterministically
detectable by a test; reviewed by reading).

# MUTATION_B: mutation checks for M19 (PPO), M20 (codecs, registry)

Same procedure as above: one single-line mutation at a time, the targeted tests run with
`pytest -x -k <selector>`, original bytes restored and verified by sha1 before and after.
Every mutant was restored byte-for-byte (`restored: true` in the run log).

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| P1 | expb/ppo.py | clipped term `clamp(ratio, 1-clip, 1+clip) * adv` -> `ratio * adv` (no ratio clipping) | killed | test_expb_ppo::test_clipped_loss_matches_hand_computation |
| P2 | expb/ppo.py | advantage `returns - values_old` -> `values_old - returns` (sign flipped) | killed | test_expb_ppo::test_policy_learns_the_optimal_threshold_on_the_one_stage_toy |
| P3 | expb/ppo.py | handoff reward `-H/E0` -> `-H` (not scaled by E0) | killed | test_expb_ppo::test_step_reward_is_scaled_by_E0_and_follows_the_breach_convention |
| P4 | expb/ppo.py | recomputed GRU state uses zero hidden at each stage (state not carried) | killed | test_expb_ppo::test_gru_state_is_carried_across_decision_stages |
| P5 | expb/ppo.py | handoff masked only for `k < m` -> `k <= m` (handoff allowed at stage m) | killed | test_expb_ppo::test_masked_logits_give_zero_handoff_probability_at_stage_m |
| P6 | expb/ppo.py | initial policy bias `logit(init_p)` -> `0` (initial handoff 0.5) | killed | test_expb_ppo::test_initial_handoff_probability_is_five_percent_at_every_state |
| P7 | expb/ppo.py | return recursion `r + gamma*acc` -> `r + acc` (gamma ignored) | killed | test_expb_ppo::test_discounted_returns_match_hand_computation_for_gamma_half_and_one |
| P8 | expb/ppo.py | empty-episode skip `if ep.ks.size == 0` -> `if False` (update on no decisions) | killed (by a raised error in the update, not an assertion) | test_expb_ppo::test_episodes_without_decisions_make_no_update |
| C1 | expb/codecs.py | unpack builds an untrained network (`load_state_dict` removed) | killed | test_expb_codecs::test_round_trip_predictions_are_identical_for_every_arm[dqn] and test_unpacked_network_carries_the_trained_weights |
| C2 | expb/codecs.py | shared-network index check `id(net) not in index` -> `True` (one copy per predictor) | killed | test_expb_codecs::test_shared_network_is_stored_once_per_job (3 networks, expected 1) |
| C3 | expb/codecs.py | grid tie tolerance dropped on pack (`tie_tol` -> 0.0) | killed | test_expb_codecs::test_round_trip_predictions_are_identical_for_every_arm[thr] |
| C4 | expb/codecs.py | DOS input dimension on pack `in_features` -> 14 | killed | test_expb_codecs::test_round_trip_predictions_are_identical_for_every_arm[dos] (load error) |
| C5 | expb/codecs.py | `libraries` not recorded (`library_versions()` -> `{}`) | killed | test_expb_codecs::test_blob_records_the_library_versions_of_the_running_process |
| C6 | expb/codecs.py | PPO predictor E0 doubled on unpack | killed | test_expb_codecs::test_round_trip_predictions_are_identical_for_every_arm[ppo] |
| C7 | expb/codecs.py | constant predictor value dropped on unpack (`value` -> 0.0) | killed | test_expb_codecs::test_constant_fallback_predictors_round_trip |
| R1 | expb/registry.py | ALP-Z bound to the THR fit function | killed | test_expb_registry::test_alpz_budget_reports_the_lp_size_and_thr_budget_the_grid |
| R2 | expb/registry.py | THR `split=False` -> `True` | killed | test_expb_registry::test_split_and_predictor_kind_follow_section_3 |
| R3 | expb/registry.py | PPO predictor kind "surrogate" -> "cost" | killed | test_expb_registry::test_split_and_predictor_kind_follow_section_3 |
| R4 | expb/registry.py | config class check `elif not isinstance(...)` -> `elif False` | killed | test_expb_registry::test_fit_arm_refuses_a_config_of_the_wrong_class |
| R5 | expb/registry.py | `ARM_POLICY_B` not identity (ppo dropped) | killed | test_expb_registry::test_b_arms_and_policy_map_are_the_six_named_arms |
| R6 | expb/registry.py | unknown-arm check `if arm not in ARM_SPECS` -> `if False` | killed | test_expb_registry::test_fit_arm_refuses_an_unknown_arm |

## Survivors and how they were closed

- **C4 as first written (DOS `width` -> 55) survived.** The mutant is equivalent under the
  current fit: `dos.init_net` always builds `DOSNet()` (width 55) and ignores `cfg.width`
  (dos.py:82), so the packed width is 55 whichever config was passed. This is a defect in
  M16, not in the codec, and is reported rather than fixed here (dos.py is out of scope).
  C4 was replaced by the input-dimension mutant above, which is not equivalent.
- **C5 first run selected no test** (`-k libraries` matches no test name; exit code 5).
  Re-run with `-k library_versions`: killed.

## Limits of the check

- Single-line mutants, each checked against the named tests. A kill shows a test detects
  the mutant, not that it is the only detectable fault.
- P8 is killed by an exception inside the update, not by an assertion; it is recorded as such.
- Not mutated: the entropy term (coefficient 0 in every spec config), the GRU cache
  bookkeeping in `PolicyPredictor._reusable` (covered only by the cached/recomputed and
  out-of-order tests), the pass-level checkpoint bookkeeping, and the budget counters.

## M21 (sweep integration), single-line mutants of the new sweep logic

Procedure as above: one single-line mutation of `expa/sweep.py` at a time, the targeted
node ids run with `pytest -x`, original bytes restored and verified by sha256 afterwards
(final check: restored, `if False:` absent). Driver: scratchpad script, not committed.
Node ids are in `tests/test_expb_sweep.py` unless stated.

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| M1 | expa/sweep.py | `stage_paths`: `if not sspec.independent:` -> `if False:` (independence world on every stage) | killed; the independence test alone survived, the identity test for non-independent stages kills it | test_world_is_identity_for_a_non_independent_stage |
| M2 | expa/sweep.py | `stage_paths`: `d=permute_columns(paths.d, key)` -> `d=paths.d` (increments not moved together) | killed | test_independent_world_permutation_is_deterministic_and_keeps_marginals |
| M3 | expa/sweep.py | `stage_paths`: purpose dropped from the seed key (`train` for every purpose) | killed | test_independent_world_permutation_is_deterministic_and_keeps_marginals |
| M4 | expa/sweep.py | `arm_config`: `len(entries) != 1` -> `> 1` (missing arm configuration not refused) | killed | test_arm_config_lookup_builds_the_configuration_class |
| M5 | expa/sweep.py | `_fit_b_payload`: `cfg.passes != sspec.pass_curve` -> `False` (pass count not checked) | killed | test_tuning_stage_refuses_a_configuration_with_another_pass_count |
| M6 | expa/sweep.py | `_curve_results`: policy `@p{p}` -> `@p{p - 1}` | killed | test_curve_results_name_one_policy_per_pass |
| M7 | expa/sweep.py | `check_stream_for_stage`: `eval_set != spec.test_stream` -> `False` (stream mismatch accepted) | killed | test_dev_stream_refusal_names_the_declared_stream |
| M8 | expa/sweep.py | `Ctx.models_dir`: reuse branch -> `False` (reads the stage's own, empty, directory) | killed | test_reuse_stage_reads_a_main_models_directory |
| M9 | expa/sweep.py | `merge_stage`: `[columns]` -> `[ROW_COLUMNS]` (test_b columns dropped) | killed | test_test_b_rows_carry_family_model_and_library_columns |
| M10 | expa/sweep.py | `run_fit`: reuse refusal -> `False` | killed (first run NOT APPLIED: driver matched LF on CRLF text; rerun after fix) | test_reuse_stage_has_no_fit |
| M11 | expa/sweep.py | `run_fit_job`: `libraries` only for test_b stages -> `False` | killed | test_b_smoke_fit_eval_merge_gives_exactly_the_ten_policies |
| M12 | expa/sweep.py | `stage_eligibility`: `SMOKE_LIKE` -> `("smoke", "pilot")` | killed | test_b_dev_stages_use_the_in_memory_con3_eligibility |

Limits: single-line mutants of the M21 logic only. Not mutated and not tested: the
ladder-label filter of reuse evaluation, `planned_keys`, and the CLI `manifest` branch.
The test_b evaluation path of `run_eval_job` (`_test_b_columns`, the family hash from the
gate token) has no test: a test_b payload cannot be produced without a pinned FAMILY line,
so it is left to the family run and to the M23 row checks.
Also fixed in M21 (not a mutant): `dos.init_net` now honours `cfg.width` (the defect
reported in the earlier MUTATION_B entry C4 is closed by tests in test_expb_dos.py).

# MUTATION_B: mutation checks for M23 (expb/analysis_b.py)

Procedure as above: one single-line mutation at a time, `tests/test_expb_analysis_b.py`
run with `pytest -x`, the original bytes restored in a `finally` block and verified by
sha256 (restored byte-identical after the full run). "killed" = at least one test fails.

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| A1 | expb/analysis_b.py | `adjusted = A.holm(raw)` -> `adjusted = list(raw)` (no Holm) | killed | test_family_table_known_answers_and_branches |
| A2 | expb/analysis_b.py | branch on `p_adj` -> branch on raw two-sided p | killed | test_family_table_applies_holm_over_the_six_arms |
| A3 | expb/analysis_b.py | `WIN_MIN_COUNT = A.WIN_MIN_COUNT` -> `12` | killed | test_win_count_thirteen_is_win_and_twelve_is_tie_weak |
| A4 | expb/analysis_b.py | WIN lower bound `lo95 > 0` -> `lo90 > 0` (CI90 for CI95) | killed | test_win_needs_ci95_lower_bound_positive_not_ci90 |
| A5 | expb/analysis_b.py | family hash check `not (...).all()` -> `if False` (mismatched family_sha256 accepted) | killed | test_load_refuses_family_hash_mismatch |
| A6 | expb/analysis_b.py | `for arm in family:` -> `for arm in list(family)[1:]` (first arm skipped) | killed | test_family_table_known_answers_and_branches |
| A7 | expb/analysis_b.py | Holm-level CI quantile `2.0 * size` -> `1.0 * size` (CI90 level) | killed | test_holm_ci_size_one_is_ci95 |
| A8 | expb/analysis_b.py | family contrast sides swapped (arm as A, FULL as B) | killed | test_family_table_known_answers_and_branches |
| A9 | expb/analysis_b.py | WIN boundary `mean >= delta - TOL` -> `delta + TOL` | killed | test_delta_boundary_is_included_in_win_and_just_below_is_not |
| A10 | expb/analysis_b.py | coverage check call made a no-op (19 instances accepted) | killed | test_family_table_refuses_instance_coverage_below_twenty |
| A11 | expb/analysis_b.py | day-count check `n_eval == TEST_B_N` -> `if False` | killed | test_load_refuses_wrong_day_count |
| A12 | expb/analysis_b.py | one eval commit `!= 1` -> `< 1` (mixed commits accepted) | killed | test_load_refuses_mixed_eval_commits |
| A13 | expb/analysis_b.py | `if absent:` (listed arm without rows) -> `if False` | killed | test_family_arm_with_no_rows_is_refused |
| A14 | expb/analysis_b.py | FROZEN fit-commit check -> `if False` | killed | test_load_refuses_fit_commit_not_frozen_for_the_stage |
| A15 | expb/analysis_b.py | TIE_STRONG on CI95 instead of CI90 | killed | test_tie_strong_is_decided_by_ci90_not_ci95 |
| A16 | expb/analysis_b.py | LOSE sign flipped (`hi95 < 0` -> `hi95 > 0`) | killed | test_go_no_go_holm_reproduces_a_vectors_with_raw_p |

Survivors found on the first run and closed by tests: A4 (no test distinguished CI90 from
CI95 in the WIN bound; added test_win_needs_ci95_lower_bound_positive_not_ci90), A13 (the
refusal was tested without its message; the message is now asserted, since `main_stage`
would also refuse, only less clearly), A8 (the first anchor matched two lines and was not
applied; the anchor is now unique and the mutant applied and killed). A15 was added as a
new test before its run.

Limits: each row is one single-line change checked by the named tests only. The CLI
(`main`) is exercised only for argument refusal and through the pinned-log helpers of
gatecheck (not run end to end on a git repository); M23's own row checks are covered
by the constructed-table tests above.

## M22 (expb/fidelity.py): dev gates, family derivation

Procedure as above, with `tests/test_expb_fidelity.py` (36 tests). Mutants are applied by a
script that replaces one unique line fragment, runs the file with `-x`, and restores the
original bytes (restoration checked by sha256; `True` in the run). The share comparisons
were moved into `_meets(share, threshold)` so that the boundary is tested directly.

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| F1 | expb/fidelity.py | B-THR band `<=` -> `<` (boundary at 3 SE) | killed | test_expb_fidelity::test_within_band_boundary_is_inclusive_at_three_se |
| F2 | expb/fidelity.py | `T4_SE = 3.0` -> `2.0` (wrong SE multiplier) | killed | test_expb_fidelity::test_within_band_boundary_is_inclusive_at_three_se |
| F3 | expb/fidelity.py | `_meets`: `>=` -> `>` (share threshold strict) | killed | test_expb_fidelity::test_share_helper_is_inclusive_at_the_threshold |
| F4 | expb/fidelity.py | THR DP breach `g1 > B` -> `g1 >= B` | killed | test_expb_fidelity::test_thr_dp_breach_is_strict_at_the_load_limit |
| F5 | expb/fidelity.py | per-day SE with ddof 0 | killed | test_expb_fidelity::test_per_day_standard_error_uses_the_sample_variance |
| F6 | expb/fidelity.py | P1 bound `2 SE` -> `1 SE` | killed | test_expb_fidelity::test_p1_boundary_is_two_standard_errors_and_the_competitor_is_the_minimum |
| F7 | expb/fidelity.py | P1 competitor always H1 (minimum dropped) | killed | test_expb_fidelity::test_p1_boundary_is_two_standard_errors_and_the_competitor_is_the_minimum |
| F8 | expb/fidelity.py | `derive_family` skips the last arm (ppo) | killed | test_expb_fidelity::test_family_includes_every_arm_that_passes |
| F9 | expb/fidelity.py | DQN pass range `1..P_max` -> `1..P_max-1` | killed | test_expb_fidelity::test_dqn_gate_accepts_three_passes |
| F10 | expb/fidelity.py | B-RLSM threshold 0.90 -> 1.0 | killed | test_expb_fidelity::test_rlsm_gate_pass_and_share_boundary_at_ninety_percent |
| F11 | expb/fidelity.py | RLSM 100-row minimum `<` -> `<=` | killed | test_expb_fidelity::test_rlsm_gate_pass_and_share_boundary_at_ninety_percent |
| F12 | expb/fidelity.py | RLSM residual 1e-8 -> 1e-7 | killed | test_expb_fidelity::test_rlsm_stages_below_100_val_rows_are_not_counted_and_residual_is_gated |
| F13 | expb/fidelity.py | B-DOS tolerance 1e-6 dropped | killed | test_expb_fidelity::test_dos_gate_uses_min_of_always_and_never_with_tolerance |
| F14 | expb/fidelity.py | B-DOS always-stop not divided by E0 | killed | test_expb_fidelity::test_dos_gate_uses_min_of_always_and_never_with_tolerance |
| F15 | expb/fidelity.py | B-DOS unrecorded objectives ignored | killed | test_expb_fidelity::test_dos_unrecorded_objective_fails_even_when_the_share_passes |
| F16 | expb/fidelity.py | B0 oracle comparison `arm < oracle` -> `arm > oracle` | killed | test_expb_fidelity::test_b0_fails_when_an_arm_row_beats_the_oracle |

Survivors and closures: the first run had 7 survivors (F3, F4, F5, F6, F7, F8, F9, F15) and
one unmatched pattern (F16). Each closed by a test added for it (boundary of the share helper,
the THR breach at exactly B, the sample SE, the P1 2-SE boundary with `none` as the minimum
competitor, PPO inside the family, three DQN passes, one unrecorded DOS objective with share
0.95). Re-run: all 16 killed.

## M16 (expb/dos.py): never-stop objective recorded in the stage diag

Added `_never_val` and a `never_val` key on all three stage diag entries (trained, constant,
carry). Fit output is unchanged: the fitted state_dicts and surrogate values on all rows
hash to the reference digest recorded before the change (`test_fitted_models_are_bitwise_unchanged_on_a_fixed_seed`).
Procedure as above, `tests/test_expb_dos.py` with `-x`; the mutated file was restored and its
sha256 compared with the pre-mutation hash (identical: True).

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| N1 | expb/dos.py | `if y_val.size == 0:` -> `if False:` (empty validation gives NaN, not None) | killed | test_expb_dos::test_never_val_is_none_without_validation_rows |
| N2 | expb/dos.py | `float(np.mean(y_val) / E0)` -> `float(np.mean(y_val))` (no E0 scale) | killed | test_expb_dos::test_never_val_is_the_loss_of_the_constant_continue_decision_on_val_rows |
| N3 | expb/dos.py | `_never_val(y_val, E0)` -> `_never_val(y_fit, E0)` (fitting rows used) | killed | test_expb_dos::test_never_val_is_the_loss_of_the_constant_continue_decision_on_val_rows |
| N4 | expb/dos.py | constant-branch entry drops `never_val=never` | killed | test_expb_dos::test_never_val_is_recorded_for_every_stage_including_constant_and_carry_branches |

Result: 4/4 killed; the file is restored byte-for-byte.

## M16b (expb/dos.py): held-stage objective (constant and carry stages)

Constant and carry stages now record `val_loss_per_ckpt=[(0, loss)]` and `chosen_ckpt=0`
(the loss of the decision deployed at that stage on its validation rows, E0 scale; `None`
and `[]` without validation rows). The carried network is evaluated on a deep copy, so the
carried predictor is not modified. The fitted-model digest (`REFERENCE_DIGEST`) is unchanged.
Procedure as above, `tests/test_expb_dos.py` with `-x`; restoration checked by sha256
(identical: True).

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| R1 | expb/dos.py | constant held decision `float(pi)` -> `1.0 - float(pi)` (stop/continue swapped) | killed | test_expb_dos::test_constant_stage_objective_is_the_loss_of_the_deployed_constant_decision |
| R2 | expb/dos.py | carry evaluated on `warm.net` (no copy; mutates eval mode) | killed | test_expb_dos::test_carry_objective_leaves_the_carried_network_untouched |
| R3 | expb/dos.py | no-validation return `None` -> recorded `0.0` at step 0 | killed | test_expb_dos::test_held_stage_without_validation_rows_has_no_objective |
| R4 | expb/dos.py | `chosen_ckpt=0` -> `chosen_ckpt=1` (objective not found by the gate) | killed | test_expb_dos::test_constant_stage_objective_is_the_loss_of_the_deployed_constant_decision |
| R5 | expb/dos.py | loss uses `F * Hk` (unscaled stop cost) | killed | test_expb_dos::test_constant_stage_objective_is_the_loss_of_the_deployed_constant_decision |

Result: 5/5 killed; the file is restored byte-for-byte. Also fixed in the test file: the carry
test of M16 used stage k=3 on a 3-route (outside [1, m-1]); now k=2.


## CODE_REVIEW_B follow-up (expb/fidelity.py, analysis_b.py, codecs.py, dos.py)

Procedure as above: single-line mutants, tests of the touched module with `-x`, file restored
byte-for-byte after the run (sha256 compared with the pre-mutation hash: identical, True for
every file). Test files: `test_expb_fidelity.py`, `test_expb_analysis_b.py`,
`test_expb_codecs.py`, `test_expb_dos.py`.

### M6 (expb/codecs.py): one PolicyPredictor per network after unpack

| id | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|
| C1 | `if key not in ppo:` -> `if True:` (fresh predictor per stage again) | killed | test_expb_codecs::test_unpacked_ppo_stages_share_one_predictor |
| C2 | cache key gains `len(ppo)` (never a hit) | killed | test_expb_codecs::test_unpacked_ppo_stages_share_one_predictor |

2/2 killed. Costs after a round trip equal the old one-predictor-per-stage path bitwise
(`test_unpacked_ppo_evaluation_uses_the_gru_cache_and_keeps_decisions`).

### M2 and M5 and parameters (expb/fidelity.py, expb/dos.py)

| id | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|
| F1 | `if reported:` -> `if False:` (cut arm with a report accepted) | killed | test_cut_arm_with_a_report_is_refused_and_unknown_cut_arm_too |
| F2 | cut entry `arm:cut` -> `arm:b_arm` | killed | test_cut_arm_needs_no_report_and_appears_as_dropped_cut |
| F3 | family rule inverted for reported arms | killed | test_family_lists_follow_gate_verdicts |
| F4 | cut arm left in `descriptive` | killed | test_cut_arm_needs_no_report_and_appears_as_dropped_cut |
| F5 | missing-report check only for cut arms | killed | test_family_derivation_requires_every_arm |
| F6 | `if unknown:` -> `if False:` | killed | test_cut_arm_with_a_report_is_refused_and_unknown_cut_arm_too |
| F7 | CLI loops over every arm (reads the cut arm's file) | killed | test_cli_skips_cut_arm_files |
| F8 | `T4_SE = 4.0` -> `3.0` | killed | test_thr_gate_false_drop_rate_of_a_correct_thr_is_at_most_one_percent |
| F9 | DP replicate mean -> `max` | killed | test_thr_route_statistic_uses_the_rep_mean_and_its_own_per_day_se |
| F10 | replicate-mean day series -> first replicate | killed | test_thr_route_statistic_uses_the_rep_mean_and_its_own_per_day_se |
| F11 | route total counted per job | killed | test_thr_gate_passes_on_constructed_independent_world |
| F12 | routes keyed per (route, rep) | killed (after adding the pooling test) | test_thr_gate_pools_the_replicates_of_a_route_before_testing |
| F13 | gate slack `ctx.dos_tol` -> constant `DOS_TOL` | killed | test_dos_tolerance_is_a_parameter_of_the_gate_and_defaults_to_the_spec_value |
| F14 | THR DP values dropped from the derivation block | killed | test_thr_dp_values_and_budget_labels_enter_the_derivation_block |
| F15 | budget labels dropped from the derivation block | killed | test_thr_dp_values_and_budget_labels_enter_the_derivation_block |
| F16 | cut arm block `passed: False` -> `True` | killed | test_cut_family_line_round_trips_through_gatecheck |
| F17 | DP block keeps rep 0 only | killed | test_thr_dp_values_and_budget_labels_enter_the_derivation_block |
| F18 | B-DQN ignores the chosen pass count | killed | test_dqn_gate_checks_the_chosen_pass_count_when_given |
| F19 | CLI does not pass `--p-main` to the DQN gate | killed | test_cli_passes_the_main_fit_pass_count_to_the_dqn_gate_only |
| F20 | B-PPO mask check returns True | killed | test_ppo_gate_refuses_a_head_that_can_hand_off_at_the_last_stage |
| F21 | B-PPO gate ignores the mask check | killed | test_ppo_gate_passes_on_a_real_fit_and_uses_the_mask_check |
| D1 | `stage_steps` reads the module constant, not `cfg.epochs_factor` | killed | test_epochs_factor_is_a_config_parameter_with_the_documented_default |
| D2 | `epochs_factor` default 20 -> 10 | killed | test_stage_steps_follow_the_formula_and_its_floor_and_cap |

23/23 killed.

### M4, M2, M7 (expb/analysis_b.py)

| id | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|
| A1 | `G.check_analysis_clean(head)` -> `pass` | killed | test_main_calls_check_analysis_clean_with_head_before_any_table |
| A2 | `cut_arms` returns `[]` | killed | test_cut_arm_without_rows_is_accepted_and_reported_as_cut |
| A3 | cut arm with rows accepted | killed | test_cut_arm_with_rows_is_refused |
| A4 | cut arm required to have rows | killed | test_cut_arm_without_rows_is_accepted_and_reported_as_cut |
| A5 | D_a contrast swapped (baton_eq vs arm) | killed | test_da_table_is_each_arm_against_baton_eq_with_a_unadjusted_branch |
| A6 | identical share inverted | killed | test_alpz_vs_thr_reports_paired_difference_and_identical_share |
| A7 | ALP-Z vs THR `or` -> `and` | killed | test_alpz_vs_thr_is_none_when_either_arm_is_absent |
| A8 | strata not filtered | killed | test_strata_table_splits_instance_deltas_by_stratum |
| A9 | replicate SD `ddof=1` -> `0` | killed | test_replicate_sd_is_the_between_replicate_sd_in_percent_of_the_route_mean |
| A10 | budget-limited label never applied | killed | test_budget_limited_label_is_in_every_claim_sentence_of_the_arm |
| A11 | THR excess over cost instead of DP | killed | test_thr_over_dp_is_the_pooled_percentage_above_the_model_value |
| A12 | partial DP table accepted | killed | test_thr_over_dp_refuses_a_partial_dp_table |
| A13 | outside reason label inverted | killed | test_build_report_covers_every_listed_arm_and_keeps_dropped_ones |
| A14 | labels from the derivation block ignored in `main` | killed (after adding the end-to-end test) | test_main_end_to_end_reads_labels_and_dp_values_from_the_derivation_block |
| A15 | DP values from the derivation block ignored in `main` | killed | same |
| A16 | cut entries listed as dropped arms | killed | test_cut_arm_without_rows_is_accepted_and_reported_as_cut |
| A17 | D_a and strata only over family arms | killed (after adding the test) | test_da_and_strata_cover_descriptive_and_dropped_arms_but_not_cut_ones |
| A18 | `da.csv` not written | killed | test_summary_labels_budget_limited_arms_and_lists_the_reports |
| A19 | `--n-reps` ignored | killed | test_cli_cut_four_reps_argument_reaches_the_coverage_check |
| A20 | `--n-routes` ignored | killed | test_main_end_to_end_reads_labels_and_dp_values_from_the_derivation_block |

20/20 killed.

## Review fixes C1, C2, M2 (cuts, dos:cut), M3, M4 (gate side), M8 (sweep.py, gatecheck.py, stages.py, choose.py)

Single-line mutants applied one at a time by script, each run against the tests named in the
last column's file (`-x`), file restored byte-for-byte and its sha256 compared afterwards
(identical: True for sweep.py, gatecheck.py, choose.py, stages.py). Five survivors of the
first pass (G7, G23, G24, C3, C13) got new tests and were re-run: all killed.

| id | module | mutation (original -> mutant) | result | first failing test |
|---|---|---|---|---|
| G1 | sweep.py | B default results dir -> `DEFAULT_RESULTS` (A's) | killed | test_expb_results_dir::test_default_results_dir_of_b_stages_is_where_the_gate_reads |
| G2 | sweep.py | `MAX_GPU_JOBS = 1` -> 2 | killed | ::test_gpu_semantics_is_one_process |
| G3 | sweep.py | `effective_jobs` returns `jobs` | killed | ::test_cuda_stages_are_capped_and_cpu_stages_are_not |
| G4 | sweep.py | CUDA resolution `== "cuda"` -> `if False` | killed | ::test_cuda_stages_are_capped_and_cpu_stages_are_not |
| G5 | sweep.py | `require_enabled` never refuses | killed | ::test_reuse_stages_are_marked_disabled_and_refuse_every_entry_point |
| G6 | sweep.py | `run_eval` dev path skips the disabled check | killed | ::test_reuse_stages_are_marked_disabled_and_refuse_every_entry_point |
| G7 | sweep.py | `check_test_gate` skips the disabled check | killed (after new test) | ::test_stage_level_gate_refuses_a_disabled_stage_without_the_commit_checks |
| G8 | sweep.py | `run_fit` drops `jobs = effective_jobs(...)` | killed | ::test_run_fit_hands_the_capped_job_count_to_the_pool |
| G9 | sweep.py | `subset6` takes 2 per stratum | killed | test_expb_cuts::test_cut_two_tunes_on_the_six_routes_3_medium_3_long |
| G10 | sweep.py | payload `cuts` = `{}` | killed | test_expb_results_dir::test_b_smoke_fit_manifest_and_gate_model_checks_agree_in_a_tmp_repo |
| G11 | sweep.py | CLI default results dir -> A's | killed | ::test_cli_without_results_dir_resolves_per_stage |
| G12 | sweep.py | `manifest` ignores `--results-dir` | killed | ::test_cli_without_results_dir_resolves_per_stage |
| G13 | gatecheck.py | `write_manifest` ignores `results_dir` | killed | ::test_write_manifest_honours_an_explicit_results_dir |
| G14 | gatecheck.py | `verify_stage` does not refuse `_reuse` | killed | ::test_reuse_stages_are_marked_disabled_and_refuse_every_entry_point |
| G15 | gatecheck.py | analysis check accepts a non-HEAD commit | killed | test_expb_gate::test_analysis_clean_refuses_a_working_tree_edit_or_a_non_head_commit |
| G16 | gatecheck.py | working bytes vs blob comparison off | killed | ::test_analysis_clean_compares_working_bytes_with_the_blob_even_without_status |
| G17 | gatecheck.py | "added once" count check off | killed | ::test_analysis_clean_refuses_a_patch_line_added_removed_and_added_again |
| G18 | gatecheck.py | "below the pinning commit" check off | killed | ::test_analysis_clean_refuses_a_patch_line_from_a_commit_that_is_not_below_the_pin |
| G19 | gatecheck.py | numstat file-count check off | killed | ::test_analysis_clean_refuses_a_patch_line_that_shares_its_commit_with_code |
| G20 | gatecheck.py | numstat `1 0` check off | killed | ::test_analysis_clean_refuses_two_patch_lines_in_one_commit |
| G21 | gatecheck.py | last patch hash not used | killed | ::test_analysis_clean_accepts_one_patch_line_in_its_own_commit |
| G22 | gatecheck.py | final analysis hash comparison off | killed | ::test_analysis_clean_refuses_a_committed_edit_without_a_patch_line |
| G23 | gatecheck.py | first patch hash used instead of last | killed (after new test) | ::test_analysis_clean_accepts_a_second_patch_and_uses_the_last_hash |
| G24 | gatecheck.py | only the last patch line checked | killed (after new test) | ::test_analysis_clean_checks_every_patch_line_not_only_the_last |
| G25 | gatecheck.py | FAMILY regex `dropped` class loses `:` | killed | ::test_dos_cut_spelling_is_accepted_by_the_family_regex_and_the_gate |
| S1 | stages.py | cut (iv) passes 1 -> 2 | killed | test_expb_cuts::test_cut_four_fixes_one_pass_for_dqn_and_ppo_only |
| S2 | stages.py | cut (iii) reps ignored | killed | ::test_cut_three_uses_two_replicates_for_every_main_stage_but_not_the_rest |
| S3 | stages.py | cut (i) drops ppo instead of dos | killed | ::test_cut_one_drops_dos_everywhere_and_labels_it |
| S4 | stages.py | cut (ii) tuning routes ignored | killed | ::test_cut_two_tunes_on_the_six_routes_3_medium_3_long |
| S5 | stages.py | label `dos:cut` -> `dos: cut` | killed | ::test_cut_one_drops_dos_everywhere_and_labels_it |
| S6 | stages.py | cut (iv) skips PPO | killed | ::test_cut_four_fixes_one_pass_for_dqn_and_ppo_only |
| S7 | stages.py | `chosen_arm_cfg` ignores the record | killed | test_expb_choose::test_chosen_arm_cfg_is_generated_from_the_record_not_typed |
| S8 | stages.py | `cut_record` reps constant | killed | test_expb_cuts::test_cut_record_is_json_able_and_lists_active_cuts |
| C1 | choose.py | tie window 0.10 -> 0.20 | killed | test_expb_choose::test_tie_window_is_a_tenth_of_a_point |
| C2 | choose.py | no tie window | killed | ::test_tie_goes_to_fewer_passes_then_fewer_seconds_then_lower_index |
| C3 | choose.py | seconds before passes | killed (after new test) | ::test_fewer_passes_beat_fewer_seconds_inside_the_tie_window |
| C4 | choose.py | index before seconds | killed | ::test_tie_goes_to_fewer_passes_then_fewer_seconds_then_lower_index |
| C5 | choose.py | higher index wins | killed | ::test_tie_goes_to_fewer_passes_then_fewer_seconds_then_lower_index |
| C6 | choose.py | score not in percent | killed | ::test_score_is_mean_percent_gap_to_baton_eq |
| C7 | choose.py | non-finite costs admitted | killed | ::test_non_finite_cost_and_missing_routes_exclude_a_config |
| C8 | choose.py | missing routes admitted | killed | ::test_non_finite_cost_and_missing_routes_exclude_a_config |
| C9 | choose.py | highest score wins | killed | ::test_lowest_score_wins_outside_the_tie_window |
| C10 | choose.py | record pairs not compared with the grid | killed | ::test_record_with_hand_edited_pairs_or_choice_is_refused |
| C11 | choose.py | record choice not re-derived | killed | ::test_record_with_hand_edited_pairs_or_choice_is_refused |
| C12 | choose.py | source CSV hash not compared | killed | ::test_source_hash_mismatch_is_refused |
| C13 | choose.py | stored seconds set to 0 | killed (after new test) | ::test_record_keeps_the_seconds_that_decided_a_tie |
| C14 | choose.py | curve arm needs only 2 passes | killed | ::test_curve_arm_requires_all_three_passes |

Result: 47/47 killed (5 after adding a test). Not mutated: the redundant `check_clean` call in
`check_analysis_clean` is covered by G15 and the monkeypatched-status test of G16.

## Revision 3.4 changes (2026-11-19)

Single-line mutants, each restored byte for byte after its run (checked by the final test run).

| id | file | mutant | result | test |
|---|---|---|---|---|
| F14 | expb/fidelity.py | B-DOS tolerance absolute (drops `abs(always - never)`) | killed | test_expb_fidelity::test_dos_gate_uses_min_of_always_and_never_with_relative_tolerance |
| F15 | expb/fidelity.py | B-DOS gap signed (no `abs`) | killed | ::test_dos_gate_uses_min_of_always_and_never_with_relative_tolerance |
| F16 | expb/fidelity.py | B-DOS `min` -> `max` | killed | ::test_dos_gate_uses_min_of_always_and_never_with_relative_tolerance |
| F17 | expb/fidelity.py | `DOS_TOL` 0.02 -> 0.2 | killed | ::test_dos_tolerance_scales_with_the_gap_between_always_and_never |
| F18 | expb/fidelity.py | `T4_SE` 4 -> 3 | killed | ::test_thr_gate_false_drop_rate_of_a_correct_thr_is_at_most_one_percent |
| F19 | expb/fidelity.py | `THR_FINE_GRID` 4001 -> 2001 | killed | ::test_thr_grid_refinement_is_bounded_by_the_base_grid_step |
| F20 | expb/fidelity.py | refinement over every job, not only G2 | killed | ::test_refinement_is_reported_for_g2_and_never_changes_the_verdict |
| D1 | expb/dos.py | `s_min` 400 -> 50 | killed | test_expb_dos::test_stage_steps_follow_the_formula_and_its_floor_and_cap |
| A1 | expb/analysis_b.py | `--n-reps` vs payload cut check off | killed | test_expb_analysis_b::test_n_reps_option_must_match_the_payload_cut |
| A2 | expb/analysis_b.py | payload cut ignored (default always from ACTIVE_CUTS) | killed | ::test_n_reps_follows_the_cut_recorded_in_the_payloads |
| A3 | expb/analysis_b.py | differing payload cuts accepted | killed | ::test_n_reps_refuses_payloads_recording_different_cuts |
| A4 | expb/analysis_b.py | `main` ignores the resolver | killed | ::test_main_takes_n_reps_from_the_payload_cut |
| W1 | expa/sweep.py | `peak_rss_mb` missing in the main B budget | killed | test_expb_sweep::test_b_fit_payload_budget_records_peak_rss |
| W2 | expa/sweep.py | `peak_rss_mb` missing in the pass-curve budget | killed | ::test_b_fit_payload_budget_records_peak_rss |

Result: 14/14 killed.
