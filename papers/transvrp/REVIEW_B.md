# Review of EXPERIMENT_B.md (FULL against published 2023-2026 algorithms)

Reviewer: independent adversarial pass, 2026-11-18. Read: EXPERIMENT_B.md (rev 1), EXPERIMENT_A.md
conventions, REVIEW_A.md (format), LIT_2025_2026.md, the tail of RESEARCH_LOG.md, and the code in
`code/transvrp/expa/` (paths.py, sweep.py, backward.py, simulate.py, features.py, model.py, train.py,
boost.py, analysis.py, eligibility.py) plus `results/expA/eligibility_dev.csv`. No experiment was run;
two small read-only Python calls (eligibility table, SeedSequence state). Nothing was edited except this file.

Sources I tried for the two methods the drafter could not read:
- Zhu, Adulyasak, Rousseau (Networks 87(3), doi 10.1002/net.70024; GERAD G-2024-70). The GERAD PDF
  (`gerad.ca/papers/G-2024-70.pdf`) returns 404; the GERAD catalogue page
  (`gerad.ca/en/papers/G-2024-70`) gives only the abstract ("affine-function lower bounds,
  decomposition-based value function approximations, constraint sampling", single vehicle,
  re-optimisation); PolyPublie `publications.polymtl.ca/60329/` rendered empty and the guessed PDF
  link hung. **I could not read the method section either.** So the ALP reconstruction cannot be
  checked against the source by me; finding 5 says what follows.
- Kadyrov et al. (ORP 15:100351): ScienceDirect and DOAJ return 403; search snippets give only the
  abstract (GNN encoder, PPO, action masking). **PPO settings remain unverified.**
- Dastpak et al. 2023 (C&OR) and Venkateshan-Mathur 2026: not retrieved; the DQN settings in 3.5 come
  from the 2026 arXiv version via LIT_2025_2026.md (the spec already flags this).
Consequence: for ALP and PPO the fidelity claim rests on generic methods plus gates, exactly as the
spec says in section 12. I agree with that wording and require it to be kept (finding 5, 6).

## Verdict (short)

ACCEPT WITH REQUIRED CHANGES. The statistical frame is a faithful extension of A, the arm list is
sensible, the THR decision (separate arm, not dp relabelled) is correct, and the THR DP indexing
(E[k], H[k-1], V_m = 0, breach only by the new W) checks out against `simulate_policy`. But the
test_b gate as specified cannot be satisfied and is bypassable (finding 1), the DQN and PPO
setups are likely under-trained by construction and the compute plan then forces that (2, 3), and
several fidelity gates are arbitrary or near-tautological (8). None of this needs a redesign; all
fixes are specification changes.

## Findings

### 1. [critical] The test_b gate is self-contradictory and is not the barrier the spec claims

Location: section 5.2 (`verify_test_gate`), 5.3, M12. Code: paths.py:137-160, 163-167, 313, 331;
sweep.py:275-286, 440-447.

(a) Contradiction. The gate requires `HEAD == frozen` (as A) AND a `TEST_B FAMILY CLOSED` line in
RESEARCH_LOG.md that was "written and committed" after the fit commit (5.3 step 4). A commit cannot
contain a log line written after it, so at HEAD == frozen the committed log lacks the family line.
In A this worked only because `tree_is_clean` runs `git status -- .` inside `code/` (paths.py:102-106)
and RESEARCH_LOG.md lives outside it, so the log is read from the *working tree*, uncommitted.
The "committed" wording is therefore false, and the gate's evidence is a free-text file nobody
tracks.
(b) Bypass 1: the log is an editable working-tree file outside the clean-tree check. Writing a family
line and hashes by hand satisfies every B check. The spec's section 1.5 says "the gate code
enforces the order"; it enforces only that three strings appear in a mutable file.
(c) Bypass 2 (cross-stream): A's check is a *substring* test (`frozen not in read_text()`,
paths.py:152). A `FROZEN COMMIT (test_b): <h>` line contains the hash, so calling
`route_pool(inst, "test", frozen=<B hash>)` directly passes A's gate and reads A's test stream. The
"B cannot read A's test" guarantee (5.2, last bullet of the StageSpec item) exists only at the CLI
layer. Same in reverse is blocked only if the new anchored regex is used for test_b.
(d) Bypass 3: `check_test_gate` returns immediately for any eval_set other than "test"
(sweep.py:279-280), and `run_eval_job` applies the fit-commit and clean-tree checks only under
literal `eval_set == "test"` (sweep.py:440-445). M12 lists the changes to `paths.py` and the
StageSpec but not these literals. If they are not edited, `--eval test_b` skips the per-model checks.
(e) Nothing ties the family line to the models actually being evaluated: any stage with a
`FROZEN COMMIT (test_b)` line can be evaluated, whether or not its arm is in `family=[...]` or was
in `dropped=[...]`, and nothing ties the line to the dev gate report.
(f) Several frozen commits (repaired arms) mean stages are evaluated with different `expa/`
code (simulate.py, features.py, backward.py can change between commits). Cross-arm contrasts would
then use different simulators.

Required change:
- Define the gate as: `frozen` is an ancestor of HEAD (or equal), `git diff frozen..HEAD -- code/`
  is empty, and the log is read **from the committed blob** at HEAD (`git show HEAD:papers/transvrp/RESEARCH_LOG.md`)
  with the log file itself clean in `git status`. Then "committed" is true and the order of lines is
  verifiable from `git log -p` timestamps.
- Make A's gate anchored too (`^FROZEN COMMIT: <h>$`), unchanged for A's existing lines and tests.
- Add `test_b` to every literal: `EVAL_SETS`, `check_test_gate`, `run_eval_job` (440-445), `route_pool`
  (313, 331), `_stream` (227), `merge_stage`. Add mutation tests that remove each literal edit.
- The gate reads `family=[...]`, `dropped=[...]` and refuses a stage whose arm is not in
  family or references. The `TEST_B FAMILY CLOSED` line embeds the sha256 of
  `results/expB/fidelity_dev.md`; `analysis_b` refuses if the file differs.
- Pre-register a manifest: sha256 of `simulate.py`, `features.py`, `backward.py`, `train.py`, `model.py`
  and the `analysis` helpers must be identical at every frozen commit used (checked at the gate).
- Add to M12(a): compare `SeedSequence(seed).generate_state(4)` of test_b against dev, test,
  test_std and train (all reps, chunks 0-9) for all 20 instances. (SeedSequence pads entropy with
  zeros, so `[base, 3]` equals `[base, 3, 0, 0]`; comparing the first 100 rows is weaker than comparing
  states. I checked: `[b, 20261118]` is a genuinely new state and `20261118` appears nowhere else
  in the repo; A's bootstrap seed is 20261115.)

### 2. [critical] DQN and PPO are set up so that they are likely under-trained; "honest tuning budget" is not met

Location: 3.5, 3.6, 6, 9.2 cut 1, 12 ("Adaptations ...").

- Exploration starves late stages. The behaviour policy is epsilon-greedy with a random action drawn
  as continue/handoff 1/2 each; handoff ends the episode and its transition is not stored (3.5). During
  the first 50% of pass 1 (epsilon from 1.0 downward) the probability of reaching stage k is about
  0.5^k, so stages beyond about 6 receive almost no continue transitions while the network is still
  random. The late stages are exactly where decisions are close. Since Q(s,handoff) = H is known and
  handoff transitions are discarded, random handoff gives no data at all. Q-learning is off-policy:
  the behaviour policy should be "continue" with epsilon exploration only in the greedy sense (or
  forced continue until a stage is visited), which loses nothing.
- One pass is the minimum. Cut 1 (very likely applied: 190 h > 7 days) removes passes = 3 from the
  grid, so the tuned grid is two learning rates at one pass; epsilon reaches its floor only half way
  through that single pass. Replay ratio, target period, n_env, epsilon fraction, entropy coefficient,
  rollout size, epochs are all fixed at defaults. PPO's rollout 2048, 4 epochs, minibatch 256 gives
  about 15k gradient steps total from ~1e6 transitions, and PPO defaults come from Schulman et al.,
  not the source (unread).
- There is no convergence evidence anywhere, so the paper could not tell "FULL beats DQN" from
  "DQN stopped early". Section 12 promises a "passes = 3 descriptive cell if cut 1 is not applied",
  i.e. never.
- The budget estimate also counts fewer transitions than the plan implies, see finding 3.

Required change:
- Replace epsilon-greedy over {continue, handoff} by: act continue (epsilon-greedy only among
  greedy-continue/greedy-handoff for transitions that are stored anyway, i.e. a continue-forced
  behaviour with the greedy action used only to terminate the episode in the evaluation-checkpoint
  rollouts). Pre-register it; pilot Q2/Q4 verify that every stage has at least `learn_start`-scale
  transitions (log per-stage transition counts in the payload).
- Keep one dev-only convergence check that survives every cut: DQN and PPO at passes 1 vs 3 (or 1 vs
  4) on the 12 tuning routes, rep 0 (about 6 h wall at 12 workers). Pre-register the rule: if the
  3-pass dev score is better than the 1-pass score by more than 0.5 points, the arm is labelled
  "budget-limited" in the paper and no sentence of the form "FULL beats <arm>" is written without that
  qualifier. Without this, cut 1 silently converts the contrast into a statement about 1 pass.
- Add to the grids at least one of {target period 500/2000, epsilon fraction 0.25/0.5} for DQN and
  {rollout 2048/8192, entropy 0.01/0.001} for PPO within the same compute (replace the passes axis if
  compute forces it, as the spec already does).

### 3. [major] Compute projection: inconsistent arithmetic, missing feature-build cost, unbudgeted repairs, cut ordering by harm

Location: 9.1, 9.2, 7 (Repair).

- Table consistency. I recompute 94 h as 0.2+10+5+33+16+26+4 = 94. But the 26 h tuning row (18,600
  job-min) is the *full* grid including passes = 3 (DQN 12 x (2x126 + 2x378) = 12,096; PPO 5,760).
  After cut 1 the tuning is about 4,900 job-min (about 7 h), so the passes-1 plan is about 75 h, not
  94 h. The 94 h figure mixes a worst-case tuning with a best-case main fit.
- Feature building is omitted. 3.5 rebuilds features "from the cache when sampled" for batches of 32
  with mixed k. `stage_batch` (features.py:158) and `_stage_arrays` are per-k and per-call; a mixed-k
  batch of 32 means up to 17 numpy calls per update, twice (s and s'), and 2.5e5 updates per pass, i.e.
  several ms per update on top of the 3e8 FLOP. That is plausibly +50-100% on DQN and PPO. The
  FLOP figure (about 3.5e8 per update, 2.2 MFLOP per padded sample times about 5 forward
  equivalents times 32) is right, but at 15 GFLOP/s sustained for 544-row matmuls on one core is
  optimistic. Pre-registering a pilot-measured projection (the spec's Q5) is right; the pilot must
  time the *real* update including feature building and the 8 checkpoint validations.
- Repairs are not in the budget. A repair "re-fits all 186 jobs" (7). One DQN repair costs 33 h and a
  PPO repair 16 h; two repairs alone exceed the ceiling.
- Cuts: the order (1 DQN/PPO passes, 2 reuse A models, 3 drop DOS, 4 reps, 5 tuning routes) goes from
  most to least harmful to the competitors, the opposite of what fairness needs. Cut 2 is
  harmless to fairness (see 4) but saves only about 10 h; cut 3 saves about 3 h; cut 1 is the one
  that actually removes the 190 -> 94 gap and it hits the competitors' budget.
- No trigger rule: "applied after the pilot" does not say whether all five are applied or applied in
  order until the projection fits. Cut 4 (reps 3 -> 2 for DQN/PPO) as written leaves FULL on 3 reps and
  the arm on 2, so contrasts are unpaired in the training reps. Cut 5 (tune on the 6 long routes)
  biases the chosen config to long routes and applies it to medium ones.
- GPU. The global environment notes a Nvidia GPU. The spec says "CPU only" without saying why. Batch-32
  DQN updates will not speed up much, but 12 independent small jobs per GPU (or vectorised envs) might.
  State what was considered.

Required change: reorder to (i) cut 2 (conditions in 4), (ii) cut 3, (iii) cut 5 only if both
strata are kept (3 medium + 3 long), (iv) cut 4 with FULL recomputed on the same two reps, (v) cut 1
last; declare "apply in order until the pilot projection (with the measured factor, not a x2 guess)
is <= 0.8 x ceiling". Recompute the table with tuning rows tied to the grid used. Budget at least
one repair of the cheapest-to-refit failing arm, or declare that a failing DQN/PPO is dropped without
repair (and say so).

### 4. [major] Ruling on the open item: cut 2 (reuse A's FULL/ladder models) -- ACCEPT WITH CONDITIONS

Reasoning. The rule "every evaluated model is fitted at the frozen commit" exists so the models are
fixed before test data are read and the code that evaluates them is the code that fitted them. A's
models were fixed before any B test stream existed, were fitted on a clean tree (`fit_tree_clean`
recorded), and are the models behind A's published numbers. Reusing them is *safer* against
accidental refit differences than refitting, and it spends nothing from the competitors' budget. It
does not touch any competitor. It does, however, weaken the machine-checkable invariant, and the
pilot test as written (Q3: one route, one rep) is too thin to license it for 186 jobs.

Conditions:
1. Q3 covers at least 6 jobs: one short, one medium, one long route, both CON3 and SCA3, reps 0-2,
   including the slowest observed job; `np.array_equal` on all 5,000 dev rows of every fold net.
   Deterministic-algorithm flags do not guarantee bit-equality across BLAS/thread/driver changes.
2. The manifest rule of finding 1: sha256 of the code files that define features, nets, backward
   induction, simulation and the classical ladder identical at c37ad7c and at the B frozen commit; plus
   a test that the train pool at B equals A's (hash of the first 10^5 rows per instance, rep).
3. The manifest of used model files (sha256, fit_commit == c37ad7c, `fit_tree_clean == True`) is
   committed *before* the family line and the gate verifies it (not the author).
4. REP is then computed with A's models; the paper says so, because REP no longer tests a refit
   (see 7).
5. Ordered first among cuts (finding 3), applied only if the projection requires it. If the refit
   fits the budget, refit (preferred: fewer special cases).

### 5. [major] Fidelity to ALP (Zhu et al.) and to THR: the independence assumption makes them lose by construction; ALP unread

Location: 3.1, 3.2, 12 ("Information class confound").

- THR/ALP use the marginals S_j and treat stops as independent, as in their source models. The
  generator uses a Gaussian copula (rho = 0.6) over stops (data.py:73-94; REVIEW_A item 12), so the
  mis-specification is large and systematic. A FULL "WIN" against THR/ALP then says little about
  FULL being a better *method*; it measures broken-assumption cost. The spec acknowledges this in
  section 12 but offers no way to separate it.
- ALP: I could not read the method section (finding intro). The spec's own wording ("in the spirit of
  Zhu et al.") is correct. But two modelling choices are the drafter's, not the paper's, and bear on
  strength: (i) the basis is per-stage affine plus q hinge terms in W only, without Zhu et al.'s
  decomposition/remaining-demand features; (ii) the LP maximises an unweighted sum over sampled states
  (no state-relevance weights), and the lower-bound character biases Ĉ downward, i.e. later handoffs.
  A published ALP could be stronger or weaker.
- The fidelity "lower bound" gate G2-ALP (section 7) is mostly tautological: the constraints hold at
  the sampled states by construction, and the bound can fail only at off-sample states w+g because
  theta_{k+1} is evaluated off-sample (so the induction to V* does not close). The gate as worded
  "at sampled states" will pass whenever the LP solved.

Required change:
- Add a **descriptive, pre-registered** evaluation of THR and ALP on an independence-resampled copy of
  test_b days (columns permuted by seeded permutations, as in G2-THR). It is outside the family and
  states what each method achieves in its own model world; with it the paper can say "THR and ALP lose
  here because of dependence, not because of implementation" or the opposite.
- Redefine G2-ALP to use held-out alive rows (not the LP's constraint sample) and report the
  excess over V^THR; the threshold of 1e-3 E0 and 99% is [DC] and should say so (finding 8).
- In the paper and in 3.2, state the two drafter-chosen elements above. Re-check the mapping when the
  source becomes readable (author access to the Networks paper or the GERAD report); if the source's
  basis differs materially, add the source basis to the ALP grid before tuning.

### 6. [major] Other fidelity issues in 3.3-3.6

- RLSM (Herrera et al.): spec uses N(0,1) weights, LeakyReLU slope 0.01, K in {20,100,400}, ridge
  1e-6 n. All [VERIFY]; I could not retrieve the paper. The RLSM literature uses data-dependent
  scaling of the random layer; with inputs clipped to [-5,5] and N(0,1) weights the hidden units are
  in a sane range, but K = 400 on 40k rows with ridge 1e-6 n is close to unregularised; add a ridge
  axis (1e-6, 1e-3 times n) so the arm has the regularisation the published method tunes.
- DOS (Becker et al.): width d+40, two hidden layers, batch norm, Adam, batch 8192 match my recollection
  of the paper. Their method resamples fresh paths every step; here a fixed 40k rows per fold is
  visited about 20 times (S_k = ceil(20 n_k/8192)), which is an over-fit regime. It is optional and
  likely a TIE; the only requirement is that the paper says "on a fixed sample".
- Surrogates (section 2): S = H + E0(pi - 0.5) passes through `simulate_policy` as a mean over the
  predictor list. For DOS the fold mean of F decides at 0.5, consistent with `ls_backward` (backward.py:
  170-176 averages predictions of the two fold nets on validation rows). OK. State that the PPO
  surrogate has no fold structure (single net).
- DQN: learning Q_cont only with H known is a favourable adaptation (stated). The target uses
  `V' = H[k] if Q_online(s') > H[k] else Q_target(s')`; that is the correct double-Q evaluation for a
  min problem. The checkpoint selection runs `simulate_policy` on 20k validation rows for 8
  checkpoints; budgeted in my 3 only if included.
- G-level source checks: 3.0 requires filling a [VERIFY] table before the pilot. Good; add that for
  ALP and PPO the table says "source unread" and the paper wording is "in the spirit of / PPO with a set
  encoder" (the spec already has this).

### 7. [major] Statistics: family, Holm, delta, REP, power

- Family F is fixed pre-test_b from dev gates. Acceptable. But a *dropped* arm disappears from the
  test_b analysis. A gate-dropped arm should still be evaluated on test_b as a descriptive row
  (test_b is B's own stream and costs minutes), with the family unchanged. As written, "never
  evaluated on test_b" removes information while protecting nothing: the family decision is already
  fixed.
- Mixed decision logic: WIN requires CI95 lower > 0 (unadjusted) and Holm p < 0.05, plus Delta-bar >=
  delta and 13/20. With |F| = 6 the CI95 is the looser condition. Report CI at the Holm-adjusted
  level (0.05/|F|, two-sided) alongside, or state that the CI95 is descriptive and Holm p is the gate.
  Equivalence (CI90, unadjusted) across 6 arms has no multiplicity control: stated [DC] in 1.3; say
  the equivalence claims are per-arm and not simultaneous.
- Power. With n = 20 and instance-level SD s, CI90 half-width is 0.387 s and CI95 half-width 0.468 s.
  A's s = 0.365 gives 0.14 and 0.17. TIE_STRONG for delta = 0.7 needs |Delta-bar| + 0.387 s < 0.7,
  attainable up to s about 1.8 at zero effect. For DQN and PPO s is likely 1-3% (between-replicate
  noise of RL), so TIE_STRONG is unlikely and WIN needs Delta-bar >= 0.7 with s <= about 2 for
  Holm p < 0.05/(6): a likely TIE_WEAK. Say now that a weak tie against a high-variance RL arm is
  expected and is reported as "no difference established", not as FULL = arm.
- REP is not a replication in the sense of independent training: models are bitwise the same (if
  Q3 holds) and only test days are new, so REP replicates the test-day sampling noise (which is
  small: per-day paired SE) and nothing about training-draw variance. Rename it "re-evaluation on a
  fresh test stream", and keep the three training replicates as the only training-noise source. If
  the paper wants a training replication, three new training reps (3..5) are needed.
- delta = 0.7% inherited from A. Fine for comparability. A is a WIN margin chosen from measured noise
  at 10^5; for RL arms the between-rep SD may exceed it. Report each arm's between-rep SD in the
  compute table (8 item 4 does).
- Tuning/test separation: the 12 tuning routes come from only 10 instances (CON3-0,1,2,3,5,6,7,9 and
  SCA3-1,2; 10 CON3 and 2 SCA3 routes), also in the primary set. Tuning uses dev days only: OK. State
  that half the instances are also tuning instances.
- The tuning score is relative to baton_eq, not FULL: good, and "no dev contrast FULL vs arm" is
  correct. But the author has seen A's dev and test FULL numbers; the priors in 1.1 are stated, which
  is the right disclosure.

### 8. [major] Fidelity-gate thresholds are arbitrary, partly tautological, and one is gameable

All thresholds are [DC] with no pilot evidence, as the spec admits. Specific issues:
- G1 (C <= 1.005 C_none on >= 90% of route-reps): sensible as a sanity bound, but it also drops an
  arm for being *weak*. A DQN that cannot beat "never hand off" at one pass is a result, not a bug.
  Separate bug gates (G0, G2) from performance gates; evaluate a G1-failing arm on test_b
  descriptively (finding 7).
- G2-DQN "learns (0.98 x update-0)": update-0 is a random network whose greedy policy is arbitrary
  (could be near never/always handoff), so 2% below it is a very weak bar for "learns" and varies
  with initialisation. Use "validation cost of the kept checkpoint <= validation cost of the
  best of {never handoff, always handoff at the cheapest stage}" or reuse G1 on validation rows.
  "Calibration within 15%" pools stages (errors cancel) and the 75% share is untested; require
  stage-wise calibration on the late stages or drop it.
- G2-PPO "mean |pi - 0.5| >= 0.3": a well-calibrated stochastic policy near the boundary fails; an
  overconfident wrong policy passes. It can be gamed by raising the entropy coefficient down or
  lr. Replace by a performance sanity bound on validation rows.
- G2-RLSM and G2-DOS (beat the constant predictor / min of always-stop, never-stop on validation):
  nearly always true; they will never fail and so protect nothing. Acceptable as bug detectors, but do
  not present them as "fidelity to the published method".
- G2-THR (3 SE on >= 95% of route-reps): correct in design, and the one gate that tests the
  implementation exactly. 3 SE per route-rep should pass 99.7% by chance, so 95% is loose;
  tighten to 99%. SE must be the per-day SE of the 5,000 permuted days (state it).
- G2-ALP: finding 5.
- The "2 repair iterations" are a tuning channel on dev; allowed, but the repair log must record whether
  each repair changed a hyper-parameter or fixed a bug, and a hyper-parameter repair counts against the
  grid-based tuning of section 6 (otherwise a gate becomes a second tuning pass for arms only).
- Gaming direction: since gates only drop arms and the authors favour FULL, dropping a competitor for
  G1 reduces what can be claimed against FULL, i.e. it is conservative; but the *repair* channel lets
  one arm get more tuning than another. Pre-register repair budgets equal across arms.

### 9. [major] Implementability against the real code (exact mismatches)

1. `run_fit_job` unpacks three values from `fit_neural`/`fit_boost` (sweep.py:392-397); B's
   `fit_*` return four. M21 says dispatch is added; it must also change `payload` construction
   (`models=_pack(...)`) for B arms. OK as planned, but the `_pack`/`_unpack` pair raises `TypeError`/
   builds `SetNet()` for any non-known predictor (sweep.py:300-321); `codecs` must be hooked in both.
2. `StageSpec` has no `arm_cfg` or `test_stream` (sweep.py:105-117); `plan_stage` (240) reads
   `spec.neural` as `(arm, N)` pairs. B's per-stage config means tuning stages are distinct stages
   with their own models dir; `Job.key` carries no config, which is fine only because the directory
   is per stage. Say so.
3. Literal "test" checks: `check_test_gate` (279-285), `run_eval_job` (440-445), `route_pool`
   (313, 331), `_stream` (227), `EVAL_SETS` (73), `merge_stage` (543). M12 names the stream, token and
   `verify_test_gate`; the per-model checks at 441/444 are not named (finding 1d).
4. `GateToken` is a frozen dataclass with a custom issuer check (paths.py:125-134); adding `stream`
   is fine, but `_require_gate` (165) compares purpose to "test" only; the mutation "accept stream
   mismatch" needs both `test` and `test_b` branches covered.
5. M17: A's `SetNet.forward(glob, tok)` (model.py:51-61) builds `h = cat[g, tok]`, then applies
   `nn.TransformerEncoderLayer` instances from a `ModuleList` (model.py:41-47), not an
   `nn.TransformerEncoder`. `enable_nested_tensor=False` is irrelevant there; the mask must be passed
   per layer as `src_key_padding_mask` with a prepended False for the global token, and the pooled mean
   (line 59) must be masked. In eval mode PyTorch may use its fast path with a padding mask and
   nested tensors; the atol 1e-5 test (a) can then fail for reasons unrelated to correctness. Disable the
   fast path in tests (`torch.backends.mha.set_fastpath_enabled(False)`) or test in train mode with dropout 0.
6. `padded_batch(cache, ks, rows)` requires per-k `stage_batch` calls (features.py:158 takes a scalar k,
   `_check_stage` 92-95); it is not a batch builder over mixed k. That is the cost in finding 3; either
   precompute per-(row,k) feature tensors once per job (about 80k x 16 x 177 floats = 0.9 GB per job,
   12 workers = 11 GB, fits 64 GB) or accept the cost.
7. `NetPredictor` only accepts `ARMS_NEURAL` and `_inputs` (train.py:95-99); new predictors must
   implement `predict(cache, k, idx)` as the Protocol (backward.py:34) says. Fine.
8. `FitFn` is `(k, cache, fit_idx, y_fit, val_idx, y_val, warm)` (backward.py:40-41); RLSM/DOS plans match.
   DOS warm start: `warm` is the previous stage's predictor of the *same fold* (backward.py:170-185), but
   when the previous stage was a `ConstantPredictor` (n < min_rows) the DOS net must cold-start;
   specify as `neural_fit_fn` does (train.py:222-224).
9. Analysis: `analysis.write_summary` rejects eval_set other than dev/test (analysis.py:486), and `FAMILY_SIZE
   = 11` is hard-coded (38). M23 says it reuses helpers only; confirm it never calls `write_summary`. B
   stages produce separate CSVs (`<stage>.test_b.csv`); the A boost run showed that concatenation was
   a manual hack (`main_boost.test.csv`). M23 should take a list of stage tables and assert equal
   test_b day counts and fit commits per row. `classify()` (analysis.py:184) lacks the 13/20 count of the
   WIN rule, which is why `go_no_go_holm` is separate; test it against A's boundary vectors as planned.
10. M23 `replication` uses A's `go_no_go` (one-sided 0.025): consistent with A.
11. Tests on `b_smoke`: "exactly {none, baton_eq, oracle, nn_full, thr, ...}" conflicts with the A
    ladder labels (`LADDER_LABELS` includes `tuned_tau`; `ARM_POLICY` maps `full -> nn_full`); say which
    `ladder_labels` the b_* stages use.

### 10. [minor]

- Section 3.1 grid: THR at 2,001 points and linear interpolation clipped at w_lo introduces interpolation
  bias near the threshold; report the grid-refinement check (4001) on the G2-THR route-reps.
- Section 4.2 `path_stop_reads` for DQN/PPO should count steps actually taken (with early termination),
  not a nominal N*m.
- Section 5.1: the bootstrap seed 20261118 equals the stream entropy; harmless, but use 20261119 to avoid
  confusion in logs.
- 1.3: "p_Holm < 0.05 equals A's one-sided 0.025 per direction when |F| = 1": correct for the exact
  symmetric test.
- Wording: "published algorithms" for ALP and PPO should be "reconstructed in the spirit of" in the abstract
  of any paper, not only in 3.2/3.6.
- 12 stop rule 2 (fewer than 3 published arms pass): cut 3 (drop DOS) is both a compute cut and an arm
  count reduction; count it before stop rule 2.
- M12 mutation list: add "use `[base, 20261118]` twice with different `purpose` caches" (chunk_file key
  includes purpose, so the cache is safe, but test it).

## Items checked and found sound

- The new seed `[base, 20261118]` is unused anywhere (repo grep), and differs as a SeedSequence state
  from every existing stream; chunk cache keys include purpose and seed (paths.py:230-235).
- Fidelity gate G2-THR as an exact-under-own-model test is a good design.
- THR recursion indexes: breach at stop k+1 costs E[k] (= E[tau-1]), handoff at epoch k costs H[k-1], no
  decision at k = m: consistent with `simulate_policy` (simulate.py:55-77). The THR vs dp vs baton_eq
  discussion (baton_eq is already a threshold policy) is correct.
- ALP LP direction (maximise, V <= H, V <= TV) gives the greatest lower bound; formulation is standard.
- Cross-fitting via `ls_backward` and the folds is reused for RLSM and DOS exactly as A's boost; every
  fit row is never a decision row at the same epoch (backward.py:152-185).
- Paired design: every arm uses the same training rows, dev rows and test_b days; instance-level n = 20
  inherited from REVIEW_A finding 1; 20 instances, 62 routes, 12-route tuning subset match
  `eligibility_dev.csv`.
- Arithmetic of the 94 h / 190 h rows reproduces from the table; the DQN FLOP count per update agrees with
  my hand count.

## Verdict: ACCEPT WITH REQUIRED CHANGES

Required changes before the pilot (sections frozen after these edits):

1. Rewrite the test_b gate (finding 1): committed-blob log, ancestor-of-HEAD with empty code diff,
   anchored regexes for both streams, all literal "test" checks edited, family/dropped enforced per
   stage, fidelity-file hash in the family line, equal-code manifest across frozen commits, SeedSequence
   state comparison test.
2. Fix the DQN behaviour policy and keep a dev convergence check (1 vs 3+ passes) that survives every
   cut; add one more hyper-parameter axis for DQN and PPO; pre-register the "budget-limited" wording rule
   (finding 2).
3. Recompute section 9: tuning rows tied to the actual grid, include feature-building and checkpoint
   validation costs, budget (or forbid) repairs, reorder cuts by harm to competitors, state the trigger,
   keep FULL paired with arms on the same reps under cut 4 (finding 3).
4. Cut 2: accept with the five conditions of finding 4; applied first and only if needed.
5. Add the independence-resampled descriptive test_b evaluation of THR and ALP; redefine G2-ALP on
   held-out states (finding 5).
6. Keep "in the spirit of" for ALP and PPO and record that neither source was read (finding intro, 6);
   re-check when a copy is obtained, before tuning.
7. Evaluate gate-dropped arms on test_b descriptively; separate bug gates from performance gates;
   replace G2-DQN "learns", G2-DQN calibration and G2-PPO "converged" by non-gameable validation-cost
   bounds; tighten G2-THR to 99%; equal repair budgets (findings 7, 8).
8. Rename REP "re-evaluation on a fresh stream"; report Holm-level CI beside CI95; state expected
   weak ties for high-variance RL arms (finding 7).
9. Fix the code mismatches of finding 9 (per-model literals, `_pack/_unpack` hooks, SetNet masking in
   per-layer calls and fast-path test settings, analysis `write_summary` guard, ladder labels).

---

## Author response (EXPERIMENT_B.md revision 2, 2026-11-18)

Added by the spec author; the review text above is unchanged. Section numbers refer to
EXPERIMENT_B.md revision 2. Sources now read in full: `METHODS_ALP_PPO.md`.

| finding | severity | resolution | where |
|---|---|---|---|
| 1a contradiction (HEAD == frozen vs committed log line) | critical | Log read from `git show HEAD:…/RESEARCH_LOG.md`; HEAD equality replaced by: fit commit F ancestor of HEAD, `git diff --quiet F HEAD -- SHARED ∪ OWN(arm)`, F's log blob lacks the FAMILY line and the stage's FROZEN line. Commit roles (F, repair F2, log commit L) and numbered commands given. | 5.3 |
| 1b log is an editable working-tree file | critical | Committed blob only; log and source must be clean in `git status`. | 5.3 checks 1-2 |
| 1c cross-stream substring bypass | critical | A's gate uses anchored `^FROZEN COMMIT: <h>$` (A's lines already have that form); B has its own anchored regex and token stream; tests for both directions. | 5.2, M12 (d) |
| 1d literal "test" checks | critical | `TEST_STREAMS` and an exact line list (paths.py 45, 126-167, 214-227, 313, 331; sweep.py 76, 105, 275, 435-445, 543); one mutation per edit. | 5.2, M12 |
| 1e nothing binds family line to stages | critical | FROZEN lines name their stages; each stage in exactly one; arm must be in family/descriptive/dropped/references; fidelity_dev.md sha256 in the FAMILY line, checked by gate and analysis_b. | 5.3 checks 3, 7, 8 |
| 1f different code across frozen commits | critical | Per-stage diff on SHARED ∪ OWN; a repair may touch only the arm's own module, else all stages are refitted. | 5.3 check 6; 7 |
| 1 SeedSequence state comparison | required | Added (states, not only first rows), all 20 instances; test_std also excluded from B. | 5.1, M12 (a), (f) |
| 2 exploration starves late stages | critical | DQN behaviour is continue-forced (off-policy, known handoff value): every alive (row, k) is stored; the exact count is a bug gate. PPO: initial handoff probability 0.05; visits per stage logged (pilot Q6). | 3.5, 3.6, 6.3, 7 |
| 2 one pass minimum / no convergence evidence | critical | Every DQN/PPO tuning run trains to 3 passes with pass-end checkpoints scored on dev (learning curve); pass count chosen by the dev rule; "budget-limited" label rule pre-registered; curves survive every cut; P = 1 is the last cut. | 6.3, 9.2 (v) |
| 2 more grid axes | required | DQN: target period {8k, 32k} stored transitions; PPO: γ {0.5 (source), 1.0}. Rollout and entropy axes not used because the source settles them (one episode per update, no entropy term). | 3.5, 3.6 |
| 2 PPO defaults from Schulman, not source | critical | Replaced by Kadyrov et al.'s stated settings (p. 4); unstated items checked in their code before tuning. | 3.6; METHODS §2.3 |
| 3 table arithmetic / tuning row | major | Recomputed; tuning rows tied to the grids with P_max = 3; P = 1 / 3 totals ≈ 83 h / ≈ 135 h (GPU for DQN), ≈ 271 h CPU-only at P = 3. | 9.1 |
| 3 feature building cost | major | `PaddedTables` built once per job (≈ 5 min, ≈ 0.9 GB) and counted; pilot Q5 times the real loop incl. tables and checkpoint validation. | M17, 9.1, 9.3 |
| 3 repairs unbudgeted | major | One repair per arm, bug fixes only, paid from headroom above the 0.8 × ceiling trigger; unaffordable → dropped (descriptive on test_b). | 7, 9 |
| 3 cut order, trigger, cut 4 pairing, cut 5 strata | major | Order (i) reuse A, (ii) drop DOS, (iii) 6 routes with 3 medium + 3 long, (iv) reps 3 → 2 for all arms incl. FULL and ladder, (v) P = 1; applied in order until the measured projection ≤ 134.4 h. | 9.2 |
| 3 GPU | major | Device rule: GPU iff ≥ 3× faster per pass and bit-identical reruns on CUDA; DQN batch 256 at the source's replay ratio expected on GPU, PPO per-episode updates on CPU. | 9 |
| 4 cut 2 ruling | major | Accepted with all five conditions, plus: re-evaluating A's files on dev with B's code must reproduce A's committed dev rows exactly. Applied first, only if needed. | 9.2 (i) |
| 5 THR/ALP lose by construction | major | Independence world added: test_b with stop columns permuted (THR, ALP-Z, none, oracle, THR's DP value) and FULL refitted on permuted training rows for the 12 subset routes. Descriptive. | 3.8 |
| 5 ALP unread / drafter's basis and objective | major | Source read. Zhu et al.'s basis is indexed by load (eq. 23.2), objective = initial state (24.1), all loads constrained (26). On a fixed route the basis is exact, so their ALP is the LP of the independent-demand DP. Arm redesigned as ALP-Z, faithful; revision 1's reconstruction withdrawn; ALP-Z enters the family only if its dev decisions differ from THR's on ≥ 1% of alive states. | 3.2; METHODS §1, §3.1 |
| 5 G2-ALP tautological | major | Replaced by exact identities: LP objective = lattice-DP initial value; uniform-weight LP = lattice DP everywhere; off-path vertex gaps reported. | 7 B-ALPZ |
| 6 RLSM ridge | major | Ridge axis {1e-6, 1e-3}·n. | 3.3 |
| 6 DOS fixed sample, warm start | major | Stated "on a fixed sample"; cold start after a constant predictor. | 3.4 |
| 6 PPO surrogate has no fold structure | major | Stated. | 2 |
| 6 "source unread" wording | major | ALP and PPO now read; paper wording rule given for every arm (header). DQN/RLSM/DOS remain [VERIFY]. | header, 3.0 |
| 7 dropped arms not evaluated | major | Dropped, P1-failing and non-distinct arms get descriptive test_b rows. | 1.4, 7 |
| 7 mixed CI95 / Holm logic | major | p_Holm is the multiplicity gate; CI95 a consistency condition; Holm-level CI reported; equivalence per arm, not simultaneous. | 1.3 |
| 7 power / weak ties | major | Stated with numbers; a weak tie is expected for RL arms and worded "no difference established". | 1.1, 8 item 8 |
| 7 REP not a replication | major | Renamed RE-EVAL; it re-samples test days only; a training replication is out of scope. | 1.1, 1.4 |
| 7 tuning instances | major | 10 of 20 instances hold tuning routes; stated. | 1.2, 12 |
| 8 G1 conflates bug and weakness | major | Bug gates (B0, B-<arm>) separated from the performance bound P1; P1 failure → descriptive, no repair. | 7 |
| 8 G2-DQN learns / calibration; G2-PPO converged | major | Removed. Replaced by exact bug gates (per-stage transition counts; masking; cached = recomputed predictor) and P1. | 7 |
| 8 G2-RLSM/DOS near-tautological | major | Kept as bug detectors, labelled as such, not presented as fidelity. | 7 |
| 8 G2-THR 95% loose; SE definition | major | 99%, per-day SE of the 5,000 permuted days; monotonicity of C_k is now an exact gate. | 7 B-THR |
| 8 repairs as tuning channel; equal budgets | major | One repair per arm, bug fixes only; hyper-parameter changes are not repairs. | 7 |
| 9.1 four return values, `_pack/_unpack` | major | New branch before the generic neural one; `_pack/_unpack` delegate to codecs on their first line. | M21 |
| 9.2 StageSpec, Job.key | major | `arm_cfg`, `test_stream`, `independent` added; per-stage models dir stated. | 5.2, 6.1, M21 |
| 9.3, 9.4 literals, `_require_gate` branches | major | See 1d; both stream branches tested. | 5.2, M12 |
| 9.5 SetNet masking and fast path | major | Mask passed to each layer of the ModuleList with a prepended False; masked mean; fast path disabled only inside a test fixture; test that importing expb leaves A's SetNet outputs bit-identical. | M17 |
| 9.6 mixed-k `padded_batch` | major | Replaced by `PaddedTables` (one `stage_batch` per k per job, then gathers). | M17 |
| 9.7, 9.8 predictor protocol, DOS warm start | major | B predictors implement `predict(cache, k, idx)`; DOS cold start specified. | 3.4, M20 |
| 9.9, 9.10 analysis guard, stage tables | major | analysis_b never calls `write_summary`; `load_stage_tables` asserts day counts, one eval commit, fit commits; the 13/20 count is in `go_no_go_holm`. | M23 |
| 9.11 ladder labels | major | b_* stages use ("none", "baton_eq", "oracle"). | 3.7, M21 |
| 10 minor items | minor | 4,001-point refinement reported; actual steps counted; bootstrap seed 20261119; wording rule; stop rule 2 counts cut (ii); seed-reuse mutation added. | 3.1, 4.2, 1.2, header, 12, M12 |

Required changes 1-9 of the verdict map to the rows above (1 → finding 1; 2 → finding 2;
3 → finding 3; 4 → finding 4; 5 → finding 5; 6 → finding 6 and METHODS_ALP_PPO.md;
7 → findings 7-8; 8 → finding 7; 9 → finding 9).
