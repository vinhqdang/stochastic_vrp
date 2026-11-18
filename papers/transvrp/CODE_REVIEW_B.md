# CODE_REVIEW_B: independent whole-code review of experiment B

Date: 2026-11-19. Scope: `code/transvrp/expb/*` (gatecheck, grid, thr, alp, rlsm, dos, padded, dqn,
ppo, codecs, registry, stages, curves, fidelity, analysis_b), the edits in `expa/paths.py`,
`expa/sweep.py`, `expa/analysis.py` (`git diff c37ad7c`), `MUTATION_B.md`, against
`EXPERIMENT_B.md` revision 3.2, `METHODS_ALP_PPO.md`, `VERIFY_CITATIONS.md`. Reviewer worked from
the code and the spec only; no test_b data exists or was read, nothing was evaluated on test or
test_b, no tracked file other than this one was changed, nothing was committed. Throwaway scripts
live in the session scratchpad (a local clone of the repository was used for the gate walk).

**Verdict: accept with required changes.** No result-bearing arm code is wrong, and no path to
test_b data without the gate was found. But the commit sequence of 5.3 cannot be executed as
written (R1, R2), one arm is dropped by a gate no correct implementation meets (R3), and the
analysis can be run from uncommitted code (R5). R1-R3 and R5 block step 3 of the sequence.

## 0. Test suite

Full suite, once, `python -m pytest tests -q -p no:cacheprovider` from `code/` (slow tests
included: `pytest.ini` has no `addopts`): **582 passed, 0 failed, 0 skipped-reported, in 1365.72 s (22 min 45 s)**, while two of the
reviewer's own scripts ran alongside. Git state of the repository checked out as before
(no tracked file changed by the run).

Order and wall-clock dependence: none found. Module-scoped fixtures (`test_expb_fidelity.py:89,94`,
`test_expb_codecs.py:52`) are read-only. Wall-clock appears only as `st_mtime_ns` equality before
and after a rerun (`test_expb_sweep.py:254-263`, as in A), which cannot flake. The slow tests are
part of the count above because the marker is not deselected by default.

## 1. Critical findings (block the sequence)

**C1. The sequence's commands write B models where the gate does not look.** `sweep.py:76`
`DEFAULT_RESULTS = code/results/expA`, and `fit`, `eval`, `merge`, `timing` default `--results-dir`
to it (`sweep.py:783, 801`); `sweep manifest` has no `--results-dir` at all (`sweep.py:797`). The
gate, the manifest writer, `family_report` and `fidelity` assume `code/results/expB`
(`gatecheck.py:30, 157, 440-455`, `fidelity.py:62`). Spec 5.3 steps 3, 4, 10 give no
`--results-dir`. Demonstrated in a clone: `sweep fit --stage b_thr --only ...` with the spec's
command produced `results/expA/models/b_thr/thr__CON3-0__r1__N100000__rep0.pkl`, mixing B models
into A's results tree; `sweep manifest` then reads `expB/models/b_thr`, finds nothing, writes an
empty manifest, and the gate refuses ("0 model files pinned; 186 planned"). It fails closed, but a
literal run deadlocks. The spec's M21 text says results go under `expB`; the code default is wrong.

**C2. Cut (i), which the spec expects to apply, cannot pass the gate.** In the clone, with 186
fabricated A files for `b_full_reuse` and `b_ladder_reuse` in `expA/models/main` (fit commit
c37ad7c, no `libraries` key, as A wrote them):
1. `gatecheck.model_dir` (`gatecheck.py:157`) ignores `StageSpec.reuse_models`, so
   `write_manifest` lists `expB/models/b_full_reuse` (empty): manifest of 0 lines;
   `family_report` then fails ("model files from 0 commits") or the stage is missing from every
   FROZEN line, and `verify_stage` refuses ("appears in 0 FROZEN lines"). Observed.
2. Even with the directory fixed, `check_models` (`gatecheck.py:345-349`) requires the files on
   disk to equal the manifest, but `models/main` holds A's other jobs (N = 100..10^6, scalar, mlp,
   big); the reuse stage must filter to its planned keys the way `run_eval` does
   (`sweep.py:652-654`).
3. `check_libraries` (`gatecheck.py:386-399`) compares the running versions with
   `payload["libraries"]`; A's payloads have none, so every reuse stage is refused (and
   `_test_b_columns` would write `None` library columns). Spec 5.3 item 8 has no rule for this.
   The spec's substitute is condition 3 of cut (i) (dev re-evaluation reproduces A's rows
   bit-for-bit); the gate should require that record (or a pinned library record) for reuse stages.

## 2. Major findings

**M1. B-DOS cannot be met by a correct DOS at the specified size, so DOS would be dropped by our
defect.** Measured on a real route (CON3-6 route 1, m = 12, N = 10^5, rep 0, `DOSConfig(lr=1e-3)`,
validation objective read as `fidelity._gate_dos` does): 17 of 22 stages (77%) meet
`obj <= min(always, never) + 1e-6`; the gate needs 95%. Re-running with 400 steps in every stage
(the cap) gives the same 17/22. The failing stages are the near-constant ones, where a sigmoid
network cannot reach a constant to 1e-6 of E0: k = 10 gives 0.016373 against never-stop
0.016349 (gap 2.4e-5, 400 steps); at the specified ~84-98 steps k = 11 gives 0.0296 against
0.0048, i.e. the network is also under-trained (6 times the constant's loss). Two defects: (a)
`DOS_TOL = 1e-6` (`fidelity.py:53`, spec T7/section 7) is an absolute tolerance no sigmoid
output meets on clear-cut stages; (b) `stage_steps` (`dos.py:71-74`, 3.4) gives ~90 steps at
n_fit = 40,000 (`ceil(20 n / 8192)`), too few to reach even the better constant (spec 14(c)
discloses fewer steps than the source, not this consequence). Spec 7 says a hyper-parameter change
is not a repair, so a failure here would end DOS in `dropped`. Decide before the pilot: a relative
tolerance (for example 1% of |H - never|/E0, or 1e-3), and/or a floor on steps. The spec's
"pilot changes no default or grid" text then applies to the corrected value.

**M2. Cuts (ii) to (v) have no code path, and cut (ii) is blocked by the tooling.**
`fidelity.derive_family` raises when an arm has no report (`fidelity.py:574-580`), and `main`
reads `b_dos.csv` unconditionally (`fidelity.py:653-658`); the spec's own line
`dropped=[dos: compute cut]` contains a space the FAMILY regex forbids (`gatecheck.py:52-54`
class `[a-z0-9_:,]`); `analysis_b.check_membership` refuses a listed arm without rows
(`analysis_b.py:173-175`). Cut (iv) needs `reps` edits in `stages.py` and `N_ROUTES/N_REPS`
constants in `analysis_b.py:40-41`; cut (iii) (6 tuning routes) and (v) (P = 1) have no switch.
These are edits before F, so they are allowed, but the cut the spec says a 1% slower pilot will
trigger cannot be executed. Provide an `--skip dos` path, a spelling such as
`dos:compute_cut`, and the stage/analysis constants as one pre-committed parameter.

**M3. DQN with `--jobs 12` spawns twelve CUDA processes.** `_run_all` (`sweep.py:609-615`) spawns
`jobs` workers for every stage; spec 5.3 step 3 runs every stage with `--jobs 12`; header and 9.1
say one GPU process. Twelve processes each hold a ~0.9-1.2 GB `PaddedTables` (m = 19: 18 stages x
80,000 rows x 18 tokens x 11 x 4 B = 1.14 GB) plus a CUDA context, more than the 16 GB card.
Needs a guard (force `jobs = 1` for a stage whose arm config resolves to CUDA) or a spec line.
CPU stages are fine (12 x ~1.4 GB for PPO).

**M4. `analysis_b` can run from uncommitted code, and the ANALYSIS PATCH rule is thinly enforced.**
`main` (`analysis_b.py:527-560`) hashes the HEAD blobs of the analysis files
(`G.tree_hash(head, ANALYSIS_RELS)`) but executes whatever is in the working tree and never calls
`check_clean`. An edited `analysis_b.py`, `expa/analysis.py` or `fidelity.py` passes the check as
long as HEAD's blobs match. Item 6: accepting "the last ANALYSIS PATCH" hash is licensed by spec
5.3 (verbatim: "the hash in the last ANALYSIS PATCH line ... each patch line is added once, in
its own commit, before the patched analysis runs, with a reason"). The code accepts the last
patch (`analysis_b.py:511-517`) but enforces none of the parenthesis: the `-G` pin pattern
(`gatecheck.py:65`) deliberately excludes PATCH lines, so a patch line can be rewritten or
amended freely. A patch can change any statistic after test_b is seen; the family and the
FAMILY-line lists remain pinned. Required: `check_clean` plus byte equality of the executed files
with the HEAD blobs inside `analysis_b.main`; a patch line must come from exactly one commit
that is a descendant of L and touches only that line.

**M5. B-THR (a) drops a correct THR about 15% of the time.** The dev stream does not depend on
the replicate, so the three reps of a route share the same permuted dev days and fail together;
the gate loops over 186 jobs and needs 99% (at most 1 failure, `fidelity.py:295-314`). One route
outside 3 SE therefore fails the gate. For an exact THR, P(some of 62 routes beyond 3 SE) =
1 - 0.9973^62 = 15%. Measured z = 0.98 on the one real THR fit made (CON3-0 route 1), so no bias
was found, only the false-drop probability. Fix before the pilot: test per route on the rep mean
and keep the 99% share over 62 routes, or widen the band to 4 SE; amend T4.

**M6. PPO evaluation recomputes the GRU from stage 1 at every stage.** `unpack_models`
(`codecs.py:98-106`) builds one `PolicyPredictor` per k, so the h_k cache
(`ppo.py` `PolicyPredictor._reusable`) never fires in `simulate_policy` after a payload round
trip: m(m-1)/2 stage forwards instead of m - 1. Verified (`um[1][0] is um[2][0]` is False with
one net stored); values are bitwise equal (round trip, m = 19, 1,000 rows, max gap 0.0).
Measured 9.4 s per 1,000 rows at m = 19 under load, about 3 min per job at 20,000 days, about 2 h
for the dev and test_b evaluations of 186 jobs on 12 workers, which section 9's 4 h line does not
cover. Share one predictor per net in `unpack_models`.

**M7. `analysis_b` omits reports the spec requires.** Absent: D_a (each arm vs baton_eq,
section 1.4); ALP-Z vs THR test_b agreement and paired difference (1.4); strata (8.5);
between-replicate SD (8.4); the budget-limited label in every claim sentence (1.3, `arm_claim`,
`analysis_b.py:405-433`, has no label input); THR's realised-cost-over-DP-value line (3.8);
`peak_rss_mb` is a compute-table column (`analysis_b.py:63`) but no fit writes it into `budget`.
Everything omitted here can only be added later through an ANALYSIS PATCH, after test_b is open.
Add before M.

**M8. The section 6.1 selection rule is not implemented.** No code applies the 0.10-point tie
window, fewer passes, fewer seconds, lower index, or the crash exclusion;
`fidelity.tuning_scores` serves DQN/PPO pass curves only (`arm@pP`) and fails for RLSM and DOS
tuning rows. Choosing by hand on dev is allowed (dev only), but the spec calls the rule fixed;
write `choose_config` with tests so `CHOSEN_ARM_CFG` (`stages.py:40`) is produced, not typed.

## 3. Item-by-item answers

### 3.1 Spec-to-code fidelity per arm

- **THR** (`thr.py`): matches 3.1 step by step (V_m = 0; C_k uses E[k] and S_{k+1}; V_j =
  min(H[j-1], C_j); w_lo; 2,001 grid; T1 tie by adding tau; clipping at w_lo; all N rows; only
  (k, W_k)). No handicap found. The 4,001-point refinement of 7(c) is not implemented (not
  gated).
- **ALP-Z** (`alp.py`): the LP is the spec's (handoff rows, continue rows, initial-state
  objective, breach constant, free variables bounded above only, `highs-ds`). Deviations: no
  `random_seed` (scipy 1.15.3 has none; stated in the docstring); the lattice rounds increments
  to h = (B - w_lo)/400, which is a stated discretisation handicap relative to THR. Run on five
  real routes (CON3-7 r1, CON3-6 r1, SCA3-2 r3 and r0, CON3-0 r1; N = 10^5): LP status 0 in 0.2-0.4
  s, LP objective equals the lattice-DP initial value to 1e-13 (B-ALPZ b), the vertex equals V*
  at every lattice node (off-path share 0.0), deployed decisions agree with the DP on
  0.99869-0.99998 of dev-visited states (gate d is pooled, needs 0.999; CON3-7 r1 alone is 0.99869
  so the pooled margin depends on the mix; no exceptions at 1e-6 tolerance except one state). No
  hidden handicap beyond the lattice.
- **RLSM** (`rlsm.py`): per-stage, per-fold N(0,1) layer, LeakyReLU 0.5, ridge lambda = ridge x n,
  clipping, constant fallback, A's folds and cached features `mlp_batch` (15 inputs). It sees the
  MLP summary, not FULL's tokens; spec 3.3 says so (1.1's "see FULL's information" is looser than
  3.3). Real-size check (CON3-6 r1, N = 10^5): B-RLSM passes 22/22 at all four tested
  (K, ridge) pairs, max normal-equation residual 3e-16.
- **DOS** (`dos.py`): architecture, loss, warm and cold start, fold-mean decision at 0.5 as
  specified. Under-training and the gate: see M1.
- **DQN** (`dqn.py`): continue-forced behaviour, transitions `(row, k, c, done)` with c = E[k] on
  breach at k+1, V' = H[k] if Q_online > H[k] else Q_target (indices checked: `H_t[ks]` is 0-based
  H[k]), Huber on the E0 scale, hard target by stored transitions, learn start 1,000, replay 50,000,
  per-update 32, checkpoints at update 0, 1/8 pass and pass end, greedy selection on the last 20%
  rows (ties earlier), per-pass selection cumulative (so a 3-pass run contains the 1- and 2-pass
  runs). No future information (features at stage k read columns 1..k; the next state of a
  transition is the next decision epoch, which is the environment). Mini run on a real route (CON3-7
  r1, m = 19, N = 4,000, CPU): transitions per stage equal `stage_counts` exactly (shapes (18,)),
  pack/unpack predictions equal. No handicap found beyond the spec's replay window (the 50,000
  buffer holds only the latest ~3,000 paths).
- **PPO** (`ppo.py`): one episode per update, 4 epochs, clip 0.2, no advantage normalisation, no
  entropy, return minus value, rewards in E0 units (handoff -H[k-1], breach at k+1 -E[k]),
  initial handoff probability 0.05 through the bias, GRU over the pooled embedding, masked softmax
  (handoff impossible at m), greedy deployment pi > 0.5. Rollout feature gather is causal. No
  hidden handicap found; initial bias is a stated favour. See M6 for evaluation cost.
- **Seeds and streams.** Every fit reads `route_pool(..., "train", ...)` only; learned arms use
  the first 80% of rows for fitting and the last 20% for in-job selection (A's split). Seeds are
  crc32 keys (`inst|route|N|rep|arm`, plus fold/k or stream). Ties: T1 by `+tau` for THR and
  ALP-Z (checked `GridPredictor.predict`), strict `>` for the cost predictors, S = H + E0(pi-0.5)
  for the surrogates.
- **FULL and baton_eq favoured?** No: `b_full` uses A's `fit_neural` with A's default
  `StageConfig()`, `b_ladder` uses A's `classical_ladder`; neither is tuned.

### 3.2 Leakage and stream discipline

Every site that accepts a purpose was checked (`grep` over `transvrp/`). `_stream` calls
`_require_gate` before any seed is computed, also on the on-disk cache path
(`paths.py:250, 284-300`), and a token must carry `stream == purpose`. B modules request only
`dev` and `train` (`fidelity.ALLOWED_PURPOSES`, enforced in `DevSource.pool`; `require_dev` rejects
non-dev rows and any `.test` file in the rows directory); `expb` has no `test`/`test_std`
request. `check_stream_for_stage` is applied in `check_test_gate` and `run_eval_job`; tuning,
pilot and smoke stages are `test_allowed=False` (refused even though their `test_stream` is
test_b); A's stages refuse `--eval test_b`, B's refuse `--eval test`; models are never evaluated
under another stage's gate (owner check). `analysis_b` and `merge` read only CSVs. The
independence-world paths permute after the gate and by purpose key. Residual: `test_std` is
ungated by A's design (`paths.py` 11-13): `route_pool(inst, "test_std")` returns the first 2,000
rows of A's test stream with no gate. B never requests it, but a one-line deny in
`route_pool` for B's code would close it; recorded, not required. test_b reads are possible by
design at any time after L through `route_pool(..., "test_b", frozen=F, stage=...)`, which is the
spec's gate, not a leak.

### 3.3 Gate soundness, 10-step walk in a clone

Real code, real eligibility (186 jobs per stage), a clone of the repository at HEAD with
CRLF checkout: fabricated payloads for `b_thr`, `b_alpz`, `b_full`, `b_ladder` (one real THR fit
copied under the 186 keys), `sweep manifest`-equivalent `write_manifest`, `fidelity_dev.md` with
the derivation block, commit M, `gatecheck family-line --commit M`, FROZEN + summary + FAMILY lines
appended, commit L, `verify_stage` per stage, no mocks. Result: all four main stages pass at L
(10.4 s per `verify_stage` call under load; `run_eval_job` calls the gate twice per job, once
explicitly and once inside `route_pool`, so about 20 s of git per eval job; cache the verdict per
process and per (stage, HEAD)). Where spec and code disagree:

1. Results directory: C1. Code (`expB`) is right; the sweep default and spec commands are wrong.
2. Reuse stages: C2. The spec text (items 5, 6, 8) is not executable for A's files.
3. Repair (step 5): `run_fit_job` returns "skip" when the model file exists (`sweep.py:468`) and
   `run_eval_job` skips existing row files, so a refit at F2 without deleting the stage's
   `models/<stage>` and `rows/` silently keeps old models and stale dev CSVs; the gate fails
   closed on the code check, but the dev rows and `fidelity_dev.md` would be computed from stale
   files. Add "delete the stage's models and rows" to step 5.
4. The `-G` pin pattern needed git's extended regex; it matched (exactly one commit, L), so
   item 2 works as specified.
5. FAMILY semantics: `derive_family` puts every non-family arm in `descriptive` and every failing
   gate in `dropped`, including P1 as `arm:p1`; the spec (section 7 consequences) says a P1
   failure is not "dropped". An arm can therefore sit in both lists. `analysis_b` handles it
   (the dropped reason wins), the gate accepts it; align the spec wording.

No deadlock found after C1/C2 are fixed. The per-stage rule `git diff --quiet F_s HEAD` and the
gate's normalised hashes agree.

### 3.4 Fidelity gates (M22) against section 7

- B0, B-ALPZ (a)-(d), B-RLSM, B-DQN, B-PPO, P1: as written; thresholds are the section 0 values
  (`fidelity.py:41-55`). FAMILY derivation (`derive_family`): family = arms passing every gate,
  as spec; the lists are rendered mechanically (the block can still be hand-edited before M; the
  gate only pins the file hash, as the spec says).
- Defect-of-ours failures: B-DOS (M1), B-THR (M5). B-RLSM passes on real data; B-ALPZ passes on
  real data. B-DQN counts: exact equality for P in {1, 2, 3} (any P accepted, not the chosen
  one; minor). B-PPO "zero actions at k = m" is vacuous (the dict has no key m); cached vs
  recomputed compares one shared predictor with fresh ones, so it does test the cache, but the
  deployed predictors after unpack do not use the cache (M6).
- P1: the SE assumes the three reps are independent (they share dev days), which makes it too
  small and so can only over-drop; effect small. Not blocking.
- Real-sized layouts: DQN `transitions_per_stage` length m-1 equal to `stage_counts` (checked at
  m = 19); PPO key set 1..m-1; RLSM and DOS diag keys (`n_val`, `val_mse`, `const_val_mse`,
  `never_val`, `val_loss_per_ckpt`) present on a real fit.

### 3.5 Sweep integration (M21)

- `b_smoke` and `b_pilot` job lists are as specified (CON3-0; smoke = 1 route; pilot = eligible
  routes of CON3-0, reps 0-1, N = 10^4, timing cells at route 3 for the six arms).
  `b_tune_<arm>_c<i>` carries `baton_eq` for scoring and `pass_curve = 3` for DQN and PPO;
  `core_of_stage` resolves `tune_*`, `_ind`, `_reuse`.
- Independence world: spec 3.8 says 12 subset routes from 10 instances, rep 0 (not 20); code:
  `routes="subset"` (12), `reps=(0,)`, `IND_INSTANCES = 10`. Permutation is one seeded
  permutation per column applied identically to p and d (g marginals preserved exactly),
  separate keys for train and each evaluation pool. The spec's check "`b_thr_ind` tables equal
  `b_thr` tables (sha256)" cannot hold: the grid bounds use `min_k W_k`, a joint quantity that
  changes under permutation (`thr._grid_bounds`, shared by ALP-Z's lattice); it is not
  implemented either. Reword as "decisions agree within the interpolation tolerance".
- Reuse filter in `run_eval` is correct (planned keys), the gate is not (C2).
- DQN GPU: `CUBLAS_WORKSPACE_CONFIG` is set by `setdefault` at `dqn.py` import, which precedes CUDA
  initialisation because `sweep` imports the arm modules at import time; the evaluated object is
  the hashed file on CPU (`QPredictor` uses the net's device after unpack), threads set to 1 in
  workers. M3 for process count; M6 for PPO evaluation time.
- Memory at 12 CPU workers: PPO about 1.4 GB per worker (tables 1.14 GB at m = 19 plus cache),
  about 17 GB total: fine on 64 GB; the GPU process is the problem (M3).
- Fit payloads carry `libraries`; `peak_rss_mb` is never recorded (M7).

### 3.6 analysis_b against section 8

Holm over F (`A.holm`), the Holm-level CI (`1 - 0.05/(2|F|)` quantile), CI95, CI90, WIN/LOSE
with Holm p and the 13/20 count, TIE_STRONG on the unadjusted CI90 inside (-delta, delta),
RE-EVAL with A's unadjusted `go_no_go`, refusal on non-test_b rows, wrong `family_sha256`, mixed
eval commits, a fit commit different from the FROZEN commit, wrong day counts, missing coverage
(20 instances / 62 routes / 3 reps), arms listed but without rows: all present and as specified.
Gaps: M4 (code identity), M7 (missing reports and the budget-limited label), constants for the
cuts (M2). The ANALYSIS PATCH acceptance is licensed by the spec (see M4) but under-enforced.

### 3.7 Anything that lets the author choose after seeing test_b

- ANALYSIS PATCH (M4): arbitrary statistics change after the data is open, with only a log line;
  mitigated by reporting every patch (spec) and made worse by the missing `check_clean`.
- Deleting and re-evaluating test_b rows is allowed by the spec ("rows deleted and re-evaluated");
  evaluation is deterministic on CPU so it changes nothing, but nothing logs how many test_b
  evaluations ran.
- The family lists and stage membership cannot change after L (pin, item 2); no second
  configuration of an arm can enter, because a stage needs a FROZEN line. Chosen configurations
  are typed into `stages.py` (M8) before F, which is dev-only.
- `fidelity_dev.md` can be edited by hand before M; only its hash is pinned. Stated in the spec.

## 4. Required changes (numbered)

1. **R1** `expa/sweep.py:76, 783, 801` (and `gatecheck.model_dir`): make `results/expB` the default
   results directory for every B stage (or add `--results-dir` to every command in spec 5.3 and to
   `sweep manifest`, `sweep.py:797`); add a test that a B stage fit writes where `gatecheck.model_dir`
   reads.
2. **R2** `expb/gatecheck.py:157, 345-349, 386-399, 440-455` and `stages.py:122-136`: make
   `model_dir`, `write_manifest`, `check_models` and `family_report` honour `reuse_models` and the
   planned-key filter; replace the library check for reuse stages by a pinned record of cut (i)
   condition 3; add a temporary-repo test that a reuse stage passes the gate with A-style payloads.
3. **R3** `expb/fidelity.py:53, 430-449` and `expb/dos.py:39-40, 71-74` (spec T7/3.4): a tolerance
   and step floor under which a correct DOS passes B-DOS on real data; show the rate on 3+ real
   routes before sign-off; amend the spec.
4. **R4** `expb/fidelity.py:295-314` (spec T4): fix the B-THR (a) false-drop rate (per route on
   the rep mean, or 4 SE); amend the spec.
5. **R5** `expb/analysis_b.py:527-560`: call `check_clean`, require the executed analysis files
   equal the HEAD blobs, and require each ANALYSIS PATCH line to come from one commit descending from L.
6. **R6** `expb/fidelity.py:574-580, 653-658`, `gatecheck.py:52-54`, `analysis_b.py:40-41, 173-175`,
   `stages.py`: implement cut (ii) (spelling without a space, skip path, membership check) and make
   cut (iv)/(iii)/(v) one pre-committed parameter set; test the cut (ii) line end to end.
7. **R7** `expa/sweep.py:609-615` (+ spec 5.3 step 3): `jobs = 1` for a stage whose arm resolves
   to CUDA.
8. **R8** `expb/codecs.py:98-106`: one `PolicyPredictor` per net in `unpack_models`; test that a
   round trip uses the cache (call count) and keeps predictions equal.
9. **R9** `expb/analysis_b.py` (before M): D_a, ALP-Z vs THR on test_b, strata, between-replicate SD,
   the budget-limited label in `arm_claim`, THR cost-over-DP line; record `peak_rss_mb` in every
   `budget`.
10. **R10** `expb/fidelity.py`/`stages.py`: `choose_config` implementing 6.1 with tests; have
    `tuning_scores` handle non-curve arms.
11. **R11** (minor) cache the gate verdict per process and (stage, HEAD) (`paths.py` `verify_test_b_gate`,
    `sweep.py` `run_eval_job`, two calls per job today); step 5 of the sequence: delete the repaired
    stage's models and rows before refit; reword the 3.8 sha-equality check; align `dropped` vs P1
    wording; implement or drop the 4,001-point THR refinement; B-DQN should check the chosen P.
