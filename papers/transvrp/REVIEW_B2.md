# Round-2 review of EXPERIMENT_B.md (revision 2)

Reviewer: same independent reviewer as REVIEW_B.md, 2026-11-18. Read: EXPERIMENT_B.md rev 2 (all),
the response table in REVIEW_B.md, METHODS_ALP_PPO.md, and the real code `code/transvrp/expa/paths.py`
(55-170) and `sweep.py` (275-286, 378-452), plus `.gitignore` and the log's FROZEN lines. Spot-checked
the Zhu PDF (PDF pp. 8-11 = journal pp. 273-276: eqs 14-27 incl. 23.2, 24, 26). Nothing was run or edited
except this file. `expb/` does not exist yet, so gate findings are against the spec text plus the A code it modifies.

## Verdict: ACCEPT WITH REQUIRED CHANGES

The rewrite fixes every round-1 critical item in substance. The new gate is a real improvement and I
found no way to make the 10-step sequence deadlock on its main path. What remains: four ways an author
could still alter the family or the analysis after seeing test_b (all detectable only after the fact),
two sequence gaps that would bite in practice, one substantive problem in ALP-Z's deployment rule, and
a PPO specification conflict. None needs a redesign.

## 1. Round-1 findings: status

| R1 finding | status | evidence |
|---|---|---|
| 1a gate self-contradiction | **resolved** | 5.3: log read from `git show HEAD:` blob; ancestor + diff replaces HEAD equality; F / F2 / L roles are consistent with A's code (A's `tree_is_clean` only covers `code/`, paths.py:94-122). |
| 1b editable log | **resolved** (see N1 for the residual) | checks 1-2. |
| 1c substring bypass | **resolved** | anchored regex replaces paths.py:152 `frozen not in read_text()`; A's lines at RESEARCH_LOG.md:131,174 do have the exact form (verified). |
| 1d literal "test" checks | **resolved on paper** | exact edit list covers paths.py 45/126-167/214-227/313/331 and sweep.py 76/275-285/435-445/543; I confirmed lines 280, 435, 441, 444 are the ones named. Mutation per edit. |
| 1e nothing binds family to stages / fidelity | **partly** | stages in FROZEN lines, arm in a list, fidelity hash: yes. But the lists themselves are free text (N1), and models are not hashed (N2). |
| 1f different code across commits | **resolved with caveat** | per-stage diff on SHARED+OWN; but SHARED is an allow-list (N3) and step 8's check is not per-stage (N5). |
| 2 DQN behaviour policy | **resolved** | see section 3. |
| 2 no convergence evidence | **resolved** | 3-pass curves survive every cut; label rule pre-registered (caveats in section 3). |
| 2 PPO defaults | **resolved**, one new conflict | Kadyrov settings verified against METHODS_ALP_PPO §2.3; n_env vs one-episode-per-update is unspecified (N6). |
| 3 compute | **partly** | grids tied, tables built once, cuts reordered, trigger stated, repairs from headroom, GPU rule: all resolved. Table arithmetic reproduces (83.05 / 134.85) but has two inconsistencies (section 4). |
| 4 cut (i) | **resolved** | all five conditions plus the dev-row identity check. |
| 5 independence / ALP | **partly** | independence world added; ALP-Z redesigned from the source (good), but its deploy rule is flawed (N4) and the independence evaluation of THR/ALP-Z/none/oracle has no stage in the gate (N5). |
| 6, 7, 8 | **resolved** | RLSM ridge axis, DOS note, surrogate note; dropped arms descriptive; Holm-level CI; weak-tie wording; bug gates vs P1; THR 99%. |
| 9 code mismatches | **resolved on paper** | `_pack/_unpack` hooks, four-value branch, per-layer masks, `PaddedTables`, no `write_summary`, ladder labels: all addressed; unverifiable until code exists. |

## 2. The new gate, adversarially

Works as intended: A's stream cannot be unlocked by a test_b line (anchored regex); check 5 forces both the
FAMILY line and the stage's FROZEN line to postdate F; check 6 includes paths.py and sweep.py, so any later
edit of the gate or the sweep invalidates every stage (strong); checks 1-2 close the working-tree log hole.
I tried to deadlock the sequence: reuse stages with F = c37ad7c (c37ad7c's blob has no test_b line: passes),
repair at F2, step 7 ordering, step 9 skip-if-exists: all consistent. Remaining holes and gaps:

- **N1 [major] Family line is not pinned after L.** The gate demands exactly one FAMILY line in HEAD's
  blob, not the one that existed when test_b was first read. After seeing test_b, the author can commit
  L2 with an edited `family=`/`descriptive=`/`dropped=` list; every check still passes, and `analysis_b`
  reads the lists from HEAD. Nothing records which FAMILY line a test_b row was evaluated under. Also,
  nothing checks that the lists equal what `fidelity_dev.md` implies (P1 outcomes, bug-gate drops, the
  distinctness rule); the author decides after seeing dev results and the A priors.
  Fix: `fidelity.py` emits a machine-readable block (arm: gate results, family/descriptive/dropped as
  computed); the gate verifies the FAMILY lists equal that block; every test_b row records
  `sha256(FAMILY line)` and `analysis_b` refuses rows whose recorded hash differs from HEAD's line.
- **N2 [major] Models are unhashed, and `results/**/models/` is gitignored.** Only reuse stages have a
  model manifest. A model pickle's `fit_commit` and `fit_tree_clean` are self-reported fields; a
  hand-edited or later-refitted pickle passes. Fix: a manifest of sha256 for every model file of every
  stage, committed in L, its hash in the FAMILY line, verified per file by the gate (as for reuse).
- **N3 [major] SHARED is an allow-list.** `expa/pilot.py`, `__init__.py`, any new `expb` helper, and ignored
  files (`*.py[codz]` is gitignored; a source-less `.pyc` is importable, and `git status` hides it)
  can change behaviour without entering the diff. Fix: default-deny: diff set = everything under
  `code/transvrp/` except an explicit exclusion list (analysis.py, pilot.py, fidelity.py, analysis_b.py), and
  add `git status --ignored` for `*.py*` under `code/transvrp`. Log torch/scipy/HiGHS versions in payloads.
- **N3b [major] Analysis modules are outside the gate.** Their post-family change rests on "sha256 logged
  before use", which nothing verifies; `analysis_b` determines what is reported (eligibility, aggregation,
  branch rules). Fix: put sha256 of `analysis_b.py`, `fidelity.py` and the `expa/analysis.py` helpers
  it imports into the FAMILY line; `analysis_b` refuses a mismatch unless an appended, dated patch line
  names the new hash and the reason.
- **N5 [minor, practical] Sequence gaps.** (a) Step 8's `git diff --name-only $F L` must be per stage
  (F2 legitimately changes the repaired arm's OWN file, which is in the diff of F..L for other stages' F).
  (b) THR / ALP-Z / none / oracle on the permuted test_b (3.8) need stages (e.g. `b_ladder_ind`,
  `b_thr_ind`, `b_alpz_ind`, or an eval flag that is part of the stage name); only `b_full_ind` is in S, so
  check 3 would refuse them. (c) The "one common eval_commit" rule in `analysis_b` breaks if any commit is
  made between step-9 stages and eval rows are skipped on rerun; say "no commits between step 9 evals" or
  key on the log-blob hash instead. (d) The regexes: normalise CRLF (Windows checkout; git already warns
  LF to CRLF on this repo) and restrict `(.*)` to `[a-z0-9_:, ]*` so a list cannot contain the delimiters.

Not a hole: gate is in-process and a deliberate bypass (editing code to skip it) remains possible; the
spec says so and, since paths.py is in SHARED, the edit is at least visible in history.

## 3. ALP-Z, DQN, PPO

**ALP-Z vs the PDF.** Verified on journal pp. 274-276: 23.2 has theta_{l,0,q} and alpha_{l,j,q} indexed
by q; 24.1 maximises theta_{0,0,Q} only; 26 constrains every feasible q for each sampled (l, j, R, u).
On a fixed route (l = stop k, R fixed by k) the basis is a free value per (stage, load): exact, so
ALP-Z equals the exact independent-demand DP on the lattice, and THR is the same object up to
discretisation. The spec's reading is correct.

**N4 [major] The deploy rule `hand off iff L_k(w) > H[k-1]` ("ties have probability 0") is wrong for an
LP vertex.** With the initial-state objective, at a handoff-optimal state f_k(i) = H needs only
L_k(i) >= H, with slack unconstrained and no lower bound on f. Basic solutions are generically tight
(L = H, in floating point either side of H) exactly on the handoff-optimal states the policy visits.
There, "L > H" gives "continue", the reverse of the exact decision. So ALP-Z will deviate from THR on
tie states because of the vertex, not because of a method difference, and will likely continue too
often. Consequences: (i) the distinctness rule (>= 1% of decisions differ) can be triggered by this
artefact plus the 401-point lattice vs 2,001-point grid and binned pmfs, so it does not measure what
it is for; (ii) no gate checks the deployed decisions (B-ALPZ checks objective values and the uniform
variant only). Fix: either deploy with `L >= H - tol` (tol pre-registered, e.g. 1e-6 E0) or add a
lexicographic tiny uniform weight to the objective so the maximal vertex (V*) is returned, and add a
bug gate: deployed ALP-Z decisions equal the lattice-DP argmin except within tol. Judge the 1% rule on
the repaired rule, and note that including ALP-Z unconditionally costs only a Holm factor 5/6;
I would prefer unconditional inclusion with the "coincides with THR" sentence if decisions agree
to > 99% (it keeps the family fixed by rule rather than by a threshold whose value is contaminated).

**DQN continue-forced.** Off-policy correct: the target uses the known H[k] and one continue transition;
handoff is terminal with known price, so Q(handoff) is not learned and nothing is missing. The behaviour
state distribution at each stage is the alive-path distribution, a superset of what the deployed policy
visits, so support is covered (same as LS rows). The per-stage count gate is exact. Residual: shared net
across stages with hard target period 8,000 transitions: about 125 target updates per pass, enough to
propagate 16 stages, but the 3-pass curve is the only evidence; fine. Indexing (c = E[k] on breach at
k+1, V' = H[k]) agrees with simulate.py.

**PPO.** The 5% initial handoff probability gives 0.95^k reach (44% at k = 16): fixes starvation.
**N6 [minor]** 3.6 says one episode per update, 4 epochs, but M19/section 9 use n_env = 64 lockstep
rollouts; 63 of 64 sequential single-episode updates then act on data from a stale policy. Specify
which (e.g. collect 64 episodes, update on them one at a time with ratios against the stored
log-probs, and say so) or use n_env = 1 with the cost accounted. Also say the gamma = 0.5 grid point is
myopic on a problem whose breach cost lies up to 16 steps ahead; it is in the grid only as the source's
value and will probably not be chosen.

**Pass-count rule and label.** Reasonable. Pass-end checkpoints of one 3-pass run equal 1- and 2-pass runs
only because lr and target period are constant (true here: no schedule; say it). Tie window 0.10 vs
label margin 0.5: coherent. Label uses rep 0 and 12 routes only, and detects only "still improving
between pass 2 and 3"; so wording must be "not labelled budget-limited", never "converged". "Budget-limited" itself is an
appropriate, honest label.

## 4. Independence world and compute

Independence world: sound as descriptive (marginals unchanged by column permutation, so THR/ALP-Z models
need no refit; FULL refit on permuted training rows isolates approximation loss). It reads test_b but
decides nothing. Gate gap in N5(b). One wording risk: "THR loses x% of which y% is mis-specification" needs
the DP value under the true copula to be unavailable; use only realised-vs-own-model gap, as stated.

Compute: totals reproduce (83.05 h and 134.85 h). Two inconsistencies: (1) DQN main wall 24.8 / 43.4 h
is 8 / 14 min per job whereas the job-hours rows (28 / 47) are 9 / 15 min; on a serial GPU wall equals
job-hours unless table building (5 min) runs on CPU workers in parallel, which is not stated. (2) The
DQN tuning row (8.2 h) implies the padded tables are built once per route and shared by the 4 configs,
but each config is a separate stage/process; either cache tables on disk or add about 4 h (48 x 5 min),
which moves P = 3 to about 139 h > 134.4 h and triggers cut (i) (125 h still fits). With a 16 GB CUDA GPU
running 2-3 DQN processes concurrently (0.9 GB tables each) would cut the serial GPU term; not needed
if the above holds. Cut order (reuse, drop DOS, 6 tuning routes, reps 3 to 2, P = 1 last) is now least-harm-first
and the trigger is mechanical: accepted.

## 5. Can the author still pick arms after seeing test_b?

Not by rerunning the gate: stages, F, fidelity file, and arm membership are pinned. Yes, by (a) editing
the FAMILY line again at a later commit (N1), (b) swapping a model file (N2), (c) changing analysis code
(N3b), (d) classifying arms into family/descriptive/dropped by hand from dev output (N1, no mechanical
link to the fidelity report). Items (a)-(c) leave a git trail; the paper should still say so. With N1-N3b
fixed, the only remaining discretion is before the family line, on dev data, plus the A priors already
disclosed.

## Remaining required changes

1. (N1) Machine-readable family derivation in `fidelity_dev.md`; gate checks the FAMILY lists equal it;
   test_b rows carry sha256(FAMILY line); `analysis_b` refuses a mismatch.
2. (N2) Model-file sha256 manifest for every stage, committed in L, hash in the FAMILY line, verified by the gate.
3. (N3, N3b) Default-deny diff set (exclusion list instead of allow-list), `--ignored` check, pin sha256 of
   `analysis_b.py`, `fidelity.py` and imported analysis helpers in the FAMILY line.
4. (N4) Fix ALP-Z deployment ties (tolerance or maximal-vertex regularisation), add a deployed-decision
   gate against the lattice DP, correct the "ties have probability 0" sentence, and restate or drop the 1%
   distinctness rule (preferably include ALP-Z unconditionally).
5. (N5) Per-stage diff in step 8; define gated stages for the independence-world evaluations of
   ladder/THR/ALP-Z; no commits between step-9 evals; CRLF normalisation and tightened regex classes.
6. (N6) Resolve n_env = 64 vs one-episode-per-update for PPO; note gamma = 0.5 myopia; write "not labelled
   budget-limited" rather than "converged".
7. Reconcile the DQN wall vs job-hour rows and the tuning-table sharing (cache tables or add about 4 h) and
   restate the trigger outcome (cut (i) expected).

---

## Author response, round 2 (EXPERIMENT_B.md revision 3, 2026-11-18)

Added by the spec author; the review text above is unchanged. Section numbers refer to
EXPERIMENT_B.md revision 3.

| item | resolution | where |
|---|---|---|
| 1 (N1) family not pinned; lists hand-made | `fidelity.py` writes a `family-derivation` JSON block; the gate requires the FAMILY lists to equal it. FAMILY and FROZEN lines must be added in exactly one commit and never edited (`git log -G` returns one commit; L's line equals HEAD's byte for byte). Every test_b row records `family_sha256`; `analysis_b` refuses a mismatch. | 5.3 items 2, 7; analysis_b |
| 2 (N2) models unhashed | Per-stage manifest `results/expB/<stage>/models.sha256` (raw-byte sha256, fit commit, clean flag), committed in M **before** the FAMILY line, combined into `manifests_sha256`; the gate hashes every model file and checks the count. Reuse stages use the same mechanism. | 5.3 definitions, item 6, steps 4, 7 |
| 3 (N3) allow-list, `.pyc`, versions | Default-deny: CODE = every git-tracked file under `code/transvrp/`; only other arms' own modules are excluded per stage. Hashes are of git blobs, CRLF normalised (`tree_hash`). `git status --ignored` refuses stray `.py/.pyd/.so/.pth`, `.pyc` outside `__pycache__`, and orphan bytecode. python/numpy/scipy/torch versions recorded in payloads and rows and checked at eval. | 5.3 item 1, 5, 8 |
| 3 (N3b) analysis modules outside the gate | `analysis.py`, `analysis_b.py`, `fidelity.py` are inside CODE (`code_sha256`) and also pinned separately (`analysis_sha256`); a later analysis change needs a pinned, dated `TEST_B ANALYSIS PATCH` line with the new hash and reason. | 5.3 |
| 4 (N4) ALP-Z ties | Deploy rule "hand off iff L ≥ H − 1e-6·E0" for ALP-Z and THR alike (ties to the preventive action, as Zhu eq. 29.1); the "ties have probability 0" sentence is corrected and explained. New bug gate: deployed decisions equal the lattice-DP decisions on ≥ 99.9% of visited dev states except within the tolerance; tie unit test with the strict rule as a mutation. The 1% distinctness rule is dropped; ALP-Z is in F unconditionally (subject to the same gates as all arms); agreement with THR is reported descriptively. | 3.1, 3.2, 1.3, 1.4, 7, M14 |
| 5 (N5a) step-8 diff not per stage | Step 6 is a per-stage diff with the other arms' own modules excluded; repeated against L. | 5.3 steps 6, 9 |
| 5 (N5b) independence evaluations ungated | Stages `b_ladder_ind`, `b_thr_ind`, `b_alpz_ind`, `b_full_ind` (12 subset routes, rep 0) are fitted at F, named in the FROZEN line and in the FAMILY line's `independence=[...]`. THR/ALP-Z `_ind` models must hash-equal the main ones. | 3.8, 5.3 |
| 5 (N5c) one eval commit | "No commit between the first and last test_b evaluation"; otherwise all test_b rows are re-evaluated. | 5.3 step 10, analysis_b |
| 5 (N5d) CRLF, regex classes | `norm` before every text hash; list classes `[a-z0-9_,]*` / `[a-z0-9_:,]*`. | 5.3 |
| 6 (N6) PPO n_env vs one episode per update | n_env = 1 as in Kadyrov et al.: roll out one path with the current policy, update on it (4 epochs), then the next; cost re-estimated. γ = 0.5 noted as myopic and in the grid only as the source value. Wording "not labelled budget-limited", never "converged"; the pass-end equivalence (no schedules) stated and tested. | 3.6, 6.3, header, 9.1 |
| 7 compute | DQN rows equal job-hours on the serial GPU (9 / 15 min per job); tuning tables cached once for the 12 routes; PPO at n_env = 1. Totals ≈ 89 h (P = 1) and ≈ 144 h (P = 3), reconciled line by line with the reviewer's ≈ 139 h. **Cut (i) is expected to apply** (→ 133.6 h, 0.8 h below the trigger; a slower pilot also triggers cut (ii)). | 9.1 |
| platform | DQN on RTX 5000 Ada 16 GB (CUDA verified, torch 2.6.0+cu124); all other arms and all evaluations on 12 CPU workers. CUDA training need not be bitwise reproducible: replicates are seed-level as in A, models are hashed, evaluation is on the CPU and deterministic. Pilot Q2 and stop rule 5 adjusted. | header, 9, 9.3, 12 |
| sign-off | New section 0, "Design thresholds needing author sign-off" (T1-T15). | 0 |
