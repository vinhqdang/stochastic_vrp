# Code review: experiment B modules M12 (test_b stream and gate) and M13 (THR, grid)

Reviewer: independent pass, 2026-11-18. Spec: EXPERIMENT_B.md revision 3.1, sections 3.1, 5.1-5.3, 7, 10.
Scope: `git diff -- papers/transvrp/code` (expa/paths.py, expa/sweep.py, three A tests) plus the new
`expb/{gatecheck,grid,thr,__init__}.py`, `tests/test_expb_{gate,thr}.py`, `code/MUTATION_B.md`.
Nothing was modified except this file. Throwaway scripts were kept in the scratchpad; the three
mutations I re-ran were reverted and `git diff --stat` / `git status --short` are identical to before.

Test run: `test_expb_gate`, `test_expb_thr`, `test_expa_gate_tree`, `test_expa_sweep`, `test_expa_measure`:
99 passed in 207 s.

## 1. A's gate is not weakened

The three A fixtures changed from `frozen <hash>` to `FROZEN COMMIT: <hash>` (own line). The only
code change to A's rule is substring-in-text -> `^FROZEN COMMIT: ([0-9a-f]{40})\r?$` (MULTILINE) with the
group equal to `frozen`. Every string accepted by the new rule contains the old substring, so the
accepted set only shrinks. HEAD equality, clean-tree check and 40-hex check are untouched.

Throwaway experiment (`verify_test_gate`, log/HEAD/tree state patched as A's own tests do):

| case | result |
|---|---|
| anchored line | accept |
| anchored line, CRLF file | accept |
| hash only in prose; `see FROZEN COMMIT: <h>`; indented line; trailing space | refuse |
| `FROZEN COMMIT (test_b): <h> stages=[..]` line | refuse |
| dirty tree | refuse |
| HEAD != hash | refuse |
| uppercase hash | refuse |
| REAL `RESEARCH_LOG.md` with `c37ad7c4685714b0991b5c4b0f857ee00c0ddb55` (line 131) and `9f2a29bf...` (line 174) | accept (parses) |

Cross-stream: a `stream="test"` token is refused for `test_b`, a `stream="test_b"` token is refused for
`test`. `verify_test_b_gate` on the real repo refuses (dirty tree), and its prefix `FROZEN COMMIT (test_b)`
cannot match A's `FROZEN COMMIT:` and vice versa. Nit: a hash followed by trailing space is now refused
by A (stricter, harmless; A's real lines are clean).

## 2. Bypass attempts on the test_b gate

Built on the test module's own temporary-repo fixture (`build_project`), attacks run through `verify_stage`.

Refused (good): stage case/whitespace/newline variants (`B_THR`, ` b_thr`, `b_thr\n`: not in any FROZEN
line, and stage reaches no path before the list match); a FROZEN line (also behind a form feed) below
the FAMILY line; CRLF-converted log (uncommitted: dirty; committed: second pinning commit); a later
commit whose log text mentions the pinned phrases. git calls use argument lists (no shell). `fit_commit`
is compared with a 40-hex value read from the pinned log before any path or git use. A git failure
in `check=True` calls raises `GateRefused`; `check=False` is used only for `log_text`, where an empty
result can only cause refusal (0 FAMILY lines) except in `check_fit_blob`, which is guarded by the
preceding ancestor check. Exceptions are never converted to acceptance (`TimeoutExpired` and
`FileNotFoundError` propagate). Edited FAMILY line in a second commit, two FAMILY lines, two FROZEN lines,
working-tree-only log, orphan/source-less `.pyc`: covered by the suite and pass.

Accepted when they should not be (findings):

- F1 (medium). **Stray model files.** An extra `thr__EVIL__...pkl` placed in the stage's model directory
  (git-ignored, so invisible to item 1) is accepted. `check_models` iterates the manifest, not the
  directory, while `sweep.run_eval` evaluates `models_dir.glob("*.pkl")` (sweep.py:520). `check_libraries`
  also reads `sorted(glob)[0]`, which may be that stray file. The spec (item 6) says "every model file of the
  stage ... file count equals the planned job count"; the intent is a closed set. Fix: assert the set of
  `*.pkl` names in the directory equals the manifest names.
- F2 (medium). **Pin pattern not anchored.** `PIN_PATTERN` is `TEST_B FAMILY CLOSED|FROZEN COMMIT [(]test_b[)]`;
  the spec has `^(...)`. Verified: after a correct pin, a later log commit whose prose or ANALYSIS PATCH
  `reason=` merely mentions either phrase is refused as a "second pinning commit" (a clean PATCH line is
  accepted). The same would happen if any pre-L commit of the log (planning notes) quotes the phrase.
  Fails closed but can lock the author out of test_b for good. Fix: add the `^` anchor as in the spec.
- F3 (low-medium). **Edited tracked file hidden by index flags.** With `git update-index --skip-worktree`
  (or `--assume-unchanged`) on `thr.py`, a changed working-tree file is accepted: hashes are taken from
  blobs and status reports clean. Deliberate bypass, outside the spec's stated threat model, but one extra
  line closes it: reject any `git ls-files -v -- CODE_REL` entry whose tag is not `H`.
- F4 (low). **Item 1 pathspec is CODE only.** `papers/transvrp/code` is the cwd for every documented
  command, so an untracked `code/numpy.py` / `scipy.py` / `sitecustomize.py` shadows libraries and is accepted
  (verified `numpy.py`). Spec-conformant, but cheap to extend: add `papers/transvrp/code/*.py` (top level) to
  the status pathspec and the source-suffix rule. An untracked `.json`/data file inside CODE is accepted
  (not source; informational).
- F5 (low). **Merge commits.** `git log -G` shows no diff for merge commits, so an evil merge that edits
  the pinned lines is not counted. Use `-m` (or reject non-linear history under the log path).
- F6 (low). **Library check reads only the first payload** and requires exact dict equality; spec item 8 says
  "the stage's payloads". Check all manifested payloads (after F1, by name).
- F7 (low, performance). `run_eval_job` verifies the gate per job and `route_pool` verifies again, each time
  tree-hashing CODE, hashing every model file, and re-planning jobs. Hundreds of jobs per stage make this
  quadratic in model bytes. Verify once per (stage, HEAD) per process and pass the token, or memoise.

## 3. THR (M13)

Correct. Indexing agrees with `simulate_policy` (breach at stop k+1 costs `E[k]`, handoff after k costs
`H[k-1]`, strict `chat > H` with `+tau`); `V_m = 0`; the backward recursion, first-crossing thresholds and
the lower clip match spec 3.1 (hand computation tests, T1/T2/T7/T8 kill). `tau = 1e-6*E0`, grid 2,001, and
`C + tau` giving "hand off iff C > H - tau" (differs from `>=` only on a float-measure-zero set; documented).
The independence assumption is implemented as described by Yang-Mathur-Ballou/Venkateshan-Mathur
(per-stop empirical marginals).

Implementer's interpolation finding: **real but not a threat here.** Reproduced on the 4-atom toy
(max |C_grid - C_exact| on integer loads 0.18 at n=101, 2001, 4001; 0.14 at 10000): linear interpolation of a
step function V (value jumps at B - g) does not converge on a discrete support. Checks on our data:
- Every g column of every dev route in all 20 instances has 5,000 distinct values (min unique = 5,000;
  gamma marginals with a Gaussian copula, `g = p - d`), so V is continuous and the error is O(h^2).
- CON3-0 routes 0-2 (5,000 dev rows): `max|C_2001 - C_20001| / E0` = 2.0e-4, 7e-5, 1.6e-4; threshold shifts of
  4,001 vs 20,001 are below one grid step; C_k monotone gap exactly 0.
- `E[k] >= H[k]` for k < m-1 on all 79 routes, so the monotonicity argument behind B-THR(b) holds
  (a sufficient condition; the gate's tolerance 1e-9*E0 is safe).
So B-THR (spec 7) is not endangered. Residual risk: a future instance set with atoms (e.g. integer demands)
or a zero-demand stop. Optional hardening (state only): (a) `fit_thr` raises/logs when any column has fewer
than ~50 distinct values; (b) for lattice supports build the grid with spacing `h = lattice/q` and anchor it at
B so every `w + g` is a node (the test's `integer_grid`); (c) keep the continuous-support claim in the test
comment as a precondition.

Minor: BLOCK_ELEMS blocks by elements rather than "2,048 grid points" (spec 3.1), results identical,
harmless. Cost at N=1e5 is about 20x the 5,000-row timing (3-4 s/route at 2,001), acceptable.

## 4. Mutation record

Re-ran three, restored, tree identical to before (`git diff --stat`, `git status --short` equal):
- G6 (`excl = []`): `test_repaired_other_arm_is_accepted_for_thr` fails. Killed.
- T5 (`>=` -> `>` in `thresholds`): `test_thresholds_use_the_tie_direction_of_t1` fails. Killed.
- T8 (`x > B` -> `x >= B`): `test_two_stop_route_matches_hand_computation` fails. Killed.
The record is plausible (including the two honest survivors, G8 and T2, and their fixes).
Gaps against the spec's mutation list: no mutants for item 3 (stage in exactly one FROZEN line),
item 4 (ancestor and fit-blob), manifests hash (item 6 first half), fidelity hash (item 7 first half),
default-deny vs allow-list, "hash working-tree instead of blobs", and the "`in TEST_STREAMS` reverted to
`== "test"`" edits in sweep.run_eval_job. The bytecode scan is stated as unmutated; that is a real gap
because it is gate item 1.

## 5. Gaps against the spec

- Gate test (g) (rows carry `family_sha256`; `analysis_b` refusals): the `analysis_b` half belongs to M23 and
  deferral is fine. The producer half is not: `ROW_COLUMNS` has no `family_sha256`, `model_sha256`,
  library versions (spec 5.3 says `run_eval_job` writes them into every test_b row); `eval_commit`/`fit_commit`
  exist. Must land (with a test) before any test_b evaluation.
- Gate item 8 depends on payloads recording `libraries`; `sweep.py` writes no such key, so the gate
  currently always refuses (fail closed, safe) until M21 fit adds it. Add it and a test now.
- Spec step 4 calls `sweep manifest --stage`; the CLI lives in `gatecheck manifest`. Reconcile spec or
  code.
- `run_eval_job` (public) does not itself enforce `eval_set == stage.test_stream`; only `run_eval` ->
  `check_test_gate` does. A direct call with a B stage and `eval_set="test"` reaches A's gate.
  Spec: "B never reads test". Move the match into a shared helper called from both.
- Reuse-stage scope (cut (i)) is implemented (`REUSE_PREFIXES` equals the spec list) but untested, and
  `planned_job_count` for reuse stages needs StageSpecs that do not exist yet. Acceptable to defer to M17/cut (i),
  provided a test is added when that cut is taken; the gate is fail-closed meanwhile.
- `GateToken.__post_init__` message still says "verify_test_gate only" (stale text); `_require_gate` does not
  compare `token.stage` (route_pool issues its own token, so not exploitable).

## Verdict: accept with required changes

Numbered required changes
1. Anchor `PIN_PATTERN` with `^` as in the spec (F2) and add a test for a later prose mention.
2. Make `check_models` compare the directory's `*.pkl` names to the manifest names exactly, and make the library
   check cover all manifested payloads (F1, F6), with tests.
3. Record `libraries` in fit payloads; add `family_sha256`, `model_sha256`, `libraries` columns to test_b
   rows with a test (gaps above); reconcile `sweep manifest` vs `gatecheck manifest`.
4. Enforce stream/stage matching inside `run_eval_job` (shared helper with `check_test_gate`).
5. Add mutants for items 3, 4, manifests/fidelity hashes, bytecode scan, default-deny and blob-vs-working-tree
   hashing; record kills in MUTATION_B.md.
6. Recommended: reject non-`H` entries in `git ls-files -v` (F3), cover top-level `code/*.py` (F4), use
   `git log -m` (F5), and verify the gate once per stage rather than per job (F7).
7. Recommended for THR: add a discrete-support guard (or lattice-aligned grid) and keep the
   continuous-support precondition written next to the gate-B-THR test; no change to thresholds or
   grid size is needed.
