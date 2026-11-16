# TRANSVRP — handoff for a local session (2026-11-16)

This file is for a fresh session that will run the **full experiment A sweep** on
the author's own machine. Read it top to bottom before doing anything.

## 0. First actions

```bash
git clone https://github.com/vinhqdang/stochastic_vrp.git   # or: git pull origin main
cd stochastic_vrp
git log --oneline -3          # expect d64c6ae (pilot report) or later on main
```

Then read, in this order:
1. `CLAUDE.md` (repo rules) and `papers/transvrp/STATUS.md`
2. `papers/transvrp/EXPERIMENT_A.md` — the pre-registered spec (sections 1, 5, 7, 8, 12)
3. `papers/transvrp/RESEARCH_LOG.md` — every decision so far
4. `papers/transvrp/code/results/expA/pilot_report.md` — the pilot outcome
5. `papers/transvrp/CODE_REVIEW_A.md`, `CODE_REVIEW_A2.md` — known gaps

## 1. Where things stand

- Paper 10 (TRANSVRP), direction A: does a set-encoder continuation-value model beat
  BATON's scalar isotonic models once training data is plentiful? Two-action
  {continue, hand off} stopping problem on one route; unit of analysis = instance
  (n = 20); equivalence margin δ = 0.7%; decision rules in spec section 1.
- Code: self-contained in `papers/transvrp/code/` (no imports from `svrpspd_wdro/`).
  Modules `transvrp/expa/` M0-M11, **230 tests green**.
- **Pilot done on the dev stream, P1-P6 all PASS** (CON3-0, 4 routes). Projection for
  the full sweep: **62.8 h ≈ 2.6 days on 4 CPU cores**, so no GPU (spec platform rule:
  ≤ 4 days → CPU). Projection is order-of-magnitude (10^6 cells extrapolated).
- Eligibility frozen on dev: `code/results/expA/eligibility_dev.csv` (80 routes, 62
  eligible, 20 instances).
- **The test stream has never been evaluated by any neural code.** One BATON-only pass
  touched it during design (disclosed in spec 12.1). Keep it that way until step 5.

## 2. Rules for this session

- Git: commit author AND committer `Vinh <dqvinh87@gmail.com>`
  (`git config user.name "Vinh"; git config user.email "dqvinh87@gmail.com"`).
  Push to `main` only, no new branches, no pull requests. **No AI/assistant identity
  anywhere** — no Co-Authored-By or "Generated with" lines, not in files either.
- Model roles (author's choice): latest Opus plans and designs, latest Haiku implements,
  latest Sonnet reviews. Any code change → Haiku with tests that can fail
  (mutation-checked), then an independent Sonnet review.
- The design is frozen (spec section 1, 7.3). Do not tune anything on dev or test
  results. Any deviation goes into `RESEARCH_LOG.md` first, with date (GMT+7) and reason.
- Other papers: BATON, TEMPO, QUASAR, RARECERT, SORTIE are under review and frozen. Do
  not touch `papers/{baton,tempo,quasar,rarecert,sortie}` or `svrpspd_wdro/core|ev`.
- Report times to the author in GMT+7.

## 3. Environment setup

Python 3.11+ (the code was developed on 3.13). From `papers/transvrp/code`:

```bash
python3 -m venv .venv && source .venv/bin/activate      # optional
pip install numpy scipy scikit-learn pandas pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pytest tests -q            # expect 230 passed (~9 min)
```

Pinned versions used so far are in `code/requirements-expA.txt` (torch 2.15.0+cpu,
numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5). Use the same versions
if you can. Whatever you use, **fit every model of the sweep on one machine with one
environment**; do not mix in the pilot models from the cloud session (they are
git-ignored anyway). Log the machine (cores, RAM, OS, versions) in `RESEARCH_LOG.md`.

Resources needed: 4+ cores, ~2 GB RAM per worker (1.8 GB measured at 10^6), and
**about 50-60 GB of free disk for the chunk cache** (estimate). Put the chunk cache
outside the repo.

## 4. The commit discipline that the test gate enforces

`transvrp/expa/paths.py::verify_test_gate` opens the test stream only if ALL hold:
- the full 40-character hash passed as `--frozen-commit` appears in
  `papers/transvrp/RESEARCH_LOG.md`;
- it equals the checked-out `HEAD`;
- `papers/transvrp/code` has no tracked changes and no untracked `.py` files;
- every model being evaluated was fitted at that same commit on a clean tree.

So: **commit and push before fitting, then make no commits until the test evaluation is
done.** `RESEARCH_LOG.md` sits outside `code/`, so editing it does not dirty the tree;
CSV/NPZ/pickle outputs under `code/results/` do not count as dirty either. Do not edit
any `.py` file during the run. If a code change becomes necessary, it means refitting
everything at the new commit (and logging why).

## 5. Run plan

Run from `papers/transvrp/code`. Set once:

```bash
R=results/expA
CH=/path/outside/repo/transvrp_chunks      # >= 60 GB free
E=$R/eligibility_dev.csv
```

**Step 1 — clean start.** `git status` clean, `git pull origin main`, then
`X=$(git rev-parse HEAD)`. Add a log entry "sweep started at commit <short X>, machine
…" to `RESEARCH_LOG.md` — **without committing it** (it stays uncommitted until step 6).

**Step 2 — fit (≈ 2.5 days, the long part).** Run detached (tmux/screen/nohup):

```bash
python3 -m transvrp.expa.sweep fit --stage main --jobs 4 --chunk-dir $CH --results-dir $R --eligibility $E
python3 -m transvrp.expa.sweep fit --stage big  --jobs 4 --chunk-dir $CH --results-dir $R --eligibility $E
```

Fitting is resumable: rerunning the same command skips models already on disk. If a
run dies, just start it again (check free disk and RAM first).

**Step 3 — dev evaluation and sanity (minutes).**

```bash
for s in main big; do
  python3 -m transvrp.expa.sweep eval  --stage $s --eval dev --jobs 4 --results-dir $R
  python3 -m transvrp.expa.sweep merge --stage $s --eval dev --results-dir $R
done
python3 -m transvrp.expa.analysis --merged $R/main.csv --big $R/big.csv --rows-dir $R/main/rows --eval dev --out $R/analysis_dev
```

Dev results are for **sanity only** (coverage 20 instances / 62 routes / 3 reps, no
policy below the oracle, nothing absurd). Do not change anything because of what dev
says; if something is broken, stop, log it, fix via Haiku + review, refit.

**Step 4 — freeze.** Append exactly one line to `RESEARCH_LOG.md`:

```
FROZEN COMMIT: <full 40-character X>
```

Check `git status --porcelain -- papers/transvrp/code` prints nothing except untracked
non-`.py` results.

**Step 5 — test evaluation (minutes).**

```bash
for s in main big; do
  python3 -m transvrp.expa.sweep eval  --stage $s --eval test --frozen-commit $X --jobs 4 --results-dir $R
  python3 -m transvrp.expa.sweep merge --stage $s --eval test --results-dir $R
done
python3 -m transvrp.expa.analysis --merged $R/main.test.csv --big $R/big.test.csv --rows-dir $R/main/rows --eval test --out $R/analysis_test
```

`$R/analysis_test/summary.md` holds the primary decision (WIN / LOSE / TIE_STRONG /
TIE_WEAK) and the 11 Holm-corrected secondary contrasts.

**Step 6 — record.** Log the outcome in `RESEARCH_LOG.md` (dates, durations, decision,
anything unexpected), check the size of `code/results/expA` (models and chunks are
git-ignored; if the per-day `.npz` rows are very large, ask the author before
committing them), then commit everything and push to `main`. Update `STATUS.md`.

**Step 7 — branch on the outcome (spec section 1).** WIN → run the boost arm
(`--stage boost`, M11) at the same frozen-commit discipline; TIE or LOSE → no boost,
the paper reports the pre-planned negative or equivalence result. Then plan the
manuscript with the author.

## 6. Known caveats

- The P5 projection uses 2 measured cells and a linear extrapolation to 10^6 on one
  route; real time may be off by ×2. The hard ceiling in the spec is 7 days, with the
  ordered cuts of section 7.2 (each logged, all before any test evaluation).
- Chunk-cache size (~50 GB) is an estimate; memory at 10^6 was measured on one route
  only (1.8 GB peak RSS).
- Gate gap still open: `git update-index --assume-unchanged` edits are not detected.
  Simply do not use it.
- `RESEARCH_LOG.md` is the record a referee will see. Write in it before looking at
  results whenever possible.
- The pilot's CON3-0 routes are also in the primary set; this is disclosed (spec 7.3).

## 7. File map

| path | what |
|---|---|
| `EXPERIMENT_A.md` | pre-registered spec (revision 2 + errata 12.1, 12.2) |
| `REVIEW_A.md`, `CODE_REVIEW_A.md`, `CODE_REVIEW_A2.md` | spec review, code review, re-review |
| `RESEARCH_LOG.md` | decisions and runs, dated |
| `PROJECT.md`, `STATUS.md`, `VERIFY_CITATIONS.md` | direction notes, state, citation checks |
| `code/transvrp/baton/` | verbatim BATON copies (do not edit) |
| `code/transvrp/expa/` | experiment A modules M1-M11 |
| `code/results/expA/` | eligibility table, pilot results and report |
