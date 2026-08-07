# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Research codebase for **stochastic vehicle routing**, structured as one
shared engine (`svrpspd_wdro/`) plus one directory per paper under
`papers/`. More papers will be added over time; new per-paper code goes
in a subpackage of `svrpspd_wdro/` (e.g. `core/`, `ev/`) so everything
shares the same instances, simulator, cost model, and test suite.

### Paper 1 — BATON (papers/baton/, UNDER REVIEW at Computers & OR)

SVRPSPD with two coupled layers:

1. **Planning** — ALNS route construction under six capacity-feasibility
   gates (deterministic, SAA-CVaR, Wasserstein-DRO, plus published robust
   baselines: Gounaris inflation, Bertsimas–Sim budget, moment-DRO).
2. **Execution** — online mid-route handoff policies under demand
   uncertainty. Contribution: **BATON** (Backward-induction AcTion
   pricing for ONline recourse; code labels keep the historical v2_lsm/
   v2_act names): peak-aware labels + a Longstaff–Schwartz optimal-
   stopping trigger over {continue, handoff, depot-restock} — no
   threshold parameter. Benchmarked against the endpoint-threshold
   predecessor (ablation), tuned thresholds, published rule-based
   recourse (pi1–pi3), a plug-in DP, and a clairvoyant oracle.

Costs follow a three-class fleet model (planned / standby / emergency
vehicles) with per-stop price schedules built from real route geometry —
see `svrpspd_wdro/core/costs.py`.

**Do not modify `papers/baton/` while the paper is under review** (see
its STATUS.md); revisions only when the decision arrives.

### Paper 2 — TEMPO (papers/tempo/, UNDER REVIEW at Transportation Science)

E-process (e-value) monitoring of a running routing plan under multiple
stochastic factors — demand, diurnal/weather travel times, congestion
shocks, accidents, vehicle breakdowns, dwell times. One master
likelihood-ratio martingale tests H0 "the day is following the planning
model"; crossing 1/alpha triggers re-optimization with anytime-valid
type-I error control. The novel coupling: predictable (previsible)
tilt/weighting of the per-channel bets driven by decision-relevance
computed from BATON's fitted continuation costs and price schedules.
TEMPO = Tilted E-Martingale Process for Online re-optimization.
Evaluation set is identical to BATON's (Dethloff + Salhi-Nagy + city,
same plan cache). Spec: `papers/tempo/PROJECT.md`. Code: `svrpspd_wdro/ev/`.

**Do not modify `papers/tempo/` while the paper is under review** (see
its STATUS.md); revisions only when the decision arrives.

### Paper 3 — CSoNet 2026 / JOCO combinatorial-scheduling paper
### (papers/csonet2026/, SUBMITTED to JOCO)

A standalone, independent combinatorial-optimization theory paper:
Minimum Weighted Hazard-Exposure Dispatch (MWHED), a single-machine
scheduling problem for dispatching a vehicle/crew from one depot to
sites racing a spreading hazard (wildfire, flood, contamination,
congestion) before a per-site deadline. NP-hardness via an equal-
deadline-reduces-to-knapsack argument, an exact pseudo-polynomial DP,
an FPTAS, and a matroid-greedy tractable special case (equal dispatch
cost). Target: CSoNet 2026's Journal Track → Journal of Combinatorial
Optimization. Deliberately shares NO code, instances, or results with
BATON or TEMPO — see `papers/csonet2026/README.md` for why (both other
papers are under review elsewhere; JOCO/CSoNet explicitly prohibit
simultaneous submission and salami-slicing).

### Paper 4 — WRNF (papers/wrnf/, ABANDONED 2026-07-27 — scooped)

Abandoned before drafting. The thesis — that type-1 Wasserstein
robustness over unbounded support leaves the decision unchanged — is
**published**: Mohajerin Esfahani & Kuhn (2018) Remark 6.7 states it
verbatim (naming the newsvendor), and Duque, Mehrotra & Morton, SIAM J.
Optimization 32(3):1499-1522 (2022) do the two-stage version, including
the bounded-support contrast, with a supply-allocation testbed that is
essentially the model we built. Byeon-Fang-Kim (SIOPT 2025) and Byeon
(arXiv:2501.05619) own the dual-vertex modulus mechanism.

Do NOT resume it. `papers/wrnf/PROJECT.md` opens with an abandonment
banner listing every scooping citation plus two technical errors found
in our own claim; `STATUS.md` records the decision. Kept as a
documented negative result. The prototype numerics are correct, just
not novel.

### Paper 5 — ECLAIR (papers/eclair/, ACTIVE — the one open paper)

*E-process Certification of LLM-generated Constraint Models with
Adaptive Instance Routing.* Target: **Constraints (Springer)**, LLM+CP
special-issue CFP. An LLM generates k candidate constraint models from
a natural-language spec; ECLAIR runs a stream of imperfect probes
(exact micro-instance oracles, provably valid metamorphic relations,
cross-model feasibility checks), bets against each candidate's
faithfulness with an e-process (anytime-valid via Ville; e-BH across
the repair tournament), and routes solver-seconds Kelly-style to the
(candidate, probe) pair with the best expected log-evidence per
second. Ports TEMPO's e-process *technique* but deliberately shares
NO code, instances, or results with papers 1–3 (same separation
argument as paper 3) — all ECLAIR code stays inside `papers/eclair/`.
Spec: `papers/eclair/PROJECT.md`; adoption record: `RESEARCH_LOG.md`.

One live thread from paper 4: BATON's W-DRO gate in
`svrpspd_wdro/core/wdro_exact.py` adds a **route-independent** constant
`epsilon/(1-alpha)` to a CVaR gate. As a feasibility constraint that
tightens the threshold rather than cancelling, so not vacuous — but it
does make the W-DRO gate equivalent to the SAA-CVaR gate at a shifted
threshold. Check against the manuscript's claims when BATON's decision
arrives; a referee could raise it. BATON is frozen, so do not act now.

## Layout

- `svrpspd_wdro/` — the shared engine + all experiment scripts (see its
  README for the module map and reproduction commands). `core/` is
  BATON-era machinery reused everywhere; `ev/` is paper 2. Paper 3
  (`papers/csonet2026/`) does NOT use this engine at all — its code is
  self-contained inside its own directory.
- `papers/baton/` — C&OR manuscript (frozen; tables ONLY via its
  `make_tables.py`, never hand-edit `tables/*`).
- `papers/tempo/` — paper 2 (TEMPO), frozen under review (see above).
- `papers/csonet2026/` — paper 3, **SUBMITTED** to JOCO's Editorial
  Manager; treat as frozen except for editor-requested fixes.
- `papers/wrnf/` — paper 4, **ABANDONED (scooped)**; kept as a
  documented negative result. Do not resume — see its PROJECT.md banner.
- `papers/eclair/` — paper 5 (ECLAIR), **ACTIVE and open for edits**;
  self-contained (no `svrpspd_wdro/` imports — see its README.md).
- **Parallel papers, independent review clocks** — always check each
  paper's own `STATUS.md` before touching it; "frozen" is a per-paper
  state, not a repo-wide one, and a paper can move from editable to
  frozen mid-session the moment the author submits it. As of now
  papers 1–3 are closed to edits, paper 4 is abandoned, and paper 5
  (ECLAIR, `papers/eclair/`) is the **one open paper**.
- **Adding a new paper:** create `papers/<shortname>/` holding the
  manuscript, a `STATUS.md` (venue, review state, freeze policy), a
  `README.md` with the file manifest and how each table/figure
  regenerates, `references.bib`, and the cover letter. New shared
  machinery goes in a new subpackage of `svrpspd_wdro/` (alongside
  `core/` and `ev/`) with tests in `svrpspd_wdro/tests/` — *unless*
  the paper must demonstrably share nothing with a paper under review
  elsewhere, in which case keep its code self-contained in its own
  directory and record that argument in its README, as paper 3 does.
- Citations: real papers with DOIs; anything unverified goes in the
  paper's `VERIFY_CITATIONS.md`.
- `RESULTS_OTR2.md` — BATON results summary.
- `legacy/` — archived ECHO-era code; do not extend, do not import.

## Working rules

- Python 3.11, no conda needed: `pip install -r requirements.txt`.
- Run tests from `svrpspd_wdro/`: `python -m pytest tests/ -q`
  (~180+ tests; keep green).
- Commit author: Vinh <dqvinh87@gmail.com>. Always commit and push to
  `origin main` after every change. No pull requests unless asked.
- Long evaluations run in background with logs under
  `svrpspd_wdro/results/*.log`; solved plans are cached in
  `results/plans/*.json` (per-instance, gates merged) so eval reruns
  skip ALNS.
- Watch working-directory drift: run pipeline commands from
  `svrpspd_wdro/` (scripts resolve paths relative to their own location,
  but log/output conventions assume that cwd).
- Gurobi: WLS licence at `/root/gurobi.lic`
  (`export GRB_LICENSE_FILE=/root/gurobi.lic`); HiGHS is the default
  solver so nothing breaks without it.
- City instances are generated from OSM via `make_city_instances.py`;
  distances are real road distances symmetrized for the ALNS moves —
  do NOT feed asymmetric matrices to the 2-opt (it cycles).
- Report times to the user in GMT+7.
