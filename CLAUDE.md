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
### (papers/csonet2026/, PREPARED, NOT YET SUBMITTED)

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

## Layout

- `svrpspd_wdro/` — the shared engine + all experiment scripts (see its
  README for the module map and reproduction commands). `core/` is
  BATON-era machinery reused everywhere; `ev/` is paper 2. Paper 3
  (`papers/csonet2026/`) does NOT use this engine at all — its code is
  self-contained inside its own directory.
- `papers/baton/` — C&OR manuscript (frozen; tables ONLY via its
  `make_tables.py`, never hand-edit `tables/*`).
- `papers/tempo/` — paper 2 (TEMPO), frozen under review (see above).
- `papers/csonet2026/` — paper 3, prepared but not yet submitted;
  freely editable until the STATUS.md there says otherwise.
- **Three parallel papers, three independent review clocks** — always
  check each paper's own `STATUS.md` before touching it; "frozen" is a
  per-paper state, not a repo-wide one, and a paper can move from
  editable to frozen mid-session the moment the author submits it.
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
