# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Research codebase for the **Stochastic Vehicle Routing Problem with
Simultaneous Pickup and Delivery (SVRPSPD)**. Two coupled layers:

1. **Planning** — ALNS route construction under six capacity-feasibility
   gates (deterministic, SAA-CVaR, Wasserstein-DRO, plus published robust
   baselines: Gounaris inflation, Bertsimas–Sim budget, moment-DRO).
2. **Execution** — online mid-route handoff policies under demand
   uncertainty. The paper's contribution is **BATON** (Backward-induction AcTion
   pricing for ONline recourse; code labels keep the historical v2_lsm/
   v2_act names): peak-aware labels + a Longstaff–Schwartz optimal-
   stopping trigger over {continue, handoff, depot-restock} — no
   threshold parameter. Benchmarked against the endpoint-threshold predecessor (ablation), tuned thresholds, published
   rule-based recourse (pi1–pi3), a plug-in DP (the exact method for the
   stopping stage), and a clairvoyant oracle.

Costs follow a three-class fleet model (planned / standby / emergency
vehicles) with per-stop price schedules built from real route geometry —
see `svrpspd_wdro/core/costs.py`.

## Layout

- `svrpspd_wdro/` — the maintained pipeline (see its README for the
  module map and reproduction commands for every results table).
- `paper/` — Springer sn-jnl manuscript (double-blind, `\ifanon`);
  tables regenerate ONLY via `paper/make_tables.py` from result CSVs —
  never hand-edit `paper/tables/*`. Citations: real papers with DOIs;
  anything unverified goes in `paper/VERIFY_CITATIONS.md`.
- `RESULTS_OTR2.md` — running results summary.
- `legacy/` — archived ECHO-era code; do not extend, do not import.

## Working rules

- Python 3.11, no conda needed: `pip install -r requirements.txt`.
- Run tests from `svrpspd_wdro/`: `python -m pytest tests/ -q`
  (~180 tests; keep green).
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
