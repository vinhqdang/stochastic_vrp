# TRANSVRP — status

- **Working title:** open (placeholder `TRANSVRP`). Transformer / LLM methods
  for stochastic VRP (SVRPSPD) optimisation.
- **Venue:** Transportation Science (decided 2026-11-25).
- **State:** **ACTIVE — experiments A and B complete (2026-11-25); next: manuscript.**
  - *Experiment A* (test stream opened once at `c37ad7c`): FULL vs BATON at equal
    paths (10^5) **WIN**, +0.82% [CI95 +0.65, +0.99], 19/20 instances. The gain is
    information (H2), architecture is irrelevant (ARCH, MLP ~ set encoder; boosted
    trees M11 equivalent). Crossover N = 500; FULL worse at N = 100. Report:
    `code/results/expA/analysis_test/summary.md`.
  - *Experiment B* (published 2023–2026 baselines, spec `EXPERIMENT_B.md` rev. 3.4;
    fit `bf3edab`, gate M `5d47d9e`, FAMILY line L `fab25c5`, results `744d7b6`).
    Cuts (i)–(iii) by compute trigger: DOS dropped, 6 tuning routes, 2 replicates.
    Dev gates: THR fails B-THR (descriptive), ALP-Z, RLSM, DQN, PPO in the family.
    test_b, FULL vs arm (Holm over 4): ALP-Z (Zhu et al. 2026) +11.3% WIN; PPO
    (Kadyrov et al. 2025) +1.7% WIN; RLSM (Herrera et al. 2024) +0.16% and DQN
    (Dastpak et al. 2023 style) +0.16% TIE_STRONG (within ±0.7%). THR (Yang–Mathur–
    Ballou / Venkateshan–Mathur) +11.0% descriptive. RE-EVAL FULL vs BATON +0.85%.
    Independence world: FULL = BATON = THR = ALP-Z within ±0.1%, so the load-only
    methods lose through dependence between stops. Report:
    `code/results/expB/analysis_test_b/summary.md`.
  - Story: **the state matters, the learner does not** — any regressor or TD
    learner given the rich state reaches the same cost; load-only DP/ALP lose 11–13%.
  - Weaknesses to state: DQN/PPO budgets far below their source papers; DOS cut for
    time; 2 replicates; THR fails its own-model check (5 of 7 failures are SE = 0
    routes); DQN tuning noisy.
  - **BATON accepted at C&OR (24 Nov 2026)** — it can now be cited (in press).
- **Freeze policy:** open for edits.
- **Seed paper:** Liu et al., OR-Transformer, arXiv 2609.01933v2 (full text
  read). Local copy in `refs_local/`.
- **Separation from papers under review:** BATON (C&OR, r2) and TEMPO (TS) are
  frozen. This paper may *read* BATON's engine as a baseline (as TEMPO does) but
  must not edit `papers/baton/`, `papers/tempo/`, or `svrpspd_wdro/core/`/`ev/`
  behaviour. **Decided 2026-11-15 (author): self-contained.** The needed BATON
  files, instances and plans were copied into `code/` (provenance in
  `code/transvrp/baton/PROVENANCE.md`); nothing imports `svrpspd_wdro/`.
- **Manuscript (decided 2026-11-25):** Transportation Science, single author (Quang-Vinh Dang), story "the state matters, the learner does not"; plan in `OUTLINE.md`.
