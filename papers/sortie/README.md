# SORTIE — fair vehicle routing when route durations are random

**SORTIE**: *Stochastic Ordered Route fairness with Tour-consistency via
Integral Evaluation.* The planner splits customers among m vehicles so that
the workload, measured by random route durations, is fair. Fairness is an
ordered (OWA-type) measure of the m durations: the maximum, range, Gini
mean difference, MAD-median or empirical variance. Each driver drives a
tour that is optimal for that driver: a TSP tour when travel is
deterministic, and a risk-consistent tour (mean plus κ standard deviations)
when it is not. The deterministic version of the problem with v = 0 and
κ = 0 is the bilevel fair CVRP of Ljubić, Puerto and Torrejón (2026).

Review state and freeze policy: `STATUS.md`. Experiment report: `REPORT.md`.

## Files

| Path | Content |
|---|---|
| `sortie.tex`, `sortie.pdf` | Manuscript: literature review, model, theory with proofs, algorithm, computational study. |
| `THEORY.md` | Working notes for the model, Propositions 1–6 and the algorithm (longer than the paper). |
| `REPORT.md` | Experiment report: setup, every table, figures, findings and limitations. |
| `cover_letter.tex` | Cover letter (venue to be decided; see `STATUS.md`). |
| `STATUS.md` | Venue candidates, review state, separation from the other papers. |
| `VERIFY_CITATIONS.md` | Citations whose content still needs a full-text check. |
| `lit/references.bib` | Bibliography; bibliographic data checked against Crossref or the publisher. |
| `lit/LIT_NOTES.md` | Literature review notes (six clusters, gaps G1–G3, unverified items). |
| `data/X-n641-k35.vrp` | CVRPLIB instance (Uchoa et al. 2017); the source of every test instance. |
| `code/SPEC.md` | Module and experiment specifications (modules A–F, experiments E0–E8). |
| `code/sortie/instances.py` | Instance slicing from X-n641-k35, congestion multiplier H. |
| `code/sortie/tsp.py` | Held–Karp, local search, risk-consistent tours (dichotomic parametric search). |
| `code/sortie/pboi.py` | Ordered measures, the Poisson-binomial ordered integral (PBOI), rank and σ prices. |
| `code/sortie/sim.py` | Arc-level Monte Carlo with common random numbers (out-of-sample evaluation). |
| `code/sortie/search.py` | Cost seed, the SORTIE search, the det/SAA baselines and the ablations. |
| `code/tests/` | 57 unit tests (`python -m pytest tests -q` from `code/`). |
| `code/experiments/` | E0–E8 drivers, `run_all.sh`, `analyze.py`. |
| `results/*.csv`, `results/*.log` | Raw results, one row per run; logs of each experiment. |
| `results/plans/` | Every solved plan (JSON), so the evaluation can be rerun without the search. |
| `results/tables/` | Generated LaTeX tables and `ALL_TABLES.md`. |
| `figures/` | Generated figures (PDF and PNG). |

## Reproducing

Python 3.11 with numpy, scipy, numba, pandas, matplotlib and tabulate.

```bash
cd papers/sortie/code
python -m pytest tests -q
bash experiments/run_all.sh          # E0..E8, about 3 h on 4 cores; add --quick for a smoke run
python experiments/analyze.py        # tables and figures
```

Each experiment can be run on its own (`python experiments/e3_main.py`).
Drivers append to their CSV and skip finished rows, so an interrupted run
resumes where it stopped. `SORTIE_RESULTS` and `SORTIE_PLANS` redirect
output, and `SORTIE_DATA` points at another copy of the instance file.

## Where each table and figure comes from

All tables are written by `code/experiments/analyze.py` to
`results/tables/<name>.tex`, and all of them are collected in
`results/tables/ALL_TABLES.md`.

| Output | Source CSV | Experiment |
|---|---|---|
| `e1_accuracy.tex`, `e1_timing.tex` | `e1_accuracy.csv`, `e1_timing.csv` | E1: PBOI against Monte Carlo, evaluation time |
| `e2_exact_raw.tex` | `e2_exact.csv` | E2: heuristics against full enumeration (n = 10) |
| `e3_improvement.tex`, `e3_by_n.tex`, `e3_illusion.tex`, `e3_levels.tex`, `e3_cvar.tex`, `e3_effort.tex` | `e3_main.csv` | E3: main grid, n = 15–35 |
| `e4_inconsistency.tex` | `e4_inconsistency.csv` | E4: free tours against lower-level tours |
| `e5_ablation.tex`, `e5_effort.tex` | `e5_ablation.csv` | E5: rank-price screening against random moves |
| `e6_sensitivity.tex` | `e6_sensitivity.csv` | E6: arc noise v |
| `e7_scale.tex` | `e7_scale.csv` | E7: n = 50–100 |
| `e8_congestion.tex`, `e8_by_n.tex` | `e8_congestion.csv` | E8: heterogeneous (congested) arc noise |
| `figures/e3_improvement.*` | `e3_main.csv` | Improvement over the deterministic fair plan |
| `figures/illusion_range.*` | `e3_main.csv` | Believed against true expected range |
| `figures/e6_illusion_v.*` | `e6_sensitivity.csv` | Unfairness hidden by the deterministic plan, against v |

The plan cache `results/plans/seed_*.json` (E0) holds the cost-efficient
plans whose cost C* defines the budget B·C* in every other experiment.

## Self-containment

SORTIE shares no code, instances, results or scenario generators with
BATON, TEMPO, QUASAR or RARECERT, which are under review. It does not import
`svrpspd_wdro/`. Its instances are slices of the public X-n641-k35 file,
which none of those papers uses.
