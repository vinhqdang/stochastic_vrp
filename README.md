# Stochastic VRPSPD with Distributionally Robust Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status: Research](https://img.shields.io/badge/Status-Research-green.svg)](https://github.com/vinhqdang/stochastic_vrp)

## Overview

This repository implements a unified optimization framework for the **Vehicle Routing Problem with Simultaneous Pickup and Delivery under Stochastic Demand (SVRPSPD)**. It contains two complementary components:

| Component | What it does |
|---|---|
| **ALNS + W-DRO** (`svrpspd_wdro/`) | Offline route planning: builds routes that are robust to demand uncertainty via Wasserstein Distributionally Robust Optimization |
| **OTR** (`svrpspd_wdro/core/otr.py`) | Online execution policy: monitors live demand as a truck drives its route and triggers a proactive spare-truck handoff when overflow risk crosses a learned threshold |

The two components are designed to be used together: ALNS+W-DRO plans the routes before departure; OTR manages residual uncertainty during execution.

---

## Problem Definition

Each vehicle departs the depot preloaded with all deliveries and simultaneously collects pickups from customers. The on-board load after visiting customer $k$ is:

$$L_k = \underbrace{\sum_{i \in \text{route}} d_i}_{L_0} - \sum_{j=1}^k d_j + \sum_{j=1}^k p_j$$

Because pickup demands $p_i$ are stochastic, the peak load $f_r(\xi) = \max_k L_k$ is random. The route overflows when $f_r(\xi) > Q$.

---

## Repository Structure

```
stochastic_vrp/
│
├── svrpspd_wdro/                    # Main research package
│   ├── core/
│   │   ├── instance.py              # Instance data structure
│   │   ├── route.py                 # Route + peak-load computation
│   │   ├── cache.py                 # Prefix/suffix-peak cache (O(Nm) build)
│   │   ├── wdro_exact.py            # Exact W-DRO evaluator — CVaR + Wasserstein
│   │   ├── wdro_fast.py             # O(N log N) insertion/removal via cache
│   │   ├── alns_wdro.py             # ALNS-SA main loop with W-DRO objective
│   │   ├── filter.py                # Phase-2 cheap-proxy pruning filter
│   │   ├── otr.py                   # OTR: Online Threshold Reassignment
│   │   ├── scenarios.py             # Scenario generation
│   │   └── seeding.py               # RNG seeding helpers
│   ├── data/
│   │   └── Dethloff/                # 40 VRPSPD benchmark instances (*.vrpspd)
│   ├── scripts/
│   │   ├── dethloff_runner.py       # Batch ALNS evaluation on Dethloff instances
│   │   ├── run_otr_eval.py          # ← full OTR evaluation (single entry point)
│   │   ├── run_test1_wdro.py        # W-DRO unit regression
│   │   ├── run_test2_filter_speedup.py
│   │   └── run_test3_stress.py
│   └── tests/                       # pytest suite
│
├── algorithms/                      # Earlier callback-routing baselines
│   ├── alns.py                      # ALNS + M-DRO (moment-based DRO)
│   ├── echo.py                      # ECHO: MDP rollout policy
│   ├── apex_v3.py                   # APEX v3 deterministic heuristic
│   ├── pomo_simplified.py           # POMO
│   ├── drl_du_simplified.py         # DRL-DU
│   ├── sro_ev.py / gnn_cb.py / th_cb.py
│   └── ...
├── evaluation/                      # Experiment runner and metrics
├── scenarios/                       # YAML scenario configs and generator
├── results/                         # Saved outputs (JSON, CSV, plots)
├── main.py                          # Entry point for baseline comparisons
└── requirements.txt
```

---

## Algorithms

### 1. ALNS + W-DRO (offline planning)

The core planner in `svrpspd_wdro/core/alns_wdro.py`. Objective:

$$F(\text{solution}) = \underbrace{\sum_r \text{dist}(r)}_{\text{travel}} + \lambda \underbrace{\sum_r \Phi(r)}_{\text{W-DRO penalty}}$$

where the W-DRO penalty for route $r$ is:

$$\Phi(r) = \mathrm{CVaR}_\alpha^{F_0}\!\bigl(\max(0,\, f_r(\xi) - Q)\bigr) + \frac{\varepsilon}{1-\alpha}$$

- $F_0$ = empirical distribution over $N$ historical scenarios  
- $\varepsilon$ = Wasserstein ambiguity radius  
- The $+\varepsilon/(1-\alpha)$ term follows from Universal Lipschitz Invariance (every route in $\mathcal{R}_n$ has $\|\beta_{r,k}\|_\infty = 1$)

**Speed:** candidate insertions are evaluated in $O(N \log N)$ using the prefix/suffix-peak cache (`cache.py` + `wdro_fast.py`), independent of route length.

Three planning policies are compared:

| Policy | Capacity gate |
|---|---|
| **Det** | Nominal (mean-demand) peak $\leq Q$ |
| **SAA** | Empirical CVaR$_\alpha \leq Q$ |
| **WDRO** | Empirical CVaR$_\alpha \leq Q(1 - \varepsilon_{\text{frac}})$ |

### 2. OTR — Online Threshold Reassignment (execution policy)

Defined in `svrpspd_wdro/core/otr.py`. After route planning, OTR operates in real time as the truck visits customers.

**Offline phase** — fit once on historical routes:

```python
models = fit_otr(g_hist, B)   # g_hist: (N, m) net increments; B = Q - L0
```

Each `models[k]` is an isotonic regression mapping the running net-increment sum $W_k$ to $\hat{p}(W_k) = P(W > B \mid W_k)$.

**Online phase** — called at every customer stop:

```
for k = 1 .. m:
    observe d_k, p_k              ← revealed on arrival
    W_k += p_k - d_k
    if W_k > B  → EMERGENCY       ← overflow already happened
    if k == m   → COMPLETE
    if models[k].predict(W_k) > τ → HANDOFF  ← proactive spare truck
```

**Threshold selection:**

```python
tau_myopic(omegaF, Cfail)          # break-even: omegaF / Cfail
tune_tau(g_train, B, models, ...)  # grid-search on training data (recommended when Cfail/omegaF > 5)
```

---

## Dataset

The `svrpspd_wdro/data/Dethloff/` directory contains **40 standard VRPSPD benchmark instances** from Dethloff (2001), organized in four families:

| Family | Topology | Vehicles | Customers |
|---|---|---|---|
| `CON3-*` (10 instances) | Concentrated, 3 clusters | 4 | 50 |
| `CON8-*` (10 instances) | Concentrated, 8 clusters | 4 | 50 |
| `SCA3-*` (10 instances) | Scattered, 3 clusters | 4 | 50 |
| `SCA8-*` (10 instances) | Scattered, 8 clusters | 4 | 50 |

Each `.vrpspd` file contains a full distance matrix (`EDGE_WEIGHT_SECTION`) and per-customer mean delivery/pickup demands (`PICKUP_AND_DELIVERY_SECTION`).

---

## Quick Start

### Prerequisites

```bash
conda create -n py313 python=3.13
conda activate py313
pip install -r requirements.txt
```

### Run the full OTR evaluation (single command)

```bash
cd svrpspd_wdro
python scripts/run_otr_eval.py
```

This script:
1. Solves all 40 Dethloff instances with ALNS under Det / SAA / WDRO planning policies
2. Generates train and test demand scenarios for each route
3. Fits OTR, tunes $\tau$, and simulates three execution policies (tuned, myopic, no-handoff)
4. Writes **`results_otr_eval.csv`** and **`results_otr_eval.xlsx`**

**Options** (all optional):

```bash
python scripts/run_otr_eval.py \
    tlim=60          \  # ALNS time limit per policy (seconds)
    n_train=1000     \  # training scenarios per route
    n_test=2000      \  # test scenarios per route
    cfail=5.0        \  # Cfail / omegaF ratio
    policies=SAA,WDRO\  # which ALNS plans to evaluate
    workers=8        \  # parallel worker processes (default: all CPUs)
    max=5            \  # limit to first N instances (useful for testing)
    out=results/run1    # output file stem
```

**Estimated runtime:** ~60 min ALNS solving + ~10 min OTR (all 40 instances, 3 policies, default settings).

### Run only the ALNS planner

```bash
cd svrpspd_wdro
python scripts/dethloff_runner.py dir=data/Dethloff t=60
# outputs: results_dethloff_summary.xlsx
```

Add `sweep` to re-price solutions under 7 demand mixture scenarios without re-solving:

```bash
python scripts/dethloff_runner.py dir=data/Dethloff sweep
```

### Run unit tests

```bash
cd svrpspd_wdro
python -m pytest tests/ -v
```

---

## Output Format

`results_otr_eval.csv` has one row per `(instance, planning policy)` combination:

| Column | Description |
|---|---|
| `Instance`, `Plan`, `N_cust`, `K_routes` | Instance metadata |
| `Travel`, `omega_V`, `omega_F`, `Cfail` | Route cost parameters |
| `ALNS_EVx`, `ALNS_exec`, `ALNS_TBC` | W-DRO baseline: $E[\text{extra trucks}]$, execution cost, total budget cost |
| `OTR_tuned_{exec,TBC,HO_rate,fail,ok}` | OTR with grid-tuned $\tau$ |
| `OTR_myopic_{exec,TBC,HO_rate,fail,ok}` | OTR with $\tau = \omega_F / C_{\text{fail}}$ |
| `NoHandoff_{exec,TBC,HO_rate,fail,ok}` | Reactive baseline ($\tau = 1$, overflow only) |
| `OTR_saving_pct` | `(NoHandoff_exec − OTR_tuned_exec) / NoHandoff_exec × 100` |

Total Budget Cost: $\text{TBC} = \text{Travel} + \omega_V \cdot K + \text{exec\_cost}$

---

## Hyperparameters

### ALNS + W-DRO

| Parameter | Default | Description |
|---|---|---|
| `alpha` | 0.90 | CVaR confidence level |
| `epsilon` | 0.50 | Wasserstein ambiguity radius |
| `penalty_lambda` | 1.0 | W-DRO penalty weight $\lambda$ |
| `max_iters` | 5000 | ALNS iterations |
| `alpha_cooling` | 0.9997 | SA geometric cooling rate |
| `destroy_frac` | 0.10–0.30 | Fraction of customers removed per iteration |
| `CV` | 0.30 | Demand coefficient of variation |
| `OMEGA_RATIO` | 50.0 | $\omega_F / \omega_V$ (spare-truck cost ratio) |

### OTR

| Parameter | Default | Description |
|---|---|---|
| `alpha` (CVaR level) | 0.90 | Inherited from W-DRO configuration |
| `tau` | tuned | Decision threshold — see `tune_tau()` or `tau_myopic()` |
| `cfail / omegaF` | 5.0 | Cost ratio for unplanned vs planned handoff |
| `n_train` | 1000 | Scenarios used to fit isotonic models |
| `n_test` | 2000 | Scenarios used to evaluate simulated cost |

---

## References

1. Dethloff, J. (2001). Vehicle routing and reverse logistics: The vehicle routing problem with simultaneous delivery and pick-up. *OR Spectrum*, 23(1), 79–96.
2. Ropke, S. & Pisinger, D. (2006). An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows. *Transportation Science*, 40(4), 455–472.
3. Delage, E. & Ye, Y. (2010). Distributionally Robust Optimization Under Moment Uncertainty. *Operations Research*, 58(3), 595–612.
4. Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21–41.
5. Kuhn, D., Esfahani, P.M., Nguyen, V.A. & Shafieezadeh-Abadeh, S. (2019). Wasserstein Distributionally Robust Optimization: Theory and Applications in Machine Learning. *INFORMS TutORials in Operations Research*.

---

## Citation

This work is currently under review. If you use this code, please reference:

```
Dang, V.Q. (2025). Distributionally Robust ALNS and Online Threshold Reassignment
for the Stochastic VRPSPD. https://github.com/vinhqdang/stochastic_vrp
```

## Contact

**Vinh Dang** — [dqvinh87@gmail.com](mailto:dqvinh87@gmail.com)

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
