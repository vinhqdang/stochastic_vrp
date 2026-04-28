# Stochastic VRPSPD with Distributionally Robust Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/Status-Research-green.svg)](https://github.com/vinhqdang/stochastic_vrp)

## Overview

This repository implements a unified optimization framework for the **Vehicle Routing Problem with Simultaneous Pickup and Delivery under Stochastic Demand (SVRPSPD)**. The core algorithm is an **Adaptive Large Neighborhood Search (ALNS)** metaheuristic augmented with a **Moment-Based Distributionally Robust Optimization (M-DRO)** risk layer, enabling tractable, distribution-free route planning under demand uncertainty.

The framework operates in two modes controlled by a single flag:

| Mode | Problem | Objective |
|------|---------|-----------|
| **Deterministic** | Pure CVRP (delivery only) | Minimize total Euclidean distance |
| **Stochastic** | VRPSPD with random pickup demand | Minimize distance + DRO risk penalty |

---

## Problem Definition

Each vehicle simultaneously delivers goods to and collects goods from customers. The vehicle departs the depot preloaded with all deliveries:

$$L_0 = \sum_{i \in \text{route}} D_i$$

After visiting customer $k$, the load evolves as:

$$L_k = L_0 + \sum_{j=1}^{k}(P_j - D_j)$$

Since pickup demands $P_i$ are **stochastic**, the load is random — hard capacity checks are not meaningful. Instead, we bound the worst-case violation probability at each stop using the **Cantelli–Chebyshev inequality**:

$$P(\text{overload at stop } k) \leq \frac{\text{Var}(L_k)}{\text{Var}(L_k) + \text{slack}_k^2}$$

and penalize the routing objective accordingly.

---

## Key Contributions

### 1. Spatial Demand Correlation Model
Pickup demands are modeled with a spatially decaying covariance structure:

$$\Sigma_{ij} = \sigma_i \cdot \sigma_j \cdot \exp\!\left(-\frac{d_{ij}}{\theta}\right)$$

capturing geographic demand correlation (e.g., regional weather effects) between nearby customers.

### 2. M-DRO Route Evaluation
- Per-stop failure probabilities via the tight **Cantelli–Chebyshev bound** (no distributional assumptions beyond first two moments)
- Route Risk Index (RRI) aggregated via union bound
- Penalized objective: $Z = \text{distance} + \frac{\lambda_0}{\ln(m+1)} \cdot \max(0, \text{RRI} - \alpha_0 \cdot m^{0.5})$
- Efficient $O(mn)$ prefix-sum covariance computation

### 3. ALNS with Simulated Annealing
- **Destroy operators**: random removal, worst-cost removal
- **Repair operators**: greedy insertion, regret-2 insertion (Ropke & Pisinger, 2006)
- **Adaptive weights**: online operator scoring updated every 100 iterations
- **SA acceptance**: geometric cooling with temperature restart

### 4. Four-Universe Gaussian Copula Validation
Post-optimization Monte Carlo stress test across four demand distributional shapes (Gaussian, right-skewed, left-skewed, heavy-tailed) linked by a Gaussian Copula to preserve spatial correlation.

---

## Repository Structure

```
stochastic_vrp/
├── algorithms/
│   ├── alns.py              # Main algorithm (ALNS + M-DRO) ← primary contribution
│   ├── echo.py              # ECHO: MDP rollout baseline
│   ├── apex_v3.py           # APEX v3: deterministic heuristic baseline
│   ├── pomo_simplified.py   # POMO baseline
│   ├── drl_du_simplified.py # DRL-DU baseline
│   ├── sro_ev.py            # SRO-EV baseline
│   ├── gnn_cb.py            # GNN-CB baseline
│   └── th_cb.py             # TH-CB baseline
├── evaluation/              # Experiment runner and metrics
├── scenarios/               # YAML scenario configs and generator
├── utils/                   # Shared data structures and utilities
├── results/                 # Saved outputs (JSON, CSV, plots)
├── main.py                  # Entry point for baseline comparisons
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
conda create -n vrp_env python=3.10
conda activate vrp_env
pip install -r requirements.txt
```

### Running ALNS on CVRPLIB Instances

1. Download benchmark instances from [CVRPLIB](http://vrp.atd-lab.inf.puc-rio.br/) (e.g., Augerat Set X `.vrp` files + `.sol` files into a directory).

2. Set the data directory and run:

```bash
# Set your .vrp data directory
export VRP_DATA_DIR=/path/to/your/vrp/instances

# Run (stochastic mode is enabled by default)
python algorithms/alns.py
```

Or edit `TARGET_DIR` directly at the bottom of `alns.py`:

```python
TARGET_DIR = "/path/to/your/vrp/instances"
```

### Switching Between Modes

At the top of `algorithms/alns.py`, set:

```python
STOCHASTIC_MODE = True   # SVRPSPD with DRO risk penalty + Monte Carlo validation
STOCHASTIC_MODE = False  # Deterministic CVRP, reports gap vs Best Known Solution
```

### Output

For each instance the benchmark engine produces:
- **Console table**: vehicles used, distance, runtime, per-universe failure rates
- **CSV file**: `results_stochastic.csv` / `results_deterministic.csv`
- **Convergence CSV**: `convergence_<instance>.csv` — best objective per iteration

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CV` | 0.2 | Coefficient of variation for demand noise |
| `CAPACITY_FACTOR` | 1.2 | Capacity relaxation factor for stochastic mode |
| `THETA_FRACTION` | 0.1 | Spatial correlation range (fraction of map diameter) |
| `ALPHA_BASE` | 0.2 | Base DRO risk threshold |
| `GAMMA` | 0.5 | Risk threshold scaling exponent ($\sqrt{m}$) |
| `LAMBDA_0` | 500.0 | Base DRO penalty multiplier |
| `ALNS_ITERATIONS` | 5000 | Total destroy–repair cycles |
| `SA_TEMP_INIT` | 100.0 | Initial SA temperature |
| `SA_COOLING` | 0.9997 | Geometric cooling rate |
| `MC_SAMPLES` | 10000 | Monte Carlo samples for validation |

---

## References

1. Ropke, S. & Pisinger, D. (2006). An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows. *Transportation Science*, 40(4), 455–472.
2. Delage, E. & Ye, Y. (2010). Distributionally Robust Optimization Under Moment Uncertainty. *Operations Research*, 58(3), 595–612.
3. Cantelli, F.P. (1928). Sui confini della probabilità. *Atti del Congresso Internazionale dei Matematici*.
4. Schoenberg, I.J. (1938). Metric Spaces and Positive Definite Functions. *Transactions of the AMS*, 44(3), 522–536.

---

## Citation

This work is currently under review. If you use this code, please reference:

```
Dang, V.Q. (2025). Distributionally Robust ALNS for the Stochastic VRPSPD.
Research implementation: https://github.com/vinhqdang/stochastic_vrp
```

## Contact

**Vinh Dang** — [dqvinh87@gmail.com](mailto:dqvinh87@gmail.com)

## License

MIT License — see [LICENSE](LICENSE) for details.