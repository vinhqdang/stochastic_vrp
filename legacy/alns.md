# Stochastic VRPSPD Solver

**Adaptive Large Neighborhood Search (ALNS) with Moment-Based Distributionally Robust Optimization (M-DRO) for the Vehicle Routing Problem with Simultaneous Pickup & Delivery under Spatially Correlated Demand Uncertainty.**

---

## What This Solves

A fleet of vehicles must serve customers by **delivering goods AND picking up returns** simultaneously. Demand is **uncertain** — customers may need more or less than expected — and nearby customers tend to fluctuate together (spatial contagion: e.g., a regional event affects an entire neighborhood).

The solver finds routes that are **short** (minimize distance) and **robust** (unlikely to overload vehicles even when demand deviates from expectations).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: CVRPLIB .vrp files (delivery-only benchmarks)   │
│         + optional .sol files (Best Known Solutions)     │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  AUGMENTATION                                           │
│  • Add random pickup demand: P_i = D_i × U(0.5, 1.5)   │
│  • Compute demand variance: σ²_i = (CV × (P+D))²       │
│  • Build spatial covariance: Σ_ij = σ_iσ_j·exp(-d/θ)   │
│  • Relax capacity: C × 1.2 (compensate for pickup)      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ALNS SEARCH ENGINE                                     │
│  • Destroy: Random Removal, Worst Removal               │
│  • Repair: Greedy Insertion, Regret-2 Insertion          │
│  • Adaptive weights (roulette wheel, Ropke&Pisinger '06) │
│  • Simulated Annealing acceptance + reheat               │
│  • Convergence history logged per instance               │
│       ▼                                                  │
│  M-DRO ROUTE EVALUATOR (called millions of times)       │
│  • Prefix-sum covariance tracking for fast Var(S_k)     │
│  • Cantelli-Chebyshev bound for worst-case P(overload)  │
│  • Union bound → Route Risk Index (RRI)                 │
│  • Objective: Z = distance + penalty(RRI)               │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  MULTI-SCENARIO MONTE CARLO VALIDATION                  │
│  • Gaussian Copula base with spatial correlation        │
│  • 4 marginal distributions tested simultaneously:      │
│    1. Gaussian (symmetric baseline)                     │
│    2. Skew-Right (skewnorm α=+5, retail-like)           │
│    3. Skew-Left  (skewnorm α=−5)                        │
│    4. Heavy-Tail (Student-t df=3, stress test)          │
│  • Empirical failure rate per scenario                  │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT                                                 │
│  • results_stochastic.csv / results_deterministic.csv   │
│  • convergence_{instance}.csv (ALNS Z-cost per iter)    │
│  Per instance: distance, vehicles, 4 failure rates, time│
└─────────────────────────────────────────────────────────┘
```

---

## Two Modes

A single toggle (`STOCHASTIC_MODE = True/False`) switches the entire pipeline:

| Aspect | Deterministic (`False`) | Stochastic (`True`) |
|--------|------------------------|---------------------|
| Problem | Classic CVRP (delivery only) | VRPSPD with random pickup |
| Capacity | Original from .vrp file | Original × 1.2 |
| Demand | Fixed, known exactly | Mean known, variance modeled |
| Objective | Minimize total distance | Distance + DRO risk penalty |
| Feasibility | Hard capacity check | Cantelli bound + penalty |
| Monte Carlo | Skipped | 4 distributional scenarios |
| Gap vs BKS | Meaningful (same problem) | N/A (different problem) |

---

## Monte Carlo: Four Scenarios via Gaussian Copula

The validation doesn't just test Gaussian. It uses a **Gaussian copula** (preserving the spatial correlation structure from the covariance matrix) combined with four different **marginal distributions**:

| Scenario | Marginal | Purpose |
|----------|----------|---------|
| `GAUSSIAN` | Normal | Symmetric baseline, the "kind" universe |
| `SKEW_RIGHT` | Skew-normal (α = +5) | Realistic for retail: most orders normal, occasional spikes |
| `SKEW_LEFT` | Skew-normal (α = −5) | Opposite skew, tests asymmetry sensitivity |
| `HEAVY_TAIL` | Student-t (df = 3) | Stress test: fat tails, extreme demand shocks |

All four share the **same correlation structure** (from the copula) but differ in how extreme individual demands can be. If the DRO surrogate (Cantelli bound) works, `RRI ≥ empirical failure rate` should hold across all four scenarios.

---

## Mathematical Foundations

### What is proven (theorems)

| Component | Basis | Reference |
|-----------|-------|-----------|
| Covariance matrix is PSD | Schoenberg: exp(-\|\|x-y\|\|) is PD on Euclidean space + Schur product theorem. **Only valid with Euclidean distances.** | Schoenberg (1938), Schur (1911) |
| Cantelli bound: σ²/(σ²+a²) | Tight worst-case over ALL distributions with given mean and variance | Cantelli (1928), Delage & Ye (2010) |
| Variance recurrence | Var(A+B) = Var(A) + Var(B) + 2·Cov(A,B) | Linear algebra |
| RRI ≥ P(any overload) | Union bound (Boole's inequality). Valid but can be loose. | Probability theory |

### What is heuristic (self-set, needs tuning)

| Parameter | Value | Role | How to validate |
|-----------|-------|------|-----------------|
| CV | 0.2 | Demand noise level (20%) | Estimate from historical data |
| θ = 0.1 × d_max | — | Spatial correlation range | Variogram fitting |
| α_base = 0.05 | — | Base risk threshold | Ablation study |
| γ = 0.5 | — | Threshold scaling exponent | Ablation study |
| λ₀ = 10,000 | — | Penalty strength | Ablation study |
| 1/ln(m+1) | — | Length-dependent penalty scaling | Ablation study |
| SA geometric cooling | 0.9997 | Temperature schedule | Hajek (1988) proves convergence to global optimum only with logarithmic cooling T_k = Γ/ln(k), which requires ~e^10000 iterations — completely impractical. Geometric cooling (T *= 0.9997) is the standard trade-off in applied OR: no theoretical guarantee, but reaches good solutions in feasible time. Nearly all published ALNS/SA implementations use geometric cooling for this reason. |
| Reheat at T < 0.1 | → 30% of T₀ | Escape frozen states | Empirical |
| Operator scores σ₁,σ₂,σ₃ | 33, 9, 13 | Adaptive weight rewards | Ropke & Pisinger (2006) |

### Known limitations

1. **Union bound is loose** when demands are positively correlated (which they are in this model). RRI overestimates true failure probability. It works as a ranking surrogate, not a probability estimate.

2. **Geometric SA cooling has no convergence guarantee.** Hajek (1988) requires logarithmic cooling, which is impractically slow. Geometric cooling is standard applied OR practice.

3. **Schoenberg PSD proof requires Euclidean distances.** If you replace coordinates with road-network distances, the covariance matrix may not be PSD.

4. **Capacity × 1.2 is a convention**, not an optimal value. In stochastic mode, BKS and ALNS use different capacity → gap comparison is N/A.

---

## Installation

```bash
pip install numpy scipy
```

No other dependencies. Python 3.8+.

---

## Usage

### Quick start

1. Place `.vrp` files (and optional `.sol` files) in a directory.

2. Edit the top of `alns.py`:
```python
STOCHASTIC_MODE = True   # or False for deterministic
TARGET_DIR = r"path/to/your/vrp/files"
```

3. Run:
```bash
python alns.py
```

### Environment variable (alternative)

```bash
export VRP_DATA_DIR=/path/to/vrp/files
python alns.py
```

### Quick test vs production

| Setting | Quick test | Production |
|---------|-----------|------------|
| `ALNS_ITERATIONS` | 500 | 10000–25000 |
| `MC_SAMPLES` | 1000 | 10000–50000 |
| Expected runtime per instance | seconds | minutes–hours |

---

## Output Files

### `results_stochastic.csv` / `results_deterministic.csv`

| Column | Description |
|--------|-------------|
| Instance | Filename without extension |
| Mode | STOCHASTIC or DETERMINISTIC |
| N | Number of customers |
| Capacity_Used | Vehicle capacity (possibly inflated) |
| CV | Coefficient of variation used |
| BKS_Vehicles | Vehicles in Best Known Solution |
| BKS_Distance | Total distance of BKS routes |
| ALNS_Vehicles | Vehicles found by ALNS |
| ALNS_Distance | Total distance of ALNS routes |
| ALNS_Z_Cost | Objective value (distance + penalty) |
| Gap_vs_BKS_% | Distance gap (deterministic only, N/A otherwise) |
| Fail_Gaussian | Failure rate under Gaussian scenario |
| Fail_SkewRight | Failure rate under right-skewed scenario |
| Fail_SkewLeft | Failure rate under left-skewed scenario |
| Fail_HeavyTail | Failure rate under heavy-tail scenario |
| Runtime_s | Wall-clock time in seconds |
| Seed | RNG seed for this instance |

### `convergence_{instance}.csv`

One file per instance. Columns: `Iteration, Best_Z_Cost`. Use for plotting ALNS convergence curves to verify the solver is improving over time and not stuck.

---

## File Structure

```
├── alns.py                        # The solver (single file, self-contained)
├── README.md                      # This document
├── data/                          # Your .vrp and .sol files
│   ├── X-n101-k25.vrp
│   ├── X-n101-k25.sol
│   └── ...
└── output/                        # Generated by the solver
    ├── results_stochastic.csv
    ├── results_deterministic.csv
    ├── convergence_X-n101-k25.csv
    └── ...
```

---

## Reproducibility

- Each instance uses seed `SEED + file_index` (default SEED = 42).
- Adding or removing `.vrp` files does **not** change other instances' results.
- The `Seed` column in the CSV output lets you reproduce any specific instance.
- Requires: same NumPy version (RNG stream may differ across major versions).

---

## Validating the ALNS Engine

To verify correctness of the core ALNS (independent of stochastic layer):

1. Set `STOCHASTIC_MODE = False`
2. Run on small instances (N ≤ 25)
3. Compare ALNS distance vs BKS optimal distance
4. Expected gap: 0–3% for small instances

A separate Gurobi MILP formulation was used on N=10 toy instances for exact validation. ALNS matched the Gurobi optimal (gap = 0%).

---

## References

1. Cantelli, F.P. (1928). *Sui confini della probabilità.*
2. Schoenberg, I.J. (1938). *Metric spaces and positive definite functions.* Trans. AMS.
3. Ropke, S. & Pisinger, D. (2006). *An ALNS heuristic for the PDPTW.* Transportation Science 40(4).
4. Hajek, B. (1988). *Cooling schedules for optimal annealing.* Math. Oper. Res. 13(2).
5. Delage, E. & Ye, Y. (2010). *Distributionally robust optimization under moment uncertainty.* Oper. Res. 58(3).
6. Schur, I. (1911). Hadamard product of PSD matrices is PSD.

---

## License

MIT
