"""Day 9 Part C — Test 3: Out-of-sample distributional stress.

Train W-DRO solution on Gamma scenarios. Test on shifted distributions:
    (1) Gamma cv=0.2  (in-distribution, sanity)
    (2) LogNormal cv=0.2  (heavier right tail)
    (3) Bimodal cv=0.2  (multi-modal)

Compare:
    - SAA  (epsilon = 0)        : empirical CVaR only
    - W-DRO (epsilon > 0)       : Wasserstein-protected, sweep ε

Key metric: violation rate = P(any route peak > Q) on test scenarios.
Claim: W-DRO < SAA on shifted distributions (key paper result, Table 2).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scenarios import ScenarioConfig, generate_scenarios
from core.seeding import phase0_wdro_seeding
from core.alns_wdro import WDROConfig, alns_sa_wdro
from core.wdro_exact import evaluate_phi_total

from scripts.run_test1_wdro import parse_con3_benchmark


# ============================================================
# Out-of-sample evaluation
# ============================================================


def compute_peaks(route, scenarios, n):
    """Returns array (N,) of peak loads for route under each scenario."""
    return route.peak_loads_batch(scenarios, n)


def evaluate_oos(routes, inst, scenarios, alpha):
    """Out-of-sample metrics for a multi-route solution.

    All computed with epsilon = 0 (pure empirical, no W-DRO regularization).
    """
    N = scenarios.shape[0]
    Q = inst.Q
    n = inst.n
    D = inst.distances()

    # Peaks: (R, N)
    peaks_all = np.array([compute_peaks(r, scenarios, n) for r in routes])

    # Per-route violation rate
    per_route_viol = (peaks_all > Q).mean(axis=1)

    # Any-route violation per scenario
    any_viol_per_scenario = (peaks_all > Q).any(axis=0)
    violation_rate = float(any_viol_per_scenario.mean())

    # Worst peak across all routes & scenarios
    worst_peak = float(peaks_all.max())
    overflow_pct = max(0.0, (worst_peak - Q) / Q * 100)

    # Empirical CVaR penalty (eps=0)
    phi_total = evaluate_phi_total(routes, inst, scenarios, alpha, epsilon=0.0)


    # Travel
    travel = 0.0
    for r in routes:
        prev = 0
        for c in r.customers:
            travel += D[prev, c + 1]
            prev = c + 1
        travel += D[prev, 0]

    return {
        "violation_rate": violation_rate,
        "worst_peak": worst_peak,
        "overflow_pct": overflow_pct,
        "phi_total": phi_total,
        "travel": travel,
        "per_route_viol": per_route_viol.tolist(),
    }


# ============================================================
# Main
# ============================================================


def main():
    inst_path = Path(__file__).parent.parent / "data" / "CON3-0.txt"
    inst = parse_con3_benchmark(inst_path)
    print("=" * 78)
    print(inst.summary())
    print("=" * 78)

    alpha = 0.9
    cv = 0.2
    N_train = 1000
    N_test = 5000

    # Train scenarios: Gamma
    train_scenarios = generate_scenarios(
        inst, N_train,
        ScenarioConfig(distribution="gamma", cv=cv, seed=42),
    )

    # Test scenario sets (different seed than train)
    test_configs = {
        "Gamma (in-dist)":  ScenarioConfig(distribution="gamma",     cv=cv, seed=2026),
        "LogNormal (shift)": ScenarioConfig(distribution="lognormal", cv=cv, seed=2026),
        "Bimodal (shift)":   ScenarioConfig(
            distribution="bimodal", cv=cv, bimodal_spread=0.5, bimodal_cv=0.1, seed=2026,
        ),
    }
    test_scenarios = {
        name: generate_scenarios(inst, N_test, cfg)
        for name, cfg in test_configs.items()
    }

    print(f"\nTrain: Gamma cv={cv}, N_train={N_train}")
    print(f"Test:  {list(test_configs.keys())}, N_test={N_test}\n")

    # Algorithms
    algorithms = {
        "SAA (eps=0)":      0.0,
        "W-DRO eps=1e3":    1e3,
        "W-DRO eps=1e4":    1e4,
        "W-DRO eps=1e5":    1e5,
    }

    # Train each algorithm
    trained = {}
    for name, eps in algorithms.items():
        print(f"[Training] {name}...")
        initial = phase0_wdro_seeding(inst, alpha=alpha, epsilon=eps)
        cfg = WDROConfig(
            alpha=alpha, epsilon=eps, penalty_lambda=1.0,
            max_iters=3000,
            T_init_frac=0.05, alpha_cooling=0.9995,
            seed=42, verbose_every=0,
            time_limit_sec=90.0,
            max_vehicles=inst.n_vehicles,
        )
        t0 = time.time()
        result = alns_sa_wdro(initial, inst, train_scenarios, cfg)
        print(f"  done in {time.time()-t0:.1f}s  "
              f"obj={result.best_objective:.0f} "
              f"travel={result.best_breakdown['travel']:.0f} "
              f"phi={result.best_breakdown['wdro_penalty']:.2f}  "
              f"routes={len(result.best_solution)}")
        trained[name] = result

    # Out-of-sample evaluation
    print()
    print("=" * 78)
    print("OUT-OF-SAMPLE EVALUATION (eps=0 for all measurements)")
    print("=" * 78)
    print()

    test_names = list(test_configs.keys())
    col_w = 24
    print(f"{'Algorithm':<18} {'Travel':>10}", end="")
    for tn in test_names:
        print(f" | {tn:<{col_w}}", end="")
    print()
    print(f"{'':<18} {'':>10}", end="")
    for _ in test_names:
        print(f" | {'viol%':>7} {'over%':>6} {'phi':>9}", end="")
    print()
    print("-" * (18 + 11 + (col_w + 3) * len(test_names)))

    summary_table = {}
    for algo_name in algorithms.keys():
        result = trained[algo_name]
        sol = result.best_solution
        travel = result.best_breakdown["travel"]
        print(f"{algo_name:<18} {travel:>10.0f}", end="")
        row = {"travel": travel}
        for test_name in test_names:
            metrics = evaluate_oos(sol, inst, test_scenarios[test_name], alpha)
            v = metrics["violation_rate"] * 100
            o = metrics["overflow_pct"]
            p = metrics["phi_total"]
            print(f" | {v:>6.2f}% {o:>5.1f}% {p:>9.0f}", end="")
            row[test_name] = metrics
        print()
        summary_table[algo_name] = row

    # Key result summary
    print()
    print("=" * 78)
    print("KEY RESULT (Manuscript Table 2)")
    print("=" * 78)
    saa = summary_table["SAA (eps=0)"]
    best_wdro_name = None
    best_shift_viol = float("inf")
    for name in algorithms:
        if name.startswith("W-DRO"):
            sh_viol = max(
                summary_table[name]["LogNormal (shift)"]["violation_rate"],
                summary_table[name]["Bimodal (shift)"]["violation_rate"],
            )
            if sh_viol < best_shift_viol:
                best_shift_viol = sh_viol
                best_wdro_name = name

    print(f"  SAA violation on shifts:    "
          f"LogNormal={saa['LogNormal (shift)']['violation_rate']*100:.2f}%, "
          f"Bimodal={saa['Bimodal (shift)']['violation_rate']*100:.2f}%")
    if best_wdro_name:
        wd = summary_table[best_wdro_name]
        print(f"  Best W-DRO ({best_wdro_name}): "
              f"LogNormal={wd['LogNormal (shift)']['violation_rate']*100:.2f}%, "
              f"Bimodal={wd['Bimodal (shift)']['violation_rate']*100:.2f}%")
        travel_premium = (wd["travel"] - saa["travel"]) / saa["travel"] * 100
        print(f"  Travel premium for W-DRO:   {travel_premium:+.2f}%")
        print()
        print("Claim verification:")
        for shift_name in ["LogNormal (shift)", "Bimodal (shift)"]:
            saa_v = saa[shift_name]["violation_rate"]
            wd_v = wd[shift_name]["violation_rate"]
            status = "PASS (W-DRO < SAA)" if wd_v < saa_v else "FAIL"
            print(f"  {shift_name}: SAA={saa_v*100:.2f}%  W-DRO={wd_v*100:.2f}%  {status}")


if __name__ == "__main__":
    main()