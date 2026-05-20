"""Day 9 — Filter speedup measurement on CON3-0 with REAL variance.

Setup:
    - Scenarios: Gamma distribution (cv=0.2) on nominal demand, N=1000
    - epsilon = 10000 (small but non-zero, scaled to Q)
    - alpha = 0.9
    - Filter lambda tuned to match T trajectory: lambda = T_final * N

Compares:
    - Baseline (no filter): plain W-DRO ALNS
    - Filtered: W-DRO ALNS with Phase 2 Brown filter

Expected:
    - Prune rate > 20% (filter actually active)
    - Wall-clock speedup > 1.5x
    - Cost gap < 2% (filter preserves quality)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scenarios import ScenarioConfig, generate_scenarios
from core.seeding import phase0_wdro_seeding
from core.alns_wdro import (
    WDROConfig, alns_sa_wdro,
    WDROFilteredConfig, alns_sa_wdro_filtered,
)
from core.filter import FilterConfig

# Reuse the working parser from Test 1 script
from scripts.run_test1_wdro import parse_con3_benchmark


def main():
    inst_path = Path(__file__).parent.parent / "data" / "CON3-0.txt"
    inst = parse_con3_benchmark(inst_path)
    print("=" * 70)
    print(inst.summary())
    print("=" * 70)

    # --- Stochastic scenarios ---
    N = 1000
    cv = 0.2
    cfg = ScenarioConfig(distribution="gamma", cv=cv, seed=42)
    scenarios = generate_scenarios(inst, N, cfg)
    print(f"\nGenerated N={N} scenarios (Gamma, cv={cv})")

    # --- Common hyperparameters ---
    alpha = 0.9
    epsilon = 10000.0  # small relative to Q ~ 8M, but non-zero
    max_iters = 5000
    time_limit = 120.0
    T_init_frac = 0.05
    alpha_cooling = 0.9997

    print(f"alpha={alpha}, epsilon={epsilon}, max_iters={max_iters}\n")

    # ---------- Baseline (no filter) ----------
    print("[1/2] BASELINE (no filter)...")
    initial = phase0_wdro_seeding(inst, alpha=alpha, epsilon=epsilon)
    base_cfg = WDROConfig(
        alpha=alpha, epsilon=epsilon, penalty_lambda=1.0,
        max_iters=max_iters,
        T_init_frac=T_init_frac, alpha_cooling=alpha_cooling,
        seed=42, verbose_every=1000,
        time_limit_sec=time_limit,
        max_vehicles=inst.n_vehicles,
    )
    t0 = time.time()
    base_result = alns_sa_wdro(initial, inst, scenarios, base_cfg)
    base_time = time.time() - t0

    print(f"\n  Iterations:    {base_result.n_iters}")
    print(f"  Elapsed:       {base_time:.1f}s")
    print(f"  Best obj:      {base_result.best_objective:.0f}")
    print(f"  Travel:        {base_result.best_breakdown['travel']:.0f}")
    print(f"  W-DRO Phi:     {base_result.best_breakdown['wdro_penalty']:.2f}")

    # ---------- Filtered ----------
    print("\n[2/2] FILTERED (Phase 2 Brown filter)...")

    # Estimate T trajectory to tune lambda
    initial_obj_est = base_result.cost_history[0]
    T_init_est = T_init_frac * initial_obj_est
    T_final_est = T_init_est * (alpha_cooling ** max_iters)
    # Target: n_0 in [10, N] over run. Use lambda = T_final * N.
    filter_lambda = max(T_final_est * N, 1000.0)
    print(f"  T_init={T_init_est:.0f}  T_final={T_final_est:.0f}")
    print(f"  Tuned filter lambda = {filter_lambda:.0f}")

    initial2 = phase0_wdro_seeding(inst, alpha=alpha, epsilon=epsilon)
    filt_cfg = WDROFilteredConfig(
        alpha=alpha, epsilon=epsilon, penalty_lambda=1.0,
        max_iters=max_iters,
        T_init_frac=T_init_frac, alpha_cooling=alpha_cooling,
        seed=42, verbose_every=1000,
        time_limit_sec=time_limit,
        max_vehicles=inst.n_vehicles,
        filter_cfg=FilterConfig(
        kappa=2.0,
        lambda_=filter_lambda,
        n0_min=20,
        n0_max=50,     # ← thêm dòng này
        enabled=True,
    ),
        enable_diagnostics=True,
    )
    t0 = time.time()
    filt_result, diag = alns_sa_wdro_filtered(initial2, inst, scenarios, filt_cfg)
    filt_time = time.time() - t0

    print(f"\n  Iterations:    {filt_result.n_iters}")
    print(f"  Elapsed:       {filt_time:.1f}s")
    print(f"  Best obj:      {filt_result.best_objective:.0f}")
    print(f"  Travel:        {filt_result.best_breakdown['travel']:.0f}")
    print(f"  W-DRO Phi:     {filt_result.best_breakdown['wdro_penalty']:.2f}")
    print(f"  Proxy evals:   {diag.n_proxy_evals}")
    print(f"  Exact evals:   {diag.n_exact_evals}")
    print(f"  Pruned:        {diag.n_pruned}")
    print(f"  Prune rate:    {diag.prune_rate:.2%}")

    # ---------- Summary ----------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Wall-clock speedup:      {base_time / filt_time:.2f}x")
    print(f"  Iters/sec baseline:      {base_result.n_iters / base_time:.1f}")
    print(f"  Iters/sec filtered:      {filt_result.n_iters / filt_time:.1f}")
    print(f"  Iters speedup:           "
          f"{(filt_result.n_iters / filt_time) / (base_result.n_iters / base_time):.2f}x")
    cost_gap = (filt_result.best_objective - base_result.best_objective) / base_result.best_objective
    print(f"  Cost gap:                {100 * cost_gap:+.2f}%")
    print(f"  Filter prune rate:       {diag.prune_rate:.2%}")
    print()
    print("Filter activity checks:")
    print(f"  Prune rate > 20%:        {'PASS' if diag.prune_rate > 0.20 else 'FAIL'}")
    print(f"  Cost gap < 2%:           {'PASS' if abs(cost_gap) < 0.02 else 'FAIL'}")
    print(f"  Wall-clock speedup > 1x: {'PASS' if base_time / filt_time > 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()