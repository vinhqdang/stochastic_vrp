"""

Runs on CON3-0 with collapse settings (variance=0, epsilon=0) and reports:
    - Wall-clock time comparison
    - Best cost found (should be similar)
    - Filter prune rate (% of candidates skipped)
    - Speedup factor
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.instance import Instance
from core.scenarios import ScenarioConfig, generate_scenarios
from core.seeding import phase0_wdro_seeding
from core.alns_wdro import (
    WDROConfig, alns_sa_wdro,
    WDROFilteredConfig, alns_sa_wdro_filtered,
)
from core.filter import FilterConfig


def run_baseline(inst, scenarios, max_iters, time_limit):
    """W-DRO ALNS WITHOUT filter."""
    initial = phase0_wdro_seeding(inst, alpha=0.9, epsilon=0.0)
    config = WDROConfig(
        alpha=0.9, epsilon=0.0, penalty_lambda=1.0,
        max_iters=max_iters,
        T_init_frac=0.05, alpha_cooling=0.9997,
        seed=42, verbose_every=0,
        time_limit_sec=time_limit,
        max_vehicles=inst.n_vehicles,
    )
    t0 = time.time()
    result = alns_sa_wdro(initial, inst, scenarios, config)
    return result, time.time() - t0


def run_filtered(inst, scenarios, max_iters, time_limit, filter_cfg):
    """W-DRO ALNS WITH filter."""
    initial = phase0_wdro_seeding(inst, alpha=0.9, epsilon=0.0)
    config = WDROFilteredConfig(
        alpha=0.9, epsilon=0.0, penalty_lambda=1.0,
        max_iters=max_iters,
        T_init_frac=0.05, alpha_cooling=0.9997,
        seed=42, verbose_every=0,
        time_limit_sec=time_limit,
        max_vehicles=inst.n_vehicles,
        filter_cfg=filter_cfg,
        enable_diagnostics=True,
    )
    t0 = time.time()
    result, diag = alns_sa_wdro_filtered(initial, inst, scenarios, config)
    return result, diag, time.time() - t0


def main():
    inst_path = Path(__file__).parent.parent / "data" / "CON3-0.txt"
    # Use the same parser the user has working
    from scripts.run_test1_wdro import parse_con3_benchmark
    inst = parse_con3_benchmark(inst_path)
    print("=" * 70)
    print(inst.summary())
    print("=" * 70)

    cfg = ScenarioConfig(distribution="constant", cv=0.0)
    scenarios = generate_scenarios(inst, 100, cfg)

    max_iters = 5000
    time_limit = 60.0  # 1 min per run

    # --- Baseline (no filter) ---
    print("\n[1/2] Running BASELINE (no filter)...")
    base_result, base_time = run_baseline(inst, scenarios, max_iters, time_limit)
    print(f"  Iterations:    {base_result.n_iters}")
    print(f"  Elapsed:       {base_time:.1f}s")
    print(f"  Best obj:      {base_result.best_objective:.0f}")
    print(f"  Travel:        {base_result.best_breakdown['travel']:.0f}")

    # --- Filtered ---
    print("\n[2/2] Running FILTERED (Phase 2)...")
    filter_cfg = FilterConfig(
        kappa=2.0,
        lambda_=100.0,
        n0_min=10,
        n0_max=None,
        enabled=True,
    )
    filt_result, diag, filt_time = run_filtered(
        inst, scenarios, max_iters, time_limit, filter_cfg,
    )
    print(f"  Iterations:    {filt_result.n_iters}")
    print(f"  Elapsed:       {filt_time:.1f}s")
    print(f"  Best obj:      {filt_result.best_objective:.0f}")
    print(f"  Travel:        {filt_result.best_breakdown['travel']:.0f}")
    print(f"  Proxy evals:   {diag.n_proxy_evals}")
    print(f"  Exact evals:   {diag.n_exact_evals}")
    print(f"  Pruned:        {diag.n_pruned}")
    print(f"  Prune rate:    {diag.prune_rate:.2%}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Speedup (wall-clock):    {base_time / filt_time:.2f}x")
    print(f"  Iters per sec (base):    {base_result.n_iters / base_time:.1f}")
    print(f"  Iters per sec (filter):  {filt_result.n_iters / filt_time:.1f}")
    print(f"  Iters speedup:           "
          f"{(filt_result.n_iters / filt_time) / (base_result.n_iters / base_time):.2f}x")
    print(f"  Cost gap (filter vs base): "
          f"{100 * (filt_result.best_objective - base_result.best_objective) / base_result.best_objective:+.2f}%")
    print(f"  Filter prune rate:       {diag.prune_rate:.2%}")
    print()
    print("Filter quality check (cost gap should be < 5%):")
    cost_gap_pct = abs(filt_result.best_objective - base_result.best_objective) / base_result.best_objective
    print(f"  Cost gap < 5%:  {'PASS' if cost_gap_pct < 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()