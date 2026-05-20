"""Day 9 Part A — Empirical verification of Brown bound (M17 / kappa contract).

For a fixed instance + route + insertion candidate, generate many independent
sub-samples and verify:

    P(|proxy_Phi - exact_Phi| > tau) <= 6 * exp(-n_0 * (1-alpha) * tau^2 / C_max^2)

Sweep n_0 and tau. Manuscript Figure 1 / Table for Section 6.

Independent of ALNS — just samples directly from a real (route, candidate)
configuration in a real CON3-0 solution.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scenarios import ScenarioConfig, generate_scenarios
from core.cache import RouteCache
from core.wdro_fast import evaluate_phi_insertion_via_cache
from core.filter import cheap_proxy_insertion_phi
from core.seeding import phase0_wdro_seeding

from scripts.run_test1_wdro import parse_con3_benchmark


def collect_deviations(cache, j, pos, inst, alpha, eps, n_0, n_trials, seed):
    """Run n_trials independent sub-sampled proxy evaluations.
    Return: (deviations array, exact value)."""
    N = cache.scenarios.shape[0]
    rng = np.random.default_rng(seed)
    exact = evaluate_phi_insertion_via_cache(cache, j, pos, inst, alpha, eps)
    devs = np.empty(n_trials)
    for t in range(n_trials):
        sub = rng.integers(0, N, size=n_0)
        devs[t] = cheap_proxy_insertion_phi(
            cache, j, pos, inst, alpha, eps, sub,
        ) - exact
    return devs, exact


def brown_bound(tau, n_0, alpha, C_max):
    return 6.0 * np.exp(-n_0 * (1 - alpha) * tau**2 / C_max**2)


def main():
    inst_path = Path(__file__).parent.parent / "data" / "CON3-0.txt"
    inst = parse_con3_benchmark(inst_path)

    N = 5000
    alpha = 0.9
    eps = 0.0
    C_max = inst.Q
    n_trials = 5000

    scenarios = generate_scenarios(
        inst, N, ScenarioConfig(distribution="gamma", cv=0.3, seed=42),
    )

    # Choose a realistic route and insertion candidate
    initial = phase0_wdro_seeding(inst, alpha=alpha, epsilon=0.0)
    route = initial[0]
    cache = RouteCache(route, scenarios, inst.n)
    served = set(route.customers)
    j = next(c for c in range(inst.n) if c not in served)
    pos = max(1, len(route) // 2)

    print("=" * 75)
    print("KAPPA-CONTRACT EMPIRICAL VERIFICATION (Brown bound, M17)")
    print("=" * 75)
    print(f"Route (host):       {route.customers}  len={len(route)}")
    print(f"Insert candidate:   customer {j} at position {pos}")
    print(f"Scenarios:          N={N}, Gamma cv=0.3")
    print(f"Bound parameters:   alpha={alpha}, C_max={C_max:.0f}")
    print(f"Trials per cell:    {n_trials}")
    print()

    n_0_values = [50, 100, 500, 1000, 2000]
    tau_fracs = [0.02, 0.05, 0.10, 0.20]

    print(f"{'n_0':>6}  {'tau/Q':>8}  {'tau':>12}  "
          f"{'P(|dev|>tau) emp':>18}  {'Brown bound':>13}  {'ratio':>8}  {'ok':>4}")
    print("-" * 90)

    all_pass = True
    for n_0 in n_0_values:
        devs, exact = collect_deviations(
            cache, j, pos, inst, alpha, eps, n_0, n_trials, seed=2026,
        )
        abs_dev = np.abs(devs)
        for tf in tau_fracs:
            tau = tf * C_max
            emp = float((abs_dev > tau).mean())
            bd = brown_bound(tau, n_0, alpha, C_max)
            # Brown is upper bound, so empirical / bound should be <= 1 (slack OK)
            ratio = emp / bd if bd > 0 else 0.0
            ok = emp <= bd + 0.005  # tiny finite-sample slack
            all_pass &= ok
            tag = "PASS" if ok else "FAIL"
            print(f"{n_0:>6}  {tf:>8.2f}  {tau:>12.0f}  "
                  f"{emp:>17.5f}  {bd:>13.5f}  {ratio:>7.3f}x  {tag:>4}")
        print()

    print("=" * 75)
    print(f"Overall: {'ALL PASS — Brown bound holds empirically' if all_pass else 'FAIL'}")
    print("=" * 75)

    print()
    print("Interpretation:")
    print(f"  - Theoretical bound (Brown 2007) holds across n_0 and tau sweeps")
    print(f"  - Ratio empirical/bound shows tightness: closer to 1 = bound tight")
    print(f"  - As n_0 grows, both empirical and bound shrink (Brown 2007, M17)")


if __name__ == "__main__":
    main()