"""Check the optimizer and compare policies on a regional demand model."""

from collections import Counter
from dataclasses import replace
from itertools import product
from math import dist, isclose, prod
from random import Random

from early_visits import Branch, Policy, Travel, certified_improvement, fit, immediate, mean_saving
from pilot import distribution_weight, optimal_gain


def regional_model():
    """Generate a finite law with noisy region and demand-severity signals."""
    cases = Counter()
    for east_active, severe in product((False, True), repeat=2):
        regime_mass = 0.5 * (0.35 if severe else 0.65)
        active_rate = 0.95 if severe else 0.05
        rates = (
            0.96 if east_active else 0.04,
            active_rate if east_active else 0.5,
            active_rate if not east_active else 0.5,
            0.58 if severe else 0.48,
            0.68 if severe else 0.36,
        )
        # The two downstream pickups are binary, with one unit of free capacity.
        overload_probability = (0.90 if severe else 0.22) ** 2
        for signals in product((0, 1), repeat=len(rates)):
            signal_mass = prod(p if x else 1 - p for x, p in zip(signals, rates))
            for overloaded in (False, True):
                p = overload_probability if overloaded else 1 - overload_probability
                cases[signals, overloaded] += regime_mass * signal_mass * p

    locations = ((1, 0), (3, 2), (3, -2), (0, 1), (4.7, 0))
    travel = Travel(
        tuple(dist((0, 0), point) for point in locations),
        tuple(tuple(dist(a, b) for b in locations) for a in locations),
        deadline=5.5,
        cost_per_minute=0.05,
    )
    return cases, travel


def brute_force(cases, travel, dispatch_cost, overload_cost, adaptive):
    """Enumerate executable policies independently on a tiny instance."""
    policies = [immediate(False), immediate(True)]
    for first in travel.next_stops(None):
        options = [Branch(None, (a, a)) for a in (False, True)]
        for second in travel.next_stops(first):
            options.extend(
                Branch(second, actions) for actions in product((False, True), repeat=2)
            )
        policies.extend(
            Policy(first, (left, right))
            for left, right in product(options, repeat=2)
            if adaptive or len({left.second, right.second} - {None}) <= 1
        )
    return max(
        mean_saving(policy, cases, travel, dispatch_cost, overload_cost)
        for policy in policies
    )


def check_optimizer():
    random = Random(312)
    for _ in range(12):
        travel = Travel(
            (1, 2, 4),
            ((0, 1, 4), (1, 0, 3), (4, 3, 0)),
            random.choice((0, 3, 5, 7)),
            random.choice((0, 0.1, 0.5)),
        )
        cases = {
            (signals, y): random.randint(0, 9)
            for signals in product((0, 1), repeat=3)
            for y in (0, 1)
        }
        for adaptive in (True, False):
            policy, gain = fit(cases, travel, 1.28, 4, adaptive=adaptive)
            expected = brute_force(cases, travel, 1.28, 4, adaptive)
            assert isclose(gain, expected, abs_tol=1e-12)
            assert isclose(gain, mean_saving(policy, cases, travel, 1.28, 4), abs_tol=1e-12)


def check_original_pilot():
    pair_count, alpha, eta = 3, 0.04, 0.6
    count = 2 * pair_count
    cases = Counter()
    for mask in range(1 << (count + 2)):
        signals = tuple((mask >> index) & 1 for index in range(count))
        y = (mask >> count) == 3
        cases[signals, y] += distribution_weight(mask, count + 2, alpha, eta)
    travel = Travel((1,) * count, ((1,) * count,) * count, deadline=2, cost_per_minute=0)
    _, actual = fit(cases, travel, 1 + alpha, 4)
    expected = optimal_gain(pair_count, alpha, [eta] + [0] * (pair_count - 1))
    assert isclose(actual, expected, abs_tol=1e-12)


def main():
    check_optimizer()
    check_original_pilot()
    print("Exhaustive policy checks passed on 12 small instances.")
    print("The general optimizer reproduced the original parity example.")
    cases, travel = regional_model()
    dispatch_cost, overload_cost = 1.28, 4
    assert isclose(sum(cases.values()), 1)
    scenarios = list(cases)
    weights = list(cases.values())
    training = Counter(Random(641).choices(scenarios, weights, k=4096))
    holdout = Counter(Random(281).choices(scenarios, weights, k=16384))
    optimal, optimal_gain = fit(cases, travel, dispatch_cost, overload_cost)
    _, fixed_gain = fit(cases, travel, dispatch_cost, overload_cost, adaptive=False)
    learned, _ = fit(training, travel, dispatch_cost, overload_cost)
    learned_gain = mean_saving(learned, cases, travel, dispatch_cost, overload_cost)
    lower = certified_improvement(learned, holdout, travel, dispatch_cost, overload_cost)
    best_immediate = max(0, overload_cost * sum(w for (_, y), w in cases.items() if y) - dispatch_cost)
    print(f"Optimal adaptive saving vs best immediate action: {optimal_gain - best_immediate:.6f}")
    print(f"Optimal fixed-order saving vs best immediate action: {fixed_gain - best_immediate:.6f}")
    print(f"Learned policy saving on the known law: {learned_gain - best_immediate:.6f}")
    print(f"95% holdout lower bound vs best immediate action: {lower:.6f}")
    names = ("district", "east", "west", "nearby", "remote")
    for label, policy in (("Optimal", optimal), ("Learned", learned)):
        first = "none" if policy.first is None else names[policy.first]
        branches = ["stop" if row.second is None else names[row.second] for row in policy.branches]
        print(f"{label} visits: {first}; next after signals 0 and 1: {', '.join(branches)}")

    print("Deadline  Adaptive saving  Fixed-order saving")
    for deadline in (2, 4, 5.5, 8):
        variant = replace(travel, deadline=deadline)
        _, adaptive_saving = fit(cases, variant, dispatch_cost, overload_cost)
        _, fixed_saving = fit(cases, variant, dispatch_cost, overload_cost, adaptive=False)
        print(f"{deadline:>8}  {adaptive_saving - best_immediate:>15.6f}  {fixed_saving - best_immediate:>18.6f}")


if __name__ == "__main__":
    main()
