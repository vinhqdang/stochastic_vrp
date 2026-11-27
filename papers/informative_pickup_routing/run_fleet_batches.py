"""Verify the fleet batch algorithm and reproduce the synthetic experiments."""

from itertools import combinations, product
from math import comb, fsum, isclose, prod
from random import Random

from fleet_batches import Fleet, Signal, expected_standby_savings


def enumerate_savings(laws, standby):
    result = 0.0
    for scenario in product(*laws):
        values = sorted((max(0, value) for value, _ in scenario), reverse=True)
        result += prod(p for _, p in scenario) * fsum(values[:standby])
    return result


def greedy_singletons(fleet):
    observed = set()
    value = 0.0
    while len(observed) < len(fleet.signals):
        remaining = set(range(len(fleet.signals))) - observed
        best_value, best_route = max(
            (fleet.net_value(observed | {i}), i) for i in remaining
        )
        if best_value <= value + 1e-12:
            break
        observed.add(best_route)
        value = best_value
    return value


def check_small_fleets():
    rng = Random(615)
    ratios = []
    for size in (3, 5, 8):
        for _ in range(12):
            standby = rng.randrange(size + 1)
            signals = []
            for _ in range(size):
                low, high = sorted(rng.sample(range(-3, 11), 2))
                probability = rng.choice((0.2, 0.5, 0.7))
                cost = rng.choice((0, 0.03, 0.12, 0.45))
                signals.append(Signal(((low, probability), (high, 1 - probability)), cost))
            laws = [signal.outcomes for signal in signals]
            expected = enumerate_savings(laws, standby)
            assert isclose(expected_standby_savings(laws, standby), expected, abs_tol=1e-11)
            fleet = Fleet(signals, standby)
            batch = fleet.select_batch()
            optimum = max(
                fleet.net_value(subset)
                for count in range(size + 1)
                for subset in combinations(range(size), count)
            )
            assert batch.net_saving <= optimum + 1e-10
            assert optimum <= batch.upper_bound + 1e-10
            if optimum > 1e-10:
                ratios.append(batch.net_saving / optimum)
    example = Fleet((Signal(((4, 0.5), (8, 0.5)), 0.0625),
                     Signal(((0, 0.5), (5, 0.5)), 0.0625)), 1)
    batch = example.select_batch()
    assert batch.exchange_degree == 1
    assert batch.routes == (0, 1)
    assert isclose(batch.net_saving, 0.125, abs_tol=1e-11)
    assert isclose(batch.net_saving, batch.upper_bound, abs_tol=1e-11)
    print("Verified threshold evaluation and batch bounds against enumeration on 36 small fleets.")
    print(f"Positive-optimum fleets: {len(ratios)}; mean and minimum fraction attained: "
          f"{fsum(ratios) / len(ratios):.2%}, {min(ratios):.2%}")


def check_complementary_fleets():
    print("Routes  Standby  Selected  Net saving  Upper bound  Singleton greedy")
    first = Signal(((4, 0.5), (8, 0.5)), 0.0625)
    second = Signal(((0, 0.5), (5, 0.5)), 0.0625)
    for standby in (4, 16, 64, 128):
        fleet = Fleet((first,) * standby + (second,) * standby, standby)
        batch = fleet.select_batch()
        greedy = greedy_singletons(fleet)
        full_value = standby / 2 * (1 - comb(2 * standby, standby) / 4**standby)
        expected = full_value - 2 * standby * first.cost
        assert len(batch.routes) == 2 * standby
        assert isclose(batch.net_saving, expected, abs_tol=1e-10)
        assert isclose(batch.upper_bound, full_value, abs_tol=1e-10)
        assert greedy == 0
        print(f"{2 * standby:>6}  {standby:>7}  {len(batch.routes):>8}  "
              f"{batch.net_saving:>10.4f}  {batch.upper_bound:>11.4f}  {greedy:>16.4f}")


if __name__ == "__main__":
    check_small_fleets()
    check_complementary_fleets()
