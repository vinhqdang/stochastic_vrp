"""Exact routing values and a synthetic audit experiment for binary pickups."""

from itertools import combinations, product
from math import ceil, isclose, log
from random import Random


OUTCOMES = (-1, 1)


def distribution_weight(mask, size, alpha, eta):
    """Density relative to independent fair signs, with probes 0 and 1 useful."""
    signed_sum = 2 * mask.bit_count() - size
    pair_sum = (signed_sum**2 - size) // 2
    outcome_bits = mask >> (size - 2)
    even_parity = ((mask & 3).bit_count() + outcome_bits.bit_count()) % 2 == 0
    return 1 + alpha * pair_sum + eta * (1 if even_parity else -1)


def joint_savings(alpha, interaction, first, second):
    """Weight the saving from dispatching by the probability of the two signs."""
    signs_product = first * second
    return (
        2 * alpha * (first + second) + (interaction - alpha**2) * signs_product
    ) / 4


def optimal_gain(pair_count, alpha, interactions):
    """Solve both steps exactly, letting the second visit depend on the first."""
    best_gain = 0.0
    for first in range(2 * pair_count):
        gain = 0.0
        for observed in OUTCOMES:
            best_branch = 0.0
            for second in range(2 * pair_count):
                if first == second:
                    continue
                interaction = (
                    interactions[first // 2] if first // 2 == second // 2 else 0
                )
                branch_gain = sum(
                    max(0, joint_savings(alpha, interaction, observed, result))
                    for result in OUTCOMES
                )
                best_branch = max(best_branch, branch_gain)
            gain += best_branch
        best_gain = max(best_gain, gain)
    return best_gain


def verify_by_enumeration():
    """Independently check full support, low-order marginals and outcome tables."""
    pair_count, alpha, eta = 3, 0.04, 0.6
    size = 2 * pair_count + 2
    states = list(product(OUTCOMES, repeat=size))
    moments = [
        tuple(indexes)
        for order in (1, 2, 3)
        for indexes in combinations(range(size), order)
    ]
    pairs = list(combinations(range(size - 2), 2))
    for useful in range(pair_count):
        weights = []
        for signs in states:
            sum_pairs = (sum(signs)**2 - size) // 2
            high_order = signs[2 * useful] * signs[2 * useful + 1]
            high_order *= signs[-2] * signs[-1]
            weights.append((1 + alpha * sum_pairs + eta * high_order) / len(states))

        assert min(weights) > 0 and isclose(sum(weights), 1)
        if useful == 0:
            for signs, weight in zip(states, weights):
                mask = sum(1 << index for index, sign in enumerate(signs) if sign == 1)
                assert isclose(
                    weight * len(states), distribution_weight(mask, size, alpha, eta)
                )
        for indexes in moments:
            actual = 0.0
            for signs, weight in zip(states, weights):
                parity = sum(signs[index] == -1 for index in indexes) % 2
                actual += weight * (-1 if parity else 1)
            expected = alpha if len(indexes) == 2 else 0.0
            assert isclose(actual, expected, abs_tol=1e-12)

        for first, second in pairs:
            totals = {(x, y): 0.0 for x, y in product(OUTCOMES, repeat=2)}
            for signs, weight in zip(states, weights):
                overloaded = signs[-2:] == (1, 1)
                totals[signs[first], signs[second]] += weight * (
                    4 * overloaded - (1 + alpha)
                )
            interaction = eta if (first, second) == (2 * useful, 2 * useful + 1) else 0
            for (x, y), actual in totals.items():
                assert isclose(
                    actual, joint_savings(alpha, interaction, x, y), abs_tol=1e-12
                )

def audit(pair_count, alpha, eta, trials=256):
    """Identify the useful pair from synthetic logs tagged with overload status."""
    sample_sizes = (16, 64, 256, 1024)
    correct = dict.fromkeys(sample_sizes, 0)
    random = Random(20261126)
    size = 2 * pair_count + 2
    envelope = 1 + alpha * size * (size - 1) / 2 + eta
    for _ in range(trials):
        scores = [0] * pair_count
        for sample_size in range(1, sample_sizes[-1] + 1):
            while True:
                mask = random.getrandbits(size)
                if random.random() * envelope < distribution_weight(
                    mask, size, alpha, eta
                ):
                    break
            if mask >> (size - 2) == 3:
                for pair in range(pair_count):
                    bits = (mask >> (2 * pair)) & 3
                    scores[pair] += 1 if bits in (0, 3) else -1
            if sample_size in correct:
                best = max(scores)
                winners = [pair for pair, score in enumerate(scores) if score == best]
                correct[sample_size] += random.choice(winners) == 0

    print(f"Audit with {pair_count} pairs and {trials} trials")
    for sample_size, successes in correct.items():
        print(f"{sample_size:>4} days  {successes:>3}/{trials} identified")
    confidence_size = ceil(128 * log(2 * pair_count / 0.05) / eta**2)
    print(f"Sufficient sample size from Hoeffding for 95% confidence: {confidence_size}")


def main():
    verify_by_enumeration()
    print("Exhaustive verification passed for marginals through order 3 and gain tables.")
    print("pairs   corr       known gain  minimax gain  fraction retained")
    eta = 0.6
    for pair_count in (2, 4, 8, 16, 32):
        alpha = eta / (16 * pair_count)
        known = optimal_gain(pair_count, alpha, [eta] + [0] * (pair_count - 1))
        minimax = optimal_gain(pair_count, alpha, [eta / pair_count] * pair_count)
        assert isclose(known, (eta - alpha**2) / 2)
        assert isclose(minimax, (eta / pair_count - alpha**2) / 2)
        print(
            f"{pair_count:>5}  {alpha:.6f}    {known:.6f}    "
            f"{minimax:.6f}    {minimax / known:>7.2%}"
        )
    pair_count, alpha = 8, 0.04
    size = 2 * pair_count + 2
    assert 1 - alpha * size / 2 - eta > 0
    known = optimal_gain(pair_count, alpha, [eta] + [0] * (pair_count - 1))
    minimax = optimal_gain(pair_count, alpha, [eta / pair_count] * pair_count)
    print(f"Fixed-correlation check: M={pair_count}, corr={alpha:.2f}")
    print(
        f"Known gain={known:.6f}, minimax gain={minimax:.6f}, "
        f"retained={minimax / known:.2%}"
    )
    audit(pair_count, alpha, eta)


if __name__ == "__main__":
    main()
