"""Choose one batch of independent route signals for a shared standby pool."""

from dataclasses import dataclass
from itertools import product
from math import fsum, isclose, isfinite

from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Signal:
    """Possible posterior expected standby savings with their probabilities."""

    outcomes: tuple[tuple[float, float], ...]
    cost: float

    def __post_init__(self):
        if not isfinite(self.cost) or self.cost < 0:
            raise ValueError("Acquisition costs must be finite and nonnegative")
        if any(not isfinite(value) or not isfinite(p) or p <= 0 for value, p in self.outcomes):
            raise ValueError("Outcomes must be finite and probabilities positive")
        if not isclose(fsum(p for _, p in self.outcomes), 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Outcome probabilities must sum to one")

    @property
    def mean(self):
        return fsum(value * p for value, p in self.outcomes)


def expected_standby_savings(laws, standby):
    """Integrate the expected number selected over positive score thresholds."""
    if standby <= 0:
        return 0.0
    capacity = min(standby, len(laws))
    events = {}
    tails = [0.0] * len(laws)
    for index, law in enumerate(laws):
        for value, probability in law:
            if value > 0:
                events.setdefault(value, []).append((index, probability))
                tails[index] += probability
    previous = 0.0
    savings = 0.0
    for threshold in sorted(events):
        counts = [1.0] + [0.0] * capacity
        for p in tails:
            p = min(1.0, max(0.0, p))
            counts[capacity] += p * counts[capacity - 1]
            for k in range(capacity - 1, 0, -1):
                counts[k] = (1 - p) * counts[k] + p * counts[k - 1]
            counts[0] *= 1 - p
        savings += (threshold - previous) * fsum(k * p for k, p in enumerate(counts))
        for index, probability in events[threshold]:
            tails[index] -= probability
        previous = threshold
    return savings


@dataclass(frozen=True)
class Batch:
    routes: tuple[int, ...]
    net_saving: float
    upper_bound: float
    exchange_degree: int


class Fleet:
    def __init__(self, signals, standby):
        self.signals = tuple(signals)
        if not isinstance(standby, int) or standby < 0:
            raise ValueError("The standby count must be a nonnegative integer")
        self.standby = min(standby, len(self.signals))
        self.means = tuple(signal.mean for signal in self.signals)
        ranked = sorted((max(0, mean) for mean in self.means), reverse=True)
        self.baseline = fsum(ranked[:self.standby])

    def net_value(self, observed):
        observed = frozenset(observed)
        if any(index < 0 or index >= len(self.signals) for index in observed):
            raise ValueError("Every observation must refer to a route in the fleet")
        cost = fsum(self.signals[index].cost for index in observed)
        return self._gross_value(observed) - cost

    def _gross_value(self, observed):
        if not observed:
            return 0.0
        laws = [
            signal.outcomes if i in observed else ((self.means[i], 1.0),)
            for i, signal in enumerate(self.signals)
        ]
        return expected_standby_savings(laws, self.standby) - self.baseline

    def select_batch(self):
        """Select an observation batch and bound the best net batch value."""
        if self.standby == 0:
            return Batch((), 0.0, 0.0, 0)
        count = len(self.signals)
        dummy = Signal(((0.0, 1.0),), 0.0)
        padded = self.signals + (dummy,) * self.standby
        means = self.means + (0.0,) * self.standby
        ranks = sorted(range(len(padded)), key=lambda index: means[index], reverse=True)
        incumbent, challenger = ranks[:self.standby], ranks[self.standby:]

        edges = {}
        degree = [0] * len(padded)
        lowest = [min(value for value, _ in signal.outcomes) for signal in padded]
        highest = [max(value for value, _ in signal.outcomes) for signal in padded]
        for row, i in enumerate(incumbent):
            for col, j in enumerate(challenger):
                if highest[j] <= lowest[i]:
                    continue
                choices = []
                left_modes = (False, True) if i < count else (False,)
                right_modes = (False, True) if j < count else (False,)
                for observe_i, observe_j in product(left_modes, right_modes):
                    left = padded[i].outcomes if observe_i else ((means[i], 1.0),)
                    right = padded[j].outcomes if observe_j else ((means[j], 1.0),)
                    gross = fsum(
                        p * q * max(0, b - a) for (a, p), (b, q) in product(left, right)
                    )
                    selected = tuple(
                        index for index, use in ((i, observe_i), (j, observe_j)) if use
                    )
                    cost = fsum(padded[index].cost for index in selected)
                    choices.append((gross, cost, selected))
                edges[row, col] = choices
                degree[i] += 1
                degree[j] += 1
        max_degree = max(degree)
        if max_degree == 0:
            return Batch((), 0.0, 0.0, 0)

        def solve_matching(cost_fraction):
            weights = [[0.0] * len(challenger) for _ in incumbent]
            observations = {}
            for (row, col), choices in edges.items():
                gross, cost, selected = max(
                    choices, key=lambda option: option[0] - cost_fraction * option[1]
                )
                value = gross - cost_fraction * cost
                if value > 0:
                    weights[row][col] = value
                    observations[row, col] = selected
            rows, cols = linear_sum_assignment(weights, maximize=True)
            keys = list(zip(rows, cols))
            score = fsum(weights[row][col] for row, col in keys)
            selected = tuple(
                sorted(index for edge in keys for index in observations.get(edge, ()))
            )
            return score, selected

        _, selected = solve_matching(1)
        lower = self.net_value(selected)
        scaled, _ = solve_matching(1 / max_degree)
        all_gross = self._gross_value(range(count))
        upper = min(max_degree * scaled, all_gross)
        return Batch(selected, lower, upper, max_degree)
