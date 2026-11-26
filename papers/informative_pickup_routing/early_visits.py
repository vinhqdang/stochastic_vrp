"""Optimize up to two early visits using synchronized, binary pickup logs."""

from dataclasses import dataclass
from math import isfinite, log, sqrt


@dataclass(frozen=True)
class Travel:
    """Deterministic time and cost of the early visits."""

    start: tuple[float, ...]
    between: tuple[tuple[float, ...], ...]
    deadline: float
    cost_per_minute: float

    def __post_init__(self):
        size = len(self.start)
        if len(self.between) != size or any(len(row) != size for row in self.between):
            raise ValueError("Travel times must cover the same customers")
        values = (*self.start, self.deadline, self.cost_per_minute)
        if any(v < 0 or not isfinite(v) for v in values):
            raise ValueError("Travel times, deadline, and cost must be finite and nonnegative")
        if any(v < 0 or not isfinite(v) for row in self.between for v in row):
            raise ValueError("Travel times must be finite and nonnegative")

    def next_stops(self, first):
        if first is None:
            return [i for i, minutes in enumerate(self.start) if minutes <= self.deadline]
        return [
            j for j, minutes in enumerate(self.between[first])
            if j != first and self.start[first] + minutes <= self.deadline
        ]


@dataclass(frozen=True)
class Branch:
    second: int | None
    dispatch: tuple[bool, bool]


@dataclass(frozen=True)
class Policy:
    first: int | None
    branches: tuple[Branch, Branch]

    def saving(self, signals, overloaded, travel, dispatch_cost, overload_cost):
        """Score a policy relative to waiting without any visits."""
        minutes = 0
        branch = self.branches[0]
        second_signal = 0
        if self.first is not None:
            branch = self.branches[int(signals[self.first])]
            minutes = travel.start[self.first]
            if branch.second is not None:
                second_signal = int(signals[branch.second])
                minutes += travel.between[self.first][branch.second]
        avoided = overload_cost * overloaded - dispatch_cost
        return branch.dispatch[second_signal] * avoided - travel.cost_per_minute * minutes


def immediate(dispatch):
    branch = Branch(None, (dispatch, dispatch))
    return Policy(None, (branch, branch))


def mean_saving(policy, cases, travel, dispatch_cost, overload_cost):
    return sum(
        weight * policy.saving(signals, overloaded, travel, dispatch_cost, overload_cost)
        for (signals, overloaded), weight in cases.items()
    ) / sum(cases.values())


def fit(cases, travel, dispatch_cost, overload_cost, *, adaptive=True):
    """Return the optimal policy for weighted logs or a known finite law.

    A log key is (tuple of 0/1 pickup signals, overload indicator).
    Its weight can be a count or a probability. In fixed-order mode the
    second customer is preselected, but either first outcome may stop early.
    """
    if not 0 <= dispatch_cost <= overload_cost or not isfinite(overload_cost):
        raise ValueError("Require finite costs with 0 <= dispatch_cost <= overload_cost")
    total_weight = sum(cases.values())
    if not cases or not isfinite(total_weight) or total_weight <= 0:
        raise ValueError("At least one observation with positive weight is required")
    rows = []
    for (signals, overloaded), weight in cases.items():
        if len(signals) != len(travel.start) or any(x not in (0, 1) for x in signals):
            raise ValueError("Each observation must contain one binary signal per customer")
        if overloaded not in (0, 1) or weight < 0 or not isfinite(weight):
            raise ValueError("Overload must be binary and weights finite and nonnegative")
        binary_signals = tuple(int(signal) for signal in signals)
        rows.append(
            (binary_signals, weight, weight * (overload_cost * overloaded - dispatch_cost))
        )

    initial = sum(reward for _, _, reward in rows)
    best_policy, best_value = immediate(initial > 0), max(0, initial)
    for first in travel.next_stops(None):
        mass = [0.0, 0.0]
        first_reward = [0.0, 0.0]
        for signals, weight, reward in rows:
            mass[signals[first]] += weight
            first_reward[signals[first]] += reward
        stop_values = [max(0, reward) for reward in first_reward]
        stop_branches = [Branch(None, (reward > 0, reward > 0)) for reward in first_reward]
        values, branches = stop_values.copy(), stop_branches.copy()

        for second in travel.next_stops(first):
            joint_reward = [[0.0, 0.0], [0.0, 0.0]]
            for signals, _, reward in rows:
                joint_reward[signals[first]][signals[second]] += reward
            follow_cost = travel.cost_per_minute * travel.between[first][second]
            candidate_values = [
                sum(max(0, reward) for reward in joint_reward[x]) - mass[x] * follow_cost
                for x in (0, 1)
            ]
            candidate_branches = [
                Branch(second, tuple(reward > 0 for reward in joint_reward[x]))
                for x in (0, 1)
            ]
            for x in (0, 1):
                if stop_values[x] >= candidate_values[x]:
                    candidate_values[x], candidate_branches[x] = stop_values[x], stop_branches[x]
            if adaptive:
                for x in (0, 1):
                    if candidate_values[x] > values[x]:
                        values[x], branches[x] = candidate_values[x], candidate_branches[x]
            elif sum(candidate_values) > sum(values):
                values, branches = candidate_values, candidate_branches

        value = sum(values) - total_weight * travel.cost_per_minute * travel.start[first]
        if value > best_value:
            best_policy, best_value = Policy(first, tuple(branches)), value
    return best_policy, best_value / total_weight


def certified_improvement(
    policy, validation_counts, travel, dispatch_cost, overload_cost, *, delta=0.05
):
    """One-sided holdout lower bound against the best immediate action.

    The feasible policy must be fixed independently of these IID validation days.
    """
    n = sum(validation_counts.values())
    if any(not isinstance(count, int) or count < 0 for count in validation_counts.values()):
        raise ValueError("Validation weights must be nonnegative integer counts")
    if n < 2 or not 0 < delta < 1:
        raise ValueError("Use at least two validation days and 0 < delta < 1")
    learned = mean_saving(policy, validation_counts, travel, dispatch_cost, overload_cost)
    best_immediate = max(
        mean_saving(immediate(dispatch), validation_counts, travel, dispatch_cost, overload_cost)
        for dispatch in (False, True)
    )
    width = overload_cost + travel.cost_per_minute * travel.deadline
    return learned - best_immediate - width * sqrt(log(2 / delta) / (2 * n))
