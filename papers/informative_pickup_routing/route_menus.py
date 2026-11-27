"""Choose route-specific early visits with an optimality certificate."""

from bisect import bisect_left
from dataclasses import dataclass
from functools import cache
from heapq import heappop, heappush
from itertools import count
from math import fsum, isfinite

from fleet_batches import Signal, expected_standby_savings, improve_choices


@dataclass(frozen=True)
class RouteOption:
    """Expected primary cost and posterior savings from assigning standby."""

    primary_cost: float
    signal: Signal

    def __post_init__(self):
        if not isfinite(self.primary_cost):
            raise ValueError("Primary cost must be finite")

    @property
    def cost(self):
        return self.primary_cost + self.signal.cost


@dataclass(frozen=True)
class MenuSolution:
    choices: tuple[int, ...]
    cost: float
    lower_bound: float
    expanded_nodes: int


def standby_envelope(options):
    """Couple option-specific scores by their quantiles before choosing an option."""
    cheapest = min(option.cost for option in options)
    cdfs = []
    endpoints = {1.0}
    for option in options:
        shifted_scores = []
        cdf = []
        mass = 0.0
        for score, probability in sorted(option.signal.outcomes):
            mass += probability
            cdf.append(min(mass, 1.0))
            shifted_scores.append(score + cheapest - option.cost)
        cdf[-1] = 1.0
        endpoints.update(cdf)
        cdfs.append((cdf, shifted_scores))

    law = {}
    previous = 0.0
    for endpoint in sorted(endpoints):
        if endpoint > previous:
            score = max(values[bisect_left(cdf, endpoint)] for cdf, values in cdfs)
            law[score] = law.get(score, 0.0) + endpoint - previous
        previous = endpoint
    return cheapest, tuple(law.items())


class RouteMenus:
    def __init__(self, menus, standby):
        self.menus = tuple(tuple(menu) for menu in menus)
        if any(not menu for menu in self.menus):
            raise ValueError("Every route needs at least one feasible option")
        if not isinstance(standby, int) or standby < 0:
            raise ValueError("Standby count must be a nonnegative integer")
        self.standby = standby
        self.options = tuple(tuple(range(len(menu))) for menu in self.menus)

    def cost(self, choices):
        """Evaluate a complete route choice with optimal signal-based dispatch."""
        if len(choices) != len(self.options) or any(
            j not in options for j, options in zip(choices, self.options)
        ):
            raise ValueError("Select exactly one feasible option per route")
        options = [self.menus[i][j] for i, j in enumerate(choices)]
        gross = expected_standby_savings(
            [option.signal.outcomes for option in options], self.standby
        )
        return fsum(option.cost for option in options) - gross

    def mean_score_options(self):
        """Best primary and standby options and their prior standby advantages."""
        without = tuple(
            min(indices, key=lambda j: self.menus[i][j].cost)
            for i, indices in enumerate(self.options)
        )
        with_standby = tuple(
            min(indices, key=lambda j: self.menus[i][j].cost - self.menus[i][j].signal.mean)
            for i, indices in enumerate(self.options)
        )
        benefit = [
            self.menus[i][without[i]].cost - self.menus[i][j].cost + self.menus[i][j].signal.mean
            for i, j in enumerate(with_standby)
        ]
        return without, with_standby, benefit

    def mean_score_plan(self, standby_fee=0.0):
        """Exact route choice if standby allocation must precede all signals."""
        without, with_standby, benefit = self.mean_score_options()
        choices = list(without)
        ranked = sorted(range(len(self.menus)), key=benefit.__getitem__, reverse=True)
        for i in ranked[:self.standby]:
            if benefit[i] > standby_fee:
                choices[i] = with_standby[i]
        return tuple(choices)

    def solve(self, node_limit=None):
        """Return a feasible choice and a lower bound, with optional early stopping."""
        if node_limit is not None and (not isinstance(node_limit, int) or node_limit < 0):
            raise ValueError("Node limit must be a nonnegative integer or None")

        @cache
        def value(choices):
            return -self.cost(choices)

        @cache
        def route_relaxation(route, indices):
            return standby_envelope([self.menus[route][j] for j in indices])

        def bound(node):
            terms = [route_relaxation(i, indices) for i, indices in enumerate(node)]
            return fsum(cost for cost, _ in terms) - expected_standby_savings(
                [law for _, law in terms], self.standby
            )

        starts = {(0,) * len(self.menus), self.mean_score_plan()}
        best = max(
            (improve_choices(start, self.options, value) for start in sorted(starts)),
            key=value,
        )
        best_cost = -value(best)
        order = count()
        queue = [(bound(self.options), next(order), self.options)]
        expanded = 0
        while queue and queue[0][0] < best_cost - 1e-10:
            if node_limit is not None and expanded >= node_limit:
                break
            _, _, node = heappop(queue)
            route = max(range(len(node)), key=lambda i: len(node[i]))
            if len(node[route]) == 1:
                choice = tuple(indices[0] for indices in node)
                cost = -value(choice)
                if cost < best_cost:
                    best, best_cost = choice, cost
                continue
            expanded += 1
            for option in node[route]:
                child = node[:route] + ((option,),) + node[route+1:]
                if all(len(indices) == 1 for indices in child):
                    choice = tuple(indices[0] for indices in child)
                    cost = -value(choice)
                    if cost < best_cost:
                        best, best_cost = choice, cost
                    continue
                child_bound = bound(child)
                if child_bound < best_cost:
                    heappush(queue, (child_bound, next(order), child))
        lower = min(best_cost, queue[0][0]) if queue else best_cost
        return MenuSolution(best, best_cost, lower, expanded)
