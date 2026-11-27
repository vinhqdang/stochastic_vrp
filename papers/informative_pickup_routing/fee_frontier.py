"""Bound the cost of early-visit plans over a continuous standby fee range."""

from dataclasses import dataclass, replace
from functools import cache
from math import isfinite

from route_menus import RouteMenus


TOLERANCE = 1e-8


def with_fee(model, fee):
    """Charge the same incremental fee for every assignment of a standby."""
    return RouteMenus(
        [
            [
                replace(
                    option,
                    signal=replace(
                        option.signal,
                        outcomes=tuple((score - fee, p) for score, p in option.signal.outcomes),
                    ),
                )
                for option in menu
            ]
            for menu in model.menus
        ],
        model.standby,
    )


@dataclass(frozen=True)
class FeeSegment:
    start: float
    end: float
    choices: tuple[int, ...]
    prior_choices: tuple[int, ...]
    gain_start: float
    gain_end: float

    def above(self, gain):
        """Part of the open segment whose fitted savings strictly exceed gain."""
        low = self.gain_start > gain
        high = self.gain_end > gain
        if low and high:
            return self.start, self.end
        if not low and not high:
            return None
        fraction = (gain - self.gain_start) / (self.gain_end - self.gain_start)
        crossing = self.start + (self.end - self.start) * fraction
        return (self.start, crossing) if low else (crossing, self.end)


@dataclass(frozen=True)
class FeeProfile:
    segments: tuple[FeeSegment, ...]
    solves: int


def fee_profile(model, low, high):
    """Return optimal-visit and prior-mean segments over the requested fee range.

    Results are certified to numerical tolerance for the finite fitted laws.
    Prior choices are defined on segment interiors. At fee knots, call the
    prior planner directly if its choice at the tie is needed.
    """
    if not (0 <= low < high and isfinite(high)):
        raise ValueError("Require a finite fee interval with 0 <= low < high")

    @cache
    def priced(fee):
        return with_fee(model, fee)

    @cache
    def optimum(fee):
        return priced(fee).solve()

    @cache
    def cost(fee, choices):
        return priced(fee).cost(choices)

    def refine(start, end):
        left = optimum(start)
        right = optimum(end)
        left_at_end = cost(end, left.choices)
        right_at_start = cost(start, right.choices)
        if left_at_end <= right.lower_bound + TOLERANCE:
            yield start, end, left.choices
            return
        if right_at_start <= left.lower_bound + TOLERANCE:
            yield start, end, right.choices
            return

        advantage_start = right_at_start - left.cost
        advantage_end = left_at_end - right.cost
        fraction = advantage_start / (advantage_start + advantage_end)
        split = start + (end - start) * fraction
        if not start < split < end:
            split = (start + end) / 2
        middle = optimum(split)
        left_at_split = cost(split, left.choices)
        right_at_split = cost(split, right.choices)
        if max(left_at_split, right_at_split) <= middle.lower_bound + TOLERANCE:
            yield start, split, left.choices
            yield split, end, right.choices
            return
        yield from refine(start, split)
        yield from refine(split, end)

    _, _, prior_switches = model.mean_score_options()
    knots = {
        score for menu in model.menus for option in menu
        for score, _ in option.signal.outcomes if low < score < high
    }
    knots.update(fee for fee in prior_switches if low < fee < high)
    knots = [low, *sorted(knots), high]
    segments = []
    for a, b in zip(knots, knots[1:]):
        for start, end, choices in refine(a, b):
            prior = model.mean_score_plan((start + end) / 2)
            gains = tuple(cost(fee, prior) - cost(fee, choices) for fee in (start, end))
            segments.append(FeeSegment(start, end, choices, prior, *gains))
    return FeeProfile(tuple(segments), optimum.cache_info().misses)
