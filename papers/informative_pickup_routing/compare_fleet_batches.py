"""Compare batch and sequential observation methods under matched score laws."""

import json
import sys
from functools import cache
from itertools import combinations
from math import comb, fsum, isclose
from pathlib import Path
from random import Random

from fleet_batches import Fleet, Signal
from run_fleet_batches import enumerate_savings, greedy_singletons


def exact_sequential(fleet):
    """Allow each new observation to depend on earlier outcomes."""
    signals = fleet.signals

    @cache
    def value(state):
        scores = [
            fleet.means[i] if outcome < 0 else signals[i].outcomes[outcome][0]
            for i, outcome in enumerate(state)
        ]
        best = fsum(sorted((max(0, score) for score in scores), reverse=True)[:fleet.standby])
        for i, outcome in enumerate(state):
            if outcome < 0:
                probe = signals[i]
                continuation = fsum(
                    probability * value(state[:i] + (j,) + state[i+1:])
                    for j, (_, probability) in enumerate(probe.outcomes)
                ) - probe.cost
                best = max(best, continuation)
        return best

    return value((-1,) * len(signals)) - fleet.baseline


def reservation(signal):
    """Solve E[(score - reservation)_+] = inspection cost."""
    outcomes = sorted(signal.outcomes, reverse=True)
    mass = weighted_score = 0.0
    for i, (score, p) in enumerate(outcomes):
        mass += p
        weighted_score += p * score
        level = (weighted_score - signal.cost) / mass
        if i + 1 == len(outcomes) or level >= outcomes[i+1][0]:
            return level


def committing(fleet):
    """Best Beyhaghi–Kleinberg committing policy for one standby vehicle."""
    if fleet.standby != 1:
        raise ValueError("The published committing policy selects one option")
    signals = fleet.signals
    levels = tuple(reservation(signal) for signal in signals)
    order = sorted(range(len(signals)), key=levels.__getitem__, reverse=True)

    def utility(backup):
        inspect = [i for i in order if i != backup]
        outside = max(0.0, fleet.means[backup]) if backup is not None else 0.0

        @cache
        def value(index, best):
            if index == len(inspect) or best >= levels[inspect[index]]:
                return best
            signal = signals[inspect[index]]
            return fsum(
                p * value(index + 1, max(best, score)) for score, p in signal.outcomes
            ) - signal.cost

        return value(0, outside)

    return max(utility(i) for i in (None, *range(len(signals)))) - fleet.baseline


def exact_batch(value, size):
    return max(
        value(frozenset(batch))
        for k in range(size + 1)
        for batch in combinations(range(size), k)
    )


def local_batch(value, size):
    """Two-start local search with add, drop, and up to two flips."""
    flips = [
        frozenset(items) for k in (1, 2) for items in combinations(range(size), k)
    ]
    best = 0.0
    for start in (frozenset(), frozenset(range(size))):
        batch = start
        while True:
            current = value(batch)
            neighbor = max((batch ^ flip for flip in flips), key=value)
            if value(neighbor) <= current + 1e-10:
                best = max(best, current)
                break
            batch = neighbor
    return best


def exact_complementary_batch(standby, cost):
    """Optimize over the observation counts in the symmetric two-group model."""
    tails = []
    for count in range(standby + 1):
        probabilities = [0.0] * count
        mass = 0.0
        for k in range(count, 0, -1):
            mass += comb(count, k) / 2**count
            probabilities[k-1] = mass
        tails.append(probabilities)
    return max(
        fsum(p * q for p, q in zip(tails[a], tails[b])) - cost * (a + b)
        for a in range(standby + 1) for b in range(a + 1)
    )


def synthetic_fleets():
    random = Random(9884)
    for family in ("Heterogeneous", "Near cutoff"):
        for instance in range(24):
            signals = []
            for route in range(8):
                if family == "Heterogeneous":
                    low, high = sorted(random.sample(range(-4, 16), 2))
                elif route < 4:
                    low, high = random.uniform(3, 6), random.uniform(7, 10)
                else:
                    low, high = random.uniform(-2, 2), random.uniform(4, 8)
                p = random.uniform(0.25, 0.75)
                cost = random.choice((0.05, 0.25, 0.75, 1.5))
                signals.append(Signal(((low, p), (high, 1-p)), cost))
            standby = 1 if instance % 2 == 0 else 4
            yield family, Fleet(signals, standby)


def route_fleets():
    """Build empirical score laws on saved routes with simulated demands."""
    import numpy as np

    engine = Path(__file__).resolve().parents[2] / "svrpspd_wdro"
    sys.path.insert(0, str(engine))
    sys.path.insert(0, str(engine / "scripts"))
    from core.costs import LastMileCosts, route_cost_schedules
    from dethloff_runner import parse_dethloff, sample_demands

    random = np.random.default_rng(16252)
    costs = LastMileCosts()
    instances = {
        "Dethloff": [f"CON3-{i}" for i in range(8)],
        "City": [f"{city}-100-{i}" for city in ("HANOI", "HCMC") for i in (1, 2)],
    }
    for source, name in ((source, name) for source, names in instances.items() for name in names):
        plan_file = engine / "results" / "plans" / f"{name}.json"
        plan = json.loads(plan_file.read_text())
        distance, _, capacity, _, scale = parse_dethloff(
            engine / "data" / source / f"{name}.vrpspd"
        )
        laws = []
        for route in plan["res"]["Det"]["plan"]:
            if len(route) < 2:
                continue
            order = [0, *route]
            n = len(order)
            delivery, pickup = [
                sample_demands(np.asarray(plan[key])[order], n, 8192, 0.30, "gamma", random, rho=0.6)
                for key in ("dbar", "pbar")
            ]
            loads = delivery[:, 1:].sum(axis=1, keepdims=True) + np.cumsum(
                pickup[:, 1:] - delivery[:, 1:], axis=1
            )
            loads = loads[(delivery[:, 1:].sum(axis=1) <= capacity) & (loads[:, 0] <= capacity)]
            if not len(loads):
                raise ValueError(f"No eligible post-first-stop samples for {name}, route {route}")
            handoff, emergency = route_cost_schedules(route, distance, scale, costs)
            failures = loads[:, 1:] > capacity
            losses = np.where(failures.any(axis=1), emergency[1:][failures.argmax(axis=1)], 0.0)
            savings = losses - handoff[0]
            high = loads[:, 0] > np.median(loads[:, 0])
            laws.append(tuple(
                (float(savings[mask].mean()), float(mask.mean())) for mask in (~high, high) if mask.any()
            ))
        for cost in (0.05, 0.25, 1.0):
            signals = [Signal(law, cost) for law in laws]
            for standby in ((1, 2) if source == "Dethloff" else (1, 2, 4)):
                yield f"{source} with simulated demands", Fleet(signals, standby)


def evaluate(fleet):
    size = len(fleet.signals)
    value = cache(fleet.net_value)
    optimal = exact_batch(value, size)
    matching = fleet.select_batch()
    local = local_batch(value, size)
    sequential = exact_sequential(fleet)
    published = committing(fleet) if fleet.standby == 1 else None
    assert matching.net_saving <= optimal + 1e-9
    assert optimal <= matching.upper_bound + 1e-9
    assert local <= optimal + 1e-9 and optimal <= sequential + 1e-9
    if published is not None:
        assert published <= sequential + 1e-9
    return (
        greedy_singletons(fleet), matching.net_saving, local, optimal, published, sequential
    )


def summarize(rows):
    columns = zip(*rows)
    means = []
    for values in columns:
        measured = [x for x in values if x is not None]
        means.append(f"{fsum(measured) / len(measured):.6f}" if measured else "n/a")
    print("  ".join(means))


def check_committing():
    signals = (
        Signal(((0, 0.3), (1, 0.2), (8, 0.5)), 0.0),
        Signal(((2, 0.5), (9, 0.5)), 0.8),
        Signal(((0, 0.5), (7, 0.5)), 1.9),
    )
    fleet = Fleet(signals, 1)
    capped = []
    for signal in signals:
        level = reservation(signal)
        assert isclose(
            fsum(p * max(0, score - level) for score, p in signal.outcomes),
            signal.cost, abs_tol=1e-11,
        )
        capped.append(tuple((min(score, level), p) for score, p in signal.outcomes))
    utilities = [enumerate_savings(capped, 1)]
    for backup in range(len(signals)):
        laws = capped.copy()
        laws[backup] = ((fleet.means[backup], 1.0),)
        utilities.append(enumerate_savings(laws, 1))
    assert isclose(committing(fleet) + fleet.baseline, max(utilities), abs_tol=1e-11)


def main():
    check_committing()
    groups = {}
    for family, fleet in (*synthetic_fleets(), *route_fleets()):
        groups.setdefault((family, fleet.standby), []).append(evaluate(fleet))
    print("Mean incremental net savings")
    print("Greedy  Matching  Two-flip  Batch optimum  BK committing  Sequential optimum")
    for (family, standby), rows in sorted(groups.items()):
        print(f"{family}, B={standby}, cases={len(rows)}")
        summarize(rows)
        positive = sum(row[3] > 1e-9 for row in rows)
        exact_local = sum(isclose(row[2], row[3], abs_tol=1e-9) for row in rows)
        wins = sum(matching > local + 1e-9 for _, matching, local, *_ in rows)
        losses = sum(matching + 1e-9 < local for _, matching, local, *_ in rows)
        print(f"Positive batch optimum: {positive}/{len(rows)}. "
              f"Two-flip attained the exact optimum: {exact_local}/{len(rows)}.")
        print(f"Matching vs two-flip: {wins} wins, {losses} losses, {len(rows)-wins-losses} ties.")

    print("Complementary construction: matching and exact batch savings")
    for standby in (4, 16, 64, 128):
        low = Signal(((4, 0.5), (8, 0.5)), 0.0625)
        high = Signal(((0, 0.5), (5, 0.5)), 0.0625)
        fleet = Fleet((low,) * standby + (high,) * standby, standby)
        matching = fleet.select_batch().net_saving
        optimal = exact_complementary_batch(standby, low.cost)
        if standby == 4:
            assert isclose(optimal, exact_batch(fleet.net_value, 2 * standby), abs_tol=1e-9)
        assert matching <= optimal + 1e-9
        print(f"routes={2*standby}: matching={matching:.6f}, exact batch={optimal:.6f}")


if __name__ == "__main__":
    main()
