"""Validate the menu solver and evaluate feasible early visits on saved routes."""

import json
import sys
from dataclasses import dataclass
from itertools import product
from math import isclose
from pathlib import Path
from random import Random

import numpy as np

from fleet_batches import Signal
from route_menus import RouteMenus, RouteOption


def verify_solver():
    random = Random(734)
    larger_gaps = 0
    for n in (3, 4, 5):
        for _ in range(10):
            menus = [
                [
                    RouteOption(
                        random.randint(0, 30),
                        Signal(((random.randint(-7, 10), 0.35), (random.randint(10, 45), 0.65)), 0),
                    )
                    for _ in range(random.randint(1, 3))
                ]
                for _ in range(n)
            ]
            model = RouteMenus(menus, random.randrange(n + 1))
            optimum = min(model.cost(choice) for choice in product(*model.options))
            exact = model.solve()
            assert isclose(exact.cost, optimum, abs_tol=1e-9)
            assert isclose(exact.lower_bound, optimum, abs_tol=1e-9)
            for limit in (0, 1):
                early = model.solve(node_limit=limit)
                assert early.lower_bound <= optimum + 1e-9 <= early.cost + 2e-9
                assert isclose(model.cost(early.choices), early.cost, abs_tol=1e-9)
            larger_gaps += model.solve(node_limit=0).cost - optimum > 1e-9
    # The two-group construction makes every one- or two-signal move unprofitable.
    # The eight signals together are profitable at this acquisition fee.
    menus = []
    for low, high in ((4, 8), (0, 5)):
        unobserved = RouteOption(0, Signal((((low + high) / 2, 1),), 0))
        observed = RouteOption(0, Signal(((low, 0.5), (high, 0.5)), 0.15))
        menus.extend([(unobserved, observed)] * 4)
    complementary = RouteMenus(menus, 4)
    root = complementary.solve(node_limit=0)
    exact = complementary.solve()
    enumerated = min(complementary.cost(choice) for choice in product(*complementary.options))
    assert root.lower_bound < exact.cost < root.cost - 1e-9
    assert exact.expanded_nodes > 0
    assert isclose(exact.cost, enumerated, abs_tol=1e-9)
    print(
        "Vetted 30 random menu problems by exhaustive enumeration, including "
        f"early-stopping certificates; {larger_gaps} needed improvement after local search."
    )
    print(
        f"Complementary fleet: branch-and-bound improved two-flip by "
        f"{root.cost - exact.cost:.6f} in {exact.expanded_nodes} expanded nodes."
    )


def travel_cost(path, distances, per_km):
    return per_km * sum(distances[i, j] for i, j in zip(path, path[1:]))


def residual_cost(start, suffix, loads, pickups, distances, capacity, costs):
    """Serve a fixed suffix with emergency depot unloading as needed."""
    result = np.full(len(loads), travel_cost([start, *suffix, 0], distances, costs.c_km))
    for index, stop in enumerate(suffix):
        overflow = loads + pickups[stop] > capacity
        if overflow.any():
            detour = costs.c_km * (distances[stop, 0] + distances[0, stop])
            fee = costs.F_emg + costs.p_breach + costs.p_late * (len(suffix) - index)
            result[overflow] += detour + fee
        loads = np.where(overflow, 0.0, loads) + pickups[stop]
    return result


@dataclass(frozen=True)
class HeldOut:
    primary: np.ndarray
    standby_saving: np.ndarray
    predicted_saving: np.ndarray


def evaluate_policy(choices, held_out, standby, standby_fee=0.0):
    """Use training posterior scores to dispatch, then pay realized costs."""
    routes = [options[choice] for options, choice in zip(held_out, choices)]
    predicted = np.vstack([run.predicted_saving for run in routes]) - standby_fee
    actual = np.vstack([run.standby_saving for run in routes]) - standby_fee
    initial = np.sum([run.primary for run in routes], axis=0)
    selected = np.argsort(-predicted, axis=0, kind="stable")[:standby]
    score = np.take_along_axis(predicted, selected, axis=0)
    saved = np.take_along_axis(actual, selected, axis=0)
    return initial - np.where(score > 0, saved, 0.0).sum(axis=0)


def build_menus(name, source, engine, random, demand_multiplier, standby_fee, samples=4096):
    from core.costs import LastMileCosts
    from dethloff_runner import parse_dethloff, sample_demands

    plan_file = engine / "results" / "plans" / f"{name}.json"
    plan = json.loads(plan_file.read_text())
    distance, _, capacity, _, scale = parse_dethloff(engine / "data" / source / f"{name}.vrpspd")
    distances = distance / scale
    routes = plan["res"]["Det"]["plan"]
    cutoff = max(distances[0, route[0]] / 25 * 60 + 2 for route in routes)
    costs = LastMileCosts(F_standby=standby_fee)
    menus, held_out = [], []
    for route in routes:
        eligible = [
            first for first in route[1:] if distances[0, first] / 25 * 60 + 2 <= cutoff
        ]
        eligible.sort(key=lambda first: travel_cost(
            [0, first, *(stop for stop in route if stop != first), 0], distances, costs.c_km
        ))
        candidates = [route[0], *eligible[:4]]
        means = np.asarray(plan["pbar"])[[0, *route]]
        rows = demand_multiplier * sample_demands(
            means, len(means), 2 * samples, 0.30, "gamma", random, rho=0.6
        )
        pickups = {i: np.minimum(rows[:, offset], capacity) for offset, i in enumerate(route, 1)}
        menu, validation = [], []
        for first in candidates:
            suffix = [i for i in route if i != first]
            prefix = travel_cost([0, first], distances, costs.c_km)
            primary = prefix + residual_cost(
                first, suffix, pickups[first], pickups, distances, capacity, costs
            )
            alternate = np.zeros_like(primary)
            if suffix:
                replacement = residual_cost(
                    0, suffix, np.zeros_like(primary), pickups, distances, capacity, costs
                )
                alternate = primary - (
                    prefix + travel_cost([first, 0], distances, costs.c_km)
                    + costs.handoff_cost(0) + replacement
                )
            cutoff_load = np.median(pickups[first][:samples])
            high = pickups[first] > cutoff_load
            training_high = high[:samples]
            scores, probabilities = {}, {}
            for outcome in (False, True):
                mask = training_high == outcome
                if mask.any():
                    scores[outcome] = float(alternate[:samples][mask].mean())
                    probabilities[outcome] = float(mask.mean())
            menu.append(RouteOption(
                float(primary[:samples].mean()),
                Signal(tuple((score, probabilities[outcome]) for outcome, score in scores.items()), 0),
            ))
            fallback = float(alternate[:samples].mean())
            predicted = np.where(high[samples:], scores.get(True, fallback), scores.get(False, fallback))
            validation.append(HeldOut(primary[samples:], alternate[samples:], predicted))
        menus.append(menu)
        held_out.append(validation)
    return menus, held_out


def routing_benchmark():
    engine = Path(__file__).resolve().parents[2] / "svrpspd_wdro"
    sys.path.insert(0, str(engine))
    sys.path.insert(0, str(engine / "scripts"))
    sources = {
        "Dethloff": [f"CON3-{i}" for i in range(8)],
        "City": [f"{city}-100-{i}" for city in ("HANOI", "HCMC") for i in (1, 2)],
    }
    print("Plan  Load  Fee  B  Routes  Options  Changed  Train vs fixed  Train vs mean  Test vs fixed  Test vs mean  Nodes  Gap")
    summary = {}
    for source, names in sources.items():
        multipliers = (1.0, 1.1 if source == "Dethloff" else 1.8)
        for (index, name), multiplier, fee in product(enumerate(names), multipliers, (20, 0)):
            random = np.random.default_rng(3915 + index)
            options, held_out = build_menus(name, source, engine, random, multiplier, fee)
            for standby in (1, 2):
                model = RouteMenus(options, standby)
                fixed = (0,) * len(options)
                mean = model.mean_score_plan()
                chosen = model.solve()
                assert chosen.cost - chosen.lower_bound <= 1e-8
                changed = sum(j != 0 for j in chosen.choices)
                test = evaluate_policy(chosen.choices, held_out, standby)
                test_fixed = evaluate_policy(fixed, held_out, standby)
                test_mean = evaluate_policy(mean, held_out, standby)
                gains = ((test_fixed - test).mean(), (test_mean - test).mean())
                summary.setdefault((source, multiplier, fee), []).append((changed, *gains))
                print(
                    f"{name} {multiplier:.1f} {fee} {standby} {len(options)} {sum(map(len, options))} "
                    f"{changed} {model.cost(fixed) - chosen.cost:.6f} "
                    f"{model.cost(mean) - chosen.cost:.6f} {gains[0]:.6f} {gains[1]:.6f} "
                    f"{chosen.expanded_nodes} {chosen.cost - chosen.lower_bound:.3g}"
                )
    print("Held-out summary by group")
    for (source, multiplier, fee), rows in summary.items():
        counts, fixed, mean = zip(*rows)
        print(
            f"{source} x{multiplier}, fee={fee}: "
            f"{sum(n > 0 for n in counts)}/{len(rows)} plans changed; "
            f"mean saving vs fixed={np.mean(fixed):.6f}, vs mean={np.mean(mean):.6f}; "
            f"losses vs fixed={sum(value < -1e-9 for value in fixed)}"
        )


if __name__ == "__main__":
    verify_solver()
    routing_benchmark()
