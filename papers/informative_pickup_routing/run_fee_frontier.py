"""Verify the continuous fee profile and evaluate fixed-fee holdout policies."""

import sys
from itertools import product
from math import fsum, isclose
from pathlib import Path
from random import Random

import numpy as np

from fee_frontier import fee_profile, with_fee
from fleet_batches import Signal
from route_menus import RouteMenus, RouteOption
from run_route_menus import build_menus, evaluate_policy


def verify_frontier():
    original = RouteOption(0, Signal(((6, 1),), 0))
    informative = RouteOption(1, Signal(((2, 0.5), (10, 0.5)), 0))
    model = RouteMenus([(original, informative)], 1)
    profile = fee_profile(model, 0, 12)
    profitable = [interval for segment in profile.segments if (interval := segment.above(0))]
    assert profitable == [(4.0, 6.0), (6.0, 8.0)], profitable
    print("Verified the exact profitable fee band (4, 8) in the analytic example.")

    rng = Random(13253)
    for _ in range(24):
        menus = [
            [
                RouteOption(
                    rng.randint(0, 16) / 4,
                    Signal(((rng.randint(-4, 5), 0.4), (rng.randint(5, 16), 0.6)), 0),
                )
                for _ in range(rng.randint(1, 3))
            ]
            for _ in range(3)
        ]
        model = RouteMenus(menus, rng.randrange(4))
        profile = fee_profile(model, 0, 18)
        assert profile.segments[0].start == 0 and profile.segments[-1].end == 18
        assert all(a.end == b.start for a, b in zip(profile.segments, profile.segments[1:]))
        for segment in profile.segments:
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                fee = segment.start + (segment.end - segment.start) * fraction
                priced = with_fee(model, fee)
                exact = min(priced.cost(choices) for choices in product(*model.options))
                actual = priced.cost(segment.choices)
                assert isclose(actual, exact, rel_tol=0, abs_tol=2e-8)
                gain = priced.cost(segment.prior_choices) - actual
                affine = segment.gain_start + fraction * (segment.gain_end - segment.gain_start)
                assert isclose(gain, affine, abs_tol=2e-8)
                if 0 < fraction < 1:
                    assert segment.prior_choices == model.mean_score_plan(fee)
    print("Compared the certified segments with enumeration on 24 small menu problems.")


def holdout_saving(model, profile, held_out, fee):
    chosen = next(segment.choices for segment in profile.segments if segment.end >= fee)
    prior = model.mean_score_plan(fee)
    proposed = evaluate_policy(chosen, held_out, model.standby, fee)
    baseline = evaluate_policy(prior, held_out, model.standby, fee)
    return baseline - proposed


def benchmark():
    engine = Path(__file__).resolve().parents[2] / "svrpspd_wdro"
    sys.path.insert(0, str(engine))
    sys.path.insert(0, str(engine / "scripts"))
    cases = {
        "Dethloff": [f"CON3-{i}" for i in range(8)],
        "City": [f"{city}-100-{i}" for city in ("HANOI", "HCMC") for i in (1, 2)],
    }
    fees = (0, 5, 10, 15, 20)
    for source, names in cases.items():
        for volume in (1.0, 1.1 if source == "Dethloff" else 1.8):
            rows = []
            best = None
            for index, name in enumerate(names):
                rng = np.random.default_rng(3915 + index)
                menus, evaluation = build_menus(name, source, engine, rng, volume, standby_fee=0)
                for standby in (1, 2):
                    model = RouteMenus(menus, standby)
                    profile = fee_profile(model, 0, 20)
                    above = [
                        range_ for segment in profile.segments if (range_ := segment.above(1.0))
                    ]
                    width = fsum(b - a for a, b in above)
                    gains = tuple(holdout_saving(model, profile, evaluation, fee) for fee in fees)
                    rows.append((width, profile.solves, gains))
                    if best is None or width > best[0]:
                        best = width, name, standby, above
            print(
                f"{source} x{volume}: {sum(width > 0 for width, _, _ in rows)}/{len(rows)} "
                f"cases with fitted saving >1; average fee span {np.mean([r[0] for r in rows]):.3f}/20; "
                f"mean oracle solves {np.mean([r[1] for r in rows]):.1f}"
            )
            paired_days = np.mean([row[2] for row in rows], axis=0)
            averages = paired_days.mean(axis=1)
            margins = 1.96 * paired_days.std(axis=1, ddof=1) / np.sqrt(paired_days.shape[1])
            print(
                "Held-out mean gains and 95% Monte Carlo margins at fees 0, 5, 10, 15, 20: "
                + " ".join(f"{mean:.6f} ± {margin:.6f}" for mean, margin in zip(averages, margins))
            )
            width, name, standby, intervals = best
            if width > 0:
                merged = []
                for a, b in intervals:
                    if merged and isclose(a, merged[-1][1], rel_tol=0, abs_tol=1e-8):
                        merged[-1] = (merged[-1][0], b)
                    else:
                        merged.append((a, b))
                ranges = " ".join(f"({a:.3f}, {b:.3f})" for a, b in merged)
                print(f"Largest fitted fee span: {name}, B={standby}, intervals {ranges}")


if __name__ == "__main__":
    verify_frontier()
    benchmark()
