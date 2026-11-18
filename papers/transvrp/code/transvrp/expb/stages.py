"""EXPERIMENT B stage definitions as plain data (EXPERIMENT_B.md 10 M21).

``expa.sweep`` turns each entry of ``b_stage_defs()`` into a ``StageSpec``. Arm
configurations are tuples of (field, value) pairs applied to the arm's configuration
class (registry.ARM_SPECS); ``()`` is the class default. A stage's ``arm_cfg`` is a tuple
of (arm, pairs) so that one stage can hold several arms (smoke).

The grids are those of sections 3 and 6.1. ``CHOSEN_ARM_CFG`` is written from the logged
tuning before the main fit; an arm missing from it has no chosen configuration and its
main stage refuses to fit.
"""
from __future__ import annotations

import itertools

from ..expa.train import StageConfig
from .registry import B_ARMS

FULL_N = 100_000
Pairs = tuple[tuple[str, object], ...]

# 3.3 RLSM: K in {20, 100, 400} x ridge in {1e-6, 1e-3} (lambda = ridge * n_fit)
# 3.4 DOS: lr in {1e-3, 3e-4}
# 3.5 DQN: lr in {1e-4, 3e-4} x target period in {8,000, 32,000} transitions
# 3.6 PPO: lr in {1e-4, 3e-4} x gamma in {0.5, 1.0}
# THR and ALP-Z: no grid (one configuration each, the default)
TUNE_GRID: dict[str, tuple[Pairs, ...]] = {
    "thr": ((),),
    "alpz": ((),),
    "rlsm": tuple((("K", K), ("ridge", r))
                  for K, r in itertools.product((20, 100, 400), (1e-6, 1e-3))),
    "dos": tuple((("lr", lr),) for lr in (1e-3, 3e-4)),
    "dqn": tuple((("lr", lr), ("target_every_tr", T))
                 for lr, T in itertools.product((1e-4, 3e-4), (8_000, 32_000))),
    "ppo": tuple((("lr", lr), ("gamma", g))
                 for lr, g in itertools.product((1e-4, 3e-4), (0.5, 1.0))),
}

# Chosen configurations (6.1). THR and ALP-Z have no grid and are chosen now.
CHOSEN_ARM_CFG: dict[str, Pairs] = {"thr": (), "alpz": ()}

# Tiny configurations for b_smoke (CPU, minutes); not a tuning result.
SMOKE_CFG: dict[str, Pairs] = {
    "thr": (),
    "alpz": (),
    "rlsm": (("K", 4),),
    "dos": (("s_min", 2), ("s_max", 4), ("n_ckpt", 1)),
    "dqn": (("passes", 1), ("buffer", 2_000), ("learn_start", 100), ("batch", 32),
            ("target_every_tr", 500), ("ckpt_per_pass", 2), ("n_env", 8), ("device", "cpu")),
    "ppo": (("passes", 1), ("epochs", 1), ("ckpt_per_pass", 2)),
}

LADDER_B = ("none", "baton_eq", "oracle")


def pilot_cfg(arm: str) -> Pairs:
    """Pilot cells use the first grid point, with one pass for DQN and PPO (9.3 times one
    full pass; the pilot changes no grid)."""
    first = TUNE_GRID[arm][0]
    return first + (("passes", 1),) if arm in ("dqn", "ppo") else first


def _arm_cfg(arms: tuple[str, ...], pairs_of) -> tuple:
    return tuple((a, pairs_of(a)) for a in arms)


def _main_arm_cfg(arm: str) -> tuple:
    return ((arm, CHOSEN_ARM_CFG[arm]),) if arm in CHOSEN_ARM_CFG else ()


def _smoke_def() -> dict:
    return dict(reps=(0,), ladder_N=(2_000,), ladder_labels=LADDER_B,
                neural=(("full", 2_000),) + tuple((a, 2_000) for a in B_ARMS),
                routes="smoke", cfg=StageConfig(s_max=10), test_allowed=False,
                arm_cfg=_arm_cfg(B_ARMS, lambda a: SMOKE_CFG[a]))


def _pilot_def() -> dict:
    return dict(reps=(0, 1), ladder_N=(10_000,), ladder_labels=LADDER_B,
                neural=(("full", 10_000),) + tuple((a, 10_000) for a in B_ARMS),
                timing=tuple((a, FULL_N) for a in B_ARMS), routes="eligible",
                test_allowed=False, arm_cfg=_arm_cfg(B_ARMS, pilot_cfg))


def _tune_defs() -> dict[str, dict]:
    """6.1: dev only, the 12 subset routes, rep 0; baton_eq on the same rows for scoring."""
    out = {}
    for arm in B_ARMS:
        for i, pairs in enumerate(TUNE_GRID[arm]):
            out[f"b_tune_{arm}_c{i}"] = dict(
                reps=(0,), ladder_N=(FULL_N,), ladder_labels=("baton_eq",),
                neural=((arm, FULL_N),), routes="subset", test_allowed=False,
                arm_cfg=((arm, pairs),),
                pass_curve=3 if arm in ("dqn", "ppo") else 0)
    return out


def _main_defs() -> dict[str, dict]:
    out = {
        "b_ladder": dict(reps=(0, 1, 2), ladder_N=(FULL_N,), ladder_labels=LADDER_B,
                         routes="eligible"),
        "b_full": dict(reps=(0, 1, 2), neural=(("full", FULL_N),), routes="eligible"),
    }
    for arm in B_ARMS:
        out[f"b_{arm}"] = dict(reps=(0, 1, 2), neural=((arm, FULL_N),), routes="eligible",
                               arm_cfg=_main_arm_cfg(arm))
    return out


def _independence_defs() -> dict[str, dict]:
    """3.8: the 12 subset routes, rep 0, every stop column permuted."""
    ind = dict(reps=(0,), routes="subset", independent=True)
    out = {
        "b_ladder_ind": dict(ind, ladder_N=(FULL_N,), ladder_labels=LADDER_B),
        "b_full_ind": dict(ind, neural=(("full", FULL_N),)),
    }
    for arm in ("thr", "alpz"):
        out[f"b_{arm}_ind"] = dict(ind, neural=((arm, FULL_N),), arm_cfg=_main_arm_cfg(arm))
    return out


def _reuse_defs() -> dict[str, dict]:
    """Cut (i), 9.2: A's own main-stage files, re-evaluated under B's code."""
    src = "expA/models/main"
    return {
        "b_full_reuse": dict(reps=(0, 1, 2), neural=(("full", FULL_N),), routes="eligible",
                             reuse_models=src),
        "b_ladder_reuse": dict(reps=(0, 1, 2), ladder_N=(FULL_N,), ladder_labels=LADDER_B,
                               routes="eligible", reuse_models=src),
    }


def b_stage_defs() -> dict[str, dict]:
    """Keyword arguments of every EXPERIMENT B stage. Dev-only stages (smoke, pilot, tune)
    and the test_b stages all declare ``test_stream="test_b"``."""
    defs: dict[str, dict] = {"b_smoke": _smoke_def(), "b_pilot": _pilot_def()}
    defs.update(_tune_defs())
    defs.update(_main_defs())
    defs.update(_independence_defs())
    defs.update(_reuse_defs())
    for kw in defs.values():
        kw.setdefault("test_stream", "test_b")
    return defs
