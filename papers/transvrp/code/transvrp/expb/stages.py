"""EXPERIMENT B stage definitions as plain data (EXPERIMENT_B.md 10 M21).

``expa.sweep`` turns each entry of ``b_stage_defs()`` into a ``StageSpec``. Arm
configurations are tuples of (field, value) pairs applied to the arm's configuration
class (registry.ARM_SPECS); ``()`` is the class default. A stage's ``arm_cfg`` is a tuple
of (arm, pairs) so that one stage can hold several arms (smoke).

The grids are those of sections 3 and 6.1. ``CHOSEN_ARM_CFG`` is read from the committed
tuning record (``choose.py``) before the main fit; an arm missing from it has no chosen
configuration and its main stage refuses to fit. The ordered cuts of section 9.2 are the
``Cuts`` configuration below.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

from ..expa.train import StageConfig
from .choose import load_chosen
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

# Chosen configurations (6.1): THR and ALP-Z have no grid; every other arm is read from the
# committed tuning record written by ``python -m transvrp.expb.choose`` (never typed).
TUNING_RECORD = Path(__file__).resolve().parents[2] / "results" / "expB" / "tuning_record.json"


def chosen_arm_cfg(record: Path) -> dict[str, Pairs]:
    """THR and ALP-Z (no grid) plus every arm of the committed tuning record."""
    return {"thr": (), "alpz": (), **load_chosen(record, TUNE_GRID)}


CHOSEN_ARM_CFG: dict[str, Pairs] = chosen_arm_cfg(TUNING_RECORD)

REUSE_DISABLED = ("cut (i), reuse of A's models, was struck in EXPERIMENT_B revision 3.3: "
                  "A's payloads carry no library record (gate item 8)")
TUNE_ROUTES = ("subset", "subset6")


@dataclass(frozen=True)
class Cuts:
    """Ordered cuts of section 9.2 (revision 3.3) as a pre-committed configuration. The
    default is no cut; applying one is an edit of ``ACTIVE_CUTS`` before the fit commit F,
    so it enters code_sha256, every B fit payload (``cuts``) and, for the first cut, the
    FAMILY ``dropped`` list as ``dos:cut``."""
    drop_dos: bool = False                  # cut (i)
    tune_routes: str = "subset"             # cut (ii): "subset6" = 3 medium + 3 long
    reps: tuple = (0, 1, 2)                 # cut (iii): (0, 1)
    p_main: int | None = None               # cut (iv): 1 pass for DQN and PPO main fits

    def __post_init__(self) -> None:
        if self.tune_routes not in TUNE_ROUTES:
            raise ValueError(f"tune_routes must be one of {TUNE_ROUTES}")
        if tuple(self.reps) not in ((0, 1, 2), (0, 1)):
            raise ValueError("reps must be (0, 1, 2) or, under cut (iii), (0, 1)")
        if self.p_main not in (None, 1):
            raise ValueError("p_main is None or 1 (cut (iv))")


ACTIVE_CUTS = Cuts()

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


def cut_record(cuts: Cuts) -> dict:
    return {"drop_dos": cuts.drop_dos, "tune_routes": cuts.tune_routes,
            "reps": list(cuts.reps), "p_main": cuts.p_main}


def dropped_labels(cuts: Cuts) -> list[str]:
    """``dropped`` entries a cut forces into the FAMILY line (spelling without spaces)."""
    return ["dos:cut"] if cuts.drop_dos else []


def active_arms(cuts: Cuts) -> tuple[str, ...]:
    return tuple(a for a in B_ARMS if not (cuts.drop_dos and a == "dos"))


def pilot_cfg(arm: str) -> Pairs:
    """Pilot cells use the first grid point, with one pass for DQN and PPO (9.3 times one
    full pass; the pilot changes no grid)."""
    first = TUNE_GRID[arm][0]
    return first + (("passes", 1),) if arm in ("dqn", "ppo") else first


def _arm_cfg(arms: tuple[str, ...], pairs_of) -> tuple:
    return tuple((a, pairs_of(a)) for a in arms)


def _main_arm_cfg(arm: str, chosen: dict, cuts: Cuts) -> tuple:
    if arm not in chosen:
        return ()
    pairs = chosen[arm]
    if cuts.p_main is not None and arm in ("dqn", "ppo"):     # cut (iv): P = 1 regardless of P*
        pairs = tuple(p for p in pairs if p[0] != "passes") + (("passes", cuts.p_main),)
    return ((arm, pairs),)


def _smoke_def(arms: tuple) -> dict:
    return dict(reps=(0,), ladder_N=(2_000,), ladder_labels=LADDER_B,
                neural=(("full", 2_000),) + tuple((a, 2_000) for a in arms),
                routes="smoke", cfg=StageConfig(s_max=10), test_allowed=False,
                arm_cfg=_arm_cfg(arms, lambda a: SMOKE_CFG[a]))


def _pilot_def(arms: tuple) -> dict:
    return dict(reps=(0, 1), ladder_N=(10_000,), ladder_labels=LADDER_B,
                neural=(("full", 10_000),) + tuple((a, 10_000) for a in arms),
                timing=tuple((a, FULL_N) for a in arms), routes="eligible",
                test_allowed=False, arm_cfg=_arm_cfg(arms, pilot_cfg))


def _tune_defs(arms: tuple, cuts: Cuts) -> dict[str, dict]:
    """6.1: dev only, the 12 subset routes (6 under cut (ii)), rep 0; baton_eq on the same
    rows for scoring."""
    out = {}
    for arm in arms:
        for i, pairs in enumerate(TUNE_GRID[arm]):
            out[f"b_tune_{arm}_c{i}"] = dict(
                reps=(0,), ladder_N=(FULL_N,), ladder_labels=("baton_eq",),
                neural=((arm, FULL_N),), routes=cuts.tune_routes, test_allowed=False,
                arm_cfg=((arm, pairs),),
                pass_curve=3 if arm in ("dqn", "ppo") else 0)
    return out


def _main_defs(arms: tuple, chosen: dict, cuts: Cuts) -> dict[str, dict]:
    reps = tuple(cuts.reps)
    out = {
        "b_ladder": dict(reps=reps, ladder_N=(FULL_N,), ladder_labels=LADDER_B,
                         routes="eligible"),
        "b_full": dict(reps=reps, neural=(("full", FULL_N),), routes="eligible"),
    }
    for arm in arms:
        out[f"b_{arm}"] = dict(reps=reps, neural=((arm, FULL_N),), routes="eligible",
                               arm_cfg=_main_arm_cfg(arm, chosen, cuts))
    return out


def _independence_defs(chosen: dict, cuts: Cuts) -> dict[str, dict]:
    """3.8: the 12 subset routes, rep 0, every stop column permuted."""
    ind = dict(reps=(0,), routes="subset", independent=True)
    out = {
        "b_ladder_ind": dict(ind, ladder_N=(FULL_N,), ladder_labels=LADDER_B),
        "b_full_ind": dict(ind, neural=(("full", FULL_N),)),
    }
    for arm in ("thr", "alpz"):
        out[f"b_{arm}_ind"] = dict(ind, neural=((arm, FULL_N),),
                                   arm_cfg=_main_arm_cfg(arm, chosen, cuts))
    return out


def _reuse_defs() -> dict[str, dict]:
    """Cut (i) of revisions 1 to 3.2, struck in 3.3: kept as disabled stages (every entry
    point refuses; see ``REUSE_DISABLED``)."""
    src = "expA/models/main"
    return {
        "b_full_reuse": dict(reps=(0, 1, 2), neural=(("full", FULL_N),), routes="eligible",
                             reuse_models=src, disabled=REUSE_DISABLED),
        "b_ladder_reuse": dict(reps=(0, 1, 2), ladder_N=(FULL_N,), ladder_labels=LADDER_B,
                               routes="eligible", reuse_models=src, disabled=REUSE_DISABLED),
    }


def b_stage_defs(cuts: Cuts | None = None, chosen: dict | None = None) -> dict[str, dict]:
    """Keyword arguments of every EXPERIMENT B stage under a cut set (default
    ``ACTIVE_CUTS``) and the chosen configurations (default ``CHOSEN_ARM_CFG``). Dev-only
    stages (smoke, pilot, tune) and the test_b stages all declare ``test_stream="test_b"``."""
    cuts = ACTIVE_CUTS if cuts is None else cuts
    chosen = CHOSEN_ARM_CFG if chosen is None else chosen
    arms = active_arms(cuts)
    defs: dict[str, dict] = {"b_smoke": _smoke_def(arms), "b_pilot": _pilot_def(arms)}
    defs.update(_tune_defs(arms, cuts))
    defs.update(_main_defs(arms, chosen, cuts))
    defs.update(_independence_defs(chosen, cuts))
    defs.update(_reuse_defs())
    for kw in defs.values():
        kw.setdefault("test_stream", "test_b")
    return defs
