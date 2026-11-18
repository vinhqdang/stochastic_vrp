"""Registry of the six B arms (EXPERIMENT_B.md 10 M20).

Each arm maps to its fit function, its configuration class and two descriptive flags
from section 3: ``split`` (learned arms fit on the first 80% of paths, 3.0) and
``predictor`` ("cost" for a continuation-cost estimate, "surrogate" for a decision
surrogate, 2). ``fit_arm`` returns the four values of the fit functions:
(models, RouteStats, diag, budget). The stage specifications that use these arms are
built in M21 (sweep integration).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..expa.features import RouteStats
from ..expa.paths import Paths, RouteSpec
from .alp import ALPZConfig, fit_alpz
from .dos import DOSConfig, fit_dos
from .dqn import DQNConfig, fit_dqn
from .ppo import PPOConfig, fit_ppo
from .rlsm import RLSMConfig, fit_rlsm
from .thr import THRConfig, fit_thr

FitResult = tuple[dict, RouteStats, list, dict]


@dataclass(frozen=True)
class ArmSpec:
    fit: Callable[..., FitResult]
    config_cls: type
    split: bool
    predictor: str           # "cost" or "surrogate"


ARM_SPECS: dict[str, ArmSpec] = {
    "thr": ArmSpec(fit_thr, THRConfig, split=False, predictor="cost"),
    "alpz": ArmSpec(fit_alpz, ALPZConfig, split=False, predictor="cost"),
    "rlsm": ArmSpec(fit_rlsm, RLSMConfig, split=True, predictor="cost"),
    "dos": ArmSpec(fit_dos, DOSConfig, split=True, predictor="surrogate"),
    "dqn": ArmSpec(fit_dqn, DQNConfig, split=True, predictor="cost"),
    "ppo": ArmSpec(fit_ppo, PPOConfig, split=True, predictor="surrogate"),
}
B_ARMS: tuple[str, ...] = ("thr", "alpz", "rlsm", "dos", "dqn", "ppo")
ARM_POLICY_B: dict[str, str] = {a: a for a in B_ARMS}


def fit_arm(arm: str, spec: RouteSpec, paths: Paths, N: int, rep: int,
            cfg: object | None = None) -> FitResult:
    """Fit one B arm on one route. ``cfg`` is the arm's configuration class instance, or
    None for the class defaults."""
    if arm not in ARM_SPECS:
        raise ValueError(f"arm must be one of {B_ARMS}, got {arm!r}")
    entry = ARM_SPECS[arm]
    if cfg is None:
        cfg = entry.config_cls()
    elif not isinstance(cfg, entry.config_cls):
        raise TypeError(f"arm {arm!r} needs a {entry.config_cls.__name__}, "
                        f"got {type(cfg).__name__}")
    return entry.fit(spec, paths, N, rep, cfg)
