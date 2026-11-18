"""Per-pass checkpoint curves of the DQN and PPO tuning runs (EXPERIMENT_B.md 6.3, M21).

One run trains P_max passes. The in-run selection restricted to checkpoints up to pass p
is the state kept at the end of pass p (``pass_states`` of the training record). Each such
state becomes a predictor set, so a tuning stage can score every pass of one run. The
prediction and budget conventions are those of ``fit_dqn`` and ``fit_ppo``.
"""
from __future__ import annotations

from ..expa.features import RouteStats
from ..expa.paths import Paths, RouteSpec
from .dqn import DQNConfig, QPredictor, train_dqn_record
from .padded import masked_net
from .ppo import PolicyPredictor, PPOConfig, PPONet, train_ppo_record

CURVE_ARMS: tuple[str, ...] = ("dqn", "ppo")


def _predictor(arm: str, state: dict, cfg: DQNConfig | PPOConfig, E0: float):
    """Predictor of one pass-end state (the network is loaded in eval mode)."""
    if arm == "dqn":
        net = masked_net()
        net.load_state_dict(state)
        net.eval()
        return QPredictor(net, E0)
    net = PPONet(cfg.init_handoff_p)
    net.load_state_dict(state)
    net.eval()
    return PolicyPredictor(net, E0)


def fit_pass_curves(arm: str, spec: RouteSpec, paths: Paths, N: int, rep: int,
                    cfg: DQNConfig | PPOConfig) -> tuple[dict, RouteStats, list, dict]:
    """Trains one DQN or PPO run and returns ({pass: {k: [predictor]}}, RouteStats, diag,
    budget) with one entry per pass 1..cfg.passes."""
    if arm not in CURVE_ARMS:
        raise ValueError(f"pass curves exist for {CURVE_ARMS}, not for {arm!r}")
    if arm == "dqn":
        rec = train_dqn_record(spec, paths, N, rep, cfg, keep_states=True)
    else:
        rec = train_ppo_record(spec, paths, N, rep, cfg, keep_states=True)
    per_pass = {}
    for p, state in enumerate(rec["pass_states"], start=1):
        pred = _predictor(arm, state, cfg, spec.E0)
        per_pass[p] = {k: [pred] for k in range(1, spec.m)}
    n_ckpt = len(rec["checkpoints"])
    diag = list(rec["checkpoints"])
    diag.append(dict(kind="summary", updates=rec["updates"], pass_best=rec["pass_best"]))
    budget = dict(
        unique_paths=int(N), path_stop_reads=rec["env_steps"],
        val_reads=int(n_ckpt * rec["n_val"] * spec.m), updates=rec["updates"],
        n_params=rec["n_params"], feature_seconds=rec["feature_seconds"],
        train_seconds=rec["train_seconds"], val_seconds=rec["val_seconds"],
        device=rec["device"], passes=int(cfg.passes),
    )
    return per_pass, rec["stats"], diag, budget
