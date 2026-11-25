"""Payload codecs for the B arms (EXPERIMENT_B.md 10 M20; REVIEW_B 9.1, gate item 8).

``pack_models`` turns the {k: [predictor, ...]} dict of a B fit into plain Python, NumPy
and CPU-tensor data (picklable). Every predictor type of the six arms is handled. The DQN
and PPO networks are stored once per job in ``nets`` and referenced by index. The blob
records ``libraries`` (the gate's library versions, gatecheck.library_versions).
``unpack_models`` rebuilds predictors whose predictions equal the packed ones, with the
packed weights.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..expa.train import ConstantPredictor
from .dos import DOSNet, DOSPredictor
from .dqn import QPredictor
from .gatecheck import library_versions
from .grid import GridPredictor
from .padded import SetNetMasked, masked_net
from .ppo import PolicyPredictor, PPONet
from .registry import B_ARMS
from .rlsm import RLSMPredictor


def _cpu_state(net: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _pack_one(pred: Any, nets: list, index: dict) -> dict:
    """One predictor -> plain entry. Shared networks go to ``nets`` once (keyed by id)."""

    def net_ref(net: torch.nn.Module, kind: str) -> int:
        if id(net) not in index:
            index[id(net)] = len(nets)
            nets.append({"kind": kind, "state": _cpu_state(net)})
        return index[id(net)]

    if isinstance(pred, GridPredictor):
        return {"kind": "grid", "w_grid": np.array(pred.w_grid, copy=True),
                "c": np.array(pred.c, copy=True), "tie_tol": float(pred.tie_tol)}
    if isinstance(pred, ConstantPredictor):
        return {"kind": "const", "value": float(pred.value)}
    if isinstance(pred, RLSMPredictor):
        # the feature-set mask (EXPERIMENT_C 1.3) travels with the predictor by name; B's
        # predictors have no set and store None (their input map is mlp_batch)
        return {"kind": "rlsm", "A": np.array(pred.A, copy=True),
                "b": np.array(pred.b, copy=True), "theta": np.array(pred.theta, copy=True),
                "slope": float(pred.slope), "E0": float(pred.E0),
                "set": getattr(getattr(pred, "featurize", None), "set_name", None)}
    if isinstance(pred, DOSPredictor):
        return {"kind": "dos", "in_dim": int(pred.net.body[0].in_features),
                "width": int(pred.net.out.in_features), "state": _cpu_state(pred.net),
                "E0": float(pred.E0)}
    if isinstance(pred, QPredictor):
        return {"kind": "q", "net": net_ref(pred.net, "q"), "E0": float(pred.E0)}
    if isinstance(pred, PolicyPredictor):
        return {"kind": "ppo", "net": net_ref(pred.net, "ppo"), "E0": float(pred.E0)}
    raise TypeError(f"cannot pack predictor {type(pred).__name__}")


def pack_models(models: dict, arm: str, E0: float) -> dict:
    """Payload form of the models of one B fit."""
    if arm not in B_ARMS:
        raise ValueError(f"arm must be one of {B_ARMS}, got {arm!r}")
    nets: list = []
    index: dict = {}
    states = {int(k): [_pack_one(p, nets, index) for p in preds]
              for k, preds in models.items()}
    return {"arm": arm, "E0": float(E0), "libraries": library_versions(),
            "nets": nets, "states": states}


def _build_net(entry: dict) -> torch.nn.Module:
    if entry["kind"] == "q":
        net: torch.nn.Module = masked_net()
    elif entry["kind"] == "ppo":
        net = PPONet()
    else:
        raise ValueError(f"unknown network kind {entry['kind']!r}")
    net.load_state_dict(entry["state"])
    net.eval()
    return net


def _unpack_one(item: dict, nets: list, ppo: dict) -> Any:
    kind = item["kind"]
    if kind == "grid":
        return GridPredictor(item["w_grid"], item["c"], item["tie_tol"])
    if kind == "const":
        return ConstantPredictor(item["value"])
    if kind == "rlsm":
        set_name = item.get("set")
        featurize = None
        if set_name:                      # lazy: expc depends on expb, not the reverse
            from ..expc.rlsm_sets import masked_featurize   # noqa: PLC0415
            featurize = masked_featurize(set_name)
        return RLSMPredictor(item["A"], item["b"], item["theta"], item["slope"], item["E0"],
                             featurize=featurize)
    if kind == "dos":
        net = DOSNet(in_dim=item["in_dim"], width=item["width"])
        net.load_state_dict(item["state"])
        net.eval()
        return DOSPredictor(net, item["E0"])
    if kind in ("q", "ppo"):
        net = nets[item["net"]]
        if kind == "q" and not isinstance(net, SetNetMasked):
            raise ValueError("a DQN predictor must reference a masked SetNet")
        if kind == "ppo" and not isinstance(net, PPONet):
            raise ValueError("a PPO predictor must reference a PPO network")
        if kind == "q":
            return QPredictor(net, item["E0"])
        # one predictor per net: the GRU cache then serves every stage (M6)
        key = (item["net"], item["E0"])
        if key not in ppo:
            ppo[key] = PolicyPredictor(net, item["E0"])
        return ppo[key]
    raise ValueError(f"unknown predictor kind {kind!r}")


def unpack_models(blob: dict) -> dict:
    """Inverse of ``pack_models``: {k: [predictor, ...]} with shared networks shared."""
    if blob.get("arm") not in B_ARMS:
        raise ValueError(f"payload arm must be one of {B_ARMS}, got {blob.get('arm')!r}")
    nets = [_build_net(entry) for entry in blob["nets"]]
    ppo: dict = {}
    return {int(k): [_unpack_one(it, nets, ppo) for it in items]
            for k, items in blob["states"].items()}
