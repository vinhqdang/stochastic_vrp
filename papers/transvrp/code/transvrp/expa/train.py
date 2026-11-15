"""Per-stage neural training for EXPERIMENT A (module M5).

EXPERIMENT_A.md sections 2.3, 4.3 and M5.

``neural_fit_fn`` returns a ``FitFn`` for ``backward.ls_backward``. For each
epoch k, ls_backward calls the fit function once for fold A and once for fold
B (in that order). The fold of a call is read from the parity of its fitting
rows: fold A holds the even global indices of the fitting range, fold B the
odd ones (``backward.split_80_20``).

Stage procedure (one fold, one epoch k):

* no training if fewer than ``min_rows`` alive fitting rows: the warm net is
  carried over, and at k = m-1 without a warm net a constant predictor is
  used (mean target, or E[min(k, m-1)] if no row is alive);
* otherwise initialise from the warm net (deep copy) or from scratch with the
  seed ``crc32("inst|route|N|rep|fold|k")`` (arm excluded);
* Adam, batch 512 drawn with replacement from a seeded numpy Generator, cosine
  learning rate to 10% of its start value, gradient-norm clip;
* checkpoints at step 0 and every ceil(S/n_ckpt) steps up to S; the one with
  the lowest validation MSE is kept (ties to the earlier checkpoint).

Targets are u = cost / E0 in [0, 1]; predictions are returned as costs.
"""
from __future__ import annotations

import copy
import math
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .backward import FitFn, ls_backward, split_80_20
from .features import RouteCache, RouteStats, mlp_batch, stage_batch
from .model import MLPNet, SetNet
from .paths import Paths, RouteSpec

ARMS_NEURAL = ("full", "scalar", "mlp")
PRED_CHUNK = 8192


@dataclass
class StageConfig:
    lr_cold: float = 1e-3
    lr_warm: float = 3e-4
    batch: int = 512
    s_min: int = 50
    s_min_cold: int = 200
    s_max: int = 400
    epochs_factor: int = 20
    n_ckpt: int = 4
    clip: float = 1.0
    min_rows: int = 20


def _set_determinism() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def stage_steps(n: int, cold: bool, cfg: StageConfig) -> int:
    """S_k = clip(ceil(epochs_factor * n / batch), floor, s_max) with floor
    s_min_cold for the cold stage and s_min otherwise (section 4.3)."""
    floor = cfg.s_min_cold if cold else cfg.s_min
    raw = math.ceil(cfg.epochs_factor * n / cfg.batch)
    return int(min(cfg.s_max, max(floor, raw)))


class ConstantPredictor:
    """Constant cost, the neural analogue of BATON's _ConstantModel."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(idx).shape[0], self.value)


class NetPredictor:
    """A fitted network bound to a RouteCache at call time: features are
    built from the cache passed to ``predict`` (section 9, M5)."""

    def __init__(self, arm: str, net: nn.Module, E0: float):
        if arm not in ARMS_NEURAL:
            raise ValueError(f"arm must be one of {ARMS_NEURAL}")
        self.arm = arm
        self.net = net
        self.E0 = float(E0)
        self.net.eval()

    def _inputs(self, cache: RouteCache, k: int, idx: np.ndarray) -> tuple:
        if self.arm == "mlp":
            return (torch.from_numpy(mlp_batch(cache, k, idx)),)
        glob, tok = stage_batch(cache, k, idx, self.arm)
        return (torch.from_numpy(glob), torch.from_numpy(tok))

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros(idx.size, dtype=np.float64)
        for s in range(0, idx.size, PRED_CHUNK):
            sl = idx[s:s + PRED_CHUNK]
            xs = self._inputs(cache, k, sl)
            with torch.no_grad():
                u = self.net(*xs).numpy().astype(np.float64)
            out[s:s + sl.size] = self.E0 * u
        return out


def _chunked_forward(net: nn.Module, xs: tuple, n: int) -> torch.Tensor:
    parts = []
    with torch.no_grad():
        for s in range(0, n, PRED_CHUNK):
            parts.append(net(*[x[s:s + PRED_CHUNK] for x in xs]))
    return torch.cat(parts)


def _mse(net: nn.Module, xs: tuple, y: torch.Tensor) -> float | None:
    n = int(y.shape[0])
    if n == 0:
        return None
    was_training = net.training
    net.eval()
    pred = _chunked_forward(net, xs, n)
    net.train(was_training)
    return float(((pred - y) ** 2).mean())


def _fold_label(fit_idx: np.ndarray) -> str:
    if fit_idx.size == 0:
        return "none"
    if np.all(fit_idx % 2 == 0):
        return "A"
    if np.all(fit_idx % 2 == 1):
        return "B"
    raise ValueError("fitting rows of one call must belong to a single fold")


def _seed(seed_key: str, fold: str, k: int) -> int:
    return zlib.crc32(f"{seed_key}|{fold}|{k}".encode())


def _train_stage(net: nn.Module, xs: tuple, y: torch.Tensor,
                 xs_val: tuple | None, y_val: torch.Tensor, *,
                 lr0: float, cfg: StageConfig, steps: int, seed: int) -> dict:
    n = int(y.shape[0])
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr0)
    every = math.ceil(steps / cfg.n_ckpt)
    ckpt_set = {0} | {s for s in range(every, steps + 1, every)}

    train_first = _mse(net, xs, y)
    val_hist: list[tuple[int, float]] = []
    best_mse = math.inf
    best_step = 0
    best_state = None
    select_on = "val" if y_val.shape[0] else "train"

    def consider(step: int) -> None:
        nonlocal best_mse, best_step, best_state
        v = _mse(net, xs_val, y_val) if select_on == "val" else _mse(net, xs, y)
        if v is None:
            v = _mse(net, xs, y)
        val_hist.append((step, float(v)))
        if best_state is None or v < best_mse:
            best_mse, best_step = float(v), step
            best_state = copy.deepcopy(net.state_dict())

    consider(0)
    net.train()
    for t in range(steps):
        lr = lr0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t / steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        bi = torch.from_numpy(rng.integers(0, n, size=cfg.batch))
        pred = net(*[x[bi] for x in xs])
        loss = ((pred - y[bi]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
        opt.step()
        if (t + 1) in ckpt_set:
            consider(t + 1)
    train_last = _mse(net, xs, y)
    net.load_state_dict(best_state)
    return dict(train_mse_first=train_first, train_mse_last=train_last,
                val_mse_per_ckpt=val_hist, chosen_ckpt=best_step,
                select_on=select_on, ckpt_every=every)


def neural_fit_fn(arm: str, seed_key: str, cfg: StageConfig,
                  diag: list) -> FitFn:
    """FitFn for ls_backward. ``seed_key`` is ``"inst|route_id|N|rep"``;
    diagnostics are appended to ``diag``."""
    if arm not in ARMS_NEURAL:
        raise ValueError(f"arm must be one of {ARMS_NEURAL}")

    def fit(k, cache, fit_idx, y_fit, val_idx, y_val, warm):
        t0 = time.perf_counter()
        spec: RouteSpec = cache.spec
        m, E0 = cache.m, spec.E0
        if E0 <= 0:
            raise ValueError("E0 must be positive")
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        y_fit = np.asarray(y_fit, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        n = int(fit_idx.size)
        fold = _fold_label(fit_idx)
        if n and (y_fit.max() > E0 + 1e-9 or y_fit.min() < -1e-9):
            raise ValueError("fitting targets must lie in [0, E0]")
        if y_val.size and (y_val.max() > E0 + 1e-9 or y_val.min() < -1e-9):
            raise ValueError("validation targets must lie in [0, E0]")

        if n < cfg.min_rows:
            if warm is not None:
                action, pred = "carry", warm
            else:
                value = float(y_fit.mean()) if n else float(spec.E[min(k, m - 1)])
                action, pred = "constant", ConstantPredictor(value)
            diag.append(dict(kind="stage", arm=arm, k=k, fold=fold, n_fit=n,
                             n_val=int(val_idx.size), action=action, steps=0,
                             seconds=time.perf_counter() - t0))
            return pred

        warm_net = isinstance(warm, NetPredictor)
        cold = not warm_net
        _set_determinism()
        torch.manual_seed(_seed(seed_key, fold, k))
        if warm_net:
            net = copy.deepcopy(warm.net)
        elif arm == "mlp":
            net = MLPNet()
        else:
            net = SetNet()
        lr0 = cfg.lr_cold if cold else cfg.lr_warm
        steps = stage_steps(n, cold, cfg)

        if arm == "mlp":
            xs = (torch.from_numpy(mlp_batch(cache, k, fit_idx)),)
            xs_val = ((torch.from_numpy(mlp_batch(cache, k, val_idx)),)
                      if val_idx.size else None)
        else:
            glob, tok = stage_batch(cache, k, fit_idx, arm)
            xs = (torch.from_numpy(glob), torch.from_numpy(tok))
            if val_idx.size:
                gv, tv = stage_batch(cache, k, val_idx, arm)
                xs_val = (torch.from_numpy(gv), torch.from_numpy(tv))
            else:
                xs_val = None
        y = torch.from_numpy((y_fit / E0).astype(np.float32))
        y_v = torch.from_numpy((y_val / E0).astype(np.float32))

        info = _train_stage(net, xs, y, xs_val, y_v, lr0=lr0, cfg=cfg,
                            steps=steps, seed=_seed(seed_key, fold, k))
        diag.append(dict(kind="stage", arm=arm, k=k, fold=fold, n_fit=n,
                         n_val=int(val_idx.size),
                         action="warm" if warm_net else "cold", steps=steps,
                         lr0=lr0, seconds=time.perf_counter() - t0, **info))
        return NetPredictor(arm, net, E0)

    return fit


def fit_neural(spec: RouteSpec, paths: Paths, arm: str, rep: int, N: int,
               cfg: StageConfig) -> tuple[dict, RouteStats, list]:
    """Cross-fitted neural continuation models for one route.

    Returns ({k: [net_A, net_B]}, RouteStats fitted on the fitting rows, diag).
    Evaluation caches must be built with the returned RouteStats."""
    if paths.p.shape[0] != N:
        raise ValueError("paths must hold exactly N rows")
    if np.any(spec.H > spec.E0 + 1e-12) or np.any(spec.E > spec.E0 + 1e-12):
        raise ValueError("H and E must not exceed E0")
    folds, val = split_80_20(N)
    fit_all = np.concatenate(folds)
    stats = RouteStats.fit(Paths(p=paths.p[fit_all], d=paths.d[fit_all]))
    cache = RouteCache(spec, paths, stats)
    seed_key = f"{spec.inst}|{spec.route_id}|{N}|{rep}"
    diag: list = []
    log: list = []
    models = ls_backward(cache, spec.B, spec.H, spec.E,
                         neural_fit_fn(arm, seed_key, cfg, diag),
                         folds, val, log=log)
    for e in log:
        diag.append(dict(kind="ls", k=e["k"], fold=e["fold"],
                         n_fit=int(e["fit_rows"].size),
                         n_applied=int(e["applied_rows"].size),
                         disagree_rate=float(e["disagree_rate"])))
    return models, stats, diag
