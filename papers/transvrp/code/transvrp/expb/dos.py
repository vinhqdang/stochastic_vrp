"""DOS: deep optimal stopping (Becker, Cheridito & Jentzen, JMLR 2019; EXPERIMENT_B.md 3.4;
M16).

A FitFn for ``backward.ls_backward`` with A's folds. The network F(x) in (0, 1) on the
15 ``mlp_batch`` features minimises mean over fitting rows of [F H[k-1] + (1 - F) y] / E0
(stop costs H, continue costs the target y). Architecture: two hidden layers of width
55 with batch normalisation, ReLU, sigmoid output, Xavier init, Adam at a constant
learning rate (placement of batch norm and the schedule are [DC]). Batches are drawn with
replacement from the fitting rows (the fixed-sample adaptation of 3.4).

Decision surrogate: the predictor returns S = H[k-1] + E0 (pi - 1/2), pi = F. S exceeds
H[k-1] iff pi > 1/2, so the fold mean of S decides at pi = 1/2. S is never read as a cost.
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

from ..expa.backward import FitFn, ls_backward, split_80_20
from ..expa.features import MLP_DIM, RouteCache, RouteStats, mlp_batch
from ..expa.paths import Paths, RouteSpec
from ..expa.train import ConstantPredictor, _fold_label

EPOCHS_FACTOR = 20          # S_k = clip(ceil(20 n_k / batch), s_min, s_max) (3.4)
PRED_CHUNK = 8192


@dataclass(frozen=True)
class DOSConfig:
    lr: float = 1e-3         # grid {1e-3, 3e-4} (3.4)
    batch: int = 8192
    s_min: int = 50
    s_max: int = 400
    width: int = 55
    n_ckpt: int = 4
    min_rows: int = 20


class DOSNet(nn.Module):
    """Two hidden layers (batch norm, ReLU) and a sigmoid output."""

    def __init__(self, in_dim: int = MLP_DIM, width: int = 55):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.out = nn.Linear(width, 1)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (n, 15) -> (n,) in (0, 1)."""
        return torch.sigmoid(self.out(self.body(x))).squeeze(-1)


def dos_loss(F: torch.Tensor, u: torch.Tensor, Hs: float) -> torch.Tensor:
    """Mean of F Hs + (1 - F) u, with u = y / E0 and Hs = H[k-1] / E0."""
    return (F * Hs + (1.0 - F) * u).mean()


def stage_steps(n: int, cfg: DOSConfig) -> int:
    raw = math.ceil(EPOCHS_FACTOR * n / cfg.batch)
    return int(min(cfg.s_max, max(cfg.s_min, raw)))


def init_net(warm, seed: int) -> DOSNet:
    """Warm start: a copy of the same fold's stage-(k+1) network. Otherwise (no warm net,
    or a ConstantPredictor) a fresh network seeded by ``seed``."""
    if isinstance(warm, DOSPredictor):
        return copy.deepcopy(warm.net)
    torch.manual_seed(seed)
    return DOSNet()


class DOSPredictor:
    """Decision surrogate of a fitted network, read through ``simulate_policy``."""

    def __init__(self, net: DOSNet, E0: float):
        self.net = net
        self.E0 = float(E0)

    def pi(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        """Stop probability F(x) of the rows ``idx``."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros(idx.size)
        if idx.size == 0:
            return out
        self.net.eval()
        with torch.no_grad():
            for s in range(0, idx.size, PRED_CHUNK):
                sl = idx[s:s + PRED_CHUNK]
                x = torch.from_numpy(mlp_batch(cache, k, sl))
                out[s:s + sl.size] = self.net(x).numpy().astype(np.float64)
        return out

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        """Surrogate S = H[k-1] + E0 (pi - 1/2)."""
        return cache.spec.H[k - 1] + self.E0 * (self.pi(cache, k, idx) - 0.5)


def _mean_loss(net: nn.Module, x: torch.Tensor, u: torch.Tensor, Hs: float) -> float:
    net.eval()
    total = 0.0
    with torch.no_grad():
        for s in range(0, x.shape[0], PRED_CHUNK):
            F = net(x[s:s + PRED_CHUNK])
            total += float((F * Hs + (1.0 - F) * u[s:s + PRED_CHUNK]).sum())
    return total / x.shape[0]


def _seed(seed_key: str, fold: str, k: int) -> int:
    return zlib.crc32(f"{seed_key}|{fold}|{k}".encode("utf-8"))


def _constant_surrogate(y_fit: np.ndarray, Hk: float, E0: float, fallback: float
                        ) -> ConstantPredictor:
    """Constant decision with the lower empirical loss: stop (pi = 1) iff H < mean y."""
    cont = float(y_fit.mean()) if y_fit.size else fallback
    pi = 1.0 if Hk < cont else 0.0
    return ConstantPredictor(Hk + E0 * (pi - 0.5))


def _train(net: nn.Module, x: torch.Tensor, u: torch.Tensor, Hs: float,
           x_val, u_val, *, cfg: DOSConfig, steps: int, seed: int) -> dict:
    n = int(x.shape[0])
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    every = math.ceil(steps / cfg.n_ckpt)
    ckpt_set = {0} | set(range(every, steps + 1, every))
    has_val = x_val is not None
    train_hist: list = []
    val_hist: list = []
    best = {"score": math.inf, "step": 0, "state": None}

    def consider(step: int) -> None:
        tl = _mean_loss(net, x, u, Hs)
        train_hist.append((step, tl))
        score = tl
        if has_val:
            vl = _mean_loss(net, x_val, u_val, Hs)
            val_hist.append((step, vl))
            score = vl
        if best["state"] is None or score < best["score"]:
            best.update(score=score, step=step, state=copy.deepcopy(net.state_dict()))

    consider(0)
    for t in range(steps):
        net.train()
        bi = torch.from_numpy(rng.integers(0, n, size=cfg.batch))
        F = net(x[bi])
        loss = dos_loss(F, u[bi], Hs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (t + 1) in ckpt_set:
            consider(t + 1)
    net.load_state_dict(best["state"])
    return dict(train_loss_per_ckpt=train_hist, val_loss_per_ckpt=val_hist,
                chosen_ckpt=int(best["step"]), select_on="val" if has_val else "train",
                ckpt_every=every)


def dos_fit_fn(seed_key: str, cfg: DOSConfig, diag: list) -> FitFn:
    """FitFn for ls_backward; diagnostics are appended to ``diag``."""

    def fit(k, cache, fit_idx, y_fit, val_idx, y_val, warm):
        t0 = time.perf_counter()
        spec: RouteSpec = cache.spec
        m, E0, Hk = cache.m, spec.E0, float(spec.H[k - 1])
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        y_fit = np.asarray(y_fit, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        if y_fit.size and (y_fit.max() > E0 + 1e-9 or y_fit.min() < -1e-9):
            raise ValueError("fitting targets must lie in [0, E0]")
        n = int(fit_idx.size)
        fold = _fold_label(fit_idx)
        if n < cfg.min_rows:
            action = "carry" if isinstance(warm, DOSPredictor) else "constant"
            pred = warm if action == "carry" else _constant_surrogate(
                y_fit, Hk, E0, float(spec.E[min(k, m - 1)]))
            diag.append(dict(kind="stage", arm="dos", k=k, fold=fold, n_fit=n,
                             n_val=int(val_idx.size), action=action, steps=0,
                             seconds=time.perf_counter() - t0))
            return pred

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        seed = _seed(seed_key, fold, k)
        net = init_net(warm, seed)
        x = torch.from_numpy(mlp_batch(cache, k, fit_idx))
        u = torch.from_numpy((y_fit / E0).astype(np.float32))
        if val_idx.size:
            x_val = torch.from_numpy(mlp_batch(cache, k, val_idx))
            u_val = torch.from_numpy((y_val / E0).astype(np.float32))
        else:
            x_val, u_val = None, None
        steps = stage_steps(n, cfg)
        info = _train(net, x, u, Hk / E0, x_val, u_val, cfg=cfg, steps=steps, seed=seed)
        diag.append(dict(kind="stage", arm="dos", k=k, fold=fold, n_fit=n,
                         n_val=int(val_idx.size),
                         action="warm" if isinstance(warm, DOSPredictor) else "cold",
                         steps=steps, lr0=cfg.lr, seconds=time.perf_counter() - t0, **info))
        return DOSPredictor(net, E0)

    return fit


def fit_dos(spec: RouteSpec, paths: Paths, N: int, rep: int, cfg: DOSConfig
            ) -> tuple[dict, RouteStats, list, dict]:
    """Cross-fitted DOS surrogates for one route (A's split and statistics). Returns
    (models {k: [net_A, net_B]}, RouteStats, diag, budget)."""
    if paths.p.shape[0] != N:
        raise ValueError(f"expected {N} training paths, got {paths.p.shape[0]}")
    t0 = time.perf_counter()
    folds, val = split_80_20(N)
    fit_all = np.concatenate(folds)
    stats = RouteStats.fit(Paths(p=paths.p[fit_all], d=paths.d[fit_all]))
    cache = RouteCache(spec, paths, stats)
    seed_key = f"{spec.inst}|{spec.route_id}|{N}|{rep}|dos"
    diag: list = []
    log: list = []
    models = ls_backward(cache, spec.B, spec.H, spec.E, dos_fit_fn(seed_key, cfg, diag),
                         folds, val, log=log)
    for e in log:
        diag.append(dict(kind="ls", k=e["k"], fold=e["fold"],
                         n_fit=int(e["fit_rows"].size),
                         n_applied=int(e["applied_rows"].size),
                         disagree_rate=float(e["disagree_rate"])))
    net_params = sum(p.numel() for p in DOSNet(width=cfg.width).parameters())
    fitted = sum(1 for d in diag if d.get("action") in ("cold", "warm"))
    budget = {"unique_paths": int(N), "path_stop_reads": int(N * spec.m),
              "val_reads": int(sum(d.get("n_val", 0) for d in diag if d.get("kind") == "stage")),
              "n_params": int(fitted * net_params), "feature_seconds": 0.0,
              "train_seconds": time.perf_counter() - t0, "val_seconds": 0.0,
              "device": "cpu"}
    return models, stats, diag, budget
