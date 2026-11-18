"""PPO after Kadyrov et al. (EXPERIMENT_B.md 3.6; 10 M19).

Two actions per decision epoch k = 1..m-1: CONTINUE (0) and HANDOFF (1). The policy is a
masked SetNet trunk (``SetNetMasked``, padded.py) pooled to 2d = 128 features, a GRU cell
(128 -> 64) carrying the episode state across decision epochs, a linear policy head and a
value head 64 -> 64 -> 1 (ReLU between). Handoff is never an action at k = m.

Rewards are in E0 units: a handoff at k earns -H[k-1]/E0 and ends the episode; a
continuation that breaches at stop k+1 earns -E[k]/E0 and ends it; completion earns 0.
Returns are discounted with gamma; the advantage is the return minus the value of the
rollout (no GAE, no advantage normalisation). Training is one episode per update (n_env =
1): roll out the next fitting path with the current policy (sampling), then take
``epochs`` gradient steps on the clipped surrogate of that episode. The policy head's
weights start at zero and its bias makes the handoff probability exactly ``init_handoff_p``
(T11) at every state.

The deployed ``PolicyPredictor`` returns the surrogate H[k-1] + E0 (pi - 1/2), with pi the
greedy handoff probability, and caches the GRU state between consecutive calls k-1 -> k on
the same cache. Seeds: crc32 of "inst|route|N|rep|ppo|main|<stream>".
"""
from __future__ import annotations

import copy  # noqa: F401  (kept for parity with dqn.py helpers)
import hashlib
import math
import os
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..expa.backward import split_80_20
from ..expa.features import RouteCache, RouteStats, stage_batch
from ..expa.model import NetConfig
from ..expa.paths import Paths, RouteSpec
from ..expa.simulate import simulate_policy
from .padded import PaddedTables, SetNetMasked, masked_net

# cuBLAS determinism (3.0); must be set before CUDA initialises, as in dqn.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CONTINUE, HANDOFF = 0, 1
POOL_DIM = 2 * NetConfig().d          # global token and masked token mean: 128
GRU_HIDDEN = 64


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 1e-4                  # grid {1e-4, 3e-4} (3.6)
    gamma: float = 0.5                # grid {0.5, 1.0} (3.6)
    passes: int = 3                   # P_max = 3 for tuning (6.3)
    epochs: int = 4                   # epochs per episode (source)
    clip: float = 0.2                 # epsilon of the clipped surrogate (source)
    max_grad: float = 1.0             # gradient-norm clip (source)
    vf_coef: float = 0.5              # value-loss coefficient (T12, [DC])
    ent_coef: float = 0.0             # no entropy bonus (source)
    init_handoff_p: float = 0.05      # T11
    ckpt_per_pass: int = 8

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.max_grad <= 0 or self.clip <= 0:
            raise ValueError("lr, max_grad and clip must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if self.passes < 1 or self.epochs < 1 or self.ckpt_per_pass < 1:
            raise ValueError("passes, epochs and ckpt_per_pass must be at least 1")
        if self.vf_coef < 0 or self.ent_coef < 0:
            raise ValueError("vf_coef and ent_coef must be non-negative")
        if not 0.0 < self.init_handoff_p < 1.0:
            raise ValueError("init_handoff_p must lie in (0, 1)")


class PPONet(nn.Module):
    """Masked SetNet trunk -> GRU cell (128 -> 64) -> policy (64 -> 2), value (64 -> 64 -> 1)."""

    def __init__(self, init_handoff_p: float = 0.05, net_cfg: NetConfig | None = None):
        super().__init__()
        if not 0.0 < init_handoff_p < 1.0:
            raise ValueError("init_handoff_p must lie in (0, 1)")
        self.trunk: SetNetMasked = masked_net(net_cfg)
        self.gru = nn.GRUCell(POOL_DIM, GRU_HIDDEN)
        self.policy = nn.Linear(GRU_HIDDEN, 2)
        self.value = nn.Sequential(nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.ReLU(),
                                   nn.Linear(GRU_HIDDEN, 1))
        with torch.no_grad():
            nn.init.zeros_(self.policy.weight)
            logit = math.log(init_handoff_p / (1.0 - init_handoff_p))
            self.policy.bias.copy_(torch.tensor([0.0, logit], dtype=torch.float32))

    def encode(self, glob: torch.Tensor, tok: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """(n, 128) pooled features of the masked trunk."""
        return self.trunk.pooled(glob, tok, pad)

    def step(self, pooled: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.gru(pooled, h)

    def heads(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy logits (n, 2) for (continue, handoff) and values (n,)."""
        return self.policy(h), self.value(h).squeeze(-1)


# ---------------------------------------------------------------- pure helpers

def allowed_mask(ks: np.ndarray, m: int) -> np.ndarray:
    """(n, 2) bool: continue always allowed; handoff allowed iff k < m."""
    ks = np.asarray(ks)
    mask = np.ones((ks.size, 2), dtype=bool)
    mask[:, HANDOFF] = ks < m
    return mask


def mask_logits(logits: torch.Tensor, ks, m: int) -> torch.Tensor:
    """Forbid handoff at stage m by setting its logit to -inf."""
    ks_t = torch.as_tensor(np.asarray(ks), dtype=torch.long, device=logits.device)
    neg_inf = torch.full_like(logits[:, HANDOFF], float("-inf"))
    hand = torch.where(ks_t < m, logits[:, HANDOFF], neg_inf)
    return torch.stack([logits[:, CONTINUE], hand], dim=1)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """G_t = r_t + gamma G_{t+1} with G_{T+1} = 0."""
    r = np.asarray(rewards, dtype=float)
    out = np.zeros(r.size)
    acc = 0.0
    for t in range(r.size - 1, -1, -1):
        acc = r[t] + gamma * acc
        out[t] = acc
    return out


def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                  clip: float) -> torch.Tensor:
    """Negative mean of min(r A, clip(r, 1-eps, 1+eps) A), r = exp(logp_new - logp_old)."""
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    return -torch.min(unclipped, clipped).mean()


def step_reward(spec: RouteSpec, W_row: np.ndarray, k: int, action: int) -> tuple[float, bool]:
    """(reward in E0 units, episode ends) for action at decision epoch k.
    W_row is the cumulative load of the row (W_row[j-1] = W_j)."""
    if action == HANDOFF:
        return -float(spec.H[k - 1]) / spec.E0, True
    if W_row[k] > spec.B:                       # stop k+1 breaches: charged E[k]
        return -float(spec.E[k]) / spec.E0, True
    if k + 1 == spec.m:                         # completion
        return 0.0, True
    return 0.0, False


def _seed(seed_key: str, stream: str) -> int:
    return zlib.crc32(f"{seed_key}|{stream}".encode("utf-8"))


def _state_sha(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _cpu_state(net: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def n_trainable(net: nn.Module) -> int:
    """Parameters that take part in PPO (the trunk's unused classification head excluded)."""
    return int(sum(p.numel() for name, p in net.named_parameters()
                   if not name.startswith("trunk.head_")))


# ---------------------------------------------------------------- deployment

class PolicyPredictor:
    """Surrogate H[k-1] + E0 (pi - 1/2) with pi the greedy handoff probability. The GRU
    state of the rows of the last call is kept; a call at k+1 on a subset of those rows
    on the same cache reuses it, any other call recomputes from stage 1."""

    def __init__(self, net: PPONet, E0: float):
        self.net = net
        self.E0 = float(E0)
        self._cache: RouteCache | None = None
        self._k = 0
        self._rows = np.zeros(0, dtype=np.int64)
        self._h: torch.Tensor | None = None

    def _step_stage(self, cache: RouteCache, k: int, idx: np.ndarray,
                    h_prev: torch.Tensor) -> torch.Tensor:
        g, t = stage_batch(cache, k, idx, "full")
        pad = torch.zeros((idx.size, cache.m - k), dtype=torch.bool)
        pooled = self.net.encode(torch.from_numpy(g), torch.from_numpy(t), pad)
        return self.net.step(pooled, h_prev)

    def _reusable(self, cache: RouteCache, k: int, idx: np.ndarray) -> bool:
        if self._cache is not cache or k != self._k + 1 or self._rows.size == 0:
            return False
        pos = np.searchsorted(self._rows, idx)
        if np.any(pos >= self._rows.size):
            return False
        return bool(np.array_equal(self._rows[pos], idx))

    def _hidden(self, cache: RouteCache, k: int, idx: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            if self._reusable(cache, k, idx):
                pos = torch.as_tensor(np.searchsorted(self._rows, idx))
                h_prev = self._h[pos]
            else:
                h_prev = torch.zeros(idx.size, GRU_HIDDEN)
                for j in range(1, k):
                    h_prev = self._step_stage(cache, j, idx, h_prev)
            h = self._step_stage(cache, k, idx, h_prev)
        order = np.argsort(idx, kind="stable")
        self._cache, self._k = cache, k
        self._rows = idx[order]
        self._h = h[torch.as_tensor(order)]
        return h

    def handoff_prob(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros(0)
        with torch.no_grad():
            h = self._hidden(cache, k, idx)
            logits, _ = self.net.heads(h)
            logits = mask_logits(logits, np.full(idx.size, k), cache.m)
            return torch.softmax(logits, dim=1)[:, HANDOFF].numpy().astype(np.float64)

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        pi = self.handoff_prob(cache, k, idx)
        return cache.spec.H[k - 1] + self.E0 * (pi - 0.5)


def _policy_cost(net: nn.Module, val_cache: RouteCache, spec: RouteSpec, E0: float) -> float:
    pred = PolicyPredictor(net, E0)
    models = {k: [pred] for k in range(1, spec.m)}
    costs, _ = simulate_policy(val_cache, spec.B, spec.H, spec.E, models)
    return float(costs.mean())


# ---------------------------------------------------------------- episodes

@dataclass(frozen=True)
class Episode:
    pos: int
    ks: np.ndarray           # decision epochs visited (consecutive from 1)
    actions: np.ndarray
    logp_old: np.ndarray
    values_old: np.ndarray
    rewards: np.ndarray      # E0 units, one per decision


def _stage_pooled(net: PPONet, table: PaddedTables, pos: int, m: int) -> torch.Tensor:
    """Pooled features of every stage 1..m-1 of one row (no action dependence)."""
    stages = np.arange(1, m)
    glob, tok, pad = table.batch(np.full(stages.size, pos), stages)
    return net.encode(glob, tok, pad)


def rollout_episode(net: PPONet, table: PaddedTables, pos: int, W_row: np.ndarray,
                    spec: RouteSpec, rng: np.random.Generator) -> Episode:
    """Sample one episode with the current policy. The GRU is causal, so the pooled
    features of all stages can be computed up front without leaking later decisions."""
    m = spec.m
    ks, acts, logps, vals, rews = [], [], [], [], []
    with torch.no_grad():
        pooled = _stage_pooled(net, table, pos, m)
        h = torch.zeros(1, GRU_HIDDEN)
        for k in range(1, m):
            if W_row[k - 1] > spec.B:           # emergency before the decision
                break
            h = net.step(pooled[k - 1:k], h)
            logits, v = net.heads(h)
            logits = mask_logits(logits, np.array([k]), m)
            probs = torch.softmax(logits, dim=1)[0].numpy().astype(np.float64)
            action = HANDOFF if rng.random() < probs[HANDOFF] else CONTINUE
            reward, done = step_reward(spec, W_row, k, action)
            ks.append(k)
            acts.append(action)
            logps.append(math.log(probs[action]))
            vals.append(float(v.item()))
            rews.append(reward)
            if done:
                break
    return Episode(pos=int(pos), ks=np.asarray(ks, dtype=np.int64),
                   actions=np.asarray(acts, dtype=np.int64),
                   logp_old=np.asarray(logps, dtype=np.float64),
                   values_old=np.asarray(vals, dtype=np.float64),
                   rewards=np.asarray(rews, dtype=np.float64))


def ppo_update(net: PPONet, opt: torch.optim.Optimizer, table: PaddedTables, ep: Episode,
               m: int, cfg: PPOConfig) -> None:
    """``cfg.epochs`` gradient steps on one episode; ratios against the rollout's log-probs,
    advantages fixed at rollout time."""
    ks_t = torch.as_tensor(ep.ks, dtype=torch.long)
    acts_t = torch.as_tensor(ep.actions, dtype=torch.long)
    logp_old = torch.as_tensor(ep.logp_old, dtype=torch.float32)
    returns = discounted_returns(ep.rewards, cfg.gamma)
    adv = torch.as_tensor(returns - ep.values_old, dtype=torch.float32)
    ret_t = torch.as_tensor(returns, dtype=torch.float32)
    glob, tok, pad = table.batch(np.full(ep.ks.size, ep.pos), ep.ks)
    for _ in range(cfg.epochs):
        pooled = net.encode(glob, tok, pad)
        h = torch.zeros(1, GRU_HIDDEN)
        states = []
        for t in range(pooled.shape[0]):
            h = net.step(pooled[t:t + 1], h)
            states.append(h)
        logits, values = net.heads(torch.cat(states, dim=0))
        logp_all = torch.log_softmax(mask_logits(logits, ep.ks, m), dim=1)
        logp_new = logp_all.gather(1, acts_t.unsqueeze(1)).squeeze(1)
        loss = ppo_clip_loss(logp_new, logp_old, adv, cfg.clip)
        loss = loss + cfg.vf_coef * nn.functional.mse_loss(values, ret_t)
        if cfg.ent_coef > 0:
            entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()
            loss = loss - cfg.ent_coef * entropy
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad)
        opt.step()


class _Run:
    """Checkpoints, in-job selection (lowest validation cost, ties earlier) and the
    per-pass selection (cumulative: checkpoints up to that pass)."""

    def __init__(self, keep_states: bool):
        self.keep_states = keep_states
        self.checkpoints: list[dict] = []
        self.states: list[dict] = []
        self.best_idx = -1
        self.best_cost = float("inf")
        self.best_state: dict | None = None
        self.val_seconds = 0.0

    def record(self, net, val_cache, spec, E0, pass_idx, pass_end, n_updates) -> None:
        t0 = time.perf_counter()
        cost = _policy_cost(net, val_cache, spec, E0)
        state = _cpu_state(net)
        self.val_seconds += time.perf_counter() - t0
        idx = len(self.checkpoints)
        self.checkpoints.append(dict(kind="ckpt", index=idx, update=int(n_updates),
                                     **{"pass": int(pass_idx)}, pass_end=bool(pass_end),
                                     val_cost=cost, sha=_state_sha(state)))
        if self.keep_states:
            self.states.append(state)
        if cost < self.best_cost:
            self.best_cost, self.best_idx, self.best_state = cost, idx, state


def train_ppo_record(spec: RouteSpec, paths: Paths, N: int, rep: int, cfg: PPOConfig,
                     keep_states: bool = False) -> dict:
    """Train one PPO on the fitting rows of a route (section 3.6). Returns the run record;
    ``fit_ppo`` is the public wrapper."""
    t_start = time.perf_counter()
    if paths.p.shape[0] != N:
        raise ValueError(f"expected {N} training paths, got {paths.p.shape[0]}")
    if spec.m < 2:
        raise ValueError("PPO needs at least two stops (one decision epoch)")
    m, E0 = spec.m, spec.E0
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    folds, val = split_80_20(N)
    fit_rows = np.sort(np.concatenate(folds))
    stats = RouteStats.fit(Paths(p=paths.p[fit_rows], d=paths.d[fit_rows]))
    cache = RouteCache(spec, paths, stats)
    val_cache = RouteCache(spec, Paths(p=paths.p[val], d=paths.d[val]), stats)
    W_fit = cache.W[fit_rows]
    seed_key = f"{spec.inst}|{spec.route_id}|{N}|{rep}|ppo|main"

    t0 = time.perf_counter()
    table = PaddedTables(cache, fit_rows)
    feature_seconds = time.perf_counter() - t0

    torch.manual_seed(_seed(seed_key, "init"))
    net = PPONet(cfg.init_handoff_p)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(_seed(seed_key, "sample"))

    updates_per_pass = int(np.sum(W_fit[:, 0] <= spec.B))   # episodes with a decision
    ckpt_every = max(1, updates_per_pass // cfg.ckpt_per_pass)

    run = _Run(keep_states)
    run.record(net, val_cache, spec, E0, 0, False, 0)
    buf_visits = np.zeros(m - 1, dtype=np.int64)
    env_steps = 0
    n_updates = 0
    pass_best: list[dict] = []
    pass_states: list[dict] = []
    train_seconds = 0.0
    for p in range(cfg.passes):
        t_pass = time.perf_counter()
        order = np.random.default_rng(_seed(seed_key, f"order{p}")).permutation(fit_rows.size)
        for pos in order:
            ep = rollout_episode(net, table, int(pos), W_fit[pos], spec, rng)
            np.add.at(buf_visits, ep.ks - 1, 1)
            env_steps += int(ep.ks.size)
            if ep.ks.size == 0:
                continue
            net.train()
            ppo_update(net, opt, table, ep, m, cfg)
            n_updates += 1
            if n_updates % ckpt_every == 0:
                run.record(net, val_cache, spec, E0, p + 1, False, n_updates)
        run.record(net, val_cache, spec, E0, p + 1, True, n_updates)
        pass_best.append(dict(**{"pass": p + 1}, index=run.best_idx,
                              val_cost=run.best_cost))
        if keep_states:
            pass_states.append(copy.deepcopy(run.best_state))
        train_seconds += time.perf_counter() - t_pass
    return dict(
        stats=stats, checkpoints=run.checkpoints, chosen_index=run.best_idx,
        best_state=run.best_state, pass_best=pass_best,
        pass_states=pass_states if keep_states else None,
        visits_per_stage=[int(x) for x in buf_visits], env_steps=int(env_steps),
        updates=int(n_updates), gradient_steps=int(n_updates * cfg.epochs),
        ckpt_every=int(ckpt_every), n_params=n_trainable(net), n_val=int(val.size),
        device="cpu", feature_seconds=feature_seconds,
        train_seconds=train_seconds - run.val_seconds, val_seconds=run.val_seconds,
        total_seconds=time.perf_counter() - t_start,
    )


def fit_ppo(spec: RouteSpec, paths: Paths, N: int, rep: int,
            cfg: PPOConfig | None = None) -> tuple[dict, RouteStats, list, dict]:
    """PPO for one route: returns ({k: [PolicyPredictor]}, RouteStats, diag, budget). The
    kept network (in-job selection over all checkpoints of the run) is shared by all
    stages' predictors."""
    cfg = PPOConfig() if cfg is None else cfg
    rec = train_ppo_record(spec, paths, N, rep, cfg, keep_states=False)
    net = PPONet(cfg.init_handoff_p)
    net.load_state_dict(rec["best_state"])
    net.eval()
    pred = PolicyPredictor(net, spec.E0)
    models = {k: [pred] for k in range(1, spec.m)}
    diag = list(rec["checkpoints"])
    diag.append(dict(kind="summary", updates=rec["updates"],
                     gradient_steps=rec["gradient_steps"], ckpt_every=rec["ckpt_every"],
                     chosen_index=rec["chosen_index"], pass_best=rec["pass_best"]))
    n_ckpt = len(rec["checkpoints"])
    budget = dict(
        unique_paths=int(N), path_stop_reads=rec["env_steps"],
        val_reads=int(n_ckpt * rec["n_val"] * spec.m), updates=rec["updates"],
        gradient_steps=rec["gradient_steps"], n_params=rec["n_params"],
        feature_seconds=rec["feature_seconds"], train_seconds=rec["train_seconds"],
        val_seconds=rec["val_seconds"], device=rec["device"],
        visits_per_stage=rec["visits_per_stage"],
    )
    return models, rec["stats"], diag, budget
