"""DQN after Dastpak, Errico & Jabali (EXPERIMENT_B.md 3.5; 10 M18).

One masked SetNet (``SetNetMasked``, padded.py) on the FULL state learns Q(s, continue)
in units of E0. Q(s, handoff) is the known route cost H[k-1], so the greedy rule is to hand
off at k iff Q_cont > H[k-1].

Training environment: the fitting rows only. The behaviour policy is always "continue"
(``always_continue``): every path runs until breach or completion, so the number of
stored transitions at stage k equals the number of alive fitting pairs per pass
(``stage_counts``). Transitions carry (position, k, c, done) and the next state is
(position, k+1), read from ``PaddedTables``. Double-Q target: y = c + (1 - done) V',
V' = H[k] if Q_online(s') > H[k] else Q_target(s'). Huber loss on the E0 scale (delta 1),
Adam, gradient-norm clip 10. One update per ``per_update`` stored transitions once the
buffer holds ``learn_start`` transitions; a hard target copy every ``target_every_tr``
stored transitions.

Checkpoints: update 0, every U_pass // ckpt_per_pass updates (U_pass = pass transitions //
per_update, independent of the total number of passes), and at every pass end. The kept
checkpoint has the lowest greedy-policy cost on the validation rows (ties to the earlier).
Evaluation models are copies on the CPU with features from ``stage_batch`` (no table).

Seeds: crc32 of "inst|route|N|rep|dqn|main|<stream>" (stream: init, order<p>, sample).
Training uses the GPU when ``device`` resolves to CUDA; CUDA runs are not required to be
bitwise reproducible (section 9.3, Q2).
"""
from __future__ import annotations

import copy
import hashlib
import os
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..expa.backward import split_80_20
from ..expa.features import RouteCache, RouteStats, stage_batch
from ..expa.paths import Paths, RouteSpec
from ..expa.simulate import simulate_policy
from .padded import PaddedTables, SetNetMasked, masked_net

# cuBLAS determinism (3.0); must be set before CUDA initialises, so it is set here.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PRED_CHUNK = 8192
_RING = 4


@dataclass(frozen=True)
class DQNConfig:
    lr: float = 1e-4                  # grid {1e-4, 3e-4} (3.5)
    passes: int = 3                   # P_max = 3 for tuning (6.3)
    n_env: int = 64                   # lockstep paths of the continue-forced behaviour
    buffer: int = 50_000
    batch: int = 256
    per_update: int = 32              # one update per 32 stored transitions (T10)
    learn_start: int = 1_000
    target_every_tr: int = 8_000      # grid {8,000, 32,000} stored transitions
    huber: float = 1.0                # on the E0 scale
    clip: float = 10.0
    ckpt_per_pass: int = 8
    device: str = "auto"              # "auto": CUDA when available, else CPU


def resolve_device(name: str) -> torch.device:
    """'auto' picks CUDA when available; an explicit 'cuda' without a device is an error."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return dev


def _no_nan_fill() -> None:
    """Deterministic mode would fill every ``torch.empty`` with NaN (one kernel launch each);
    only uninitialised memory is affected, never a computed value."""
    torch.utils.deterministic.fill_uninitialized_memory = False


def make_optimizer(net: nn.Module, lr: float, device: torch.device) -> torch.optim.Adam:
    """Adam; the fused capturable CUDA kernel on the GPU (one launch for all parameters),
    the stock single-tensor implementation on the CPU (bitwise equal to the reference)."""
    cuda = device.type == "cuda"
    return torch.optim.Adam(net.parameters(), lr=lr, fused=cuda, capturable=cuda)


def always_continue(pos: np.ndarray, k: int) -> np.ndarray:
    """Behaviour policy of 3.5: continue at every decision (every stage gets its data)."""
    return np.ones(np.asarray(pos).shape[0], dtype=bool)


def stage_counts(W: np.ndarray, B: float, m: int) -> np.ndarray:
    """Alive fitting pairs per stage, length m with index k (index 0 unused = 0): rows
    with no breach at stops 1..k, i.e. the decisions a continue-forced pass visits."""
    breach = np.asarray(W)[:, : m - 1] > B
    through = np.maximum.accumulate(breach, axis=1)
    counts = np.zeros(m, dtype=np.int64)
    for k in range(1, m):
        counts[k] = int((~through[:, k - 1]).sum())
    return counts


def _batch_transitions(W: np.ndarray, rows: np.ndarray, B: float, E: np.ndarray, m: int,
                       cont_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """One lockstep batch of paths (simulate_policy order: emergency check at each stop,
    then the behaviour decision). Returns (pos, k, c, done, stop visits)."""
    alive = np.ones(rows.size, dtype=bool)
    out_pos, out_k, out_c, out_d = [], [], [], []
    visits = 0
    for k in range(1, m):
        act = np.flatnonzero(alive)
        if act.size == 0:
            break
        visits += int(act.size)
        emergency = W[rows[act], k - 1] > B
        alive[act[emergency]] = False
        act = act[~emergency]
        if act.size == 0:
            continue
        cont = np.asarray(cont_fn(rows[act], k), dtype=bool)
        alive[act[~cont]] = False
        act = act[cont]
        if act.size == 0:
            continue
        r = rows[act]
        breach_next = W[r, k] > B
        done = breach_next | (k + 1 == m)
        cost = np.where(breach_next, E[k], 0.0)
        out_pos.append(r)
        out_k.append(np.full(r.size, k, dtype=np.int64))
        out_c.append(cost.astype(np.float64))
        out_d.append(done)
        alive[act[done]] = False
    if not out_pos:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), np.zeros(0, dtype=bool), visits
    return (np.concatenate(out_pos), np.concatenate(out_k), np.concatenate(out_c),
            np.concatenate(out_d), visits)


def collect_transitions(W: np.ndarray, order: np.ndarray, B: float, E: np.ndarray, m: int,
                        n_env: int, cont_fn=always_continue):
    """All transitions of one pass over ``order`` (positions into W), lockstep batches of
    ``n_env``. Returns (pos, k, c, done, stop visits summed over batches)."""
    order = np.asarray(order, dtype=np.int64)
    parts = []
    visits = 0
    for s in range(0, order.size, n_env):
        p, kk, cc, dd, v = _batch_transitions(W, order[s:s + n_env], B, E, m, cont_fn)
        parts.append((p, kk, cc, dd))
        visits += v
    if not parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), np.zeros(0, dtype=bool), 0
    return (np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts]),
            np.concatenate([x[2] for x in parts]), np.concatenate([x[3] for x in parts]),
            visits)


class ReplayBuffer:
    """FIFO ring buffer of (position, k, c, done)."""

    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self.pos = np.zeros(self.cap, dtype=np.int64)
        self.k = np.zeros(self.cap, dtype=np.int64)
        self.c = np.zeros(self.cap, dtype=np.float64)
        self.done = np.zeros(self.cap, dtype=bool)
        self.head = 0
        self.size = 0

    def add(self, pos, ks, cs, dones) -> None:
        n = len(pos)
        if n > self.cap:
            pos, ks, cs, dones = pos[-self.cap:], ks[-self.cap:], cs[-self.cap:], dones[-self.cap:]
            n = self.cap
        if n == 0:
            return
        idx = (self.head + np.arange(n)) % self.cap
        self.pos[idx] = pos
        self.k[idx] = ks
        self.c[idx] = cs
        self.done[idx] = dones
        self.head = (self.head + n) % self.cap
        self.size = min(self.cap, self.size + n)

    def sample(self, rng: np.random.Generator, n: int):
        i = rng.integers(0, self.size, size=n)
        return self.pos[i], self.k[i], self.c[i], self.done[i]


def double_q_target(c: torch.Tensor, done: torch.Tensor, q_on: torch.Tensor,
                    q_tg: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
    """y = c + (1 - done) V', V' = H (known handoff cost) if Q_online > H else Q_target.
    Works in any consistent unit (cost or E0-scaled)."""
    v_next = torch.where(q_on > h_next, h_next, q_tg)
    return c + (1.0 - done) * v_next


class QPredictor:
    """Q_cont in cost units from a (shared) network; features from ``stage_batch``."""

    def __init__(self, net: SetNetMasked, E0: float):
        self.net = net
        self.E0 = float(E0)

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros(idx.size, dtype=np.float64)
        dev = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            for s in range(0, idx.size, PRED_CHUNK):
                sl = idx[s:s + PRED_CHUNK]
                g_np, t_np = stage_batch(cache, k, sl, "full")
                g = torch.from_numpy(g_np).to(dev)
                t = torch.from_numpy(t_np).to(dev)
                pad = torch.zeros(sl.size, cache.m - k, dtype=torch.bool, device=dev)
                u = self.net(g, t, pad).cpu().numpy().astype(np.float64)
                out[s:s + sl.size] = self.E0 * u
        return out


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


def _policy_cost(net: nn.Module, val_cache: RouteCache, spec: RouteSpec, E0: float) -> float:
    pred = QPredictor(net, E0)
    models = {k: [pred] for k in range(1, spec.m)}
    costs, _ = simulate_policy(val_cache, spec.B, spec.H, spec.E, models)
    return float(costs.mean())


def make_step(net, target, opt, table, cfg, H_t, E0, m):
    """(step(buf, rng), name): the CUDA-graph update on a GPU, the eager one otherwise or
    when capture fails (the eager path computes the same update)."""
    if table.device.type == "cuda":
        try:
            gu = GraphUpdater(net, target, opt, table, cfg, H_t, E0, m)
            return gu.update, "cuda_graph"
        except RuntimeError:
            pass
    return (lambda buf, rng: _update(net, target, opt, table, buf, rng, cfg, H_t, E0, m),
            "eager")


def _pad_table(table: PaddedTables) -> torch.Tensor:
    """(L, L) bool: row k-1 marks tokens t >= m-k as padding. Cached on the table."""
    pt = getattr(table, "_dqn_pad_tab", None)
    if pt is None:
        stage_real = table.m - torch.arange(1, table.L + 1, device=table.device)
        slots = torch.arange(table.L, device=table.device)
        pt = slots.unsqueeze(0) >= stage_real.unsqueeze(1)
        table._dqn_pad_tab = pt
    return pt


def _pack_batch(buf: ReplayBuffer, rng: np.random.Generator, n: int,
                out: np.ndarray | None = None) -> np.ndarray:
    """Sample ``n`` transitions (same draw as ``ReplayBuffer.sample``) into one (4, n)
    float64 array [pos, k, c, done] so that a single host-to-device copy carries a batch
    (integers up to 2^53 are exact)."""
    i = rng.integers(0, buf.size, size=n)
    out = np.empty((4, n), dtype=np.float64) if out is None else out
    out[0], out[1], out[2], out[3] = buf.pos[i], buf.k[i], buf.c[i], buf.done[i]
    return out


def _update_core(net, target, opt, table: PaddedTables, pk: torch.Tensor,
                 cfg: DQNConfig, H_t: torch.Tensor, E0: float, m: int) -> torch.Tensor:
    """The update on a device-resident packed batch ``pk``; no host synchronisation and
    no allocation that depends on the data, so it can be captured in a CUDA graph."""
    n = cfg.batch
    pos = pk[0].long()
    ks = pk[1].long()
    c_t = pk[2].float()
    d_t = pk[3].float()
    k_next = torch.clamp(ks + 1, max=m - 1)
    si = torch.cat([ks, k_next]) - 1
    pos2 = torch.cat([pos, pos])
    glob2 = table.glob[si, pos2]
    tok2 = table.tok[si, pos2]
    pad2 = _pad_table(table)[si]
    glob, gn = glob2[:n], glob2[n:]
    tok, tn = tok2[:n], tok2[n:]
    pad, pn = pad2[:n], pad2[n:]
    q_now = net(glob, tok, pad)
    with torch.no_grad():
        q_on = E0 * net(gn, tn, pn)
        q_tg = E0 * target(gn, tn, pn)
        y_u = double_q_target(c_t, d_t, q_on, q_tg, H_t[ks]) / E0
    loss = nn.functional.huber_loss(q_now, y_u, delta=cfg.huber)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
    opt.step()
    return loss.detach()


def _update(net, target, opt, table: PaddedTables, buf: ReplayBuffer,
            rng: np.random.Generator, cfg: DQNConfig, H_t: torch.Tensor, E0: float,
            m: int) -> torch.Tensor:
    """One double-Q update (eager). Same algorithm and RNG use as ``_update_reference`` in
    tests/test_expb_dqn.py; the detached loss is returned as a device tensor (no sync)."""
    net.train()
    pk = torch.from_numpy(_pack_batch(buf, rng, cfg.batch)).to(table.device)
    return _update_core(net, target, opt, table, pk, cfg, H_t, E0, m)


class GraphUpdater:
    """The same update replayed from a CUDA graph (one launch per update instead of
    about 600). Capture runs warm-up steps on a dummy batch, then restores the parameters
    and zeroes the optimiser state in place, so training starts from the untouched
    initial state; ``opt`` must be created with ``capturable=True``."""

    def __init__(self, net, target, opt, table, cfg, H_t, E0, m):
        self.n, self.dev = cfg.batch, table.device
        # ring of pinned host buffers: a buffer is rewritten only after the copy that read
        # it (4 updates earlier) has completed
        self.hosts = [torch.zeros((4, self.n), dtype=torch.float64).pin_memory()
                      for _ in range(_RING)]
        self.copied = [torch.cuda.Event() for _ in range(_RING)]
        self.calls = 0
        self.pk = torch.zeros((4, self.n), dtype=torch.float64, device=self.dev)
        self.pk[0], self.pk[1] = 0.0, 1.0                  # a valid (row 0, stage 1) batch
        net.train()
        saved = [p.detach().clone() for p in net.parameters()]
        step = lambda: _update_core(net, target, opt, table, self.pk, cfg, H_t, E0, m)
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            opt.zero_grad(set_to_none=True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loss = step()
        finally:                                  # also on failure: untouched initial state
            with torch.no_grad():
                for p, q in zip(net.parameters(), saved):
                    p.copy_(q)
                for st in opt.state.values():
                    for v in st.values():
                        v.zero_()
        self.net = net

    def update(self, buf: ReplayBuffer, rng: np.random.Generator) -> torch.Tensor:
        self.net.train()
        j = self.calls % _RING
        self.calls += 1
        if self.calls > _RING:
            self.copied[j].synchronize()
        _pack_batch(buf, rng, self.n, out=self.hosts[j].numpy())
        self.pk.copy_(self.hosts[j], non_blocking=True)
        self.copied[j].record()
        self.graph.replay()
        return self.loss


class _Run:
    """Mutable bookkeeping of one training run (checkpoints, counters, kept state)."""

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


def train_dqn_record(spec: RouteSpec, paths: Paths, N: int, rep: int, cfg: DQNConfig,
                     keep_states: bool = False) -> dict:
    """Train one DQN on the fitting rows of a route (section 3.5). Returns the run record;
    ``fit_dqn`` is the public wrapper."""
    t_start = time.perf_counter()
    if paths.p.shape[0] != N:
        raise ValueError(f"expected {N} training paths, got {paths.p.shape[0]}")
    if cfg.per_update < 1 or cfg.batch < 1 or cfg.passes < 1:
        raise ValueError("per_update, batch and passes must be positive")
    m, E0 = spec.m, spec.E0
    device = resolve_device(cfg.device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _no_nan_fill()

    folds, val = split_80_20(N)
    fit_rows = np.sort(np.concatenate(folds))
    stats = RouteStats.fit(Paths(p=paths.p[fit_rows], d=paths.d[fit_rows]))
    cache = RouteCache(spec, paths, stats)
    val_cache = RouteCache(spec, Paths(p=paths.p[val], d=paths.d[val]), stats)
    W_fit = cache.W[fit_rows]
    seed_key = f"{spec.inst}|{spec.route_id}|{N}|{rep}|dqn|main"

    t0 = time.perf_counter()
    table = PaddedTables(cache, fit_rows, device=device)
    feature_seconds = time.perf_counter() - t0

    torch.manual_seed(_seed(seed_key, "init"))
    net = masked_net().to(device)
    target = copy.deepcopy(net)
    target.eval()
    opt = make_optimizer(net, cfg.lr, device)
    H_t = torch.as_tensor(spec.H, dtype=torch.float32, device=device)
    rng = np.random.default_rng(_seed(seed_key, "sample"))

    step_fn, update_path = make_step(net, target, opt, table, cfg, H_t, E0, m)

    counts = stage_counts(W_fit, spec.B, m)
    updates_per_pass = int(counts.sum()) // cfg.per_update
    ckpt_every = max(1, updates_per_pass // cfg.ckpt_per_pass)

    run = _Run(keep_states)
    run.record(net, val_cache, spec, E0, 0, False, 0)
    buf = ReplayBuffer(cfg.buffer)
    n_total = 0
    since = 0
    n_updates = 0
    next_target = cfg.target_every_tr
    target_copies: list[int] = []
    env_steps = 0
    per_stage = np.zeros(m, dtype=np.int64)
    visited = np.zeros(fit_rows.size, dtype=bool)
    train_seconds = 0.0
    pass_best: list[dict] = []
    pass_states: list[dict] = []
    for p in range(cfg.passes):
        t_pass = time.perf_counter()
        order = np.random.default_rng(_seed(seed_key, f"order{p}")).permutation(fit_rows.size)
        for s in range(0, order.size, cfg.n_env):
            pos, ks, cs, dn, visits = _batch_transitions(
                W_fit, order[s:s + cfg.n_env], spec.B, spec.E, m, always_continue)
            env_steps += visits
            per_stage += np.bincount(ks, minlength=m)[:m]
            visited[pos] = True
            buf.add(pos, ks, cs, dn)
            n_total += pos.size
            since += pos.size
            if buf.size < cfg.learn_start:
                since = 0
            while since >= cfg.per_update:
                step_fn(buf, rng)
                since -= cfg.per_update
                n_updates += 1
                if n_updates % ckpt_every == 0:
                    run.record(net, val_cache, spec, E0, p + 1, False, n_updates)
            while n_total >= next_target:
                target.load_state_dict(net.state_dict())
                target_copies.append(next_target)
                next_target += cfg.target_every_tr
        run.record(net, val_cache, spec, E0, p + 1, True, n_updates)
        # per-pass accessor (EXPERIMENT_B 6.3): selection restricted to checkpoints up to
        # this pass; read-only, the update path above is unchanged
        pass_best.append(dict(**{"pass": p + 1}, index=run.best_idx, val_cost=run.best_cost))
        if keep_states:
            pass_states.append(copy.deepcopy(run.best_state))
        train_seconds += time.perf_counter() - t_pass
    chosen = run.best_idx
    return dict(
        stats=stats, checkpoints=run.checkpoints, chosen_index=chosen,
        best_state=run.best_state,
        last_state=run.states[-1] if keep_states else None,
        pass_best=pass_best, pass_states=pass_states if keep_states else None,
        transitions_per_stage=[int(x) for x in per_stage], env_steps=int(env_steps),
        env_rows=fit_rows[visited], updates=int(n_updates), target_copies=target_copies,
        ckpt_every=int(ckpt_every), n_params=int(sum(q.numel() for q in net.parameters())),
        n_val=int(val.size), device=device.type, update_path=update_path, feature_seconds=feature_seconds,
        train_seconds=train_seconds - run.val_seconds, val_seconds=run.val_seconds,
        total_seconds=time.perf_counter() - t_start,
    )


def fit_dqn(spec: RouteSpec, paths: Paths, N: int, rep: int, cfg: DQNConfig
            ) -> tuple[dict, RouteStats, list, dict]:
    """DQN for one route: returns ({k: [QPredictor]}, RouteStats, diag, budget). The
    kept network is copied to the CPU and shared by all stages' predictors."""
    rec = train_dqn_record(spec, paths, N, rep, cfg, keep_states=False)
    net = masked_net()
    net.load_state_dict(rec["best_state"])
    net.eval()
    pred = QPredictor(net, spec.E0)
    models = {k: [pred] for k in range(1, spec.m)}
    diag = list(rec["checkpoints"])
    diag.append(dict(kind="summary", updates=rec["updates"], ckpt_every=rec["ckpt_every"],
                     chosen_index=rec["chosen_index"], target_copies=rec["target_copies"]))
    n_ckpt = len(rec["checkpoints"])
    budget = dict(
        unique_paths=int(N), path_stop_reads=rec["env_steps"],
        val_reads=int(n_ckpt * rec["n_val"] * spec.m), updates=rec["updates"],
        n_params=rec["n_params"], feature_seconds=rec["feature_seconds"],
        train_seconds=rec["train_seconds"], val_seconds=rec["val_seconds"],
        device=rec["device"], transitions_per_stage=rec["transitions_per_stage"][1:],
        target_copies=len(rec["target_copies"]),
    )
    return models, rec["stats"], diag, budget
