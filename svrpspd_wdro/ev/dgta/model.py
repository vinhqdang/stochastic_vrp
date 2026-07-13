"""model.py — DGTA: Dynamic Graph Temporal Attention model.

Faithful, scaled-down reimplementation of Chen, Imdahl, Lai & Van
Woensel (2025), "The Dynamic Traveling Salesman Problem with
Time-Dependent and Stochastic travel times: A deep reinforcement
learning approach," Transportation Research Part C 172:105022, DOI
10.1016/j.trc.2025.105022 — the DGTA-RL baseline TEMPO is compared
against (paper's own Section 5, Fig. 2; MDP of Section 3; training
Algorithm 1). Reused here as a re-implemented BASELINE for TEMPO's
traffic channel, not as a method of this project.

Architecture (paper's Fig. 2, matching section numbers in comments):
  Embedding layer (Eq. 7)        node coords + time-interval index
                                 -> H^(0), one vector per (node, time
                                 interval) pair.
  Dual attention layer (§5.1,
    Eqs. 8-9), L1 blocks         spatial MHA (Eq. 8, dependencies
                                 between nodes at a fixed time
                                 interval) and temporal MHA (across
                                 time intervals for a fixed node) in
                                 parallel, concatenated, projected,
                                 each with the usual add&norm + FF
                                 + add&norm transformer block.
  Dynamic encoder (§3, "dynamic
    encoder" in Fig. 2), L2
    blocks                       folds in the REALIZED arrival times
                                 A_d and current decision time tau_d
                                 (state components specific to step d)
                                 via further MHA blocks over the
                                 spatial-temporal representation.
  Spatial + temporal pointer
    decoder                      two separate multi-head "pointer"
                                 attentions (query = dynamic-encoder
                                 output at the current node/time,
                                 keys/values = per-node and per-time
                                 representations) whose logits are
                                 combined and masked by the visited-
                                 node vector V_d to give the next-node
                                 distribution (Eq. 6's p_theta).

Scaled down from the paper for a tractable from-scratch reproduction:
hidden dim D=64 (paper's D unspecified exactly, larger in practice),
M=4 attention heads, L1=L2=2 blocks, T=8 time intervals over one
8-hour route-day (vs. the paper's finer discretization) — chosen to
match TEMPO's own diurnal-hours-scale travel-time process rather than
the paper's larger benchmark instances.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mha(q, k, v, n_heads, mask=None):
    """Scaled dot-product multi-head attention. q,k,v: (B, Lq/Lk, D)."""
    B, Lq, D = q.shape
    Lk = k.shape[1]
    Dh = D // n_heads
    qh = q.view(B, Lq, n_heads, Dh).transpose(1, 2)      # (B,H,Lq,Dh)
    kh = k.view(B, Lk, n_heads, Dh).transpose(1, 2)
    vh = v.view(B, Lk, n_heads, Dh).transpose(1, 2)
    scores = qh @ kh.transpose(-1, -2) / math.sqrt(Dh)   # (B,H,Lq,Lk)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ vh                                       # (B,H,Lq,Dh)
    return out.transpose(1, 2).reshape(B, Lq, D)


class MHABlock(nn.Module):
    """One transformer block: MHA(q,k,v) -> linear -> add&norm -> FF
    -> add&norm, exactly the residual pattern in Fig. 2's dual
    attention / dynamic encoder blocks."""

    def __init__(self, dim, n_heads, ff_mult=4):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.ReLU(),
                                nn.Linear(dim * ff_mult, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x_q, x_kv=None):
        x_kv = x_q if x_kv is None else x_kv
        h = _mha(self.q(x_q), self.k(x_kv), self.v(x_kv), self.n_heads)
        x = self.norm1(x_q + self.proj(h))
        x = self.norm2(x + self.ff(x))
        return x


class DGTA(nn.Module):
    """DGTA model: embedding -> dual attention (spatial+temporal) x L1
    -> dynamic encoder x L2 -> spatial+temporal pointer -> masked
    next-node distribution. See module docstring for the section map."""

    def __init__(self, n_time: int, dim: int = 64, n_heads: int = 4,
                L1: int = 2, L2: int = 2):
        super().__init__()
        self.T = n_time
        self.D = dim
        self.embed = nn.Linear(3, dim)          # (x, y, time_idx) -> H0

        self.spatial_blocks = nn.ModuleList(
            MHABlock(dim, n_heads) for _ in range(L1))
        self.temporal_blocks = nn.ModuleList(
            MHABlock(dim, n_heads) for _ in range(L1))
        self.merge = nn.Linear(2 * dim, dim)

        self.dyn_in = nn.Linear(dim + 2, dim)   # + (arrival_time, tau_d)
        self.dyn_blocks = nn.ModuleList(
            MHABlock(dim, n_heads) for _ in range(L2))

        self.temporal_ptr_q = nn.Linear(dim, dim)
        self.temporal_ptr_k = nn.Linear(dim, dim)
        self.spatial_ptr_q = nn.Linear(dim, dim)
        self.spatial_ptr_k = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.out_proj = nn.Linear(2, 1)

    def encode(self, coords):
        """coords: (B, N, 2). Returns per-(node,time) representation
        Hbar: (B, N, T, D), and its time-pooled per-node form (B,N,D)."""
        B, N, _ = coords.shape
        T = self.T
        xy = coords.unsqueeze(2).expand(B, N, T, 2)
        t_idx = torch.arange(T, device=coords.device).float()
        t_idx = t_idx.view(1, 1, T, 1).expand(B, N, T, 1)
        h0 = torch.sigmoid(self.embed(torch.cat([xy, t_idx], dim=-1)))
        h0 = h0.view(B, N * T, self.D)

        # spatial attention: within each time slice, across nodes
        hs = h0.view(B, N, T, self.D)
        sp = hs.permute(0, 2, 1, 3).reshape(B * T, N, self.D)
        for blk in self.spatial_blocks:
            sp = blk(sp)
        sp = sp.view(B, T, N, self.D).permute(0, 2, 1, 3)   # (B,N,T,D)

        # temporal attention: within each node, across time slices
        tp = hs.reshape(B * N, T, self.D)
        for blk in self.temporal_blocks:
            tp = blk(tp)
        tp = tp.view(B, N, T, self.D)

        merged = torch.tanh(self.merge(torch.cat([sp, tp], dim=-1)))
        return merged                                        # (B,N,T,D)

    def step_logits(self, Hbar, cur_node, t_idx, arrival_times, tau_d,
                    visited_mask):
        """One decision step. Hbar: (B,N,T,D). cur_node: (B,) long.
        t_idx: (B,) long (current discretized time). arrival_times:
        (B,N) realized/expected arrival time at each node so far (0 for
        unvisited). tau_d: (B,) current decision-step scalar time.
        visited_mask: (B,N) bool, True = already visited.
        Returns logits (B,N) over the next node."""
        B, N, T, D = Hbar.shape
        node_rep = Hbar[torch.arange(B), :, t_idx.clamp(max=T - 1), :]  # (B,N,D)
        dyn_in = torch.cat([node_rep, arrival_times.unsqueeze(-1),
                            tau_d.view(B, 1, 1).expand(B, N, 1)], dim=-1)
        dyn = torch.tanh(self.dyn_in(dyn_in))                 # (B,N,D)
        for blk in self.dyn_blocks:
            dyn = blk(dyn)

        cur = dyn[torch.arange(B), cur_node].unsqueeze(1)     # (B,1,D)

        tq = self.temporal_ptr_q(cur)
        tk = self.temporal_ptr_k(dyn)
        t_score = (tq @ tk.transpose(-1, -2)) / math.sqrt(D)  # (B,1,N)

        sq = self.spatial_ptr_q(cur)
        sk = self.spatial_ptr_k(dyn)
        s_score = (sq @ sk.transpose(-1, -2)) / math.sqrt(D)

        both = torch.stack([t_score.squeeze(1), s_score.squeeze(1)],
                           dim=-1)                            # (B,N,2)
        logits = self.out_proj(both).squeeze(-1)              # (B,N)
        logits = logits.masked_fill(visited_mask, float("-inf"))
        return logits
