"""Padded FULL-state tables and the masked SetNet (EXPERIMENT_B.md 3.5 and 3.6, 10 M17;
fixes REVIEW_B 9.5 and 9.6).

``SetNetMasked`` is A's ``SetNet`` (same parameter names, so its state dict loads
strictly) with a key-padding mask passed to every encoder layer. A prepended False
column keeps the global token visible; the token mean runs over unpadded tokens only.
Padding uses PyTorch's convention: True marks a padded position that is ignored.

``PaddedTables`` holds, for every row in ``rows`` and every stage k = 1..m-1, the FULL
globals and the remaining-stop tokens of ``stage_batch(cache, k, rows, "full")``, padded
to m-1 tokens. One ``stage_batch`` call per k is made per job, so mixed-stage minibatches
are plain gathers. Memory is about (m-1) * n * (m-1) * F * 4 bytes for the tokens.

The MHA fast path is never switched off at import time; tests that compare against the
reference use a fixture that restores the previous flag.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..expa.features import F, G, RouteCache, stage_batch
from ..expa.model import NetConfig, SetNet


class SetNetMasked(SetNet):
    """A's SetNet with key-padding masks (EXPERIMENT_B.md 3.5)."""

    def _encode(self, glob: Tensor, tok: Tensor, pad: Tensor) -> Tensor:
        """Pooled (n, 2d) features; ``pad`` is (n, T) bool, True = padding."""
        n = glob.shape[0]
        g = self.glob_proj(glob).unsqueeze(1)
        h = torch.cat([g, self.tok_proj(tok)], dim=1)
        keep_pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=pad.device), pad],
                             dim=1)
        for layer in self.encoder:
            h = layer(h, src_key_padding_mask=keep_pad)
        tok_out = h[:, 1:].masked_fill(pad.unsqueeze(-1), 0.0)
        n_real = (~pad).sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        tok_mean = tok_out.sum(dim=1) / n_real
        return torch.cat([h[:, 0], tok_mean], dim=1)

    def pooled(self, glob: Tensor, tok: Tensor, pad: Tensor) -> Tensor:
        """(n, 2d): the global token output and the masked token mean (PPO's GRU input)."""
        return self._encode(glob, tok, pad)

    def forward(self, glob: Tensor, tok: Tensor, pad: Tensor) -> Tensor:
        """glob (n, G), tok (n, T, F), pad (n, T) bool. Returns (n,) in (0, 1)."""
        z = torch.nn.functional.gelu(self.head_hidden(self._encode(glob, tok, pad)))
        return torch.sigmoid(self.head_out(z)).squeeze(-1)


def masked_net(cfg: NetConfig | None = None) -> SetNetMasked:
    return SetNetMasked(cfg or NetConfig())


class PaddedTables:
    """Per-job FULL features of ``rows`` for every stage, padded to m-1 tokens."""

    def __init__(self, cache: RouteCache, rows: np.ndarray,
                 device: str | torch.device = "cpu"):
        rows = np.asarray(rows, dtype=np.int64)
        self.m = int(cache.m)
        self.L = self.m - 1
        self.n = int(rows.size)
        self.device = torch.device(device)
        self.glob = torch.zeros((self.L, self.n, G), dtype=torch.float32, device=self.device)
        self.tok = torch.zeros((self.L, self.n, self.L, F), dtype=torch.float32,
                               device=self.device)
        for k in range(1, self.m):
            g_np, t_np = stage_batch(cache, k, rows, "full")
            self.glob[k - 1] = torch.from_numpy(g_np).to(self.device)
            self.tok[k - 1, :, : self.m - k, :] = torch.from_numpy(t_np).to(self.device)

    def batch(self, pos: np.ndarray, ks: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        """Gather (glob (B, G), tok (B, m-1, F), pad (B, m-1) bool) for positions ``pos``
        at stages ``ks``. Padding marks tokens t >= m-k (stage k has m-k real tokens)."""
        pos_t = torch.as_tensor(np.asarray(pos, dtype=np.int64), device=self.device)
        ks_t = torch.as_tensor(np.asarray(ks, dtype=np.int64), device=self.device)
        if ks_t.numel() and (int(ks_t.min()) < 1 or int(ks_t.max()) > self.L):
            raise ValueError(f"stages must lie in [1, m-1] = [1, {self.L}]")
        si = ks_t - 1
        glob = self.glob[si, pos_t]
        tok = self.tok[si, pos_t]
        n_real = self.m - ks_t
        slots = torch.arange(self.L, device=self.device).unsqueeze(0)
        pad = slots >= n_real.unsqueeze(1)
        return glob, tok, pad
