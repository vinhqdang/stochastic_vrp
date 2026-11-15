"""Neural continuation networks for EXPERIMENT A (module M4).

EXPERIMENT_A.md sections 3.4 and 4.1.

* ``SetNet`` (arms FULL and SCALAR): a global token and one token per
  remaining stop, two pre-norm Transformer encoder layers without positional
  encoding, then a head on [global, mean of stop tokens]. Output in (0, 1),
  the continuation cost divided by E0.
* ``MLPNet`` (arm MLP): three 128-unit GELU layers on the 15-dimensional
  summary vector, then a sigmoid output.

Both return values in (0, 1). The caller multiplies by E0 to get a cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .features import F, G, MLP_DIM


@dataclass
class NetConfig:
    d: int = 64
    layers: int = 2
    heads: int = 4
    ff: int = 128


class SetNet(nn.Module):
    """Permutation-equivariant encoder over the remaining-stop tokens."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        self.cfg = cfg
        self.glob_proj = nn.Linear(G, cfg.d)
        self.tok_proj = nn.Linear(F, cfg.d)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.d, nhead=cfg.heads, dim_feedforward=cfg.ff,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True)
            for _ in range(cfg.layers)
        ])
        self.head_hidden = nn.Linear(2 * cfg.d, cfg.d)
        self.head_out = nn.Linear(cfg.d, 1)

    def forward(self, glob: Tensor, tok: Tensor) -> Tensor:
        """glob (n, G), tok (n, T, F) with T >= 1. Returns (n,) in (0, 1)."""
        if tok.dim() != 3 or tok.shape[1] < 1:
            raise ValueError("tok must have shape (n, T >= 1, F)")
        g = self.glob_proj(glob).unsqueeze(1)
        h = torch.cat([g, self.tok_proj(tok)], dim=1)
        for layer in self.encoder:
            h = layer(h)
        pooled = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        z = torch.nn.functional.gelu(self.head_hidden(pooled))
        return torch.sigmoid(self.head_out(z)).squeeze(-1)


class MLPNet(nn.Module):
    """Three 128-unit GELU layers on the 15-dimensional summary vector."""

    def __init__(self, in_dim: int = MLP_DIM, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.out = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        """x (n, 15). Returns (n,) in (0, 1)."""
        return torch.sigmoid(self.out(self.body(x))).squeeze(-1)


def n_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))
