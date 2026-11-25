"""Nested feature sets for Q1 (EXPERIMENT_C.md section 1.3, module MC2).

Column indices refer to the 15-column ``expa.features.mlp_batch``: columns 0..11
are the globals g1..g12 (0-based), columns 12..14 the token summaries s1..s3.
"""
from __future__ import annotations

import numpy as np

from ..expa.features import MLP_DIM

_TOKENS = (12, 13, 14)

FEATURE_SETS: dict[str, tuple[int, ...]] = {
    "F0": (0, 2, 3, 4, 5, 6) + _TOKENS,                  # g1, g3..g7
    "F0n": (0, 1, 2, 3, 4, 5, 6, 11) + _TOKENS,          # + g2, g12
    "F1": (0, 2, 3, 4, 5, 6, 7, 8) + _TOKENS,            # + g8, g9
    "F2": (0, 2, 3, 4, 5, 6, 7, 8, 9, 10) + _TOKENS,     # + g10, g11
    "F3": tuple(range(MLP_DIM)),                         # all 12 globals + s1..s3
}


def columns_of(set_name: str) -> tuple[int, ...]:
    """Column indices of a named set; unknown names raise ValueError."""
    if set_name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {set_name!r}; expected one of "
                         f"{tuple(FEATURE_SETS)}")
    return FEATURE_SETS[set_name]


def _check_width(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != MLP_DIM:
        raise ValueError(f"expected an (n, {MLP_DIM}) array, got shape {x.shape}")
    return x


def mask_columns(x: np.ndarray, set_name: str) -> np.ndarray:
    """Copy of x with the columns outside the set set to 0.0 (shape unchanged)."""
    cols = columns_of(set_name)
    out = np.array(_check_width(x), copy=True)
    keep = np.zeros(MLP_DIM, dtype=bool)
    keep[list(cols)] = True
    out[:, ~keep] = 0.0
    return out


def select_columns(x: np.ndarray, set_name: str) -> np.ndarray:
    """Only the set's columns of x, in index order (n, len(set))."""
    return _check_width(x)[:, list(columns_of(set_name))]
