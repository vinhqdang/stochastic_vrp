"""RLSM for Q1 feature sets (EXPERIMENT_C.md 1.3, module MC2).

B's RLSM (``expb.rlsm``, K = 100, ridge 1e-6 n) on the 15 mlp inputs with the
columns outside the set set to zero. Zero-masking keeps the random layer (A, b)
the same for every set: it is drawn from the set-independent seed key
``rlsm_seed_key`` by ``draw_layer``, with the same 15 input rows. The same mask is
applied at fit and at predict time (through ``RLSMPredictor.featurize``).
"""
from __future__ import annotations

import numpy as np

from ..expa.features import RouteCache, mlp_batch
from ..expa.paths import Paths, RouteSpec
from ..expb.rlsm import RLSMConfig, fit_rlsm, rlsm_fit_fn, rlsm_seed_key
from .featuresets import columns_of, mask_columns


class MaskedFeaturize:
    """(cache, k, idx) -> mlp_batch with the columns outside the set zeroed. A module-level
    callable class (not a closure), so that a fitted RLSMPredictor pickles and reloads."""

    def __init__(self, set_name: str):
        columns_of(set_name)
        self.set_name = set_name

    def __call__(self, cache: RouteCache, k: int, idx) -> "np.ndarray":
        return mask_columns(mlp_batch(cache, k, idx), self.set_name)


def masked_featurize(set_name: str) -> MaskedFeaturize:
    """The picklable input map of one feature set (see MaskedFeaturize)."""
    return MaskedFeaturize(set_name)


def rlsm_set_fit_fn(seed_key: str, cfg: RLSMConfig, diag: list, set_name: str):
    """FitFn for ls_backward on one feature set (B's RLSM FitFn with the mask)."""
    return rlsm_fit_fn(seed_key, cfg, diag, featurize=masked_featurize(set_name))


def fit_rlsm_set(spec: RouteSpec, paths: Paths, N: int, rep: int, cfg: RLSMConfig,
                 set_name: str):
    """Cross-fitted RLSM models for one route and feature set. Same return value as
    ``expb.rlsm.fit_rlsm``: (models, RouteStats, diag, budget)."""
    return fit_rlsm(spec, paths, N, rep, cfg, featurize=masked_featurize(set_name),
                    seed_key=rlsm_seed_key(spec, N, rep))
