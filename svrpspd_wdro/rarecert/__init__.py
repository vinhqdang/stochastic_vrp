"""RARECERT numerics: closed-form stage-mixture importance sampling (SMIS) for the
rare overflow probability of a route. See papers/rarecert/THEORY.md and
:mod:`rarecert.smis`. The route simulator and the mixture estimator are imported
from :mod:`quantum.rare`."""

from rarecert.smis import (
    block_direction,
    block_scale,
    dominating_point,
    smis_estimate,
    smis_mixture,
    stage_scales,
)

__all__ = [
    "block_direction",
    "block_scale",
    "dominating_point",
    "smis_estimate",
    "smis_mixture",
    "stage_scales",
]
