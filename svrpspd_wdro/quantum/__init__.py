"""Classical (exact) peak-load distribution tools for SVRPSPD routes.

See :mod:`quantum.classical` for the load model, the backward (A, G)
dynamic program for the exact peak-load pmf, and the insertion lemma.
"""

from quantum.classical import (
    DiscreteDemand,
    chance,
    cvar,
    discretize_instance,
    discretize_samples,
    insertion_window,
    peak_distribution,
    peak_load,
)

__all__ = [
    "DiscreteDemand",
    "chance",
    "cvar",
    "discretize_instance",
    "discretize_samples",
    "insertion_window",
    "peak_distribution",
    "peak_load",
]
