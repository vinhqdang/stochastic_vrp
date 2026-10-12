"""Classical (exact) peak-load distribution tools and Qiskit peak-load oracles
for SVRPSPD routes.

See :mod:`quantum.classical` for the load model, the backward (A, G)
dynamic program for the exact peak-load pmf, and the insertion lemma. See
:mod:`quantum.oracle` for the coherent peak-load and difference oracles.
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
from quantum.oracle import (
    PeakOracle,
    build_difference_oracle,
    build_peak_oracle,
    estimate_probability,
    evaluate_basis,
    exact_probability,
    resource_counts,
)

__all__ = [
    "DiscreteDemand",
    "PeakOracle",
    "build_difference_oracle",
    "build_peak_oracle",
    "chance",
    "cvar",
    "discretize_instance",
    "discretize_samples",
    "estimate_probability",
    "evaluate_basis",
    "exact_probability",
    "insertion_window",
    "peak_distribution",
    "peak_load",
    "resource_counts",
]
