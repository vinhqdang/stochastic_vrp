"""SORTIE: ordered route fairness under stochastic travel times.

Modules
-------
instances  CVRPLIB parsing and instance construction (module A)
tsp        Held-Karp, tour solvers and the Theorem 6 dichotomic search (module B)
pboi       Poisson-binomial order-statistic evaluator, rank prices (module C)
sim        arc-level Monte Carlo with common random numbers (module D)
search     upper-level screening and search over plans (module E)
"""

__version__ = "0.1.0"

__all__ = ["instances", "tsp", "pboi", "sim", "search", "__version__"]
