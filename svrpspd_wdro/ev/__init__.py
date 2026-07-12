"""ev — paper 2: anytime-valid re-optimization for stochastic routing.

An e-process monitors a running routing plan against the planning
model's joint law over demand, travel time, dwell time, accidents and
breakdowns; crossing 1/alpha triggers a (warm-started ALNS) replan with
anytime-valid type-I error control. See papers/ev_reopt/PROJECT.md.
"""

from .world import DayParams, DriftSpec, simulate_route_day  # noqa: F401
from .eprocess import MasterEProcess, THETA_GRID             # noqa: F401
from .baselines import CusumMonitor, PeriodicMonitor, BonferroniFixed  # noqa: F401
