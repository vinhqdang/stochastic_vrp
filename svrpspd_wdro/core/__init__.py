"""Core data structures: Instance, Route, scenario generation."""

from core.instance import Instance
from core.route import Route
from core.scenarios import ScenarioConfig, generate_scenarios

__all__ = ["Instance", "Route", "ScenarioConfig", "generate_scenarios"]