"""
General helper functions for the VRP implementation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import random


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def calculate_callback_priority(package, current_time: float, shipper_proximity: float,
                              available_capacity: float) -> float:
    """Calculate priority score for a callback."""
    # Time factor (more urgent if callback is older)
    time_factor = min(current_time - package.last_attempt_time if package.last_attempt_time else 0, 30) / 30

    # Value factor (higher value packages get priority)
    value_factor = package.value / 150.0  # Normalize assuming max value ~150

    # Customer tier factor
    tier_multipliers = {"standard": 1.0, "premium": 1.5, "vip": 2.0}
    tier_factor = tier_multipliers.get(package.customer_tier, 1.0)

    # Proximity factor (closer shippers = higher priority)
    proximity_factor = max(0, 1 - shipper_proximity / 50.0)  # Normalize by max distance

    # Capacity factor (can we actually handle this?)
    capacity_factor = 1.0 if available_capacity >= package.weight else 0.1

    # Attempt factor (fewer attempts = higher priority)
    attempt_factor = max(0, 1 - package.attempt_count * 0.2)

    priority = (0.3 * time_factor +
                0.25 * value_factor +
                0.2 * tier_factor +
                0.15 * proximity_factor +
                0.1 * capacity_factor) * attempt_factor

    return priority


def find_nearest_location(current_location: int, target_locations: List[int],
                         distance_matrix: np.ndarray) -> Tuple[int, float]:
    """Find the nearest location from a list of candidates."""
    if not target_locations:
        return None, float('inf')

    min_distance = float('inf')
    nearest_location = None

    for location in target_locations:
        distance = distance_matrix[current_location, location]
        if distance < min_distance:
            min_distance = distance
            nearest_location = location

    return nearest_location, min_distance


def calculate_insertion_cost(route: List[int], new_location: int, position: int,
                           distance_matrix: np.ndarray) -> float:
    """Calculate cost of inserting a location into a route at given position."""
    if position == 0:
        if len(route) == 0:
            return 0
        else:
            return distance_matrix[new_location, route[0]]
    elif position >= len(route):
        if len(route) == 0:
            return 0
        else:
            return distance_matrix[route[-1], new_location]
    else:
        # Insert between two locations
        prev_loc = route[position - 1]
        next_loc = route[position]

        old_cost = distance_matrix[prev_loc, next_loc]
        new_cost = distance_matrix[prev_loc, new_location] + distance_matrix[new_location, next_loc]

        return new_cost - old_cost


def find_best_insertion_position(route: List[int], new_location: int,
                               distance_matrix: np.ndarray) -> Tuple[int, float]:
    """Find the best position to insert a new location in a route."""
    if len(route) == 0:
        return 0, 0.0

    best_position = 0
    best_cost = float('inf')

    for position in range(len(route) + 1):
        cost = calculate_insertion_cost(route, new_location, position, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_position = position

    return best_position, best_cost


def calculate_route_distance(route: List[int], distance_matrix: np.ndarray,
                           start_location: int = 0) -> float:
    """Calculate total distance for a route."""
    if len(route) == 0:
        return 0.0

    total_distance = 0.0
    current_location = start_location

    for next_location in route:
        total_distance += distance_matrix[current_location, next_location]
        current_location = next_location

    return total_distance


def format_time(minutes: float) -> str:
    """Format time in minutes to hours:minutes format."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def validate_state_consistency(state) -> bool:
    """Validate that the state is consistent (for debugging)."""
    # Check that shipper loads match their packages
    for shipper in state.shippers:
        expected_load = sum(pkg.weight for pkg in shipper.packages)
        if abs(shipper.current_load - expected_load) > 1e-6:
            return False

    # Check that no package is assigned to multiple shippers
    all_package_ids = set()
    for shipper in state.shippers:
        for pkg in shipper.packages:
            if pkg.id in all_package_ids:
                return False
            all_package_ids.add(pkg.id)

    return True