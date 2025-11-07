"""
Distance calculation utilities for the VRP.
"""

import numpy as np
from typing import List, Tuple


def euclidean_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


def create_distance_matrix(locations: List[Tuple[float, float]]) -> np.ndarray:
    """Create distance matrix from list of (x, y) coordinates."""
    n = len(locations)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = euclidean_distance(locations[i], locations[j])

    return matrix


def generate_random_locations(n_locations: int, width: float = 100.0, height: float = 100.0,
                            seed: int = None) -> List[Tuple[float, float]]:
    """Generate random locations uniformly in a rectangle."""
    if seed is not None:
        np.random.seed(seed)

    locations = []
    for _ in range(n_locations):
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        locations.append((x, y))

    return locations


def generate_clustered_locations(n_clusters: int, points_per_cluster: int,
                               cluster_radius: float = 10.0,
                               width: float = 100.0, height: float = 100.0,
                               seed: int = None) -> List[Tuple[float, float]]:
    """Generate clustered locations."""
    if seed is not None:
        np.random.seed(seed)

    locations = []

    # Generate cluster centers
    centers = generate_random_locations(n_clusters, width, height)

    # Generate points around each center
    for center in centers:
        for _ in range(points_per_cluster):
            # Generate point within cluster radius
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, cluster_radius)

            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)

            # Ensure within bounds
            x = max(0, min(width, x))
            y = max(0, min(height, y))

            locations.append((x, y))

    return locations


def generate_hub_spoke_locations(hub_location: Tuple[float, float],
                               spoke_locations: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Generate hub-and-spoke network topology."""
    locations = [hub_location]
    locations.extend(spoke_locations)
    return locations