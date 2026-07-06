"""
Probability distribution utilities for stochastic elements.
"""

import numpy as np
from typing import List, Tuple


def generate_delivery_probabilities(n_locations: int,
                                  prob_range: Tuple[float, float] = (0.6, 0.9),
                                  distribution: str = "uniform",
                                  seed: int = None) -> np.ndarray:
    """Generate delivery success probabilities for each location."""
    if seed is not None:
        np.random.seed(seed)

    min_prob, max_prob = prob_range

    if distribution == "uniform":
        return np.random.uniform(min_prob, max_prob, n_locations)
    elif distribution == "normal":
        mean = (min_prob + max_prob) / 2
        std = (max_prob - min_prob) / 4  # 95% within range
        probs = np.random.normal(mean, std, n_locations)
        return np.clip(probs, min_prob, max_prob)
    elif distribution == "heterogeneous":
        # Mix of high, medium, and low probability zones
        high_count = n_locations // 3
        medium_count = n_locations // 3
        low_count = n_locations - high_count - medium_count

        high_probs = np.random.uniform(0.8, 0.95, high_count)
        medium_probs = np.random.uniform(0.5, 0.8, medium_count)
        low_probs = np.random.uniform(0.2, 0.5, low_count)

        probs = np.concatenate([high_probs, medium_probs, low_probs])
        np.random.shuffle(probs)
        return probs
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def generate_time_dependent_probabilities(n_locations: int, n_time_slots: int = 4,
                                        base_probs: np.ndarray = None,
                                        time_multipliers: List[float] = None,
                                        seed: int = None) -> np.ndarray:
    """Generate time-dependent delivery probabilities."""
    if seed is not None:
        np.random.seed(seed)

    if base_probs is None:
        base_probs = generate_delivery_probabilities(n_locations, seed=seed)

    if time_multipliers is None:
        time_multipliers = [0.7, 0.85, 1.0, 0.8]  # morning, midday, afternoon, evening

    # Ensure we have the right number of multipliers
    time_multipliers = time_multipliers[:n_time_slots]
    while len(time_multipliers) < n_time_slots:
        time_multipliers.append(1.0)

    probs = np.zeros((n_locations, n_time_slots))
    for i, base_prob in enumerate(base_probs):
        for j, multiplier in enumerate(time_multipliers):
            probs[i, j] = np.clip(base_prob * multiplier, 0.1, 0.95)

    return probs


def generate_callback_probabilities(n_locations: int,
                                  prob_range: Tuple[float, float] = (0.3, 0.7),
                                  seed: int = None) -> np.ndarray:
    """Generate callback probabilities for failed deliveries."""
    if seed is not None:
        np.random.seed(seed)

    min_prob, max_prob = prob_range
    return np.random.uniform(min_prob, max_prob, n_locations)


def sample_delivery_outcome(location: int, time: float,
                          delivery_probs: np.ndarray,
                          seed: int = None) -> bool:
    """Sample whether a delivery attempt succeeds."""
    if seed is not None:
        np.random.seed(seed)

    if delivery_probs.ndim == 1:
        prob = delivery_probs[location]
    else:
        # Time-dependent
        time_slot = min(int(time / 60), delivery_probs.shape[1] - 1)
        prob = delivery_probs[location, time_slot]

    return np.random.random() < prob


def sample_callback_occurrence(location: int, callback_probs: np.ndarray,
                             seed: int = None) -> bool:
    """Sample whether a callback occurs after failed delivery."""
    if seed is not None:
        np.random.seed(seed)

    prob = callback_probs[location]
    return np.random.random() < prob


def sample_callback_delay(mean_delay: float = 15.0, seed: int = None) -> float:
    """Sample callback delay using exponential distribution."""
    if seed is not None:
        np.random.seed(seed)

    return np.random.exponential(mean_delay)