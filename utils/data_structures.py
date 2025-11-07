"""
Core data structures for the Stochastic VRP with Callbacks problem.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import copy


@dataclass
class Package:
    """Represents a package to be delivered."""
    id: int
    weight: float
    destination: int  # location index
    value: float = 100.0
    expected_delivery_time: float = 0.0
    attempt_count: int = 0
    assignment_time: float = 0.0
    last_attempt_time: Optional[float] = None
    customer_tier: str = "standard"  # standard, premium, vip


@dataclass
class Shipper:
    """Represents a delivery shipper/vehicle."""
    id: int
    capacity: float
    location: int = 0  # current location index
    current_load: float = 0.0
    packages: List[Package] = field(default_factory=list)
    completion_time: float = 0.0
    load_history: List[float] = field(default_factory=list)
    route_history: List[int] = field(default_factory=list)

    @property
    def remaining_capacity(self) -> float:
        return self.capacity - self.current_load

    def can_carry(self, package: Package) -> bool:
        return self.remaining_capacity >= package.weight

    def add_package(self, package: Package) -> bool:
        if self.can_carry(package):
            self.packages.append(package)
            self.current_load += package.weight
            return True
        return False

    def remove_package(self, package: Package) -> bool:
        if package in self.packages:
            self.packages.remove(package)
            self.current_load -= package.weight
            return True
        return False


@dataclass
class Callback:
    """Represents a customer callback for re-delivery."""
    package: Package
    callback_time: float
    priority_score: float = 0.0
    accepted: bool = False
    reattempt_time: Optional[float] = None
    successful: bool = False


@dataclass
class DeliveryAttempt:
    """Represents a delivery attempt."""
    package_id: int
    shipper_id: int
    location: int
    attempt_time: float
    successful: bool
    attempt_number: int = 1
    delivery_time: Optional[float] = None


class CallbackQueue:
    """Priority queue for managing callbacks."""

    def __init__(self):
        self.callbacks: List[Callback] = []

    def add(self, callback: Callback):
        self.callbacks.append(callback)
        # Sort by priority score (higher is better)
        self.callbacks.sort(key=lambda x: x.priority_score, reverse=True)

    def peek(self) -> Optional[Callback]:
        return self.callbacks[0] if self.callbacks else None

    def pop(self) -> Optional[Callback]:
        return self.callbacks.pop(0) if self.callbacks else None

    def is_empty(self) -> bool:
        return len(self.callbacks) == 0

    def size(self) -> int:
        return len(self.callbacks)


@dataclass
class State:
    """Represents the current system state."""
    shippers: List[Shipper]
    pending_deliveries: List[Package] = field(default_factory=list)
    callback_queue: CallbackQueue = field(default_factory=CallbackQueue)
    completed_deliveries: List[DeliveryAttempt] = field(default_factory=list)
    failed_deliveries: List[DeliveryAttempt] = field(default_factory=list)
    callbacks: List[Callback] = field(default_factory=list)
    current_time: float = 0.0
    total_cost: float = 0.0

    def copy(self):
        """Create a deep copy of the state."""
        return copy.deepcopy(self)

    def get_active_shippers(self) -> List[Shipper]:
        """Get shippers that still have packages to deliver."""
        return [s for s in self.shippers if len(s.packages) > 0]

    def all_packages_processed(self) -> bool:
        """Check if all packages have been delivered or failed."""
        total_assigned = sum(len(s.packages) for s in self.shippers)
        return total_assigned == 0 and len(self.pending_deliveries) == 0


@dataclass
class Action:
    """Represents an action a shipper can take."""
    shipper_id: int
    next_location: Optional[int] = None
    packages_to_attempt: List[int] = field(default_factory=list)  # package IDs
    callback_response: Optional[str] = None  # "accept", "defer", "reject"
    callback_id: Optional[int] = None

    @property
    def is_movement(self) -> bool:
        return self.next_location is not None

    @property
    def is_delivery_attempt(self) -> bool:
        return len(self.packages_to_attempt) > 0

    @property
    def is_callback_response(self) -> bool:
        return self.callback_response is not None


@dataclass
class ProblemInstance:
    """Represents a problem instance with all parameters."""
    n_shippers: int
    n_packages: int
    n_locations: int
    shipper_capacities: List[float]
    package_weights: List[float]
    package_destinations: List[int]
    locations: List[Tuple[float, float]]  # (x, y) coordinates
    distance_matrix: np.ndarray
    delivery_probabilities: np.ndarray  # P(success | location, time)
    callback_probabilities: np.ndarray  # P(callback | location)

    # Reward parameters
    R_success_base: float = 100.0
    R_failure: float = -50.0
    R_callback_success: float = 80.0
    cost_per_km_kg: float = 0.5
    time_window: float = 120.0

    # Additional parameters
    network_type: str = "uniform"
    scenario_name: str = "default"

    def get_distance(self, loc1: int, loc2: int) -> float:
        """Get distance between two locations."""
        return self.distance_matrix[loc1, loc2]

    def get_delivery_probability(self, location: int, time: float = 0.0) -> float:
        """Get probability of successful delivery at location and time."""
        if self.delivery_probabilities.ndim == 1:
            return self.delivery_probabilities[location]
        else:
            # Time-dependent probabilities (simplified)
            time_index = min(int(time / 60), self.delivery_probabilities.shape[1] - 1)
            return self.delivery_probabilities[location, time_index]

    def get_callback_probability(self, location: int) -> float:
        """Get probability of callback after failed delivery."""
        return self.callback_probabilities[location]

    def calculate_movement_cost(self, distance: float, load: float) -> float:
        """Calculate cost of moving distance with given load."""
        return distance * load * self.cost_per_km_kg

    def calculate_success_reward(self, delivery_time: float) -> float:
        """Calculate reward for successful delivery with time decay."""
        time_factor = max(0, 1 - delivery_time / self.time_window)
        return self.R_success_base * time_factor