"""
GNN-CB: Greedy Nearest Neighbor with Callback Queue

A simple baseline that always selects the nearest undelivered package,
with a FIFO callback queue mechanism.
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.helpers import find_nearest_location
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


class GNN_CB:
    """
    Greedy Nearest Neighbor with Callback Queue

    Simple baseline that uses greedy nearest neighbor selection
    with FIFO callback handling.
    """

    def __init__(self, config: Dict = None):
        self.max_detour_threshold = config.get('max_detour_threshold', 25.0) if config else 25.0

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """Main solving routine for GNN-CB algorithm."""
        state = self._initialize_state(problem_instance)
        total_reward = 0
        episode_history = []

        iteration = 0
        max_iterations = 1000

        while not self._is_terminal(state) and iteration < max_iterations:
            # Check callback queue first
            if not state.callback_queue.is_empty():
                callback_handled = self._process_callbacks(state, problem_instance)
                if callback_handled:
                    continue

            # For each shipper, greedily select nearest package
            active_shippers = state.get_active_shippers()

            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._select_greedy_action(state, shipper, problem_instance)
                if action is not None:
                    next_state, reward = self._execute_action(state, shipper, action, problem_instance)
                    total_reward += reward
                    episode_history.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
                    state = next_state

            iteration += 1

        return {
            'total_reward': total_reward,
            'state': state,
            'history': episode_history,
            'iterations': iteration
        }

    def _initialize_state(self, problem_instance: ProblemInstance) -> State:
        """Initialize the state from problem instance."""
        # Create shippers
        shippers = []
        for i in range(problem_instance.n_shippers):
            shipper = Shipper(
                id=i,
                capacity=problem_instance.shipper_capacities[i],
                location=0  # All start at depot
            )
            shippers.append(shipper)

        # Create packages
        packages = []
        for i in range(problem_instance.n_packages):
            package = Package(
                id=i,
                weight=problem_instance.package_weights[i],
                destination=problem_instance.package_destinations[i]
            )
            packages.append(package)

        # Assign packages to shippers
        self._assign_packages_to_shippers(shippers, packages)

        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _assign_packages_to_shippers(self, shippers: List[Shipper], packages: List[Package]):
        """Simple greedy package assignment."""
        packages.sort(key=lambda p: p.weight, reverse=True)

        for package in packages:
            best_shipper = None
            best_remaining = -1

            for shipper in shippers:
                if shipper.can_carry(package) and shipper.remaining_capacity > best_remaining:
                    best_remaining = shipper.remaining_capacity
                    best_shipper = shipper

            if best_shipper is not None:
                best_shipper.add_package(package)

    def _select_greedy_action(self, state: State, shipper: Shipper,
                             problem_instance: ProblemInstance) -> Optional[Action]:
        """Select the nearest undelivered package."""
        if len(shipper.packages) == 0:
            return None

        # Find nearest package
        destinations = [pkg.destination for pkg in shipper.packages]
        nearest_location, min_distance = find_nearest_location(
            shipper.location, destinations, problem_instance.distance_matrix
        )

        if nearest_location is not None:
            # Get all packages at this location
            packages_at_location = [pkg.id for pkg in shipper.packages
                                  if pkg.destination == nearest_location]

            return Action(
                shipper_id=shipper.id,
                next_location=nearest_location,
                packages_to_attempt=packages_at_location
            )

        return None

    def _process_callbacks(self, state: State, problem_instance: ProblemInstance) -> bool:
        """Process callbacks using FIFO with detour cost check."""
        if state.callback_queue.is_empty():
            return False

        callback = state.callback_queue.peek()

        if callback.callback_time > state.current_time:
            return False

        # Find nearest shipper with capacity
        nearest_shipper = self._find_nearest_shipper_with_capacity(
            callback, state, problem_instance
        )

        if nearest_shipper is not None:
            # Calculate detour cost
            distance = problem_instance.get_distance(nearest_shipper.location, callback.package.destination)
            detour_cost = problem_instance.calculate_movement_cost(distance, nearest_shipper.current_load)

            # Accept if within threshold
            if detour_cost < self.max_detour_threshold:
                state.callback_queue.pop()  # Remove from queue
                callback.accepted = True
                callback.reattempt_time = state.current_time

                # Add package back to shipper
                nearest_shipper.add_package(callback.package)
                return True

        # Reject callback if no suitable shipper found
        state.callback_queue.pop()
        return False

    def _find_nearest_shipper_with_capacity(self, callback: Callback, state: State,
                                          problem_instance: ProblemInstance) -> Optional[Shipper]:
        """Find nearest shipper that can handle the callback."""
        best_shipper = None
        min_distance = float('inf')

        for shipper in state.shippers:
            if shipper.can_carry(callback.package):
                distance = problem_instance.get_distance(shipper.location, callback.package.destination)
                if distance < min_distance:
                    min_distance = distance
                    best_shipper = shipper

        return best_shipper

    def _execute_action(self, state: State, shipper: Shipper, action: Action,
                       problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Execute an action and return resulting state and reward."""
        next_state = state.copy()
        next_shipper = next_state.shippers[shipper.id]

        immediate_reward = 0.0

        # Movement
        if action.is_movement:
            distance = problem_instance.get_distance(next_shipper.location, action.next_location)
            movement_cost = problem_instance.calculate_movement_cost(distance, next_shipper.current_load)
            immediate_reward -= movement_cost
            next_state.total_cost += movement_cost

            # Update location and time
            next_shipper.location = action.next_location
            next_state.current_time += distance / 10.0  # Assume speed of 10 units/minute
            next_shipper.route_history.append(action.next_location)

        # Delivery attempts
        if action.is_delivery_attempt:
            packages_to_remove = []

            for package_id in action.packages_to_attempt:
                package = next((p for p in next_shipper.packages if p.id == package_id), None)
                if package is None:
                    continue

                # Sample delivery outcome
                success = sample_delivery_outcome(
                    package.destination,
                    next_state.current_time,
                    problem_instance.delivery_probabilities
                )

                package.attempt_count += 1
                package.last_attempt_time = next_state.current_time

                attempt = DeliveryAttempt(
                    package_id=package.id,
                    shipper_id=shipper.id,
                    location=package.destination,
                    attempt_time=next_state.current_time,
                    successful=success,
                    attempt_number=package.attempt_count
                )

                if success:
                    # Successful delivery
                    reward = problem_instance.calculate_success_reward(next_state.current_time)
                    immediate_reward += reward

                    attempt.delivery_time = next_state.current_time
                    next_state.completed_deliveries.append(attempt)
                    packages_to_remove.append(package)
                else:
                    # Failed delivery
                    immediate_reward += problem_instance.R_failure
                    next_state.failed_deliveries.append(attempt)

                    # Check for callback (FIFO ordering)
                    callback_occurs = sample_callback_occurrence(
                        package.destination,
                        problem_instance.callback_probabilities
                    )

                    if callback_occurs:
                        callback_delay = sample_callback_delay()
                        callback_time = next_state.current_time + callback_delay

                        callback = Callback(
                            package=package,
                            callback_time=callback_time,
                            priority_score=callback_time  # FIFO: earlier callbacks have higher priority
                        )

                        next_state.callback_queue.add(callback)
                        next_state.callbacks.append(callback)
                        packages_to_remove.append(package)

            # Remove packages
            for package in packages_to_remove:
                next_shipper.remove_package(package)

        # Update load history
        next_shipper.load_history.append(next_shipper.current_load)

        return next_state, immediate_reward

    def _is_terminal(self, state: State) -> bool:
        """Check if the state is terminal."""
        return state.all_packages_processed()