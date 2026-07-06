"""
TH-CB: Threshold-based Callback Policy

Uses a simple threshold-based decision rule for callback acceptance:
accept callback if expected benefit exceeds threshold. Routes are planned
dynamically using nearest neighbor with callback scoring.
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.helpers import calculate_callback_priority, find_nearest_location
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


class TH_CB:
    """
    Threshold-based Callback Policy

    Uses threshold-based decisions for callback acceptance with
    weighted scoring system for both routing and callback handling.
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = {}

        # Threshold parameters
        self.theta_accept = config.get('theta_accept', 0.6)  # Callback acceptance threshold
        self.theta_priority = config.get('theta_priority', 0.3)  # Callback priority threshold

        # Weight parameters for scoring
        self.omega_distance = config.get('omega_distance', 0.3)
        self.omega_value = config.get('omega_value', 0.25)
        self.omega_prob = config.get('omega_prob', 0.2)
        self.omega_time = config.get('omega_time', 0.15)
        self.omega_capacity = config.get('omega_capacity', 0.1)

        # Time decay parameter
        self.lambda_decay = config.get('lambda_decay', 0.05)

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """Main solving routine for TH-CB algorithm."""
        state = self._initialize_state(problem_instance)
        total_reward = 0
        episode_history = []

        iteration = 0
        max_iterations = 1000

        while not self._is_terminal(state) and iteration < max_iterations:
            # Process callback queue with threshold-based decisions
            if not state.callback_queue.is_empty():
                callback_handled = self._process_callbacks(state, problem_instance)

            # Dynamic routing for regular deliveries
            active_shippers = state.get_active_shippers()

            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._select_action_with_scoring(state, shipper, problem_instance)
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
                location=0
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

        # Assign packages using weighted scoring
        self._assign_packages_with_scoring(shippers, packages, problem_instance)

        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _assign_packages_with_scoring(self, shippers: List[Shipper], packages: List[Package],
                                    problem_instance: ProblemInstance):
        """Assign packages using capacity and efficiency scoring."""
        # Sort packages by value/weight ratio
        packages.sort(key=lambda p: p.value / max(p.weight, 0.1), reverse=True)

        for package in packages:
            best_shipper = None
            best_score = -1

            for shipper in shippers:
                if shipper.can_carry(package):
                    # Calculate assignment score
                    utilization = (shipper.current_load + package.weight) / shipper.capacity
                    capacity_score = utilization if utilization <= 0.95 else 0.5

                    # Distance from depot (prefer closer initial assignments)
                    distance_score = max(0, 1 - problem_instance.get_distance(0, package.destination) / 100.0)

                    # Delivery probability
                    prob_score = problem_instance.get_delivery_probability(package.destination)

                    score = (0.4 * capacity_score +
                            0.3 * prob_score +
                            0.3 * distance_score)

                    if score > best_score:
                        best_score = score
                        best_shipper = shipper

            if best_shipper is not None:
                best_shipper.add_package(package)

    def _select_action_with_scoring(self, state: State, shipper: Shipper,
                                  problem_instance: ProblemInstance) -> Optional[Action]:
        """Select next package using weighted scoring system."""
        if len(shipper.packages) == 0:
            return None

        best_package = None
        best_score = -np.inf

        for package in shipper.packages:
            score = self._calculate_package_score(package, shipper, state, problem_instance)

            if score > best_score:
                best_score = score
                best_package = package

        if best_package is not None:
            # Get all packages at this destination
            packages_at_dest = [pkg.id for pkg in shipper.packages
                              if pkg.destination == best_package.destination]

            return Action(
                shipper_id=shipper.id,
                next_location=best_package.destination,
                packages_to_attempt=packages_at_dest
            )

        return None

    def _calculate_package_score(self, package: Package, shipper: Shipper, state: State,
                               problem_instance: ProblemInstance) -> float:
        """Calculate weighted score for package selection."""
        # Distance factor (prefer closer packages)
        distance = problem_instance.get_distance(shipper.location, package.destination)
        max_distance = np.max(problem_instance.distance_matrix)
        distance_score = 1 - (distance / max_distance) if max_distance > 0 else 1.0

        # Delivery probability factor
        prob_score = problem_instance.get_delivery_probability(package.destination, state.current_time)

        # Value factor
        value_score = package.value / 150.0  # Normalize by typical max value

        # Time urgency factor
        time_factor = max(0, 1 - state.current_time / problem_instance.time_window)

        # Attempt penalty (prefer packages with fewer attempts)
        attempt_penalty = max(0, 1 - package.attempt_count * 0.2)

        # Combined weighted score
        score = (self.omega_distance * distance_score +
                self.omega_prob * prob_score +
                self.omega_value * value_score +
                self.omega_time * time_factor) * attempt_penalty

        return score

    def _process_callbacks(self, state: State, problem_instance: ProblemInstance) -> bool:
        """Process callback queue with threshold-based decisions."""
        callbacks_processed = False

        while not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()

            if callback.callback_time > state.current_time:
                break

            # Find best shipper for this callback
            best_shipper, best_score = self._find_best_shipper_for_callback(
                callback, state, problem_instance
            )

            if best_score > self.theta_accept:
                # Accept callback
                state.callback_queue.pop()
                callback.accepted = True
                callback.reattempt_time = state.current_time

                # Calculate insertion benefit
                callback_value = problem_instance.R_callback_success * \
                               problem_instance.get_delivery_probability(callback.package.destination)

                distance = problem_instance.get_distance(best_shipper.location, callback.package.destination)
                insertion_cost = problem_instance.calculate_movement_cost(distance, best_shipper.current_load)

                # Accept if net benefit is positive
                if callback_value - insertion_cost > 0:
                    best_shipper.add_package(callback.package)
                    callbacks_processed = True
            else:
                # Reject callback
                state.callback_queue.pop()
                callbacks_processed = True

        return callbacks_processed

    def _find_best_shipper_for_callback(self, callback: Callback, state: State,
                                      problem_instance: ProblemInstance) -> Tuple[Optional[Shipper], float]:
        """Find best shipper for callback using composite scoring."""
        best_shipper = None
        best_score = -np.inf

        for shipper in state.shippers:
            if not shipper.can_carry(callback.package):
                continue

            score = self._evaluate_callback_score(callback, shipper, state, problem_instance)

            if score > best_score:
                best_score = score
                best_shipper = shipper

        return best_shipper, best_score

    def _evaluate_callback_score(self, callback: Callback, shipper: Shipper, state: State,
                               problem_instance: ProblemInstance) -> float:
        """Calculate composite score for callback-shipper assignment."""
        # Distance factor (normalized)
        distance = problem_instance.get_distance(shipper.location, callback.package.destination)
        max_distance = np.max(problem_instance.distance_matrix)
        distance_score = 1 - (distance / max_distance) if max_distance > 0 else 1.0

        # Package value factor
        value_score = callback.package.value / 150.0

        # Delivery probability factor
        prob_score = problem_instance.get_delivery_probability(
            callback.package.destination, state.current_time
        )

        # Time factor (callback freshness)
        time_wait = state.current_time - callback.callback_time
        time_score = np.exp(-self.lambda_decay * max(0, time_wait))

        # Capacity factor
        capacity_score = 1.0 if shipper.can_carry(callback.package) else 0.0

        # Current route efficiency
        route_efficiency = len(shipper.packages) / max(1, shipper.capacity / 10.0)
        efficiency_score = min(route_efficiency / 3.0, 1.0)  # Normalize

        # Composite score
        score = (self.omega_distance * distance_score +
                self.omega_value * value_score +
                self.omega_prob * prob_score +
                self.omega_time * time_score +
                self.omega_capacity * capacity_score +
                0.1 * efficiency_score)

        return score

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

            next_shipper.location = action.next_location
            next_state.current_time += distance / 10.0
            next_shipper.route_history.append(action.next_location)

        # Delivery attempts
        if action.is_delivery_attempt:
            packages_to_remove = []

            for package_id in action.packages_to_attempt:
                package = next((p for p in next_shipper.packages if p.id == package_id), None)
                if package is None:
                    continue

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
                    reward = problem_instance.calculate_success_reward(next_state.current_time)
                    immediate_reward += reward

                    attempt.delivery_time = next_state.current_time
                    next_state.completed_deliveries.append(attempt)
                    packages_to_remove.append(package)
                else:
                    immediate_reward += problem_instance.R_failure
                    next_state.failed_deliveries.append(attempt)

                    # Add to callback queue with priority score
                    callback_occurs = sample_callback_occurrence(
                        package.destination,
                        problem_instance.callback_probabilities
                    )

                    if callback_occurs:
                        callback_delay = sample_callback_delay()
                        callback_time = next_state.current_time + callback_delay

                        # Calculate priority score for this callback
                        priority_score = self._evaluate_callback_score(
                            Callback(package, callback_time),
                            next_shipper, next_state, problem_instance
                        )

                        # Only add to queue if above priority threshold
                        if priority_score > self.theta_priority:
                            callback = Callback(
                                package=package,
                                callback_time=callback_time,
                                priority_score=priority_score
                            )

                            next_state.callback_queue.add(callback)
                            next_state.callbacks.append(callback)

                        packages_to_remove.append(package)

            for package in packages_to_remove:
                next_shipper.remove_package(package)

        next_shipper.load_history.append(next_shipper.current_load)

        return next_state, immediate_reward

    def _is_terminal(self, state: State) -> bool:
        """Check if the state is terminal."""
        return state.all_packages_processed()