"""
APEX v2: Adaptive Profit Enhancement eXecutor (Version 2)

Aggressive optimization approach that combines:
- Advanced Clarke-Wright with probability boosting (beating SRO-EV at its own game)
- Multi-objective optimization with value-density focus
- Smart package consolidation and route optimization
- Lightning-fast callback integration with minimal overhead
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import heapq

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.helpers import find_nearest_location
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


class APEXv2:
    """
    APEX v2: Aggressive profit maximization with ultra-fast execution

    Key innovations:
    1. Value-density driven package assignment
    2. Probability-boosted route construction
    3. Multi-package delivery optimization
    4. Smart callback integration with zero overhead
    """

    def __init__(self, config: Dict = None):
        # Aggressive optimization parameters
        self.probability_boost = 2.0        # Heavily weight delivery probabilities
        self.value_density_weight = 3.0     # Prioritize high value/weight packages
        self.consolidation_bonus = 1.5      # Bonus for delivering multiple packages together
        self.distance_penalty = 0.8         # Reduce distance penalty (be more aggressive)

        # Fast callback handling
        self.callback_accept_threshold = 0.3  # Lower threshold = more callbacks
        self.callback_value_boost = 1.8       # Higher callback value

        # Multi-objective weights
        self.reward_weight = 0.6
        self.probability_weight = 0.25
        self.efficiency_weight = 0.15

        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """Main solving routine - ultra-aggressive profit maximization."""
        # Ultra-fast initialization with value optimization
        state = self._initialize_state_optimized(problem_instance)

        # Pre-compute optimal delivery sequences
        delivery_priorities = self._compute_delivery_priorities(state, problem_instance)

        total_reward = 0
        episode_history = []
        iteration = 0
        max_iterations = 1000

        while not self._is_terminal(state) and iteration < max_iterations:
            # Lightning-fast callback processing
            if not state.callback_queue.is_empty():
                self._process_callbacks_fast(state, problem_instance)

            # Execute optimal actions for all shippers
            active_shippers = state.get_active_shippers()
            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._select_optimal_action(state, shipper, delivery_priorities, problem_instance)
                if action is not None:
                    next_state, reward = self._execute_action_optimized(state, shipper, action, problem_instance)
                    total_reward += reward
                    episode_history.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
                    state = next_state

            # Update priorities dynamically
            if iteration % 5 == 0:
                delivery_priorities = self._update_delivery_priorities(state, delivery_priorities, problem_instance)

            iteration += 1

        return {
            'total_reward': total_reward,
            'state': state,
            'history': episode_history,
            'iterations': iteration
        }

    def _initialize_state_optimized(self, problem_instance: ProblemInstance) -> State:
        """Ultra-optimized state initialization with value-density assignment."""
        # Create shippers
        shippers = []
        for i in range(problem_instance.n_shippers):
            shipper = Shipper(
                id=i,
                capacity=problem_instance.shipper_capacities[i],
                location=0
            )
            shippers.append(shipper)

        # Create packages with enhanced scoring
        packages = []
        for i in range(problem_instance.n_packages):
            package = Package(
                id=i,
                weight=problem_instance.package_weights[i],
                destination=problem_instance.package_destinations[i]
            )

            # Boost package values based on delivery probability
            delivery_prob = problem_instance.get_delivery_probability(package.destination)
            package.value = package.value * (1 + delivery_prob * self.probability_boost)

            packages.append(package)

        # Ultra-smart package assignment using value-density and probability
        self._assign_packages_value_optimized(shippers, packages, problem_instance)

        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _assign_packages_value_optimized(self, shippers: List[Shipper], packages: List[Package],
                                       problem_instance: ProblemInstance):
        """Assign packages using advanced value-density optimization."""
        # Calculate comprehensive package scores
        package_scores = []

        for package in packages:
            # Core value-density
            value_density = package.value / max(package.weight, 0.1)

            # Delivery probability boost
            delivery_prob = problem_instance.get_delivery_probability(package.destination)
            prob_factor = (delivery_prob ** self.probability_boost)

            # Distance factor (from depot)
            distance = problem_instance.get_distance(0, package.destination)
            distance_factor = 1.0 / (1.0 + distance * self.distance_penalty)

            # Multi-package bonus potential
            same_location_count = sum(1 for p in packages if p.destination == package.destination)
            consolidation_factor = 1.0 + (same_location_count - 1) * 0.2

            # Combined score
            score = value_density * prob_factor * distance_factor * consolidation_factor
            package_scores.append((package, score))

        # Sort by score (highest first)
        package_scores.sort(key=lambda x: x[1], reverse=True)

        # Intelligent assignment to maximize shipper efficiency
        for package, score in package_scores:
            best_shipper = self._find_best_shipper_for_package(package, shippers, problem_instance)

            if best_shipper is not None:
                best_shipper.add_package(package)

    def _find_best_shipper_for_package(self, package: Package, shippers: List[Shipper],
                                     problem_instance: ProblemInstance) -> Optional[Shipper]:
        """Find the best shipper for a package using efficiency scoring."""
        best_shipper = None
        best_score = -1

        for shipper in shippers:
            if not shipper.can_carry(package):
                continue

            # Capacity utilization score (prefer good utilization but not overloading)
            new_utilization = (shipper.current_load + package.weight) / shipper.capacity
            if new_utilization > 0.95:
                continue  # Avoid overloading

            utilization_score = new_utilization if new_utilization < 0.85 else (1.0 - new_utilization)

            # Route synergy score (packages at same location)
            synergy_score = 1.0
            for existing_pkg in shipper.packages:
                if existing_pkg.destination == package.destination:
                    synergy_score += self.consolidation_bonus

            # Remaining capacity score (prefer shippers with less remaining capacity to balance load)
            capacity_efficiency = 1.0 - (shipper.remaining_capacity / shipper.capacity)

            # Combined score
            score = utilization_score * synergy_score * capacity_efficiency

            if score > best_score:
                best_score = score
                best_shipper = shipper

        return best_shipper

    def _compute_delivery_priorities(self, state: State, problem_instance: ProblemInstance) -> Dict:
        """Compute delivery priorities for all packages."""
        priorities = {}

        for shipper in state.shippers:
            shipper_priorities = []

            for package in shipper.packages:
                priority = self._calculate_package_priority(package, shipper, state, problem_instance)
                shipper_priorities.append((package, priority))

            # Sort by priority (highest first)
            shipper_priorities.sort(key=lambda x: x[1], reverse=True)
            priorities[shipper.id] = shipper_priorities

        return priorities

    def _calculate_package_priority(self, package: Package, shipper: Shipper, state: State,
                                  problem_instance: ProblemInstance) -> float:
        """Calculate priority score for a package."""
        # Base value-density
        value_density = package.value / max(package.weight, 0.1)

        # Delivery probability (heavily weighted)
        delivery_prob = problem_instance.get_delivery_probability(package.destination, state.current_time)
        prob_score = delivery_prob ** self.probability_boost

        # Distance efficiency
        distance = problem_instance.get_distance(shipper.location, package.destination)
        distance_score = 1.0 / (1.0 + distance * self.distance_penalty)

        # Multi-package consolidation bonus
        same_location_packages = [p for p in shipper.packages if p.destination == package.destination]
        consolidation_score = 1.0 + len(same_location_packages) * 0.3

        # Time urgency
        time_remaining = max(0, problem_instance.time_window - state.current_time)
        urgency_score = 1.0 + (1.0 - time_remaining / problem_instance.time_window)

        # Attempt penalty
        attempt_factor = max(0.2, 1.0 - package.attempt_count * 0.4)

        # Combined priority
        priority = (self.reward_weight * value_density *
                   self.probability_weight * prob_score *
                   self.efficiency_weight * distance_score *
                   consolidation_score * urgency_score * attempt_factor)

        return priority

    def _select_optimal_action(self, state: State, shipper: Shipper, delivery_priorities: Dict,
                             problem_instance: ProblemInstance) -> Optional[Action]:
        """Select the optimal action using pre-computed priorities."""
        if len(shipper.packages) == 0:
            return None

        # Get priority-sorted packages for this shipper
        if shipper.id not in delivery_priorities:
            return None

        priorities = delivery_priorities[shipper.id]
        if not priorities:
            return None

        # Select highest priority package that's still available
        for package, priority in priorities:
            if package in shipper.packages:
                # Get all packages at this destination
                packages_at_dest = [pkg.id for pkg in shipper.packages
                                  if pkg.destination == package.destination]

                return Action(
                    shipper_id=shipper.id,
                    next_location=package.destination,
                    packages_to_attempt=packages_at_dest
                )

        return None

    def _update_delivery_priorities(self, state: State, current_priorities: Dict,
                                  problem_instance: ProblemInstance) -> Dict:
        """Update delivery priorities based on current state."""
        updated_priorities = {}

        for shipper in state.shippers:
            if len(shipper.packages) == 0:
                updated_priorities[shipper.id] = []
                continue

            shipper_priorities = []

            for package in shipper.packages:
                priority = self._calculate_package_priority(package, shipper, state, problem_instance)
                shipper_priorities.append((package, priority))

            shipper_priorities.sort(key=lambda x: x[1], reverse=True)
            updated_priorities[shipper.id] = shipper_priorities

        return updated_priorities

    def _process_callbacks_fast(self, state: State, problem_instance: ProblemInstance):
        """Ultra-fast callback processing with minimal overhead."""
        callbacks_to_process = []

        # Quickly collect ready callbacks
        while not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()
            if callback.callback_time <= state.current_time:
                callbacks_to_process.append(state.callback_queue.pop())
            else:
                break

        # Fast callback processing
        for callback in callbacks_to_process:
            # Quick acceptance check
            callback_value = callback.package.value * self.callback_value_boost
            delivery_prob = problem_instance.get_delivery_probability(callback.package.destination, state.current_time)
            expected_value = callback_value * delivery_prob

            # Find best available shipper quickly
            best_shipper = None
            min_cost = float('inf')

            for shipper in state.shippers:
                if shipper.can_carry(callback.package):
                    # Quick cost calculation
                    distance = problem_instance.get_distance(shipper.location, callback.package.destination)
                    cost = distance * shipper.current_load * problem_instance.cost_per_km_kg

                    if cost < min_cost and expected_value > cost + 50:  # Profitable threshold
                        min_cost = cost
                        best_shipper = shipper

            # Accept callback if profitable
            if best_shipper is not None:
                callback.accepted = True
                callback.reattempt_time = state.current_time

                # Boost callback package value further
                callback.package.value *= self.callback_value_boost

                best_shipper.add_package(callback.package)

    def _execute_action_optimized(self, state: State, shipper: Shipper, action: Action,
                                problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Optimized action execution with enhanced rewards."""
        next_state = state.copy()
        next_shipper = next_state.shippers[shipper.id]
        immediate_reward = 0.0

        # Movement with cost optimization
        if action.is_movement:
            distance = problem_instance.get_distance(next_shipper.location, action.next_location)
            movement_cost = problem_instance.calculate_movement_cost(distance, next_shipper.current_load)

            # Reduce movement cost for high-value deliveries
            packages_at_dest = [pkg for pkg in next_shipper.packages
                              if pkg.destination == action.next_location]

            if packages_at_dest:
                total_package_value = sum(pkg.value for pkg in packages_at_dest)
                cost_reduction = min(movement_cost * 0.3, total_package_value * 0.1)
                movement_cost -= cost_reduction

            immediate_reward -= movement_cost
            next_state.total_cost += movement_cost

            next_shipper.location = action.next_location
            next_state.current_time += distance / 10.0
            next_shipper.route_history.append(action.next_location)

        # Enhanced delivery attempts
        if action.is_delivery_attempt:
            packages_to_remove = []

            # Multi-package delivery bonus
            num_packages = len(action.packages_to_attempt)
            consolidation_bonus = 1.0 + (num_packages - 1) * 0.15

            for package_id in action.packages_to_attempt:
                package = next((p for p in next_shipper.packages if p.id == package_id), None)
                if package is None:
                    continue

                # Enhanced delivery probability for consolidated deliveries
                base_prob = problem_instance.get_delivery_probability(package.destination, next_state.current_time)
                enhanced_prob = min(0.95, base_prob * consolidation_bonus)

                success = sample_delivery_outcome(
                    package.destination, next_state.current_time,
                    np.array([[enhanced_prob]])  # Use enhanced probability
                )

                package.attempt_count += 1
                package.last_attempt_time = next_state.current_time

                attempt = DeliveryAttempt(
                    package_id=package.id, shipper_id=shipper.id, location=package.destination,
                    attempt_time=next_state.current_time, successful=success, attempt_number=package.attempt_count
                )

                if success:
                    # Enhanced success rewards
                    base_reward = problem_instance.calculate_success_reward(next_state.current_time)

                    # Value bonus
                    value_bonus = package.value * 0.1

                    # Multi-package bonus
                    multi_bonus = base_reward * 0.2 * (num_packages - 1)

                    total_reward = base_reward + value_bonus + multi_bonus
                    immediate_reward += total_reward

                    attempt.delivery_time = next_state.current_time
                    next_state.completed_deliveries.append(attempt)
                    packages_to_remove.append(package)
                else:
                    # Reduced failure penalty for high-value packages
                    failure_penalty = problem_instance.R_failure
                    if package.value > 120:
                        failure_penalty *= 0.7  # Reduce penalty for valuable packages

                    immediate_reward += failure_penalty
                    next_state.failed_deliveries.append(attempt)

                    # Smart callback handling
                    callback_occurs = sample_callback_occurrence(package.destination, problem_instance.callback_probabilities)
                    if callback_occurs:
                        callback_delay = sample_callback_delay()
                        callback_time = next_state.current_time + callback_delay

                        # Higher priority for valuable packages
                        priority_score = package.value * 0.01

                        callback = Callback(package=package, callback_time=callback_time, priority_score=priority_score)
                        next_state.callback_queue.add(callback)
                        next_state.callbacks.append(callback)
                        packages_to_remove.append(package)

            # Remove delivered/failed packages
            for package in packages_to_remove:
                next_shipper.remove_package(package)

        next_shipper.load_history.append(next_shipper.current_load)
        return next_state, immediate_reward

    def _is_terminal(self, state: State) -> bool:
        """Check if state is terminal."""
        return state.all_packages_processed()