"""
APEX: Adaptive Profit Enhancement eXecutor

A hybrid algorithm that combines the best aspects of all baseline approaches:
- Static route optimization for initial planning (from SRO-EV)
- Dynamic greedy selection with smart scoring (from GNN-CB + TH-CB)
- Intelligent callback integration (improved from ECHO)
- Adaptive strategy switching based on scenario characteristics
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.helpers import calculate_callback_priority, find_nearest_location
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


@dataclass
class APEXConfig:
    """Configuration for APEX algorithm."""
    # Strategy selection thresholds
    uncertainty_threshold: float = 0.7  # Switch strategies based on delivery probability variance
    density_threshold: float = 2.0      # Packages per location ratio

    # Route optimization parameters
    savings_weight: float = 0.6         # Weight for Clarke-Wright savings
    prob_weight: float = 0.4           # Weight for probability adjustment

    # Dynamic scoring weights
    distance_weight: float = 0.25
    probability_weight: float = 0.30
    value_weight: float = 0.20
    urgency_weight: float = 0.15
    capacity_weight: float = 0.10

    # Callback handling
    callback_value_multiplier: float = 1.2  # Boost for callback packages
    max_detour_ratio: float = 0.3          # Max detour as fraction of current route
    callback_decay: float = 0.1             # Priority decay over time

    # Adaptive parameters
    performance_window: int = 5             # Moving window for performance tracking
    strategy_switch_threshold: float = 0.15 # Performance difference to trigger switch


class APEX:
    """
    APEX: Adaptive Profit Enhancement eXecutor

    Hybrid algorithm that adapts its strategy based on problem characteristics
    and real-time performance feedback.
    """

    def __init__(self, config: Dict = None):
        self.config = APEXConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Strategy tracking
        self.current_strategy = "hybrid"  # "static", "dynamic", "hybrid"
        self.performance_history = []
        self.strategy_performance = {"static": [], "dynamic": [], "hybrid": []}

        # Route optimization state
        self.initial_routes = {}
        self.route_deviations = {}

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """Main solving routine for APEX algorithm."""
        # Analyze problem characteristics
        scenario_profile = self._analyze_scenario(problem_instance)

        # Select initial strategy
        self.current_strategy = self._select_initial_strategy(scenario_profile)

        # Initialize state
        state = self._initialize_state(problem_instance)

        # Pre-compute optimal routes for static fallback
        if self.current_strategy in ["static", "hybrid"]:
            self._precompute_routes(state, problem_instance)

        total_reward = 0
        episode_history = []
        iteration = 0
        max_iterations = 1000

        while not self._is_terminal(state) and iteration < max_iterations:
            # Adaptive strategy switching
            if iteration % 10 == 0 and iteration > 0:
                self._evaluate_strategy_performance(total_reward, iteration)

            # Process callbacks with advanced prioritization
            if not state.callback_queue.is_empty():
                self._process_callbacks_advanced(state, problem_instance)

            # Execute current strategy
            active_shippers = state.get_active_shippers()
            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._select_action_by_strategy(state, shipper, problem_instance)
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
            'iterations': iteration,
            'strategy_used': self.current_strategy,
            'strategy_switches': len(set(h[1].shipper_id for h in episode_history))
        }

    def _analyze_scenario(self, problem_instance: ProblemInstance) -> Dict:
        """Analyze scenario characteristics to guide strategy selection."""
        # Uncertainty level
        prob_variance = np.var(problem_instance.delivery_probabilities.flatten())
        mean_prob = np.mean(problem_instance.delivery_probabilities.flatten())
        uncertainty_level = prob_variance / (mean_prob * (1 - mean_prob)) if mean_prob > 0 else 1.0

        # Density
        density = problem_instance.n_packages / problem_instance.n_locations

        # Capacity pressure
        total_weight = sum(problem_instance.package_weights)
        total_capacity = sum(problem_instance.shipper_capacities)
        capacity_pressure = total_weight / total_capacity if total_capacity > 0 else 1.0

        # Network spread
        locations = np.array(problem_instance.locations)
        if len(locations) > 1:
            distances = []
            for i in range(len(locations)):
                for j in range(i+1, len(locations)):
                    dist = np.linalg.norm(locations[i] - locations[j])
                    distances.append(dist)
            network_spread = np.std(distances) / np.mean(distances) if distances else 0
        else:
            network_spread = 0

        return {
            'uncertainty_level': uncertainty_level,
            'density': density,
            'capacity_pressure': capacity_pressure,
            'network_spread': network_spread,
            'mean_delivery_prob': mean_prob
        }

    def _select_initial_strategy(self, profile: Dict) -> str:
        """Select initial strategy based on scenario profile."""
        uncertainty = profile['uncertainty_level']
        density = profile['density']

        if uncertainty < 0.3 and profile['mean_delivery_prob'] > 0.8:
            return "static"  # Low uncertainty, high success rate
        elif uncertainty > 0.7 or density > 3.0:
            return "dynamic"  # High uncertainty or high density
        else:
            return "hybrid"  # Balanced approach

    def _precompute_routes(self, state: State, problem_instance: ProblemInstance):
        """Pre-compute optimal routes using enhanced Clarke-Wright algorithm."""
        # Enhanced savings matrix with probability weighting
        savings_matrix = self._compute_enhanced_savings_matrix(state, problem_instance)

        # Group packages by location
        location_packages = {}
        all_packages = []
        for shipper in state.shippers:
            all_packages.extend(shipper.packages)

        for package in all_packages:
            if package.destination not in location_packages:
                location_packages[package.destination] = []
            location_packages[package.destination].append(package)

        # Build routes using savings
        routes = self._build_routes_from_savings(savings_matrix, location_packages, problem_instance)

        # Store routes for later use
        self.initial_routes = routes

    def _compute_enhanced_savings_matrix(self, state: State, problem_instance: ProblemInstance) -> np.ndarray:
        """Compute enhanced Clarke-Wright savings with probability and value weighting."""
        n_locations = len(problem_instance.locations)
        savings = np.zeros((n_locations, n_locations))

        for i in range(1, n_locations):
            for j in range(i + 1, n_locations):
                # Base Clarke-Wright savings
                base_saving = (problem_instance.distance_matrix[0, i] +
                              problem_instance.distance_matrix[0, j] -
                              problem_instance.distance_matrix[i, j])

                # Probability enhancement
                prob_i = problem_instance.get_delivery_probability(i)
                prob_j = problem_instance.get_delivery_probability(j)
                prob_factor = (prob_i + prob_j) / 2

                # Value enhancement (packages at these locations)
                value_factor = 1.0
                for shipper in state.shippers:
                    for package in shipper.packages:
                        if package.destination == i or package.destination == j:
                            value_factor += package.value / 1000  # Normalize

                # Combined savings
                enhanced_saving = (self.config.savings_weight * base_saving *
                                 self.config.prob_weight * prob_factor *
                                 value_factor)

                savings[i, j] = enhanced_saving

        return savings

    def _build_routes_from_savings(self, savings_matrix: np.ndarray,
                                 location_packages: Dict,
                                 problem_instance: ProblemInstance) -> Dict:
        """Build routes from savings matrix with capacity constraints."""
        # Convert to list and sort
        savings_list = []
        n_locations = savings_matrix.shape[0]

        for i in range(1, n_locations):
            for j in range(i + 1, n_locations):
                if savings_matrix[i, j] > 0:
                    savings_list.append((i, j, savings_matrix[i, j]))

        savings_list.sort(key=lambda x: x[2], reverse=True)

        # Initialize routes
        routes = {loc: [loc] for loc in location_packages.keys()}

        # Merge routes based on savings
        for i, j, saving in savings_list:
            if i in routes and j in routes and routes[i] != routes[j]:
                # Check capacity feasibility
                weight_i = sum(pkg.weight for pkg in location_packages.get(i, []))
                weight_j = sum(pkg.weight for pkg in location_packages.get(j, []))

                # Find shipper that can handle combined weight
                for shipper in problem_instance.shipper_capacities:
                    if shipper >= weight_i + weight_j:
                        # Merge routes
                        route_i = routes[i]
                        route_j = routes[j]

                        merged_route = route_i + route_j

                        # Update all locations in merged routes
                        for loc in route_i + route_j:
                            routes[loc] = merged_route

                        break

        return routes

    def _select_action_by_strategy(self, state: State, shipper: Shipper,
                                 problem_instance: ProblemInstance) -> Optional[Action]:
        """Select action based on current strategy."""
        if self.current_strategy == "static":
            return self._select_static_action(state, shipper, problem_instance)
        elif self.current_strategy == "dynamic":
            return self._select_dynamic_action(state, shipper, problem_instance)
        else:  # hybrid
            return self._select_hybrid_action(state, shipper, problem_instance)

    def _select_static_action(self, state: State, shipper: Shipper,
                            problem_instance: ProblemInstance) -> Optional[Action]:
        """Static strategy: follow pre-computed optimal routes."""
        if len(shipper.packages) == 0:
            return None

        # Find the best package from pre-computed routes
        best_package = None
        best_score = -1

        for package in shipper.packages:
            # Check if this package is in a good route
            if package.destination in self.initial_routes:
                route = self.initial_routes[package.destination]
                route_score = len(route)  # Longer routes are better (more savings)

                # Adjust by delivery probability
                prob_score = problem_instance.get_delivery_probability(package.destination)

                combined_score = route_score * prob_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_package = package

        # Fallback to nearest if no route package found
        if best_package is None:
            destinations = [pkg.destination for pkg in shipper.packages]
            nearest_location, _ = find_nearest_location(
                shipper.location, destinations, problem_instance.distance_matrix
            )
            best_package = next((pkg for pkg in shipper.packages
                               if pkg.destination == nearest_location), None)

        if best_package is not None:
            packages_at_dest = [pkg.id for pkg in shipper.packages
                              if pkg.destination == best_package.destination]

            return Action(
                shipper_id=shipper.id,
                next_location=best_package.destination,
                packages_to_attempt=packages_at_dest
            )

        return None

    def _select_dynamic_action(self, state: State, shipper: Shipper,
                             problem_instance: ProblemInstance) -> Optional[Action]:
        """Dynamic strategy: intelligent scoring with multiple factors."""
        if len(shipper.packages) == 0:
            return None

        best_package = None
        best_score = -np.inf

        for package in shipper.packages:
            score = self._calculate_advanced_package_score(
                package, shipper, state, problem_instance
            )

            if score > best_score:
                best_score = score
                best_package = package

        if best_package is not None:
            packages_at_dest = [pkg.id for pkg in shipper.packages
                              if pkg.destination == best_package.destination]

            return Action(
                shipper_id=shipper.id,
                next_location=best_package.destination,
                packages_to_attempt=packages_at_dest
            )

        return None

    def _select_hybrid_action(self, state: State, shipper: Shipper,
                            problem_instance: ProblemInstance) -> Optional[Action]:
        """Hybrid strategy: combine static routes with dynamic adjustments."""
        if len(shipper.packages) == 0:
            return None

        # Get static and dynamic preferences
        static_action = self._select_static_action(state, shipper, problem_instance)
        dynamic_action = self._select_dynamic_action(state, shipper, problem_instance)

        # If both agree, use that
        if (static_action and dynamic_action and
            static_action.next_location == dynamic_action.next_location):
            return static_action

        # Otherwise, score both options
        static_score = -np.inf
        dynamic_score = -np.inf

        if static_action:
            static_package = next((pkg for pkg in shipper.packages
                                 if pkg.destination == static_action.next_location), None)
            if static_package:
                static_score = self._calculate_advanced_package_score(
                    static_package, shipper, state, problem_instance
                ) * 1.1  # Slight bias towards static routes

        if dynamic_action:
            dynamic_package = next((pkg for pkg in shipper.packages
                                  if pkg.destination == dynamic_action.next_location), None)
            if dynamic_package:
                dynamic_score = self._calculate_advanced_package_score(
                    dynamic_package, shipper, state, problem_instance
                )

        # Return the better option
        return static_action if static_score >= dynamic_score else dynamic_action

    def _calculate_advanced_package_score(self, package: Package, shipper: Shipper,
                                        state: State, problem_instance: ProblemInstance) -> float:
        """Calculate advanced scoring for package selection."""
        # Distance factor (prefer closer)
        distance = problem_instance.get_distance(shipper.location, package.destination)
        max_distance = np.max(problem_instance.distance_matrix)
        distance_score = 1 - (distance / max_distance) if max_distance > 0 else 1.0

        # Probability factor
        prob_score = problem_instance.get_delivery_probability(package.destination, state.current_time)

        # Value factor
        value_score = package.value / 150.0  # Normalize

        # Urgency factor (time-based)
        time_factor = max(0, 1 - state.current_time / problem_instance.time_window)

        # Capacity efficiency (prefer packages that use capacity well)
        capacity_score = package.weight / shipper.capacity

        # Attempt penalty
        attempt_penalty = max(0.1, 1 - package.attempt_count * 0.3)

        # Multi-package bonus (if multiple packages at same location)
        location_packages = [pkg for pkg in shipper.packages if pkg.destination == package.destination]
        multi_package_bonus = 1 + 0.1 * (len(location_packages) - 1)

        # Combine all factors
        score = (self.config.distance_weight * distance_score +
                self.config.probability_weight * prob_score +
                self.config.value_weight * value_score +
                self.config.urgency_weight * time_factor +
                self.config.capacity_weight * capacity_score) * attempt_penalty * multi_package_bonus

        return score

    def _process_callbacks_advanced(self, state: State, problem_instance: ProblemInstance):
        """Advanced callback processing with intelligent prioritization."""
        callbacks_to_process = []

        # Collect ready callbacks
        while not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()
            if callback.callback_time <= state.current_time:
                callbacks_to_process.append(state.callback_queue.pop())
            else:
                break

        # Sort callbacks by advanced priority
        callbacks_to_process.sort(key=lambda c: self._calculate_callback_priority_advanced(c, state, problem_instance), reverse=True)

        for callback in callbacks_to_process:
            # Find best shipper for this callback
            best_shipper, benefit = self._find_best_shipper_for_callback_advanced(callback, state, problem_instance)

            if best_shipper and benefit > 0:
                # Accept callback
                callback.accepted = True
                callback.reattempt_time = state.current_time

                # Boost callback package value
                callback.package.value *= self.config.callback_value_multiplier

                best_shipper.add_package(callback.package)

    def _calculate_callback_priority_advanced(self, callback: Callback, state: State,
                                            problem_instance: ProblemInstance) -> float:
        """Calculate advanced priority score for callbacks."""
        # Base priority factors
        value_factor = callback.package.value / 150.0
        prob_factor = problem_instance.get_delivery_probability(callback.package.destination, state.current_time)

        # Time decay
        wait_time = state.current_time - callback.callback_time
        time_factor = np.exp(-self.config.callback_decay * wait_time)

        # Customer tier
        tier_multiplier = {"standard": 1.0, "premium": 1.5, "vip": 2.0}.get(callback.package.customer_tier, 1.0)

        # Attempt penalty
        attempt_factor = max(0.5, 1 - callback.package.attempt_count * 0.2)

        priority = value_factor * prob_factor * time_factor * tier_multiplier * attempt_factor

        return priority

    def _find_best_shipper_for_callback_advanced(self, callback: Callback, state: State,
                                               problem_instance: ProblemInstance) -> Tuple[Optional[Shipper], float]:
        """Find best shipper for callback with cost-benefit analysis."""
        best_shipper = None
        best_benefit = 0

        for shipper in state.shippers:
            if not shipper.can_carry(callback.package):
                continue

            # Calculate detour cost
            current_location = shipper.location
            callback_location = callback.package.destination

            # Estimate current route value
            current_route_value = self._estimate_remaining_route_value(shipper, state, problem_instance)

            # Calculate detour
            detour_distance = problem_instance.get_distance(current_location, callback_location)
            detour_cost = problem_instance.calculate_movement_cost(detour_distance, shipper.current_load)

            # Calculate callback value
            callback_success_prob = problem_instance.get_delivery_probability(callback_location, state.current_time)
            callback_value = callback_success_prob * problem_instance.R_callback_success

            # Net benefit
            net_benefit = callback_value - detour_cost

            # Check detour ratio constraint
            if shipper.packages:
                # Estimate total remaining route distance
                remaining_distances = []
                current_pos = current_location
                for pkg in shipper.packages:
                    dist = problem_instance.get_distance(current_pos, pkg.destination)
                    remaining_distances.append(dist)
                    current_pos = pkg.destination

                total_route_distance = sum(remaining_distances)
                detour_ratio = detour_distance / max(total_route_distance, 1)

                if detour_ratio > self.config.max_detour_ratio:
                    continue

            if net_benefit > best_benefit:
                best_benefit = net_benefit
                best_shipper = shipper

        return best_shipper, best_benefit

    def _estimate_remaining_route_value(self, shipper: Shipper, state: State,
                                      problem_instance: ProblemInstance) -> float:
        """Estimate value of shipper's remaining route."""
        total_value = 0

        for package in shipper.packages:
            prob = problem_instance.get_delivery_probability(package.destination, state.current_time)
            success_reward = problem_instance.calculate_success_reward(state.current_time)
            expected_value = prob * success_reward
            total_value += expected_value

        return total_value

    def _evaluate_strategy_performance(self, current_reward: float, iteration: int):
        """Evaluate current strategy performance and potentially switch."""
        self.performance_history.append(current_reward)

        if len(self.performance_history) >= self.config.performance_window:
            recent_performance = np.mean(self.performance_history[-self.config.performance_window:])

            # Store performance for current strategy
            self.strategy_performance[self.current_strategy].append(recent_performance)

            # Check if we should switch strategies
            if len(self.strategy_performance[self.current_strategy]) >= 3:
                current_avg = np.mean(self.strategy_performance[self.current_strategy][-3:])

                # Check other strategies
                best_strategy = self.current_strategy
                best_performance = current_avg

                for strategy, performances in self.strategy_performance.items():
                    if len(performances) >= 2:
                        strategy_avg = np.mean(performances[-2:])
                        if strategy_avg > best_performance + self.config.strategy_switch_threshold:
                            best_performance = strategy_avg
                            best_strategy = strategy

                if best_strategy != self.current_strategy:
                    self.current_strategy = best_strategy

    def _execute_action(self, state: State, shipper: Shipper, action: Action,
                       problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Execute action (same as baseline implementations)."""
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
                    package.destination, next_state.current_time, problem_instance.delivery_probabilities
                )

                package.attempt_count += 1
                package.last_attempt_time = next_state.current_time

                attempt = DeliveryAttempt(
                    package_id=package.id, shipper_id=shipper.id, location=package.destination,
                    attempt_time=next_state.current_time, successful=success, attempt_number=package.attempt_count
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

                    callback_occurs = sample_callback_occurrence(package.destination, problem_instance.callback_probabilities)
                    if callback_occurs:
                        callback_delay = sample_callback_delay()
                        callback_time = next_state.current_time + callback_delay
                        priority_score = self._calculate_callback_priority_advanced(
                            Callback(package, callback_time), next_state, problem_instance
                        )

                        callback = Callback(package=package, callback_time=callback_time, priority_score=priority_score)
                        next_state.callback_queue.add(callback)
                        next_state.callbacks.append(callback)
                        packages_to_remove.append(package)

            for package in packages_to_remove:
                next_shipper.remove_package(package)

        next_shipper.load_history.append(next_shipper.current_load)
        return next_state, immediate_reward

    def _initialize_state(self, problem_instance: ProblemInstance) -> State:
        """Initialize state (same as baselines)."""
        shippers = []
        for i in range(problem_instance.n_shippers):
            shipper = Shipper(id=i, capacity=problem_instance.shipper_capacities[i], location=0)
            shippers.append(shipper)

        packages = []
        for i in range(problem_instance.n_packages):
            package = Package(
                id=i, weight=problem_instance.package_weights[i],
                destination=problem_instance.package_destinations[i]
            )
            packages.append(package)

        self._assign_packages_optimally(shippers, packages, problem_instance)
        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _assign_packages_optimally(self, shippers: List[Shipper], packages: List[Package],
                                 problem_instance: ProblemInstance):
        """Optimal package assignment using value-density and probability."""
        # Score packages by value density and delivery probability
        package_scores = []
        for package in packages:
            prob = problem_instance.get_delivery_probability(package.destination)
            value_density = package.value / package.weight
            score = prob * value_density
            package_scores.append((package, score))

        # Sort by score (best first)
        package_scores.sort(key=lambda x: x[1], reverse=True)

        for package, score in package_scores:
            # Find best shipper (highest remaining capacity that can carry it)
            best_shipper = None
            best_capacity = -1

            for shipper in shippers:
                if shipper.can_carry(package) and shipper.remaining_capacity > best_capacity:
                    best_capacity = shipper.remaining_capacity
                    best_shipper = shipper

            if best_shipper is not None:
                best_shipper.add_package(package)

    def _is_terminal(self, state: State) -> bool:
        """Check if state is terminal."""
        return state.all_packages_processed()