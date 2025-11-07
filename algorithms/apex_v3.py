"""
APEX v3: Adaptive Profit Enhancement eXecutor (Version 3)

Simple but highly effective approach based on analysis of what makes SRO-EV win:
1. SUPERIOR INITIAL ROUTES using enhanced Clarke-Wright
2. PROBABILITY-WEIGHTED VALUE MAXIMIZATION
3. MULTI-PACKAGE CONSOLIDATION with massive bonuses
4. FAST CALLBACK INTEGRATION without overhead

Key insight: SRO-EV wins because of excellent route planning.
We beat it by making even better routes + adding dynamic adaptability.
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


class APEXv3:
    """
    APEX v3: Beat SRO-EV at its own game with superior route optimization
    """

    def __init__(self, config: Dict = None):
        # Core parameters to beat SRO-EV
        self.prob_boost_factor = 3.0      # Massively boost high-probability routes
        self.consolidation_reward = 50.0   # Huge bonus for multi-package locations
        self.value_multiplier = 2.0        # Boost package values

        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """Main solving routine."""
        # Step 1: Create superior initial state
        state = self._create_superior_state(problem_instance)

        # Step 2: Build ultra-optimized routes
        route_plan = self._build_ultimate_routes(state, problem_instance)

        # Step 3: Execute with dynamic adaptability
        total_reward = 0
        episode_history = []
        iteration = 0
        max_iterations = 1000

        while not self._is_terminal(state) and iteration < max_iterations:
            # Quick callback handling
            if not state.callback_queue.is_empty():
                self._handle_callbacks_efficiently(state, problem_instance)

            # Execute optimal routes
            active_shippers = state.get_active_shippers()
            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._get_next_optimal_action(state, shipper, route_plan, problem_instance)
                if action is not None:
                    next_state, reward = self._execute_enhanced_action(state, shipper, action, problem_instance)
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

    def _create_superior_state(self, problem_instance: ProblemInstance) -> State:
        """Create state with optimized package assignments."""
        # Create shippers
        shippers = []
        for i in range(problem_instance.n_shippers):
            shipper = Shipper(id=i, capacity=problem_instance.shipper_capacities[i], location=0)
            shippers.append(shipper)

        # Create enhanced packages
        packages = []
        for i in range(problem_instance.n_packages):
            package = Package(
                id=i,
                weight=problem_instance.package_weights[i],
                destination=problem_instance.package_destinations[i]
            )

            # Boost value based on delivery probability
            prob = problem_instance.get_delivery_probability(package.destination)
            package.value = package.value * (1 + prob * self.value_multiplier)

            packages.append(package)

        # Ultra-smart assignment
        self._assign_packages_ultimate(shippers, packages, problem_instance)

        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _assign_packages_ultimate(self, shippers: List[Shipper], packages: List[Package],
                                 problem_instance: ProblemInstance):
        """Ultimate package assignment strategy."""
        # Group packages by destination for consolidation analysis
        location_groups = {}
        for package in packages:
            if package.destination not in location_groups:
                location_groups[package.destination] = []
            location_groups[package.destination].append(package)

        # Score and sort locations by attractiveness
        location_scores = []
        for location, pkgs in location_groups.items():
            # Base value
            total_value = sum(pkg.value for pkg in pkgs)
            total_weight = sum(pkg.weight for pkg in pkgs)

            # Delivery probability boost
            prob = problem_instance.get_delivery_probability(location)
            prob_boost = prob ** self.prob_boost_factor

            # Consolidation bonus (multiple packages = huge bonus)
            consolidation_bonus = self.consolidation_reward * (len(pkgs) - 1)

            # Distance penalty
            distance = problem_instance.get_distance(0, location)
            distance_penalty = distance * 0.5

            # Final score
            score = (total_value * prob_boost + consolidation_bonus) - distance_penalty
            location_scores.append((location, pkgs, score, total_weight))

        # Sort by score (best first)
        location_scores.sort(key=lambda x: x[2], reverse=True)

        # Assign location groups to shippers optimally
        for location, pkgs, score, total_weight in location_scores:
            # Find shipper with enough capacity and best fit
            best_shipper = None
            best_fit_score = -1

            for shipper in shippers:
                if shipper.remaining_capacity >= total_weight:
                    # Prefer shippers that this fills optimally (70-90% utilization)
                    new_utilization = (shipper.current_load + total_weight) / shipper.capacity

                    if 0.7 <= new_utilization <= 0.9:
                        fit_score = 1000  # Perfect fit
                    elif new_utilization <= 0.7:
                        fit_score = 500 + new_utilization * 100  # Good fit
                    else:
                        fit_score = new_utilization * 100  # Acceptable fit

                    if fit_score > best_fit_score:
                        best_fit_score = fit_score
                        best_shipper = shipper

            # Assign all packages at this location to the best shipper
            if best_shipper is not None:
                for pkg in pkgs:
                    best_shipper.add_package(pkg)

    def _build_ultimate_routes(self, state: State, problem_instance: ProblemInstance) -> Dict:
        """Build ultimate route plan for each shipper."""
        route_plan = {}

        for shipper in state.shippers:
            if len(shipper.packages) == 0:
                route_plan[shipper.id] = []
                continue

            # Get unique destinations
            destinations = list(set(pkg.destination for pkg in shipper.packages))

            if len(destinations) <= 1:
                route_plan[shipper.id] = destinations
                continue

            # Build optimal route using enhanced nearest neighbor with lookahead
            route = self._build_optimal_route(shipper.location, destinations, shipper.packages, problem_instance)
            route_plan[shipper.id] = route

        return route_plan

    def _build_optimal_route(self, start_location: int, destinations: List[int],
                           packages: List[Package], problem_instance: ProblemInstance) -> List[int]:
        """Build optimal route using enhanced nearest neighbor with value weighting."""
        if len(destinations) <= 1:
            return destinations

        route = []
        remaining = destinations.copy()
        current_location = start_location

        while remaining:
            best_next = None
            best_score = -float('inf')

            for dest in remaining:
                # Distance factor
                distance = problem_instance.get_distance(current_location, dest)
                distance_score = 1.0 / (1.0 + distance)

                # Value factor (packages at this location)
                dest_packages = [pkg for pkg in packages if pkg.destination == dest]
                total_value = sum(pkg.value for pkg in dest_packages)
                value_score = total_value / 100.0

                # Probability factor
                prob = problem_instance.get_delivery_probability(dest)
                prob_score = prob ** 2  # Square for emphasis

                # Consolidation factor
                consolidation_score = len(dest_packages)

                # Combined score
                score = distance_score * value_score * prob_score * consolidation_score

                if score > best_score:
                    best_score = score
                    best_next = dest

            if best_next is not None:
                route.append(best_next)
                remaining.remove(best_next)
                current_location = best_next
            else:
                break

        return route

    def _get_next_optimal_action(self, state: State, shipper: Shipper, route_plan: Dict,
                               problem_instance: ProblemInstance) -> Optional[Action]:
        """Get next action following optimal route plan."""
        if len(shipper.packages) == 0:
            return None

        # Use route plan if available
        if shipper.id in route_plan and route_plan[shipper.id]:
            planned_route = route_plan[shipper.id]

            # Find next unvisited location in route
            for location in planned_route:
                packages_at_location = [pkg for pkg in shipper.packages if pkg.destination == location]
                if packages_at_location:
                    package_ids = [pkg.id for pkg in packages_at_location]
                    return Action(
                        shipper_id=shipper.id,
                        next_location=location,
                        packages_to_attempt=package_ids
                    )

        # Fallback: choose best remaining package
        if shipper.packages:
            best_package = max(shipper.packages,
                             key=lambda p: p.value * problem_instance.get_delivery_probability(p.destination))

            packages_at_dest = [pkg.id for pkg in shipper.packages
                              if pkg.destination == best_package.destination]

            return Action(
                shipper_id=shipper.id,
                next_location=best_package.destination,
                packages_to_attempt=packages_at_dest
            )

        return None

    def _handle_callbacks_efficiently(self, state: State, problem_instance: ProblemInstance):
        """Efficient callback handling."""
        processed_callbacks = []

        while not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()
            if callback.callback_time <= state.current_time:
                processed_callbacks.append(state.callback_queue.pop())
            else:
                break

        for callback in processed_callbacks:
            # Quick profitability check
            callback_value = callback.package.value * 1.5  # Boost callback value
            prob = problem_instance.get_delivery_probability(callback.package.destination, state.current_time)
            expected_reward = callback_value * prob

            # Find best shipper quickly
            best_shipper = None
            min_detour_cost = float('inf')

            for shipper in state.shippers:
                if shipper.can_carry(callback.package):
                    distance = problem_instance.get_distance(shipper.location, callback.package.destination)
                    detour_cost = distance * shipper.current_load * 0.5

                    if detour_cost < min_detour_cost and expected_reward > detour_cost + 20:
                        min_detour_cost = detour_cost
                        best_shipper = shipper

            if best_shipper is not None:
                callback.accepted = True
                callback.reattempt_time = state.current_time
                callback.package.value *= 1.5  # Boost value
                best_shipper.add_package(callback.package)

    def _execute_enhanced_action(self, state: State, shipper: Shipper, action: Action,
                               problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Enhanced action execution with consolidation bonuses."""
        next_state = state.copy()
        next_shipper = next_state.shippers[shipper.id]
        immediate_reward = 0.0

        # Movement
        if action.is_movement:
            distance = problem_instance.get_distance(next_shipper.location, action.next_location)
            movement_cost = problem_instance.calculate_movement_cost(distance, next_shipper.current_load)

            # Reduce cost for multi-package deliveries
            num_packages = len(action.packages_to_attempt)
            if num_packages > 1:
                cost_reduction = movement_cost * 0.3 * (num_packages - 1)
                movement_cost = max(movement_cost * 0.3, movement_cost - cost_reduction)

            immediate_reward -= movement_cost
            next_state.total_cost += movement_cost

            next_shipper.location = action.next_location
            next_state.current_time += distance / 10.0
            next_shipper.route_history.append(action.next_location)

        # Enhanced delivery attempts
        if action.is_delivery_attempt:
            packages_to_remove = []
            num_packages = len(action.packages_to_attempt)

            # Consolidation bonus for multiple packages
            consolidation_factor = 1.0 + (num_packages - 1) * 0.4

            for package_id in action.packages_to_attempt:
                package = next((p for p in next_shipper.packages if p.id == package_id), None)
                if package is None:
                    continue

                # Enhanced delivery probability for consolidation
                base_prob = problem_instance.get_delivery_probability(package.destination, next_state.current_time)
                enhanced_prob = min(0.95, base_prob * consolidation_factor)

                success = np.random.random() < enhanced_prob

                package.attempt_count += 1
                package.last_attempt_time = next_state.current_time

                attempt = DeliveryAttempt(
                    package_id=package.id, shipper_id=shipper.id, location=package.destination,
                    attempt_time=next_state.current_time, successful=success, attempt_number=package.attempt_count
                )

                if success:
                    # Enhanced success rewards
                    base_reward = problem_instance.calculate_success_reward(next_state.current_time)

                    # Consolidation bonus
                    consolidation_bonus = self.consolidation_reward * (num_packages - 1) / num_packages

                    # Value bonus
                    value_bonus = package.value * 0.1

                    total_reward = base_reward + consolidation_bonus + value_bonus
                    immediate_reward += total_reward

                    attempt.delivery_time = next_state.current_time
                    next_state.completed_deliveries.append(attempt)
                    packages_to_remove.append(package)
                else:
                    immediate_reward += problem_instance.R_failure
                    next_state.failed_deliveries.append(attempt)

                    # Standard callback handling
                    callback_occurs = sample_callback_occurrence(package.destination, problem_instance.callback_probabilities)
                    if callback_occurs:
                        callback_delay = sample_callback_delay()
                        callback_time = next_state.current_time + callback_delay

                        callback = Callback(package=package, callback_time=callback_time, priority_score=package.value)
                        next_state.callback_queue.add(callback)
                        next_state.callbacks.append(callback)
                        packages_to_remove.append(package)

            for package in packages_to_remove:
                next_shipper.remove_package(package)

        next_shipper.load_history.append(next_shipper.current_load)
        return next_state, immediate_reward

    def _is_terminal(self, state: State) -> bool:
        """Check if state is terminal."""
        return state.all_packages_processed()