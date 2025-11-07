"""
SRO-EV: Static Route Optimization with Expected Values

Constructs optimal routes upfront using expected delivery probabilities,
then follows static routes with minimal dynamic adjustments for callbacks.
Uses Clarke-Wright savings algorithm adapted for stochastic demands.
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.helpers import find_best_insertion_position
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


class SRO_EV:
    """
    Static Route Optimization with Expected Values

    Uses Clarke-Wright savings algorithm to construct initial routes
    based on expected delivery probabilities, then executes with
    minimal dynamic adjustments.
    """

    def __init__(self, config: Dict = None):
        self.insertion_threshold = config.get('insertion_threshold', 20.0) if config else 20.0

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """Main solving routine for SRO-EV algorithm."""
        state = self._initialize_state(problem_instance)

        # Phase 1: Construct initial routes using Clarke-Wright
        self._construct_initial_routes(state, problem_instance)

        # Phase 2: Execute routes with callback handling
        total_reward = 0
        episode_history = []

        iteration = 0
        max_iterations = 1000

        while not self._is_terminal(state) and iteration < max_iterations:
            active_shippers = state.get_active_shippers()

            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._get_next_route_action(state, shipper, problem_instance)
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

        return State(
            shippers=shippers,
            pending_deliveries=packages,
            current_time=0.0,
            total_cost=0.0
        )

    def _construct_initial_routes(self, state: State, problem_instance: ProblemInstance):
        """Construct initial routes using Clarke-Wright savings algorithm."""
        # Calculate savings matrix
        savings_matrix = self._compute_savings_matrix(state, problem_instance)

        # Sort savings in descending order
        savings_list = []
        n_locations = len(problem_instance.locations)

        for i in range(1, n_locations):
            for j in range(i + 1, n_locations):
                if savings_matrix[i, j] > 0:
                    savings_list.append((i, j, savings_matrix[i, j]))

        savings_list.sort(key=lambda x: x[2], reverse=True)

        # Initialize individual routes for each package
        location_packages = {}
        for package in state.pending_deliveries:
            if package.destination not in location_packages:
                location_packages[package.destination] = []
            location_packages[package.destination].append(package)

        routes = {}  # location -> route (list of locations)
        for location in location_packages:
            routes[location] = [location]

        # Merge routes based on savings
        for i, j, saving in savings_list:
            route_i = self._find_route_containing(i, routes)
            route_j = self._find_route_containing(j, routes)

            if route_i is not None and route_j is not None and route_i != route_j:
                # Check if routes can be merged (capacity constraint)
                combined_weight = self._calculate_route_weight(route_i, location_packages) + \
                                self._calculate_route_weight(route_j, location_packages)

                # Find shipper with enough capacity
                suitable_shipper = None
                for shipper in state.shippers:
                    if shipper.capacity >= combined_weight and len(shipper.packages) == 0:
                        suitable_shipper = shipper
                        break

                if suitable_shipper is not None:
                    # Merge routes
                    merged_route = self._merge_routes(route_i, route_j, i, j)

                    # Remove old routes
                    del routes[route_i[0]]
                    del routes[route_j[0]]

                    # Add merged route
                    routes[merged_route[0]] = merged_route

                    # Assign packages to shipper
                    for location in merged_route:
                        if location in location_packages:
                            for package in location_packages[location]:
                                suitable_shipper.add_package(package)

        # Assign remaining unassigned packages
        for location, packages in location_packages.items():
            if not any(pkg in shipper.packages for shipper in state.shippers for pkg in packages):
                # Find shipper with most remaining capacity
                best_shipper = None
                best_capacity = -1

                total_weight = sum(pkg.weight for pkg in packages)

                for shipper in state.shippers:
                    if shipper.remaining_capacity >= total_weight and shipper.remaining_capacity > best_capacity:
                        best_capacity = shipper.remaining_capacity
                        best_shipper = shipper

                if best_shipper is not None:
                    for package in packages:
                        best_shipper.add_package(package)

        # Clear pending deliveries
        state.pending_deliveries.clear()

    def _compute_savings_matrix(self, state: State, problem_instance: ProblemInstance) -> np.ndarray:
        """Compute Clarke-Wright savings matrix adjusted for expected probabilities."""
        n_locations = len(problem_instance.locations)
        savings = np.zeros((n_locations, n_locations))

        for i in range(1, n_locations):
            for j in range(i + 1, n_locations):
                # Classical Clarke-Wright savings
                distance_saving = (problem_instance.distance_matrix[0, i] +
                                 problem_instance.distance_matrix[0, j] -
                                 problem_instance.distance_matrix[i, j])

                # Adjust by expected delivery probabilities
                expected_prob_i = problem_instance.get_delivery_probability(i)
                expected_prob_j = problem_instance.get_delivery_probability(j)
                prob_factor = (expected_prob_i + expected_prob_j) / 2

                savings[i, j] = distance_saving * prob_factor

        return savings

    def _find_route_containing(self, location: int, routes: Dict) -> Optional[List[int]]:
        """Find the route that contains a given location."""
        for route in routes.values():
            if location in route:
                return route
        return None

    def _calculate_route_weight(self, route: List[int], location_packages: Dict) -> float:
        """Calculate total weight of packages in a route."""
        total_weight = 0.0
        for location in route:
            if location in location_packages:
                total_weight += sum(pkg.weight for pkg in location_packages[location])
        return total_weight

    def _merge_routes(self, route1: List[int], route2: List[int], connect_i: int, connect_j: int) -> List[int]:
        """Merge two routes by connecting them at specified locations."""
        # Simple merge: connect routes at the connecting points
        merged = route1.copy()

        # Find positions of connecting points
        if connect_j in route2:
            idx = route2.index(connect_j)
            # Add route2 from connect_j onwards
            for k in range(idx, len(route2)):
                if route2[k] not in merged:
                    merged.append(route2[k])
            # Add route2 before connect_j
            for k in range(idx):
                if route2[k] not in merged:
                    merged.append(route2[k])

        return merged

    def _get_next_route_action(self, state: State, shipper: Shipper,
                              problem_instance: ProblemInstance) -> Optional[Action]:
        """Get next action following the static route with callback insertion."""
        if len(shipper.packages) == 0:
            return None

        # Handle callbacks first with simple insertion
        if not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()
            if (callback.callback_time <= state.current_time and
                shipper.can_carry(callback.package)):

                # Check if insertion cost is acceptable
                distance = problem_instance.get_distance(shipper.location, callback.package.destination)
                insertion_cost = problem_instance.calculate_movement_cost(distance, shipper.current_load)

                if insertion_cost < self.insertion_threshold:
                    state.callback_queue.pop()
                    callback.accepted = True
                    callback.reattempt_time = state.current_time
                    shipper.add_package(callback.package)

        # Select next package using nearest neighbor from current location
        nearest_package = None
        min_distance = float('inf')

        for package in shipper.packages:
            distance = problem_instance.get_distance(shipper.location, package.destination)
            if distance < min_distance:
                min_distance = distance
                nearest_package = package

        if nearest_package is not None:
            # Get all packages at this destination
            packages_at_dest = [pkg.id for pkg in shipper.packages
                              if pkg.destination == nearest_package.destination]

            return Action(
                shipper_id=shipper.id,
                next_location=nearest_package.destination,
                packages_to_attempt=packages_at_dest
            )

        return None

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

                    # Handle callback with simple re-insertion
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
                            priority_score=callback_time
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