"""
ECHO: Efficient Callback Handling Optimizer

Adaptive Route-based MDP with intelligent callback management
for uncertain last-mile delivery scenarios.

"Listen to the echoes, adapt to uncertainty"
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.helpers import calculate_callback_priority, find_nearest_location, validate_state_consistency
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


class ECHO:
    """
    ECHO: Efficient Callback Handling Optimizer

    Uses route-based Markov Decision Process formulation with rollout-based
    lookahead to optimize delivery routing under uncertainty with dynamic callbacks.
    """

    def __init__(self, config: Dict):
        self.gamma = config.get('discount_factor', 0.95)
        self.horizon = config.get('rollout_horizon', 3)
        self.n_samples = config.get('n_rollout_samples', 10)
        self.callback_threshold = config.get('callback_threshold', 0.5)
        self.max_callback_wait = config.get('max_callback_wait', 30.0)

        # Value function weights (can be learned)
        self.value_weights = np.array([
            1.0,   # remaining_high_value_packages
            0.8,   # average_shipper_utilization
            1.2,   # total_expected_delivery_probability
            -0.6,  # callback_queue_size
            -0.4,  # time_remaining_penalty
            -0.3,  # average_distance_to_packages
            -1.0,  # failed_delivery_count
            0.9    # capacity_efficiency
        ])

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """Main solving routine for ECHO algorithm."""
        state = self._initialize_state(problem_instance)
        total_reward = 0
        episode_history = []

        iteration = 0
        max_iterations = 1000  # Safety limit

        while not self._is_terminal(state) and iteration < max_iterations:
            # Process callbacks first
            if not state.callback_queue.is_empty():
                self._process_callbacks(state, problem_instance)

            # Make decisions for each active shipper
            active_shippers = state.get_active_shippers()

            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._select_action(state, shipper, problem_instance)
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
                location=0  # All start at depot (location 0)
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

        # Assign packages to shippers using simple greedy assignment
        self._assign_packages_to_shippers(shippers, packages)

        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _assign_packages_to_shippers(self, shippers: List[Shipper], packages: List[Package]):
        """Assign packages to shippers based on capacity constraints."""
        # Sort packages by weight (heaviest first)
        packages.sort(key=lambda p: p.weight, reverse=True)

        for package in packages:
            # Find shipper with most remaining capacity that can carry this package
            best_shipper = None
            best_capacity = -1

            for shipper in shippers:
                if shipper.can_carry(package) and shipper.remaining_capacity > best_capacity:
                    best_capacity = shipper.remaining_capacity
                    best_shipper = shipper

            if best_shipper is not None:
                best_shipper.add_package(package)
                package.assignment_time = 0.0

    def _select_action(self, state: State, shipper: Shipper, problem_instance: ProblemInstance) -> Optional[Action]:
        """Select best action using approximate value iteration with rollout."""
        feasible_actions = self._generate_feasible_actions(state, shipper, problem_instance)

        if not feasible_actions:
            return None

        best_action = None
        best_value = -np.inf

        for action in feasible_actions:
            # Estimate Q-value for this state-action pair
            q_value = self._estimate_q_value(state, shipper, action, problem_instance)

            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def _generate_feasible_actions(self, state: State, shipper: Shipper,
                                 problem_instance: ProblemInstance) -> List[Action]:
        """Generate feasible actions for a shipper."""
        actions = []

        if len(shipper.packages) == 0:
            return actions

        # Get unique destinations from shipper's packages
        destinations = list(set(pkg.destination for pkg in shipper.packages))

        for destination in destinations:
            # Create action to move to this destination and attempt delivery
            packages_at_dest = [pkg.id for pkg in shipper.packages if pkg.destination == destination]

            action = Action(
                shipper_id=shipper.id,
                next_location=destination,
                packages_to_attempt=packages_at_dest
            )
            actions.append(action)

        return actions

    def _estimate_q_value(self, state: State, shipper: Shipper, action: Action,
                         problem_instance: ProblemInstance) -> float:
        """Estimate Q(s, a) using sampling and rollout."""
        q_estimate = 0.0

        for _ in range(self.n_samples):
            # Sample outcome
            next_state, immediate_reward = self._sample_transition(state, shipper, action, problem_instance)

            # Estimate future value using rollout
            future_value = self._rollout_value(next_state, self.horizon, problem_instance)

            # Q-value estimate
            q_estimate += immediate_reward + self.gamma * future_value

        return q_estimate / self.n_samples

    def _sample_transition(self, state: State, shipper: Shipper, action: Action,
                         problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Sample a transition given state, shipper, and action."""
        next_state = state.copy()
        next_shipper = next_state.shippers[shipper.id]

        immediate_reward = 0.0

        # Movement cost
        if action.is_movement:
            distance = problem_instance.get_distance(next_shipper.location, action.next_location)
            movement_cost = problem_instance.calculate_movement_cost(distance, next_shipper.current_load)
            immediate_reward -= movement_cost
            next_state.total_cost += movement_cost

            # Update location and time
            next_shipper.location = action.next_location
            next_state.current_time += distance / 10.0  # Assume 10 units/minute speed
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

                    # Check for callback
                    callback_occurs = sample_callback_occurrence(
                        package.destination,
                        problem_instance.callback_probabilities
                    )

                    if callback_occurs:
                        callback_delay = sample_callback_delay()
                        callback_time = next_state.current_time + callback_delay

                        priority_score = calculate_callback_priority(
                            package, next_state.current_time, 0.0, next_shipper.remaining_capacity
                        )

                        callback = Callback(
                            package=package,
                            callback_time=callback_time,
                            priority_score=priority_score
                        )

                        next_state.callback_queue.add(callback)
                        next_state.callbacks.append(callback)

                        packages_to_remove.append(package)

            # Remove delivered/failed packages
            for package in packages_to_remove:
                next_shipper.remove_package(package)

        # Update load history
        next_shipper.load_history.append(next_shipper.current_load)

        return next_state, immediate_reward

    def _rollout_value(self, state: State, horizon: int, problem_instance: ProblemInstance) -> float:
        """Estimate value of state using greedy rollout policy."""
        if horizon == 0 or self._is_terminal(state):
            return self._approximate_value_function(state, problem_instance)

        simulated_state = state.copy()
        total_value = 0.0

        for h in range(horizon):
            active_shippers = simulated_state.get_active_shippers()

            if not active_shippers:
                break

            # Greedy policy: select action with highest expected immediate reward
            best_reward = 0.0
            best_next_state = simulated_state

            for shipper in active_shippers:
                actions = self._generate_feasible_actions(simulated_state, shipper, problem_instance)

                for action in actions:
                    expected_reward = self._expected_immediate_reward(simulated_state, shipper, action, problem_instance)

                    if expected_reward > best_reward:
                        best_reward = expected_reward
                        best_next_state, _ = self._expected_transition(simulated_state, shipper, action, problem_instance)

            total_value += (self.gamma ** h) * best_reward
            simulated_state = best_next_state

            if self._is_terminal(simulated_state):
                break

        return total_value

    def _expected_immediate_reward(self, state: State, shipper: Shipper, action: Action,
                                 problem_instance: ProblemInstance) -> float:
        """Calculate expected immediate reward for an action."""
        expected_reward = 0.0

        # Movement cost
        if action.is_movement:
            distance = problem_instance.get_distance(shipper.location, action.next_location)
            movement_cost = problem_instance.calculate_movement_cost(distance, shipper.current_load)
            expected_reward -= movement_cost

        # Expected delivery rewards
        if action.is_delivery_attempt:
            for package_id in action.packages_to_attempt:
                package = next((p for p in shipper.packages if p.id == package_id), None)
                if package is None:
                    continue

                delivery_prob = problem_instance.get_delivery_probability(package.destination, state.current_time)

                success_reward = problem_instance.calculate_success_reward(state.current_time)
                failure_reward = problem_instance.R_failure

                expected_reward += delivery_prob * success_reward + (1 - delivery_prob) * failure_reward

        return expected_reward

    def _expected_transition(self, state: State, shipper: Shipper, action: Action,
                           problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Calculate expected next state (deterministic version for rollout)."""
        next_state = state.copy()
        next_shipper = next_state.shippers[shipper.id]

        # Update location and time
        if action.is_movement:
            distance = problem_instance.get_distance(next_shipper.location, action.next_location)
            next_shipper.location = action.next_location
            next_state.current_time += distance / 10.0

        # Remove packages with high delivery probability
        if action.is_delivery_attempt:
            packages_to_remove = []

            for package_id in action.packages_to_attempt:
                package = next((p for p in next_shipper.packages if p.id == package_id), None)
                if package is None:
                    continue

                delivery_prob = problem_instance.get_delivery_probability(package.destination, next_state.current_time)

                # If delivery probability is high, assume success for rollout
                if delivery_prob > 0.6:
                    packages_to_remove.append(package)

            for package in packages_to_remove:
                next_shipper.remove_package(package)

        return next_state, 0.0

    def _approximate_value_function(self, state: State, problem_instance: ProblemInstance) -> float:
        """Approximate value function using learned features."""
        features = self._extract_features(state, problem_instance)
        return np.dot(self.value_weights, features)

    def _extract_features(self, state: State, problem_instance: ProblemInstance) -> np.ndarray:
        """Extract feature vector for value function approximation."""
        features = np.zeros(8)

        total_packages = sum(len(s.packages) for s in state.shippers)

        if total_packages == 0:
            return features

        # Remaining high-value packages
        high_value_packages = sum(1 for s in state.shippers for p in s.packages if p.value > 80)
        features[0] = high_value_packages / max(1, total_packages)

        # Average shipper utilization
        utilizations = [s.current_load / s.capacity for s in state.shippers if s.capacity > 0]
        features[1] = np.mean(utilizations) if utilizations else 0

        # Total expected delivery probability
        total_prob = 0
        for shipper in state.shippers:
            for package in shipper.packages:
                prob = problem_instance.get_delivery_probability(package.destination, state.current_time)
                total_prob += prob
        features[2] = total_prob / max(1, total_packages)

        # Callback queue size (normalized)
        features[3] = min(state.callback_queue.size() / 10.0, 1.0)

        # Time remaining penalty
        features[4] = min(state.current_time / problem_instance.time_window, 1.0)

        # Average distance to undelivered packages
        total_distance = 0
        count = 0
        for shipper in state.shippers:
            for package in shipper.packages:
                distance = problem_instance.get_distance(shipper.location, package.destination)
                total_distance += distance
                count += 1
        features[5] = (total_distance / max(1, count)) / 100.0  # Normalize by typical distance

        # Failed delivery count
        features[6] = len(state.failed_deliveries) / max(1, problem_instance.n_packages)

        # Capacity efficiency
        total_capacity = sum(s.capacity for s in state.shippers)
        used_capacity = sum(s.current_load for s in state.shippers)
        features[7] = used_capacity / max(1, total_capacity)

        return features

    def _process_callbacks(self, state: State, problem_instance: ProblemInstance):
        """Process pending callbacks using value-based acceptance."""
        callbacks_to_handle = []

        # Find callbacks that are ready to be handled
        while not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()
            if callback.callback_time <= state.current_time:
                callbacks_to_handle.append(state.callback_queue.pop())
            else:
                break

        for callback in callbacks_to_handle:
            # Find best shipper for this callback
            best_shipper = self._find_best_shipper_for_callback(callback, state, problem_instance)

            if best_shipper is not None:
                # Accept callback
                callback.accepted = True
                callback.reattempt_time = state.current_time

                # Add package back to shipper
                best_shipper.add_package(callback.package)

    def _find_best_shipper_for_callback(self, callback: Callback, state: State,
                                      problem_instance: ProblemInstance) -> Optional[Shipper]:
        """Find the best shipper to handle a callback."""
        best_shipper = None
        best_score = self.callback_threshold  # Minimum score to accept

        for shipper in state.shippers:
            if not shipper.can_carry(callback.package):
                continue

            # Calculate callback handling score
            distance = problem_instance.get_distance(shipper.location, callback.package.destination)
            proximity_score = max(0, 1 - distance / 50.0)

            # Current route efficiency
            current_efficiency = len(shipper.packages) / max(1, shipper.capacity / 10.0)

            # Delivery probability
            delivery_prob = problem_instance.get_delivery_probability(
                callback.package.destination, state.current_time
            )

            # Combined score
            score = (0.4 * proximity_score +
                    0.3 * delivery_prob +
                    0.2 * callback.priority_score +
                    0.1 * current_efficiency)

            if score > best_score:
                best_score = score
                best_shipper = shipper

        return best_shipper

    def _execute_action(self, state: State, shipper: Shipper, action: Action,
                       problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Execute an action and return the resulting state and reward."""
        # Use the sample transition function with a fixed seed for consistency
        return self._sample_transition(state, shipper, action, problem_instance)

    def _is_terminal(self, state: State) -> bool:
        """Check if the state is terminal."""
        return state.all_packages_processed()