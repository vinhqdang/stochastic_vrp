"""
DRL-DU-Simplified: Deep RL for Dynamic and Uncertain VRP (Simplified Implementation)

Key innovation: Belief state tracking with dynamic replanning without requiring
full POMDP solver. Captures DRL-DU's core benefit of adapting to uncertainty
while remaining computationally efficient.
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, Action, ProblemInstance, Package, Shipper, Callback, DeliveryAttempt
from utils.helpers import find_nearest_location
from utils.probability import sample_delivery_outcome, sample_callback_occurrence, sample_callback_delay


@dataclass
class BeliefState:
    """Simplified belief state for tracking uncertainty."""
    expected_demands: Dict[int, float]  # location -> expected demand
    demand_confidence: Dict[int, float]  # location -> confidence (0-1)
    callback_predictions: Dict[int, float]  # location -> callback probability
    dynamic_events: List[Dict]  # List of recent events


class DRLDUSimplified:
    """
    DRL-DU-Simplified: Dynamic and Uncertain VRP with Belief State Tracking

    Captures DRL-DU's key insight of maintaining belief states and adapting
    to revealed information, but uses efficient heuristics instead of neural networks.
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = {}

        # Belief update parameters
        self.belief_decay = config.get('belief_decay', 0.1)  # How fast beliefs fade
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.replan_threshold = config.get('replan_threshold', 0.3)

        # Dynamic adaptation weights
        self.uncertainty_weight = config.get('uncertainty_weight', 0.3)
        self.confidence_weight = config.get('confidence_weight', 0.2)
        self.value_weight = config.get('value_weight', 0.25)
        self.distance_weight = config.get('distance_weight', 0.25)

        # Callback handling
        self.callback_value_threshold = config.get('callback_value_threshold', 50.0)

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """
        Main DRL-DU solving routine with belief state tracking.
        """
        # Initialize state and belief
        state = self._initialize_state(problem_instance)
        belief_state = self._initialize_belief_state(problem_instance)

        total_reward = 0
        episode_history = []
        iteration = 0
        max_iterations = 1000
        replanning_events = 0

        while not self._is_terminal(state) and iteration < max_iterations:
            # Update beliefs based on current information
            belief_state = self._update_belief_state(belief_state, state, problem_instance)

            # Check if major replanning needed
            if self._should_replan(belief_state, state):
                state = self._dynamic_replan(state, belief_state, problem_instance)
                replanning_events += 1

            # Process callbacks with belief-informed decisions
            if not state.callback_queue.is_empty():
                self._process_callbacks_belief_based(state, belief_state, problem_instance)

            # Execute actions with uncertainty awareness
            active_shippers = state.get_active_shippers()
            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._select_action_belief_based(state, shipper, belief_state, problem_instance)
                if action is not None:
                    # Execute and update beliefs with revealed information
                    next_state, reward = self._execute_action_with_learning(
                        state, shipper, action, belief_state, problem_instance
                    )
                    total_reward += reward
                    episode_history.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
                    state = next_state

            iteration += 1

        return {
            'total_reward': total_reward,
            'state': state,
            'history': episode_history,
            'iterations': iteration,
            'replanning_events': replanning_events,
            'final_confidence': np.mean(list(belief_state.demand_confidence.values())) if belief_state.demand_confidence else 0
        }

    def _initialize_state(self, problem_instance: ProblemInstance) -> State:
        """Initialize state (similar to other algorithms)."""
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

        # Assign packages using uncertainty-aware assignment
        self._assign_packages_uncertainty_aware(shippers, packages, problem_instance)

        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _initialize_belief_state(self, problem_instance: ProblemInstance) -> BeliefState:
        """Initialize belief state with initial uncertainty."""
        expected_demands = {}
        demand_confidence = {}
        callback_predictions = {}

        for i in range(problem_instance.n_locations):
            # Initial beliefs based on problem instance
            expected_demands[i] = 0.5  # Neutral expectation
            demand_confidence[i] = 0.3  # Low initial confidence
            callback_predictions[i] = problem_instance.callback_probabilities[i] if i < len(problem_instance.callback_probabilities) else 0.5

        return BeliefState(
            expected_demands=expected_demands,
            demand_confidence=demand_confidence,
            callback_predictions=callback_predictions,
            dynamic_events=[]
        )

    def _assign_packages_uncertainty_aware(self, shippers: List[Shipper], packages: List[Package],
                                         problem_instance: ProblemInstance):
        """Assign packages considering uncertainty."""
        # Score packages by expected success under uncertainty
        package_scores = []
        for package in packages:
            # Base delivery probability
            prob = problem_instance.get_delivery_probability(package.destination)

            # Uncertainty factor (prefer more certain outcomes)
            uncertainty_bonus = prob * 1.5  # Boost high-probability packages more

            # Value factor
            value_factor = package.value / max(package.weight, 0.1)

            score = prob + uncertainty_bonus + value_factor * 0.01
            package_scores.append((package, score))

        # Sort by uncertainty-adjusted score
        package_scores.sort(key=lambda x: x[1], reverse=True)

        # Greedy assignment to shippers
        for package, score in package_scores:
            best_shipper = None
            best_capacity = -1

            for shipper in shippers:
                if shipper.can_carry(package) and shipper.remaining_capacity > best_capacity:
                    best_capacity = shipper.remaining_capacity
                    best_shipper = shipper

            if best_shipper is not None:
                best_shipper.add_package(package)

    def _update_belief_state(self, belief_state: BeliefState, state: State,
                           problem_instance: ProblemInstance) -> BeliefState:
        """Update beliefs based on observed outcomes (Bayesian-like update)."""
        # Decay old beliefs
        for location in belief_state.demand_confidence:
            belief_state.demand_confidence[location] *= (1 - self.belief_decay)

        # Update based on recent delivery outcomes
        for delivery in state.completed_deliveries[-5:]:  # Last 5 deliveries
            location = delivery.location
            if location in belief_state.expected_demands:
                # Successful delivery increases confidence in that location
                if delivery.successful:
                    belief_state.expected_demands[location] = min(1.0, belief_state.expected_demands[location] + 0.2)
                    belief_state.demand_confidence[location] = min(1.0, belief_state.demand_confidence[location] + 0.1)

        # Update based on failures
        for failure in state.failed_deliveries[-5:]:  # Last 5 failures
            location = failure.location
            if location in belief_state.expected_demands:
                # Failed delivery decreases expected success
                belief_state.expected_demands[location] = max(0.0, belief_state.expected_demands[location] - 0.1)
                belief_state.demand_confidence[location] = min(1.0, belief_state.demand_confidence[location] + 0.05)

        # Update callback predictions based on observed callbacks
        recent_callbacks = [cb for cb in state.callbacks if state.current_time - cb.callback_time < 30]
        callback_locations = {}
        for cb in recent_callbacks:
            loc = cb.package.destination
            callback_locations[loc] = callback_locations.get(loc, 0) + 1

        for loc, count in callback_locations.items():
            if loc in belief_state.callback_predictions:
                # Increase callback prediction for locations with recent callbacks
                belief_state.callback_predictions[loc] = min(1.0, belief_state.callback_predictions[loc] + count * 0.1)

        return belief_state

    def _should_replan(self, belief_state: BeliefState, state: State) -> bool:
        """Decide if major replanning is needed based on belief changes."""
        # Check if confidence has dropped significantly
        avg_confidence = np.mean(list(belief_state.demand_confidence.values()))

        if avg_confidence < self.replan_threshold:
            return True

        # Check for significant callback accumulation
        callback_queue_size = state.callback_queue.size()
        if callback_queue_size > 3:  # Many pending callbacks
            return True

        return False

    def _dynamic_replan(self, state: State, belief_state: BeliefState,
                       problem_instance: ProblemInstance) -> State:
        """Perform dynamic replanning based on current beliefs."""
        # Reassign packages based on updated beliefs
        all_packages = []
        for shipper in state.shippers:
            all_packages.extend(shipper.packages)
            shipper.packages.clear()
            shipper.current_load = 0.0

        # Re-assign using updated belief information
        package_scores = []
        for package in all_packages:
            loc = package.destination

            # Use belief state for scoring
            expected_success = belief_state.expected_demands.get(loc, 0.5)
            confidence = belief_state.demand_confidence.get(loc, 0.3)

            # Higher score for high-confidence, high-success locations
            score = expected_success * confidence + package.value * 0.01

            package_scores.append((package, score))

        package_scores.sort(key=lambda x: x[1], reverse=True)

        # Reassign to shippers
        for package, score in package_scores:
            best_shipper = None
            best_fit = -1

            for shipper in state.shippers:
                if shipper.can_carry(package):
                    # Prefer shippers with good utilization
                    utilization = (shipper.current_load + package.weight) / shipper.capacity
                    if 0.3 <= utilization <= 0.9:
                        fit_score = utilization
                    else:
                        fit_score = 0.5

                    if fit_score > best_fit:
                        best_fit = fit_score
                        best_shipper = shipper

            if best_shipper is not None:
                best_shipper.add_package(package)

        return state

    def _select_action_belief_based(self, state: State, shipper: Shipper, belief_state: BeliefState,
                                   problem_instance: ProblemInstance) -> Optional[Action]:
        """Select action incorporating belief state information."""
        if len(shipper.packages) == 0:
            return None

        best_package = None
        best_score = -float('inf')

        for package in shipper.packages:
            score = self._calculate_belief_score(package, shipper, state, belief_state, problem_instance)

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

    def _calculate_belief_score(self, package: Package, shipper: Shipper, state: State,
                               belief_state: BeliefState, problem_instance: ProblemInstance) -> float:
        """Calculate package score using belief state."""
        location = package.destination

        # Belief-based probability
        expected_success = belief_state.expected_demands.get(location, 0.5)
        confidence = belief_state.demand_confidence.get(location, 0.3)

        # Distance factor
        distance = problem_instance.get_distance(shipper.location, location)
        max_distance = np.max(problem_instance.distance_matrix) if problem_instance.distance_matrix.size > 0 else 1.0
        distance_score = 1.0 - (distance / max_distance)

        # Value factor
        value_score = package.value / 150.0

        # Uncertainty penalty (prefer high-confidence decisions)
        uncertainty_penalty = 1.0 - confidence

        # Combined belief-based score
        score = (self.uncertainty_weight * expected_success +
                self.confidence_weight * confidence +
                self.distance_weight * distance_score +
                self.value_weight * value_score -
                0.1 * uncertainty_penalty)

        return score

    def _process_callbacks_belief_based(self, state: State, belief_state: BeliefState,
                                       problem_instance: ProblemInstance):
        """Process callbacks using belief state for better decisions."""
        callbacks_to_process = []

        while not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()
            if callback.callback_time <= state.current_time:
                callbacks_to_process.append(state.callback_queue.pop())
            else:
                break

        for callback in callbacks_to_process:
            # Use belief state to evaluate callback value
            location = callback.package.destination
            expected_success = belief_state.expected_demands.get(location, 0.5)
            confidence = belief_state.demand_confidence.get(location, 0.3)

            # Callback value considering uncertainty
            callback_value = (callback.package.value * expected_success * confidence +
                            problem_instance.R_callback_success * expected_success)

            if callback_value > self.callback_value_threshold:
                # Find best shipper considering belief state
                best_shipper = self._find_best_shipper_belief_based(
                    callback, state, belief_state, problem_instance
                )

                if best_shipper is not None:
                    callback.accepted = True
                    callback.reattempt_time = state.current_time
                    callback.package.value *= 1.4  # Boost callback value
                    best_shipper.add_package(callback.package)

                    # Update belief: accepting callback shows confidence in location
                    belief_state.demand_confidence[location] = min(1.0, confidence + 0.1)

    def _find_best_shipper_belief_based(self, callback: Callback, state: State,
                                       belief_state: BeliefState,
                                       problem_instance: ProblemInstance) -> Optional[Shipper]:
        """Find best shipper for callback using belief information."""
        best_shipper = None
        best_expected_value = 0

        location = callback.package.destination
        expected_success = belief_state.expected_demands.get(location, 0.5)

        for shipper in state.shippers:
            if shipper.can_carry(callback.package):
                # Calculate expected net value
                distance = problem_instance.get_distance(shipper.location, location)
                detour_cost = problem_instance.calculate_movement_cost(distance, shipper.current_load)

                expected_reward = expected_success * problem_instance.R_callback_success
                net_value = expected_reward - detour_cost

                if net_value > best_expected_value:
                    best_expected_value = net_value
                    best_shipper = shipper

        return best_shipper

    def _execute_action_with_learning(self, state: State, shipper: Shipper, action: Action,
                                    belief_state: BeliefState,
                                    problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Execute action and update beliefs with revealed information."""
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

        # Delivery attempts with belief updates
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

                # Update beliefs based on outcome
                location = package.destination
                if success:
                    reward = problem_instance.calculate_success_reward(next_state.current_time)
                    immediate_reward += reward
                    attempt.delivery_time = next_state.current_time
                    next_state.completed_deliveries.append(attempt)
                    packages_to_remove.append(package)

                    # Successful delivery increases belief in location
                    if location in belief_state.expected_demands:
                        belief_state.expected_demands[location] = min(1.0, belief_state.expected_demands[location] + 0.1)
                        belief_state.demand_confidence[location] = min(1.0, belief_state.demand_confidence[location] + 0.05)

                else:
                    immediate_reward += problem_instance.R_failure
                    next_state.failed_deliveries.append(attempt)

                    # Failed delivery decreases belief in location
                    if location in belief_state.expected_demands:
                        belief_state.expected_demands[location] = max(0.0, belief_state.expected_demands[location] - 0.05)
                        belief_state.demand_confidence[location] = min(1.0, belief_state.demand_confidence[location] + 0.02)

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