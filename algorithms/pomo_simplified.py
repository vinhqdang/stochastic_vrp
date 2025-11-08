"""
POMO-Simplified: Policy Optimization with Multiple Optima (Simplified Implementation)

Key innovation: Multiple starting points for exploration without requiring
full Transformer architecture. Captures POMO's core benefit of diverse
solution generation while remaining computationally efficient.
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


class POMOSimplified:
    """
    POMO-Simplified: Multiple Optima Policy Optimization

    Captures POMO's key insight of generating multiple solutions from different
    starting points, but uses efficient greedy policy instead of neural networks.
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = {}

        # POMO's key parameter: number of different starting solutions
        self.num_starts = config.get('num_starts', 20)  # Reduced from 50 for speed

        # Attention-like scoring weights
        self.distance_weight = config.get('distance_weight', 0.3)
        self.probability_weight = config.get('probability_weight', 0.4)
        self.value_weight = config.get('value_weight', 0.2)
        self.consolidation_weight = config.get('consolidation_weight', 0.1)

        # Callback handling
        self.callback_threshold = config.get('callback_threshold', 0.4)

    def solve(self, problem_instance: ProblemInstance) -> Dict:
        """
        POMO main solving routine: Generate multiple solutions and return best.
        """
        best_solution = None
        best_reward = -float('inf')
        all_solutions = []

        # Generate multiple solutions from different starting approaches
        for start_idx in range(self.num_starts):
            try:
                # Each iteration uses different strategy/randomization
                solution = self._generate_solution(problem_instance, start_idx)

                if solution['total_reward'] > best_reward:
                    best_reward = solution['total_reward']
                    best_solution = solution

                all_solutions.append(solution)

            except Exception as e:
                # Continue with other starts if one fails
                continue

        # POMO's aggregation: use best solution but track diversity
        return {
            'total_reward': best_reward,
            'state': best_solution['state'] if best_solution else None,
            'history': best_solution['history'] if best_solution else [],
            'iterations': best_solution['iterations'] if best_solution else 0,
            'num_solutions_generated': len(all_solutions),
            'solution_diversity': self._calculate_diversity(all_solutions)
        }

    def _generate_solution(self, problem_instance: ProblemInstance, start_idx: int) -> Dict:
        """Generate single solution using POMO-style diverse exploration."""
        # Initialize state with variation based on start_idx
        state = self._initialize_state_varied(problem_instance, start_idx)

        total_reward = 0
        episode_history = []
        iteration = 0
        max_iterations = 1000

        while not self._is_terminal(state) and iteration < max_iterations:
            # Process callbacks with POMO-style value estimation
            if not state.callback_queue.is_empty():
                self._process_callbacks_pomo(state, problem_instance)

            # Multi-optima action selection
            active_shippers = state.get_active_shippers()
            if not active_shippers:
                break

            for shipper in active_shippers:
                action = self._select_action_pomo(state, shipper, problem_instance, start_idx)
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

    def _initialize_state_varied(self, problem_instance: ProblemInstance, start_idx: int) -> State:
        """Initialize state with variation based on start index (POMO's multiple optima)."""
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

        # POMO variation: different assignment strategies based on start_idx
        assignment_strategy = start_idx % 4

        if assignment_strategy == 0:
            # Strategy 1: Value-density priority
            self._assign_by_value_density(shippers, packages, problem_instance)
        elif assignment_strategy == 1:
            # Strategy 2: Probability-first priority
            self._assign_by_probability(shippers, packages, problem_instance)
        elif assignment_strategy == 2:
            # Strategy 3: Distance-based clustering
            self._assign_by_distance_clustering(shippers, packages, problem_instance)
        else:
            # Strategy 4: Balanced approach
            self._assign_balanced(shippers, packages, problem_instance)

        return State(shippers=shippers, current_time=0.0, total_cost=0.0)

    def _assign_by_value_density(self, shippers: List[Shipper], packages: List[Package],
                                problem_instance: ProblemInstance):
        """Assignment prioritizing value/weight ratio."""
        packages.sort(key=lambda p: p.value / max(p.weight, 0.1), reverse=True)
        self._greedy_assign(shippers, packages)

    def _assign_by_probability(self, shippers: List[Shipper], packages: List[Package],
                             problem_instance: ProblemInstance):
        """Assignment prioritizing delivery probability."""
        packages.sort(key=lambda p: problem_instance.get_delivery_probability(p.destination), reverse=True)
        self._greedy_assign(shippers, packages)

    def _assign_by_distance_clustering(self, shippers: List[Shipper], packages: List[Package],
                                     problem_instance: ProblemInstance):
        """Assignment trying to cluster by distance."""
        # Group packages by proximity
        location_groups = {}
        for package in packages:
            dest = package.destination
            if dest not in location_groups:
                location_groups[dest] = []
            location_groups[dest].append(package)

        # Assign groups to shippers
        shipper_idx = 0
        for dest, pkgs in location_groups.items():
            total_weight = sum(p.weight for p in pkgs)

            # Find shipper with enough capacity
            for i in range(len(shippers)):
                shipper = shippers[(shipper_idx + i) % len(shippers)]
                if shipper.remaining_capacity >= total_weight:
                    for pkg in pkgs:
                        shipper.add_package(pkg)
                    shipper_idx = (shipper_idx + i + 1) % len(shippers)
                    break
            else:
                # If no shipper can take all, distribute individually
                for pkg in pkgs:
                    for shipper in shippers:
                        if shipper.can_carry(pkg):
                            shipper.add_package(pkg)
                            break

    def _assign_balanced(self, shippers: List[Shipper], packages: List[Package],
                        problem_instance: ProblemInstance):
        """Balanced assignment considering multiple factors."""
        # Score packages by composite metric
        package_scores = []
        for package in packages:
            prob = problem_instance.get_delivery_probability(package.destination)
            value_density = package.value / max(package.weight, 0.1)
            distance = problem_instance.get_distance(0, package.destination)

            score = (0.4 * prob + 0.3 * (value_density / 100.0) + 0.3 * (1.0 / (1.0 + distance)))
            package_scores.append((package, score))

        package_scores.sort(key=lambda x: x[1], reverse=True)
        packages = [p for p, s in package_scores]
        self._greedy_assign(shippers, packages)

    def _greedy_assign(self, shippers: List[Shipper], packages: List[Package]):
        """Simple greedy assignment to shippers."""
        for package in packages:
            best_shipper = None
            best_remaining = -1

            for shipper in shippers:
                if shipper.can_carry(package) and shipper.remaining_capacity > best_remaining:
                    best_remaining = shipper.remaining_capacity
                    best_shipper = shipper

            if best_shipper is not None:
                best_shipper.add_package(package)

    def _select_action_pomo(self, state: State, shipper: Shipper, problem_instance: ProblemInstance,
                           start_idx: int) -> Optional[Action]:
        """POMO-style action selection with attention-like scoring."""
        if len(shipper.packages) == 0:
            return None

        # POMO uses attention mechanism - we simulate with weighted scoring
        best_package = None
        best_score = -float('inf')

        # Add some randomization based on start_idx for diversity
        randomization_factor = 0.1 + (start_idx % 10) * 0.05

        for package in shipper.packages:
            score = self._calculate_attention_score(package, shipper, state, problem_instance)

            # Add randomization for diversity (POMO exploration)
            if randomization_factor > 0:
                noise = np.random.normal(0, randomization_factor)
                score += noise

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

    def _calculate_attention_score(self, package: Package, shipper: Shipper, state: State,
                                 problem_instance: ProblemInstance) -> float:
        """Calculate attention-like score for package selection."""
        # Distance factor (query-key attention simulation)
        distance = problem_instance.get_distance(shipper.location, package.destination)
        max_distance = np.max(problem_instance.distance_matrix) if problem_instance.distance_matrix.size > 0 else 1.0
        distance_score = 1.0 - (distance / max_distance)

        # Probability factor (value component)
        prob_score = problem_instance.get_delivery_probability(package.destination, state.current_time)

        # Value factor
        value_score = package.value / 150.0  # Normalize

        # Consolidation factor (multiple packages at same location)
        same_location_count = len([p for p in shipper.packages if p.destination == package.destination])
        consolidation_score = 1.0 + (same_location_count - 1) * 0.2

        # Weighted combination (attention output)
        attention_score = (self.distance_weight * distance_score +
                          self.probability_weight * prob_score +
                          self.value_weight * value_score +
                          self.consolidation_weight * consolidation_score)

        return attention_score

    def _process_callbacks_pomo(self, state: State, problem_instance: ProblemInstance):
        """Process callbacks using POMO-style value estimation."""
        callbacks_to_process = []

        while not state.callback_queue.is_empty():
            callback = state.callback_queue.peek()
            if callback.callback_time <= state.current_time:
                callbacks_to_process.append(state.callback_queue.pop())
            else:
                break

        for callback in callbacks_to_process:
            # POMO-style value estimation for callback
            callback_value = self._estimate_callback_value(callback, state, problem_instance)

            if callback_value > self.callback_threshold:
                # Find best shipper
                best_shipper = self._find_best_shipper_for_callback(callback, state, problem_instance)

                if best_shipper is not None:
                    callback.accepted = True
                    callback.reattempt_time = state.current_time
                    callback.package.value *= 1.3  # Boost callback value
                    best_shipper.add_package(callback.package)

    def _estimate_callback_value(self, callback: Callback, state: State,
                                problem_instance: ProblemInstance) -> float:
        """Estimate value of accepting callback (POMO value function simulation)."""
        package = callback.package

        # Expected reward from successful callback
        prob = problem_instance.get_delivery_probability(package.destination, state.current_time)
        expected_reward = prob * problem_instance.R_callback_success

        # Time decay
        time_factor = max(0, 1 - state.current_time / problem_instance.time_window)

        # Package value factor
        value_factor = package.value / 100.0

        return expected_reward * time_factor * value_factor

    def _find_best_shipper_for_callback(self, callback: Callback, state: State,
                                      problem_instance: ProblemInstance) -> Optional[Shipper]:
        """Find best shipper for callback with cost consideration."""
        best_shipper = None
        best_net_value = 0

        for shipper in state.shippers:
            if shipper.can_carry(callback.package):
                # Calculate detour cost
                distance = problem_instance.get_distance(shipper.location, callback.package.destination)
                detour_cost = problem_instance.calculate_movement_cost(distance, shipper.current_load)

                # Calculate net value
                callback_value = self._estimate_callback_value(callback, state, problem_instance)
                net_value = callback_value - detour_cost

                if net_value > best_net_value:
                    best_net_value = net_value
                    best_shipper = shipper

        return best_shipper

    def _calculate_diversity(self, solutions: List[Dict]) -> float:
        """Calculate diversity of generated solutions."""
        if len(solutions) < 2:
            return 0.0

        rewards = [sol['total_reward'] for sol in solutions if sol is not None]
        if not rewards:
            return 0.0

        return np.std(rewards) / max(np.mean(rewards), 1.0)

    def _execute_action(self, state: State, shipper: Shipper, action: Action,
                       problem_instance: ProblemInstance) -> Tuple[State, float]:
        """Execute action (same as other algorithms)."""
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