"""
Comprehensive evaluation metrics for the stochastic VRP problem.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import State, ProblemInstance, DeliveryAttempt, Callback


class MetricsCalculator:
    """Calculate comprehensive performance metrics for algorithm evaluation."""

    @staticmethod
    def calculate_all_metrics(result: Dict, problem_instance: ProblemInstance) -> Dict[str, float]:
        """Calculate all evaluation metrics for a single run."""
        state = result['state']
        history = result.get('history', [])

        metrics = {}

        # Primary performance metrics
        metrics.update(MetricsCalculator._calculate_primary_metrics(result, problem_instance))

        # Callback performance metrics
        metrics.update(MetricsCalculator._calculate_callback_metrics(state))

        # Efficiency metrics
        metrics.update(MetricsCalculator._calculate_efficiency_metrics(state, problem_instance))

        # Route quality metrics
        metrics.update(MetricsCalculator._calculate_route_quality_metrics(state, problem_instance))

        # Fairness metrics
        metrics.update(MetricsCalculator._calculate_fairness_metrics(state))

        return metrics

    @staticmethod
    def _calculate_primary_metrics(result: Dict, problem_instance: ProblemInstance) -> Dict[str, float]:
        """Calculate primary performance metrics."""
        state = result['state']
        metrics = {}

        # Total reward
        metrics['total_reward'] = result.get('total_reward', 0.0)

        # Delivery success rate
        total_deliveries = len(state.completed_deliveries) + len(state.failed_deliveries)
        if total_deliveries > 0:
            metrics['delivery_success_rate'] = len(state.completed_deliveries) / total_deliveries * 100
        else:
            metrics['delivery_success_rate'] = 0.0

        # First-attempt success rate
        first_attempts = [d for d in state.completed_deliveries + state.failed_deliveries
                         if d.attempt_number == 1]
        successful_first = [d for d in first_attempts if d.successful]
        if first_attempts:
            metrics['first_attempt_success_rate'] = len(successful_first) / len(first_attempts) * 100
        else:
            metrics['first_attempt_success_rate'] = 0.0

        # Package completion rate
        total_packages = problem_instance.n_packages
        completed_packages = len(state.completed_deliveries)
        metrics['package_completion_rate'] = completed_packages / total_packages * 100

        return metrics

    @staticmethod
    def _calculate_callback_metrics(state: State) -> Dict[str, float]:
        """Calculate callback-related performance metrics."""
        metrics = {}

        callbacks = state.callbacks
        if callbacks:
            # Callback response rate
            accepted_callbacks = [c for c in callbacks if c.accepted]
            metrics['callback_response_rate'] = len(accepted_callbacks) / len(callbacks) * 100

            # Average callback response time
            if accepted_callbacks:
                response_times = [
                    c.reattempt_time - c.callback_time
                    for c in accepted_callbacks
                    if c.reattempt_time is not None
                ]
                metrics['callback_response_time'] = np.mean(response_times) if response_times else float('inf')

                # Callback success rate
                successful_callbacks = [
                    c for c in accepted_callbacks if c.successful
                ]
                metrics['callback_success_rate'] = len(successful_callbacks) / len(accepted_callbacks) * 100
            else:
                metrics['callback_response_time'] = float('inf')
                metrics['callback_success_rate'] = 0.0
        else:
            metrics['callback_response_rate'] = 0.0
            metrics['callback_response_time'] = 0.0
            metrics['callback_success_rate'] = 0.0

        # Callback queue efficiency (lower is better)
        metrics['max_callback_queue_size'] = max(
            [len(state.callback_queue.callbacks)] + [0]
        )

        return metrics

    @staticmethod
    def _calculate_efficiency_metrics(state: State, problem_instance: ProblemInstance) -> Dict[str, float]:
        """Calculate efficiency-related metrics."""
        metrics = {}

        # Average delivery time
        successful_deliveries = [d for d in state.completed_deliveries if d.delivery_time is not None]
        if successful_deliveries:
            delivery_times = [d.delivery_time for d in successful_deliveries]
            metrics['average_delivery_time'] = np.mean(delivery_times)
        else:
            metrics['average_delivery_time'] = float('inf')

        # Total distance traveled
        total_distance = 0.0
        for shipper in state.shippers:
            if len(shipper.route_history) > 0:
                current_location = 0  # Start from depot
                for location in shipper.route_history:
                    total_distance += problem_instance.get_distance(current_location, location)
                    current_location = location

        metrics['total_distance_traveled'] = total_distance

        # Capacity utilization
        if state.shippers:
            utilizations = []
            for shipper in state.shippers:
                if shipper.load_history:
                    avg_load = np.mean(shipper.load_history)
                    utilization = avg_load / shipper.capacity * 100 if shipper.capacity > 0 else 0
                    utilizations.append(utilization)

            metrics['average_capacity_utilization'] = np.mean(utilizations) if utilizations else 0.0
        else:
            metrics['average_capacity_utilization'] = 0.0

        # Cost per successful delivery
        successful_count = len([d for d in state.completed_deliveries if d.successful])
        if successful_count > 0:
            metrics['cost_per_successful_delivery'] = state.total_cost / successful_count
        else:
            metrics['cost_per_successful_delivery'] = float('inf')

        # Distance efficiency (successful deliveries per unit distance)
        if total_distance > 0:
            metrics['delivery_efficiency'] = successful_count / total_distance
        else:
            metrics['delivery_efficiency'] = 0.0

        return metrics

    @staticmethod
    def _calculate_route_quality_metrics(state: State, problem_instance: ProblemInstance) -> Dict[str, float]:
        """Calculate route quality metrics."""
        metrics = {}

        # Makespan (maximum completion time)
        if state.shippers:
            completion_times = [s.completion_time for s in state.shippers]
            metrics['makespan'] = max(completion_times) if completion_times else 0.0
        else:
            metrics['makespan'] = 0.0

        # Route length variance (measure of route balance)
        route_lengths = []
        for shipper in state.shippers:
            if len(shipper.route_history) > 0:
                length = 0.0
                current_location = 0
                for location in shipper.route_history:
                    length += problem_instance.get_distance(current_location, location)
                    current_location = location
                route_lengths.append(length)

        if route_lengths:
            metrics['route_length_variance'] = np.var(route_lengths)
            metrics['route_length_std'] = np.std(route_lengths)
        else:
            metrics['route_length_variance'] = 0.0
            metrics['route_length_std'] = 0.0

        # Average visits per location
        location_visits = {}
        for shipper in state.shippers:
            for location in shipper.route_history:
                location_visits[location] = location_visits.get(location, 0) + 1

        if location_visits:
            metrics['average_visits_per_location'] = np.mean(list(location_visits.values()))
        else:
            metrics['average_visits_per_location'] = 0.0

        return metrics

    @staticmethod
    def _calculate_fairness_metrics(state: State) -> Dict[str, float]:
        """Calculate fairness-related metrics."""
        metrics = {}

        # Shipper workload balance
        deliveries_per_shipper = []
        distance_per_shipper = []

        for shipper in state.shippers:
            # Count completed deliveries for this shipper
            shipper_deliveries = len([
                d for d in state.completed_deliveries if d.shipper_id == shipper.id
            ])
            deliveries_per_shipper.append(shipper_deliveries)

            # Calculate total distance for this shipper
            total_distance = 0.0
            if len(shipper.route_history) > 0:
                current_location = 0
                for location in shipper.route_history:
                    # Note: We'd need distance matrix here, using approximation
                    total_distance += 1.0  # Placeholder
                    current_location = location
            distance_per_shipper.append(total_distance)

        # Workload balance (coefficient of variation)
        if deliveries_per_shipper and np.mean(deliveries_per_shipper) > 0:
            metrics['workload_balance'] = np.std(deliveries_per_shipper) / np.mean(deliveries_per_shipper)
        else:
            metrics['workload_balance'] = 0.0

        # Distance balance
        if distance_per_shipper and np.mean(distance_per_shipper) > 0:
            metrics['distance_balance'] = np.std(distance_per_shipper) / np.mean(distance_per_shipper)
        else:
            metrics['distance_balance'] = 0.0

        return metrics

    @staticmethod
    def calculate_aggregate_statistics(metric_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics across multiple runs."""
        if not metric_results:
            return {}

        # Get all metric names
        metric_names = set()
        for result in metric_results:
            metric_names.update(result.keys())

        aggregate = {}

        for metric_name in metric_names:
            values = [result.get(metric_name, 0.0) for result in metric_results]
            # Filter out infinite values for statistics
            finite_values = [v for v in values if np.isfinite(v)]

            if finite_values:
                aggregate[metric_name] = {
                    'mean': np.mean(finite_values),
                    'std': np.std(finite_values),
                    'min': np.min(finite_values),
                    'max': np.max(finite_values),
                    'median': np.median(finite_values),
                    'q25': np.percentile(finite_values, 25),
                    'q75': np.percentile(finite_values, 75),
                    'count': len(finite_values)
                }
            else:
                aggregate[metric_name] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan'),
                    'median': float('nan'),
                    'q25': float('nan'),
                    'q75': float('nan'),
                    'count': 0
                }

        return aggregate

    @staticmethod
    def compare_algorithms(results_by_algorithm: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
        """Compare performance across algorithms."""
        comparison = {}

        # Calculate statistics for each algorithm
        for algorithm_name, results in results_by_algorithm.items():
            comparison[algorithm_name] = MetricsCalculator.calculate_aggregate_statistics(results)

        # Rank algorithms by key metrics
        key_metrics = ['total_reward', 'delivery_success_rate', 'callback_response_rate']
        rankings = {}

        for metric in key_metrics:
            algorithm_means = {}
            for alg_name, stats in comparison.items():
                if metric in stats:
                    algorithm_means[alg_name] = stats[metric]['mean']

            # Sort by metric value (higher is better for these metrics)
            sorted_algorithms = sorted(algorithm_means.items(), key=lambda x: x[1], reverse=True)
            rankings[metric] = [alg_name for alg_name, _ in sorted_algorithms]

        comparison['rankings'] = rankings

        return comparison