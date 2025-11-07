"""
Experiment runner for comparing algorithms across scenarios.
"""

import time
import numpy as np
from typing import Dict, List, Any, Type
from tqdm import tqdm
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.echo import ECHO
from algorithms.gnn_cb import GNN_CB
from algorithms.sro_ev import SRO_EV
from algorithms.th_cb import TH_CB
from scenarios.scenario_generator import ScenarioGenerator
from evaluation.metrics import MetricsCalculator
from utils.helpers import set_random_seeds


class ExperimentRunner:
    """Orchestrate evaluation of algorithms across scenarios."""

    def __init__(self, scenario_config_file: str, num_runs: int = 10):
        self.scenario_generator = ScenarioGenerator(scenario_config_file)
        self.num_runs = num_runs
        self.results = {}

        # Initialize algorithms
        self.algorithms = {
            'ECHO': ECHO,
            'GNN-CB': GNN_CB,
            'SRO-EV': SRO_EV,
            'TH-CB': TH_CB
        }

        # Algorithm configurations
        self.algorithm_configs = {
            'ECHO': {
                'discount_factor': 0.95,
                'rollout_horizon': 3,
                'n_rollout_samples': 10,
                'callback_threshold': 0.5
            },
            'GNN-CB': {
                'max_detour_threshold': 25.0
            },
            'SRO-EV': {
                'insertion_threshold': 20.0
            },
            'TH-CB': {
                'theta_accept': 0.6,
                'theta_priority': 0.3,
                'omega_distance': 0.3,
                'omega_value': 0.25,
                'omega_prob': 0.2
            }
        }

    def run_experiments(self, scenario_names: List[str] = None, algorithm_names: List[str] = None) -> Dict:
        """Run experiments for specified scenarios and algorithms."""
        if scenario_names is None:
            scenario_names = self.scenario_generator.get_scenario_names()

        if algorithm_names is None:
            algorithm_names = list(self.algorithms.keys())

        print(f"Running experiments for {len(scenario_names)} scenarios and {len(algorithm_names)} algorithms")
        print(f"Number of runs per combination: {self.num_runs}")

        for scenario_name in scenario_names:
            print(f"\n=== Running scenario: {scenario_name} ===")
            scenario_results = {}

            for algorithm_name in algorithm_names:
                print(f"  Running algorithm: {algorithm_name}")
                algorithm_results = []

                # Progress bar for runs
                for run_id in tqdm(range(self.num_runs), desc=f"{algorithm_name}"):
                    # Set random seed for reproducibility
                    set_random_seeds(run_id)

                    try:
                        # Generate problem instance
                        problem_instance = self.scenario_generator.generate_instance(
                            scenario_name, seed=run_id
                        )

                        # Run algorithm
                        result = self._run_single_experiment(
                            algorithm_name, problem_instance, run_id
                        )

                        # Calculate metrics
                        metrics = MetricsCalculator.calculate_all_metrics(result, problem_instance)
                        metrics['runtime'] = result.get('runtime', 0.0)
                        metrics['iterations'] = result.get('iterations', 0)

                        algorithm_results.append(metrics)

                    except Exception as e:
                        print(f"    Error in run {run_id}: {e}")
                        # Add empty result to maintain run count
                        algorithm_results.append(self._get_empty_metrics())

                scenario_results[algorithm_name] = algorithm_results

            self.results[scenario_name] = scenario_results

        return self.results

    def _run_single_experiment(self, algorithm_name: str, problem_instance, run_id: int) -> Dict:
        """Run a single algorithm on a problem instance."""
        # Initialize algorithm
        algorithm_class = self.algorithms[algorithm_name]
        config = self.algorithm_configs.get(algorithm_name, {})
        algorithm = algorithm_class(config)

        # Run algorithm with timing
        start_time = time.time()
        result = algorithm.solve(problem_instance)
        end_time = time.time()

        # Add runtime to result
        result['runtime'] = end_time - start_time
        result['algorithm'] = algorithm_name
        result['run_id'] = run_id

        return result

    def _get_empty_metrics(self) -> Dict[str, float]:
        """Get empty metrics for failed runs."""
        return {
            'total_reward': 0.0,
            'delivery_success_rate': 0.0,
            'first_attempt_success_rate': 0.0,
            'package_completion_rate': 0.0,
            'callback_response_rate': 0.0,
            'callback_response_time': float('inf'),
            'callback_success_rate': 0.0,
            'average_delivery_time': float('inf'),
            'total_distance_traveled': 0.0,
            'average_capacity_utilization': 0.0,
            'cost_per_successful_delivery': float('inf'),
            'makespan': float('inf'),
            'workload_balance': 0.0,
            'runtime': 0.0,
            'iterations': 0
        }

    def get_results_summary(self) -> Dict:
        """Get summary of all results."""
        if not self.results:
            return {}

        summary = {}

        for scenario_name, scenario_results in self.results.items():
            print(f"\n=== Summary for {scenario_name} ===")

            scenario_summary = {}
            for algorithm_name, algorithm_results in scenario_results.items():
                # Calculate aggregate statistics
                stats = MetricsCalculator.calculate_aggregate_statistics(algorithm_results)
                scenario_summary[algorithm_name] = stats

                # Print key metrics
                if 'total_reward' in stats:
                    reward_mean = stats['total_reward']['mean']
                    reward_std = stats['total_reward']['std']
                    success_rate = stats.get('delivery_success_rate', {}).get('mean', 0)
                    runtime = stats.get('runtime', {}).get('mean', 0)

                    print(f"  {algorithm_name:8s}: "
                          f"Reward={reward_mean:7.1f}±{reward_std:5.1f}, "
                          f"Success={success_rate:5.1f}%, "
                          f"Runtime={runtime:6.3f}s")

            # Compare algorithms for this scenario
            comparison = MetricsCalculator.compare_algorithms(scenario_results)
            scenario_summary['comparison'] = comparison

            summary[scenario_name] = scenario_summary

        return summary

    def export_results(self, filename: str):
        """Export results to a file."""
        import json

        # Convert results to JSON-serializable format
        export_data = {
            'experiment_config': {
                'num_runs': self.num_runs,
                'algorithm_configs': self.algorithm_configs,
                'scenarios': list(self.results.keys())
            },
            'results': self.results,
            'summary': self.get_results_summary()
        }

        # Handle numpy types and inf values
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                if np.isfinite(obj):
                    return float(obj)
                else:
                    return str(obj)
            elif obj == float('inf'):
                return "Infinity"
            elif obj == float('-inf'):
                return "-Infinity"
            elif isinstance(obj, float) and np.isnan(obj):
                return "NaN"
            return obj

        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {key: recursive_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_for_json(obj)

        export_data = recursive_convert(export_data)

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Results exported to {filename}")

    def run_quick_test(self) -> Dict:
        """Run a quick test with minimal scenarios and runs."""
        print("Running quick test...")

        # Create simple test scenario
        from scenarios.scenario_generator import create_test_scenarios
        test_scenarios = create_test_scenarios()

        quick_results = {}

        for scenario_name, problem_instance in test_scenarios.items():
            print(f"Testing scenario: {scenario_name}")
            scenario_results = {}

            for algorithm_name in ['ECHO', 'GNN-CB']:  # Test only 2 algorithms
                print(f"  Testing {algorithm_name}")
                algorithm_results = []

                for run_id in range(3):  # Only 3 runs
                    set_random_seeds(run_id)

                    try:
                        result = self._run_single_experiment(
                            algorithm_name, problem_instance, run_id
                        )
                        metrics = MetricsCalculator.calculate_all_metrics(result, problem_instance)
                        metrics['runtime'] = result.get('runtime', 0.0)
                        algorithm_results.append(metrics)

                        # Print immediate feedback
                        reward = metrics.get('total_reward', 0)
                        success_rate = metrics.get('delivery_success_rate', 0)
                        print(f"    Run {run_id}: Reward={reward:.1f}, Success={success_rate:.1f}%")

                    except Exception as e:
                        print(f"    Error in run {run_id}: {e}")
                        algorithm_results.append(self._get_empty_metrics())

                scenario_results[algorithm_name] = algorithm_results

            quick_results[scenario_name] = scenario_results

        return quick_results