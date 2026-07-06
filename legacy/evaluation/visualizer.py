"""
Visualization tools for experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import os


class ResultsVisualizer:
    """Create plots and tables for results presentation."""

    def __init__(self):
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def create_comparison_plots(self, results: Dict, output_dir: str):
        """Create comparison plots for all scenarios."""
        os.makedirs(output_dir, exist_ok=True)

        # Key metrics to plot
        metrics_to_plot = [
            ('total_reward', 'Total Reward'),
            ('delivery_success_rate', 'Delivery Success Rate (%)'),
            ('callback_response_rate', 'Callback Response Rate (%)'),
            ('average_delivery_time', 'Average Delivery Time (min)'),
            ('runtime', 'Runtime (seconds)')
        ]

        for metric_key, metric_label in metrics_to_plot:
            self._create_metric_comparison_plot(
                results, metric_key, metric_label, output_dir
            )

        # Create overall summary plot
        self._create_overall_summary_plot(results, output_dir)

        print(f"Comparison plots created in {output_dir}")

    def _create_metric_comparison_plot(self, results: Dict, metric: str, label: str, output_dir: str):
        """Create a comparison plot for a specific metric."""
        scenarios = list(results.keys())
        if not scenarios:
            return

        algorithms = list(results[scenarios[0]].keys())
        n_scenarios = len(scenarios)
        n_algorithms = len(algorithms)

        # Create subplot for each scenario
        fig, axes = plt.subplots(1, min(n_scenarios, 3), figsize=(15, 5))
        if n_scenarios == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

        for i, scenario in enumerate(scenarios[:3]):  # Limit to 3 scenarios
            ax = axes[i] if i < len(axes) else axes[-1]

            scenario_data = []
            labels = []

            for j, algorithm in enumerate(algorithms):
                if algorithm in results[scenario]:
                    values = [
                        run.get(metric, 0) for run in results[scenario][algorithm]
                        if np.isfinite(run.get(metric, 0))
                    ]

                    if values:
                        scenario_data.append(values)
                        labels.append(algorithm)

            if scenario_data:
                bp = ax.boxplot(scenario_data, labels=labels, patch_artist=True)

                # Color the boxes
                for patch, color in zip(bp['boxes'], colors[:len(scenario_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_title(f'{scenario}', fontsize=10)
                ax.set_ylabel(label)
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_overall_summary_plot(self, results: Dict, output_dir: str):
        """Create an overall summary plot showing algorithm rankings."""
        scenarios = list(results.keys())
        if not scenarios:
            return

        algorithms = list(results[scenarios[0]].keys())

        # Calculate average total reward for each algorithm across scenarios
        algorithm_scores = {alg: [] for alg in algorithms}

        for scenario in scenarios:
            for algorithm in algorithms:
                if algorithm in results[scenario]:
                    rewards = [
                        run.get('total_reward', 0) for run in results[scenario][algorithm]
                        if np.isfinite(run.get('total_reward', 0))
                    ]
                    if rewards:
                        algorithm_scores[algorithm].append(np.mean(rewards))

        # Create summary bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Average performance across scenarios
        avg_scores = {alg: np.mean(scores) if scores else 0
                     for alg, scores in algorithm_scores.items()}

        algorithms_sorted = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars1 = ax1.bar(algorithms_sorted, [avg_scores[alg] for alg in algorithms_sorted],
                       color=colors[:len(algorithms_sorted)])
        ax1.set_title('Average Total Reward Across Scenarios')
        ax1.set_ylabel('Total Reward')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, alg in zip(bars1, algorithms_sorted):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')

        # Performance consistency (std deviation)
        std_scores = {alg: np.std(scores) if len(scores) > 1 else 0
                     for alg, scores in algorithm_scores.items()}

        bars2 = ax2.bar(algorithms_sorted, [std_scores[alg] for alg in algorithms_sorted],
                       color=colors[:len(algorithms_sorted)])
        ax2.set_title('Performance Consistency (Lower is Better)')
        ax2.set_ylabel('Standard Deviation of Rewards')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, alg in zip(bars2, algorithms_sorted):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/overall_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_table(self, summary: Dict, output_file: str):
        """Create a summary table of results."""
        with open(output_file, 'w') as f:
            f.write("STOCHASTIC VRP WITH CALLBACKS - EXPERIMENT RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            for scenario_name, scenario_data in summary.items():
                f.write(f"SCENARIO: {scenario_name}\n")
                f.write("-" * 50 + "\n")

                # Write algorithm performance
                algorithms = [alg for alg in scenario_data.keys() if alg != 'comparison']

                if algorithms:
                    # Header
                    f.write(f"{'Algorithm':<12} {'Reward':<12} {'Success':<10} {'Callbacks':<12} {'Runtime':<10}\n")
                    f.write("-" * 65 + "\n")

                    for alg in algorithms:
                        stats = scenario_data[alg]
                        reward = stats.get('total_reward', {})
                        success = stats.get('delivery_success_rate', {})
                        callbacks = stats.get('callback_response_rate', {})
                        runtime = stats.get('runtime', {})

                        f.write(f"{alg:<12} "
                               f"{reward.get('mean', 0):6.1f}±{reward.get('std', 0):4.1f} "
                               f"{success.get('mean', 0):6.1f}%   "
                               f"{callbacks.get('mean', 0):6.1f}%     "
                               f"{runtime.get('mean', 0):6.3f}s\n")

                    # Write rankings if available
                    if 'comparison' in scenario_data and 'rankings' in scenario_data['comparison']:
                        rankings = scenario_data['comparison']['rankings']
                        f.write("\nRANKINGS:\n")
                        for metric, ranking in rankings.items():
                            f.write(f"  {metric}: {' > '.join(ranking)}\n")

                f.write("\n" + "=" * 70 + "\n\n")

        print(f"Summary table saved to {output_file}")

    def plot_algorithm_convergence(self, results: Dict, algorithm_name: str, output_dir: str):
        """Plot convergence behavior for a specific algorithm."""
        os.makedirs(output_dir, exist_ok=True)

        scenarios = list(results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, scenario in enumerate(scenarios[:4]):
            if algorithm_name in results[scenario]:
                rewards = [run.get('total_reward', 0) for run in results[scenario][algorithm_name]]
                iterations = [run.get('iterations', 0) for run in results[scenario][algorithm_name]]

                if i < len(axes):
                    ax = axes[i]
                    ax.scatter(iterations, rewards, alpha=0.6)
                    ax.set_xlabel('Iterations')
                    ax.set_ylabel('Total Reward')
                    ax.set_title(f'{scenario}')

        plt.suptitle(f'{algorithm_name} - Convergence Analysis')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{algorithm_name}_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_profile(self, results: Dict, output_dir: str):
        """Create performance profile showing relative algorithm performance."""
        os.makedirs(output_dir, exist_ok=True)

        # Collect all performance ratios
        algorithm_names = None
        performance_ratios = {}

        for scenario_results in results.values():
            if algorithm_names is None:
                algorithm_names = list(scenario_results.keys())
                for alg in algorithm_names:
                    performance_ratios[alg] = []

            for run_idx in range(len(list(scenario_results.values())[0])):
                # Get performance of all algorithms for this run
                performances = {}
                for alg in algorithm_names:
                    if run_idx < len(scenario_results[alg]):
                        performances[alg] = scenario_results[alg][run_idx].get('total_reward', 0)

                # Best performance for this instance
                best_performance = max(performances.values()) if performances.values() else 1

                # Compute ratios
                if best_performance > 0:
                    for alg in performances:
                        ratio = performances[alg] / best_performance
                        performance_ratios[alg].append(ratio)

        # Plot performance profile
        plt.figure(figsize=(10, 6))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (alg, ratios) in enumerate(performance_ratios.items()):
            if ratios:
                sorted_ratios = np.sort(ratios)
                cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
                plt.plot(sorted_ratios, cumulative, label=alg, linewidth=2, color=colors[i % len(colors)])

        plt.xlabel('Performance Ratio (Algorithm Performance / Best Performance)')
        plt.ylabel('Cumulative Probability')
        plt.title('Performance Profile - Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1.1)
        plt.ylim(0, 1)

        plt.savefig(f'{output_dir}/performance_profile.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance profile saved to {output_dir}/performance_profile.png")