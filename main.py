#!/usr/bin/env python3
"""
Main execution script for the Stochastic VRP with Callbacks experiments.

Usage:
    python main.py --quick                  # Run quick test
    python main.py --full                   # Run full experiments
    python main.py --scenario Low_Uncertainty_Sparse  # Run specific scenario
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.runner import ExperimentRunner
from evaluation.visualizer import ResultsVisualizer


def run_quick_test():
    """Run a quick test to verify implementation."""
    print("=" * 60)
    print("STOCHASTIC VRP WITH CALLBACKS - QUICK TEST")
    print("=" * 60)

    # Initialize runner (will use create_test_scenarios)
    runner = ExperimentRunner("scenarios/scenarios.yaml", num_runs=3)

    # Run quick test
    try:
        results = runner.run_quick_test()

        # Display results
        print("\n" + "=" * 50)
        print("QUICK TEST RESULTS")
        print("=" * 50)

        for scenario_name, scenario_results in results.items():
            print(f"\nScenario: {scenario_name}")
            print("-" * 30)

            for algorithm_name, algorithm_results in scenario_results.items():
                if algorithm_results:
                    rewards = [r.get('total_reward', 0) for r in algorithm_results]
                    success_rates = [r.get('delivery_success_rate', 0) for r in algorithm_results]
                    runtimes = [r.get('runtime', 0) for r in algorithm_results]

                    print(f"{algorithm_name:10s}: "
                          f"Reward={sum(rewards)/len(rewards):6.1f} "
                          f"Success={sum(success_rates)/len(success_rates):5.1f}% "
                          f"Runtime={sum(runtimes)/len(runtimes):6.3f}s")

        print("\n✅ Quick test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_experiments():
    """Run full experiments across all scenarios."""
    print("=" * 60)
    print("STOCHASTIC VRP WITH CALLBACKS - FULL EXPERIMENTS")
    print("=" * 60)

    # Check if scenario file exists
    scenario_file = "scenarios/scenarios.yaml"
    if not os.path.exists(scenario_file):
        print(f"❌ Scenario file {scenario_file} not found!")
        return False

    # Initialize runner
    runner = ExperimentRunner(scenario_file, num_runs=10)

    try:
        # Run all experiments
        print("Starting full experiment suite...")
        results = runner.run_experiments()

        # Get and display summary
        summary = runner.get_results_summary()

        # Export results
        results_file = "results/experiment_results.json"
        os.makedirs("results", exist_ok=True)
        runner.export_results(results_file)

        # Create visualizations if possible
        try:
            visualizer = ResultsVisualizer()
            visualizer.create_comparison_plots(results, "results/")
            visualizer.create_summary_table(summary, "results/summary_table.txt")
            print("📊 Visualizations created in results/ directory")
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")

        print(f"\n✅ Full experiments completed! Results saved to {results_file}")
        return True

    except Exception as e:
        print(f"❌ Full experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_specific_scenario(scenario_name: str):
    """Run experiments for a specific scenario."""
    print(f"Running experiments for scenario: {scenario_name}")

    scenario_file = "scenarios/scenarios.yaml"
    if not os.path.exists(scenario_file):
        print(f"❌ Scenario file {scenario_file} not found!")
        return False

    runner = ExperimentRunner(scenario_file, num_runs=5)

    try:
        results = runner.run_experiments([scenario_name])
        summary = runner.get_results_summary()

        print(f"\n✅ Scenario {scenario_name} completed!")

        # Print results for this scenario
        if scenario_name in summary:
            scenario_summary = summary[scenario_name]
            print(f"\nResults for {scenario_name}:")
            print("-" * 40)

            for alg_name, stats in scenario_summary.items():
                if alg_name != 'comparison' and 'total_reward' in stats:
                    reward = stats['total_reward']
                    success = stats.get('delivery_success_rate', {})
                    print(f"{alg_name:10s}: "
                          f"Reward={reward.get('mean', 0):6.1f}±{reward.get('std', 0):4.1f} "
                          f"Success={success.get('mean', 0):5.1f}%")

        return True

    except Exception as e:
        print(f"❌ Scenario experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stochastic VRP with Callbacks Experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full experiments")
    parser.add_argument("--scenario", type=str, help="Run specific scenario")

    args = parser.parse_args()

    # Set Python path to use the conda environment
    python_executable = "/opt/miniconda3/envs/py313/bin/python"
    if os.path.exists(python_executable):
        os.environ["PYTHON"] = python_executable

    success = False

    if args.quick:
        success = run_quick_test()
    elif args.full:
        success = run_full_experiments()
    elif args.scenario:
        success = run_specific_scenario(args.scenario)
    else:
        print("Please specify --quick, --full, or --scenario <name>")
        print("Available scenarios:")

        try:
            from scenarios.scenario_generator import ScenarioGenerator
            generator = ScenarioGenerator("scenarios/scenarios.yaml")
            for name in generator.get_scenario_names():
                description = generator.describe_scenario(name)
                print(f"  {name}: {description}")
        except Exception:
            print("  (Could not load scenarios)")

        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())