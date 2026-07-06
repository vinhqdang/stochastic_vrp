"""
Scenario generator for creating problem instances from YAML configurations.
"""

import yaml
import numpy as np
from typing import Dict, List, Tuple, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_structures import ProblemInstance
from utils.distance import (
    generate_random_locations, generate_clustered_locations,
    generate_hub_spoke_locations, create_distance_matrix
)
from utils.probability import (
    generate_delivery_probabilities, generate_callback_probabilities,
    generate_time_dependent_probabilities
)


class ScenarioGenerator:
    """Generate problem instances from scenario configurations."""

    def __init__(self, config_file: str = None):
        self.scenarios = {}
        if config_file:
            self.load_scenarios(config_file)

    def load_scenarios(self, config_file: str):
        """Load scenario configurations from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            for scenario_config in config['scenarios']:
                name = scenario_config['name']
                self.scenarios[name] = scenario_config

        except FileNotFoundError:
            print(f"Configuration file {config_file} not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    def generate_instance(self, scenario_name: str, seed: int = None) -> ProblemInstance:
        """Generate a problem instance from scenario configuration."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        if seed is not None:
            np.random.seed(seed)

        config = self.scenarios[scenario_name]['parameters']

        # Extract basic parameters
        n_shippers = config['n_shippers']
        n_packages = config['n_packages']
        n_locations = config['n_locations']

        # Generate shipper capacities
        if isinstance(config['shipper_capacities'], list):
            shipper_capacities = config['shipper_capacities'][:n_shippers]
            while len(shipper_capacities) < n_shippers:
                shipper_capacities.append(shipper_capacities[-1])
        else:
            # Single value - replicate for all shippers
            shipper_capacities = [config['shipper_capacities']] * n_shippers

        # Generate package weights
        weight_range = config['package_weights']
        if isinstance(weight_range, list) and len(weight_range) == 2:
            package_weights = np.random.uniform(weight_range[0], weight_range[1], n_packages).tolist()
        else:
            package_weights = [weight_range] * n_packages

        # Generate locations based on network type
        locations = self._generate_locations(config, n_locations, seed)

        # Create distance matrix
        distance_matrix = create_distance_matrix(locations)

        # Generate package destinations (excluding depot at index 0)
        package_destinations = np.random.choice(
            range(1, n_locations), size=n_packages, replace=True
        ).tolist()

        # Generate delivery probabilities
        delivery_probs = self._generate_delivery_probabilities(config, n_locations, seed)

        # Generate callback probabilities
        callback_probs = generate_callback_probabilities(
            n_locations,
            (config['callback_probability'] * 0.8, config['callback_probability'] * 1.2),
            seed
        )

        # Create problem instance
        instance = ProblemInstance(
            n_shippers=n_shippers,
            n_packages=n_packages,
            n_locations=n_locations,
            shipper_capacities=shipper_capacities,
            package_weights=package_weights,
            package_destinations=package_destinations,
            locations=locations,
            distance_matrix=distance_matrix,
            delivery_probabilities=delivery_probs,
            callback_probabilities=callback_probs,
            R_success_base=config.get('R_success_base', 100.0),
            R_failure=config.get('R_failure', -50.0),
            R_callback_success=config.get('R_callback_success', 80.0),
            cost_per_km_kg=config.get('cost_per_km_kg', 0.5),
            time_window=config.get('time_window', 120.0),
            network_type=config.get('network_type', 'uniform'),
            scenario_name=scenario_name
        )

        return instance

    def _generate_locations(self, config: Dict, n_locations: int, seed: int = None) -> List[Tuple[float, float]]:
        """Generate location coordinates based on network type."""
        network_type = config.get('network_type', 'uniform')
        bounds = config.get('location_bounds', [100, 100])
        width, height = bounds

        if seed is not None:
            np.random.seed(seed)

        if network_type == 'uniform':
            return generate_random_locations(n_locations, width, height, seed)

        elif network_type == 'clustered':
            n_clusters = max(2, n_locations // 4)  # Reasonable number of clusters
            points_per_cluster = n_locations // n_clusters
            extra_points = n_locations % n_clusters

            locations = []
            for i in range(n_clusters):
                cluster_size = points_per_cluster + (1 if i < extra_points else 0)
                if cluster_size > 0:
                    cluster_locations = generate_clustered_locations(
                        1, cluster_size, cluster_radius=width/10,
                        width=width, height=height, seed=seed
                    )
                    locations.extend(cluster_locations)

            return locations[:n_locations]

        elif network_type == 'hub_spoke':
            # Create hub at center
            hub = (width / 2, height / 2)

            # Create spokes in a circle around hub
            spokes = []
            radius = min(width, height) / 3

            for i in range(n_locations - 1):  # -1 for the hub
                angle = 2 * np.pi * i / (n_locations - 1)
                x = hub[0] + radius * np.cos(angle)
                y = hub[1] + radius * np.sin(angle)
                # Ensure within bounds
                x = max(0, min(width, x))
                y = max(0, min(height, y))
                spokes.append((x, y))

            return [hub] + spokes

        else:
            # Default to uniform
            return generate_random_locations(n_locations, width, height, seed)

    def _generate_delivery_probabilities(self, config: Dict, n_locations: int,
                                       seed: int = None) -> np.ndarray:
        """Generate delivery probabilities based on configuration."""
        if seed is not None:
            np.random.seed(seed)

        prob_range = config.get('delivery_probability_range', [0.6, 0.9])

        # Check if time-dependent probabilities are specified
        if 'time_dependent' in config and config['time_dependent']:
            # Generate time-dependent probabilities
            base_probs = generate_delivery_probabilities(n_locations, prob_range, seed=seed)
            time_multipliers = config.get('time_multipliers', [0.7, 0.85, 1.0, 0.8])
            return generate_time_dependent_probabilities(
                n_locations, len(time_multipliers), base_probs, time_multipliers, seed
            )
        else:
            # Generate location-specific probabilities
            distribution = config.get('probability_distribution', 'uniform')
            return generate_delivery_probabilities(n_locations, prob_range, distribution, seed)

    def get_scenario_names(self) -> List[str]:
        """Get list of available scenario names."""
        return list(self.scenarios.keys())

    def describe_scenario(self, scenario_name: str) -> str:
        """Get description of a scenario."""
        if scenario_name in self.scenarios:
            return self.scenarios[scenario_name].get('description', 'No description available')
        return f"Unknown scenario: {scenario_name}"


def create_test_scenarios() -> Dict[str, ProblemInstance]:
    """Create a set of test scenarios for quick testing."""
    generator = ScenarioGenerator()

    # Simple test scenario
    test_config = {
        'name': 'Simple_Test',
        'parameters': {
            'n_shippers': 2,
            'n_packages': 6,
            'n_locations': 4,
            'shipper_capacities': [30, 30],
            'package_weights': [5, 10],
            'delivery_probability_range': [0.7, 0.9],
            'callback_probability': 0.5,
            'network_type': 'uniform',
            'location_bounds': [20, 20],
            'R_success_base': 100,
            'R_failure': -50,
            'R_callback_success': 75,
            'cost_per_km_kg': 0.5,
            'time_window': 100
        }
    }

    generator.scenarios['Simple_Test'] = test_config

    return {
        'Simple_Test': generator.generate_instance('Simple_Test', seed=42)
    }