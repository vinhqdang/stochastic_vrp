# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python research project that implements and evaluates the **ECHO (Efficient Callback Handling Optimizer)** algorithm for the Stochastic Multi-Agent Vehicle Routing Problem with Uncertain Delivery and Dynamic Callbacks (SMAVRP-UDC). The project focuses on optimizing last-mile delivery routing when delivery success is uncertain and customers can request re-delivery callbacks.

## Core Algorithm Architecture

**ECHO** uses a Markov Decision Process formulation with:
- **Route-based state representation** for scalability
- **Callback priority queue management** for dynamic re-delivery requests
- **Approximate value function** using rollout policy for lookahead optimization
- **Multi-agent coordination** through decentralized decision-making

The algorithm addresses real-world scenarios where delivery attempts can fail stochastically, and customers can callback requesting re-delivery, requiring intelligent dynamic routing decisions.

## Development Workflow

create conda environment called py313 if not created, otherwise use the environment
remember to commit to github for all the changes. The github keys are setup already.

### Common Commands
```bash
# Install dependencies (when implemented)
pip install -r requirements.txt

# Run experiments (when implemented)
python main.py --config scenarios/scenarios.yaml --output results/

# Run specific scenario (when implemented)
python main.py --scenario "High_Uncertainty_Dense" --algorithms ECHO GNN-CB

# Run tests (when implemented)
python -m pytest tests/

# Run linting (when implemented)
python -m flake8 algorithms/ evaluation/ utils/
```

### Project Structure (Planned)
```
algorithms/          # Algorithm implementations
├── echo.py         # Main ECHO algorithm (MDP with rollout)
├── gnn_cb.py       # Greedy Nearest Neighbor baseline
├── sro_ev.py       # Static Route Optimization baseline
└── th_cb.py        # Threshold-based Callback baseline

scenarios/          # Test scenario definitions
├── scenario_generator.py
└── scenarios.yaml  # 10 scenarios with varying uncertainty

evaluation/         # Performance evaluation framework
├── metrics.py      # 20+ evaluation metrics
├── runner.py       # Experiment orchestration
├── analyzer.py     # Statistical analysis
└── visualizer.py   # Results visualization

utils/             # Helper functions
├── distance.py    # Distance calculations
├── probability.py # Probability distributions
└── helpers.py     # Utility functions
```

## Key Implementation Concepts

### State Space
The ECHO algorithm operates on states containing:
- Shipper positions, loads, and remaining capacity
- Pending deliveries with attempt counts
- Callback queue with priority scores
- Failed delivery tracking
- Current time and accumulated costs

### Action Space
At each decision epoch, actions include:
- Delivery attempts at current location
- Movement to next location
- Callback acceptance/rejection decisions
- Package selection for delivery attempts

### Reward Structure
- **Success rewards**: Time-decayed positive rewards for successful deliveries
- **Failure penalties**: Negative rewards for failed delivery attempts
- **Callback rewards**: Bonuses for successful callback handling
- **Movement costs**: Distance × weight cost function

### Algorithm Comparison
The project evaluates ECHO against three baselines:
1. **GNN-CB**: Greedy nearest neighbor with simple callback queue
2. **SRO-EV**: Static route optimization using expected values
3. **TH-CB**: Threshold-based callback acceptance policy

## Evaluation Framework

### Test Scenarios (10 total)
- **Low/Medium/High uncertainty** with varying delivery success rates (0.3-0.95)
- **Different network topologies**: clustered, uniform, hub-spoke
- **Capacity constraints**: tight vs. loose shipper capacities
- **Time-critical delivery** with steep reward decay
- **Large-scale deployment** with 50+ packages
- **Heterogeneous delivery probabilities** across locations
- **Time-dependent success rates** varying throughout the day

### Performance Metrics
- **Primary**: Total reward, delivery success rate, first-attempt success
- **Callback-specific**: Response rate, response time, callback success rate
- **Efficiency**: Average delivery time, distance traveled, capacity utilization
- **Quality**: Route deviation, makespan, cost per delivery
- **Robustness**: Performance variability across runs

## Research Context

This project implements algorithms described in PLANv1.md, which references foundational VRP literature and extends it to handle:
- **Stochastic delivery outcomes** with location-specific success probabilities
- **Dynamic callback events** requiring real-time route replanning
- **Multi-objective optimization** balancing delivery success vs. operational costs

The "ECHO" name reflects the callback mechanism - when deliveries fail, customer callbacks "echo" back to the system, requiring intelligent navigation through delivery uncertainty.

## Implementation Guidelines

### Key Classes to Implement
- `State`: Current system state (shippers, packages, callbacks, time)
- `Action`: Decision choices (movement, delivery attempts, callback responses)
- `Problem Instance`: Scenario configuration (locations, probabilities, costs)
- `Algorithm`: Base class for ECHO and baseline algorithms

### Critical Design Considerations
- **Scalability**: Use route-based state representation, not full combinatorial states
- **Stochastic modeling**: Sample delivery outcomes according to location probabilities
- **Rollout efficiency**: Limit horizon depth (3 steps) and sample count (10 samples)
- **Callback prioritization**: Score callbacks based on value, proximity, and timing
- **Statistical rigor**: Run 30+ replications per scenario for significance testing

## Testing and Validation

### Unit Tests Required
- State transition functions with stochastic outcomes
- Callback handling and priority queue management
- Metric computation accuracy
- Scenario generation from YAML configurations

### Integration Tests
- End-to-end algorithm execution on sample scenarios
- Statistical analysis pipeline
- Visualization generation

### Performance Benchmarks
- Runtime complexity validation: O(n×m×s×h×k) per decision
- Memory usage tracking for large-scale scenarios
- Comparison against theoretical bounds where available
