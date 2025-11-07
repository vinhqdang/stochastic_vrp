```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗ ██████╗██╗  ██╗ ██████╗                          ║
║   ██╔════╝██╔════╝██║  ██║██╔═══██╗                         ║
║   █████╗  ██║     ███████║██║   ██║                         ║
║   ██╔══╝  ██║     ██╔══██║██║   ██║                         ║
║   ███████╗╚██████╗██║  ██║╚██████╔╝                         ║
║   ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝                          ║
║                                                               ║
║          Efficient Callback Handling Optimizer               ║
║      "Listen to the echoes, adapt to uncertainty"           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

# ECHO: Efficient Callback Handling Optimizer
## Navigating Uncertain Last-Mile Delivery with Adaptive Intelligence

*A Route-based Markov Decision Process for Stochastic Package Delivery with Dynamic Callbacks*

---

## Executive Summary

This document presents a comprehensive algorithmic solution for the **Stochastic Multi-Agent Vehicle Routing Problem with Uncertain Delivery and Dynamic Callbacks (SMAVRP-UDC)**. The problem addresses real-world last-mile delivery scenarios where delivery success is uncertain and customers can callback requesting re-delivery, requiring dynamic routing decisions.

Our proposed solution, **ECHO (Efficient Callback Handling Optimizer)**, uses adaptive route-based planning with lookahead optimization to maximize delivery success rates while minimizing costs in highly uncertain environments.

---

## 1. Problem Formulation

### 1.1 Problem Definition

**Given:**
- **Shippers:** Set of n shippers S = {s₁, s₂, ..., sₙ}
- **Packages:** Set of m packages P = {p₁, p₂, ..., pₘ}
- **Locations:** Set of delivery locations L = {l₁, l₂, ..., lₖ}
- **Shipper Capacity:** Each shipper sᵢ has capacity Cᵢ (kg)
- **Package Weight:** Each package pⱼ has weight wⱼ (kg)
- **Travel Time:** Known travel time matrix T[i][j] between locations
- **Delivery Probability:** Known probability distribution P(success|location) for each location
- **Cost Function:** cost = time × weight (linear in both dimensions)

**Objectives:**
- Maximize total reward (timely deliveries)
- Minimize delivery failures (negative rewards)
- Optimize dynamic routing decisions when callbacks occur

**Constraints:**
- Capacity: Σ(packages on shipper) ≤ Cᵢ
- Time: Each delivery attempt has associated time cost
- Callback response: Decision required when callback received

### 1.2 Formal Notation

| Symbol | Description |
|--------|-------------|
| S | Set of shippers {s₁, ..., sₙ} |
| P | Set of packages {p₁, ..., pₘ} |
| L | Set of locations {l₁, ..., lₖ} |
| Cᵢ | Capacity of shipper i (kg) |
| wⱼ | Weight of package j (kg) |
| T[i][j] | Travel time from location i to j |
| P(lᵢ, t) | Probability of successful delivery at location lᵢ at time t |
| R_success(t) | Reward for successful delivery at time t |
| R_failure | Penalty for failed delivery |
| cost(d, w) | Cost function = d × w (distance × weight) |

---

## 2. Proposed Algorithm: ECHO (Efficient Callback Handling Optimizer)

### 2.1 Algorithm Overview

The **ECHO (Efficient Callback Handling Optimizer)** algorithm formulates the problem as a Markov Decision Process where states represent current shipper positions, package assignments, and callback queues. The algorithm uses approximate dynamic programming with rollout-based lookahead to make real-time routing decisions.

**Why ECHO?** The name captures the essence of the callback mechanism—when a delivery fails, the customer's callback "echoes" back to the system, requiring intelligent re-routing decisions. Like sonar using echoes to navigate uncertain waters, ECHO navigates the uncertainty of last-mile delivery.

**Key Features:**
1. **Route-based state representation** for scalability
2. **Callback priority queue** management
3. **Approximate value function** using rollout policy
4. **Dynamic replanning** upon callback events
5. **Multi-agent coordination** through decentralized decision-making

### 2.2 State Space Definition

```
State s_t = {
    shipper_states: [(location, current_load, remaining_capacity, route_plan)₁, ..., (...)ₙ],
    pending_deliveries: [(package_id, destination, delivery_probability, attempt_count)₁, ..., (...)ₘ],
    callback_queue: [(package_id, callback_time, priority_score)₁, ..., (...)_c],
    failed_deliveries: [package_ids],
    current_time: t,
    total_cost: accumulated_cost
}
```

### 2.3 Action Space

For each decision epoch (arrival at location or callback event):
```
Action a = {
    delivery_action: [attempt_delivery | skip],
    movement_action: [next_location | return_to_callback | continue_route],
    callback_response: [accept_callback | defer_callback | reject_callback],
    package_selection: [package_ids_to_attempt]
}
```

### 2.4 Transition Function

```python
def transition(state, action, stochastic_outcome):
    """
    State transition based on action and stochastic delivery outcome
    """
    new_state = copy(state)
    
    # Update shipper position and time
    new_state.current_time += travel_time
    new_state.shipper_location = action.next_location
    
    # Process delivery attempts with stochastic outcome
    for pkg in action.package_selection:
        success_prob = get_delivery_probability(pkg.destination, new_state.current_time)
        if stochastic_outcome <= success_prob:
            # Successful delivery
            new_state.shipper_load -= pkg.weight
            new_state.pending_deliveries.remove(pkg)
            new_state.reward += calculate_success_reward(new_state.current_time)
        else:
            # Failed delivery - add to callback queue with probability
            pkg.attempt_count += 1
            callback_prob = get_callback_probability(pkg, new_state.current_time)
            if random() < callback_prob:
                callback_time = sample_callback_time_distribution()
                new_state.callback_queue.add((pkg, callback_time))
    
    # Update costs
    new_state.total_cost += calculate_movement_cost(travel_distance, shipper_load)
    
    return new_state
```

### 2.5 Reward Function

```python
def reward(state, action, next_state):
    """
    Calculate immediate reward for state-action-next_state transition
    """
    reward = 0
    
    # Success rewards (time-dependent)
    for pkg in successful_deliveries:
        time_factor = max(0, 1 - (current_time - pkg.expected_delivery_time) / time_window)
        reward += R_success_base * time_factor
    
    # Failure penalties
    for pkg in failed_deliveries:
        reward += R_failure * pkg.attempt_count  # Increasing penalty
    
    # Movement costs
    reward -= cost(travel_distance, shipper_load)
    
    # Callback handling rewards
    for callback in handled_callbacks:
        reward += R_callback_success
    
    # Capacity utilization bonus (encourage efficient loading)
    utilization = shipper_load / shipper_capacity
    if 0.7 <= utilization <= 0.95:
        reward += R_utilization_bonus
    
    return reward
```

### 2.6 Callback Handling Mechanism

```python
def handle_callback(state, callback_event):
    """
    Decision logic for responding to callback
    """
    pkg, callback_time = callback_event
    
    # Calculate callback priority score
    priority_score = calculate_callback_priority(
        package_value=pkg.value,
        customer_tier=pkg.customer_tier,
        time_since_attempt=callback_time - pkg.last_attempt_time,
        shipper_proximity=min_distance_to_shippers(pkg.destination),
        remaining_capacity=available_shipper_capacity()
    )
    
    # Find best shipper for callback
    best_shipper = None
    best_value = -inf
    
    for shipper in get_available_shippers(pkg.weight):
        # Estimate opportunity cost of diversion
        current_route_value = estimate_route_value(shipper.planned_route)
        callback_route_value = estimate_callback_route_value(shipper, pkg)
        diversion_cost = calculate_diversion_cost(shipper.location, pkg.destination)
        
        net_value = callback_route_value - current_route_value - diversion_cost
        
        if net_value > best_value and net_value > threshold_callback_acceptance:
            best_value = net_value
            best_shipper = shipper
    
    if best_shipper:
        return accept_callback(best_shipper, pkg)
    else:
        return defer_or_reject_callback(pkg, priority_score)
```

### 2.7 Value Function Approximation

```python
def approximate_value_function(state):
    """
    Approximate value function using weighted features
    """
    features = extract_features(state)
    
    # Feature vector
    f = [
        features.remaining_high_value_packages,
        features.average_shipper_utilization,
        features.total_expected_delivery_probability,
        features.callback_queue_size,
        features.time_remaining,
        features.distance_to_undelivered_packages,
        features.failed_delivery_count,
        features.average_delivery_probability
    ]
    
    # Learned weights (via temporal difference learning or batch regression)
    weights = trained_weights
    
    return np.dot(weights, f)
```

### 2.8 Rollout Policy for Lookahead

```python
def rollout_policy(state, horizon=3):
    """
    Greedy rollout policy for approximate lookahead
    """
    simulated_state = copy(state)
    total_expected_value = 0
    
    for h in range(horizon):
        # For each shipper, select next best action greedily
        for shipper in simulated_state.shippers:
            if len(shipper.packages) == 0:
                continue
            
            # Evaluate possible next locations
            best_action = None
            best_expected_value = -inf
            
            for next_location in get_feasible_locations(shipper):
                # Calculate expected reward
                delivery_prob = get_delivery_probability(next_location)
                expected_reward = delivery_prob * R_success - (1 - delivery_prob) * abs(R_failure)
                travel_cost = cost(shipper.location, next_location, shipper.load)
                
                expected_value = expected_reward - travel_cost
                
                if expected_value > best_expected_value:
                    best_expected_value = expected_value
                    best_action = (next_location, expected_reward)
            
            # Simulate action
            if best_action:
                simulated_state = simulate_transition(simulated_state, best_action)
                total_expected_value += best_expected_value
    
    return total_expected_value
```

### 2.9 ECHO Algorithm Pseudocode

```
Algorithm: ECHO (Efficient Callback Handling Optimizer)
         "Listen to the echoes, adapt to uncertainty"

Input: 
    - Shippers S with capacities C
    - Packages P with weights w and destinations L
    - Travel time matrix T
    - Delivery probability distributions P(l, t)
    - Reward parameters (R_success, R_failure, R_callback)

Output:
    - Routing decisions for each shipper over time
    - Total accumulated reward

Initialize:
    state ← initial_state(S, P, L)
    total_reward ← 0
    episode_complete ← False

While NOT episode_complete:
    
    // Check for callback events
    If callback_event_occurred():
        callback ← get_callback_event()
        decision ← handle_callback(state, callback)
        If decision == ACCEPT:
            update_route_plan(best_shipper, callback.package)
    
    // For each shipper at decision point
    For each shipper in get_active_shippers(state):
        
        // Extract current shipper state
        location ← shipper.location
        packages ← shipper.packages
        capacity ← shipper.remaining_capacity
        
        // Generate feasible actions
        feasible_actions ← generate_feasible_actions(shipper, state)
        
        If length(feasible_actions) == 0:
            Continue
        
        // Evaluate actions using approximate value iteration with rollout
        best_action ← None
        best_value ← -infinity
        
        For each action in feasible_actions:
            // Estimate expected next states (sample multiple outcomes)
            expected_value ← 0
            num_samples ← 10
            
            For sample in range(num_samples):
                // Sample stochastic delivery outcome
                delivery_outcome ← sample_delivery_outcome(action, state)
                next_state ← transition(state, action, delivery_outcome)
                
                // Calculate immediate reward
                immediate_reward ← reward(state, action, next_state)
                
                // Estimate future value using rollout
                future_value ← rollout_policy(next_state, horizon=3)
                
                // Discounted total value
                expected_value += (immediate_reward + γ * future_value)
            
            expected_value /= num_samples
            
            If expected_value > best_value:
                best_value ← expected_value
                best_action ← action
        
        // Execute best action
        Execute action best_action for shipper
        
        // Update state based on actual outcome
        actual_outcome ← observe_delivery_outcome()
        state ← transition(state, best_action, actual_outcome)
        total_reward += reward(state, best_action, state)
        
        // Check for new callbacks
        process_callback_arrivals(state)
    
    // Check termination conditions
    If all_packages_delivered_or_failed(state) OR time_limit_reached(state):
        episode_complete ← True

Return total_reward, state.delivery_statistics
```

### 2.10 Computational Complexity

**Time Complexity per decision:**
- State space: O(n × 2^m) in worst case, but pruned to O(n × m) with route-based representation
- Action space: O(k) where k = number of feasible next locations
- Rollout samples: O(s × h × k) where s = samples, h = horizon, k = actions per step
- **Overall:** O(n × m × s × h × k) per decision epoch

**Space Complexity:**
- O(n × m) for state storage
- O(k × s) for rollout simulation

---

## 3. Baseline Algorithms

### 3.1 Baseline 1: Greedy Nearest Neighbor with Callback Queue (GNN-CB)

**Reference:** Clarke, G., & Wright, J. W. (1964). "Scheduling of vehicles from a central depot to a number of delivery points." *Operations Research*, 12(4), 568-581.

**Description:**  
A myopic greedy algorithm that always selects the nearest undelivered package, with a simple callback queue mechanism that prioritizes callbacks based on waiting time.

**Pseudocode:**

```
Algorithm: GNN-CB (Greedy Nearest Neighbor with Callback Queue)

Input: Same as ECHO

Initialize:
    state ← initial_state(S, P, L)
    callback_queue ← empty priority queue
    total_reward ← 0

While NOT all_packages_processed():
    
    // Check callback queue first
    If callback_queue NOT empty:
        callback ← callback_queue.pop()
        nearest_shipper ← find_nearest_shipper_with_capacity(callback.destination, callback.weight)
        
        If nearest_shipper exists:
            // Calculate detour cost
            detour_cost ← cost(nearest_shipper.location, callback.destination, nearest_shipper.load)
            
            // Accept if within threshold
            If detour_cost < MAX_DETOUR_THRESHOLD:
                route_to_location(nearest_shipper, callback.destination)
                attempt_delivery(nearest_shipper, callback.package)
                continue
    
    // For each shipper, greedily select nearest package
    For each shipper in active_shippers:
        
        If shipper.packages is empty:
            continue
        
        // Find nearest undelivered package considering capacity
        nearest_package ← None
        min_distance ← infinity
        
        For each package in shipper.packages:
            distance ← T[shipper.location][package.destination]
            
            If distance < min_distance:
                min_distance ← distance
                nearest_package ← package
        
        // Move to nearest package location
        shipper.location ← nearest_package.destination
        shipper.time += T[shipper.location][nearest_package.destination]
        
        // Attempt delivery with stochastic outcome
        delivery_success ← random() < P(nearest_package.destination, shipper.time)
        
        If delivery_success:
            // Successful delivery
            total_reward += R_success * time_decay_factor(shipper.time)
            shipper.packages.remove(nearest_package)
            shipper.load -= nearest_package.weight
        Else:
            // Failed delivery
            total_reward += R_failure
            nearest_package.attempt_count += 1
            
            // Add to callback queue with probability
            callback_occurs ← random() < P_callback
            If callback_occurs:
                callback_time ← shipper.time + sample_callback_delay()
                priority ← callback_time  // FIFO ordering
                callback_queue.add((nearest_package, callback_time), priority)

Return total_reward
```

**Advantages:**
- Simple and fast (O(m log m) per shipper)
- No lookahead computation required
- Easy to implement

**Disadvantages:**
- Myopic: doesn't consider future implications
- Poor callback handling (FIFO only)
- No capacity optimization
- Suboptimal route structure

---

### 3.2 Baseline 2: Static Route Optimization with Expected Values (SRO-EV)

**Reference:** Laporte, G., Louveaux, F., & Mercure, H. (1992). "The vehicle routing problem with stochastic travel times." *Transportation Science*, 26(3), 161-170.

**Description:**  
Constructs optimal routes upfront using expected delivery probabilities, then follows static routes with minimal dynamic adjustments for callbacks. Uses Clarke-Wright savings algorithm adapted for stochastic demands.

**Pseudocode:**

```
Algorithm: SRO-EV (Static Route Optimization with Expected Values)

Input: Same as ECHO

Phase 1: Initial Route Construction

// Build distance savings matrix considering expected delivery probabilities
savings_matrix ← compute_savings_matrix()

For i in locations:
    For j in locations where j ≠ i:
        // Classical Clarke-Wright savings
        distance_saving ← T[depot][i] + T[depot][j] - T[i][j]
        
        // Adjust by expected delivery probabilities
        expected_prob ← (P(i) + P(j)) / 2
        adjusted_saving ← distance_saving * expected_prob
        
        savings_matrix[i][j] ← adjusted_saving

// Sort savings in descending order
sorted_savings ← sort_descending(savings_matrix)

// Construct routes by merging based on savings
routes ← []
For each shipper in S:
    routes[shipper] ← []

For each (i, j, saving) in sorted_savings:
    // Check if i and j can be merged into a route
    route_i ← find_route_containing(i, routes)
    route_j ← find_route_containing(j, routes)
    
    If route_i ≠ route_j:
        // Check capacity feasibility
        combined_weight ← route_weight(route_i) + route_weight(route_j)
        shipper ← assign_shipper_with_capacity(combined_weight)
        
        If shipper exists:
            // Merge routes
            merged_route ← merge_routes(route_i, route_j, i, j)
            routes[shipper] ← merged_route

Phase 2: Static Route Execution with Callback Handling

For each shipper in S:
    route ← routes[shipper]
    
    While route NOT empty:
        next_location ← route.pop_front()
        
        // Execute movement
        shipper.location ← next_location
        shipper.time += T[shipper.location][next_location]
        
        // Attempt delivery
        package ← packages_for_location(next_location)
        delivery_success ← random() < P(next_location, shipper.time)
        
        If delivery_success:
            total_reward += R_success * time_decay_factor(shipper.time)
            shipper.load -= package.weight
        Else:
            total_reward += R_failure
            
            // Handle callback with simple heuristic
            callback_occurs ← random() < P_callback
            If callback_occurs:
                callback_time ← shipper.time + sample_callback_delay()
                
                // Insert callback back into route at optimal position
                If callback_time < route_completion_time AND shipper.capacity_available:
                    insertion_position ← find_best_insertion(route, next_location)
                    route.insert(insertion_position, next_location)

Return total_reward
```

**Advantages:**
- Good route structure from upfront optimization
- Efficient use of savings heuristic
- Considers expected delivery probabilities

**Disadvantages:**
- Static plan doesn't adapt well to stochastic outcomes
- Limited callback handling flexibility
- Requires accurate probability estimates upfront
- Poor performance with high uncertainty

---

### 3.3 Baseline 3: Threshold-based Callback Policy (TH-CB)

**Reference:** Bertazzi, L., & Secomandi, N. (2018). "Faster rollout search for the vehicle routing problem with stochastic demands and restocking." *European Journal of Operational Research*, 270(2), 487-497.

**Description:**  
Uses a simple threshold-based decision rule for callback acceptance: accept callback if expected benefit exceeds threshold. Routes are planned dynamically using nearest neighbor, but callback decisions use a learned threshold value.

**Pseudocode:**

```
Algorithm: TH-CB (Threshold-based Callback Policy)

Input: Same as ECHO
Parameters:
    - θ_accept: callback acceptance threshold
    - θ_priority: callback priority threshold
    - ω_distance: weight for distance factor
    - ω_value: weight for package value factor
    - ω_prob: weight for delivery probability factor

Initialize:
    state ← initial_state(S, P, L)
    callback_queue ← priority queue
    total_reward ← 0

Function: evaluate_callback_score(callback, shipper):
    """Calculate composite score for callback decision"""
    
    // Distance factor (normalized)
    distance ← T[shipper.location][callback.destination]
    max_distance ← max(T)
    distance_score ← 1 - (distance / max_distance)
    
    // Package value factor
    value_score ← callback.package.value / max_package_value
    
    // Delivery probability factor
    prob_score ← P(callback.destination, current_time)
    
    // Time factor (callback freshness)
    time_wait ← current_time - callback.callback_time
    time_score ← exp(-λ * time_wait)  // Exponential decay
    
    // Capacity factor
    capacity_score ← 1 if shipper.capacity >= callback.weight else 0
    
    // Composite score
    score ← ω_distance * distance_score + 
            ω_value * value_score + 
            ω_prob * prob_score + 
            ω_time * time_score + 
            ω_capacity * capacity_score
    
    return score

Main Algorithm:

While NOT all_packages_processed():
    
    // Process callback queue with threshold-based decisions
    If callback_queue NOT empty:
        callback ← callback_queue.peek()  // Look at highest priority
        
        // Find best shipper for this callback
        best_shipper ← None
        best_score ← -infinity
        
        For each shipper in active_shippers:
            score ← evaluate_callback_score(callback, shipper)
            
            If score > best_score:
                best_score ← score
                best_shipper ← shipper
        
        // Accept callback if score exceeds threshold
        If best_score > θ_accept:
            callback_queue.pop()
            
            // Insert callback into shipper's route
            // Calculate insertion cost
            insertion_cost ← calculate_insertion_cost(best_shipper, callback)
            current_route_value ← estimate_route_value(best_shipper.route)
            callback_value ← R_callback_success * P(callback.destination)
            
            // Accept if net benefit is positive
            If callback_value - insertion_cost > 0:
                insert_callback_in_route(best_shipper, callback)
                continue
    
    // Dynamic routing for regular deliveries
    For each shipper in active_shippers:
        
        If shipper.packages is empty:
            continue
        
        // Select next package using weighted scoring
        best_package ← None
        best_score ← -infinity
        
        For each package in shipper.packages:
            distance ← T[shipper.location][package.destination]
            prob ← P(package.destination, shipper.time)
            
            // Weighted score
            score ← (prob * R_success - distance * cost_per_km) / (1 + distance)
            
            If score > best_score:
                best_score ← score
                best_package ← package
        
        // Move to selected location
        shipper.location ← best_package.destination
        shipper.time += T[shipper.location][best_package.destination]
        
        // Attempt delivery
        delivery_success ← random() < P(best_package.destination, shipper.time)
        
        If delivery_success:
            total_reward += R_success * time_decay_factor(shipper.time)
            shipper.packages.remove(best_package)
            shipper.load -= best_package.weight
        Else:
            total_reward += R_failure
            best_package.attempt_count += 1
            
            // Add to callback queue with priority score
            callback_occurs ← random() < P_callback
            If callback_occurs:
                callback_time ← shipper.time + sample_callback_delay()
                priority_score ← evaluate_callback_score(
                    Callback(best_package, callback_time), 
                    shipper
                )
                
                // Only add to queue if above priority threshold
                If priority_score > θ_priority:
                    callback_queue.add(
                        Callback(best_package, callback_time), 
                        priority=priority_score
                    )

Return total_reward
```

**Advantages:**
- Explicit callback decision mechanism
- Tunable thresholds for different scenarios
- Balances immediate routing with callback handling
- O(m log m) complexity per decision

**Disadvantages:**
- Threshold selection is problem-dependent
- No global optimization
- Limited lookahead
- Doesn't coordinate multiple shippers optimally

---

## 4. Algorithm Comparison Matrix

| Algorithm | Approach | Callback Handling | Lookahead | Complexity | Key Strength |
|-----------|----------|-------------------|-----------|------------|--------------|
| **ECHO** (Proposed) 🎯 | MDP with rollout | Dynamic value-based | 3-step rollout | O(n×m×s×h×k) | Optimal balance of accuracy and computation |
| **GNN-CB** | Greedy nearest | FIFO queue | None | O(m log m) | Speed and simplicity |
| **SRO-EV** | Static optimization | Minimal (re-insertion) | Full route upfront | O(m² log m) | Good initial routes |
| **TH-CB** | Threshold-based | Weighted scoring | None | O(m log m) | Tunable callback behavior |

---

## 5. Test Scenarios

### 5.1 Scenario Dimensions

Each scenario varies along these dimensions:
- **Number of shippers** (n): {2, 3, 5}
- **Number of packages** (m): {10, 20, 50}
- **Number of locations** (k): {5, 10, 15}
- **Delivery uncertainty**: {Low (0.8-0.9), Medium (0.6-0.8), High (0.3-0.6)}
- **Callback rate**: {Low (0.2), Medium (0.5), High (0.8)}
- **Network topology**: {Clustered, Uniform, Hub-spoke}

### 5.2 Detailed Scenarios

#### Scenario 1: Low Uncertainty - Sparse Network
```yaml
name: "Low_Uncertainty_Sparse"
description: "Favorable conditions with high delivery success rates"
parameters:
  n_shippers: 2
  n_packages: 10
  n_locations: 5
  shipper_capacities: [50, 50]  # kg
  package_weights: uniform(2, 8)  # kg
  delivery_probability: uniform(0.80, 0.95)
  callback_probability: 0.20
  network_type: "clustered"
  locations: [[0, 0], [10, 5], [15, 10], [8, 15], [5, 8]]  # (x, y) coordinates
  time_matrix: euclidean_distance(locations)
  R_success_base: 100
  R_failure: -50
  R_callback_success: 80
  cost_per_km_kg: 0.5
  time_window: 120  # minutes
```

#### Scenario 2: High Uncertainty - Dense Network
```yaml
name: "High_Uncertainty_Dense"
description: "Challenging conditions with frequent failures and callbacks"
parameters:
  n_shippers: 3
  n_packages: 20
  n_locations: 10
  shipper_capacities: [40, 50, 45]
  package_weights: uniform(3, 10)
  delivery_probability: uniform(0.30, 0.60)
  callback_probability: 0.80
  network_type: "uniform"
  locations: random_uniform(20, 20, 10)  # 10 locations in 20x20 grid
  time_matrix: euclidean_distance(locations)
  R_success_base: 100
  R_failure: -80
  R_callback_success: 70
  cost_per_km_kg: 0.8
  time_window: 180
```

#### Scenario 3: Medium Uncertainty - Hub-Spoke Network
```yaml
name: "Medium_Uncertainty_HubSpoke"
description: "Realistic urban delivery with central depot"
parameters:
  n_shippers: 3
  n_packages: 15
  n_locations: 8
  shipper_capacities: [60, 55, 50]
  package_weights: uniform(2, 12)
  delivery_probability: uniform(0.60, 0.80)
  callback_probability: 0.50
  network_type: "hub_spoke"
  hub_location: [10, 10]
  spoke_locations: [
    [10, 20], [20, 15], [20, 5], [10, 0], 
    [0, 5], [0, 15], [5, 10], [15, 10]
  ]
  time_matrix: hub_spoke_distance(hub, spokes)
  R_success_base: 100
  R_failure: -60
  R_callback_success: 75
  cost_per_km_kg: 0.6
  time_window: 150
```

#### Scenario 4: Capacity-Constrained
```yaml
name: "Capacity_Constrained"
description: "Heavy packages with limited shipper capacity"
parameters:
  n_shippers: 2
  n_packages: 12
  n_locations: 6
  shipper_capacities: [30, 35]  # Tight capacity
  package_weights: uniform(8, 15)  # Heavy packages
  delivery_probability: uniform(0.70, 0.85)
  callback_probability: 0.40
  network_type: "clustered"
  locations: generate_clusters(n_clusters=2, points_per_cluster=3)
  time_matrix: euclidean_distance(locations)
  R_success_base: 120
  R_failure: -70
  R_callback_success: 90
  cost_per_km_kg: 1.0  # Higher cost due to heavy loads
  time_window: 100
```

#### Scenario 5: Time-Critical Delivery
```yaml
name: "Time_Critical"
description: "Short time window with high time decay on rewards"
parameters:
  n_shippers: 4
  n_packages: 20
  n_locations: 12
  shipper_capacities: [50, 50, 45, 55]
  package_weights: uniform(3, 7)
  delivery_probability: uniform(0.65, 0.85)
  callback_probability: 0.60
  network_type: "uniform"
  locations: random_uniform(25, 25, 12)
  time_matrix: euclidean_distance(locations)
  R_success_base: 150  # High base reward
  time_decay_lambda: 0.05  # Steep decay: R = R_base * exp(-λ * t)
  R_failure: -100
  R_callback_success: 100
  cost_per_km_kg: 0.7
  time_window: 60  # Very short window
```

#### Scenario 6: Large-Scale Deployment
```yaml
name: "Large_Scale"
description: "Large fleet with many deliveries"
parameters:
  n_shippers: 5
  n_packages: 50
  n_locations: 15
  shipper_capacities: [60, 55, 50, 65, 55]
  package_weights: uniform(2, 10)
  delivery_probability: uniform(0.60, 0.80)
  callback_probability: 0.50
  network_type: "uniform"
  locations: random_uniform(30, 30, 15)
  time_matrix: euclidean_distance(locations)
  R_success_base: 100
  R_failure: -60
  R_callback_success: 75
  cost_per_km_kg: 0.6
  time_window: 200
```

#### Scenario 7: Heterogeneous Delivery Probabilities
```yaml
name: "Heterogeneous_Probability"
description: "Locations with vastly different success rates"
parameters:
  n_shippers: 3
  n_packages: 18
  n_locations: 9
  shipper_capacities: [50, 50, 50]
  package_weights: uniform(3, 8)
  delivery_probability:  # Specified per location
    location_1: 0.90  # Residential (high success)
    location_2: 0.85
    location_3: 0.80
    location_4: 0.70
    location_5: 0.60
    location_6: 0.50  # Commercial (medium success)
    location_7: 0.40
    location_8: 0.30  # Industrial (low success)
    location_9: 0.25
  callback_probability: 0.55
  network_type: "mixed"
  locations: generate_mixed_zones(residential=3, commercial=3, industrial=3)
  time_matrix: euclidean_distance(locations)
  R_success_base: 100
  R_failure: -70
  R_callback_success: 80
  cost_per_km_kg: 0.6
  time_window: 150
```

#### Scenario 8: Time-Dependent Delivery Probabilities
```yaml
name: "Time_Dependent_Probability"
description: "Success rates vary by time of day"
parameters:
  n_shippers: 3
  n_packages: 15
  n_locations: 8
  shipper_capacities: [50, 55, 50]
  package_weights: uniform(3, 9)
  delivery_probability_function:
    # P(success) = base_prob * time_multiplier(t)
    base_probability: uniform(0.60, 0.75)
    time_multipliers:
      morning (0-60 min): 0.70  # People at work
      midday (60-120 min): 0.85  # Higher availability
      afternoon (120-180 min): 1.00  # Peak availability
      evening (180-240 min): 0.80  # Moderate availability
  callback_probability: 0.50
  network_type: "clustered"
  locations: generate_clusters(n_clusters=2, points_per_cluster=4)
  time_matrix: euclidean_distance(locations)
  R_success_base: 110
  R_failure: -65
  R_callback_success: 85
  cost_per_km_kg: 0.7
  time_window: 240
```

#### Scenario 9: High Callback Responsiveness
```yaml
name: "High_Callback_Responsiveness"
description: "Frequent callbacks require dynamic rerouting"
parameters:
  n_shippers: 4
  n_packages: 25
  n_locations: 12
  shipper_capacities: [50, 50, 50, 50]
  package_weights: uniform(3, 8)
  delivery_probability: uniform(0.50, 0.70)
  callback_probability: 0.85  # Very high
  callback_delay_distribution: exponential(mean=10)  # Minutes
  callback_window: 30  # Must respond within 30 minutes
  network_type: "uniform"
  locations: random_uniform(25, 25, 12)
  time_matrix: euclidean_distance(locations)
  R_success_base: 100
  R_failure: -80
  R_callback_success: 120  # High reward for successful callback
  R_callback_ignore: -150  # Severe penalty for ignored callback
  cost_per_km_kg: 0.7
  time_window: 180
```

#### Scenario 10: Multi-Modal Cost Structure
```yaml
name: "Multi_Modal_Cost"
description: "Different cost structures for different times and locations"
parameters:
  n_shippers: 3
  n_packages: 20
  n_locations: 10
  shipper_capacities: [55, 50, 55]
  package_weights: uniform(4, 10)
  delivery_probability: uniform(0.65, 0.80)
  callback_probability: 0.50
  network_type: "mixed"
  locations: generate_mixed_zones(urban=5, suburban=3, rural=2)
  time_matrix: euclidean_distance(locations)
  cost_structure:
    # Cost = base_cost * distance * weight * zone_multiplier * time_multiplier
    urban_cost_multiplier: 1.5  # Higher cost due to traffic
    suburban_cost_multiplier: 1.0
    rural_cost_multiplier: 0.8
    peak_time_multiplier: 2.0  # Rush hour (60-120 min)
    normal_time_multiplier: 1.0
  R_success_base: 100
  R_failure: -70
  R_callback_success: 85
  base_cost_per_km_kg: 0.6
  time_window: 200
```

---

## 6. Evaluation Metrics

### 6.1 Primary Performance Metrics

#### 6.1.1 Total Reward
```python
def total_reward(deliveries, failures, callbacks, costs):
    """
    Overall performance metric combining all factors
    """
    success_rewards = sum([
        R_success_base * time_decay(d.delivery_time) 
        for d in deliveries if d.successful
    ])
    
    failure_penalties = len(failures) * R_failure
    
    callback_rewards = sum([
        R_callback_success 
        for c in callbacks if c.successful
    ])
    
    total_costs = sum(costs)
    
    return success_rewards + failure_penalties + callback_rewards - total_costs
```

#### 6.1.2 Delivery Success Rate
```python
def delivery_success_rate(deliveries):
    """
    Percentage of packages successfully delivered
    """
    successful = len([d for d in deliveries if d.successful])
    total_attempts = len(deliveries)
    return (successful / total_attempts) * 100
```

#### 6.1.3 First-Attempt Success Rate
```python
def first_attempt_success_rate(deliveries):
    """
    Success rate on first delivery attempt (excludes callbacks)
    """
    first_attempts = [d for d in deliveries if d.attempt_number == 1]
    successful_first = [d for d in first_attempts if d.successful]
    return (len(successful_first) / len(first_attempts)) * 100
```

### 6.2 Callback Performance Metrics

#### 6.2.1 Callback Response Rate
```python
def callback_response_rate(callbacks):
    """
    Percentage of callbacks accepted and serviced
    """
    accepted = len([c for c in callbacks if c.accepted])
    total = len(callbacks)
    return (accepted / total) * 100 if total > 0 else 0
```

#### 6.2.2 Callback Response Time
```python
def average_callback_response_time(callbacks):
    """
    Average time between callback and re-delivery attempt
    """
    response_times = [
        c.reattempt_time - c.callback_time 
        for c in callbacks if c.accepted
    ]
    return np.mean(response_times) if len(response_times) > 0 else float('inf')
```

#### 6.2.3 Callback Success Rate
```python
def callback_success_rate(callbacks):
    """
    Success rate for callback deliveries
    """
    accepted_callbacks = [c for c in callbacks if c.accepted]
    successful_callbacks = [c for c in accepted_callbacks if c.successful]
    return (len(successful_callbacks) / len(accepted_callbacks)) * 100 if len(accepted_callbacks) > 0 else 0
```

### 6.3 Efficiency Metrics

#### 6.3.1 Average Delivery Time
```python
def average_delivery_time(deliveries):
    """
    Mean time from package assignment to successful delivery
    """
    successful = [d for d in deliveries if d.successful]
    delivery_times = [d.delivery_time - d.assignment_time for d in successful]
    return np.mean(delivery_times)
```

#### 6.3.2 Total Distance Traveled
```python
def total_distance_traveled(shipper_routes):
    """
    Sum of distances traveled by all shippers
    """
    total_distance = 0
    for shipper_route in shipper_routes:
        for i in range(len(shipper_route) - 1):
            total_distance += distance(shipper_route[i], shipper_route[i+1])
    return total_distance
```

#### 6.3.3 Capacity Utilization
```python
def average_capacity_utilization(shippers):
    """
    Average percentage of capacity used across all shippers
    """
    utilizations = []
    for shipper in shippers:
        total_load_over_time = shipper.load_history  # Load at each time step
        avg_load = np.mean(total_load_over_time)
        utilization = (avg_load / shipper.capacity) * 100
        utilizations.append(utilization)
    return np.mean(utilizations)
```

#### 6.3.4 Cost Efficiency
```python
def cost_per_successful_delivery(total_cost, successful_deliveries):
    """
    Average cost per successfully delivered package
    """
    return total_cost / len(successful_deliveries) if len(successful_deliveries) > 0 else float('inf')
```

### 6.4 Route Quality Metrics

#### 6.4.1 Route Deviation Score
```python
def route_deviation_score(planned_routes, actual_routes):
    """
    Measure of deviation from planned routes (for ECHO vs baselines)
    Lower score = more adaptive
    """
    deviations = []
    for planned, actual in zip(planned_routes, actual_routes):
        # Edit distance between planned and actual sequences
        deviation = edit_distance(planned.sequence, actual.sequence)
        deviations.append(deviation / len(planned.sequence))
    return np.mean(deviations)
```

#### 6.4.2 Makespan
```python
def makespan(shippers):
    """
    Maximum completion time among all shippers
    """
    completion_times = [shipper.completion_time for shipper in shippers]
    return max(completion_times)
```

### 6.5 Robustness Metrics

#### 6.5.1 Performance Variability
```python
def performance_variability(rewards_over_episodes):
    """
    Standard deviation of total reward across multiple runs
    Lower = more robust algorithm
    """
    return np.std(rewards_over_episodes)
```

#### 6.5.2 Worst-Case Performance
```python
def worst_case_performance_ratio(algorithm_rewards, optimal_rewards):
    """
    Ratio of algorithm performance to optimal in worst case
    """
    min_algorithm = min(algorithm_rewards)
    min_optimal = min(optimal_rewards)
    return min_algorithm / min_optimal if min_optimal != 0 else 0
```

### 6.6 Fairness Metrics

#### 6.6.1 Shipper Workload Balance
```python
def shipper_workload_balance(shippers):
    """
    Coefficient of variation in number of deliveries per shipper
    Lower = more balanced
    """
    deliveries_per_shipper = [len(shipper.completed_deliveries) for shipper in shippers]
    mean_deliveries = np.mean(deliveries_per_shipper)
    std_deliveries = np.std(deliveries_per_shipper)
    return std_deliveries / mean_deliveries if mean_deliveries > 0 else 0
```

### 6.7 Computational Metrics

#### 6.7.1 Average Decision Time
```python
def average_decision_time(decision_times):
    """
    Mean time to make a routing decision (in milliseconds)
    """
    return np.mean(decision_times)
```

#### 6.7.2 Scalability Factor
```python
def scalability_factor(times_vs_problem_size):
    """
    Slope of log-log plot of runtime vs problem size
    """
    log_sizes = np.log(times_vs_problem_size['sizes'])
    log_times = np.log(times_vs_problem_size['times'])
    slope, _ = np.polyfit(log_sizes, log_times, 1)
    return slope  # ~1 for linear, ~2 for quadratic, etc.
```

---

## 7. Evaluation Framework

### 7.1 Experimental Setup

```python
class ExperimentRunner:
    """
    Orchestrates evaluation of all algorithms across all scenarios
    """
    
    def __init__(self, algorithms, scenarios, num_runs=30):
        self.algorithms = algorithms  # [ECHO, GNN-CB, SRO-EV, TH-CB]
        self.scenarios = scenarios    # 10 scenarios defined above
        self.num_runs = num_runs      # Statistical significance
        self.results = {}
    
    def run_experiments(self):
        """
        Run all algorithms on all scenarios multiple times
        """
        for scenario in self.scenarios:
            print(f"Running Scenario: {scenario.name}")
            scenario_results = {alg.name: [] for alg in self.algorithms}
            
            for run_id in range(self.num_runs):
                # Set random seed for reproducibility
                np.random.seed(run_id)
                
                # Initialize scenario instance
                problem_instance = scenario.generate_instance()
                
                for algorithm in self.algorithms:
                    # Run algorithm
                    start_time = time.time()
                    result = algorithm.solve(problem_instance)
                    end_time = time.time()
                    
                    # Collect metrics
                    metrics = self.compute_metrics(result, problem_instance)
                    metrics['runtime'] = end_time - start_time
                    
                    scenario_results[algorithm.name].append(metrics)
            
            self.results[scenario.name] = scenario_results
        
        return self.results
    
    def compute_metrics(self, result, problem_instance):
        """
        Compute all evaluation metrics for a single run
        """
        metrics = {
            # Primary metrics
            'total_reward': total_reward(result),
            'delivery_success_rate': delivery_success_rate(result.deliveries),
            'first_attempt_success_rate': first_attempt_success_rate(result.deliveries),
            
            # Callback metrics
            'callback_response_rate': callback_response_rate(result.callbacks),
            'callback_response_time': average_callback_response_time(result.callbacks),
            'callback_success_rate': callback_success_rate(result.callbacks),
            
            # Efficiency metrics
            'average_delivery_time': average_delivery_time(result.deliveries),
            'total_distance_traveled': total_distance_traveled(result.routes),
            'capacity_utilization': average_capacity_utilization(result.shippers),
            'cost_per_delivery': cost_per_successful_delivery(result.total_cost, result.deliveries),
            
            # Route quality metrics
            'makespan': makespan(result.shippers),
            
            # Fairness metrics
            'workload_balance': shipper_workload_balance(result.shippers),
        }
        
        return metrics
```

### 7.2 Statistical Analysis

```python
class StatisticalAnalyzer:
    """
    Performs statistical tests to compare algorithm performance
    """
    
    def compare_algorithms(self, results):
        """
        Statistical comparison of algorithms
        """
        comparisons = {}
        
        for scenario_name, scenario_results in results.items():
            print(f"\n=== Analysis for {scenario_name} ===")
            
            # Extract rewards for each algorithm
            algorithm_rewards = {
                alg_name: [run['total_reward'] for run in runs]
                for alg_name, runs in scenario_results.items()
            }
            
            # Compute statistics
            stats = {}
            for alg_name, rewards in algorithm_rewards.items():
                stats[alg_name] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards),
                    'median': np.median(rewards),
                    'q25': np.percentile(rewards, 25),
                    'q75': np.percentile(rewards, 75)
                }
            
            # Pairwise comparisons (t-tests)
            algorithm_names = list(algorithm_rewards.keys())
            pairwise_comparisons = {}
            
            for i in range(len(algorithm_names)):
                for j in range(i+1, len(algorithm_names)):
                    alg1, alg2 = algorithm_names[i], algorithm_names[j]
                    
                    # Perform two-sample t-test
                    t_stat, p_value = scipy.stats.ttest_ind(
                        algorithm_rewards[alg1], 
                        algorithm_rewards[alg2]
                    )
                    
                    # Effect size (Cohen's d)
                    mean1, mean2 = stats[alg1]['mean'], stats[alg2]['mean']
                    std1, std2 = stats[alg1]['std'], stats[alg2]['std']
                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    pairwise_comparisons[f"{alg1}_vs_{alg2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'cohens_d': cohens_d,
                        'mean_difference': mean1 - mean2
                    }
            
            comparisons[scenario_name] = {
                'statistics': stats,
                'pairwise_comparisons': pairwise_comparisons
            }
        
        return comparisons
```

### 7.3 Visualization

```python
class ResultsVisualizer:
    """
    Generate plots and tables for results presentation
    """
    
    def plot_algorithm_comparison(self, results, metric='total_reward'):
        """
        Box plot comparing algorithms across scenarios
        """
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, (scenario_name, scenario_results) in enumerate(results.items()):
            ax = axes[idx]
            
            # Prepare data for box plot
            data = [
                [run[metric] for run in runs]
                for runs in scenario_results.values()
            ]
            labels = list(scenario_results.keys())
            
            ax.boxplot(data, labels=labels)
            ax.set_title(scenario_name)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'algorithm_comparison_{metric}.png')
    
    def plot_performance_profile(self, results, metric='total_reward'):
        """
        Performance profile showing cumulative distribution of performance ratios
        """
        # Compute performance ratios relative to best algorithm per instance
        performance_ratios = {alg: [] for alg in results[list(results.keys())[0]].keys()}
        
        for scenario_results in results.values():
            for run_idx in range(len(list(scenario_results.values())[0])):
                # Get performance of all algorithms for this run
                performances = {
                    alg: scenario_results[alg][run_idx][metric]
                    for alg in scenario_results.keys()
                }
                
                # Best performance for this instance
                best_performance = max(performances.values())
                
                # Compute ratios
                for alg in performances:
                    ratio = performances[alg] / best_performance if best_performance > 0 else 0
                    performance_ratios[alg].append(ratio)
        
        # Plot cumulative distribution
        plt.figure(figsize=(10, 6))
        
        for alg, ratios in performance_ratios.items():
            sorted_ratios = np.sort(ratios)
            cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
            plt.plot(sorted_ratios, cumulative, label=alg, linewidth=2)
        
        plt.xlabel('Performance Ratio')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Performance Profile - {metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'performance_profile_{metric}.png')
    
    def generate_summary_table(self, results):
        """
        Generate LaTeX table with summary statistics
        """
        # Aggregate across all scenarios
        aggregated = {}
        
        for scenario_results in results.values():
            for alg, runs in scenario_results.items():
                if alg not in aggregated:
                    aggregated[alg] = {metric: [] for metric in runs[0].keys()}
                
                for run in runs:
                    for metric, value in run.items():
                        aggregated[alg][metric].append(value)
        
        # Compute mean and std for each metric
        summary = {}
        for alg in aggregated:
            summary[alg] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for metric, values in aggregated[alg].items()
            }
        
        # Generate LaTeX table
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{l|cccc}\n"
        latex += "\\hline\n"
        latex += "Metric & ECHO & GNN-CB & SRO-EV & TH-CB \\\\\n"
        latex += "\\hline\n"
        
        metrics_to_show = [
            'total_reward', 'delivery_success_rate', 'callback_response_rate', 
            'average_delivery_time', 'cost_per_delivery'
        ]
        
        for metric in metrics_to_show:
            latex += f"{metric.replace('_', ' ').title()} & "
            values = []
            for alg in ['ECHO', 'GNN-CB', 'SRO-EV', 'TH-CB']:
                mean = summary[alg][metric]['mean']
                std = summary[alg][metric]['std']
                values.append(f"{mean:.2f} ± {std:.2f}")
            latex += " & ".join(values) + " \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\caption{Algorithm Performance Comparison (Mean ± Std)}\n"
        latex += "\\end{table}"
        
        return latex
```

---

## 8. Implementation Guidelines

### 8.1 Code Structure

```
project/
│
├── algorithms/
│   ├── __init__.py
│   ├── arm_cb.py          # ECHO implementation
│   ├── gnn_cb.py          # GNN-CB baseline
│   ├── sro_ev.py          # SRO-EV baseline
│   └── th_cb.py           # TH-CB baseline
│
├── scenarios/
│   ├── __init__.py
│   ├── scenario_generator.py
│   └── scenarios.yaml     # Scenario configurations
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py         # All metric implementations
│   ├── runner.py          # Experiment runner
│   ├── analyzer.py        # Statistical analysis
│   └── visualizer.py      # Plotting and visualization
│
├── utils/
│   ├── __init__.py
│   ├── distance.py        # Distance calculations
│   ├── probability.py     # Probability distributions
│   └── helpers.py         # Utility functions
│
├── tests/
│   ├── test_algorithms.py
│   ├── test_metrics.py
│   └── test_scenarios.py
│
├── main.py                # Main execution script
├── requirements.txt
└── README.md
```

### 8.2 Dependencies

```txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
pyyaml>=5.4.0
tqdm>=4.62.0
pytest>=6.2.0
```

### 8.3 Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python main.py --config scenarios/scenarios.yaml --output results/

# Run specific scenario
python main.py --scenario "High_Uncertainty_Dense" --algorithms ECHO GNN-CB

# Run with custom parameters
python main.py --num-runs 50 --timeout 300 --parallel 4
```

---

## 9. Expected Results and Hypotheses

### 9.1 Performance Hypotheses

**H1:** ECHO will outperform all baselines in total reward across all scenarios.
- **Justification:** Lookahead and value function approximation enable better long-term decision making.

**H2:** GNN-CB will have the fastest computation time but lowest reward.
- **Justification:** Greedy decisions are fast but myopic.

**H3:** SRO-EV will perform well in low-uncertainty scenarios but poorly in high-uncertainty ones.
- **Justification:** Static routes are optimal when outcomes match expectations.

**H4:** TH-CB will show moderate performance with tunability.
- **Justification:** Threshold-based decisions balance simplicity and adaptability.

**H5:** ECHO's advantage will increase with problem complexity (more packages, locations, uncertainty).
- **Justification:** Approximation schemes scale better than exact methods as complexity grows.

### 9.2 Scenario-Specific Predictions

| Scenario | Winner (Expected) | Reason |
|----------|-------------------|---------|
| Low Uncertainty Sparse | SRO-EV | Static routes work well with high predictability |
| High Uncertainty Dense | ECHO | Dynamic adaptation crucial with high failure rates |
| Medium Hub-Spoke | ECHO / TH-CB | Balanced performance, thresholds may work well |
| Capacity Constrained | ECHO | Requires sophisticated capacity planning |
| Time Critical | ECHO | Lookahead essential for time-optimal decisions |
| Large Scale | TH-CB / GNN-CB | Computational efficiency becomes important |
| Heterogeneous Probability | ECHO | Value function captures location-specific rewards |
| Time-Dependent | ECHO | MDP naturally models time-varying transitions |
| High Callback | ECHO | Callback handling is core strength of ECHO |
| Multi-Modal Cost | ECHO | Complex cost structure requires sophisticated modeling |

---

## 10. Extensions and Future Work

### 10.1 Potential Enhancements

1. **Deep Reinforcement Learning:** Replace value function approximation with neural networks
2. **Multi-Objective Optimization:** Pareto-optimal solutions balancing reward, cost, and time
3. **Online Learning:** Adapt delivery probability estimates from historical data
4. **Collaborative Routing:** Explicit coordination between multiple shippers
5. **Customer Preference Learning:** Personalized delivery windows and preferences

### 10.2 Real-World Integration Considerations

- **GPS Integration:** Real-time traffic and location updates
- **Communication System:** Customer notification and callback systems
- **Weather Impacts:** Include weather as state variable affecting delivery probability
- **Vehicle Heterogeneity:** Different vehicle types (bikes, vans, trucks)
- **Driver Fatigue:** Maximum working hours and break requirements

---

## References

1. **Clarke, G., & Wright, J. W. (1964).** "Scheduling of vehicles from a central depot to a number of delivery points." *Operations Research*, 12(4), 568-581.

2. **Laporte, G., Louveaux, F., & Mercure, H. (1992).** "The vehicle routing problem with stochastic travel times." *Transportation Science*, 26(3), 161-170.

3. **Bertazzi, L., & Secomandi, N. (2018).** "Faster rollout search for the vehicle routing problem with stochastic demands and restocking." *European Journal of Operational Research*, 270(2), 487-497.

4. **Ulmer, M. W., Goodson, J. C., Mattfeld, D. C., & Thomas, B. W. (2019).** "On modeling stochastic dynamic vehicle routing problems." *EURO Journal on Transportation and Logistics*, 9(2), 100008.

5. **Bent, R. W., & Van Hentenryck, P. (2004).** "Scenario-based planning for partially dynamic vehicle routing with stochastic customers." *Operations Research*, 52(6), 977-987.

6. **Secomandi, N., & Margot, F. (2009).** "Reoptimization approaches for the vehicle-routing problem with stochastic demands." *Operations Research*, 57(1), 214-230.

7. **Gendreau, M., Jabali, O., & Rei, W. (2016).** "50th anniversary invited article—Future research directions in stochastic vehicle routing." *Transportation Science*, 50(4), 1163-1173.

8. **Ge, L., Zhang, J., Sun, H., & Zhao, Y. (2020).** "Electric vehicle routing problems with stochastic demands and dynamic remedial measures." *Mathematical Problems in Engineering*, 2020.

9. **Zhou, C., Ma, J., & Douge, L. (2023).** "Reinforcement learning-based approach for dynamic vehicle routing problem with stochastic demand." *Computers & Industrial Engineering*, 183, 109475.

10. **Thomas, B. W. (2007).** "Waiting strategies for anticipating service requests from known customer locations." *Transportation Science*, 41(3), 319-331.

---

## Appendix A: Sample Code Snippets

### A.1 ECHO Core Implementation

```python
import numpy as np
from typing import List, Tuple, Dict
import copy

class ECHO:
    """
    ECHO: Efficient Callback Handling Optimizer
    
    Adaptive Route-based MDP with intelligent callback management
    for uncertain last-mile delivery scenarios.
    
    "Listen to the echoes, adapt to uncertainty"
    """
    
    def __init__(self, config: Dict):
        self.gamma = config.get('discount_factor', 0.95)
        self.horizon = config.get('rollout_horizon', 3)
        self.n_samples = config.get('n_rollout_samples', 10)
        self.value_function_weights = None
    
    def solve(self, problem_instance):
        """
        Main solving routine
        """
        state = self.initialize_state(problem_instance)
        total_reward = 0
        episode_history = []
        
        while not self.is_terminal(state):
            # Handle callbacks
            if state.callback_queue:
                self.process_callbacks(state)
            
            # Make decisions for each active shipper
            for shipper in self.get_active_shippers(state):
                action = self.select_action(state, shipper)
                next_state, reward = self.execute_action(state, shipper, action)
                
                total_reward += reward
                episode_history.append((state, action, reward, next_state))
                state = next_state
        
        return {
            'total_reward': total_reward,
            'state': state,
            'history': episode_history
        }
    
    def select_action(self, state, shipper):
        """
        Select best action using approximate value iteration with rollout
        """
        feasible_actions = self.generate_feasible_actions(state, shipper)
        
        if not feasible_actions:
            return None
        
        best_action = None
        best_value = -np.inf
        
        for action in feasible_actions:
            # Estimate Q-value for this state-action pair
            q_value = self.estimate_q_value(state, shipper, action)
            
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action
    
    def estimate_q_value(self, state, shipper, action):
        """
        Estimate Q(s, a) using sampling and rollout
        """
        q_estimate = 0
        
        for _ in range(self.n_samples):
            # Sample outcome
            next_state, immediate_reward = self.sample_transition(
                state, shipper, action
            )
            
            # Estimate future value using rollout
            future_value = self.rollout_value(next_state, self.horizon)
            
            # Q-value estimate
            q_estimate += immediate_reward + self.gamma * future_value
        
        return q_estimate / self.n_samples
    
    def rollout_value(self, state, horizon):
        """
        Estimate value of state using greedy rollout policy
        """
        if horizon == 0 or self.is_terminal(state):
            return 0
        
        simulated_state = copy.deepcopy(state)
        total_value = 0
        
        for h in range(horizon):
            # Greedy policy
            best_reward = 0
            best_next_state = simulated_state
            
            for shipper in self.get_active_shippers(simulated_state):
                actions = self.generate_feasible_actions(simulated_state, shipper)
                
                for action in actions:
                    expected_reward = self.expected_immediate_reward(
                        simulated_state, shipper, action
                    )
                    
                    if expected_reward > best_reward:
                        best_reward = expected_reward
                        # Simulate with expected outcome
                        best_next_state = self.expected_next_state(
                            simulated_state, shipper, action
                        )
            
            total_value += (self.gamma ** h) * best_reward
            simulated_state = best_next_state
            
            if self.is_terminal(simulated_state):
                break
        
        return total_value
```

### A.2 Metric Computation Example

```python
def compute_all_metrics(result, problem_instance):
    """
    Comprehensive metric computation
    """
    metrics = {}
    
    # Extract data
    deliveries = result['state'].completed_deliveries
    shippers = result['state'].shippers
    callbacks = result['state'].callbacks
    routes = result['history']
    
    # Primary metrics
    metrics['total_reward'] = result['total_reward']
    
    successful_deliveries = [d for d in deliveries if d.successful]
    metrics['delivery_success_rate'] = (
        len(successful_deliveries) / len(deliveries) * 100 
        if deliveries else 0
    )
    
    first_attempts = [d for d in deliveries if d.attempt_number == 1]
    successful_first = [d for d in first_attempts if d.successful]
    metrics['first_attempt_success_rate'] = (
        len(successful_first) / len(first_attempts) * 100 
        if first_attempts else 0
    )
    
    # Callback metrics
    accepted_callbacks = [c for c in callbacks if c.accepted]
    metrics['callback_response_rate'] = (
        len(accepted_callbacks) / len(callbacks) * 100 
        if callbacks else 0
    )
    
    if accepted_callbacks:
        response_times = [
            c.reattempt_time - c.callback_time 
            for c in accepted_callbacks
        ]
        metrics['callback_response_time'] = np.mean(response_times)
        
        successful_callbacks = [c for c in accepted_callbacks if c.successful]
        metrics['callback_success_rate'] = (
            len(successful_callbacks) / len(accepted_callbacks) * 100
        )
    else:
        metrics['callback_response_time'] = float('inf')
        metrics['callback_success_rate'] = 0
    
    # Efficiency metrics
    if successful_deliveries:
        delivery_times = [
            d.delivery_time - d.assignment_time 
            for d in successful_deliveries
        ]
        metrics['average_delivery_time'] = np.mean(delivery_times)
    else:
        metrics['average_delivery_time'] = float('inf')
    
    # Distance and cost
    total_distance = 0
    for i in range(len(routes) - 1):
        state1, action1, _, _ = routes[i]
        state2, action2, _, _ = routes[i + 1]
        # Calculate distance moved
        for shipper in state1.shippers:
            shipper2 = [s for s in state2.shippers if s.id == shipper.id][0]
            if shipper.location != shipper2.location:
                total_distance += problem_instance.distance_matrix[
                    shipper.location, shipper2.location
                ]
    metrics['total_distance_traveled'] = total_distance
    
    # Capacity utilization
    utilizations = []
    for shipper in shippers:
        if hasattr(shipper, 'load_history'):
            avg_load = np.mean(shipper.load_history)
            utilization = (avg_load / shipper.capacity) * 100
            utilizations.append(utilization)
    metrics['capacity_utilization'] = (
        np.mean(utilizations) if utilizations else 0
    )
    
    # Cost per delivery
    total_cost = sum(
        action.cost for _, action, _, _ in routes if hasattr(action, 'cost')
    )
    metrics['cost_per_delivery'] = (
        total_cost / len(successful_deliveries) 
        if successful_deliveries else float('inf')
    )
    
    # Route quality
    completion_times = [s.completion_time for s in shippers]
    metrics['makespan'] = max(completion_times) if completion_times else 0
    
    # Fairness
    deliveries_per_shipper = [
        len([d for d in deliveries if d.shipper_id == s.id]) 
        for s in shippers
    ]
    if deliveries_per_shipper:
        mean_del = np.mean(deliveries_per_shipper)
        std_del = np.std(deliveries_per_shipper)
        metrics['workload_balance'] = std_del / mean_del if mean_del > 0 else 0
    else:
        metrics['workload_balance'] = 0
    
    return metrics
```

---

## Conclusion

This document provides a complete specification for implementing and evaluating the uncertain package delivery problem with callback mechanisms. **ECHO (Efficient Callback Handling Optimizer)** leverages route-based Markov Decision Process formulations with approximate dynamic programming to handle the inherent stochasticity and dynamic nature of last-mile delivery. Like sonar using echoes to navigate uncertain waters, ECHO uses callback signals to adaptively optimize routing decisions in real-time.

Three baseline algorithms provide comparison points spanning myopic greedy approaches, static optimization, and threshold-based policies. The 10 diverse scenarios test algorithm performance across varying conditions of uncertainty, scale, network topology, and operational constraints. Comprehensive evaluation metrics enable rigorous statistical comparison of algorithm performance across multiple dimensions including total reward, success rates, callback handling, efficiency, and computational cost.

**The ECHO Advantage:**
- 🎯 **Dynamic Adaptation**: Real-time replanning based on delivery outcomes
- 🔄 **Intelligent Callbacks**: Value-based prioritization of re-delivery requests  
- 📈 **Lookahead Optimization**: 3-step rollout for informed decision-making
- ⚡ **Scalable Design**: Efficient O(n×m×s×h×k) complexity with practical performance

**Next Steps for Developer:**
1. Implement core classes for State, Action, Problem Instance
2. Implement each algorithm following the provided pseudocode
3. Create scenario generator from YAML configurations
4. Implement metric computation functions
5. Build experiment runner and statistical analyzer
6. Execute experiments and generate visualizations
7. Document results and perform sensitivity analysis

**Contact for Questions:** Please reach out if any clarifications are needed during implementation.
