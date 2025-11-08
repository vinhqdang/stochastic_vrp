# APEX: Modern Baseline Algorithms (2020-2025)

## Overview

This document provides two state-of-the-art baseline algorithms from recent literature (2020-2025) to compare against your APEX algorithm for the stochastic vehicle routing problem with uncertain delivery and callbacks.

---

## Baseline 4: POMO (Policy Optimization with Multiple Optima) - 2020

**Reference:** Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). "POMO: Policy optimization with multiple optima for reinforcement learning." *Advances in Neural Information Processing Systems*, 33, 21188-21198.

**Why This Baseline:**
- **Most cited recent work**: Foundation for many 2021-2025 papers (POMO+, MVMoE, RouteFinder, etc.)
- **State-of-the-art performance**: Consistently outperforms classical heuristics
- **Directly applicable**: Works with stochastic/dynamic variants
- **Well-implemented**: Available in RL4CO framework with production-ready code

### 4.1 Algorithm Description

POMO uses a Transformer-based encoder-decoder architecture with a novel training strategy that leverages multiple starting points (optima) to improve policy learning. Unlike traditional RL approaches that learn a single policy, POMO trains by exploring multiple trajectories from different starting nodes simultaneously.

**Key Innovation:** Multiple rollouts from different starting points within a single training iteration, dramatically improving sample efficiency and solution quality.

**Architecture:**
- **Encoder:** Multi-head attention Transformer that processes node features
- **Decoder:** Auto-regressive policy network with attention mechanism
- **Training:** REINFORCE with shared baseline across multiple trajectories

### 4.2 Adaptation for Stochastic VRP with Callbacks

```python
class POMO_Callback:
    """
    POMO adapted for stochastic VRP with uncertain delivery and callbacks
    """
    
    def __init__(self, config):
        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=128,
            num_heads=8,
            num_layers=3,
            ff_dim=512
        )
        
        # Decoder with context
        self.decoder = AttentionDecoder(
            embed_dim=128,
            num_heads=8
        )
        
        # Number of parallel rollouts (POMO's key parameter)
        self.num_starts = config.get('num_starts', 50)  # Use 50 different starting points
        
        # Stochastic elements
        self.delivery_prob_predictor = MLP([128, 64, 1])  # Predict delivery success prob
        self.callback_handler = CallbackValueEstimator([128, 64, 32, 1])
    
    def encode_state(self, problem_instance):
        """
        Encode problem instance with stochastic features
        """
        # Node features: [location, demand, delivery_prob, callback_history, time]
        node_features = torch.cat([
            problem_instance.locations,
            problem_instance.demands.unsqueeze(-1),
            problem_instance.delivery_probs.unsqueeze(-1),
            problem_instance.callback_counts.unsqueeze(-1),
            problem_instance.time_features.unsqueeze(-1)
        ], dim=-1)
        
        # Encode with Transformer
        node_embeddings = self.encoder(node_features)
        graph_embedding = node_embeddings.mean(dim=1)
        
        return node_embeddings, graph_embedding
    
    def select_action(self, node_embeddings, graph_embedding, visited_mask, 
                     current_load, current_time, stochastic_state):
        """
        Select next node using attention mechanism with stochastic awareness
        """
        # Context: current state including stochastic elements
        context = torch.cat([
            graph_embedding,
            current_load.unsqueeze(-1),
            current_time.unsqueeze(-1),
            stochastic_state  # Expected delivery probs, callback queue size, etc.
        ], dim=-1)
        
        # Compute attention scores
        query = self.decoder.query_proj(context)
        keys = self.decoder.key_proj(node_embeddings)
        values = self.decoder.value_proj(node_embeddings)
        
        # Scaled dot-product attention
        scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.decoder.embed_dim)
        
        # Mask visited nodes and infeasible nodes (capacity constraints)
        scores = scores.masked_fill(visited_mask, float('-inf'))
        
        # Stochastic adjustment: favor nodes with high delivery probability
        delivery_probs = self.delivery_prob_predictor(node_embeddings).squeeze(-1)
        stochastic_bonus = torch.log(delivery_probs + 1e-8) * 0.1  # Small bonus
        scores = scores + stochastic_bonus.unsqueeze(0)
        
        # Softmax to get probability distribution
        action_probs = F.softmax(scores, dim=-1)
        
        return action_probs
    
    def handle_callback(self, state, callback_event):
        """
        Callback handling using value-based decision
        """
        # Embed callback package info
        callback_embedding = self.encoder.embed_single_node(callback_event.package_features)
        
        # Evaluate expected value of accepting callback
        for shipper in state.available_shippers:
            shipper_state = encode_shipper_state(shipper)
            
            # Estimate value of diverting to callback
            callback_value = self.callback_handler(
                torch.cat([callback_embedding, shipper_state], dim=-1)
            )
            
            # Estimate opportunity cost (value of continuing current route)
            current_route_value = self.estimate_route_value(shipper.planned_route)
            
            # Make decision
            if callback_value - current_route_value > threshold:
                return ACCEPT_CALLBACK, shipper
        
        return DEFER_CALLBACK, None
    
    def train_step(self, batch):
        """
        POMO training with multiple starting points
        """
        # Encode problem instances
        node_embeddings, graph_embeddings = self.encode_state(batch)
        
        # Generate multiple solutions starting from different nodes
        # This is POMO's key innovation
        all_solutions = []
        all_rewards = []
        
        for start_node in range(self.num_starts):
            # Each trajectory starts from a different node
            solution, log_probs = self.generate_solution(
                node_embeddings, 
                graph_embeddings,
                start_node=start_node
            )
            
            # Evaluate solution with stochastic simulation
            reward = self.evaluate_solution_stochastic(solution, batch, num_samples=5)
            
            all_solutions.append(solution)
            all_rewards.append(reward)
        
        # POMO uses shared baseline: mean of all rewards
        baseline = torch.stack(all_rewards).mean()
        
        # Policy gradient with shared baseline
        advantages = torch.stack(all_rewards) - baseline
        policy_loss = -(advantages * log_probs).mean()
        
        return policy_loss
    
    def solve(self, problem_instance):
        """
        Inference: Generate multiple solutions and return best
        """
        self.eval()
        
        node_embeddings, graph_embeddings = self.encode_state(problem_instance)
        
        best_solution = None
        best_reward = float('-inf')
        
        # Generate solutions from multiple starting points
        for start_node in range(self.num_starts):
            solution = self.greedy_rollout(
                node_embeddings,
                graph_embeddings, 
                start_node=start_node
            )
            
            # Evaluate with Monte Carlo simulation for stochastic elements
            reward = self.evaluate_solution_stochastic(
                solution, 
                problem_instance,
                num_samples=20  # More samples for inference
            )
            
            if reward > best_reward:
                best_reward = reward
                best_solution = solution
        
        return best_solution, best_reward
```

### 4.3 Pseudocode

```
Algorithm: POMO-Callback (POMO adapted for callbacks)

Input:
    - Problem instances with locations, demands, delivery probabilities
    - Number of starting points K (typically 50-100)
    - Callback probability distribution

Training Phase:
    For each epoch:
        Sample batch of problem instances
        
        For each instance in batch:
            # Encode instance
            node_embeddings, graph_embedding ← Encode(instance)
            
            # POMO: Generate K solutions from K different starting points
            solutions ← []
            rewards ← []
            log_probs ← []
            
            For k = 1 to K:
                # Start from k-th node
                current_node ← k
                solution_k ← [current_node]
                log_prob_k ← 0
                visited ← {current_node}
                
                While not all_delivered:
                    # Select next node using attention mechanism
                    action_probs ← AttentionDecoder(node_embeddings, current_state)
                    
                    # Mask visited and infeasible nodes
                    action_probs[visited] ← 0
                    action_probs ← normalize(action_probs)
                    
                    # Sample action
                    next_node ← sample(action_probs)
                    log_prob_k += log(action_probs[next_node])
                    
                    # Simulate delivery with stochastic outcome
                    success ← sample(Bernoulli(delivery_prob[next_node]))
                    
                    If not success:
                        # Callback occurs with some probability
                        callback_occurs ← sample(Bernoulli(callback_prob))
                        If callback_occurs:
                            # Handle callback using value estimation
                            decision ← CallbackHandler(next_node, current_state)
                            If decision == ACCEPT:
                                Insert next_node back into route
                    
                    solution_k.append(next_node)
                    visited.add(next_node)
                    current_node ← next_node
                
                # Evaluate solution (Monte Carlo for stochastic elements)
                reward_k ← EvaluateSolution(solution_k, instance, num_samples=5)
                
                solutions.append(solution_k)
                rewards.append(reward_k)
                log_probs.append(log_prob_k)
            
            # POMO baseline: shared across all K solutions
            baseline ← mean(rewards)
            
            # Compute policy gradient loss
            advantages ← rewards - baseline
            loss ← -mean(advantages * log_probs)
        
        # Update parameters
        Optimize(loss)

Inference Phase:
    Given new problem instance:
        node_embeddings, graph_embedding ← Encode(instance)
        
        best_solution ← None
        best_reward ← -infinity
        
        # Generate K solutions and pick best
        For k = 1 to K:
            solution_k ← GreedyRollout(node_embeddings, start_node=k)
            reward_k ← EvaluateSolution(solution_k, num_samples=20)
            
            If reward_k > best_reward:
                best_reward ← reward_k
                best_solution ← solution_k
        
        Return best_solution, best_reward
```

### 4.4 Implementation Notes

**Hyperparameters:**
```python
config = {
    'embed_dim': 128,
    'num_encoder_layers': 3,
    'num_decoder_layers': 1,
    'num_heads': 8,
    'ff_dim': 512,
    'num_starts': 50,  # POMO parameter: number of starting points
    'learning_rate': 1e-4,
    'batch_size': 64,
    'num_epochs': 100,
    'callback_threshold': 0.3,  # For callback acceptance decision
    'monte_carlo_samples': 5,  # Training
    'monte_carlo_samples_inference': 20,  # Inference
}
```

**Complexity:**
- Training: O(K × N² × L) where K=num_starts, N=num_nodes, L=sequence_length
- Inference: O(K × N²)

**Advantages:**
- State-of-the-art performance on deterministic VRP benchmarks
- Easy to implement with PyTorch
- Highly parallelizable (K rollouts in parallel)
- Available in RL4CO framework

**Disadvantages:**
- Requires significant GPU memory (K parallel rollouts)
- Training time increases linearly with K
- May need adaptation for very dynamic scenarios

---

## Baseline 5: DRL-DU (Deep RL for Dynamic and Uncertain VRP) - 2023

**Reference:** Pan, W., & Liu, S. Q. (2023). "Deep reinforcement learning for the dynamic and uncertain vehicle routing problem." *Applied Intelligence*, 53(1), 405-422.

**Why This Baseline:**
- **Specifically designed for dynamic + uncertain scenarios** (directly relevant)
- **POMDP formulation** handles partial observability of demand
- **Recent (2023)** and well-cited
- **Proven on realistic scenarios** with real-world validation

### 5.1 Algorithm Description

DRL-DU uses a Partially Observable Markov Decision Process (POMDP) framework to handle uncertainty in customer demands that are only revealed upon arrival. The algorithm combines:
1. Actor-Critic architecture with PPO (Proximal Policy Optimization)
2. Observation model that tracks demand uncertainty
3. Dynamic replanning when new information arrives
4. Adaptive policy that balances exploration vs exploitation

**Key Innovation:** Explicit modeling of partial observability + dynamic decision-making that adapts to revealed information in real-time.

### 5.2 Architecture and Components

```python
class DRL_DU_Callback:
    """
    Deep RL for Dynamic and Uncertain VRP with Callback handling
    Based on Pan & Liu (2023) with callback extensions
    """
    
    def __init__(self, config):
        # Actor network (policy)
        self.actor = ActorNetwork(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dims=[256, 128, 64]
        )
        
        # Critic network (value function)
        self.critic = CriticNetwork(
            state_dim=config['state_dim'],
            hidden_dims=[256, 128, 64]
        )
        
        # Observation encoder for partial observability
        self.observation_encoder = ObservationEncoder(
            embed_dim=128
        )
        
        # Belief state tracker
        self.belief_tracker = BeliefStateTracker()
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
    
    def encode_observation(self, observation):
        """
        Encode partial observation into belief state
        
        Observation includes:
        - Known: vehicle position, time, visited customers
        - Uncertain: unvisited customer demands (distribution)
        - Dynamic: newly appeared customers
        """
        # Separate known and uncertain components
        known_state = observation['known_state']
        uncertain_demands = observation['demand_distributions']
        dynamic_arrivals = observation['new_customers']
        
        # Encode using GNN or MLP
        known_embedding = self.observation_encoder.encode_known(known_state)
        
        # For uncertain demands, use distributional embedding
        # E.g., embed mean and variance
        demand_embedding = self.observation_encoder.encode_uncertain(
            uncertain_demands['mean'],
            uncertain_demands['variance']
        )
        
        # Encode dynamic elements
        dynamic_embedding = self.observation_encoder.encode_dynamic(dynamic_arrivals)
        
        # Combine into belief state
        belief_state = torch.cat([
            known_embedding,
            demand_embedding,
            dynamic_embedding
        ], dim=-1)
        
        return belief_state
    
    def select_action(self, belief_state):
        """
        Select action using actor network with PPO
        """
        # Get action probabilities from actor
        action_logits = self.actor(belief_state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def update_belief(self, belief_state, action, observation):
        """
        Update belief state after taking action and observing outcome
        
        This is key for POMDP: maintaining belief over uncertain states
        """
        # Delivery outcome reveals true demand
        true_demand_revealed = observation['delivery_outcome']['actual_demand']
        delivery_success = observation['delivery_outcome']['success']
        
        # Update demand beliefs using Bayesian update
        updated_beliefs = self.belief_tracker.update(
            prior_belief=belief_state['demand_distributions'],
            observation=true_demand_revealed,
            location=action
        )
        
        # Handle callbacks dynamically
        if not delivery_success and observation['callback_occurred']:
            # Add callback to state
            callback_info = {
                'location': action,
                'callback_time': observation['callback_time'],
                'priority': self.compute_callback_priority(observation)
            }
            belief_state['active_callbacks'].append(callback_info)
        
        return belief_state
    
    def compute_callback_priority(self, observation):
        """
        Compute priority score for callback
        """
        # Factors: customer value, time since failure, current position
        time_factor = 1.0 / (1.0 + observation['time_since_failure'])
        distance_factor = 1.0 / (1.0 + observation['distance_to_location'])
        value_factor = observation['customer_value']
        
        priority = 0.4 * time_factor + 0.3 * distance_factor + 0.3 * value_factor
        
        return priority
    
    def handle_callbacks(self, belief_state):
        """
        Decide which callbacks to service
        """
        if len(belief_state['active_callbacks']) == 0:
            return None
        
        # Sort callbacks by priority
        sorted_callbacks = sorted(
            belief_state['active_callbacks'],
            key=lambda x: x['priority'],
            reverse=True
        )
        
        # Use critic to evaluate opportunity cost
        for callback in sorted_callbacks:
            # Estimate value of current plan
            current_plan_value = self.critic(belief_state)
            
            # Estimate value with callback inserted
            hypothetical_state = self.insert_callback(belief_state, callback)
            callback_plan_value = self.critic(hypothetical_state)
            
            # Accept if improves expected value
            if callback_plan_value > current_plan_value:
                return callback
        
        return None
    
    def train_step(self, trajectories):
        """
        PPO training step with trajectory data
        """
        # Unpack trajectories
        states = trajectories['states']
        actions = trajectories['actions']
        old_log_probs = trajectories['log_probs']
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        
        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones)
        values = self.critic(states)
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # K epochs
            # Get current policy
            action_logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            new_values = self.critic(states)
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def solve(self, problem_instance):
        """
        Solve problem instance with dynamic replanning
        """
        self.eval()
        
        # Initialize belief state
        belief_state = self.initialize_belief(problem_instance)
        
        total_reward = 0
        solution = []
        
        while not self.is_terminal(belief_state):
            # Check for callbacks first
            callback = self.handle_callbacks(belief_state)
            if callback is not None:
                # Accept callback - modify plan
                belief_state = self.insert_callback(belief_state, callback)
            
            # Encode observation
            encoded_state = self.encode_observation(belief_state)
            
            # Select action
            action, _, _ = self.select_action(encoded_state)
            
            # Execute action in environment
            observation, reward, done = self.environment_step(action, belief_state)
            
            # Update belief with revealed information
            belief_state = self.update_belief(belief_state, action, observation)
            
            total_reward += reward
            solution.append(action.item())
            
            # Dynamic replanning if major change
            if self.should_replan(observation):
                belief_state = self.replan(belief_state)
        
        return solution, total_reward
```

### 5.3 Pseudocode

```
Algorithm: DRL-DU-Callback (Deep RL for Dynamic Uncertain VRP with Callbacks)

Input:
    - Initial observation (partial: known locations, demand distributions)
    - Actor network π_θ (policy)
    - Critic network V_φ (value function)

Training:
    Initialize actor π_θ and critic V_φ
    
    For each episode:
        # Initialize
        observation ← Initial partial observation
        belief_state ← InitializeBelief(observation)
        trajectory ← []
        
        While not done:
            # Handle callbacks with priority
            If active_callbacks.size() > 0:
                callback ← SelectHighestPriorityCallback(belief_state)
                If EstimateCallbackValue(callback) > CurrentPlanValue:
                    belief_state ← InsertCallback(belief_state, callback)
            
            # Encode partial observation into belief state
            encoded_state ← EncodeObservation(belief_state)
            
            # Select action using policy
            action, log_prob ← π_θ(encoded_state)
            
            # Execute action in environment
            next_observation, reward, done ← Environment.step(action)
            
            # Reveal information (demand, delivery success)
            true_demand ← next_observation['actual_demand']
            delivery_success ← next_observation['success']
            
            # Update belief state with revealed information (Bayesian update)
            belief_state ← UpdateBelief(belief_state, action, next_observation)
            
            # If delivery failed, callback may occur
            If not delivery_success:
                callback_occurs ← Sample(Bernoulli(callback_prob))
                If callback_occurs:
                    callback_time ← Sample(CallbackTimeDistribution)
                    priority ← ComputeCallbackPriority(action, callback_time)
                    belief_state.active_callbacks.add((action, callback_time, priority))
            
            # Store transition
            trajectory.append((encoded_state, action, reward, log_prob))
            
            # Check if replanning needed (major dynamic event)
            If ShouldReplan(next_observation):
                belief_state ← DynamicReplan(belief_state)
            
            observation ← next_observation
        
        # PPO Update
        returns ← ComputeReturns(trajectory)
        advantages ← returns - V_φ(states)
        
        For k = 1 to K_epochs:
            # Actor update with clipped objective
            new_log_probs ← π_θ(states)
            ratio ← exp(new_log_probs - old_log_probs)
            
            surr1 ← ratio * advantages
            surr2 ← clip(ratio, 1-ε, 1+ε) * advantages
            actor_loss ← -mean(min(surr1, surr2))
            
            # Critic update
            critic_loss ← MSE(V_φ(states), returns)
            
            # Update networks
            UpdateParameters(actor_loss + critic_loss)

Inference:
    Given problem instance:
        belief_state ← InitializeBelief(instance)
        solution ← []
        
        While not done:
            # Handle callbacks
            If active_callbacks.size() > 0:
                callback ← SelectHighestPriorityCallback(belief_state)
                If EstimateCallbackValue(callback) > threshold:
                    belief_state ← InsertCallback(belief_state, callback)
            
            # Greedy action selection
            encoded_state ← EncodeObservation(belief_state)
            action ← argmax(π_θ(encoded_state))
            
            # Execute and update
            observation, reward ← Environment.step(action)
            belief_state ← UpdateBelief(belief_state, action, observation)
            
            solution.append(action)
        
        Return solution
```

### 5.4 Implementation Notes

**Hyperparameters:**
```python
config = {
    'state_dim': 256,  # Belief state dimension
    'action_dim': 100,  # Number of possible locations
    'actor_hidden_dims': [256, 128, 64],
    'critic_hidden_dims': [256, 128, 64],
    'learning_rate': 3e-4,
    'clip_epsilon': 0.2,  # PPO clipping
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'k_epochs': 10,  # PPO epochs per update
    'batch_size': 128,
    'trajectory_length': 200,
    'gamma': 0.99,  # Discount factor
    'gae_lambda': 0.95,  # GAE parameter
}
```

**Complexity:**
- Training: O(T × (A + C)) where T=trajectory_length, A=action_space, C=critic_forward
- Inference: O(N × A) where N=num_steps

**Advantages:**
- Explicitly handles partial observability (POMDP)
- Adapts dynamically to revealed information
- PPO provides stable training
- Proven effective on uncertain demand scenarios

**Disadvantages:**
- Requires more training data than supervised methods
- Belief state maintenance adds complexity
- May need careful tuning of PPO hyperparameters

---

## Comparison of All Baselines

| Algorithm | Year | Type | Key Strength | Complexity | Best For |
|-----------|------|------|--------------|------------|----------|
| **APEX** (Your algorithm) | 2025 | MDP + Rollout | Callback handling | O(n×m×s×h×k) | Uncertain delivery + callbacks |
| **POMO** | 2020 | DRL (Transformer) | Multiple optima | O(K×N²) | Large-scale, fast inference |
| **DRL-DU** | 2023 | DRL (PPO+POMDP) | Dynamic + uncertain | O(T×A) | Partial observability |
| **GNN-CB** | 1964/Adapted | Greedy | Speed | O(m log m) | Baseline comparison |
| **SRO-EV** | 1992/Adapted | Static opt | Initial routes | O(m²) | Low uncertainty |
| **TH-CB** | 2018/Adapted | Threshold | Tunability | O(m log m) | Moderate scenarios |

---

## Recommended Evaluation Strategy

### Scenario Selection
Test on scenarios where modern baselines excel:
1. **Large-scale instances** (50-100+ nodes) - POMO advantage
2. **High uncertainty** (30-60% delivery success) - DRL-DU advantage
3. **Dynamic arrivals** (callbacks during execution) - DRL-DU advantage
4. **Time-critical** - POMO (fast inference)

### Metrics to Compare
```python
metrics = {
    # Performance
    'total_reward': [],
    'delivery_success_rate': [],
    'callback_response_rate': [],
    
    # Efficiency
    'total_distance': [],
    'makespan': [],
    'capacity_utilization': [],
    
    # Computational
    'training_time': [],
    'inference_time': [],
    'gpu_memory': [],
    
    # Robustness
    'performance_std': [],  # Across multiple runs
    'generalization': [],   # On unseen distributions
}
```

### Statistical Tests
```python
# Paired t-test for significance
from scipy.stats import ttest_rel

apex_rewards = results['APEX']['total_reward']
pomo_rewards = results['POMO']['total_reward']
drl_du_rewards = results['DRL-DU']['total_reward']

# APEX vs POMO
t_stat, p_value = ttest_rel(apex_rewards, pomo_rewards)
print(f"APEX vs POMO: t={t_stat:.3f}, p={p_value:.4f}")

# APEX vs DRL-DU
t_stat, p_value = ttest_rel(apex_rewards, drl_du_rewards)
print(f"APEX vs DRL-DU: t={t_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)

print(f"Effect size APEX vs POMO: d={cohens_d(apex_rewards, pomo_rewards):.3f}")
```

---

## Implementation Resources

### POMO
- **Original paper**: [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html)
- **Official code**: [GitHub - yd-kwon/POMO](https://github.com/yd-kwon/POMO)
- **RL4CO framework**: [GitHub - ai4co/rl4co](https://github.com/ai4co/rl4co) (includes POMO)
- **Adaptations**: POMO+, MVMoE, RouteFinder all built on POMO base

### DRL-DU
- **Original paper**: [Applied Intelligence 2023](https://link.springer.com/article/10.1007/s10489-022-03456-w)
- **PPO baseline**: [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Similar implementations**: 
  - [VRP-RL by Zhou et al.](https://github.com/chenhao-zhou/RL-for-DVRPSD)
  - Generic POMDP solvers: DESPOT, POMCP

### Quick Start Code Template

```python
# Install dependencies
# pip install torch torch-geometric stable-baselines3 rl4co

# Import baselines
from rl4co.models import POMO
from stable_baselines3 import PPO

# Your APEX algorithm
from algorithms.apex import APEX

# Load your scenarios
scenarios = load_scenarios('scenarios.yaml')

# Initialize algorithms
apex = APEX(config_apex)
pomo = POMO(config_pomo)
drl_du = PPO('MlpPolicy', env, **config_drl_du)

# Run experiments
results = {}
for scenario in scenarios:
    results[scenario.name] = {
        'APEX': apex.solve(scenario),
        'POMO': pomo.solve(scenario),
        'DRL-DU': drl_du.predict(scenario),
    }

# Analyze
analyze_results(results)
```

---

## References

1. **Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020).** POMO: Policy optimization with multiple optima for reinforcement learning. *Advances in Neural Information Processing Systems*, 33, 21188-21198.

2. **Pan, W., & Liu, S. Q. (2023).** Deep reinforcement learning for the dynamic and uncertain vehicle routing problem. *Applied Intelligence*, 53(1), 405-422.

3. **Kool, W., Van Hoof, H., & Welling, M. (2019).** Attention, learn to solve routing problems! *International Conference on Learning Representations*.

4. **Berto, F., et al. (2025).** RL4CO: An Extensive Reinforcement Learning for Combinatorial Optimization Benchmark. *arXiv preprint*.

5. **Zhou, C., Ma, J., Douge, L., Chew, E. P., & Lee, L. H. (2023).** Reinforcement learning-based approach for dynamic vehicle routing problem with stochastic demand. *Computers & Industrial Engineering*, 183, 109475.

6. **Vidal, T. (2022).** Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. *Computers & Operations Research*, 140, 105643.

7. **Drakulic, D., Michel, S., Mai, F., Sors, A., & Andreoli, J. M. (2024).** BQ-NCO: Bisimulation Quotienting for Generalizable Neural Combinatorial Optimization. *NeurIPS 2024*.

8. **Gao, L., Chen, M., Chen, Q., Luo, G., Zhu, N., & Liu, Z. (2024).** Learn to Design the Heuristics for Vehicle Routing Problem. *IEEE Transactions on Intelligent Transportation Systems*.

---

## Conclusion

These two modern baselines (POMO and DRL-DU) represent the state-of-the-art in VRP solving with deep learning:

- **POMO (2020)** is the foundation for most recent neural VRP solvers and provides strong performance with fast inference
- **DRL-DU (2023)** explicitly handles dynamic and uncertain scenarios with POMDP formulation

Both are significantly more sophisticated than the 1960s-2018 baselines and will provide much stronger comparison points for your APEX algorithm. They are also well-documented with available implementations, making them practical to include in your evaluation.

**Recommendation:** Implement POMO first (easier, better documented in RL4CO) as your primary modern baseline, then add DRL-DU for scenarios with high uncertainty and partial observability.
