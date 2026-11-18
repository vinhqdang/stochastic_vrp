# Method sections of the two sources behind the ALP and PPO arms of EXPERIMENT B

Read 2026-11-18 from the local copies in `refs_local/` (full text, all pages).
Page numbers are the journals' own. Everything below is a paraphrase; equations are
restated in our notation where noted, not copied. This file replaces the "method
section not read" status of both sources in `LIT_2025_2026.md` and `EXPERIMENT_B.md`
revision 1.

- **[Z]** L. Zhu, Y. Adulyasak, L.-M. Rousseau. Partial-outsourcing strategy for the
  vehicle routing problem with stochastic demands. *Networks* 87(3):266-288, 2026.
  doi 10.1002/net.70024. `refs_local/zhu2026_networks_partial_outsourcing.pdf` (23 pp.).
- **[K]** S. Kadyrov, A. Azamov, Y. Abdumajitov, C. Turan. Deep reinforcement learning
  for dynamic vehicle routing with demand and traffic uncertainty. *Operations Research
  Perspectives* 15:100351, 2025. doi 10.1016/j.orp.2025.100351.
  `refs_local/kadyrov2025_orp_drl_dvrp.pdf` (12 pp.). Code cited as their ref. [38]:
  github.com/shiralikadyrov/CVRP_uncertainty (not yet inspected; see section 3.4).

---

## 1. Zhu, Adulyasak & Rousseau (Networks 2026)

### 1.1 Problem (pp. 267, 270)

- One private vehicle of integer capacity Q serves N customers from one depot on a
  complete Euclidean network; external carriers take any demand the vehicle cannot
  serve and are paid a fixed price b per unit (p. 267).
- Customer l's demand ξ_l is discrete on {0, …, E}, E ≤ Q, with a **known** pmf
  p_l(e), **independent across customers and of the policy**, revealed on arrival
  (p. 270, section 3.1). Split deliveries are allowed.
- **Failure recourse = outsourcing.** If the residual load q is below ξ_l, the vehicle
  delivers q and the shortfall ξ_l − q is outsourced at cost b·(ξ_l − q); the vehicle
  continues **empty** (q = 0), with no detour to the depot (p. 270; Fig. 1b, p. 268).
- **Preventive decision = restocking.** Before going to the next customer the vehicle
  may return to the depot to refill to Q, at extra distance
  Δ_lj = d_l0 + d_0j − d_lj (p. 271, eq. 8).
- **Routing is re-optimised** at every stage: the next customer is chosen dynamically
  among the unvisited set (p. 270-271).

### 1.2 Decision timing and state (pp. 270-271)

- Stages k ∈ {N, N−1, …, 0} count the unvisited customers. Stage k starts when the
  vehicle has finished serving the current customer. State s_k = (l, q, R_k(l)):
  current location, residual capacity q ∈ {0, …, Q}, unvisited set (p. 271, eq. 1).
- At each stage two decisions: the next customer j ∈ R, and direct (D) versus
  restock-first (R). At stage N only the first customer is chosen; at stage 0 the
  vehicle returns to the depot (p. 271).
- The authors **swap the usual order** (routing first, then restocking) so that the
  restocking decision is embedded in the value function; the Bellman equation becomes
  a single minimisation over j (pp. 271-272, eqs. 11-12). The state space has size
  1 + N(Q+1)2^{N−1} (p. 272).

### 1.3 Value function: decomposition plus ALP (pp. 272-277)

1. **Decomposition** (p. 272-273, eqs. 13-16). The cost-to-go is split into a routing
   travel part and a **penalty part** f (extra restocking distance plus expected
   outsourcing cost) given the next customer. Each part is replaced by a lower bound:
   the travel part by the LP relaxation of a TSP path through the remaining customers
   (p. 273, LP 18, subtour constraints, solved by min-cut separation), and the penalty
   part by an ALP lower bound L.
2. **Penalty MDP** (p. 274, eq. 19.1), restated: for the next customer J,
   - restock (u = 1): Δ_lJ + Σ_e p_J(e) f_{k−1}(J, Q − e, R∖J);
   - direct (u = 0): Σ_{e≤q} p_J(e) f_{k−1}(J, q − e, R∖J) + b Σ_{e>q} p_J(e)(e − q)
     + Σ_{e>q} p_J(e) f_{k−1}(J, 0, R∖J);
   and f is the minimum of the two. Two structural results (p. 274; proofs in
   Appendix B, pp. 286-288): **Prop. 1** the penalty cost is nonincreasing in the
   residual capacity q; **Prop. 2** the restock-or-not choice can be of threshold type
   in q (Fig. B1, p. 288).
3. **Exact LP** (p. 274-275, LP 20): maximise the penalty value of the initial state
   f_{0,N}(Q) subject to f ≤ the right-hand side of each action, for every
   (location, next customer, unvisited set, feasible q). Feasible q at a stage with
   |R| unvisited customers is the range [(Q − (N − |R|)E)^+, Q − e_min] (p. 275).
   Variable reduction via f_{l,R}(q) ≤ f^j_{l,R}(q) gives LP 22 (p. 275).
4. **Affine basis** (p. 275, eqs. 23.1-23.3):
   - f_{0,N}(Q) ≈ θ_{0,0,Q};
   - f_{l,R}(q) ≈ θ_{l,0,q} + Σ_{j∈R} (α_{l,j,q}·Δ_lj + ω_{l,j}·q) for |R| ≥ 2;
   - f_{l,{j}}(q) ≈ θ_{l,0,q} + α_{l,j,q}·Δ_lj + ω_{l,j}·q.
   Parameter dimensions: θ ∈ R^{N·Q+1}, α ∈ R^{(N²−N)(Q+1)}, ω ∈ R^{N²−N}
   (p. 275). **Note: θ and α carry the index q.** So for a fixed location and
   unvisited set the approximation is a free value per residual capacity, plus a term
   linear in q; it is affine across customers, not across loads. The ALP 24 has
   O(N²Q) variables, N²(Q+1) + N² − N in total (p. 276).
5. **Objective = initial state only** (p. 275, eq. 24.1): maximise θ_{0,0,Q}. No
   state-relevance weights over other states. The authors say the basis and the
   state-relevance distribution are chosen "in a standard manner" and name both as the
   main avenue for improvement (pp. 268, 284).
6. **Constraint sampling** (pp. 276-277, section 5.3; after de Farias & Van Roy 2004
   and Adelman 2004). A set of heuristic policies (Appendix A, p. 286: eleven
   candidates, e.g. partial re-optimisation PR^ous after Secomandi & Margot 2009, one-
   and two-step rollout, static rollout, TSP, myopic, and the traditional-recourse
   policy PR^trd, each with a "-de" decomposition variant) is run on sampled demand
   realisations; the visited (location, next customer, unvisited set, restock decision)
   pairs form a pool. For each pooled pair, constraints are added **for every feasible
   residual capacity q and both restock values** (eqs. 26-27, p. 276), so q is not
   sampled, only the routing triple is.
7. **Price-directed policy** (p. 277, section 6, eqs. 28-30): the lower bounds L are
   one-step expectations of the fitted affine functions over the next customer's
   demand (eq. 28.2). At state (l, q, R), for each candidate next customer J:
   restock iff L^R ≤ L^D (eq. 29.1; ties go to restocking); the next customer
   minimises d_lJ + l_tsp + min(L^R, L^D) over customers in the pool (eq. 29.2).

So **yes, Zhu et al.'s value function is an ALP lower bound on the recourse
(penalty) cost-to-go**, and the restock decision compares the one-step expected
approximate cost of the two actions. That is the continuation-value comparison of
our problem. What differs from revision 1's reconstruction is the basis (per-load
intercepts, not affine-plus-hinge in the load), the objective (initial state only,
not a uniform sum over sampled states) and the sampling (routing triples are
sampled; all loads are always constrained).

### 1.4 Numbers (pp. 277-283)

| item | value | page |
|---|---|---|
| LP solver, machine | Gurobi; Intel Core 3.2 GHz, 16 GB RAM | 277 |
| network | customers uniform on a 1000×1000 grid; depot at (0,0) "corner" or (500,500) "midpoint" | 277-278 |
| demand | discrete uniform on {1,…,5} (mean 3); also a normal variant with variance-to-mean 3.66 vs 0.67 for the uniform | 278, 281-282 |
| sizes | N = 10 to 40 in steps of 5 (tables report 5,10,…,40); 20 instances per setting | 278 |
| capacity | Q = round(3N / f̄), filling rate f̄ ∈ {1.9, 2.5, 3.4} | 278 |
| outsourcing price | b ∈ [max Δ_lj / Σ_e p(e)e, min 2d_0j / Σ_e p(e)e], base value at the lower end (Appendix D, pp. 287-288) | 278 |
| constraint sampling | 500 demand samples per heuristic policy (200 already stable) | 278 |
| PD vs traditional recourse | mean −4.06% (midpoint), −0.20% (corner); per-setting range −9.44% to +3.92% | 279, Tables 1-2 |
| PD vs PR^ous (re-optimisation with outsourcing) | PD worse in several settings, e.g. +4.13% (set 9, set 12); "within 5%" | 280, Tables 3-4 |
| CPU time | total dominated by the heuristic preparation (up to ≈ 4.6·10^5 s at N = 40); ALP solve 3 to 1,103 s; policy evaluation < 1 s | 281, Tables 5-6 |
| price sensitivity | total cost rises with b (b^0 to b^max) | 283, Tables 9-10 |

---

## 2. Kadyrov, Azamov, Abdumajitov & Turan (OR Perspectives 2025)

### 2.1 Problem (pp. 3-4)

- Single-vehicle **dynamic CVRP** with stochastic demand and stochastic, time-varying
  travel costs, modelled as a POMDP (p. 3, section 3). Complete directed graph; depot
  at the centre of an n×n integer grid; customers on grid points (p. 3, section 4.1).
- At each step t the vehicle at node i_t with remaining capacity C_t chooses the next
  node a_t; demand d_i is revealed on arrival; only E[d_i] is known before (p. 3).
  Objective: expected total travel cost over a finite horizon; every customer visited
  once; C_t ≥ d_i on visit; return to the depot at the end (p. 3).
- Travel cost per step (p. 3, eq. 1): exponential moving average of the Manhattan
  distance with Gaussian noise and an offset, floored at the distance. Demand (p. 3,
  eq. 2): expected demand d̄_i uniform on {1,…,5}; realised demand is the rounded-down
  normal N(d̄_i, σ_D²) clipped to [1, C−1].
- Table 1 (p. 4): β = 0.07 (smoothing), σ_D = 1.0, σ_T = 1.0, trend offset m = 5.0,
  C = 15, grid n = 8.

The vehicle's preventive choice is routing to the depot (an action among the nodes);
there is no separate stop/continue decision and no recourse cost (an infeasible move
is masked instead, p. 4).

### 2.2 Policy network (pp. 2-4)

- **Encoder** (p. 3): two NNConv layers whose edge networks map edge attributes
  (travel costs) to input×hidden weight matrices, hidden 64, attention-based message
  passing; then a **Transformer encoder with 4 heads, 2 layers, hidden dimension 64**.
- **Memory** (p. 3): a **GRU with hidden size 64** updated each step on the global
  mean-pooled node embedding (POMDP memory).
- **Policy head** (p. 4): one fully connected layer, softmax over feasible actions.
  **Action masking** sets to zero the probability of visited nodes, nodes whose
  expected demand exceeds the remaining capacity, and non-depot nodes when capacity is
  insufficient; if nothing is feasible the vehicle goes to the depot.
- **Value function** (p. 4): a separate network of **two linear layers, hidden 64**,
  used as the baseline for the returns.
- Hybrids (p. 4): GNN-DRL-G restricts the choice to the k = 3 nearest feasible nodes;
  GNN-DRL-CW mixes in Clarke-Wright savings.

### 2.3 PPO settings (p. 4)

| setting | value | page |
|---|---|---|
| reward | −ΔC_travel + 5·ΔN_visited − 0.5 per step | 4 |
| update unit | **one episode at a time** | 4 |
| episode | reset with a random number of customers (10 to 63) on an 8×8 grid; ends at termination or after 5·|N| steps | 4 |
| return / advantage | discounted return with **γ = 0.5**; the value network gives the baseline ("normalise rewards") | 4 |
| epochs | **4 PPO epochs per episode** | 4 |
| clip | ε = 0.2 (standard clipped surrogate) | 4 |
| optimiser | Adam, **constant learning rate 1e-4**, no schedule | 4 |
| gradient clipping | max norm **1.0** | 4 |
| budget | **> 200,000 episodes, > 100 h**; Google Colab, Python 3.11, PyTorch 2.6.0+cu124, Xeon 2 cores 2.0 GHz, Tesla T4 | 4 |
| **not stated** | entropy coefficient (none mentioned), value-loss coefficient, GAE (none mentioned), minibatch size (whole episode implied), advantage normalisation, NNConv internals, weight initialisation, value-network input, activation between the two value layers | 3-4 |

### 2.4 Evaluation and results (pp. 5-10)

- Baselines (p. 5, Table 2): Tabu search (tenure 20), Greedy, Random, Largest-Demand,
  Max-Reachable, Clarke-Wright, Sweep; custom implementations adapted to the dynamic
  setting.
- Sizes: N ∈ {20, 50, 100, 200} customers, grid n ∈ {8, 12, 15}, **100 random
  instances per size**; seven scenarios (standard, demand-only, traffic-only, infinite
  capacity, high traffic σ_T = 10, high demand σ_D = 3, no uncertainty) (pp. 5-9).
  Statistics: Friedman test because Shapiro-Wilk rejected normality (p. 5).
- Run time (p. 6, Table 3): GNN-DRL 0.107 s (N = 20) to 3.555 s (N = 200) per
  instance.
- Costs (Table 4, p. 6, standard scenario): plain GNN-DRL 304 / 743 / 1,916 / 4,539
  for N = 20 / 50 / 100 / 200, against Greedy 262 / 631 / 1,461 / 3,174 and Tabu
  263 / 629 / 1,456 / 3,174. The hybrid GNN-DRL-CW is best (242 / 575 / 1,352 / 2,939).
- **The plain PPO policy (GNN-DRL) is worse than Greedy and Tabu in every size of
  Tables 4-10** (pp. 6-10) and among the worst under infinite capacity (Fig. 6, p. 9).
  Only the heuristic hybrids beat the classical heuristics. The authors note they
  could not compare with other RL methods because of differing formulations (p. 11).

---

## 3. Mapping to our two-action handoff problem

Our problem (EXPERIMENT_A.md section 2.1): a fixed route of m stops, continuous net
load increments g_j (gamma marginals, Gaussian copula ρ = 0.6 across stops), running
load W_k; at epochs k = 1..m−1 on alive paths choose continue or hand off the rest of
the route at the known price H[k−1] (terminal); a breach W_k > B at stop k costs
E[k−1] (terminal); completion costs 0.

### 3.1 ALP arm (Zhu et al.)

**Carries over verbatim**
- The ALP idea: a lower bound on the recourse cost-to-go from an LP whose constraints
  are the Bellman inequalities for both actions (LP 20/24, pp. 274-276).
- The objective: maximise the value of the initial state only (eq. 24.1).
- Constraints for **every feasible load** at each visited (stage, next stop); with a
  fixed route every stage is visited by every policy, so the "sampled" routing triples
  are all of them and the constraint set is complete.
- Independent demands with known marginals (p. 270). We use the empirical marginal of
  each stop from the job's N training paths, as THR does.
- The price-directed decision rule: compare the known cost of the preventive action
  with the one-step expectation of the fitted approximation (eqs. 28.2, 29.1).
- The structural facts: monotonicity of the penalty in residual capacity (Prop. 1)
  becomes "C_k(w) nondecreasing in the load w" in our model; the threshold-type
  decision (Prop. 2) becomes a load threshold per stop.

**Must be adapted**
- **Basis on a fixed route.** With the route fixed, location l = stop k and the
  unvisited set R = {k+1, …, m} is determined by k. Eq. 23.2 then reads
  θ_{k,0,q} + Σ_{j>k} α_{k,j,q}Δ_kj + q·Σ_{j>k} ω_{k,j}, which is a **free value per
  (stage, load)**: a lookup table. Zhu et al.'s basis is therefore *exact* on our
  problem and their ALP is the exact LP of the independent-demand DP. We keep that
  basis as published; we do not invent a coarser one.
- **Discrete loads.** Their q is an integer. Ours is continuous, so loads live on a
  lattice of n_w points on [w_lo, B] and each stop's increments are rounded to the
  lattice spacing (binned empirical pmf). The lattice size is ours [DC].
- **Action semantics.** Their restock pays Δ and continues full; their failure pays
  b·(shortfall) and continues empty. Our handoff pays H[k−1] and our breach pays E[k]
  (both fixed per stop) and both are **terminal**. In LP terms the next-stage value
  after either is 0, which is a special case of their constraint form.
- **Solver.** Gurobi → HiGHS (`scipy.optimize.linprog`). The optimum value is the
  same; with the initial-state objective the optimal **vertex is not unique** (values
  at states reachable only after a handoff-optimal state are free below their exact
  value), and Gurobi and HiGHS may return different vertices. This is a property of
  the published objective, not of our adaptation; we fix the solver options and
  record the vertex (EXPERIMENT_B.md section 3.2).

**Cannot be reproduced**
- Routing re-optimisation, the TSP-path lower bound (LP 18), the decomposition's
  travel part, the next-customer choice (29.2, 30), and the multi-policy heuristic
  pool (Appendix A): with a fixed route there is no routing decision.
- The outsourcing price b and its range (Appendix D): our prices are the route's H
  and E schedules.
- Their instance family (discrete uniform demands, Euclidean grid, re-optimised
  routes) and their headline numbers (−4.06% vs traditional recourse).

**Consequence for B.** On a fixed route, Zhu et al.'s ALP computes the exact
independent-demand DP on a load lattice, i.e. the same object as THR (Yang, Mathur &
Ballou 2000) up to discretisation and the LP's off-path vertex choice. EXPERIMENT_B
revision 2 therefore (a) implements their ALP faithfully as arm **ALP-Z**, (b) uses
the equality with the lattice DP as an exact bug gate, and (c) admits ALP-Z to the
Holm family only if its dev decisions differ from THR's on at least 1% of alive
decision states; otherwise it is reported as coinciding with THR (descriptive).
Revision 1's affine-plus-hinge basis with a uniform objective over sampled loads was
our invention and is withdrawn.

### 3.2 PPO arm (Kadyrov et al.)

**Carries over verbatim**
- Transformer encoder 4 heads, 2 layers, width 64: identical to A's `SetNet`
  configuration (`NetConfig(d=64, layers=2, heads=4)`, ff 128).
- GRU memory of hidden size 64 on the pooled embedding, updated each decision step.
- Policy head: one linear layer, softmax over the feasible actions with masking.
- Value network: two linear layers, hidden 64, as the baseline.
- PPO clipped surrogate with ε = 0.2; **4 epochs per episode, one episode per
  update**; Adam with constant learning rate 1e-4; gradient-norm clip 1.0; no
  schedule; no entropy bonus (none stated).
- Advantage = discounted return minus the value baseline (no GAE stated).
- Budget order of magnitude: ≥ 200,000 training episodes. Our 80,000 fitting paths per
  job give 240,000 episodes at 3 passes.
- γ = 0.5 as one point of the tuning grid (see below).

**Must be adapted**
- **Action set**: {continue, handoff} at k = 1..m−1; no action at k = m (masking).
  Their masking of infeasible nodes has no analogue except "no decision at the last
  stop".
- **Encoder input**: the NNConv layers act on travel-cost edges of a routing graph;
  our route is fixed, so they are dropped and A's FULL stage tokens (one token per
  remaining stop, 11 features) plus the 12 globals enter the transformer directly.
- **Reward**: −(cost)/E0, i.e. −H[k−1]/E0 on handoff, −E[k]/E0 on a breach at k+1,
  0 otherwise. Their shaping terms (+5 per visited customer, −0.5 per step) reward
  route coverage, which our fixed route makes constant; added to our problem they
  would change the optimal stopping rule (the per-step penalty rewards early
  handoffs), so they are not carried over.
- **Discount**: their γ = 0.5 optimises a discounted objective; ours is undiscounted.
  Both γ = 0.5 (source) and γ = 1 (our objective) are tuning-grid points, chosen on
  dev.
- **Initial handoff probability**: their actions never end an episode early; ours
  does. With a uniform initial policy an episode reaches stage k with probability
  ≈ 2^{−k}, starving late stages. The policy-head bias is initialised so that the
  initial handoff probability is 0.05 (stated adaptation, in the arm's favour).
- **Environment**: episodes replay the job's fixed 80,000 fitting paths (equal data,
  A §1.1), not fresh simulations.
- **Hardware**: CPU (12 workers) unless the pilot's device rule picks the GPU
  (EXPERIMENT_B.md section 9).

**Cannot be reproduced**
- Hybrids GNN-DRL-G and GNN-DRL-CW (routing heuristics; their best results come from
  these), the traffic model, cross-size generalisation, their seven scenarios and
  instance sizes.
- The items in section 2.3 marked "not stated" unless the code (ref. [38]) states
  them. **Rule:** before tuning, the implementer reads that repository and records
  each such value in `VERIFY_CITATIONS.md`; a value stated in the code replaces our
  default, "not stated" keeps it.

**Expectation from the source.** In Kadyrov et al.'s own tables the plain PPO policy
never beats the Greedy or Tabu heuristics (pp. 6-10). A PPO arm that is weak on our
problem is therefore consistent with the published behaviour and is reported that
way, not treated as an implementation failure.
