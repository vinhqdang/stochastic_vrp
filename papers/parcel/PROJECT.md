# PARCEL — project spec

**Price-Aware Relay of Context over Endogenous Links**

Target: **AAMAS 2027** (26th Intl. Conf. on Autonomous Agents and
Multiagent Systems), Hanoi, Vietnam, 3–7 May 2027. Main track.
Status tracker: `STATUS.md`. File manifest: `README.md`.

Paper type: **theoretical** (formal model → structural results →
hardness → restricted-regime algorithm), with simulation as secondary
validation rather than the headline.

---

## 1. The pitch

Multi-agent LLM systems pay for coordination in tokens. The prevailing
practice is to forward whole conversation histories or whole context
blobs between agents, because deciding what to withhold is hard and
sending everything is safe. Tokens are billed, so this is a direct,
measurable operating cost, and it grows with the square of the number
of agents when every agent broadcasts to every other.

The obvious framing is influence maximization: pick seeds, let
information diffuse, cover the network cheaply. **That framing does not
survive contact with the problem**, and showing exactly where it breaks
is this paper's first contribution.

## 2. Why the standard toolchain does not transfer

Classical influence maximization (Kempe–Kleinberg–Tardos and the
twenty-plus years built on it) rests on the spread function being
**monotone** and **submodular**, which is what licenses the greedy
`(1 − 1/e)` guarantee and everything downstream of it — reverse
influence sampling, IMM, and the current GPU-accelerated and
learning-based descendants. Our objective has neither property, for two
*independent* reasons.

**(F1) Saturation kills monotonicity.** The objective is task utility,
not the count of informed agents. Delivering additional context to an
agent measurably *degrades* its performance past a point: controlled
benchmarks report step accuracy falling from 43% to 19% as irrelevant
contexts go from 1 to 15, and long-context evaluations show
non-uniform, degrading use of the context window as input length grows.
So there exist instances where adding one transmission strictly lowers
utility. Monotonicity fails not at the margin but for the exact reason
the problem is worth studying.

**(F2) Complementarity kills submodularity.** Submodularity requires
diminishing returns. Two facts can be individually inert and jointly
decisive — "the deadline is Friday" and "the build takes three days"
each change nothing alone and together force a replan. That is
*increasing* returns on a bundle. Even with saturation switched off,
diminishing returns do not hold.

Neither failure is a technicality to be patched. Together they place
the problem outside the class the entire IM scaling literature is built
for, and that gap is the paper's opening.

## 3. What is actually new here

Three axes, in decreasing order of confidence:

1. **Non-monotone, non-submodular dissemination under a global budget.**
   The money constraint is *global* (total tokens billed); the
   degradation mechanism is *per-agent* (each context window saturates
   independently). These are different objects and they pull against
   each other: the budget rewards concentrating spend where marginal
   value is highest, saturation punishes exactly that concentration.
   That tension is the theorem territory.

2. **Endogenous topology.** Temporal/dynamic influence maximization
   exists and is active, but it treats network evolution as
   *exogenous* — the graph changes, you forecast the change, you seed
   against the forecast. In an agent system the orchestrator builds the
   agent graph in response to the task, and what you transmit
   determines who talks to whom next: an agent that learns of a
   dependency goes and contacts the agent that owns it. Seeding
   perturbs the topology it is seeding over. No temporal-IM work found
   so far models that feedback.

3. **Copyable goods, rivalrous attention.** This is where the repo's
   vehicle-routing lineage legitimately connects, and it is a
   *contrast*, not an analogy. In VRP the goods are conserved: what one
   vehicle carries, another does not. Information is free to duplicate,
   so the routing intuition does not transfer. What *does* transfer is
   **capacity**: the receiving agent's context budget is a rivalrous,
   saturating resource, structurally like vehicle capacity. The paper
   should state this contrast explicitly and briefly, then move on —
   it motivates the model, it is not evidence for it.

## 4. Formal model (draft)

Agents `V = {1..n}`, horizon `t = 1..T`, context items (facts)
`F = {f_1..f_m}`. Agent `i` holds knowledge `K_i(t) ⊆ F`, with
`K_i(0)` given.

**Endogenous graph.** `G_t = (V, E_t)`, where `E_{t+1} = g(E_t, K(t))`.
The topology at `t+1` is a function of who knows what at `t`. Static
`G` is the degenerate special case and should be analyzed first as a
baseline.

**Decisions.** At each `t`, choose transmissions
`A_t ⊆ {(i, j, S) : (i,j) ∈ E_t, S ⊆ K_i(t)}` — send bundle `S` from
`i` to `j`.

**Cost / the budget.** `c(S)` = token cost of `S`. Global knapsack:

```
Σ_t Σ_{(i,j,S) ∈ A_t} c(S)  ≤  B
```

**Utility with saturation.** Per-agent,

```
u_i(K) = rel_i(K) − ρ_i( load_i(K) )
```

`rel_i` is task relevance (coverage-like, plausibly submodular *alone*,
though F2 says not in general); `ρ_i` is a saturation penalty
increasing in the tokens loaded into `i`. `ρ_i` must be **calibrated
against published degradation curves**, not invented — that is what
stops a reviewer calling the non-monotonicity an artifact of a
convenient penalty term.

Objective: maximize `U = Σ_i u_i(K_i(T))` subject to the global budget.

## 5. Theory targets

| # | Target | Confidence |
|---|---|---|
| T1 | `U` is not monotone — constructive instance, saturation-driven, calibrated to measured degradation | high |
| T2 | `U` is not submodular — constructive instance, complementarity-driven | high |
| T3 | Precise statement of which step of the greedy `(1−1/e)` argument each failure breaks, hence why the RIS/IMM/GPU-IM lineage is inapplicable | high |
| T4 | NP-hardness (knapsack for the budget; coverage for the relevance term), and inapproximability in the unconstrained case | medium-high |
| T5 | **Positive result**: identify the structural regime where guarantees return — a *sub-saturation* regime where every agent's allocation stays below its degradation knee — and give a budgeted greedy with a proved ratio there | medium |
| T6 | Endogeneity: either a competitive ratio for an online algorithm against an offline optimum that knows the realized topology, or a proof that endogeneity strictly increases hardness | low — stretch |

T5 is the one that has to land. T1–T4 without T5 is an all-negative
paper. The defense against "they assumed away the hard part" is that
the sub-saturation regime is **where real deployments already operate**
(operators cap per-agent context because they have observed the
degradation), and the *global budget stays unrestricted*, so the
economic difficulty is untouched.

## 6. Baselines

- **Full broadcast** — current practice, the cost baseline to beat.
- **No sharing** — the utility floor.
- **Fixed-schedule summarization** — periodic compaction, the common
  engineering mitigation.
- **Classical IM seeding** (degree / centrality / greedy-on-`rel`
  ignoring saturation) — the "what if you just used KKT" control. This
  one matters most: it should *visibly fail* by over-concentrating and
  saturating hubs, which is the paper's thesis made empirical.
- **Random-k** at matched budget.
- **Hindsight oracle** — offline optimum with full knowledge, for the
  gap.

## 7. Experiment plan

Secondary to the theory, and scoped to the deadline. Simulation over
synthetic agent graphs with `ρ_i` calibrated to published degradation
curves; measure utility-per-token against every baseline at matched
budgets; show the classical-IM seeding control failing in the predicted
way. A real multi-agent LLM run would strengthen the paper but is a
stretch inside four weeks and should not be promised in the abstract.

**Self-contained.** No `svrpspd_wdro/` imports, no shared instances, no
shared results — same separation argument as papers 3 and 5. See
`README.md`.

## 8. Open questions

- Does `rel_i` stay submodular once complementarity is admitted, or do
  we need a bounded-complementarity parameter (a curvature-like
  quantity) to get anything at all? This likely determines whether T5
  is provable.
- Is the sub-saturation regime definable without circularity — i.e.
  without the knee being defined by the very penalty we chose?
- For T6: is there a clean formalism for endogenous edge formation that
  is not so general it becomes trivially hard?
- Page limit and review model for the AAMAS main track are **not yet
  verified** — see `STATUS.md`.

## 9. Naming

PARCEL — **P**rice-**A**ware **R**elay of **C**ontext over
**E**ndogenous **L**inks. "Parcel" nods to the repo's routing lineage
while marking the difference the paper turns on: a parcel is conserved,
context is copyable. The rivalrous thing is the receiver's attention.
