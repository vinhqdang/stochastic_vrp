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
| T5 | **Positive result**: the admission-price rule (§9), plus a budgeted greedy with a proved ratio on the price-respecting region | medium-high — upgraded, see §9 |
| T6 | Endogeneity: either a competitive ratio for an online algorithm against an offline optimum that knows the realized topology, or a proof that endogeneity strictly increases hardness | low — stretch |

T5 is the one that has to land. T1–T4 without T5 is an all-negative
paper. **§9 now supplies it**, and on a stronger footing than the
original "assume a sub-saturation regime" plan: the regime is *derived*
as a dominance property, not assumed.

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

- ~~Does `rel_i` stay submodular once complementarity is admitted?~~
  **Resolved 2026-09-06 — see §9.** No, and in the worst case no
  parameter saves it; the fix is a two-level ground set.
- ~~Is the sub-saturation regime definable without circularity?~~
  **Resolved — §9 derives it instead of assuming it.**
- For T6: is there a clean formalism for endogenous edge formation that
  is not so general it becomes trivially hard?
- Page limit and review model for the AAMAS main track are **not yet
  verified** — see `STATUS.md`.

## 9. The admission price — resolution of the T5 structural question

Worked 2026-09-06. This section replaces the vague "assume a
sub-saturation regime" plan and is now the intended core of the paper.

### 9.1 The two failures are separable

Write `u_i(S) = rel_i(S) − ρ_i(c(S))`, with `rel_i` monotone
non-decreasing, `ρ_i` increasing and **convex** (degradation
accelerates), and `c` modular.

- `ρ` alone makes `u` non-monotone even when `rel` is perfectly
  monotone submodular.
- Complementarity in `rel` alone breaks submodularity even when
  `ρ ≡ 0`.

They are independent, so they can be parameterized independently.
Worth stating as a lemma — it is what licenses the rest.

### 9.2 Complementarity: the negative finding

The submodularity ratio `γ` (the standard weak-submodularity handle)
**collapses to exactly zero under a single hard AND-pair**. Take
`rel(∅) = rel({f₁}) = rel({f₂}) = 0`, `rel({f₁,f₂}) = 1`. With `S = ∅`,
`T = {f₁,f₂}`: the sum of individual marginals is `0`, the joint
marginal is `1`, so `γ = 0` and every `(1 − e^{−γ})` bound degenerates
to nothing.

So: *one* purely complementary pair anywhere in the instance destroys
the weak-submodularity route. This is a real finding and it belongs in
the paper — it rules out the obvious fix and justifies the next move.

### 9.3 The fix: a two-level ground set

Do not run the optimization over facts. Run it over **bundles** —
complementarity-closed groups of facts, so intra-bundle complementarity
is eliminated by construction, and only the milder inter-bundle
complementarity remains, handled by `γ`.

Reads as: *strong complementarity is local and low-order* (a fact and
its operand; a deadline and a duration), *weak complementarity is
global*. Bundle identification is a separate problem — a clustering
over the fact-dependency graph — and the paper must be explicit that it
is assumed given, not solved here. That is a real limitation and should
be stated as one rather than buried.

### 9.4 The admission price

The marginal of adding fact `f` to agent `i` holding `S`:

```
Δu_i(f | S) = Δrel_i(f | S) − [ ρ_i(c(S) + c(f)) − ρ_i(c(S)) ]
```

Non-negative iff, to first order in `c(f)`,

```
   Δrel_i(f | S) / c(f)   ≥   ρ_i′( c(S) )
                              └──── τ_i(S), the admission price ────┘
```

**Send a fact to an agent only if its relevance-per-token clears that
agent's current marginal degradation rate.** Because `ρ_i` is convex,
`ρ_i′` is increasing, so **the bar rises as the agent fills up**: early
in a task an agent accepts marginal context; as its window loads, only
high-value facts clear. That is both the operationally right behavior
and a quotable rule.

### 9.5 Why this is not "assuming away the hard part"

The price rule is a **dominance property, not an assumption**. If `f`
fails the test, including it both lowers `u_i` *and* consumes budget —
so dropping it strictly increases `U` and frees budget. No optimal
solution contains such a fact. Restricting to the price-respecting
region therefore **loses nothing**.

This is the paper's defense, and it is much stronger than the original
plan: we do not assume the sub-saturation regime, we *prove the optimum
lies in it*.

**Where the exchange argument needs care.** Dropping `f` can destroy
the value of a retained `g` if the two are complementary. So the
dominance argument is exact only when the removed item is not
complementary with retained ones — i.e. **at the bundle level**. This
is a second, independent reason the two-level ground set of §9.3 is
the right construction: it is what makes the exchange valid. With
residual inter-bundle complementarity (`γ < 1`) dominance holds only up
to a factor, and the paper must say so plainly.

### 9.6 What is still hard

The problem does not become easy. Choosing which bundles go to which
agents under a global knapsack, with **state-dependent prices** (`τ`
depends on load, which depends on the choices), still embeds knapsack,
so T4's hardness stands. T5 gives a greedy guarantee *on the
price-respecting region*, which is where the optimum already lives.

### 9.7 Lineage note

The rule "send iff `Δrel/c > ρ′`" has the same *shape* as BATON's
"hand off iff `Ĉ_k(W_k) > H_k`" — a marginal value compared against a
state-dependent price. Different problem, different derivation, no
shared artifacts. Worth one sentence in the manuscript as intellectual
through-line; **not** a claim that PARCEL extends BATON.

### 9.8 Open, downgraded

- Does `ρ_i` convexity hold empirically, or is degradation better
  modeled with a knee/cliff? Convexity is what makes `τ` monotone and
  the story clean; a cliff would still work but changes the rule's
  shape. **Depends on the citation check now running.**
- The knapsack loss: weakly-submodular maximization under a *knapsack*
  (not cardinality) constraint — confirm the best available guarantee
  rather than assuming the cardinality bound carries over.
- Bundle identification is assumed given. Can we at least bound the
  damage from imperfect bundling?

## 10. Naming

PARCEL — **P**rice-**A**ware **R**elay of **C**ontext over
**E**ndogenous **L**inks. "Parcel" nods to the repo's routing lineage
while marking the difference the paper turns on: a parcel is conserved,
context is copyable. The rivalrous thing is the receiver's attention.
