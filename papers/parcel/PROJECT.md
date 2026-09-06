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
agent can measurably *degrade* its performance: a controlled
distractor benchmark finds **all six models tested** degrade in
reasoning accuracy as irrelevant context grows (`distracted2025`), and
NoLiMa finds **11 of 13 models** claiming ≥128K context falling below
half their short-context baseline at 32K (`nolima2025`). So there exist
instances where adding one transmission strictly lowers utility.

**Calibration discipline (learned the hard way — see
`VERIFY_CITATIONS.md`).** Anchor F1 on those *cross-model* statements.
Do **not** headline the widely-quoted "43% → 19%" figure: it is one
model (Grok-3-Beta) at one reasoning depth on a synthetic benchmark,
and a referee will check. A "−55.89% average regression" circulating in
search results **is not in the cited paper at all** and must never be
used.

**And state the heterogeneity honestly** (detail in §11.3). Dense
models hold 97.5–98.5% accuracy under 15,000 words of *generic filler*
(arXiv:2601.11564), and distractor-aware truncation flattens the
degradation curve entirely for frontier models (arXiv:2608.03297). Both
results test the **easiest regime on both axes that matter** —
single-hop retrieval with non-confusable distractors. The contrast is
the point: with the answer span fixed and only distractors added,
GPT-4.1 loses **0.270** on multi-hop HotpotQA versus **0.065** on
single-span SQuAD. Degradation is task- and confusability-dependent.

PARCEL needs saturation to exist in *some* operating regime, not to be
a universal law — and it plainly does, since frontier models miss
dangerous actions **2×–30× more often** after 800K tokens of benign
context (arXiv:2605.12366). Claiming universality invites a referee to
produce the counterexample; conceding the heterogeneity first, and
showing the model handles it, is strictly stronger.

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

⚠️ **Repositioned 2026-09-06 after a prior-art sweep. Read §10 before
drafting** — several things that looked like contributions are taken.
The ordering below is the post-sweep ordering, and it is deliberate:
the negative result is now the spine, not the warm-up.

1. **The negative results — the spine of the paper.** Two of them,
   independent, pointing the same way:

   **(a) Against classical IM.** Per-agent saturation breaks
   **monotonicity** and complementarity breaks **submodularity**, which
   invalidates the KKT → RIS → IMM toolchain. Every applied paper found
   either *assumes* submodularity (PACMS, BPS) or avoids theory
   altogether.

   **(b) Against the nearest general framework** (§10.6, machine-checked
   in `code/degeneracy_check.py`). Shi & Lai's Theorem 4 needs
   `γ₁`-weak submodularity *and* `γ₂`-weak supermodularity at once, and
   PARCEL's two phenomena kill exactly one parameter each —
   complementarity kills `γ₁`, saturation kills `γ₂`. Not "the bound is
   weak": the parameters **do not exist**.

   Two degeneracy results is a substantially stronger paper than one,
   and (b) also retires the biggest scooping risk. This is what the
   paper leads with.

2. **Multi-receiver structure — where the real theorem lives.** The
   closest prior work (BPS, arXiv:2608.19993) is **one receiver with a
   modular penalty**. The tension PARCEL names — a *global* token
   knapsack against *per-agent* degradation — is untouched by it. Frame
   the contribution as a **partition / multi-knapsack problem with
   per-block supermodular penalties**, not as generic non-submodular
   knapsack maximization, which is already covered (§10).

3. **Cost and damage are different quantities** (§4, §11.2). You are
   **billed in tokens** — absolute, global, modular — but **damaged by
   confusability** — semantic, per-agent, supermodular. The measurements
   force this: raw length barely predicts degradation once it is
   controlled for, while topical proximity predicts it strongly. All the
   prior work optimizes a *single* quantity, so an item's price here
   depends on two independent properties, and a cheap highly-confusable
   item can be worse than an expensive orthogonal one. This is the
   structural primitive that gives an interpretable bound where the
   general theory offers only an opaque global `γ`, and it is what turns
   the objective into submodular-minus-*supermodular*, breaking the
   Distorted-Greedy analysis the work would otherwise inherit.

4. **Endogenous topology — last section, not the pitch.** No formal
   treatment was found, so it is genuinely open, but it is also the
   hardest to get a theorem about. Position it as the closing section
   or as future work; do not build the abstract around it.

   Temporal/dynamic influence maximization
   exists and is active, but it treats network evolution as
   *exogenous* — the graph changes, you forecast the change, you seed
   against the forecast. In an agent system the orchestrator builds the
   agent graph in response to the task, and what you transmit
   determines who talks to whom next: an agent that learns of a
   dependency goes and contacts the agent that owns it. Seeding
   perturbs the topology it is seeding over.

   ⚠️ "No temporal-IM work models this" is an **absence claim** and no
   citation can support it. It requires a documented search (queries,
   databases, dates) plus the named nearest prior work, or it must be
   softened to a search-bounded statement. **That search has not been
   run** — see `VERIFY_CITATIONS.md`.

5. **Copyable goods, rivalrous attention** — motivation only, one
   paragraph. A *contrast*, not an analogy: in VRP the goods are
   conserved, so routing intuitions apply; information duplicates
   freely, so they do not. What transfers is **capacity** — the
   receiver's context budget is rivalrous like vehicle capacity. State
   it briefly and move on. It motivates the model; it is not evidence.

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
u_i(K) = rel_i(K) − ρ_i( conf_i(K) )
```

`rel_i` is task relevance; `ρ_i` is the saturation penalty.

⚠️ **Revised 2026-09-06 — the penalty argument changed.** The draft
model penalized *absolute* token load. **The evidence does not support
that** (§12). Raw length is a weak predictor of degradation; what
predicts it is **confusable load** — irrelevant content weighted by its
semantic proximity to the task. So `ρ_i` takes `conf_i(K)`, a
confusability-weighted load, not `c(K)`.

**This decoupling is a feature, and it sharpens the paper's central
tension.** You are **billed for tokens** — absolute, global, modular —
but you are **damaged by confusability** — semantic, per-agent,
supermodular. The two quantities are not proportional and can be
manipulated independently: 15,000 words of generic filler cost a 70B
model 0.5 accuracy points, while at *fixed* ~12K tokens a change in
lexical density alone drives retrieval from near-perfect to under 60%.
The earlier framing ("money is global, damage is per-agent") is
therefore too weak. The true statement is:

> **money is global and absolute; damage is per-agent and semantic.**

Consequence for the admission price (§9.4): the numerator and
denominator no longer share units. The rule becomes "marginal relevance
per **token** against marginal degradation per unit **confusable**
load", so an item's price depends on *two* independent properties — what
it costs to send and how confusable it is with the receiver's task. A
cheap, highly confusable item can be worse than an expensive, orthogonal
one. That has no analogue in the single-quantity prior work.

Objective: maximize `U = Σ_i u_i(K_i(T))` subject to the global token
budget.

## 5. Theory targets

| # | Target | Confidence |
|---|---|---|
| T1 | `U` is not monotone — constructive instance, saturation-driven, calibrated to measured degradation | high |
| T2 | `U` is not submodular — constructive instance, complementarity-driven | high |
| T3 | Precise statement of which step of the greedy `(1−1/e)` argument each failure breaks, hence why the RIS/IMM/GPU-IM lineage is inapplicable | high |
| T4 | NP-hardness (knapsack for the budget; coverage for the relevance term), and inapproximability in the unconstrained case | medium-high |
| T5 | **Positive result — MULTI-RECEIVER.** A partition/multi-knapsack allocation with per-block supermodular penalties, bounded in interpretable primitives (receiver load, bundle closure) rather than an opaque global `γ`. The admission price (§9) is a *step inside* this, not the headline. | medium — reframed after §10 |
| T6 | Endogeneity: either a competitive ratio for an online algorithm against an offline optimum that knows the realized topology, or a proof that endogeneity strictly increases hardness | low — stretch, closing section |

**Post-sweep priority (see §10).** T1–T3 are the spine — the negative
result is the strongest unclaimed ground the paper has. T5 must be
stated in the **multi-receiver** form: the single-receiver
modular-penalty version is already published (BPS, arXiv:2608.19993),
and the general non-monotone-non-submodular-knapsack version is already
covered (Shi & Lai 2024). Neither of those touches a *global* budget
allocated across *many* saturating receivers, which is where PARCEL's
theorem has to live.

Do **not** headline the admission price itself — density-greedy
thresholds under a knapsack are textbook, and submodular-minus-modular
with a derived threshold is Harshaw et al. (ICML 2019). The defensible
increment is that the penalty is **state-dependent and rising**, which
turns the objective into submodular-minus-*supermodular* and breaks the
analysis it would otherwise inherit.

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
curves (targets ranked in §11.4); measure utility-per-token against
every baseline at matched budgets; show the classical-IM seeding
control failing in the predicted way by over-concentrating and
saturating hubs.

**Plus one small real measurement, if anything is run at all** (§11.5):
no published curve is dense enough in the 0–8K region to locate the
convexity inflection `c*`, and `c*` defines the regime where the
theorem holds. A fine sweep of that region on one model family — with a
confusability arm (topical vs random distractors) to separate the two
variables of §11.2 — is cheap, directly serves a modelling assumption,
and is the right shape of empirical work for a theory paper. Prefer it
over a broad benchmark.

A full multi-agent LLM evaluation would strengthen the paper but is a
stretch inside the remaining time and must not be promised in the
abstract.

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

Worked 2026-09-06. It replaces the vague "assume a sub-saturation
regime" plan.

⚠️ **Demoted 2026-09-06 by the §10 prior-art sweep.** This was drafted
as the core of the paper; it is not. Density-greedy thresholds under a
knapsack are textbook, submodular-minus-modular with a derived
threshold is Harshaw et al. (ICML 2019), and BPS reached the token
setting first. The derivation below is still *correct* and still needed
— it is the mechanism inside the multi-receiver theorem — but it is a
**step, not a headline**. What survives as novel is the penalty being
**state-dependent** (§9.4) rather than modular.

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

⚠️ **AAMAS review is double-blind.** BATON and TEMPO must be cited in
the **third person**, as any other prior work would be — never "our
previous work", and nothing that identifies the author group.

### 9.8 Open

- ~~**Is `ρ_i` convex?**~~ **RESOLVED 2026-09-06 — see §11.1. Answer:
  only locally, and the draft above overclaims.** ρ is **S-shaped**:
  convex up to each model's effective length (~1–8K tokens,
  capability-dependent), concave above it. Above the inflection `ρ′`
  *decreases*, so the bar would **fall** as an agent fills — the
  opposite of §9.4's claim. **Fix:** state convexity as an explicit
  regime assumption `c ≤ c*(i)`, use NoLiMa's published per-model
  "effective length" as the estimator of `c*`, and claim nothing in the
  concave tail. Encouragingly, the convex region *widens with model
  capability*, so the result is strongest for the capable agents real
  systems deploy. Also note (§11.2) the penalty's argument is
  **confusable load, not token count** — so `τ` is a price per unit
  confusability, while the budget is spent in tokens.
- **Bundle identification is assumed given** (§9.3). Can we bound the
  damage from imperfect bundling? Without such a bound this is the
  paper's most exposed assumption.

### 9.9 The knapsack result must be proved here, not imported

⚠️ **The clean "weakly submodular under a knapsack constraint" theorem
assumed in the planning draft does not exist.** Chen, Feldman & Karbasi
(`chen2017weakly`) generalize beyond cardinality to **matroids**, and a
knapsack is not a matroid.

The nearest prior guarantee is **Shi & Lai, TCS 990:114409 (2024)**
(`shilai2024`) — non-monotone *and* non-submodular *and* knapsack,
which is structurally PARCEL's exact optimization setting.

**This is a scooping risk and must be read in full before drafting.**
PARCEL's novelty has to live in the **model** — agent context
saturation, the endogenous graph, the *derived* admission price — and
not in an abstract non-monotone-non-submodular-knapsack theorem that a
2024 TCS paper may already own. Cite `shilai2024` as the nearest prior
guarantee and prove the paper's own result over it.

## 10. Prior art and positioning

Sweep run 2026-09-06. **Verdict: partially scooped — reposition, do not
abandon.** Everything below is arXiv-preprint or published prior art;
preprints still count as prior art for novelty, even though they are
non-archival for AAMAS's dual-submission rule.

### 10.1 Dead as headline claims — do not assert these

| Claim | Who owns it |
|---|---|
| "Admission price / density threshold" as a novel rule | **Harshaw, Feldman, Ward & Karbasi, ICML 2019** (Distorted-Greedy), the regularized-submodular line, **Shi & Lai Algorithm 1** (density-greedy with a positive-marginal filter — verbatim the same rule), and in the token setting **BPS got there first** |
| A general theorem for weakly-submodular maximization under a knapsack | **Shi & Lai, TCS 990:114409 (2024)** — but see §10.6: their parameters **do not exist** on PARCEL's problem class, so this does *not* subsume PARCEL |
| "Submodular context selection under a token budget for LLM agents" | **PACMS** (arXiv:2606.20047), **BPS** (arXiv:2608.19993) |

The dominance/exchange argument in §9.5 is **standard technique** in
the regularized-submodular literature, not a novel proof device.
Present it as a step, never as a headline. A referee who knows Harshaw
et al. will recognize it instantly.

### 10.2 The direct competitor — BPS

**"Optimal Skill Selection for LLM Agents with Provable Bicriteria
Guarantees"** (Chen, Chen, Wang, Li, Huang; arXiv:2608.19993, Aug
2026). Objective: `max_{S: ℓ(S)≤B} G(S) − κℓ(S)` — a **monotone
submodular** benefit minus a context penalty under a hard token budget.
Algorithm BPS is density-greedy over seed sets of size ≤2; bicriteria
`(1−1/e, 1)`.

This is PARCEL's objective *shape* and PARCEL's admission-price rule,
already published with a guarantee. **But its assumptions are exactly
what PARCEL denies:**

- **single agent** — one fixed executor, no multi-receiver allocation;
- penalty **modular and state-independent** (depends only on total
  token length);
- benefit assumed **monotone submodular**.

The opening BPS leaves: its own abstract concedes that redundant skills
"can even degrade performance" — it *names* the phenomenon and then
models it away, by keeping the benefit monotone submodular and pushing
degradation into a separate modular penalty. PARCEL's claim is that
this decomposition is inadequate: degradation is **state-dependent**
(supermodular in load), and complementarity breaks submodularity of the
benefit itself.

**Positioning: BPS is the single-receiver, modular-penalty,
submodular-benefit special case that PARCEL generalizes.** Cite it
prominently, use it as the baseline and the foil, and pre-empt the
obvious referee objection by naming the relationship explicitly rather
than letting a reviewer discover it. **Read its proofs before drafting.**

### 10.3 Must-cite, partially pre-empts the premise

**"Phase Transition for Budgeted Multi-Agent Synergy"** (Liu, Kong,
Pei; arXiv:2601.17311) — theory, *multi-agent*, models finite context
windows as hard fan-in limits and characterizes saturation via mixing
depth. The mathematics is majority-vote aggregation and correlation
exponents, not set-function optimization, so it is not a scoop — but it
partly pre-empts "context saturation is a real constraint worth
theorizing." Must cite.

⚠️ An automated summary of this paper **hallucinated** claims about
submodularity and complementarity that its actual abstract does not
contain. Read the real abstract; do not trust secondary summaries of it.

### 10.4 Empirical cluster — citations, not scoops

None of these are theoretical; all are related work. **RCR-Router**
(arXiv:2508.04903) is closest to PARCEL's *applied* problem — per-agent
memory subset selection under a strict token budget, ~30% token
reduction, no theorems. Also: AgentPrune, AgentDropout, TodyComm
(arXiv:2602.03688), Guided Topology Diffusion (arXiv:2510.07799),
KVComm (arXiv:2510.03346), "Cut the Crap" (arXiv:2410.02506), AdaGReS,
and "Token Economics for LLM Agents" (arXiv:2605.09104, a survey).

**Influence-maximization framing for LLM agents: none found.** The
nearest is arXiv:2505.23352 on information propagation in LLM-MAS
topologies, which uses diffusion *language* empirically but never
invokes KKT greedy or submodularity. **PARCEL's negative result appears
unclaimed** — this is the strongest remaining ground.

### 10.5 Adjacent theory to track

**"Stronger Approximation Guarantees for Non-Monotone γ-Weakly
DR-Submodular Maximization"** (arXiv:2601.00611) — **at AAMAS 2026,
PARCEL's own venue.** Improves `γe^{−γ}` to `Φ_γ` (0.401 at `γ=1`), but
over a down-closed convex body in *continuous* space, not a discrete
knapsack. Adjacent, not superseding — but being at the same venue makes
it a likely reviewer touchstone. Know it.

### 10.6 RESOLVED 2026-09-06 — Shi & Lai does *not* cover PARCEL

Full text obtained and read. **The scooping fear was wrong in the
direction that matters, and the finding is now the paper's best
material.** Verified exhaustively by
`code/degeneracy_check.py` (runs in a second, no dependencies).

Their Theorem 4 — the general case allowing negative objective values —
requires the objective to satisfy **both** parameters simultaneously:

- **Def 1, `γ₁`-weak submodular** (`γ₁ ≥ 1`): for `U₁ ⊊ U₂`, `x ∉ U₂`,
  `F(U₂+x) − F(U₂) ≤ γ₁·[F(U₁+x) − F(U₁)]`
- **Def 3, `γ₂`-weak supermodular** (`γ₂ ≥ 1`): for `U₁ ⊊ U₂`, `x ∉ U₂`,
  `γ₂·(F(U₂+x) − F(U₂)) ≥ F(U₁+x) − F(U₁)`

A finite parameter **fails to exist at all** — not "is large", *does
not exist* — when a triple forces an inequality no finite multiplier
can satisfy. And PARCEL's two phenomena each kill exactly one:

| Phenomenon | `γ₁` (Def 1) | `γ₂` (Def 3) |
|---|---|---|
| **Complementarity** (hard AND-pair) | **DOES NOT EXIST** | exists |
| **Saturation** (modular rel, convex penalty) | exists | **DOES NOT EXIST** |

*Complementarity:* with `F(∅)=F({f₁})=F({f₂})=0`, `F({f₁,f₂})=1`, take
`U₁=∅`, `U₂={f₂}`, `x=f₁`. Def 1 demands `1 ≤ γ₁·0`. No `γ₁` works.

*Saturation:* with modular relevance and a convex load penalty, a fact
whose marginal is `+0.5` at a lightly-loaded receiver is `−1.5` at a
saturated one. Def 3 demands `γ₂·(−1.5) ≥ 0.5`. No `γ₂` works.

**The symmetry is exact and quotable:** complementarity breaks weak
*sub*modularity; saturation breaks weak *super*modularity. Theorem 4
needs both at once, so it applies to neither phenomenon alone, let
alone together.

**Consequences.**

1. **The "already covered" risk is retired.** PARCEL cannot be dismissed
   as a special case of Shi & Lai — their parameters are undefined on
   this problem class.
2. **This becomes a formal contribution**, and it strengthens the spine
   (§3.1): it is a *second, independent* instance of "the standard
   machinery degenerates here", now against the nearest general
   framework rather than against classical IM. Two degeneracy results
   pointing the same way is a much stronger paper than one.
3. **The admission price is still not novel.** Confirmed directly from
   their Algorithm 1: line 3 accepts `x` only while
   `F(A∪{x}) − F(A) > 0`, line 4 picks
   `argmax [F(A∪{x}) − F(A)] / c(x)`. That *is* density-greedy with a
   positive-marginal filter — i.e. the admission price. §9 stays
   demoted; cite Algorithm 1 alongside Harshaw et al. and BPS.
4. Note also **Theorem 1**'s `θ`-weak monotonicity (`θF(U₁) ≤ F(U₂)`)
   *does* tolerate bounded saturation, since it only caps the
   multiplicative drop — but Theorem 1 still needs `γ₁`, which
   complementarity kills. There is no route through either theorem.

## 11. Calibrating ρ — what the measurements actually say

Evidence sweep 2026-09-06. Full extracted tables are in the session
record; the load-bearing conclusions are below. **All citations here are
UNVERIFIED by me personally** — they come from an assisted extraction
pass and must be checked before use (`VERIFY_CITATIONS.md`).

### 11.1 ρ is S-shaped, not convex — the rising bar is a *local* result

The decisive statistic is accuracy lost per 1K tokens between
consecutive NoLiMa context lengths. Convexity requires it to **rise**.
It rises for at most one or two intervals, then falls by one to two
orders of magnitude. Representative (points lost per 1K tokens):

| Model | 1→2K | 2→4K | 4→8K | 8→16K | 16→32K | peak | NoLiMa "effective length" |
|---|---|---|---|---|---|---|---|
| GPT-4.1 | 0.40 | **1.75** | 1.05 | 0.32 | 0.32 | 2–4K | 16K |
| GPT-4o | 0.10 | 1.15 | **1.62** | 0.95 | 0.74 | 4–8K | 8K |
| Claude 3.5 Sonnet | 1.40 | 3.20 | **3.97** | 2.00 | 0.99 | 4–8K | 4K |
| Llama 3.1 8B | **11.30** | 5.15 | 3.05 | 1.16 | 0.53 | 1–2K | 1K |
| Gemma 3 4B | **15.00** | 9.45 | 2.22 | 0.65 | 0.09 | 1–2K | <1K |

**Consequence — and it contradicts the §9.4 draft.** Above the
inflection `ρ′` is *decreasing*, so the admission bar would **fall** as
an agent fills, not rise. The rising-bar result holds **only on the
convex sub-saturation region** `c ≤ c*(i)`.

**The fix is clean and defensible:** state convexity as a *regime
assumption* `c ≤ c*(i)`, and use **NoLiMa's published per-model
"effective length" as the empirical estimator of `c*`** — that column
exists precisely to mark where a model stops using its context
reliably. Do not claim the guarantee in the concave tail. One honest
caveat to state: accuracy floors at zero, so *some* upper concavity is
mechanical rather than behavioural.

**Encouraging corollary:** the convex region **widens with model
capability** (weak models peak in the first interval — essentially no
convex region; strong models peak at 4–8K). So the result is *more*
valid for the capable agents that real multi-agent systems deploy.

### 11.2 The penalty argument is confusability, not length

Five independent length-matched or length-fixed designs agree, and none
found argues the opposite once length is controlled:

- 15,000 words of generic filler cost Llama-3.1-70B **0.5 points**
  (98.5 → 98.0) — arXiv:2601.11564.
- At **fixed ~12K tokens**, varying only lexical density moves
  retrieval from near-perfect to **under 60%** — arXiv:2606.06203.
- Length-matched hard negatives cost measurably more than *random*
  documents of identical length; the random arm stays near control —
  MUDDLE, arXiv:2608.29477.
- "Filler insertion has little effect… fragmentation, not prompt
  length, drives the loss" — arXiv:2608.22140 (EMNLP 2026).
- All 18 models score **higher on shuffled haystacks than on logically
  structured ones of the same length** — Context Rot.

A pure *ratio* law also fails: the ratio model of arXiv:2603.15723
saturates in its own data (0.330 → 0.305 when signal share halves),
which a ratio law cannot produce.

**So: neither absolute count nor ratio. Confusable load.** Suggested
functional forms to fit: `ρ(c) = a·log(1 + κ·c_conf/c*)` with a
topical-weight `κ`, or a two-parameter logistic in `c/c*` if both
regimes must be covered.

### 11.3 Task dependence — state it, do not hide it

Degradation is far steeper on multi-hop reasoning than single-span
retrieval. With the answer span held fixed and only distractors added,
GPT-4.1 loses **0.270** on multi-hop HotpotQA versus **0.065** on
single-span SQuAD over the same expansion (arXiv:2603.15723). GSM-DC
shows sensitivity rising monotonically with reasoning depth, with a
published functional form `E(m; rs) ∝ m^δ(rs)`, `δ` increasing in
depth — and `δ ≪ 1`, i.e. **error is concave in distractor count**.

This is why the counter-evidence (§2) is real but limited: it tests
single-hop factual QA with generic filler — the easiest regime on both
axes that matter.

⚠️ **Contrary to any "it's been fixed in 2026 models" reading:**
arXiv:2605.12366 reports current frontier models missing dangerous
actions **2×–30× more often** after 800K tokens of benign context.
Degradation has not been engineered away; it has moved out in scale.

### 11.4 Calibration targets, ranked

1. **GSM-DC** (arXiv:2505.18761) — best for a *count*-parameterized ρ:
   8 linearly-spaced points × 6 models × 4 reasoning depths, relevant
   content held fixed, **and the authors publish a fitted functional
   form**. ⚠️ Per-point values exist only as a figure; the numbers in
   the session record were **reconstructed from the source SVG** and
   validated against the two values quoted in the text. Treat them as
   reconstruction: usable for shape analysis, **never quotable as
   published values**.
2. **NoLiMa** (arXiv:2502.05167) — best for a *token*-parameterized ρ:
   7–9 log-spaced points, 22 models, plus the per-model `c*` column.
   Weakness: log spacing leaves only ~2 points inside the convex region.
3. **Dhara & Sheth** (arXiv:2603.15723) — the cleanest controlled
   design (fixed 256-token signal window, 128-token distractor chunks,
   answer-leak filtered, bootstrap CIs). Only 4 points.
4. **Levy, Jacoby & Goldberg, FLenQA** (ACL 2024, arXiv:2402.14848) —
   the best-designed length sweep for isolating padding, with a
   padding-*type* arm that directly tests length-vs-confusability.
   Per-point values are figure-only, **but the dataset and code are
   released**, so regenerating them is a modest job.

### 11.5 The experiment worth running ourselves

**No published curve is dense enough in the 0–8K convex region to
locate the inflection.** NoLiMa gives 3 intervals there, FLenQA 4 in
aggregate form. If `c*` is load-bearing for the theorem — and §12.1
says it is, since it defines the regime where the guarantee holds —
then a dense sweep of that region is the single most valuable
experiment PARCEL could run, and it is cheap: one model family, fixed
task, distractor tokens swept finely from 0 to ~8K, with a
confusability arm (topical vs random distractors) to separate the two
variables of §11.2.

That is also the ideal empirical contribution for a theory paper: small,
targeted, and directly in service of a modelling assumption rather than
a general benchmark.

## 12. Naming

PARCEL — **P**rice-**A**ware **R**elay of **C**ontext over
**E**ndogenous **L**inks. "Parcel" nods to the repo's routing lineage
while marking the difference the paper turns on: a parcel is conserved,
context is copyable. The rivalrous thing is the receiver's attention.
