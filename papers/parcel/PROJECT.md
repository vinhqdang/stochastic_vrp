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

**And state the heterogeneity honestly.** At least one 2026 preprint
reports large dense models holding 97.5–98.5% accuracy under 15,000
words of distractors. Degradation is task- and model-dependent. PARCEL
needs saturation to exist in *some* operating regime — which the
peer-reviewed evidence supports — not to be a universal law. Claiming
the latter invites a referee to produce the counterexample.

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

3. **State-dependent penalty + complementarity-closed bundles.** These
   are the structural primitives that let the bound be stated in
   *interpretable, measurable* quantities (a receiver's load, a bundle's
   closure) where the general theory gives only an opaque global `γ`.
   The increment over the known technique is that the penalty is
   **supermodular in the receiver's load** rather than modular, which
   turns the objective from submodular-minus-modular into
   submodular-minus-supermodular and makes the threshold a *moving*
   price. That breaks the Distorted-Greedy analysis it would otherwise
   inherit.

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

- **Is `ρ_i` convex?** The rising-bar property depends on it. The
  verification pass did not settle this, and the available evidence
  cuts both ways: NoLiMa's cliff-like collapses past a context
  threshold look more like a **knee than a smooth convex curve**. A
  knee still yields an admission price, but `τ` becomes a step rather
  than a continuously rising bar, and the greedy analysis changes. This
  is now the **top open modeling question** — resolve it against the
  measured curves before writing the theorem.
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

## 11. Naming

PARCEL — **P**rice-**A**ware **R**elay of **C**ontext over
**E**ndogenous **L**inks. "Parcel" nods to the repo's routing lineage
while marking the difference the paper turns on: a parcel is conserved,
context is copyable. The rivalrous thing is the receiver's attention.
