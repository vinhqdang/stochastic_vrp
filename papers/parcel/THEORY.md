# PARCEL — theory

Working statements for the manuscript. Status of each is marked:
**PROVED** (argument complete and written here), **VERIFIED**
(machine-checked on small instances by a script in `code/`),
**SKETCH** (idea sound, proof not yet written), **OPEN**.

Reframed 2026-09-07 after the experiments. Proposition 1 is now the
centrepiece: it is the theorem that matches the strongest empirical
result (per-receiver allocation beats global seeding by 19.2 points at
matched cost, p = 0.0004; `RESULTS.md` §1b) and it is what justifies
the paper existing at all.

---

## Model

Receivers (agents) `1..n`. A pool of information items `F`, each with
token cost `c(f) > 0`. An **allocation** assigns a set `S_i ⊆ F` to
each receiver.

Two cost conventions, both used below because the results hold in
either:

- **Billed per receiver** (the realistic one): total spend is
  `Σ_i c(S_i)`. Information is copyable, but each transmission is
  charged, so serving `n` receivers with the same set costs `n` times as
  much as serving one.
- **Broadcast free**: a set may be given to everyone at the cost of one
  transmission. Included to show the results do not depend on the cost
  model.

Per-receiver utility `u_i(S_i) = rel_i(S_i) − ρ_i(conf_i(S_i))`:
relevance minus a saturation penalty in *confusable* load (see
`PROJECT.md` §4 and §11.2 — the penalty's argument is confusability,
not raw token count, and that is measured rather than assumed).
Objective `U = Σ_i u_i(S_i)` subject to the budget.

---

## Proposition 1 — the seed-set formulation loses a factor n

**Status: PROVED, and VERIFIED by `code/separation_check.py`.**

Classical influence maximization selects **one** set and lets it serve
the network. PARCEL selects a possibly different set per receiver. The
difference is not presentational:

> **Proposition 1.** For every `n ≥ 1` there is an instance with `n`
> receivers on which, at an identical budget,
> `OPT_per-receiver / OPT_seed-set = n`.
> The separation holds in **both** cost conventions, for two
> independent reasons.

**Construction.** `n` receivers, `n` distinct items, each of cost `c`.
Receiver `i` is satisfied only by item `i`. Budget `B = n·c`.

**(a) Billed per receiver — the cost argument.**
*Per-receiver:* send item `i` to receiver `i`. Spend `n·c = B`,
utility `n`.
*Seed set:* one set `S` reaches all `n` receivers, so it costs
`n·|S|·c ≤ n·c`, forcing `|S| ≤ 1`. A single item satisfies exactly one
receiver. Utility `1`. ∎

**(b) Broadcast free, attention saturating — the saturation argument.**
Take the budget non-binding, and let a receiver absorb one item freely
and pay `1` for each further item. Broadcasting `S` to everyone gives
utility `|S| − n·max(0, |S| − 1)`, maximised at `|S| = 1` with value
`1`; per-receiver allocation again gives `n`. ∎

**(c) Unbounded — added and VERIFIED 2026-09-07 by
`code/unbounded_check.py`.** Same items and receivers, broadcast still
free, but a receiver pays `lam` for each item it receives that is
irrelevant to it. Broadcasting `S` gives `|S|·(1 − lam·(n−1))`: every
item satisfies its own receiver and distracts the other `n−1`. For any
`lam > 1/(n−1)` this is strictly negative for every non-empty `S`, so
the best seed set is the **empty** one, value `0`. Per-receiver
allocation sends item `i` to receiver `i` alone, pays no penalty, and
still gets `n`. Ratio `n/0`. ∎

**Consequence for how Proposition 1 should be stated.** The factor `n`
of (a) is the one-set formulation's **best case**, attained exactly
where its own assumptions hold (modular relevance, `rho ≡ 0`, so the
objective is monotone and submodular). Once saturation is strong
enough to make broadcasting actively harmful, the gap has no finite
bound. Do **not** write "Θ(n)" as if `n` were an upper bound — there
is no matching upper bound, and a referee could construct (c)
themselves. The manuscript now says "a factor `n` on benign instances,
unbounded in general".

**Why (b) matters more than (a).** A referee can dismiss (a) as an
artefact of charging per receiver. (b) removes that objection: even
with free broadcast the seed set is capped, because **an extra
broadcast item helps one receiver and harms the other `n−1`.** That
sentence is the whole multi-receiver point, and it is why an
influence-maximization formulation cannot be repaired by a better
approximation ratio — the formulation is choosing the wrong object.

**Consequence.** Any guarantee proved for a seed-set objective is
vacuous here: it bounds distance from the wrong optimum. This is a
sharper obstruction than "the objective is not submodular", because it
survives even when the objective *is* submodular — the plain-utility
instance in (a) has no saturation and no complementarity at all.

**Empirical counterpart.** `RESULTS.md` §1b: per-agent top-k beats the
centrality (seed-set) control by **+0.192, p = 0.0004**, at matched
cost. The mechanism in the data is the one in the proof — global
ranking spends budget on items that are centrally scored but
individually unneeded.

---

## Proposition 2 — NP-hardness

**Status: PROVED (routine).**

> **Proposition 2.** Maximising `U` under the budget is NP-hard, already
> for `n = 1` and with `ρ ≡ 0`.

With one receiver and no saturation the problem is: choose `S ⊆ F` with
`c(S) ≤ B` maximising a modular `rel`. That is KNAPSACK. For general
`n` under a single shared budget it contains MULTIPLE KNAPSACK. ∎

Routine, and the paper should say so — it is included for completeness,
not as a contribution. The interesting hardness question (does the
saturation penalty add hardness beyond the knapsack structure?) is
**OPEN**.

---

## Proposition 3 — the standard toolchains degenerate

**Status: (a),(b) PROVED. (c) VERIFIED by `code/degeneracy_check.py`.**

**(a) `U` is not monotone.** With `rel` modular and `ρ` increasing,
adding an item whose relevance is below its marginal saturation cost
strictly lowers `u_i`. Monotonicity is required by the greedy
`(1 − 1/e)` argument.

**(b) `U` is not submodular.** Two items individually inert and jointly
decisive give increasing returns: `rel(∅) = rel({a}) = rel({b}) = 0`,
`rel({a,b}) = 1`. Submodularity is the other requirement.

**(c) The nearest general framework's parameters do not exist.**
Shi & Lai (TCS 990:114409, 2024) Theorem 4 requires the objective to be
`γ₁`-weak submodular **and** `γ₂`-weak supermodular simultaneously. The
two phenomena kill exactly one parameter each:

| | `γ₁` (weak submodular) | `γ₂` (weak supermodular) |
|---|---|---|
| complementarity | **does not exist** | exists |
| saturation | exists | **does not exist** |

*Complementarity:* with the (b) instance, `U₁ = ∅`, `U₂ = {b}`,
`x = a`, Definition 1 demands `1 ≤ γ₁·0`.
*Saturation:* an item with marginal `+0.5` at a light receiver and
`−1.5` at a saturated one makes Definition 3 demand
`γ₂·(−1.5) ≥ 0.5`. Neither has a finite solution. ∎

The symmetry is the quotable part: **complementarity breaks weak
*sub*modularity; saturation breaks weak *super*modularity.** Theorem 4
needs both at once, so it applies to neither phenomenon alone.

---

## Proposition 4 — the admission price, and what it is worth

**Status: SKETCH. And empirically it does not beat a tuned baseline.**

The marginal of adding `f` to receiver `i` holding `S` is non-negative
iff, to first order,

```
   Δrel_i(f | S) / c(f)   ≥   ρ_i′( conf_i(S) )
```

— send an item only when its relevance per token clears that receiver's
current marginal degradation rate. An item failing the test both lowers
`u_i` and consumes budget, so no optimal solution contains one: the
restriction is a **dominance property, not an assumption**. The
exchange argument is exact only at bundle level, since dropping an item
can destroy a retained complement.

**Two reasons this is not the contribution, and the paper must say
both.**

1. **It is prior art.** Density-greedy with a positive-marginal filter
   is Harshaw et al. (ICML 2019), is verbatim Shi & Lai's own
   Algorithm 1, and reached the token-budget setting first via BPS
   (arXiv:2608.19993). See `PROJECT.md` §10.1.
2. **It does not measurably help.** At matched budget it does not beat
   per-receiver top-k on any of three budget levels; the cleanest
   comparison is exactly zero (`RESULTS.md` §1b).

So Proposition 4 belongs in the paper as *a method with a derivation*,
credited to its lineage, and explicitly not as the source of the
result. Presenting it as the contribution would be both a novelty
overclaim and unsupported by our own experiments.

---

## What the paper claims, in order

1. **Proposition 1** — the seed-set formulation loses `Θ(n)`; the IM
   formulation is choosing the wrong object, not merely lacking a good
   ratio. Matched by the largest measured effect.
2. **Proposition 3** — both the classical toolchain and the nearest
   general framework degenerate, the latter machine-checked, with the
   sub/super symmetry.
3. **Empirics** — allocation buys a 2.7× token reduction at no
   measurable accuracy cost; over-delivery costs ~5 accuracy points
   (consistent across two models, borderline power).
4. **Proposition 2, Proposition 4** — completeness and method, both
   credited as routine or prior.

---

## Corollary (of Proposition 5) — approximation transfers

**Status: PROVED, and VERIFIED by `code/unbounded_check.py` (claim B,
400 random instances against brute-force enumeration).**

> **Corollary.** If for each receiver `i` an oracle returns, for each
> level `b`, a set `S̃_i(b)` with `c(S̃_i(b)) ≤ b` and
> `u_i(S̃_i(b)) ≥ (1−eps)·v_i(b)`, then the resource-allocation DP run
> on the curves `ṽ_i(b) = u_i(S̃_i(b))` returns a feasible allocation
> worth at least `(1−eps)·OPT`.

**Proof.** Let `(b_i*)` maximise `Σ v_i(b_i)` under `Σ b_i ≤ B`; by
Proposition 5 its value is `OPT`. The DP returns
`max{Σ ṽ_i(b_i) : Σ b_i ≤ B} ≥ Σ ṽ_i(b_i*) ≥ (1−eps)·Σ v_i(b_i*)
= (1−eps)·OPT`. Feasibility and attainment hold because each `ṽ_i(b)`
is realised by an actual set of cost `≤ b`, and `ṽ_i(b) ≥ 0` because
`∅` is always available. ∎

**Why it is worth stating despite being routine.** The technique is
textbook resource allocation and the paper credits it as such. The
*conclusion* is the contribution: with `rho_i ≡ 0` each `v_i` is a
knapsack value function and such oracles exist for every `eps`, so in
that regime the `n`-receiver problem is **no harder to approximate
than the 1-receiver problem**. Paired with Proposition 1 this gives a
clean pair of statements — the seed-set object loses a factor `n` or
worse, and the correct object costs nothing in approximability. So the
reason to abandon the one-set formulation is not tractability.

This is also the project's **only positive guarantee**, and it is
conditional. The unconditional ratio for non-trivial `rho_i` remains
the biggest gap (below).

---

## Open, and honestly so

- Does saturation add hardness beyond knapsack? (Prop 2 remark.)
- A positive approximation guarantee for the *multi-receiver* problem
  with per-receiver caps. Proposition 1 says what fails; it does not
  supply an algorithm with a ratio. **This is the biggest theoretical
  gap.**
- Endogenous topology: no formal treatment (`PROJECT.md` §3.4).
- Bundle identification is assumed given (§9.3).
