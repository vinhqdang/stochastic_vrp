# Seat EIC: Journal-Fit Reviewer

## Seat and angle
I read this as an associate editor of C&OR who works on stochastic VRP and computational OR. My questions were: does the paper fit the journal, how original and significant is it after revision, are the contribution statement and the abstract, introduction and conclusion claims proportionate to the evidence, does a 62-page paper hang together, and was each round-1 item answered in substance? I judged only from the revised manuscript (main.tex and main_revised.txt), the tables and macros, and spot checks of results/r1/regret.csv and fresh.csv. I did not open r1/.

Fit: good. The paper is a data-driven execution-stage recourse policy for SVRPSPD, with benchmark families, MIP-certified plans and an honest reference ladder. That is core C&OR material. The revision is substantive. Every major reviewer request produced a new experiment or a new passage of analysis: Tables 8–15, §3.5, §4.5–4.7, and the Scope and Limitations paragraphs of §5.

## Verdicts on round-1 comments

| Item | Verdict | Evidence anchor | Residual gap |
|---|---|---|---|
| R1.1 | RESOLVED | Assumption 1 is generalised to stochastically monotone Markov kernels, and the factor-sufficiency case is stated (§3.2). Table 9 varies ρ from 0 to 0.9 and adds a day factor, plus a direct monotonicity test. | Props. 2–3 say "Under Assumption 1", but §3 intro says they hold without it on finite trees. Reconcile in the statements (see N6). |
| R1.2 | RESOLVED | §3.1 "Standby versus emergency" explains the pay-per-use convention. §4.6 and Table 12 cover a capacitated FCFS pool, a shadow price λ and S\*. | S\*=0 on every plan, so the capacitated case is almost degenerate. A multi-plan pool is asserted as the economical regime but never tested. |
| R1.3 | RESOLVED | §3.5 "The depot-return transition" gives the post-return state, the conservative reset, repeated returns and recursion (14). Table 11 has an exact-reset column. | – |
| R1.4 | RESOLVED | thr.-k (position-dependent) and roll.-θ (cost-scaled) are in Tables 1, 2 and 6, and the discussion is in §4.2, third observation. | – |
| R1.5 | RESOLVED | §4.1 Protocol: every competitor gets the same 10³ training days, with no validation set, stated explicitly. Table 14 separates offline and online time, and RL compute is in §4.7. | – |
| R1.6 | RESOLVED | The unit is the instance within a gate (§4.1). Table 3 has paired bootstrap CIs, win counts, rank-biserial r and Holm correction. Table 4 gives tail CVaR and P(emg). | Route-level results are said to be in supplementary data, which I could not see. |
| R1.7 | PARTLY_RESOLVED | The contributions paragraph (§1, "three contributions… of different kinds") now separates model, structural results and method, and calls the method "an adaptation". | Props. 2 and 4 are instances of known results: one-step-look-ahead dominance, and value monotonicity under stochastically monotone kernels (monotone MDP comparative statics). No such literature is cited (grep finds no Puterman, Müller–Stoyan or Topkis). See N3. |
| R1.8 | PARTLY_RESOLVED | §5 "Scope of the conclusions" lists time windows, heterogeneous penalties, shared resources and integrated planning, and calls absolute savings "indicative". | The same section opens with "five findings travel beyond our test bed". The abstract says the conclusions "survive… a capacitated standby pool" on evidence that is almost degenerate (S\*=0). Tone these down. |
| R2.M1 | RESOLVED | Table 9 covers ρ ∈ {0, 0.3, 0.6, 0.9} and a day factor. The share of significant monotonicity violations is 0.2/0.1/0.0%. §3.4 explains that the isotonic fit is the monotone projection under approximation. | – |
| R2.M2 | RESOLVED | §3.5 decomposes the bias into (i), (ii) and (iii). Table 11 measures the bias on Salhi–Nagy, Dethloff SAA and city, and Baton-cf fixes (iii). | Table 11 aggregation differs from Tables 2 and 6 (see C3). |
| R2.M3 | RESOLVED | The abstract and §4.3 state that on city networks the method reduces to its handoff-only form. §5 Limitations explains why partial handoff lies outside the action class. | – |
| R2.M4 | RESOLVED | New Prop. 3 bounds the over-trigger cost. Table 8 evaluates regret against the bound and says when a threshold is good enough. | The bound is violated on 1 of 735 routes and this goes unexplained (N5). |
| R2.M5 | RESOLVED | The last paragraph of §5 names co-optimisation as the most important direction and gives two reasons for not doing it: it would destroy the common-plan design, and inner-loop cost. | A short quantification would strengthen it: the share of Det-plan risk sitting at stop 1, which is already implied. |
| R2.M6 | RESOLVED | Table 10 compares pooled, day-type-specific and stale fits, and §5 gives deployment advice. | The shift tested is mild (one known promotion type). |
| R2.m1 | RESOLVED | §4.4 paragraph on DQN, PPO and attention architectures. | – |
| R2.m2 | RESOLVED | The explanation is now in the abstract, the introduction (paragraph after the pillars), §3.6, the Table 2 caption and §5. | The abstract's "it can exceed" has an ambiguous "it". The real fix is a three-action clairvoyant bound (N2). |
| R2.m3 | RESOLVED | 25% and 50% deliver-only twins are re-planned (Table 6). | This exposed a lost comparison that the abstract does not disclose (N1). |
| R2.m4 | RESOLVED | The old Figure 2 is split into Figures 2 and 3, and the fonts are legible (fig2a_how_it_works.png). | – |
| R3.1 | RESOLVED | §3.1 "Standby versus emergency vehicles". Pool size is reported as the 95th percentile of handoffs (§4.6). | – |
| R3.2 | RESOLVED | F_sb is defined as the holding cost. Eq. (7) is shown not to be needed (Assumption 2). The F_sb=60 row of Table 15 violates (7), and §4.6 covers a reserved pool. | – |
| R3.3 | PARTLY_RESOLVED | The claim is restricted to "implementable" policies and the oracle is explained (§3.6, §4.2). The oracle confusion is resolved. | The abstract's unqualified "lowest expected cost" is contradicted by Table 6, 50% deliver-only (thr.-k 3.3 vs Baton 3.1). The introduction and conclusion disclose this tie; the abstract does not. |
| R3.4 | RESOLVED | Table 3 compares full Baton, and Table 5 has a Baton column. | – |
| R3.5 | RESOLVED | §4.3 explains why the oracle (clairvoyance at the next stop) and DP50k (the finite-sample gap, which Table 13 shows closing) lie above Baton on city instances. | – |

## New issues introduced or exposed by the revision

**N1 (MAJOR): the abstract's headline numbers are wider than the evidence** (main.tex l.97–102).
- (a) "lowest expected cost of the implementable policies compared" is false as written. In Table 6, City 50% deliver-only, thr.-k scores 3.3 against Baton's 3.1.
- (b) "under six planning regimes" is attached to Salhi–Nagy and city, but Table 6 is Det-gate only.
- (c) "reaches 86%–96% of the saving of a near-exact DP" holds only on the six Dethloff gates (Table 2). Elsewhere the Baton/DP350k ratio is 10.6/12.3 = 86% on city real-shops, 4.9/6.4 = 77% on 25% deliver-only, and 3.1/6.7 = 46% on 50% deliver-only.

Fix: write "on the Dethloff benchmark, under six planning regimes", add "up to a tie on a deliver-only city variant", and scope the 86–96% to Dethloff. Apply the same edit to the introduction ("every benchmark family") and to §5, first paragraph.

**N2 (MAJOR): no clairvoyant bound applies to the proposed policy.** The introduction (l.256) says the clairvoyant bound "measures how much saving is achievable at all". It does not measure this for a three-action policy, and Baton exceeds it on 4 of 6 gates and on Salhi–Nagy. Readers are left with no hindsight upper bound for Baton; DP350k is only near-exact and is beaten by the exact reset (Table 11). A per-day clairvoyant optimum over {continue, handoff, return} is a small deterministic DP along one route with known demands, so it is cheap to compute. Fix: add a three-action oracle column, or at minimum rephrase l.256 as "for the handoff-only class".

**N3 (MAJOR, positioning): the structural contribution is overstated relative to the literature.** Prop. 2 (optimal stopping region ⊆ one-step-look-ahead region, via C_k ≤ C_k⁰) and Prop. 4 (a stochastically monotone kernel plus non-decreasing terminal and running costs gives a monotone value) are textbook-type results. Prop. 1 is a set inclusion. The contributions paragraph calls them "a set of structural results about that model" but cites no optimal-stopping or MDP monotonicity literature for them. Fix: cite the standard results (for example Puterman, Ch. 4.7, on monotone value functions; Müller and Stoyan on stochastic monotonicity; the classical monotone-case and one-step-look-ahead results in Chow et al. 1971). Recast Props. 2 and 4 as applications. Keep Prop. 3 (pricing the over-trigger) and the modelling as the genuine theoretical novelty.

**N4 (MINOR): a §4.2 attribution conflicts with Table 3.** l.1399 says Baton's Det margin over thr. (27.9 vs 24.7) "comes mostly from the trigger correction of Proposition 2". The data say otherwise:
- trigger correction (Baton-ho − thr.): 26.1 − 24.7 = 1.4 pp
- depot return (Baton − Baton-ho): 27.9 − 26.1 = 1.8 pp (Table 3 also gives +1.8 against Baton-ho)

Fix: say the margin is split roughly evenly.

**N5 (MINOR): the empirical bound violation is unexplained.** §4.4 reports that the bound "holds on 734/735 routes" and gives no reason for the one exception. regret.csv confirms exactly one route has regret > bound. Because Prop. 3 is a theorem, the violation must come from estimation or Monte Carlo error. Fix: say so, and give that route's margin next to the standard error.

**N6 (MINOR): internal coherence of the theory.** The §3 introduction says the Lean formalisation proves Props. 2–3 without Assumption 1 on finite trees. The statements of Props. 2 and 3 still say "Under Assumption 1", and the proof of Prop. 2 uses the strong Markov property. Fix: add a remark after Prop. 3 stating the history-based version, which is exactly what covers the ρ=0.6 benchmark.

**N7 (MINOR): revision-history text inside the manuscript.** Remark 1 (l.872) says "The submitted version of this paper therefore recommended…". Referee-process history does not belong in an archival paper. Fix: restate the point as a finding with no reference to the earlier version.

**N8 (MINOR): length and structure.** The paper is about 62 pages with 15 tables and 10 figures, roughly twice the submitted length. Several items could move to an online appendix without weakening the argument:
- Table 11 (fresh-start biases)
- Table 14 (timing)
- the Lean paragraph
- the second negative result
- Figure 7 narrative

Fix: move them, and keep in the body the Tables 2, 3, 6, 8 and 9 line of argument.

**N9 (MINOR): unsupported field claims.** §3.3 cites a "genuine field failure in which a deployed policy saved exactly nothing" and "tuned thresholds drifted upward… on long routes", but no data or source is given. Figure 7's caption uses "$341", while the rest of the paper uses currency units. Fix: support or soften the field claims, and unify the units.

## Claim/number consistency problems

- **C1.** Abstract and introduction say Baton has the "lowest… every benchmark family" (introduction partly qualified). Table 6, 50% deliver-only: thr.-k 3.3 > Baton 3.1. See N1.
- **C2.** The abstract's "86%–96%" is computed from Table 2 only (41.2/47.8 = 86.2% to 52.5/54.9 = 95.6%). Tables 6 and 9 go down to 46%, and on 50% deliver-only DP50k/DP350k beat Baton by 3.7/3.6 pp.
- **C3.** Table 11 vs Tables 2 and 6. Baton-ho is 41.3 vs 42.3 (Salhi–Nagy), 18.9 vs 20.1 (Dethloff SAA) and 9.2 vs 10.6 (city). DP350k is 53.5 vs 54.0 and 54.6 vs 56.4 on the first two rows, and 10.7 vs 12.3 on city. fresh.csv reproduces Table 11's Baton-ho values as pooled ratio-of-sums over routes (41.3, 18.9, 9.2), whereas Tables 2 and 6 average per-instance savings. The Table 11 caption does not say this. Fix: state the aggregation, or align it with the other tables.
- **C4.** Table 13 at N=1000 vs Table 2 and Table 6:

  | Quantity | Table 13 | Table 2 / Table 6 |
  |---|---|---|
  | SAA Baton | 53.6 | 53.7 |
  | SAA thr. | 16.5 | 16.3 |
  | City thr. | 10.0 | 9.7 |
  | City Baton | 10.5 | 10.6 |
  | SAA DP50k | 22.2 | 22.1 |
  | City DP50k | 12.2 | 12.1 |

  These are small differences, but the reader is told all tables are generated from the same files. Fix: explain in a footnote (different route subset or aggregation).
- **C5.** §4.4 says the regret is "smallest on tightly packed deterministic plans (3.8%)", but Table 8's City-Det row is 2.0%. regret.csv confirms 2.0 for City and 4.0 for all Dethloff. Fix: say "smallest on the city and short routes".
- **C6.** Table 8 overall figures check out against regret.csv: regret 3.8%, bound 7.7%, clean-day share 90%, and 734/735 containment of the bound.

## Recommendation signal
**Minor revision** (bordering on major because of N1–N3).

The revision engages every round-1 item in substance with new experiments, and the paper fits C&OR well. What remains is claim hygiene, not missing science: the abstract's headline numbers need scoping (N1, C1, C2), a three-action clairvoyant bound or a corrected "achievable at all" sentence is needed (N2), and Props. 2 and 4 should be positioned against the standard optimal-stopping and MDP monotonicity results (N3). The paper should also be trimmed to an online appendix.
