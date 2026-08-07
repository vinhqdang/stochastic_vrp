# ECLAIR — E-process Certification of LLM-generated Constraint Models with Adaptive Instance Routing

Paper 5. A sequential, anytime-valid, budget-aware certification layer
for LLM-generated constraint models. An LLM turns a natural-language
spec into candidate formal models; ECLAIR decides which candidate to
trust, with a statistical guarantee, by running each candidate against
a stream of cheap imperfect probes and betting against the null "this
model is faithful" with an e-process.

Status: ACTIVE — spec + simulation prototype. Target venue:
**Constraints (Springer)**, special-issue CFP on LLMs and constraint
programming (Topic 2, "solver-in-the-loop reasoning to control,
correct, or bound LLM outputs"; also touches Topics 1 and the
evaluation-methodology topic). CFP details (deadline, guest editors)
are recorded from the author's reading of the call and must be pinned
to the published CFP before submission — see `VERIFY_CITATIONS.md`.

Naming: follows the repo convention (evocative word backfit to an
initialism, like BATON and TEMPO). **Collision check required before
committing to the name**: "ECLAIR" is believed to collide with a
commercial static-analysis tool (BUGSENG ECLAIR, MISRA compliance) —
nearby enough to program verification to be a nuisance. Verify and, if
real, decide whether the collision matters at a CP venue (cf. the
SIDRA rejection in `RESEARCH_LOG.md`).

## 1. One-paragraph pitch

An LLM (or ensemble of prompting strategies) generates k candidate
formal models m_1..m_k from a natural-language spec. Ground truth does
not exist — that is the oracle problem. But we can generate a stream of
cheap, imperfect probes: metamorphic relations, brute-forced
micro-instances, cross-model feasibility checks. ECLAIR treats
certification as sequential hypothesis testing with e-processes: each
candidate carries a wealth process E_t(m) betting against the null
"model m is faithful." By Ville's inequality, rejecting whenever
E_t >= 1/alpha controls the probability of EVER falsely rejecting a
faithful model at level alpha — at any stopping time, under optional
stopping and continuation. A Kelly-style allocation rule decides, at
each step, which (candidate, probe) pair to run next to maximize
expected log-evidence growth per unit of solver time. The output is
not just a model but an auditable evidence certificate: "this model
survived probes carrying total evidence E, and the probability we
would certify an unfaithful model this way is at most alpha."

## 2. Positioning — the gap

Generation is crowded; verification is heuristic; nobody has a
guarantee. (All citations below are placeholders until verified — see
`VERIFY_CITATIONS.md`.)

- **Generation** (not our fight): CP-Agent (agentic constraint
  modelling with iterative refinement and solver feedback),
  ConstraintLLM (neuro-symbolic framework for industrial CP), CPEVAL
  (zero-shot CP-model-generation benchmark from CSPLib).
- **Verification** (the gap): two very recent lines attack it — (i) a
  falsification framework using provably valid metamorphic relations
  from LP duality, monotone comparative statics, and Le Chatelier
  principles, where a violation certifies unfaithfulness; (ii) an
  agent-based framework that builds a testing API, composes test
  cases, and applies optimization-oriented mutation testing. Both are
  **heuristic detectors**: empirical detection rates, no statistical
  guarantee on the accepted model, no principled stopping rule, no way
  to choose the next test under a compute budget.
- **Conformal prediction** for LLM outputs needs exchangeability
  between calibration and test data (frequently violated) and is
  fixed-sample — a poor fit for a sequential generate–test–repair loop
  that peeks at evidence continuously.

**The opening: a sequential, anytime-valid, budget-aware certification
layer for LLM-generated constraint models does not exist.** It sits
directly on top of the e-process machinery this repo built for TEMPO
(`svrpspd_wdro/ev/eprocess.py`: mixture bets over a theta-grid,
aGRAPA-style adaptive tilts, Ville-threshold alarms, dual-regime
combination) — ported into a fresh domain and reimplemented from
scratch here (see §9 on why no code is shared).

Why e-processes and not conformal / fixed-sample tests — three
structural reasons, each a selling point:

1. **Optional stopping IS the workflow.** Generate–verify–repair
   inherently peeks and adapts. Fixed-sample p-values and
   split-conformal calibration are invalidated by that; e-processes
   are designed for it. Stopping the moment evidence suffices is a
   direct solver-compute saving.
2. **Repair without p-hacking.** A falsified candidate feeds the
   violated probe back as a counterexample (CEGIS-style); the repaired
   candidate opens a FRESH e-process; evidence across candidates and
   rounds combines via e-BH for FDR control over the whole tournament.
3. **Heterogeneous, dependent probes.** Probes differ wildly in power
   and cost. E-values multiply across arbitrary dependence as long as
   each factor is a valid e-variable conditionally on the past —
   exactly what adaptively chosen probes need.

## 3. The algorithm

Inputs: NL spec s, candidate budget k, compute budget B
(solver-seconds), risk level alpha.

**Step 1 — Diversified generation.** Sample k candidates via
orthogonal variation: formalism (MiniZinc / CPMpy / OR-Tools CP-SAT /
MILP), decomposition prompt style, temperature, model family.
Diversity is instrumental: cross-checks are only informative when
errors are decorrelated.

**Step 2 — Probe pool.** Three tiers, ordered by oracle quality:

| Tier | Probe | Oracle quality | Cost | e-variable |
|---|---|---|---|---|
| A | micro-instances small enough for exhaustive enumeration (optimum must match); executable predicate checkers compiled from the NL spec and validated by cross-agreement | near-exact, calibrated error | high | Bernoulli LR with calibrated (p0, p1) from a held-out corpus |
| B | provably valid MRs: relaxation dominance (deleting a constraint cannot worsen the optimum), parameter monotonicity, symmetry/permutation invariance, LP/Lagrangian dual bounds, scaling laws | zero false-alarm BY PROOF | moderate | violation => e = infinity (deterministic falsification); pass => e = 1, zero null bleed |
| C | cross-feasibility: plug m_i's optimal solution into m_j's constraints; disagreement is soft evidence against at least one, weighted by the pool's agreement structure | weak, noisy | cheap | Bernoulli LR with CONSERVATIVE null rate (see §5, subtlety 2) |

Tier B imports and extends the falsification framework's MRs, but each
MR outcome becomes an e-variable rather than a binary alarm — the
framework gracefully contains hard certificates as the special case
e = infinity.

**Step 3 — Betting.** For each probe outcome X_t on candidate m, form

    e_t = 1 + lambda_t * (g(X_t) - E[g(X_t) | F_{t-1}, H0]),

lambda_t in [0, 1/||g||_inf) tuned by GRAPA/mixture methods (TEMPO's
construction: fixed theta-grid mixture with a 0-component as safety
net, 50/50 with an EWMA-learned adaptive tilt — both
F_{t-1}-measurable, hence validity-preserving), and update
E_t(m) = E_{t-1}(m) * e_t. For binary probes the mixture reduces to a
calibrated Bernoulli likelihood ratio.

**Step 4 — Adaptive routing (the novel layer).** At each step choose

    (m, p)* = argmax_{(m,p)} E[log e | posterior over faithfulness] / c(p),

i.e. maximize expected log-wealth growth per solver-second —
Kelly-optimal evidence acquisition under a budget (GROW criterion of
Grünwald et al.). It is a bandit over the candidates x probe-types
matrix; routing weights maintained by exponentiated gradient over
probe-type effectiveness give a regret bound against the best fixed
routing in hindsight. Routing choices are F_{t-1}-measurable, so they
never touch validity — only power.

**Step 5 — Decisions.** Reject m when E_t(m) >= 1/alpha →
counterexample-guided repair, new candidate, fresh e-process. Select
final model(s) via e-BH over survivors when B is exhausted or one
candidate's survivorship evidence dominates. If everything is
rejected, **abstain with the evidence ledger** — a principled "this
spec is ambiguous; here are the probes that killed every reading,"
doubling as a conversational-elicitation signal (CFP Topic 1).

## 4. Theory section — what we prove

1. **Anytime validity.** For each faithful m,
   P_{H0}(exists t: E_t(m) >= 1/alpha) <= alpha, robust to optional
   stopping/continuation and to adaptive probe routing (Ville +
   predictability of the routing rule).
2. **FDR control** of the final selected set via e-BH across the
   adaptively grown candidate pool (repairs included) — the guarantee
   no generate-and-vote pipeline has.
3. **Power / expected stopping time.** Under minimal detectability
   (every unfaithful model violates some probe class at rate >= delta),
   E[tau] = O(log(1/alpha) / KL), and Kelly routing attains the
   optimal growth exponent up to the EG regret term O(sqrt(T log K)).
4. **A structural result for CP** (the genuinely-new, on-brand-for-
   *Constraints* piece): a characterization of which MR families
   (relaxation dominance, symmetry, monotonicity) remain provably
   valid across CP-SAT / MILP encodings, extending the LP-centric
   results of the falsification framework to global constraints
   (alldifferent, cumulative).

## 5. Technical subtleties to resolve in the writing

These are the honest load-bearing points; each needs a paragraph or a
lemma in the paper.

1. **H0 is composite.** "m is faithful" does not pin down the law of
   probe outcomes (it depends on instance generators, checker noise,
   the other candidates). Validity therefore requires each e-factor to
   be a valid e-variable for EVERY law in H0, conditionally on the
   past. Tier B achieves this by proof (violation probability exactly
   0 under faithfulness). Tiers A and C achieve it via conservative
   calibration — see next point.
2. **Conservative calibration preserves validity.** For a binary probe
   with true null alarm rate p0_true unknown, bet with an upper bound
   p0_bar >= p0_true: e = (p1/p0_bar)^X ((1-p1)/(1-p0_bar))^(1-X) has
   E_{H0}[e] <= 1 whenever p0_true <= p0_bar <= p1 (the mean is
   increasing in p0_true and equals 1 at p0_true = p0_bar). So
   held-out calibration only needs a valid UPPER confidence bound on
   each probe family's null alarm rate — cite/verify the calibration
   route and make the corpus construction explicit (mutation-generated
   known-buggy models + hand-verified faithful models).
3. **Tier-C probes entangle candidates.** Cross-feasibility outcomes
   on m depend on which other candidates are alive; the null alarm
   rate must be bounded uniformly over pool states, or conditioned on
   the (predictable) pool composition. The clean route: calibrate
   p0_bar per (probe, pool-agreement-level) bucket, all
   F_{t-1}-measurable.
4. **Checker validity (Tier A).** Predicate checkers are themselves
   LLM-generated; cross-agreement validation bounds their error, and
   that bound feeds the same conservative-calibration argument.
5. **Repair opens a fresh e-process — but the REPAIRED candidate was
   chosen using data.** e-BH tolerates arbitrary dependence between
   e-values, which is exactly why it (and not BH on p-values) is the
   right combiner across the tournament. State this carefully.

## 6. Experiments

- **Benchmarks:** NL4Opt, ComplexOR, CPEVAL/CSPLib, logic grid
  puzzles, plus 2–3 industrial-style scheduling/routing specs
  (sanitized SmartOSC presales scenarios — industrial-relevance note).
- **Baselines:** single-shot generation, self-consistency majority
  vote, CP-Agent-style agentic refinement, fixed-sample mutation
  testing, split conformal.
- **Metrics:** certified accuracy; **empirical false-certification
  rate vs nominal alpha (the money plot)**; solver-seconds to
  decision; abstention behaviour on deliberately ambiguous specs.
- **Ablations:** routing (Kelly vs round-robin vs cost-blind), probe
  tiers, candidate diversity.

## 7. Prototype (this directory, `prototype/`)

`prototype/eclair_prototype.py` — stdlib-only Monte-Carlo simulation
of the statistical core, no LLM calls, no CP solver: candidates are faithful/unfaithful coin flips, the three probe
tiers are simulated with calibrated error rates and costs, betting is
the conservative Bernoulli LR + Tier-B hard certificates, and the
three routing policies compete under a shared solver-second budget.
Purpose: verify BEFORE any LLM/solver engineering that (i) the
e-process's empirical false-rejection rate of faithful candidates
respects alpha, and (ii) Kelly routing beats round-robin and
cost-blind routing on detection and cost-to-detection at realistic
probe powers. Results in `prototype/prototype_results.txt`;
regenerate with `python3 prototype/eclair_prototype.py`.

**Verdict (2026-08-07): precondition MET.** At alpha = 0.05, 8
candidates/rep, 200 solver-sec shared budget, 500 reps: empirical
false-rejection of faithful candidates is 0.0000 (Kelly), 0.0031
(round-robin), 0.0005 (cost-blind) — all far below alpha, as
conservative calibration predicts; detection of unfaithful candidates
is 86.4% (Kelly) vs 59.6% (round-robin) vs 76.6% (cost-blind), at
mean solver-cost per kill 12.2 vs 16.6 vs 22.6. The e-process
separates cleanly at realistic probe powers and the routing layer is
worth its section.

Next engineering step (not yet built): CPMpy + OR-Tools CP-SAT
pipeline on ~10 CSPLib problems with mutation-injected unfaithful
candidates (no LLM needed for the first real experiment — mutations
play the role of LLM errors, which also builds the calibration
corpus), then LLM generation on NL4Opt.

## 8. Why this wins at this venue

It hits three of the four CFP topics simultaneously (solver-in-the-
loop verification; multi-model generate/judge separation; systematic
evaluation methodology focused on correctness), and it answers the
question CP-community guest editors actually care about: *what
guarantee do I get?* A certification layer is complementary to the
guest editors' own agendas (modelling pipelines, constraint
acquisition) rather than competing — favourable for desk review. For
this repo it is low-risk: the e-process core, the EG routing analysis,
and the empirical discipline (calibration corpora, replication
scripts) all exist in other papers here, recombined into an unclaimed
niche.

Alternatives considered and set aside (for the record): (a) LLM-guided
cut selection with online-learning regret guarantees — crowded by
ML4CO, hard to beat learned baselines by the deadline; (b)
anytime-valid solver-portfolio selection from text features — solid
but incremental over algorithm selection. ECLAIR dominates both on
novelty-per-effort.

## 9. Relation to the rest of the repo — self-containment policy

ECLAIR shares NO code, instances, or results with BATON (under review,
Computers & OR), TEMPO (under review, Transportation Science), or the
CSoNet/JOCO paper (submitted). The e-process construction is ported at
the level of published technique (mixture bets, predictable adaptive
tilts, Ville thresholds — all citable to the public literature), and
the implementation here is written from scratch against a different
data model (candidate x probe outcomes, not routing-day event
streams). This mirrors the deliberate separation argued in
`papers/csonet2026/README.md`: with two papers under review elsewhere,
sharing artifacts would invite simultaneous-submission/salami-slicing
concerns. Consequently ECLAIR's code lives self-contained in this
directory, NOT in a new `svrpspd_wdro/` subpackage — the shared engine
(VRP instances, simulator, cost model) is useless to it anyway.

The manuscript's related-work section may cite TEMPO once it is
public; until then, cite only the underlying e-process literature.

## 10. Open actions

- [ ] Pin the CFP: URL, deadline, guest editors, topic list →
      `VERIFY_CITATIONS.md`, then `STATUS.md`.
- [ ] Name collision check (BUGSENG ECLAIR) — keep or rename.
- [ ] Verify the prior-art citations (CP-Agent, ConstraintLLM, CPEVAL,
      both verification frameworks) and the e-process/stats canon —
      all listed in `VERIFY_CITATIONS.md`.
- [ ] Prototype phase 2: CPMpy + CP-SAT on CSPLib with mutation-
      injected faults; build the calibration corpus.
- [ ] Formal statements + proofs of §4 items 1–3; scope item 4
      (which global constraints admit which MR families).
- [ ] Decide author list.
