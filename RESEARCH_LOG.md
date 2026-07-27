# Research log — the search for paper 4

**Purpose.** Papers 1–3 are frozen under review and paper 4 (WRNF) was
abandoned as scooped, so the repo currently has **no open paper**. This file is
the running record of the search for the next one: every idea evaluated, every
verdict, and — most valuable — **the citation ledger of prior art that killed
things**, so no dead idea gets re-trodden. Seven ideas have been rejected so far.
Without this note that history is invisible and repeatable.

Keep appending. Newest findings at the top of each section. Every citation here
was retrieved and checked, not recalled; anything unverified is labelled so.

---

## Working name: **KAIROS** *(on hold — the direction it named is dead)*

Greek for *the opportune moment* — coined for the decision-dependent-reference
candidate, which chose **when** to act and thereby chose which travel-time
distribution it faced. **That idea was abandoned as scooped on 2026-07-27**, so
the name is unattached. It stays here because it would still fit any routing
paper about choosing dispatch timing under uncertainty — reuse it only if that
is what the next idea turns out to be.

> **K**eyed **A**mbiguity over **I**nterval-conditioned **R**eference measures
> with **O**ptimal-transport **S**moothing

Naming notes: the repo's convention is an evocative word backfit to an
initialism (BATON, TEMPO), so this fits. **"SIDRA" was considered and
rejected** — it is established traffic-engineering software (SIDRA
Intersection), and a name collision inside the transport domain would be a
persistent nuisance. **Do not commit to this name until the tractability
verdict below lands** — and that verdict came back ABANDON, so the name is
parked rather than adopted.

---

## Current state (2026-07-27)

| | |
|---|---|
| Target venue | Computational Optimization and Applications (Springer), regular stream |
| Topic | **undecided** — the 7th candidate was just killed; no direction currently alive |
| Ideas rejected | **7** |
| Real data | **solved** — verified sources secured, see below |
| Blocking | needs a **change of strategy**, not another idea in the same area — see below |

### No live direction — and a strategy problem worth naming

All seven rejected ideas sit in **distributionally-robust / robust optimization
theory**. That area is exceptionally crowded and is being worked systematically
by identifiable groups (Kuhn–Wiesemann, Xian Yu, Byeon, Bansal, Mehrotra), who
publish in *Math. Prog.* / *OR* / *SIOPT* faster than a part-time effort can
track. Five of the seven deaths were "someone published this in the last 12–18
months"; two were our own faulty reasoning. Continuing to generate candidate
ambiguity-set variations is a losing game — the eighth will likely die the same
way.

**What this repo has actually demonstrated it is good at**, from the three
papers that exist: applied stochastic routing with a real simulator, real
instances and careful empirical evaluation (BATON, TEMPO), and self-contained
combinatorial theory with hardness + approximation results (MWHED). Neither
competes head-to-head with DRO theory groups.

**Two assets in hand that are hard to scoop**, because they are data and
engineering rather than a theorem someone can beat us to:

1. A **verified real traffic + weather data pipeline** — NYC DOT link-level
   speeds and travel times with polylines map-matchable to our OSM graphs,
   Open-Meteo hourly weather for all five cities, plus a CC0 HCMC set keyed by
   real OSM node IDs. See `data_sources/`.
2. **Empirical findings nobody's theorem gives them**: 1.94× diurnal speed
   swing; a weather effect that is *masked* by time-of-day and doubles once
   hour-of-week is removed (−0.079 → −0.195); and weather shifting the **mean
   without inflating the spread** (residual sd 1.191 wet vs 1.303 dry).

The third finding is the interesting one and it cuts *against* the robust-
optimization framing: if real weather does not widen dispersion, an ambiguity
set is the wrong instrument for it. That is a substantive modelling observation
and it points toward an applied/computational paper where the contribution is
the algorithm plus the data-grounded evaluation, rather than a new ambiguity set.

### The one verified open gap, with its risks

**There is no distributionally robust time-dependent VRP, and no DR
time-dependent shortest path.** Verified by negative search across Crossref,
arXiv, optimization-online and publisher pages. Time-dependence is currently
modelled deterministically (piecewise-linear travel-time functions), by
scenarios (Kubek 2025, DOI `10.3390/su172411308`), by a Markov background
process (Kamphuis, Levering & Mandjes 2025, *Computers & OR* 183:107148, DOI
`10.1016/j.cor.2025.107148`), or by a budget uncertainty set — **never as a
period-indexed ambiguity set with a finite-sample guarantee.**

Three risks to price in before touching it:
1. **It is publicly signposted and funded.** Filippi, Maggioni & Speranza's
   2025 survey (*Computers & OR* 182:107096, DOI
   `10.1016/j.cor.2025.107096`) names it in future directions — *"utilizing
   historical and spatio-temporal data (such as weather conditions,
   time-of-day patterns, or congestion levels) … estimate conditional
   distributions of uncertain parameters"* — under PRIN grants. We would be
   racing Speranza's and Maggioni's groups.
2. **The robust version is already being built.** Malheiros, Poss, Nesello &
   Subramanian, "The robust time-dependent VRP with time windows and budget
   uncertainty" — ROADEF 2026 and EURO 34 abstracts, no journal version yet.
   Travel time `t_ij(s,delta)` with both components piecewise-linear in
   departure time and `delta` in a Bertsimas–Sim budget set.
3. **"Index the ambiguity set by the chosen slot and dualize" is scooped
   several times over.** To survive, the contribution must be something the
   general theory does not hand you — e.g. exploiting the **finite, totally
   ordered** period index plus FIFO monotonicity to get a polynomial or exactly
   convex reformulation where generic decision-dependent DRO is bilinear, or a
   **monotone comparative statics** result on the optimal departure period as
   the radius grows.

**Recommendation on record:** stop competing in DRO theory. Get a novelty check
before any building — that discipline has now saved eleven manuscripts.

### The cells that ARE open (checked 2026-07-27) — two papers' worth

**(i) Solution path / breakpoints in the ambiguity radius, for a DISCRETE
decision.** Verifiably empty, by several independent negative searches: arXiv
`"solution path" AND "distributionally robust"` = **0**; arXiv
`"comparative statics" AND "distributionally robust"` = **0**; the 225-page
Kuhn–Shafiee–Wiesemann survey (*Acta Numerica* 34:579–804, DOI
`10.1017/S0962492924000084`, arXiv:`2411.02549`) contains **zero** occurrences of
"comparative statics" and has no monotonicity-in-radius or solution-path
section; Long, Qi & Zhang's supermodularity paper (*Management Science*, DOI
`10.1287/mnsc.2023.4748`) has **no radius parameter at all**; and Tang & Fan's
"solution path algorithm for distributionally robust regression" (*Optimization*
73:3275–3296, 2024, DOI `10.1080/02331934.2024.2341938`) paths over the two
**elastic-net hyperparameters**, *not* the Wasserstein radius. The existing
inertness theory (Bartl et al.; Gao–Chen–Kleywegt; Gotoh–Kim–Lim) is about
**smooth decisions and objective values**; none of it says anything about
*discrete solution identity*.
**Why this is the strongest thing to come out of the whole search:** it turns
the null result that has dogged every idea here — robustness barely moves the
objective — into a *theorem about when the combinatorial decision switches*,
with an exact breakpoint characterisation and a path-following algorithm
beating grid search over epsilon. Algorithm + complexity + numerics, and our
real-data pipeline supplies the instances.
**Precondition before building:** the model's path must be non-degenerate, i.e.
actually have breakpoints. Ours currently is not (see the BATON gate note
below). Check numerically first — the tooling in
`probes/sinkhorn_collapse_check.py` is the right shape for it.

**(ii) B3-1: joint optimum stratification with integer sample allocation.**
Ordered index `1..n` with per-cell populations and variances; choose a
contiguous partition into `k` strata **and** an integer allocation
`n_1..n_k` with `sum n_b <= B`, minimising `sum_b W_b^2 sigma_b^2 / n_b`. Both
halves are classical — allocation alone is separable convex resource allocation
(Hochbaum, *Math. of OR* 19:390–409, 1994, DOI `10.1287/moor.19.2.390`),
stratification alone is the Bühler–Deutler/Eubank DP — but **the joint problem
has no stated complexity class anywhere**: no hardness proof, no approximation
guarantee, no Monge characterisation. Zero MWHED adjacency, and, importantly, a
*different technical skeleton* (DP + Monge/SMAWK + convex-allocation proximity,
versus MWHED's knapsack-DP + value-scaling FPTAS + matroid greedy) — which is
what a salami-hunting referee actually pattern-matches on.
**Precondition:** settle whether it is NP-hard *before* drafting. If weakly
NP-hard, the full package follows (hardness, pseudo-poly DP, FPTAS,
Monge-tractable case). If polynomial via proximity/scaling, it is a shorter but
still clean paper — just a different one.
**Downgraded after a deep dive (same day).** The stratification half is a
**51-year-old literature already using this cost structure**: Khan, Nand & Ahmad,
*Survey Methodology* 34(2):205–214 (2008) solve `minimise sum_h phi_h(x_{h-1},x_h)`
over ordered cut points with per-stratum cost `W_h sigma_h^2 (W_h/n_h - 1/N)` —
variance divided by the stratum's sample count — by the Bühler–Deutler DP; and
Eubank's *SIAM Review* survey declares optimal grouping/spacing/stratification a
single solved breakpoint problem. So only the **integer** allocation is arguably
new, which is thin. Do not start this without first reading Wu, *J. Algorithms*
12(4):663–673 (1991), DOI `10.1016/0196-6774(91)90039-2` (O(kn) via SMAWK for
weighted least-squares interval partitioning) and Elomaa & Rousu, "On the
Complexity of Optimal Multisplitting" (ISMIS 2000, DOI
`10.1007/3-540-39963-1_58`; journal version *Fundamenta Informaticae*
47(1-2):35–52) — a direct title collision that neither agent could access.
**Runner-up:** Budgeted Active Time (maximisation-under-budget dual of the
Active Time problem; Chang, Gabow & Khuller, *Algorithmica* 70:368–405, DOI
`10.1007/s00453-013-9807-y`; NP-completeness settled recently in
arXiv:`2112.03255`). Better-shaped for JOCO but its paper skeleton is
MWHED's exactly — flagged as a real salami hazard.

**(iii) The Amazon LMRRC uncertainty derivative.** 9,184 real 2018 routes, five
US cities, per-instance point-to-point travel times **and real service times**,
CC BY-NC on AWS Open Data (Merchán et al., *Transportation Science*, DOI
`10.1287/trsc.2022.1173`). No derivative adding calibrated travel-time
uncertainty exists. Better raw material than either SVRPBench or BonnTour used,
and BonnTour's Uber Movement source is **discontinued** while ours (NYC DOT,
Open-Meteo) is live and refreshable. Pair with LaDe (arXiv:`2306.10675`) for
dwell-time laws.

### Venue correction — COAP is probably the wrong journal

Benchmark/computational precedents at COAP *do* exist (Martí, Reinelt & Duarte,
COAP 51:1297–1317, 2011, DOI `10.1007/s10589-010-9384-9`; de Moraes & Coelho,
COAP 88:349–378, 2024, DOI `10.1007/s10589-024-00551-1`), correcting an earlier
note here that claimed none. **But the journal is overwhelmingly continuous
optimization** — Riemannian methods, proximal gradient, SQP, cubic
regularisation — and routing appears roughly once a decade. Precedent exists;
the referee pool does not. Better homes for a discrete/routing computational
paper: **EJCO**, **Discrete Optimization** (the BonnTour precedent: benchmark
bundled with theory), *Transportation Science*, or *EJOR*.

---

## Ideas evaluated, with verdicts

### 8-11. Four candidates checked 2026-07-27 — all four dead as pitched

**B1 ordered partitioning of a context index — SOLVED-ALREADY, and two of our
own claims about it were technically wrong.**

Two corrections from a dedicated deep dive, both worth keeping because they
generalise:
1. **The Monge intuition was backwards.** Tested exhaustively over all valid
   quadruples: the `sigma^2/n_b` term alone gives **0/210 quadrangle-inequality
   violations** — for `w(i,j) = f(N(i,j))` with N additive over the interval, QI
   collapses to `f(P+C)+f(P+D) <= f(P+C+D)+f(P)`, which holds **iff f is
   convex**, and `1/x` is convex. So the variance term is the *friendly* one. It
   is the **bias/SSE term** that breaks QI — 0/210 violations when values are
   monotone along the index, **75/210 when they are not**. Since the diurnal
   conditional mean is *not* monotone in hour-of-day, QI generically **fails**
   for our actual use case and SMAWK/Knuth/LARSCH would not apply. That is
   precisely why Jagadish et al. only achieved O(N^2 B) and Guha–Koudas–Shim
   §3.1 wrote *"the answer, unfortunately, is no"* with an Omega(n) obstruction.
2. **The variance term risks being a placement-independent constant — the WRNF
   failure mode, third occurrence.** If per-block variance is mass-weighted as an
   integrated-MSE risk requires, `sum_b (n_b/n)(sigma^2/n_b) = sigma^2 k/n`,
   **independent of where the cuts fall**, so the objective collapses to plain
   V-optimal segmentation with a k-penalty. Avoiding that needs a
   decision-theoretic justification for weighting every block equally regardless
   of mass. **Check for this degeneracy in any partition objective before
   building on it.**

 Every piece
exists in three literatures that already cite each other: the O(n^2 k) DP is the
V-Optimal histogram problem (Jagadish et al., VLDB 1998 — *no DOI exists, do not
invent one*) and earlier Bühler & Deutler, *Metrika* 22:161–175 (1975), DOI
`10.1007/BF01899725`, unified by Eubank, *SIAM Review* 30(3):404–420 (1988), DOI
`10.1137/1030092`; (1+eps) approximation by Guha, Koudas & Shim, *ACM TODS*
31(1):396–438 (2006), DOI `10.1145/1132863.1132873`; Monge/SMAWK speedups are the
textbook application (Aggarwal et al., *Algorithmica* 2:195–208, DOI
`10.1007/BF01840359`; Burkard, Klinz & Rudolf, *Discrete Applied Math* 70:95–161,
DOI `10.1016/0166-218X(95)00103-X` — in the target journal, so a referee has read
it). The 2-D product case I hoped was open is **NP-hard with a published
sum-objective reduction**: Grigni & Manne, DOI `10.1007/BFb0030123`; Ahrens &
Boman, arXiv:`2005.12414`.

**B2 period-dependent dispatch cost with release windows — SOLVED-ALREADY, and
it is a textbook exercise.** Lenstra & Shmoys, *Elements of Scheduling*,
arXiv:`2001.06005`, Ch. 2 §2.1 pp. 37–38 constructs this exact instance as a
min-cost flow. The constraint matrix is a network matrix, totally unimodular by
Hoffman–Kruskal *regardless of which arcs are deleted*, so the "interval windows
give consecutive-ones tractability" theorem would be **vacuous**. The owning
literature is time-of-use scheduling, not time-dependent processing times: Wan &
Qi, *NRL* 57(2):159–171 (2010), DOI `10.1002/nav.20393`; Fang et al., *Annals of
OR* 238:199–227 (2016), DOI `10.1007/s10479-015-2003-5` (**already has the
polynomial algorithm for equal workload + unimodal "pyramidal" prices** — our
unimodal-congestion variant); Penn & Raviv, *J. Scheduling* 24(1):83–102 (2021),
DOI `10.1007/s10951-020-00674-3` (extends to release times and due dates).
Structurally cannot yield the FPTAS/matroid shape, because the base problem is
polynomial and the transportation polytope is already integral.

**A1 real-data uncertainty benchmark + decision-relevance study — CROWDED, the
framing SCOOPED.** Three separate occupants: **SVRPBench** (arXiv:`2505.21887`)
claims "first open benchmark ... stochastic dynamics in vehicle routing at urban
scale" — though its congestion is a hard-coded Gaussian mixture on synthetic 2-D
points, i.e. parameter-picking, not calibration; **BonnTour** (Blauth et al.,
*Discrete Optimization* 53:100848, 2024, DOI `10.1016/j.disopt.2024.100848`)
already ships OSM networks + **measured Uber Movement speeds** for 10 cities
including New York; and the decision-relevance design is **Chassein, Dokka &
Goerigk**, *EJOR* 274:671–686 (2019), DOI `10.1016/j.ejor.2018.10.006` — six
uncertainty sets built from a **live Chicago traffic feed**, evaluating which are
"actually valuable", seven years ago, on shortest path. **Mahmutoğulları & Guns**,
arXiv:`2310.17368`, already reports our null in routing: a deterministic model
with one well-chosen quantile matches formal robust optimization.

**A2 DR time-dependent VRP — DEAD, both escape hatches closed.** Hatch (a),
FIFO monotonicity for tractability, is taken by **Malheiros, Poss, Nesello &
Subramanian** (ROADEF 2026 abstract, full PDF read): their Assumption 1 *is*
FIFO, **Theorem 3.1** proves relaxing it makes robust time-dependent path
feasibility NP-complete, **Proposition 1** gives a polynomial DP under FIFO, and
they already report that their model "yields better reliability–cost trade-offs
than robust models ignoring time dependence" on real data. What remains is
"replace the budget set with a Wasserstein ball" — the move this log already
records as scooped. Hatch (b), monotone comparative statics in the radius, is
largely taken by **Blanchet, Murthy & Zhang**, *Math. of OR* 47(2):1500–1529
(2022), DOI `10.1287/moor.2021.1178`, whose §2.4 is literally titled
"Comparative statics analysis", and by **Bartl, Drapeau, Obłój & Wiesel**, *Proc.
R. Soc. A* 477(2256), DOI `10.1098/rspa.2021.0176`, which gives explicit
first-order corrections to the **optimizer** in epsilon.

### 7. Sinkhorn ball with a decision-dependent reference measure — **ABANDONED, scooped (2026-07-27)**

The candidate above. Tractability was **fine**; novelty was fatal.

**The scoop.** **Yu & Basciftci, "Distributionally robust optimization with
multimodal decision-dependent ambiguity sets," *Mathematical Programming*, DOI
`10.1007/s10107-026-02337-1`** (published 9 Mar 2026; read in full as
arXiv:`2404.19185v2`). Their eq. (3),
`Θ(y) = { Σ_{l=1}^{L} p_l P_l : p ∈ Δ(p̂(y)), P_l ∈ U_l(y) }`, is this idea's
framework:
- `L` finitely many **modes** = our (hour-of-day, weather-state) cells;
- the **reference mode probability `p̂(y)` is decision-dependent** — i.e. the
  first-stage decision selects which conditional the second stage faces, with
  their Remark 1 covering exact one-hot selection;
- two-stage with recourse, **Wasserstein-based** per-mode sets (Thms 6–9), and
  §3.2.3 makes the reference atoms themselves affine in `y`;
- **binary decisions ⇒ McCormick envelopes ⇒ exact MILP** — precisely the
  finite-disjunction rescue we hypothesised;
- a **separation-based decomposition algorithm with finite convergence** plus
  facility-location and shipment-planning numerics.

**Second, independent occupant.** Zhu, Yu & Bayraksan, arXiv:`2406.20004v2`
(preprint, read in full) — residuals-based contextual DRO whose Wasserstein
nominal is **both decision- and covariate-dependent**, general two-stage with
recourse, asymptotic optimality and finite-sample guarantees, Benders with
nonlinear cuts and proven finite convergence, shipment-planning-and-pricing
numerics. This is the idea's exact architecture, with Wasserstein.
**Third:** Hu, Tong, Peng & Tan, *J. Global Optimization*, DOI
`10.1007/s10898-026-01589-7` (abstract only, Springer-gated) — ambiguity sets
driven by first-stage **integer** decisions, MINLP via Lagrangian duality.

**Note for future searches: Xian Yu is a co-author on both killing papers.**
One group is systematically working decision-dependent DRO. Treat that whole
area as contested.

**An independent second check confirmed this and found an EARLIER, cleaner
kill.** Recorded because it predates everything above by four years:

- **Noyan, Rudolf & Lejeune (2022), "Distributionally Robust Optimization Under
  a Decision-Dependent Ambiguity Set with Applications to Machine Scheduling and
  Humanitarian Logistics," *INFORMS J. on Computing* 34(2):729–751, DOI
  `10.1287/ijoc.2021.1096`** — abstract verbatim: *"as our ambiguity sets, we
  consider **balls centered on a decision-dependent probability distribution**.
  The balls are based on a class of earth mover's distances that includes both
  the total variation distance and the Wasserstein metrics."* And the nominal is
  explicitly a **finite mixture selected by the decision** (§3):
  `P = Σ_{s=1}^S π_s(x) P_s` with `π_s` affine in `x`. Selection is the
  degenerate case `π ∈ {0,1}^S`. So "the decision picks its reference measure
  from a finite menu, with a Wasserstein ball centred on it" was **published in
  a journal in 2022**.
- **Yu & Basciftci §5.1 names our exact case by name**: *"assume p̄_l = 0, ∀l,
  Σ_l λ_l = 1 and **Σ_i y_i = 1**. This reduces to the convex combination of
  distributions discussed in Section 4.1.2 of Hellemo et al. (2018)."* With
  binary `y` summing to one, `p̂(y)` *is* a selection indicator. Our idea is
  their `ρ = 0` corner — strictly *less* general than what they published, and
  flagged in their own text.
- **Hellemo, Barton & Tomasgard (2018), *Computational Management Science*
  15(3–4):369–395, DOI `10.1007/s10287-018-0330-0`**, §4.1.2 — the
  decision-selects-a-distribution construction in two-stage recourse, pre-DRO.
  (Verified via Crossref and via Yu & Basciftci's citation; its full text not
  obtained.)
- **Qu, Jia & You, arXiv:`2508.06965`** (preprint, single-stage) — Definition 3
  plus Remark 1: nearest-neighbour interpolation over per-decision empirical
  measures *is* pure selection, with the ball centred on the selected one.
- **Chen, Sim & Xiong (2020), *Management Science* 66(8):3329–3339, DOI
  `10.1287/mnsc.2020.3603`** — **event-wise ambiguity sets**: finitely many
  events, ambiguous conditional per event. Our structure minus
  decision-dependence.
- **Field-maturity signal:** *Mathematical Programming* ran a **special issue**
  on "stochastic programming and distributionally robust optimization with
  decision-dependent uncertainty" — Lejeune, Romeijnders & Krokhmal (2026), DOI
  `10.1007/s10107-026-02383-9`. When a top journal devotes a special issue to
  the exact topic, the area is fully staffed. Likely referees include the very
  authors above.

**Confirmed dead on a THIRD, independent ground — our own numerics, plus a
conceptual objection that is worse than the novelty problem.**

`probes/sinkhorn_collapse_check.py` tested the question nobody in the
literature had: does Sinkhorn collapse to Wasserstein at a shifted radius, the
way the type-1 ball collapsed in WRNF? Results:

- **C1 — no collapse.** Best-matching Wasserstein radius leaves residual
  sd **2.02** (a reparametrisation would give ~0) and the argmins genuinely
  differ. So the WRNF trap does *not* apply — this is a real model.
- **C2 — the claimed differentiator holds.** Max |second difference| in the
  decision: **3.7e-2 (Sinkhorn) vs 4.0e-1 (Wasserstein)**, i.e. ~10.8x
  smoother, as the entropic soft-max predicts.
- **C3/C3b — but it never changes the decision.** Across a location sweep and a
  dispersion sweep, the two models selected the same cell every time
  (**0/6 flips**). Sinkhorn buys a smoother objective and never a different
  action.

**Methodological finding worth keeping** (it generalises beyond this idea): a
*location* difference between cells cannot discriminate between ambiguity
models for a translation-equivariant loss. Translating a cell's samples
translates the optimal decision and leaves the optimal value exactly unchanged,
so both models are indifferent by symmetry — the Wasserstein gaps came out
identically `0.0000` at every shift, and small non-zero Sinkhorn gaps (~1e-3)
were support-truncation artefacts. **Only dispersion differences can
discriminate.** This bites directly: our real data says weather shifts the
travel-time **mean without inflating the spread**, i.e. reality supplies
exactly the kind of difference that provably cannot separate these models. The
application could not have justified the method.

**The conceptual objection, which is the real kill.** If the decision merely
*selects* an exogenous conditional and does not deform it, the model is **not
decision-dependent ambiguity at all**: index the periods, take the ambiguity
set over the joint period-indexed vector, and the decision enters only the
objective and constraints — ordinary two-stage DRO, no new duality, no
bilinearity, no hardness. **Shehadeh, *Transportation Science* 57(1):197–229
(2023), DOI `10.1287/trsc.2022.1153`, builds exactly this construction** — one
exogenous ambiguity set per period, the schedule decides which bind — **and
correctly declines to call it decision-dependent**, while citing Luo–Mehrotra
and Basciftci et al. So the framing is either wrong (selection only) or already
published (if the decision truly deforms the conditional). Structurally the
same trap as WRNF: a robustness claim that cancels into a known model.

**And the "decision selects the travel-time distribution" framing has three
separate owners:**
- **Hall (1986), *Transportation Science* 20:182–188, DOI
  `10.1287/trsc.20.3.182`** — arc travel-time distributions conditional on
  departure time. Forty years old; it is why the optimal object is a policy,
  not a path.
- **Wang, Delage & Coelho (2026), *M&SOM*, DOI `10.1287/msom.2024.0899`** —
  "Data-Driven Stochastic VRP with Deadlines Under **Decision-Dependent Travel
  Time**": a route feature vector selects the conditional travel-time
  distribution via kNN/KDE weights, logic-based Benders, real food-delivery
  data. Owns the framing in a VRP, published. (SAA, not DRO; no departure-time
  or weather conditioning — that is all it leaves open.)
- **Kong, Li, Liu, Teo & Yan (2020), *Management Science* 66(8):3480–3500, DOI
  `10.1287/mnsc.2019.3366`** — DR appointment scheduling where the *scheduled
  time* selects the no-show distribution, explicitly endogenous-on-time,
  bilinear copositive program. "Timing selects the distribution" was proved in
  a top-5 journal six years ago.
- Also: **Cheng, Adulyasak & Rousseau (2024), *M&SOM* 26(4):1402–1421, DOI
  `10.1287/msom.2022.0339`** — real **wind data** → cluster-wise ambiguity set
  over flight times → adaptive DRO. Real weather conditioning a travel-time
  ambiguity set, in an INFORMS A-journal.

**Effect-size warning from the same problem class.** Bao, Vogiatzis & Kontou,
"Risk-Averse Stochastic User Equilibrium on Uncertain Transportation Networks"
(Optimization Online 2026/03, no DOI; PDF prefixed `TranSci__`, i.e. submitted
to the same venue as TEMPO) report that risk aversion and distributional
robustness *"refine rather than fundamentally alter"* equilibrium flow
patterns. That is the WRNF failure mode restated by someone else, in road
networks.

**One genuinely open thing found, and it matters for what comes next.** The
second check found **no routing/VRP paper** where "departure-time or
service-window selection picks the demand or travel-time distribution", and
optimization-online returns **zero** hits for "contextual decision-dependent".
So the *phrase* and the *routing application* are unclaimed even though the
mathematics is thoroughly claimed. Any future use of this must position itself
as an **instantiation** of Noyan–Rudolf–Lejeune / Yu–Basciftci, with all novelty
in the routing layer — never as a new ambiguity-set class.

**Two corrections to our own reasoning, recorded because both were wrong:**
1. **The convexity fear was unfounded.** We worried log-sum-exp composed with an
   x-dependent measure destroys convexity. It does not:
   `G(x,λ) = λε·log E_Q[e^{f(x,z)/(λε)}]` is the **perspective** (in `λε`) of the
   convex log-sum-exp functional, hence **jointly convex in (x, λ)**, and
   composing with `f` convex in `x` preserves it. Strong duality also survives a
   decision-dependent reference measure, holding pointwise in `x`. What actually
   breaks convexity is only the **reweighted outer expectation**
   `Σᵢ p̂ᵢ(x)·Gᵢ(x,λ)` — the same bilinearity as a decision-dependent *radius*,
   no worse. The log-sum-exp was never the obstruction.
2. **"Sinkhorn closed forms don't reach piecewise-linear-convex losses" was
   true but the inference from it was wrong** — Wang–Gao–Xie solve the
   piecewise-linear newsvendor numerically in their §5.1.
Also worth knowing: their "2-SDRO" means *order-2* Sinkhorn, **not** two-stage.

**Residual delta, for the record** (not enough for a paper): the joint-convexity
observation, since both Sinkhorn papers needlessly scalar-search `λ`; exact
mixed-integer-*convex* reformulation via perspective functions rather than
McCormick, since a Sinkhorn ball gives binary × nonlinear log-sum-exp rather
than binary × linear duals; and SAA consistency for the nested
log-of-expectation under mixed-integer selection, since the BSMD/MLMC
guarantees require convexity in the decision and die with binaries.

**The unanswerable referee objection:** *"this is Yu & Basciftci with the ball
swapped."* True at the level of model, tractability mechanism, algorithm class
and guarantee. And our own data findings (1.94× diurnal, 1.04 mph wet-hour
slowdown) motivate **conditioning**, which is the incumbents' contribution —
they do not discriminate between ambiguity balls.

### 6. Phase-transition Wasserstein radii on bounded support — **SCOOPED**
Characterise the ε-interval where the two-stage DR decision differs from both
SAA and the fully-robust solution.
- **Agra & Rodrigues**, *EJOR* 326(1):174–188 (2025), DOI
  `10.1016/j.ejor.2025.04.053` — publishes the *algorithm*: exact and heuristic
  methods for finding the control-parameter values yielding all relevant
  first-stage solutions, via an ε-constrained Pareto-front scheme, tested on
  scheduling, berth allocation and **facility location**.
- **Byeon**, arXiv:`2501.05619` §4 — the theory statement with explicit radii
  at both ends.
- Attracts the same citation triple that killed WRNF. A re-run of a dead thread.

### 5. Sinkhorn DRO for network flow / routing (as originally pitched) — **CROWDED; two of my own claims were wrong**
- **My "prove non-degeneracy unlike type-1 Wasserstein" was not a
  contribution** — it is the standing motivation of every Sinkhorn-DRO paper,
  and Huang–Li–Mao now have the complete characterisation of when W-DRO
  degenerates.
- **My motivation had a one-line rebuttal**: "W₁ degenerates ⇒ use Sinkhorn" is
  answered with "or just use a bounded support, as Duque–Mehrotra–Morton 2022
  did" — they prove the cone degeneracy then pivot to box support with a
  cutting-plane algorithm and a two-stage *network-flow* testbed.
- **Wang, Gao & Xie**, *Operations Research* (2026), DOI
  `10.1287/opre.2023.0294` (arXiv:`2109.11926`) — the anchor. Duality for
  arbitrary loss; closed-form/conic specialisations **only for linear and
  quadratic losses**, so a two-stage LP value function (max-of-affine) is *not*
  reached. Algorithm: biased stochastic mirror descent + bisection on λ.
  Numerics: newsvendor, portfolio, adversarial classification. No two-stage, no
  recourse, no network.
- Occupied adjacent cell: **IEEE T-Power Systems** (2026), DOI
  `10.1109/TPWRS.2026.3684359` — exponential-cone reformulation of Sinkhorn
  worst case + CVaR-approximated DR joint chance constraints, mixed-integer
  exponential cone program on a **network**, IEEE 14/118-bus. Also **Yang et
  al.**, *AIChE J.* (2023), DOI `10.1002/aic.18177` (Sinkhorn DRCCP conic
  template).
- Survives only if narrowed to the two-stage max-of-affine case with the
  exponential-cone-outer / recourse-inner decomposition as the contribution.

### 4. WRNF — two-stage Wasserstein-robust network flow — **ABANDONED, scooped**
Full record in `papers/wrnf/PROJECT.md` (abandonment banner) and its
`STATUS.md`. Summary of the kill:
- **Mohajerin Esfahani & Kuhn**, *Math. Prog.* 171:115–166 (2018), DOI
  `10.1007/s10107-017-1172-1`, **Remark 6.7 "ε-insensitive optimizers"** —
  states the thesis verbatim, naming the newsvendor: x-independent Lipschitz
  modulus ⇒ "the distributionally robust optimization problem and the SAA
  problem share the same minimizers irrespective of the Wasserstein radius ε."
  *Verified against the arXiv PDF (1505.05116), line 2273.*
- **Duque, Mehrotra & Morton**, *SIAM J. Optimization* 32(3):1499–1522 (2022),
  DOI `10.1137/20M1370227` — the two-stage version and **both** of our
  "findings": *"when using the Wasserstein distance and the support of the
  random parameters is an unbounded convex cone, there is no point in solving
  the DRTSP"*, plus §3.2 hyper-rectangular support restoring hedging, plus
  Thm 4 (quadratic transport also breaks it). Their §5 testbed is a
  supply-allocation network-flow problem — our model plus a holding cost.
  *Verified against the Optimization Online preprint.*
- **Byeon**, arXiv:`2501.05619` (Jan 2025) **Proposition 2** — sharpest form,
  for two-stage conic LPs with RHS uncertainty on **both ℝᵏ and ℝᵏ₊**, with
  ε-invariance / SAA-coincidence in the next sentence. Its Remark 1 disclaims
  novelty, crediting MEK, Hanasusanto–Kuhn and Byeon–Fang–Kim.
- **Byeon, Fang & Kim**, *SIAM J. Opt.* 35(1):506–536 (2025), DOI
  `10.1137/23M1626839` — **Lemma 2.2**: value function is L-Lipschitz with
  L = max_{π∈Π}‖π‖_q. Our "modulus = recourse penalty" is this with q=∞.
- **Lee, Kim & Moon**, *JORS* 72(8):1879–1897 (2021), DOI
  `10.1080/01605682.2020.1746203` — modulus = penalty rate for the newsvendor,
  citing MEK Remark 6.7 explicitly.
- **Huang, Li & Mao**, Optimization Online 2025/09 rev. 2026/02 — general
  type-1 regularization identity (Prop. 7, eq. 36). *Verified from full text
  that it has no two-stage/recourse/network content — its loss is static.*

**Two real errors in our own claim**, recorded so they are not repeated:
over Ξ = ℝⁿ the second stage is infeasible for negative demand so the worst
case is +∞ (needs ℝⁿ₊); and the πⱼ ≤ p argument bounds the dual from above
only, whereas the modulus needs a bound over the whole dual feasible set.
Separately, `papers/wrnf/sweep.py` has a **known bug**: its Part A demands
exact equality where the collapse is only asymptotic in box width (error
≈ ε/box). Never fixed — the paper died first.

### 3. DR shortest-path interdiction over a Wasserstein ball — **CROWDED**
**Ketkov & Prokopyev**, *Computers & OR* (2026), DOI
`10.1016/j.cor.2026.107533` already does Wasserstein-DRO interdiction with a
polynomial-size MILP reformulation and Benders decomposition; Kang & Bansal
(2022, 2024) publish finite convergence and "cutting planes beat monolithic
reformulation" for interdiction. Substituting shortest path for packing is an
increment.

### 2. ML-guided exact optimization with certified bounds — **CROWDED**
**Klamkin, Tanneau & Van Hentenryck**, arXiv:`2510.15850` (2025) is the stated
contribution: learn primal and dual, certify the gap a posteriori, fall back to
the exact solver. Also, ML for branching/node/cut selection is
exactness-preserving already, so there is no guarantee to restore.

### 1. Dispatch with decision-dependent (endogenous) deadlines — **REJECTED**
Collides with the Moving Firefighter Problem (DOIs `10.3390/math11010179`,
`10.1002/net.70037`), whose contribution list is the same one proposed, for a
closer model. And it extends our own MWHED paper under review at JOCO, by the
same authors — an editor could reasonably read it as salami-slicing.

---

## Data: real traffic and weather — **secured**

Full detail, verified URLs and licences: **`data_sources/README.md`**.
Probe: **`data_sources/probe_nyc_traffic_weather.py`**.

Recency is **not** required (author's call, 2026-07-27) — data from 2020 or
earlier is acceptable. Convenient, because every verified source is a
historical archive; nothing was ever blocked on recency.

### Headline sources (all fetched with no credentials)
- **NYC DOT Traffic Speeds** (Socrata `i4gi-tjb9`) — per-link **speed and
  travel_time** plus a `link_points` WGS84 polyline, so **map-matchable to the
  OSM edges `make_city_instances.py` already builds**. 125 links, 2017→live,
  ~110 M rows. Independently confirmed: link 4362249, 21.12 mph, 187 s,
  polyline present. **No stated licence** — cite by URL, don't redistribute bulk.
- **METR-LA / PEMS-BAY** (Zenodo `10.5281/zenodo.5146275`) — loop-detector
  speeds, 5-min, **CC BY 4.0**, the licence-clean option for a public extract.
- **HCMC segment velocities** (Kaggle `thanhnguyen2612`) — **CC0**, keyed by
  **genuine OSM node IDs** (verified against the live OSM API to 7 decimals),
  2020-07→2021-04. Unexpected find: every Vietnamese government portal has
  nothing. COVID-era and sparse (~9 obs/segment) — use to validate, not anchor.
- **NYC TLC trip records** — per-trip durations, **OD taxi-zone IDs only**.
- **Open-Meteo archive** — hourly temp/precip/snow/wind, **no API key**, any
  lat/lon, so it covers all five `City/` instances.

**Correction on record:** coordinates in TLC parquet exist **only for
2009–2010**; from 2011-01 every month including 2016-06 is zone-only. The
common belief that lat/lon persists to mid-2016 describes the *retired CSVs*.

### What the data actually shows (Jan 2025, 3,355,999 clean trips, 734 hours)
- Diurnal congestion: **19.1 mph at 05:00 vs 9.8 mph at 17:00**, a 1.94× swing.
- 24,837 (OD, hour) cells with ≥30 observations; 8,183 with ≥100. At fixed OD
  and fixed hour, duration still has **median CV 0.301** — real dispersion.
- Weather matters and **time-of-day hides it**: raw precip–speed correlation
  −0.079, rising to **−0.195** after removing hour-of-week means. Wet hours
  **1.04 mph slower** (Welch t = −5.15, p = 6.9×10⁻⁶); snow 0.90 mph slower
  (p = 1.7×10⁻⁵).
- **But weather shifts the mean without inflating the spread** (residual sd
  1.191 wet vs 1.303 dry). So weather is **context conditioning the
  distribution**, not an extra dimension widening an ambiguity set. This is
  what points at a conditional/decision-dependent formulation rather than
  "add a channel".
- **Wind is not a clean channel** — residual correlation *positive* (+0.191),
  confounded. Do not present it as a congestion driver.

### Dead ends — do not plan around these
PeMS (login-gated), UTD19 (email-gated), LargeST (7.56 GB, flow not speed,
CC BY-**NC**), **Uber Movement (decommissioned, no mirror — would have had HCMC
and Paris)**, Paris/Île-de-France (flow and occupancy only; no speed),
Barcelona (captcha, ordinal only), Hamburg (categorical, empty archive stubs),
NDW (no fetchable archive), all Vietnamese government portals, IEEE DataPort
GRACTRANET (subscription). **All four commercial APIs forbid what we need** —
HERE §6.4(b) explicitly bars exposing content under open-data licences.

For **Paris there is no honest open path** to travel times; the
transferred-congestion-multiplier approach in `data_sources/README.md` is what
keeps Paris in the instance set without inventing data.

---

## Process notes worth keeping

- **optimization-online.org is essential** and was the author's suggestion. It
  surfaced Huang–Li–Mao when Crossref and arXiv searches had not. Search it
  directly on every novelty check.
- **Recent literature is where the scooping risk is.** Also the author's point.
  Older papers matter as *constraints on what may be claimed* (Atamtürk–Zhang
  2007 still blocks a polynomial-separation claim in 2026), which is a
  different role from related work.
- **Verify agent claims against source PDFs.** Both decisive kills this session
  were confirmed by downloading the paper and grepping it — MEK Remark 6.7 at
  line 2273 of arXiv:1505.05116, and DMM's "no point in solving the DRTSP".
- **Prototype before theorising.** `papers/wrnf/prototype_check.py` falsified
  the first version of its own theory (predicted Lipschitz modulus 2.0,
  measured 10.0) — the marginal action is refusing demand at the penalty rate,
  not shipping on the cheapest arc.
- **A "complete characterisation" paper appearing recently in your corner is a
  warning.** Huang–Li–Mao (Feb 2026 revision) signalled how crowded the
  regularization-equivalence area is; the WRNF kill followed within the hour.

---

## Open BATON item (paper 1, frozen — do not act, revisit at decision)

`svrpspd_wdro/core/wdro_exact.py` computes
`Phi(r) = CVaR_alpha(max(0, f_r(xi) - Q)) + epsilon/(1-alpha)`. That regularizer
is a **route-independent additive constant** — the same degeneracy phenomenon.
It is *not* vacuous, because the gate is a feasibility constraint, so the
constant tightens the threshold rather than cancelling. But it does make the
W-DRO gate equivalent to the SAA-CVaR gate at a shifted threshold, and the
manuscript presents six distinct gates. A referee could raise it. Checked and
clear on a related worry: BATON **does** already cite Ghosal, Ho & Wiesemann,
*Operations Research* (2024), DOI `10.1287/opre.2021.0669` (`ghosal2024unifying`,
three citations in `main.tex`).
