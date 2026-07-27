# Research log — the search for paper 4

**Purpose.** Papers 1–3 are frozen under review and paper 4 (WRNF) was
abandoned as scooped, so the repo currently has **no open paper**. This file is
the running record of the search for the next one: every idea evaluated, every
verdict, and — most valuable — **the citation ledger of prior art that killed
things**, so no dead idea gets re-trodden. Six ideas have been rejected so far.
Without this note that history is invisible and repeatable.

Keep appending. Newest findings at the top of each section. Every citation here
was retrieved and checked, not recalled; anything unverified is labelled so.

---

## Working name: **KAIROS** *(provisional)*

Greek for *the opportune moment* — apt, because the live candidate direction is
about a decision that chooses **when** to act, and thereby chooses which
distribution of travel times it faces.

> **K**eyed **A**mbiguity over **I**nterval-conditioned **R**eference measures
> with **O**ptimal-transport **S**moothing

Naming notes: the repo's convention is an evocative word backfit to an
initialism (BATON, TEMPO), so this fits. **"SIDRA" was considered and
rejected** — it is established traffic-engineering software (SIDRA
Intersection), and a name collision inside the transport domain would be a
persistent nuisance. **Do not commit to this name until the tractability
verdict below lands** — if the direction changes, the name should change with
it.

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

**Recommendation on record:** stop competing in DRO theory. Next candidate
should either (a) be an applied computational paper built on the real-data
pipeline, or (b) sit in self-contained combinatorial optimization like MWHED,
where the repo has already succeeded. Get a novelty check before any building —
that discipline has now saved seven manuscripts.

---

## Ideas evaluated, with verdicts

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
