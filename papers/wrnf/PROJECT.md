# Paper 4 — WRNF: when does distributional robustness change a network-flow decision?

**Working title.** *Decision relevance and tractability of two-stage
Wasserstein-robust network flow with ambiguous demand.*

**Target venue.** Computational Optimization and Applications (Springer,
EiC W. W. Hager). Regular stream — both currently open special issues are
restricted to conference participants, and the continuous-optimization one
is out of scope for this work.

**Status.** Spec + verified prototype. No manuscript yet. See `STATUS.md`.

---

## 1. The question

Two-stage distributionally robust optimization is usually sold as a way to
get *better decisions* than sample-average approximation (SAA) under limited
data. This paper asks when that is actually true for network flow, and the
answer turns out to be sharply conditional.

Model. Sources $i=1..m$ with first-stage installed capacity $x_i$ at unit
cost $f_i$; sinks $j=1..n$ with random demand $\xi_j$; arc $(i,j)$ with unit
shipping cost $c_{ij}$; unmet demand penalised at rate $p$:

$$Q(x,\xi) \;=\; \min\Big\{\textstyle\sum_{ij} c_{ij} y_{ij} + p\sum_j u_j \;:\;
\sum_j y_{ij} \le x_i,\;\; \sum_i y_{ij} + u_j = \xi_j,\;\; y,u \ge 0\Big\}$$

$$\min_x \; f'x + \sup_{P \in \mathbb{B}_\varepsilon(\hat P_N)} \mathbb{E}_P[Q(x,\xi)]$$

with $\mathbb{B}_\varepsilon$ a type-1 Wasserstein ball of radius
$\varepsilon$ around the empirical distribution of $N$ demand samples,
transport cost measured in $\ell_1$.

## 2. What the prototype has already established

`prototype_check.py` tests three claims by brute force — computing the
worst-case distribution exactly as an LP over transport plans on a
discretised support, so there is no reliance on the reformulation being
correct. All four checks pass; raw output in `prototype_results.txt`.

**Finding 1 (collapse / decision irrelevance).** Over an *unbounded*
support,

$$\sup_{P \in \mathbb{B}_\varepsilon} \mathbb{E}_P[Q(x,\xi)] \;=\; \frac{1}{N}\sum_k Q(x,\hat\xi_k) \;+\; \varepsilon p .$$

Verified to $\le 3.6\times10^{-15}$ at $\varepsilon \in \{0, 0.25, 0.5, 1, 2\}$.
Because the added term does not depend on $x$, the robust problem is the SAA
problem plus a constant, and **the robust decision equals the SAA decision**
(both $[4,4]$ on the test instance). Robustification shifts the objective and
changes nothing about what you do.

Mechanism: the demand dual is $\pi_j = \min\{p,\ \min_i (c_{ij}+\mu_i)\}$
with $\mu_i \ge 0$ the source-capacity dual. Refusing demand is always
available so $\pi_j \le p$; as demand grows capacity is exhausted, $\mu_i$
grows, and $\pi_j \to p$. Hence the Lipschitz modulus of $Q$ in the norm dual
to $\ell_1$ is exactly $p$, independent of $x$.

**The identity itself is prior art — do not claim it.** The general mechanism
(worst case = reference expectation + $\varepsilon\cdot$Lipschitz modulus over
a type-1 ball) is established, and as of February 2026 it is stated in
considerable generality:

> Huang, Li & Mao, *When Wasserstein DRO Reduces Exactly: Complete
> Characterizations of Projection Equivalence and Regularization*,
> Optimization Online, Sept 2025, rev. Feb 2026.
> [preprint](https://optimization-online.org/2025/09/when-wasserstein-dro-reduces-exactly-complete-characterization-projection-equivalence-and-regularization/)
> Their **Proposition 7, eq. (36)**:
> $\sup_{F\in B_1(F_0,\varepsilon)} H_1^F(f(\xi)) = H_1^{F_0}(f(\zeta)) + b\,\mathrm{Lip}(f)\,\varepsilon$.
> Taking $H_1$ to be the expectation ($b=1$) gives our identity's mechanism
> exactly. Also cite Mohajerin Esfahani & Kuhn; Gao & Kleywegt;
> Shafieezadeh-Abadeh, Kuhn & Mohajerin Esfahani (regularization via mass
> transportation).

Checked directly against the full text (`pdftotext`): that paper contains
**zero** occurrences of "two-stage", "recourse", "second stage", "value
function", "network", or "newsvendor". Its $f$ is a static loss on $\xi$; its
worked applications are chance-constrained programs and classification. It
never computes $\mathrm{Lip}(f)$ for an optimization value function and never
draws a conclusion about the *decision*.

So the contribution narrows to three things, and the manuscript must be framed
this way from the first paragraph:

1. **The modulus of a two-stage recourse value function is the recourse
   penalty**: $\mathrm{Lip}(Q(x,\cdot)) = p$ for network flow with simple
   recourse. A structural computation about an optimization value function, not
   a static loss — this is what nobody has done.
2. **Decision irrelevance.** Because that modulus does not depend on $x$, the
   robust argmin equals the SAA argmin. Robustness is not merely equivalent to
   a regularizer here; it is *inert*. The corollary is the point.
3. **The frontier.** Bounded support and capacitated arcs restore relevance;
   tractability dichotomy around that boundary.

**Finding 2 (bounded support restores relevance).** Truncating the support to
$\xi \le 6$ with $\varepsilon = 2$ moves the robust optimum to $[5.5, 6.5]$
against SAA's $[4,4]$ — genuinely different decisions. The closed form
becomes a valid upper bound rather than an identity ($33.20 \ge 32.96$), and
the worst case no longer has a closed form. **Bounded support is where the
algorithmic content lives, and it is exactly the case Hanasusanto & Kuhn
(2018) show is copositive-hard in general.**

**Finding 3 (the modulus is combinatorial, in two regimes).** Measured
against brute force:
- *binding* regime (capacity can be exhausted on the support): modulus $= p$;
- *slack* regime (capacity never binds): modulus $= \max_j \min\{p, \min_i c_{ij}\}$,
  read straight off the cost matrix with no LP.

Both exact. The transition between the regimes is what an algorithm must
track, and it is a combinatorial condition, not a numerical one.

## 3. Theorems to prove formally

1. **Collapse theorem.** Formalize Finding 1 for a general network (not just
   bipartite): with a finite recourse penalty and unbounded support, the
   type-1 Wasserstein DR two-stage network-flow problem is equivalent to SAA
   plus $\varepsilon p$, hence argmin-identical. State the exact hypotheses
   ($p$ finite; support unbounded in the relevant directions; $\ell_1$
   transport cost).
2. **Sharpness.** Show each hypothesis is necessary: counterexamples for
   bounded support (have one numerically — needs to be made analytic), for
   $p = \infty$ (hard demand satisfaction), and for a transport norm whose
   dual is not $\ell_\infty$.
3. **Combinatorial modulus on bounded support.** Characterize
   $\mathrm{Lip}(Q(x,\cdot))$ over a box support as a max over sinks of a
   residual-capacity shortest-path quantity; prove it is computable in
   $O(mn)$ (bipartite) / $O(|E| + |V|\log|V|)$ per sink (general), giving a
   polynomial oracle.
4. **Tractability frontier.** Prove the separation problem is polynomial for
   the uncapacitated-arc bipartite case under a box support, and NP-hard once
   arc capacities are present — the latter by reduction along the lines of
   Atamtürk & Zhang (2007), who prove separation is NP-hard for two-stage
   robust network flow on a bipartite graph.
5. **Integrality survival.** Determine whether the robustified problem retains
   integral optimal first-stage solutions under integral data and integral
   $\varepsilon$. In the collapse regime this is immediate (the objective
   differs from SAA by a constant); the bounded-support case is open and is
   the interesting half. I found no prior claim on this.

## 4. Algorithm to build

For the bounded-support regime, a cutting-plane / dual-ascent method:
outer problem over $x$; separation generates the worst-case distribution via
the combinatorial oracle of Theorem 3 rather than a nested LP/MILP. Prove
finite convergence. Warm-start successive network-flow solves from the
previous basis.

Baseline to beat: the monolithic Wasserstein reformulation handed to Gurobi
with a hard time limit — the honest comparison, and the one COAP expects.

## 5. Experiment plan

Sized for the actual hardware (4 cores, single machine — the Forrester &
Waddell template of 2,160 instances × 2-hour limits is *not* reproducible
here, so the design targets instances where the exact method finishes in
seconds-to-minutes and the specialized algorithm's advantage shows at
moderate scale):

- **Factorial synthetic sweep.** $m,n$ grid; $N \in \{10, 50, 200\}$;
  $\varepsilon$ across the collapse boundary; $p$ spanning slack and binding
  regimes; support width. NETGEN/GRIDGEN-style generation, released as a
  public testbed (COAP norm — an explicitly contributed instance set).
- **Collapse boundary experiment.** The headline empirical figure: sweep
  $\varepsilon$ and support width, plot the decision distance
  $\|x_{DRO} - x_{SAA}\|$, showing it is identically zero in the collapse
  regime and where it departs. This is the plot that makes the theory
  operational for a practitioner.
- **Scaling.** Specialized method vs. monolithic Gurobi reformulation;
  performance profiles (COAP's reporting convention).
- **Out-of-sample value.** Does the robust decision actually pay off on
  held-out data in the regime where it differs? An honest check that
  decision relevance is not merely mathematical.

## 6. Separation from the other papers in this repo — read before writing code

BATON (`papers/baton/`, under review at Computers & OR) also uses
Wasserstein-DRO machinery, as one of six *capacity-feasibility gates* inside
an ALNS route constructor. That is a different object and must be kept
visibly different:

| | BATON | WRNF (this paper) |
|---|---|---|
| Decision | vehicle routes + online handoff policy | first-stage installed capacity |
| Role of W-DRO | a feasibility gate filtering candidate routes | the objective itself |
| Question | when to hand off mid-route | whether robustness changes the decision at all |
| Method | ALNS heuristic + optimal stopping | exact cutting plane + complexity results |
| Instances | Dethloff, Salhi–Nagy, OSM city | NETGEN/GRIDGEN-style flow networks (new) |

Rules adopted, mirroring `papers/csonet2026/`:
- **Code is self-contained in this directory.** No imports from
  `svrpspd_wdro/`, in either direction. Verified mechanically before each
  commit.
- **No shared instances or results.** New generated testbed only.
- Cite BATON/TEMPO only if genuinely relevant, and if either is published by
  submission time, disclose the relationship in the cover letter.

## 7. Prior art that must be engaged head-on

All DOIs below verified against Crossref. **This section is still too
pre-2021-weighted and must be rebuilt on 2024–2026 work before drafting** —
that is where the scooping risk actually lives, and a referee will expect the
current frontier engaged, not just the foundational papers. Research in
progress; `optimization-online.org` is the venue this community preprints to
and is worth searching directly, not only Crossref/arXiv.

- **Huang, Li & Mao (2025, rev. Feb 2026), Optimization Online** — *When
  Wasserstein DRO Reduces Exactly.* Proposition 7 states the type-1
  regularization identity in general form; our collapse identity is a special
  case of its mechanism. **Cite prominently as prior art; claim only the
  value-function modulus, the decision-irrelevance corollary, and the
  frontier.** Verified from full text: no two-stage, recourse, value-function,
  or network content anywhere in it.

- Hanasusanto & Kuhn (2018), *Operations Research* 66(3),
  [10.1287/opre.2017.1698](https://doi.org/10.1287/opre.2017.1698) — two-stage
  DR linear programs over Wasserstein balls are copositive; tractable LP only
  absent support constraints. **The reason the bounded-support case is hard,
  and the reason the collapse result is not vacuous.**
- Atamtürk & Zhang (2007), *Operations Research* 55(4),
  [10.1287/opre.1070.0428](https://doi.org/10.1287/opre.1070.0428) —
  separation is NP-hard for two-stage robust network flow on a bipartite
  graph. **Blocks any naive "network structure makes separation polynomial"
  claim; the restriction in Theorem 4 has to dodge this explicitly.**
- Wang, You, Song & Zhang (2020), *EJOR* 284(1),
  [10.1016/j.ejor.2020.01.009](https://doi.org/10.1016/j.ejor.2020.01.009) —
  Wasserstein DR shortest path; their closing section already extends to
  min-cost flow with *arc-cost* ambiguity (ours is right-hand-side/demand
  ambiguity, single- vs two-stage — state the distinction early).
- Xie (2020), *Operations Research Letters* 48(4) — conditions under which
  RHS-uncertain two-stage Wasserstein DRO *is* tractable. DOI not yet
  verified; cite by ScienceDirect pii S0167637720300857 until confirmed.
- Mohajerin Esfahani & Kuhn; Gao & Kleywegt — the known
  regularization/collapse mechanism. Cite as prior, claim nothing.
- Filippi, Maggioni & Speranza (2025), *Computers & OR*,
  [10.1016/j.cor.2025.107096](https://doi.org/10.1016/j.cor.2025.107096) —
  survey; read its open-problems section before finalizing the contribution
  list.

## 8. Immediate next steps

1. Prove Theorem 1 analytically and make the Finding-2 counterexample exact.
2. Widen the prototype into a systematic sweep — confirm collapse and its
   boundary across many instances, not the single one tested so far.
3. Then, and only then, build the cutting-plane algorithm and start the
   manuscript in Springer `sn-jnl` with **numbered** references (COAP style).

## 9. Venue facts to build against

Verified from the live journal pages: no page limit stated, but recent
research articles run **32–56 pages** (mean ≈38); references are **numbered
in square brackets**, not author-year; abstract 150–250 words, 4–6 keywords;
Springer Nature LaTeX template encouraged; a **data availability statement is
mandatory**; code availability encouraged, not required; median submission to
first decision **6 days** (implying a large, fast desk-reject population — the
introduction has to establish computational contribution immediately). Review
model is *not* stated publicly on the journal's pages; the required title page
carries author names, i.e. the non-anonymized pattern, so prepare a
non-blinded manuscript unless the editorial office says otherwise.
