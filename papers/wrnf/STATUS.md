# WRNF paper — status

- **Working title:** Decision relevance and tractability of two-stage
  Wasserstein-robust network flow with ambiguous demand
- **Venue:** Computational Optimization and Applications (Springer) —
  **NOT YET WRITTEN.** Regular stream, not a special issue: both open
  COAP special issues are restricted to conference participants, and
  the Brazilian Workshop one is continuous-optimization only.
- **State:** spec + verified prototype. `PROJECT.md` holds the plan;
  `prototype_check.py` / `prototype_results.txt` hold the numerical
  evidence for the three central claims. No manuscript file yet, no
  algorithm implementation yet.
- **Policy:** this directory is an **active work-in-progress** and is
  freely editable — unlike `papers/baton/`, `papers/tempo/`, and
  `papers/csonet2026/`, which are all frozen. Switch it to the frozen
  convention the moment it is submitted, and record the date here.
- **Authors:** not yet fixed. Decide before drafting; the other papers
  in this repo use Quang-Vinh Dang (BUV, corresponding) plus a subset
  of Minh Ngoc Dinh (Millennia Education), Hoang-Viet Vu (BUV), and
  Phuc-Son Nguyen (UEH).
- **Idea selection (for the record).** Four candidates were
  novelty-checked against the literature before this one was chosen:
  1. *DR shortest-path interdiction over a Wasserstein ball* — rejected
     as CROWDED. Ketkov & Prokopyev, *Computers & OR* (2026),
     10.1016/j.cor.2026.107533, already does Wasserstein DRO
     interdiction with a polynomial-size MILP reformulation and Benders
     decomposition; Kang & Bansal (2022, 2024) already publish finite
     convergence and "cutting planes beat monolithic reformulation" for
     interdiction. Substituting shortest path for packing is an
     increment, not a contribution.
  2. *ML-guided exact optimization with certified bounds* — rejected as
     CROWDED. Klamkin, Tanneau & Van Hentenryck, arXiv:2510.15850
     (2025), is the stated contribution: learn primal and dual, certify
     the gap a posteriori, fall back to the exact solver. Also,
     ML for branching/node/cut selection is exactness-preserving
     already, so there is no guarantee to restore there.
  3. *Wasserstein-robust network flow* — **CHOSEN**, after reframing.
     The naive form of the claim ("network structure makes the DRO
     separation oracle polynomial") is provably false: Atamtürk & Zhang
     (2007) prove separation is NP-hard for two-stage robust network
     flow on a bipartite graph, and Hanasusanto & Kuhn (2018) show
     two-stage Wasserstein DR LPs are copositive-hard with support
     constraints. That is what makes the correct paper a *dichotomy*
     rather than an algorithm-only paper.
  4. *Dispatch with decision-dependent (endogenous) deadlines* —
     rejected on two grounds. It collides with the Moving Firefighter
     Problem (10.3390/math11010179; 10.1002/net.70037), whose
     contribution list is the same one proposed here, for a closer
     model. And it is an extension of our own MWHED paper currently
     under review at JOCO, by the same authors, so an editor could
     reasonably read it as salami-slicing. Poor COAP fit besides.
- **Novelty check, first result (2026-07-27).** The collapse *identity*
  is prior art and must not be claimed: Huang, Li & Mao, "When
  Wasserstein DRO Reduces Exactly" (Optimization Online, Sept 2025,
  rev. Feb 2026), Proposition 7 eq. (36), states the type-1
  regularization identity in general form, and our identity is a
  special case of its mechanism. Verified against the paper's full
  text that it contains **no** two-stage, recourse, value-function,
  network, or newsvendor content — its loss is static and its
  applications are chance-constrained programming and classification.
  So the surviving contribution is narrower than first framed: (1) the
  Lipschitz modulus of a two-stage *recourse value function* equals the
  recourse penalty, (2) the decision-irrelevance corollary that follows
  from it being independent of x, (3) the bounded-support /
  capacitated-arc frontier. `PROJECT.md` §2 and §7 record this. Two
  further literature checks were still running when this was written.
- **Relationship to the other papers in this repo.** Independent by
  construction. BATON also uses Wasserstein-DRO machinery, but as a
  capacity-feasibility *gate* inside a route constructor, whereas here
  it is the objective of a two-stage capacity-installation problem —
  see `PROJECT.md` §6 for the full separation table and the
  no-shared-code / no-shared-instances rules adopted.
