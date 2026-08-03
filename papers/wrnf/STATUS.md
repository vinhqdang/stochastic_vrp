# WRNF paper — status

- **Working title:** Decision relevance and tractability of two-stage
  Wasserstein-robust network flow with ambiguous demand
- **Venue:** was targeting Computational Optimization and Applications.
- **State: ABANDONED, 2026-07-27 — scooped. Never drafted.** The result
  is published: MEK (2018) Remark 6.7 states the decision-irrelevance
  thesis verbatim, and Duque, Mehrotra & Morton (SIOPT 32(3), 2022) do
  the two-stage version including both of this project's "findings",
  with a supply-allocation testbed that is essentially this model.
  **Byeon (arXiv:2501.05619, Jan 2025) Proposition 2 is the sharpest
  scoop** — the theorem verbatim for two-stage conic LPs with RHS
  uncertainty on *both* the whole space and the nonnegative orthant,
  with the epsilon-invariance / SAA-coincidence conclusion stated in
  the next sentence, which also closes the one technical gap we had
  found in our own claim. Byeon-Fang-Kim (SIOPT 2025) Lemma 2.2 owns
  the dual-vertex modulus;
  Lee-Kim-Moon (JORS 2021) already have "modulus = recourse penalty"
  for the newsvendor. Full citation list and the two technical errors
  found in our own claim are recorded at the top of `PROJECT.md`.
- **Policy:** kept as a documented negative result, not a plan. Do not
  resume without reading `PROJECT.md`'s abandonment banner first. The
  numerics in `prototype_check.py` are *correct* and reproduce a real
  (published) phenomenon — they are just not novel.
- **Checked and clear:** BATON already cites Ghosal, Ho & Wiesemann,
  "A Unifying Framework for the CVRP Under Risk and Ambiguity",
  *Operations Research* (2024), DOI 10.1287/opre.2021.0669
  (`ghosal2024unifying`, three citations in `main.tex` including as the
  M-DRO gate basis). The 2024-2026 survey flagged it as unavoidable for
  BATON's gate layer; it is already there, so no action needed.
- **Worth salvaging (the one live thread).** BATON's W-DRO capacity gate
  (`svrpspd_wdro/core/wdro_exact.py`) computes
  `Phi(r) = CVaR_alpha(max(0, f_r(xi) - Q)) + epsilon/(1-alpha)`, whose
  regularizer is a **route-independent additive constant** — the same
  phenomenon. Because the gate is a *feasibility constraint* rather than
  an objective, the constant tightens the threshold instead of
  cancelling, so the gate is not vacuous; but it does make the W-DRO
  gate equivalent to the SAA-CVaR gate at a shifted threshold. Worth
  checking against the BATON manuscript's claims when its decision
  arrives (it is frozen now); a referee could plausibly raise it.
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
