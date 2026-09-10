# PARCEL — status

- **Working title:** PARCEL: Price-Aware Relay of Context over
  Endogenous Links — budget-constrained information dissemination in
  multi-agent LLM systems when utility is neither monotone nor
  submodular.
- **Venue:** **AAMAS 2027**, the 26th International Conference on
  Autonomous Agents and Multiagent Systems, **Hanoi, Vietnam,
  3–7 May 2027**. Main track.
- **State:** **ACTIVE — drafting.** Manuscript at **8 pages of body**
  (`main.tex`, AAMAS `sigconf` anonymous mode) — the full allowance,
  with references spilling to page 9 and therefore not counted.
  Compiles clean with zero errors against the official class. Theory in `THEORY.md`,
  empirical record in `RESULTS.md`, spec in `PROJECT.md`.

### Blocking before submission

1. ✅ **RESOLVED 2026-09-07 — the LLM-agent context-reduction paragraph
   is now cited.** Seven records were verified directly from the arXiv
   API (title, author list, first-posting date, arXiv comment field)
   and entered in `references.bib`: BPS (2608.19993), PACMS
   (2606.20047), RCR-Router (2508.04903), Phase Transition
   (2601.17311), AgentPrune/"Cut the Crap" (2410.02506), GTD
   (2510.07799, ACL 2026 Main), KVComm (2510.03346, ICLR 2026). The
   paragraph now also states the BPS relationship explicitly rather
   than leaving a referee to discover it.
2. ✅ **RESOLVED 2026-09-07 — `harshaw2019` verified** against the
   PMLR v97 landing page's own `citation_*` metadata: Harshaw,
   Feldman, Ward, Karbasi; pages 2634–2643; ICML 2019; ISSN 2640-3498.
   PMLR renders the title with lowercase "beyond"; the .bib now
   matches and carries the URL.
3. ✅ **RESOLVED 2026-09-07 — BPS read in full** (model, Theorem 1, and
   the Appendix A proofs of Lemmas 3–4). The positioning survives and
   is now stated precisely rather than by assertion; see
   `PROJECT.md` §10.2.
4. Author list, affiliations, `\acmSubmissionID` (blank pending
   OpenReview registration).
5. The 8 calibration citations in `VERIFY_CITATIONS.md` remain
   unverified; none of them is currently load-bearing in `main.tex`.
6. ✅ Page budget: **8 of 8 used**, body ending on page 8 with
   references starting there and running onto page 9. All proofs are
   in-body (appendices are not exempted). Any further addition now
   requires cutting something — check the count after every edit.
7. The AAMAS copyright block, `\setcopyright{ifaamas}` and
   `\acmConference` were **missing** and are now in place; the footer
   had been rendering the ACM placeholder "Conference'17, Washington
   DC". Worth re-checking after any preamble edit.
8. ⚠️ **The OpenReview submission form shown 2026-09-09 lists Abstract
   2026-10-02 / Submission 2026-10-09 — one day later than the dates
   recorded below**, verified 2026-09-06 from the call page. Flagged to
   the user; not yet resolved which is authoritative. Trust the live
   form over this file until confirmed.
9. ✅ **RESOLVED 2026-09-10 — a detailed technical review (external,
   forwarded by the user) found and we fixed six real correctness
   defects**, verified independently against the model/code/data before
   editing, not taken on the reviewer's word alone:
   - **Proposition 3 was malformed**: its formal statement claimed
     `rel` modular for *both* halves, but the submodularity-violation
     witness is necessarily non-modular (superadditive). Split into two
     explicit witnesses with their own hypotheses (§4.2 in `main.tex`).
   - **The "scalar budget price" claim was false in general.** A
     Lagrange multiplier decentralizes the DP's optimum only when every
     `v_i` is concave; Proposition 4 explicitly allows non-concave
     (S-shaped) `v_i`. Added a verified counterexample
     (`v_1=(0,0,3)`, `v_2=(0,2,2)`, `B=2`: no scalar price reaches the
     optimum) and removed every "decides locally"/"price a system
     could publish" claim; kept the DP, which is unaffected.
   - **The density-rule admission test conflated token cost with
     confusable load**, contradicting the model's own stated
     distinction. Rewrote the dominance argument to be exact (no
     first-order hedge, no differentiability needed) in terms of the
     true marginal penalty, and separated the density-*greedy
     heuristic* (which does specialize `conf_i(S):=c(S)` and needs
     `rho_i` differentiable) from the exact test.
   - **Table 3's numbers didn't match the committed data.** Recomputed
     every row directly from `results/summary_musique.json`
     (authoritative): oracle tokens 99→106, seed-set Δacc
     −0.263→−0.392, random −0.632→−0.667, no-context −0.763→−0.792.
     Table 2 gained a 95% CI column. Also fixed a genuine naming
     collision: the `q=0.3` arm was labeled "marginal allocation" in
     Tables 2–3, colliding with Algorithm 1's own name; renamed to
     "density, loose" throughout (Table 5 is the actual Algorithm 1
     results and was correctly named already).
   - **Statistical overclaiming**: "costs nothing"/"free" language
     throughout (abstract, intro, conclusion) rewritten to report the
     point estimate + 95% CI honestly (e.g. the 2.7× arm is "−4.2
     points, CI crossing zero," not "free"). Added a cluster-robustness
     check (block bootstrap by original instance, not receiver) to
     Discussion — it changes little (3/5 rows unchanged to 3dp), which
     is itself worth stating rather than assuming.
   - **Proposition 1 overclaims**: "the factor n is the best case" is
     false (an instance where every receiver needs the same item gives
     ratio 1); retracted throughout. Fixed `n≥1` vs `n≥2` for the
     unboundedness clause, and replaced "OPT_seed=0" division language
     with a rigorous derivation (ratio `1/(1-λ(n-1))→∞` as `λ`
     approaches the threshold), verified numerically.
   Also fixed: Corollary 5's knapsack-oracle remark needed
   `rel_i` modular (else only a `(1-1/e)`-oracle exists); Algorithm 1's
   pseudocode computed exact `v_i` but returned tilde-decorated
   (approximate) sets — genericized to take either oracle; the
   `u_i(\emptyset)=0` normalization was asserted but never derived from
   `rel_i,conf_i,rho_i`; the IM-identity claim ("the seed-set
   restriction is *exactly* what IM imposes") softened to an analogy,
   since IM's diffusion can still reach nodes differently once the seed
   set is fixed; the "quadratic in agents" and tokens-as-cost claims
   qualified (architecture- and billing-dependent). Two logged traces
   moved to `supplementary_material/` (now `logged_traces.md`) to make
   room. The reviewer's larger asks — a real end-to-end multi-agent
   benchmark with dependent messages, stronger receiver-aware routing
   baselines, full CIs for Table 5's four value-curve comparisons — are
   **not** addressed (would need new experiments/API budget); recorded
   as Open Problems (iii) and folded into the power caveat on Table 5.
   Body still exactly 8 pages after all of this; two real table-width
   overflows were introduced and caught by rebuilding (`\small` fonts,
   shorter labels) before this was called done.
10. ✅ **RESOLVED 2026-09-10 — a second, independent technical review
    (external, forwarded by the user) found one genuine error in our own
    round-9 fix plus several real presentation/rigor gaps**, again
    verified independently (Python re-derivation, direct code/data
    checks) rather than taken on the reviewer's word:
    - **Proposition 3's "no single instance repairs both" claim was
      itself false**, and this was *our* mistake from round 9, not the
      reviewer's. A direct-sum instance (disjoint receivers, one
      complementarity-witness, one saturation-witness) forces both
      degeneracies at once, verified exhaustively
      (`code/degeneracy_check.py`'s new `combined()` check over the
      5-item ground set). Rewrote the proposition and added an explicit
      modesty caveat that the class of escaping instances is not
      claimed to be unique or deep.
    - **Proposition 2's proof wrongly claimed the general-$n$ case
      "contains Multiple Knapsack."** False: Multiple Knapsack needs
      per-bin capacities and forbids item reuse; PARCEL has one shared
      budget and copyable items, so general $n$ reduces to ordinary
      Knapsack over receiver-item pairs instead. Removed the false
      sentence, added a correct remark.
    - **The exact dominance test (Eq. 4) was described as usable for
      incremental/forward construction**; it is only a necessary
      condition on a *completed* set — under complementarity an item
      failing it mid-construction can pass once the rest of $S_i$ is in
      place. Added that clarification explicitly.
    - **Table 3's caption called the density rule an "exact-dominance
      heuristic"** — self-contradictory. Fixed to "density-greedy
      heuristic (an approximation of the exact test, not an instance of
      it)."
    - **Table 2 vs. Table 3 appeared inconsistent**: Table 2's headline
      per-receiver-vs-seed row is actually per-receiver $k{=}3$ (220
      tokens) vs. seed $k{=}3$ (251 tokens), but Table 3 only listed
      per-receiver $k{=}5$ (420 tokens), so the two tables looked like
      they disagreed. Verified the correct numbers directly from
      `results/summary_musique.json`, added explicit configs/token
      counts to every Table 2 row label, and added the missing
      per-receiver $k{=}3$ row to Table 3.
    - **Concavification's "restores a market-clearing price" claim
      needed an ex-post/ex-ante qualification**: it holds only in
      expectation under independent per-receiver randomization; exact
      per-realization budget feasibility needs correlated (dependent)
      rounding. Added that caveat.
    - **"Free"/"no measured cost" framing persisted in a few spots**
      (abstract, Finding 1, cost-quality-frontier paragraph, conclusion)
      despite round 9's fix elsewhere; reworded to the point-estimate +
      95% CI framing throughout ("uncertain 4.2-point loss, CI −10 to
      +1").
    - Redesigned Table 2 so cluster-robust CIs/$p$-values are the
      *primary* reported statistic (previously only a Discussion
      side-check), and correspondingly compressed the Discussion's
      clustering paragraph since it no longer introduces new numbers.
      Fixed Figure 2's overlapping point labels (leader lines) and
      added CI error bars to the oracle/density-loose/uniform-$k{=}5$
      points.
    - Expanded the reproducibility protocol paragraph with exact model
      names, temperature, call date, prompt template, extraction rule,
      lexical-score formula, and embedding model/dimension, including an
      honest disclosure that there is no calibration/evaluation split,
      so "tuned uniform" cannot be certified free of leakage.
    - **Not addressed** (recorded here rather than silently dropped):
      Table 5's four value-curve comparisons still lack full per-model
      paired CIs (would need new analysis of data not in hand — stays
      an Open Problem/power caveat); the proxy-vs-real-multi-agent-system
      limitation is unchanged from round 9 (needs a new experiment,
      already listed as Open Problem (iii)); IM's positioning in Related
      Work was tightened stylistically but not restructured further, a
      judgment call. The reviewer's "missing spaces" complaints
      (`can stillreach`, `item cost it1`, etc.) were re-checked against
      the `.tex` source and confirmed, again, to be `pdftotext`
      extraction artifacts around line breaks/ligatures, not real
      source bugs — same conclusion as round 9's version of this
      complaint.
    Body still exactly 8 pages after all of this; the new Table 2 design
    introduced a real 29pt table-width overflow, fixed with `\footnotesize`
    and tighter `\tabcolsep` before this was called done.
11. ✅ **RESOLVED 2026-09-10 — a third technical review (external,
    forwarded by the user) found the "tokens" metric was never
    real, plus a self-inflicted sample-accounting error and several
    rigor gaps.** Each finding was verified against code/data (or, for
    tokens, against Gemini's own API) before acting, per the standing
    rule of this changelog:
    - **"Tokens" were word-count × 1.33, not a model tokenizer** ---
      the review's most consequential catch. Recomputed every token
      figure in Tables 2/3, Figure 2, and the composite-receiver
      discussion using Gemini's real `countTokens` endpoint on the
      exact transmitted context for every logged call (item selection
      is deterministic, so this needed no new generation calls, only
      lightweight token-count calls). New script
      `code/real_tokens.py`. Real tokens run ~9-16% higher than the
      old estimate and the bias is non-uniform (worse on short
      contexts), so every ratio changed slightly: oracle's headline
      compression is `21.9x`→`22x` (was `21.3-21.4x`, inconsistent
      between two tables besides), broadcast `2517` (was `2256`), and
      so on throughout. All of Table 2, Table 3, Figure 2's axis
      positions, and the composite-receiver `230`/`1050`-token claims
      were updated to match.
    - **The "other model" grid's sample size was simply wrong**: the
      paper claimed "69 instances, n=138 receivers" for the
      confirmatory 4-arm grid; the actual committed file
      (`musique_gemini-3.1-flash-lite_lexical_core.jsonl`) has 125
      instances (one cut short by quota, 124 complete, 248 receivers).
      The oracle-vs-broadcast delta and p-value quoted from it
      (`+0.051`, `p=0.039`) were also stale — recomputed from the
      actual file with a new cluster-robust test script
      (`code/cluster_test.py`, block bootstrap + permutation at the
      instance level): the real numbers are `+0.065`, cluster-robust
      `p=0.001`, n=124 — a *stronger*, not weaker, result, but not the
      number that was published. Fixed throughout (Protocol,
      Discussion, cost-quality-frontier paragraph, Table 3's caption).
      This was our own error, not carried over correctly from an
      earlier draft — flagged here rather than minimized.
    - **Table 5 was unreproducible**: no code implementing the paper's
      claimed "four value-curve designs" survived in this repository.
      Rewrote it as three precisely-defined curves (sum / noisy-OR /
      captured-mass) evaluated on both the plain and composite-receiver
      sets (`code/value_curves_check.py`, deterministic recall + real
      `countTokens` spend, cluster-bootstrap CIs) and relabeled the old
      table's fourth "row" for what it actually was — a second
      *condition*, not a fourth curve. The new numbers are not a
      recovery of the old ones (STATUS.md and the table caption both
      say so); qualitatively similar (sum loses, noisy-OR ties) but not
      identical (captured-mass now loses on the plain set, ties only on
      composite).
    - **Figure 2's caption made a false claim**: "only the oracle's CI
      clears [broadcast]" — oracle's CI in fact crosses zero; the one
      CI that does exclude zero (uniform k=5) does so on the *negative*
      side. Rewrote the caption to say so plainly, and fixed a
      matching `21.4×` vs.\ `21.3×` arithmetic inconsistency between
      Tables 2 and 3 (now both `21.9×` with real tokens).
    - Added the missing formal step the review's point 1 asked for: a
      lifted receiver-item ground set `E=[n]×F` stated once, so
      Proposition 3's submodularity claims have an explicit domain
      rather than an implicit one.
    - Fully instantiated Proposition 3's saturation witness (previously
      only its marginal values were given): `F={a,b}`,
      `rel=conf=|S|`, `rho(0)=0, rho(1)=0.5, rho(2)=3` — matching the
      reviewer's own suggested construction, verified in
      `degeneracy_check.py` (now a 2-item witness instead of 3, kept in
      sync with the paper).
    - Qualified Remark 2 (`rho=0` explicit) and Remark 5 ("standard
      value-oracle guarantee" instead of an unqualified "only") per the
      review's minor wording points.
    - "0% memorised" → "0% no-context accuracy" (a correct no-context
      answer is not proof of memorisation); softened "the binding
      constraint is X" to "points to X as the leading bottleneck"
      throughout (abstract, intro, Discussion, Conclusion), and
      softened the influence-maximization framing to an explicit
      analogy rather than a verdict on that literature, per the
      review's points on overclaiming.
    - Verified temperature=0 empirically (3 identical repeated calls)
      and confirmed via the API's own model-listing endpoint that
      `gemini-3.5-flash-lite` still exposes temperature/top-p/top-k as
      live parameters (contradicting the reviewer's speculative
      deprecation concern for this specific model); added a sentence
      documenting this. Verified the AAMAS copyright/conference block
      matches the official template's own anonymous-mode configuration
      exactly (not a bug). Re-checked the review's "missing spaces"
      complaints against the `.tex` source: still `pdftotext`
      extraction artifacts, not real bugs (third round confirming this
      pattern).
    - **Not addressed**: Table 5's per-model breakdown (only one model
      backs the value-curve comparisons — now stated explicitly as a
      threats-to-validity item rather than fixed, since replicating on
      the second model needs a new run); the proxy-vs-real-deployment
      limitation (needs a new experiment, unchanged from rounds 9-10).
    Body still exactly 8 pages after all of this — the largest
    trimming pass yet, needing a genuinely redundant table
    (`tab:scaling`, fully restated in adjacent prose) and one
    illustrative figure removed outright, plus a modest, standard
    caption/float-spacing tightening (`\captionsetup{skip=4pt}` and
    reduced `\floatsep`/`\textfloatsep`/`\intextsep`) in addition to
    prose cuts throughout, before references started cleanly at the
    top of page 9.

## Deadlines — VERIFIED 2026-09-06 against the official call

Source: <https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/> and its
main-track subpage (the earlier 404 was transient). All deadlines are
end-of-day **AoE (UTC−12)**.

| Milestone | Date |
|---|---|
| **OpenReview author registration** | **2026-09-17** ⚠️ |
| Abstract submission | **2026-10-01** |
| Full paper submission | **2026-10-08** |
| Author rebuttal window | 2026-11-20 – 2026-11-24 |
| Notification | 2026-12-21 |
| Camera-ready | 2027-01-25 |
| Conference | 2027-05-03 – 2027-05-07, Hanoi |

⚠️ **The 2026-09-17 OpenReview author-registration step precedes the
abstract deadline and is easy to miss.** The instructions also state
that all authors need OpenReview accounts two weeks before abstract
registration. This is the nearest hard deadline — treat it as the
first action item.

## Submission requirements — VERIFIED

- **Page limit: 8 pages**, plus **any number of additional pages for
  bibliographic references** (references do not count). The call
  explicitly warns: *"Excessive use of typesetting tricks to make
  everything fit into 8 pages is not admissible."*
- **Appendices:** the instructions do **not** separately exempt
  appendix material. Assume it counts toward the 8 pages unless
  clarified. For a theory paper this is the binding constraint — proofs
  must be budgeted, not deferred to an unlimited appendix.
- **LaTeX is mandatory.** The official template is now committed here
  and its exact invocation is recorded under "Template" below.
- **Review is DOUBLE-BLIND.** Consequence for this repo: BATON and
  TEMPO must be cited in the **third person**, never as "our previous
  work". The `PROJECT.md` §9.7 lineage note must be written
  accordingly.
- **Dual submission:** substantially similar work may not be under
  review at another archival venue simultaneously. arXiv preprints and
  non-archival workshops are permitted. Violations mean desk rejection
  at any stage.

## Topic fit — VERIFIED, strong

Three of the eleven listed areas hit directly. Quoted from the call:

- **GAAI** — *"Memory, state, context, long-lived interaction, and
  other architectural patterns for generative and agentic AI systems"*.
  This is close to a bullseye.
- **LEARN** — *"Learning agent-to-agent interactions, including
  learning to communicate and emergent communication."*
- **COINE** — *"Communication, including communication using natural
  language"*; *"Coordination and teamwork."*

A **Blue Sky Ideas track exists** (`/calls/call-for-blue-sky-ideas/`);
its deadline was not retrieved and is still unverified. Worth checking
as a home for the endogenous-topology stretch contribution (T6) if it
does not mature in time for the main track.

## Time budget

Roughly **four weeks** to the abstract deadline as of 2026-09-06. The
theory targets in `PROJECT.md` §5 are prioritized accordingly: T1–T5
are the shippable core, T6 (endogeneity competitive analysis) is a
stretch goal that should be cut without hesitation if it threatens the
deadline. A real multi-agent LLM evaluation is explicitly *not*
promised at this scope.

## Freeze policy

Editable until submitted. **On submission this directory freezes** —
same convention as papers 1–3: no edits while under review, revisions
only when a decision arrives. Update this file at submission time.

## Relationship to the other papers

Self-contained. Shares no code, instances, or results with BATON
(under review, *Computers & OR*), TEMPO (under review, *Transportation
Science*), or the CSoNet/JOCO submission. It borrows a *contrast* with
the vehicle-routing capacity model as motivation, which is a citation
to this group's own published-or-under-review work at most, not shared
artifacts. See `README.md` for the full separation argument.

## Template — received and verified 2026-09-06

The official AAMAS 2027 author template is committed to this directory
(`aamas.cls`, `ACM-Reference-Format.bst`, `template_sample.tex`, plus the
logo and CC-BY artwork the class expects).

Settled by reading it, replacing the earlier "unverified" notes:

- Class invocation for submission: `\documentclass[sigconf,anonymous]{aamas}`
  — the `anonymous` option is what enforces double-blind, and it prints
  the OpenReview submission id on page 1.
- `\acmSubmissionID{<id>}` must carry the OpenReview submission number.
- `\submissionType{Research Paper Track}` — the same template also offers
  AAAI, Demonstration, **Blue Sky Ideas**, JAAMAS and Doctoral Consortium
  tracks, so a Blue Sky variant needs no separate template.
- `balance` package is used to even the columns on the final page.

**8 pages remains the hard limit** and the author has confirmed AAMAS
enforces it strictly. References are excluded from the count; appendix
material is not exempted, so every proof must be budgeted into the 8
pages from the outline stage rather than deferred.
