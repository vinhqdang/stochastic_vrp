# PARCEL — working directory (paper 6, ACTIVE)

Budget-constrained context dissemination in multi-agent LLM systems.
The objective is neither monotone nor submodular, so the influence-
maximization toolchain does not transfer; the paper says precisely why,
proves hardness, and recovers a guarantee in the regime real
deployments operate in.

Target: **AAMAS 2027**, Hanoi, 3–7 May 2027 (main track). Spec:
`PROJECT.md`. Review-state tracker: `STATUS.md`.

## Files

- `PROJECT.md` — the spec: pitch, why classical IM fails (the two
  independent structural failures), the formal model, six theory
  targets with confidence ratings, baselines, experiment plan, open
  questions, and the naming rationale.
- `STATUS.md` — venue, deadlines (**unverified — confirm against the
  official call**), time budget, freeze policy.
- `VERIFY_CITATIONS.md` — the verification log: corrections found
  (including one fabricated statistic that must never be used), logged
  counter-evidence, a scooping risk to assess, what is still
  unverified, and the one claim no citation can support.
- `references.bib` — verified entries only. Populated 2026-09-06;
  several carry inline warning comments about metadata traps.

## Self-containment (read before adding code)

Like `papers/csonet2026/` and `papers/eclair/`, this paper shares **no
code, no instances, and no results** with BATON (under review at
*Computers & OR*), TEMPO (under review at *Transportation Science*), or
the CSoNet/JOCO submission. The shared engine `svrpspd_wdro/` is not
imported and has nothing PARCEL needs.

The vehicle-routing connection is a **contrast used as motivation**,
not shared machinery: in VRP the goods are conserved, so routing
intuitions apply; context is copyable, so they do not. What survives
the translation is *capacity* — the receiving agent's context window is
a rivalrous, saturating resource in the way vehicle capacity is. That
observation motivates the model and is worth one paragraph in the
manuscript. It is not evidence, and it must not become a claim that
PARCEL extends BATON or TEMPO. Two of those papers are under review
elsewhere and artifact sharing would invite simultaneous-submission and
salami-slicing concerns.

Keep all PARCEL code inside this directory.

## State

Planning only — no manuscript, no code, no results yet. The framing was
settled in a planning dialogue on 2026-09-06; `PROJECT.md` §5 lists what
has to be proved and §8 lists what is still unresolved.

The T5 structural question is **resolved** — `PROJECT.md` §9 derives an
*admission price* (send a fact to an agent only when its
relevance-per-token clears that agent's current marginal degradation
rate) and shows the restriction is a dominance property rather than an
assumption. That is the paper's positive result and its defense against
the "assumed away the hard part" critique.

**Next actions, in order:**

1. ⚠️ **Register authors on OpenReview by 2026-09-17** — this precedes
   the abstract deadline and is the nearest hard deadline. See
   `STATUS.md`.
2. **Read Shi & Lai, TCS 990:114409 (2024) in full.** It covers
   non-monotone, non-submodular maximization under a knapsack — 
   structurally this paper's optimization setting. The novelty must
   live in the model, not the abstract theorem. `PROJECT.md` §9.9.
3. **Settle whether degradation is convex or knee-shaped** against the
   measured curves. It decides the shape of the §9 admission price and
   is now the top open modeling question (§9.8).
4. Run the documented search behind the endogenous-topology absence
   claim, or soften it (`VERIFY_CITATIONS.md`).
5. Find and verify the 2026 counter-evidence preprint reporting models
   that resist distractors, and write the heterogeneity into §2 rather
   than waiting for a referee to raise it.
6. Bound the damage from imperfect bundle identification (§9.3).

Verified venue constraints that shape the writing: **8 pages** plus
unlimited references, appendices apparently **counted** (so proofs must
be budgeted), **LaTeX mandatory**, and **double-blind** review — which
means BATON and TEMPO get third-person citation, never "our prior
work".
