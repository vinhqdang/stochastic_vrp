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

⚠️ **Repositioned 2026-09-06 after a prior-art sweep — `PROJECT.md`
§10.** Verdict: *partially scooped, reposition, do not abandon.* The
admission price derived in §9 is correct but is **not** the headline:
density-greedy thresholds under a knapsack are textbook, and a paper
from Aug 2026 (BPS, arXiv:2608.19993) already publishes the
single-receiver token-budget version with a guarantee.

What survives, and what the paper now leads with:

1. **The negative result** — that saturation breaks monotonicity and
   complementarity breaks submodularity for inter-agent context,
   invalidating the influence-maximization toolchain. Unclaimed; every
   applied paper either assumes submodularity or avoids theory.
2. **Multi-receiver structure** — a *global* token budget allocated
   across *many* saturating receivers. BPS is single-receiver with a
   modular penalty; the general knapsack theorem (Shi & Lai 2024) does
   not see the partition. This is where the theorem lives.
3. **State-dependent (supermodular) penalty** — the increment that
   breaks the Distorted-Greedy analysis the work would otherwise
   inherit.

**Next actions, in order:**

1. ⚠️ **Register authors on OpenReview by 2026-09-17** — precedes the
   abstract deadline; nearest hard deadline. See `STATUS.md`.
2. **Get Shi & Lai (TCS 990:114409, 2024) full text** via institutional
   access — ScienceDirect 403s and no preprint exists. Check whether
   their weak-supermodular case already absorbs a rising penalty. This
   is the highest-value unresolved item (`PROJECT.md` §10.6).
3. **Read BPS (arXiv:2608.19993) proofs closely** and cite it as the
   special case PARCEL generalizes, pre-empting the obvious referee
   objection (§10.2).
4. **Settle whether degradation is convex or knee-shaped** against the
   measured curves (§9.8).
5. Run the documented search behind the endogenous-topology absence
   claim, or soften it (`VERIFY_CITATIONS.md`).
6. Bound the damage from imperfect bundle identification (§9.3).

Verified venue constraints that shape the writing: **8 pages** plus
unlimited references, appendices apparently **counted** (so proofs must
be budgeted), **LaTeX mandatory**, and **double-blind** review — which
means BATON and TEMPO get third-person citation, never "our prior
work".
