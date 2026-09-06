# PARCEL — working directory (paper 6, ACTIVE)

Budget-constrained context dissemination in multi-agent LLM systems.
The objective is neither monotone nor submodular, so the influence-
maximization toolchain does not transfer; the paper says precisely why,
proves hardness, and recovers a guarantee in the regime real
deployments operate in.

Target: **AAMAS 2027**, Hanoi, 3–7 May 2027 (main track). Spec:
`PROJECT.md`. Review-state tracker: `STATUS.md`.

## Files

- `PROJECT.md` — the spec: pitch, the two structural failures, the
  formal model, six theory targets with confidence ratings, baselines,
  experiment plan, open questions, the prior-art map and positioning
  (§10), and the ρ calibration evidence (§11).
- `STATUS.md` — venue, deadlines (verified against the official call),
  submission requirements, topic fit, time budget, freeze policy.
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

Planning — no manuscript yet. One machine-checked result exists
(`code/`). The framing was settled in a planning dialogue on
2026-09-06; `PROJECT.md` §5 lists what has to be proved and §8 what is
still unresolved.

⚠️ **Repositioned 2026-09-06 after a prior-art sweep — `PROJECT.md`
§10.** Verdict: *partially scooped, reposition, do not abandon.* The
admission price derived in §9 is correct but is **not** the headline:
density-greedy thresholds under a knapsack are textbook, and a paper
from Aug 2026 (BPS, arXiv:2608.19993) already publishes the
single-receiver token-budget version with a guarantee.

**Update 2026-09-06 — the biggest scooping risk is retired.** Shi & Lai
(2024) was obtained and read: their Theorem 4 needs `γ₁`-weak
submodularity *and* `γ₂`-weak supermodularity simultaneously, and
PARCEL's two phenomena kill one parameter each (complementarity kills
`γ₁`, saturation kills `γ₂`). The parameters **do not exist** on this
problem class, so PARCEL is not a special case of it. Machine-checked
in `code/degeneracy_check.py`; details in `PROJECT.md` §10.6.

What survives, and what the paper now leads with:

1. **Two negative results** — (a) saturation breaks monotonicity and
   complementarity breaks submodularity, invalidating the
   influence-maximization toolchain; (b) the nearest general framework's
   structural parameters degenerate on exactly these two phenomena.
   Unclaimed; every applied paper either assumes submodularity or
   avoids theory.
2. **Multi-receiver structure** — a *global* token budget allocated
   across *many* saturating receivers. BPS is single-receiver with a
   modular penalty; the general knapsack theorem (Shi & Lai 2024) does
   not see the partition. This is where the theorem lives.
3. **Cost and damage are different quantities** — you are billed in
   *tokens* (absolute, global, modular) but damaged by *confusability*
   (semantic, per-agent, supermodular), and the measurements say the
   two are not proportional. All prior work optimizes a single
   quantity. This is the increment that breaks the Distorted-Greedy
   analysis the work would otherwise inherit.

**Next actions, in order:**

1. ⚠️ **Register authors on OpenReview by 2026-09-17** — precedes the
   abstract deadline; nearest hard deadline. See `STATUS.md`.
2. ~~Get Shi & Lai full text~~ — **done, §10.6. Risk retired.**
3. **Read BPS (arXiv:2608.19993) proofs closely** and cite it as the
   special case PARCEL generalizes, pre-empting the obvious referee
   objection (§10.2).
4. ~~Settle whether degradation is convex or knee-shaped~~ — **done,
   §11. Two model changes resulted:** ρ is S-shaped (convex only below
   each model's effective length, so the rising-bar result needs an
   explicit regime assumption `c ≤ c*`), and the penalty's argument is
   **confusable load, not token count** — you are billed in tokens but
   damaged by semantic proximity, and the two are not proportional.
5. **Verify the eight citations added by the calibration sweep**
   (`VERIFY_CITATIONS.md`) — none has been checked, peer-review status
   varies from EMNLP/ACL to bare preprint, and one dataset used for
   shape analysis was digitized from a figure.
6. Run the documented search behind the endogenous-topology absence
   claim, or soften it (`VERIFY_CITATIONS.md`).
7. Bound the damage from imperfect bundle identification (§9.3).

## Code

- `code/degeneracy_check.py` — exhaustive enumeration showing that
  Shi & Lai's two structural parameters fail to exist on PARCEL's
  problem class, one killed by each phenomenon. Stdlib only, runs in a
  second: `python3 code/degeneracy_check.py`. This is the machine-checked
  backing for negative result (b) and should become a table in the
  manuscript.
- `refs_local/` — downloaded PDFs of prior work, **gitignored**. This is
  a public repo; publisher PDFs must never be committed.

Verified venue constraints that shape the writing: **8 pages** plus
unlimited references, appendices apparently **counted** (so proofs must
be budgeted), **LaTeX mandatory**, and **double-blind** review — which
means BATON and TEMPO get third-person citation, never "our prior
work".
