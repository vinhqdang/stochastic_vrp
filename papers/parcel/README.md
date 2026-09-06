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
- `VERIFY_CITATIONS.md` — every citation the paper intends to lean on,
  **all currently unverified**, each recorded with the claim it
  supports. Two entries are flagged as load-bearing.
- `references.bib` — verified entries only; **currently empty** by
  design.

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

**Next actions, in order:**

1. Verify the AAMAS 2027 call — page limit, template, blind-review
   model, deadlines (`STATUS.md` has the unverified table).
2. Verify the two load-bearing citations flagged in
   `VERIFY_CITATIONS.md`, starting with the LLM-degradation figures
   that the whole non-monotonicity argument rests on.
3. Settle `PROJECT.md` §8's first open question (does the relevance
   term stay submodular under bounded complementarity), since it
   determines whether theory target T5 is provable at all.
