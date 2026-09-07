# PARCEL — status

- **Working title:** PARCEL: Price-Aware Relay of Context over
  Endogenous Links — budget-constrained information dissemination in
  multi-agent LLM systems when utility is neither monotone nor
  submodular.
- **Venue:** **AAMAS 2027**, the 26th International Conference on
  Autonomous Agents and Multiagent Systems, **Hanoi, Vietnam,
  3–7 May 2027**. Main track.
- **State:** **ACTIVE — drafting.** Manuscript at **5 pages**
  (`main.tex`, AAMAS `sigconf` anonymous mode), compiles clean with
  zero errors against the official class. Theory in `THEORY.md`,
  empirical record in `RESULTS.md`, spec in `PROJECT.md`.

### Blocking before submission

1. ⚠️ **The related-work paragraph on LLM-agent context reduction
   makes claims with NO citations.** The prior-art sweep found the
   relevant work (message-graph pruning, learned topologies, cache
   sharing, single-agent context selection under a token budget) but
   those records are **unverified**, so per the repo rule they cannot
   enter `references.bib`. A reviewer will flag uncited claims about
   prior work. Verify and cite, or delete the claims.
2. `harshaw2019` metadata is not independently checked and it is the
   lineage that owns the density rule — getting it wrong would be an
   attribution error on the one point where we concede priority.
3. Author list, affiliations, `\acmSubmissionID` (blank pending
   OpenReview registration).
4. Page budget: 5 of 8 used. Appendices appear to count, so proofs
   must stay in-body.

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
