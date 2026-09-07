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
