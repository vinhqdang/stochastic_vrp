# CSoNet 2026 / JOCO paper — status

- **Title (revised):** Minimum Weighted Hazard-Exposure Dispatch: A
  Scheduling View with Verified Algorithms and a Wildfire Case Study
  (submitted as *...: Complexity, an Exact Algorithm, and an FPTAS*)
- **Venue:** Journal of Combinatorial Optimization (Springer), via
  CSoNet 2026's Journal Track — **SUBMITTED.**
- **Submission record:**
  - JOCO / Editorial Manager (editorialmanager.com/joco, username
    `dqvinh87@gmail.com`): submitted and **received**. The editorial
    office then sent it back with one pre-review request — *"provide
    the corresponding author email address in the manuscript."* The
    manuscript already carries it (`main.tex` line 31,
    `\author*[1]{...}\email{vinh.dq4@buv.edu.vn}`, rendering on the
    title page as "Corresponding author(s). E-mail(s):
    vinh.dq4@buv.edu.vn"), so the fix is to re-upload the current
    `main.pdf` via *Submissions Sent Back to Author → Edit Submission
    → Attach Files*, rebuild the PDF, and approve. The file uploaded
    originally predated the finalized author block.
  - CSoNet 2026 conference abstract (meteor.springer.com/CSoNet2026):
    submitted, **submission ID 374764**. Listed only Quang-Vinh Dang
    as author; a request to Meteor support to add the three co-authors
    is the author's outstanding action.
- **Review round 1 (2026-10-05): MAJOR REVISION.** Reviewer 1: novelty
  (results follow from 1||sum w_j U_j), relation to Lawler-Moore, and
  why FPTAS = 1.000 in every table. Reviewer 2: clarity of the Theorem 2
  additivity step, self-contained proof of Theorem 5, shorter
  positioning, and parameter uncertainty (refs: Wang 2020; Koca 2023).
  **Revised manuscript and `response_to_reviewers.tex` prepared**
  (2026-10-05; not yet uploaded — upload the flat package in
  `submission/` only: one `main.tex` + figure PNGs, no PDF). The
  revision went through a second pass the same day (an internal
  simulated review, `review_round1/`, then fixes), which changed the
  paper materially — see `review_round1/EDITORIAL_DECISION.md` for the
  review and the list below for what was done:
  - **Positioning:** equal-size greedy for several vehicles is
    *classical* (Lawler 1976; Dessouky–Lageweg–Lenstra–van de Velde
    1990), no longer claimed; classification = elementary corollary;
    new, modest claims: refined exactly tight FPTAS guarantee
    (Thm 7, Prop 11), hardness for a constant hazard-arrival time under
    the arrival reading (Prop 4), round-trip model conservative vs a
    chained route (Prop 1).
  - **Camp Fire rebuilt** on NIST-documented clock times for all four
    communities (the earlier extrapolated times for Magalia and Yankee
    Hill were far off; NIST defines no time zero — we use the initial
    dispatch 06:31); comparison with a chained route, stated protocol,
    correlated-error variant, sample-average plan.
  - **Bibliography** fully verified/corrected (59 entries);
    `verify_extensions.py` checks the added results; `experiment.py`
    gained a HiGHS MIP baseline; code + Lean library are public (tag
    `csonet2026-revision1`).
  - **Open items for the author:** (a) Baptiste (1999) was cited from its
    verified record and a secondary source, not read directly — the
    claim is that one machine with equal processing times, weights and
    release dates is polynomial; (b) Lawler (1976) agreeable-weights
    statement is via Lawler et al., *Elements of Scheduling*; (c) the
    Oroville depot is a modelling choice; (d) nothing has been uploaded
    to Editorial Manager.
- **Policy:** treat as frozen — **do not modify** except for
  editor-requested fixes like the one above, same convention as
  `papers/baton/STATUS.md`. Record any further editorial exchange here.
- Authors (4, real identity now on the title page — no longer
  blinded, per explicit author instruction since JOCO is not a
  double-blind venue):
  1. Quang-Vinh Dang, British University Vietnam — corresponding author
  2. Minh Ngoc Dinh, Millennia Education
  3. Hoang-Viet Vu, British University Vietnam
  4. Phuc-Son Nguyen, UEH University
  All four appear on `main.tex`'s title page, in the "Author
  Contributions" declaration, and across `cover_letter.md`,
  `title_page.md`, and `conference_abstract_submission.md`.
- Original submission set (superseded by the revision above): `main.pdf` (manuscript, 32 pages: 4 theorems, 1
  proposition, full pseudocode, a running numerical example, 5 figures
  (schematic of the running example, 3 plots of the synthetic
  experiments, a real-geography map of the case study), 4 numerical
  experiments on synthetic instances plus a real-world case study
  built from the 2018 Camp Fire (Section 5.5, `case_study_campfire.py`),
  a discussion/extensions section, 2 appendices), `cover_letter.md`,
  `conference_abstract_submission.md`, `title_page.md` (convenience
  copy of title/author/abstract/keywords for pasting into
  submission-system web forms). No separate declarations file, since
  JOCO folds Statements and Declarations into the manuscript itself
  (unlike BATON's Elsevier venue).
- Relationship to the other two papers in this repo: independent.
  Shares no code, instances, or results with BATON or TEMPO — see
  `README.md`'s "What the paper is (and is not)" section for why that
  separation was deliberate (avoiding simultaneous-submission/
  salami-slicing concerns while both other papers are under review
  elsewhere).
