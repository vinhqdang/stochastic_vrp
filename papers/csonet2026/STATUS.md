# CSoNet 2026 / JOCO paper — status

- **Title:** Minimum Weighted Hazard-Exposure Dispatch: Complexity, an
  Exact Algorithm, and an FPTAS
- **Venue:** Journal of Combinatorial Optimization (Springer), via
  CSoNet 2026's Journal Track — **PREPARED, NOT YET SUBMITTED.**
- **Deadline:** Journal Track submission — July 15, 2026 (extended),
  per the CFP text available earlier in this project. **As of today
  (2026-07-16) this date has already passed by one day** — verify the
  live CSoNet 2026 call before submitting: check whether the window
  was extended again, whether JOCO's Editorial Manager enforces this
  date at all for a direct journal submission (journals often accept
  papers year-round; the CSoNet-linked window mainly affects whether
  the abstract submission to meteor.springer.com/CSoNet2026 secures a
  presentation slot), and submit as soon as possible regardless.
- **Policy:** once actually submitted (editorialmanager.com/joco),
  update this file with the submission date and switch this directory
  to the same "do not modify while under review" policy as
  `papers/baton/`; until then it is an active work-in-progress
  directory, editable freely.
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
- Submission set: `main.pdf` (manuscript, 32 pages: 4 theorems, 1
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
