# ECLAIR — working directory (paper 5, ACTIVE)

Sequential, anytime-valid, budget-aware certification of LLM-generated
constraint models via e-processes with Kelly-style probe routing.
Target: *Constraints* (Springer), Special Issue on Large Language
Models for Combinatorial Constraint Solving (LLM-Solve), deadline
October 1, 2026. Full spec: `PROJECT.md`. Review-state tracker:
`STATUS.md`. The call itself: `CFP.md`.

## Files

- `PROJECT.md` — the spec: pitch, gap analysis, algorithm (diversified
  generation → 3-tier probe pool → betting → adaptive routing →
  reject/e-BH/abstain), the four theory targets, known technical
  subtleties, experiment plan, and open actions.
- `STATUS.md` — venue and review state (ACTIVE, not submitted).
- `CFP.md` — the call for papers, verbatim (Special Issue on Large
  Language Models for Combinatorial Constraint Solving, **deadline
  October 1, 2026**), plus ECLAIR's bullet-level mapping onto its
  four topics. The call is not yet on Springer's public collections
  page, so this file is the authoritative local record.
- `VERIFY_CITATIONS.md` — every citation this paper intends to lean
  on, ALL currently unverified; nothing moves to `references.bib`
  until checked against live publisher/DOI/arXiv records (repo rule).
- `references.bib` — verified entries only; currently empty.
- `prototype/eclair_prototype.py` — stdlib-only Monte-Carlo
  simulation of the statistical core (no LLM, no CP solver): checks
  that the e-process respects the nominal false-rejection level alpha
  on faithful candidates and that Kelly routing beats round-robin and
  cost-blind baselines on detection and solver-cost-to-detection.
  Both checks PASS — headline numbers in PROJECT.md §7. Regenerate
  `prototype/prototype_results.txt` with
  `python3 prototype/eclair_prototype.py`.

## Self-containment (read before adding code)

Like `papers/csonet2026/`, this paper deliberately shares **no code,
no instances, and no results** with BATON (under review at *Computers
& OR*), TEMPO (under review at *Transportation Science*), or the
CSoNet/JOCO submission. ECLAIR ports TEMPO's e-process *technique*
(mixture bets, predictable adaptive tilts, Ville-threshold decisions —
all citable to the public statistics literature), but the
implementation is written from scratch here against a different data
model, and the shared VRP engine (`svrpspd_wdro/`) is not imported.
Rationale: two of those papers are under review elsewhere, and
artifact sharing would invite simultaneous-submission/salami-slicing
concerns; also `svrpspd_wdro/` simply has nothing ECLAIR needs. Keep
all ECLAIR code inside this directory.
