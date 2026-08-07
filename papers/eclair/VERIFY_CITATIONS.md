# ECLAIR — citations to verify

Repo rule: `references.bib` holds only citations verified against live
publisher/DOI/arXiv records. EVERYTHING below is currently unverified
— the prior-art items come from the adoption proposal's literature
scan (not re-retrieved here), and the statistics canon is from memory.
Verify title/authors/venue/year/DOI before first use in the
manuscript; move verified entries to `references.bib` and tick them
off here.

## The CFP itself

- [x] **VERIFIED 2026-08-07** — full text supplied by the author,
      recorded verbatim in `CFP.md`: *Constraints* Special Issue on
      Large Language Models for Combinatorial Constraint Solving
      (LLM-Solve); deadline October 1, 2026; guest editors Kadıoğlu
      (lead), Guns, Rousseau, Szeider, Tsouros; two-phase review.
      Cross-checked against the LLM-Solve workshop at CP'26/FLoC'26
      (https://sites.google.com/view/llm-solve-2026 — same organizer
      group). Note: not yet indexed on Springer's public collections
      page as of 2026-08-07.

## Prior art the gap analysis leans on (all recent, all unverified)

- [ ] **CP-Agent** — agentic constraint modelling with iterative
      refinement and solver feedback.
- [ ] **ConstraintLLM** — neuro-symbolic framework for
      industrial-level CP modelling.
- [ ] **CPEVAL** — zero-shot CP model generation benchmark derived
      from CSPLib.
- [ ] **Falsification framework via provably valid metamorphic
      relations** (LP duality, monotone comparative statics,
      Le Chatelier) — violation as a certificate of unfaithfulness.
      This is the paper whose MR families Tier B imports and whose
      LP-centric validity results Theorem/Prop 4 extends.
- [ ] **Agent-based testing framework** — builds a testing API,
      composes test cases, optimization-oriented mutation testing.
- [ ] **NER4Opt** (Kadıoğlu group) — for the venue-fit paragraph.
- [ ] **Constraint acquisition** line (Tsouros & Guns) — venue fit.
- [ ] Conformal prediction for LLM outputs (pick the canonical
      reference; the criticism used is exchangeability +
      fixed-sample).

## Benchmarks

- [ ] NL4Opt (NeurIPS competition paper).
- [ ] ComplexOR.
- [ ] CSPLib (citable archive reference).

## E-process / anytime-valid statistics canon

- [ ] Ville (1939) — the inequality. (Étude critique de la notion de
      collectif.)
- [ ] Ramdas, Grünwald, Vovk, Shafer — game-theoretic statistics and
      safe anytime-valid inference survey (*Statistical Science*,
      ~2023).
- [ ] Shafer — testing by betting (*JRSS-A*, ~2021).
- [ ] Grünwald, de Heide, Koolen — safe testing / GROW criterion
      (*JRSS-B*, ~2024).
- [ ] Waudby-Smith & Ramdas — betting-based confidence sequences,
      GRAPA/aGRAPA (*JRSS-B*, ~2024).
- [ ] Wang & Ramdas — e-BH: false discovery rate control with
      e-values (*JRSS-B*, ~2022).
- [ ] Vovk & Wang — e-values / combining e-values (*Annals of
      Statistics*, ~2021).
- [ ] Kelly (1956) — the growth-rate criterion, for the routing rule's
      lineage.
- [ ] An exponentiated-gradient / online-learning regret reference for
      the routing bound (Cesa-Bianchi & Lugosi book, or the original
      EG paper — Kivinen & Warmuth 1997).
