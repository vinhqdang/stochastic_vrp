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

## Prior art the gap analysis leans on

- [x] **CP-Agent** — VERIFIED 2026-08-08 via arXiv abs page: Szeider
      (2025), arXiv:2508.07468. In references.bib.
- [x] **ConstraintLLM** — VERIFIED 2026-08-08: Shi, Liu, Zhang, Shi,
      Jia, Ma, Zhang — EMNLP 2025 main conference, arXiv:2510.05774.
      In references.bib.
- [x] **CP-Bench / DCP-Bench-Open** — VERIFIED 2026-08-08:
      Michailidis, Tsouros, Guns, arXiv:2506.06052 (CP-Bench version
      at ECAI 2025). In references.bib. Also verified:
      Michailidis–Tsouros–Guns ICL paper, LIPIcs CP 2024,
      DOI 10.4230/LIPIcs.CP.2024.20; and Szeider's MCP-Solver,
      arXiv:2501.00539. NOTE: "CPEVAL" as named in the adoption
      proposal remains unidentified — the closest verified artifacts
      are CP-Bench (above) and the Springer chapter "Do LLMs
      Understand Constraint Programming? Zero-Shot CP Model
      Generation Using LLMs" (DOI 10.1007/978-3-032-09156-7_2,
      authors not yet retrieved — verify before citing).
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

- [x] Ramdas, Grünwald, Vovk, Shafer — VERIFIED 2026-08-08: *Stat.
      Sci.* 38(4):576–601 (2023), DOI 10.1214/23-STS894.
- [x] Shafer, testing by betting — VERIFIED 2026-08-08: *JRSS-A*
      184(2):407–431 (2021), DOI 10.1111/rssa.12647.
- [x] Grünwald, de Heide, Koolen, safe testing — VERIFIED 2026-08-08:
      *JRSS-B* 86(5):1091–1128 (2024), DOI 10.1093/jrsssb/qkae011.
- [x] Waudby-Smith & Ramdas — VERIFIED 2026-08-08: *JRSS-B*
      86(1):1–27 (2024), DOI 10.1093/jrsssb/qkad009.
- [x] Wang & Ramdas, e-BH — VERIFIED 2026-08-08: *JRSS-B*
      84(3):822–852 (2022), DOI 10.1111/rssb.12489.
- [x] Vovk & Wang — VERIFIED 2026-08-08: *Ann. Statist.* 49(3)
      (2021), DOI 10.1214/20-AOS2020 (page range not re-checked;
      omitted from the bib entry).
- [ ] **In bib with coordinates FROM MEMORY — re-check DOIs before
      submission:** Ville (1939, book); Kelly (1956, BSTJ 35(4),
      DOI 10.1002/j.1538-7305.1956.tb03809.x); Longstaff & Schwartz
      (2001, RFS 14(1):113–147, DOI 10.1093/rfs/14.1.113); Clopper &
      Pearson (1934, Biometrika 26(4):404–413,
      DOI 10.1093/biomet/26.4.404); Guns (2019, ModRef — CPMpy);
      OR-Tools software citation.
- [ ] An exponentiated-gradient / online-learning regret reference if
      the EG routing-regret result enters the final theory section
      (Kivinen & Warmuth 1997 or Cesa-Bianchi & Lugosi book) — not
      currently cited in main.tex.
