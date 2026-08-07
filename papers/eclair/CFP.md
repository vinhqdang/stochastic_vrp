# The CFP — verbatim record

Received by the author and recorded here 2026-08-07 (the call is not
yet indexed on Springer's collections page, which as of today lists
only the Thirtieth-Anniversary and Holy-Grail-Progress collections;
this special issue is the journal companion of the LLM-Solve workshop
at CP'26/FLoC'26, https://sites.google.com/view/llm-solve-2026 —
same organizer group).

**Deadline math: October 1, 2026 is 55 days from the recording date.**

---

Call for Papers
Constraints Journal (Springer)
Special Issue on Large Language Models for Combinatorial Constraint Solving (LLM-Solve)
Submission Deadline: October 1, 2026

Guest Editors:
    (Lead) Serdar Kadıoğlu, Brown University & Fidelity Investments, USA
    Tias Guns, KU Leuven, Belgium
    Louis-Martin Rousseau, École Polytechnique de Montréal, Canada
    Stefan Szeider, TU Wien, Austria
    Dimos Tsouros, University of Western Macedonia, Greece

Overview

Combinatorial problem-solving paradigms, covering Constraint
Programming (CP), Boolean Satisfiability (SAT/SMT), and Operations
Research (OR), provide powerful frameworks for finding feasible and
optimal decisions. However, their industrial adoption has long faced a
persistent bottleneck: the steep expertise barrier required to
translate problem descriptions in natural language into formal
constraint models.

Concurrently, Generative AI and Large Language Models (LLMs) have
evolved into sophisticated, multi-agent reasoning engines capable of
code synthesis, retrieval-augmented generation, and step-by-step
planning. While LLMs excel at processing natural language and
generating draft hypotheses, they lack the intrinsic mathematical
guarantees required for formal correctness, leading to logical
inconsistencies or subtle modeling errors when deployed
out-of-the-box.

This Special Issue of the Constraints journal explores the synergy
between generative AI and constraint solvers. We invite high-quality,
original research submissions that leverage LLMs to lower the barrier
to combinatorial problem solving.

Topics of Interest

We welcome original research contributions spanning discrete,
continuous, and hybrid optimization and satisfaction settings. Topics
of interest include, but are not limited to:

1. LLMs for Constraint Modeling & Automated Elicitation
- Automated translation of natural language specifications into formal
  optimization models (CP, SAT, SMT, MILP, etc.).
- Conversational constraint elicitation, intermediary knowledge
  compilation from unstructured problem descriptions.
- Automated constraint acquisition, model synthesis, and semantic
  reformulation using generative AI.

2. Hybrid Architectures & Multi-Model Neuro-Symbolic Systems
- Agentic workflows, self-reflection loops, and retrieval-augmented
  generation specialized for combinatorial structures.
- Solver-in-the-loop reasoning: using CP/SAT solvers as verification
  execution engines to control, correct, or bound LLM outputs.
- Multi-model ecosystems separating generation from judging,
  evaluating, or cross-verifying optimization scripts.

3. Solver Optimization, Configuration, & Tuning
- LLM-guided search heuristics, variable/value branching, node
  ordering, and cut strategy selection.
- Automated hyperparameter tuning, algorithm selection, and solver
  portfolio configuration using text-based performance
  representations.
- Generative configurations for low-level solver mechanics, such as
  cutting-plane selection or preprocessing rules.

4. Benchmarks, Datasets, & Infrastructure
- Open-source repositories and centralized benchmarks for evaluating
  LLMs on constraint tasks.
- Systematic evaluation methodologies focusing on the correctness,
  robustness, and solution quality of LLM-driven optimization
  pipelines.

Submission Guidelines & Review Process

Please refer to the Constraints Journal submission guidelines and
ensure you select the special issue track: "SI: Large Language Models
for Combinatorial Constraint Solving".
https://link.springer.com/journal/10601/submission-guidelines

To ensure high scientific quality, streamline reviewer workload, and
provide authors with prompt feedback, this Special Issue will
implement a two-phase review process:

Phase 1 (Editorial Desk Review): Upon submission, the Guest Editorial
Board will conduct a fast-track initial evaluation to assess scope,
relevance, technical rigor, and publication readiness. Papers that do
not meet the core criteria will receive an immediate decision,
allowing authors to redirect their work without prolonged delays.

Phase 2 (Full Peer Review): Submissions passing Phase 1 will
immediately proceed to an in-depth peer review by domain experts. To
support a timely and collaborative evaluation process, authors of
submitted papers may be invited to contribute as peer reviewers.

Submission Deadline: October 1, 2026

Inquiries: For inquiries regarding scope, track fit, and submissions,
please contact the lead guest editor at serdark@cs.brown.edu

---

## ECLAIR's topic mapping (PROJECT.md §8 relies on this)

- **Topic 2, bullet 2** (solver-in-the-loop reasoning to control,
  correct, or bound LLM outputs) — the core: probes are solver
  executions; the e-process is the bounding layer.
- **Topic 2, bullet 3** (multi-model ecosystems separating generation
  from judging / cross-verifying) — the candidate ensemble + Tier-C
  cross-feasibility probes + e-BH tournament.
- **Topic 4, bullet 2** (systematic evaluation methodologies focusing
  on correctness) — the certification-with-guarantee framing and the
  false-certification-vs-alpha money plot.
- **Topic 1, bullet 2** (conversational elicitation) — the abstention
  ledger as an ambiguity signal, secondary.
- Guest-editor fit: Kadıoğlu (modeling pipelines, NER4Opt), Tsouros &
  Guns (constraint acquisition) — a certification layer complements
  rather than competes; Phase-1 desk review rewards exactly the
  "what guarantee do I get?" answer.
