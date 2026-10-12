# QUASAR (paper 7)

Quantum algorithms for the stochastic vehicle routing problem with
simultaneous pickup and delivery. Specification: `PROJECT.md`; state and
milestones: `STATUS.md`; citations to check: `VERIFY_CITATIONS.md`.

## Files
- `PROJECT.md` — research questions, algorithm design, evaluation plan.
- `STATUS.md` — venue, state, milestones, freeze policy.
- `LITERATURE.md` — literature review (2025–2026 included), built from the
  verified records.
- `references.bib` — verified references only.
- `VERIFY_CITATIONS.md` — records not yet verified.

## Code
`svrpspd_wdro/quantum/` (Qiskit); tests in
`svrpspd_wdro/tests/test_quantum_*.py`. Run from `svrpspd_wdro/`:
`python -m pytest tests/ -q -k quantum`.
