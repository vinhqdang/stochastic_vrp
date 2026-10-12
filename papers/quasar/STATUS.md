# QUASAR — status

- **Working title:** QUASAR: quantum amplitude estimation for stochastic
  vehicle routing — coherent feasibility oracles, common random numbers in
  superposition, and nested search for chance-constrained SVRPSPD.
- **Venue:** not fixed. Candidates: INFORMS Journal on Computing; IEEE
  Transactions on Quantum Engineering. Decide at M3 from the strength of the
  results.
- **State:** **ACTIVE — M4 next (cheapest-feasible search with estimated feasibility; lower bounds).** Open for
  edits.
- **Sharing:** uses the shared engine's instances, scenario generators and
  route conventions (`svrpspd_wdro/core`); quantum code is separate in
  `svrpspd_wdro/quantum/`. The research question (quantum estimation and
  search) is disjoint from BATON's (planning gates and online recourse), so
  there is no overlap of claims with papers 1–2.
- **Toolchain:** Qiskit 2.x, qiskit-aer, qiskit-algorithms (Python 3.11).
  GPU only if needed, via Colab.

## Milestones
- [x] M0 scaffold + literature review (2025–2026 included; LITERATURE.md, 139 verified records, 45 from 2026)
- [x] M1 classical references (exact peak DP, discretisation, insertion lemma; svrpspd_wdro/quantum/classical.py, 112 tests)
- [x] M2 Qiskit oracle + resource counts + correctness tests (quantum/oracle.py, DESIGN_M2.md, exact Toffoli formula)
- [x] M3 amplitude estimation experiments, coherent CRN (results/quasar/m3a_v3_summary.md, m3b_v3_summary.md). Findings: CRN gain small for inter-route insertions (rho_q ~0.9 CON/SCA, 0.4-0.7 CMT/city), large for intra-route and alternative-position moves (rho_q 0.13-0.16 CON/SCA, ~0 in-sample CMT/city); end-load lemma; certified query counts quantum/classical 0.07-0.11 (insertions) and 0.42-0.72 (signed moves) at eps 1e-3, 0.006-0.06 at 1e-4, classical cheaper at relative precision; MLAE S>=64 slope -1 vs classical -1/2 on the circuit-validated rotation model; levels b=6, factor-grid error budget ~3e-4.
- [ ] M4 cheapest-feasible search, lower bounds
- [ ] M5 resource estimation, crossover, manuscript

## Freeze policy
Editable until submission; frozen while under review, like the other papers.
