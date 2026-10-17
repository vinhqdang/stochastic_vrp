# QUASAR — status

- **Working title:** QUASAR: quantum amplitude estimation for stochastic
  vehicle routing — coherent feasibility oracles, common random numbers in
  superposition, and nested search for chance-constrained SVRPSPD.
- **Venue:** SUBMITTED to Quantum Information Processing (Springer), regular
  article, October 2026 (not JoC: RARECERT is there; IEEE TQE avoided after
  recent desk rejections). Single-blind. Files in `manuscript/` (`quasar.tex`
  in Springer `sn-jnl` format, `cover_letter.pdf`, `appendix_proofs.tex`).
  The cover letter deliberately does not discuss other manuscripts; Springer
  asks for transparency on re-use, and the rare-event numbers of Section 5.4
  also appear in the quantum note of RARECERT — be ready to answer.
- **State:** **UNDER REVIEW at Quantum Information Processing — FROZEN.** Negative results: search inside the optimiser, simultaneous and per-route certification, rare events vs mixture IS (see manuscript Table 7). Items to check if a reviewer asks: the per-route classical lower bound (appendix, new argument not independently checked) and the classical-scan constant (extra term added), STATUS/REFRAME numbers superseded by `FACTS_FOR_PAPER.md` Section 7; Declarations (funding, competing interests, AI use, contributions) were left to the author.
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
- [ ] M4 cheapest-feasible search, lower bounds. Done: THEORY_M4.md revision 2 (FIRST-FEASIBLE, O(sqrt(k*) T) up to logs; classical scan; lower bounds Omega(k a(1-a)/eps^2) classical and Omega(sqrt(k) + sqrt(a)/eps) quantum, product bound a conjecture); cost model quantum/search.py; M4b repair-step measurements on the 73-instance set (results/quasar/m4b_rr_summary.md) and on the region set (148 instances, Northern Vietnam 35.3 M and Guangdong 115.5 M people, data/Region/, results/quasar/m4b_rr_region_summary.md). Findings: on SAA plans the cheapest position is feasible in 72-83% of steps; on Det plans k* grows with scale (q90 30 / 50 / 130 for the 73-set / VNN / GD); core ratio quantum/classical 0.011-0.63, but with the explicit constants of the theorem quantum is 22-500x costlier in every step. Open: a search with smaller constants (expected O(sqrt(k*)) find-first without per-bisection re-testing), then re-measure.
- [ ] M5 resource estimation, crossover, manuscript

## Freeze policy
Editable until submission; frozen while under review, like the other papers.
