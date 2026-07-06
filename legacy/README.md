# Legacy code (archived)

This directory preserves the repository's earlier research phase — the
**ECHO** project (callback-handling optimizer for last-mile delivery with
re-delivery callbacks) and assorted early baselines — for provenance. It
is **not** part of the current SVRPSPD/OTR-2.0 pipeline, is not covered
by the test suite, and is not maintained.

| Item | What it was |
|---|---|
| `algorithms/` | ECHO + early baselines (GNN-CB, SRO-EV, TH-CB, APEX, ALNS drafts, Gounaris/Cui robust adaptations) |
| `evaluation/`, `scenarios/`, `utils/`, `main.py` | ECHO experiment framework |
| `results_echo/`, `echo_outputs/` | ECHO experiment outputs |
| `PLANv1.md`, `ALGORITHMSv2/3.md`, `BASELINEv2.md`, `alns.md` | Early design notes |
| `RESULTS_ANALYSIS.md`, `RESULTS_TABLE_LATEX.md` | Early result write-ups |

The maintained project lives in `../svrpspd_wdro/` (see its README) with
the manuscript in `../paper/`. Two ideas from this archive were carried
forward into the maintained code as planning gates: the Gounaris-style
robust inflation gate and the Bertsimas–Sim budget gate
(`svrpspd_wdro/scripts/dethloff_runner.py`).
