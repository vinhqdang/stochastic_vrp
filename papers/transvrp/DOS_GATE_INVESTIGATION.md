# B-DOS gate investigation (CODE_REVIEW_B M1)

Date: 2026-11-19. Question: can a correct DOS meet the B-DOS gate of EXPERIMENT_B.md section 7
(stage objective on validation rows <= min(always stop, never stop) + 1e-6 on >= 95% of stages with
>= 100 alive validation rows), at the specified size (`stage_steps` = clip(ceil(20 n / 8192), 50,
400), about 98 steps at n = 40,000)? No threshold or default was changed in the code; tolerance
(`ctx.dos_tol`, `fidelity --dos-tol`) and steps (`DOSConfig.epochs_factor`, `s_min`, `s_max`) are now
parameters. Reproduction: `code/dos_gate_investigation.py` (fit, then analyze).

## 1. Setup (dev and train streams only)

Real routes, N = 10^5 training paths, rep 0, `DOSConfig(lr = 1e-3)`, A's folds; the objective is read
exactly as `fidelity._gate_dos` reads it (validation loss of the kept checkpoint, E0 scale). Routes:
CON3-6 route 1 (m = 12, 22 stages), CON3-7 route 1 (m = 19, 36 stages), SCA3-2 route 3 (m = 11, 20
stages): 78 stages in all. Steps per stage fixed at S in {100, 400, 1000, 3000} (s_min = s_max = S) plus
the spec default (98 at n = 40,000, 84-91 at the fold sizes seen). No test or test_b data was read.

"gap" = objective - min(always, never) (E0 units); "rel gap" = gap / |always - never| (the share of the
span between the two trivial policies that the network fails to capture). A negative gap means the
network beats both trivial policies.

## 2. Reproduction of the review's finding

CON3-6 route 1 at the spec size: 17 of 22 stages (77%) pass 1e-6; at 400 steps also 17 of 22. Confirmed
(table 3, rows "CON3-6"). The failing stages are the near-constant ones (k = 8, 10, 11).

## 3. Results

Pooled pass share over the 78 stages (three routes):

| steps per stage | abs 1e-6 | abs 1e-4 | abs 1e-3 | abs 1e-2 | rel 0.5% | rel 1% | rel 2% | rel 5% | rel 10% |
|---|---|---|---|---|---|---|---|---|---|
| spec (about 98) | 0.705 | 0.821 | 0.846 | 0.910 | 0.846 | 0.872 | 0.885 | 0.910 | 0.962 |
| 100 | 0.833 | 0.872 | 0.910 | 0.936 | 0.910 | 0.910 | 0.923 | 0.936 | 0.987 |
| 400 | 0.859 | 0.923 | 0.949 | 1.000 | 0.974 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1000 | 0.949 | 0.949 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3000 | 0.923 | 0.987 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Per route (share passing abs 1e-6 / rel 1%): CON3-6: spec 0.77/0.86, 400: 0.77/1.00, 1000: 0.91/1.00,
3000: 0.91/1.00; CON3-7: spec 0.67/0.86, 400: 0.89/1.00, 1000: 0.94/1.00, 3000: 0.89/1.00; SCA3-2:
spec 0.70/0.90, 400: 0.90/1.00, 1000: 1.00/1.00, 3000: 1.00/1.00.

Worst gaps (all 78 stages): 98 steps abs 0.055 / rel 0.164; 400 steps abs 1.7e-3 / rel 5.4e-3; 1000
steps abs 3.1e-4 / rel 9.1e-4; 3000 steps abs 2.1e-4 / rel 6.7e-4. The 95% quantile of the abs gap is
2.4e-2 (spec), 1.0e-3 (400), 2.4e-5 (1000), 1.5e-5 (3000): the smallest absolute tolerance that lets a
pooled 95% pass is about 2.4e-2 E0 at the spec size, 1e-3 at 400 steps, 2.4e-5 at 1000, 1.5e-5 at
3000. 1e-6 is not reached at any of the sizes tried (best 0.949, one stage short of 0.95).

The one stage that stays above 1e-6 longest, CON3-6 k = 11 (never-stop is optimal, never = 0.004833,
always = 0.3396), shows the mechanism: gap 5.5e-2 (84 steps), 1.7e-3 (400), 2.9e-4 (1000), 2.5e-5
(3000). It is still falling roughly like steps^-1.5 at 3000.

(b) Wall-clock per fit job (one thread; measured with three jobs and some test runs sharing the
machine, so an upper-ish bound): m = 12: 42 s (100 steps), 131 s (400), 455 s (1000), 1122 s (3000);
m = 19: 68 s, 249 s, 710 s, 1469 s; m = 11: 37 s, 117 s, 396 s, 1027 s. Mean of the three: about 49 s,
166 s, 520 s, 1206 s. For 186 jobs on 12 workers (15.5 jobs each): about 0.2 h, 0.7 h, 2.2 h, 5.2 h.

## 4. Interpretation

(a) Tolerance. Write F-bar for the mean stop probability a network leaves on a stage where continuing
is strictly better. With the sigmoid output the objective is obj = never + F-bar * (mean H/E0 - u) up to
the within-stage correlation, so rel gap = F-bar for a clear-cut stage. An output that reaches 1e-6
absolute needs F-bar of about 3e-6 on every validation row, i.e. a logit below -12.7 everywhere: the
loss floor of a bounded output under finite training, not a bug indicator. A relative tolerance has a
direct meaning (the DOS keeps all but a fraction tau of the benefit of the better trivial policy), and
the bug classes the gate exists for are far away: a DOS whose output is stuck at 0.5 has rel gap 0.5
exactly (loss (always + never) / 2), one whose decision is inverted has rel gap 1.0.

(c) Under-training versus tolerance: both, in that order. At the specified step count the failures are
under-training (gaps of 1e-2 to 5e-2 E0, the network is still warm-start noise: k = 11 at 84 steps has
objective 6 to 12 times the constant's), and no tolerance of 1e-6 would pass them. Beyond about 400 steps
the remaining failures are pure tolerance: gaps of 1e-5 to 1e-3 that keep shrinking but do not reach 1e-6
before 3000 steps (CON3-7 at 3000 steps is not better than at 1000 on its worst stage, 2.1e-4). A
tolerance of 1e-6 therefore cannot be met by a correct DOS at any size affordable here, and the spec size
is also too few steps.

## 5. Recommendation (for the spec; nothing adopted in code)

1. **Tolerance**: replace 1e-6 absolute by a relative tolerance **tau = 0.02 of |always - never|**
   (obj <= min(always, never) + 0.02 |always - never|), per stage, share >= 95% unchanged. At 400 steps
   it passes 78/78 (worst observed 0.0054, margin 3.7x), a stuck or inverted DOS fails at 0.5 and 1.0
   (25x and 50x). (An absolute 1e-2 E0 behaves the same at 400 steps but has no scale argument.)
2. **Steps**: raise the floor to **s_min = 400** (the existing cap), i.e. 400 steps per stage and fold;
   cost about 0.7 h for the 186 jobs (166 s mean per job). Keeping about 98 steps would need
   tau = 0.10 (96% pass) and leaves a DOS that is under-trained by the source's standard (3,000+ steps
   per date; spec 14(c) discloses it); 1000 steps (2.2 h) buys no pass-rate beyond 400 under tau = 0.02.
   Add to 14(c): "400 steps per stage, about 13% of the source's per-date budget".
3. State that the changed tolerance and steps are fixed before the pilot (section 7: a hyper-parameter
   change after a failure is not a repair), and show the rate on three real routes (done here: 78/78 at
   400 steps, tau = 0.02) before sign-off. Per-job time and the fidelity wording ("bug detector only")
   do not change.

Wiring when adopted: `python -m transvrp.expb.fidelity --dos-tol` is an absolute E0-scale slack; the
relative form needs a one-line change in `_gate_dos` (slack = tau * |always - never|); `DOSConfig(s_min=400)`
in `stages.CHOSEN_ARM_CFG`. The lead owns both spec and `stages.py`.
