# QUASAR M3b v2: estimator behaviour on the validated rotation model

Produced by scripts/run_quasar_m3b.py. Seeds: root SeedSequence(20261012); 2000 independent replicates per setting; one spawned stream per amplitude and estimator.

## Step 1: circuit validation (exact statevector, k = 0..6)

| kind | plan / route | prefix L | b | customers | qubits | Q_level | P(base > Q_level) | a (enumeration) | max abs diff over k |
|---|---|---|---|---|---|---|---|---|---|
| peak | CON8-3 route 0 | 4 | 1 | 13 28 21 23 | 21 | 1 | 0.13 | 0.129723 | 1.89e-15 |
| peak | CON8-0 route 0 | 2 | 2 | 34 38 | 19 | 3 | 0.0974 | 0.097358 | 6.11e-15 |
| peak | SCA8-0 route 0 | 2 | 2 | 1 50 | 19 | 2 | 0.0397 | 0.0396668 | 8.88e-16 |
| difference | CON8-3 route 0 | 2 | 1 | 13 28 21 | 22 | 0 | 0.197 | 0.568817 | 2.66e-15 |

Maximum |p_circuit - p_model| over all oracles and k = 0..6: **6.11e-15**.

Thresholds: the plan Q lies far above the peak of these prefixes (p = 0), so each oracle uses the discrete median of its base route's peak (smallest q with P(F <= q) >= 1/2). The demand marginals are independent gamma (cv 0.3) built from the plan's dbar/pbar, for circuit validation only; the M3a statistics use the copula model.

## Amplitudes

Source: m3a_v2_moves.csv.gz: 30 random moves per class (cls) with 0.001 < p_new < 0.5; 90 moves sampled.
Unique amplitudes simulated: 183; estimation targets (pairs): 90.

| group | targets | median p' | median P(D=1) | median P(D=-1) | median rho_q (file) | median rho_q (recomputed) | share with P(D=-1) > 0 |
|---|---|---|---|---|---|---|---|
| cls1 | 30 | 0.121 | 0.0705 | 0 | 0.841 | 0.841 | 0.00 |
| cls2 | 30 | 0.0453 | 4.75e-05 | 3.5e-05 | 0.074 | 0.074 | 0.57 |
| cls3 | 30 | 0.123 | 4e-05 | 0.000178 | 0.101 | 0.101 | 0.80 |

## Fitted log-log slopes (last six settings per family; expected -1 quantum, -1/2 classical)

Median (min, max) over the targets of each group; stand-alone records (sa) and difference records (dp). The last column is the slope of the 90% quantile of |a_hat - a| for the stand-alone records.

| group | family | sa slope (RMS) | dp slope (RMS) | sa slope (90% abs error) |
|---|---|---|---|---|
| cls1 | qae | -0.912 (-4.560, 0.764) | -1.018 (-1.676, 0.824) | -0.930 (-3.597, 0.412) |
| cls1 | mlae_S1 | 0.018 (-0.052, 0.068) | 0.027 (-0.051, 0.226) | 0.004 (-0.120, 0.107) |
| cls1 | mlae_S8 | -0.228 (-0.409, 0.012) | -0.183 (-0.870, 0.503) | -0.627 (-1.191, 0.069) |
| cls1 | mlae_S64 | -0.963 (-1.130, -0.290) | -0.931 (-1.119, -0.161) | -1.001 (-1.085, -0.076) |
| cls1 | classical | -0.501 (-0.513, -0.494) | -0.501 (-0.512, -0.489) | -0.501 (-0.515, -0.487) |
| cls2 | qae | -1.066 (-1.904, 0.550) | -1.242 (-2.730, -0.471) | -0.949 (-1.857, 0.279) |
| cls2 | mlae_S1 | 0.006 (-0.272, 0.134) | 0.062 (-0.357, 0.488) | -0.002 (-0.405, 0.351) |
| cls2 | mlae_S8 | -0.305 (-1.018, 0.497) | -0.150 (-0.728, 0.231) | -0.828 (-1.587, 0.228) |
| cls2 | mlae_S64 | -0.959 (-1.115, -0.300) | -0.883 (-1.091, -0.062) | -1.002 (-1.105, -0.455) |
| cls2 | classical | -0.500 (-0.515, -0.487) | -0.497 (-0.507, -0.489) | -0.499 (-0.518, -0.481) |
| cls3 | qae | -1.229 (-4.923, 1.005) | -1.451 (-2.097, -0.631) | -1.344 (-3.975, 0.245) |
| cls3 | mlae_S1 | -0.004 (-0.113, 0.108) | 0.056 (-0.313, 0.358) | 0.003 (-0.300, 0.293) |
| cls3 | mlae_S8 | -0.145 (-0.512, 0.241) | -0.124 (-0.620, 0.088) | -0.764 (-1.243, 0.214) |
| cls3 | mlae_S64 | -0.985 (-1.149, -0.324) | -0.531 (-1.258, -0.094) | -1.004 (-1.106, -0.260) |
| cls3 | classical | -0.501 (-0.513, -0.492) | -0.499 (-0.510, -0.494) | -0.501 (-0.516, -0.487) |

## Difference versus stand-alone cost at equal RMS

cost_standalone: cost to reach RMS eps for p'. ratio_single: cost to reach RMS eps for one estimate of P(D=1) divided by cost_standalone (the relevant comparison when D is nonnegative, P(D=-1) = 0). ratio_spec: two estimates, P(D=1) and P(D=-1), each at RMS eps/2, divided by cost_standalone. Predicted single ratio: sqrt(P(D=1)(1-P(D=1))) / sqrt(p'(1-p')) for quantum families and the same squared for classical. Predicted spec ratio: 2 (sqrt v+ + sqrt v-) / sqrt(p'(1-p')) for quantum and 4 (v+ + v-)/(p'(1-p')) for classical, with v = P(1-P). rho_q (file) and rho_q (recomputed) are sqrt(Var D)/sqrt(p'(1-p')) with Var D = P(D=1) + P(D=-1) - (P(D=1)-P(D=-1))^2. Rows where a target is already met at the smallest setting are excluded (at_floor).

Medians over valid rows (at_floor = False). n = valid rows. For the single ratio only rows with P(D=-1) = 0 are used (n_single).

| group | family | target RMS | n | n_single | median ratio_single | median pred_single | ratio_single / pred p10-p90 | median ratio_spec | median pred_spec | ratio_spec / pred p10-p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| cls1 | qae | 0.001 | 30 | 30 | 0.639 | 0.841 | 0.0635 - 7.53 | 1.28 | 1.68 | 0.0582 - 4.26 |
| cls1 | qae | 0.0001 | 30 | 30 | 0.873 | 0.841 | 0.154 - 6.82 | 1.35 | 1.68 | 0.329 - 7.85 |
| cls1 | mlae_S1 | 0.001 | 30 | 30 | nan | 0.841 | - | nan | 1.68 | - |
| cls1 | mlae_S1 | 0.0001 | 30 | 30 | nan | 0.841 | - | nan | 1.68 | - |
| cls1 | mlae_S8 | 0.001 | 30 | 30 | nan | 0.841 | - | nan | 1.68 | - |
| cls1 | mlae_S8 | 0.0001 | 30 | 30 | nan | 0.841 | - | nan | 1.68 | - |
| cls1 | mlae_S64 | 0.001 | 29 | 29 | 0.859 | 0.85 | 0.879 - 1.18 | 1.85 | 1.7 | 0.907 - 1.96 |
| cls1 | mlae_S64 | 0.0001 | 30 | 30 | 0.83 | 0.841 | 0.688 - 1.51 | 1.42 | 1.68 | 0.72 - 1.45 |
| cls1 | classical | 0.001 | 30 | 30 | 0.718 | 0.708 | 0.958 - 1.03 | 2.91 | 2.83 | 0.971 - 1.05 |
| cls1 | classical | 0.0001 | 30 | 30 | 0.734 | 0.708 | 0.949 - 1.04 | 2.57 | 2.83 | 0.955 - 1.03 |
| cls2 | qae | 0.001 | 21 | 7 | 2.08 | 0.0458 | 3.71 - 1.23e+04 | 1.6 | 0.329 | 0.263 - 444 |
| cls2 | qae | 0.0001 | 21 | 7 | 0.239 | 0.0458 | 1.01 - 129 | 0.288 | 0.329 | 0.243 - 55.6 |
| cls2 | mlae_S1 | 0.001 | 21 | 7 | nan | 0.0458 | - | nan | 0.329 | - |
| cls2 | mlae_S1 | 0.0001 | 21 | 7 | nan | 0.0458 | - | nan | 0.329 | - |
| cls2 | mlae_S8 | 0.001 | 13 | 3 | nan | 0.104 | - | nan | 0.647 | - |
| cls2 | mlae_S8 | 0.0001 | 21 | 7 | nan | 0.0458 | - | nan | 0.329 | - |
| cls2 | mlae_S64 | 0.001 | 7 | 1 | 0.202 | 0.264 | 0.767 - 0.767 | 0.681 | 0.671 | 0.718 - 1.63 |
| cls2 | mlae_S64 | 0.0001 | 18 | 5 | 0.0215 | 0.0458 | 0.218 - 0.811 | 0.096 | 0.503 | 0.535 - 1.07 |
| cls2 | classical | 0.001 | 16 | 4 | 0.00684 | 0.00628 | 0.594 - 1.17 | 0.186 | 0.203 | 0.897 - 1.04 |
| cls2 | classical | 0.0001 | 21 | 7 | 0.00215 | 0.0021 | 0.787 - 1.07 | 0.0694 | 0.0683 | 0.925 - 1.03 |
| cls3 | qae | 0.001 | 22 | 2 | 53 | 0.128 | 172 - 1.54e+03 | 0.905 | 0.398 | 0.216 - 449 |
| cls3 | qae | 0.0001 | 22 | 2 | 26.5 | 0.128 | 86.3 - 770 | 0.751 | 0.398 | 0.567 - 22.5 |
| cls3 | mlae_S1 | 0.001 | 22 | 2 | nan | 0.128 | - | nan | 0.398 | - |
| cls3 | mlae_S1 | 0.0001 | 22 | 2 | nan | 0.128 | - | nan | 0.398 | - |
| cls3 | mlae_S8 | 0.001 | 14 | 1 | nan | 0.193 | - | nan | 0.518 | - |
| cls3 | mlae_S8 | 0.0001 | 22 | 2 | nan | 0.128 | - | nan | 0.398 | - |
| cls3 | mlae_S64 | 0.001 | 12 | 1 | 0.161 | 0.193 | 0.832 - 0.832 | 0.565 | 0.572 | 0.861 - 1.18 |
| cls3 | mlae_S64 | 0.0001 | 20 | 1 | nan | 0.193 | - | 0.19 | 0.446 | 0.806 - 2.32 |
| cls3 | classical | 0.001 | 19 | 1 | 0.0368 | 0.0374 | 0.983 - 0.983 | 0.133 | 0.133 | 0.963 - 1.03 |
| cls3 | classical | 0.0001 | 22 | 2 | 0.0199 | 0.0206 | 0.779 - 0.963 | 0.103 | 0.107 | 0.95 - 1.05 |

Full per-target table: m3b_v2_ratio.csv. Sampled moves: m3b_v2_moves_sample.csv.

## Notes on definitions and ranges

- Settings were extended beyond the design grid where the design grid cannot reach the targets. Canonical QAE uses m = 3..22 (design 3..12). Its RMS is dominated by outcomes far from the peak, whose total probability is about sin^2(pi frac(2^m theta/pi)) / N, so the RMS depends on the alignment of 2^m theta/pi and is not monotone in m (it oscillates with m, and is about 0.01 at m = 12 for a = 0.07). Classical samples run to N = 2^26 (design 2^22): at a = 0.07 the target RMS 1e-4 needs N near 6.5e6.
- cost_at uses the suffix maximum of RMS over cost: the cost is the smallest cost from which every larger setting also meets the target. This avoids crediting a lucky dip (the QAE RMS is not monotone).
- The 90% absolute-error column reports the high-probability error that the canonical QAE bound actually controls; it is provided beside the RMS.
- Resolution of the QAE cost ratios: the canonical QAE cost moves in factors of two (cost = 2 (2^m - 1) + 1), so a cost-to-target is only resolved to about a factor of two, and the alignment effect above adds scatter. The QAE ratio/prediction spread columns should be read with this in mind; the classical and MLAE (S = 64) families have fine cost resolution and are the sharp test of the sqrt(variance) law.

## Run

- validation time: 181.1 s
- simulation time: 3699.1 s (workers 3)
- total time: 3885.8 s
- python: 3.11.17
- seeds per setting: 2000
- classical N: 2^4 .. 2^26
- QAE m: 3 .. 22
- MLAE: J = 1 .. 10, S in {1, 8, 64}, grid 20001 points + golden-section refinement

