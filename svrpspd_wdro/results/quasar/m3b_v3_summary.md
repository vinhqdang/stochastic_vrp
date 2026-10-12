# QUASAR M3b v3: estimator behaviour on the validated rotation model

Produced by scripts/run_quasar_m3b.py. Root seed SeedSequence(20261013); 4000 independent seeds per sampled setting; canonical QAE is computed from its exact outcome pmf.

## 1. Circuit validation (reused from v2, not rerun)

The four real oracles of v2 (three peak, one difference; 19 to 22 qubits) were simulated once with an exact statevector and compared with sin^2((2k+1) theta), k = 0..6. The rows are copied unchanged to m3b_v3_validation.csv.

Maximum |p_circuit - p_model| over the four oracles and k = 0..6: **6.11e-15**.

## 2. Sample and exclusions

Source: m3a_v2_moves.csv.gz, 1369458 moves, 2e5 scenarios each (hits = amplitude x 2e5). Eligible: every nonzero amplitude has at least 200 hits, and the move is not zero-difference.

- Zero-difference moves (P(D=+1) = P(D=-1) = 0): **149236** excluded.
- Moves with a nonzero amplitude below 200 hits (stand-alone p' or a difference side): **529674** excluded.
- Eligible moves: 690548.
- Amplitude values below the threshold (1 to 199 hits): p' 28995, P(D=+1) 354488, P(D=-1) 335112; zero hits (one-sided sides, not amplitudes to estimate): P(D=+1) 232903, P(D=-1) 805815.

Stratification: move class cls (1 = insertion, 2 = intra-route, 3 = alternative position) by stand-alone p' band (low: p' < 0.02; decision: 0.02 <= p' <= 0.2; high: p' > 0.2). Up to 30 moves per cell, drawn at random from the eligible pool.

| cell | eligible pool | sampled |
|---|---|---|
| cls1 / low | 57564 | 30 |
| cls1 / decision | 219134 | 30 |
| cls1 / high | 272123 | 30 |
| cls2 / low | 651 | 30 |
| cls2 / decision | 6309 | 30 |
| cls2 / high | 13901 | 30 |
| cls3 / low | 2106 | 30 |
| cls3 / decision | 26704 | 30 |
| cls3 / high | 92056 | 30 |

Sampled moves: 270 (186 one-sided, 84 signed). Unique amplitudes simulated: 588.

## 3. Error metrics and fitted slopes

Primary metric: the 90 % quantile of |a_hat - a|. Slopes are log-log fits over the last six settings of each family in cost order (QAE m = 17..22; MLAE J = 5..10 for each S; classical N = 2^21..2^26). Medians and [min, max] over the amplitude records of each kind.

| kind | family | slope, 90 % quantile | slope, mean abs error | slope, RMS (secondary) |
|---|---|---|---|---|
| standalone | qae | -0.957 [-3.603, 0.422] (n=267) | -0.889 [-3.763, 0.572] (n=267) | -0.471 [-2.538, 0.499] (n=267) |
| standalone | mlae_S32 | -0.987 [-1.441, -0.069] (n=267) | -0.941 [-1.141, -0.250] (n=267) | -0.827 [-1.321, -0.038] (n=267) |
| standalone | mlae_S64 | -1.001 [-1.413, -0.202] (n=267) | -0.986 [-1.206, -0.485] (n=267) | -0.966 [-1.170, -0.101] (n=267) |
| standalone | mlae_S128 | -1.005 [-1.307, -0.382] (n=267) | -1.002 [-1.183, -0.649] (n=267) | -0.996 [-1.316, -0.364] (n=267) |
| standalone | classical | -0.499 [-0.512, -0.484] (n=267) | -0.500 [-0.514, -0.488] (n=267) | -0.500 [-0.511, -0.489] (n=267) |
| difference+ | qae | -0.931 [-2.839, 0.463] (n=211) | -0.922 [-3.006, 0.652] (n=211) | -0.488 [-1.812, 0.498] (n=211) |
| difference+ | mlae_S32 | -0.937 [-1.534, 0.126] (n=211) | -0.713 [-1.119, -0.168] (n=211) | -0.303 [-1.285, 0.005] (n=211) |
| difference+ | mlae_S64 | -0.991 [-1.420, 0.049] (n=211) | -0.916 [-1.160, -0.283] (n=211) | -0.690 [-1.219, -0.044] (n=211) |
| difference+ | mlae_S128 | -1.003 [-1.394, -0.148] (n=211) | -0.978 [-1.170, -0.481] (n=211) | -0.916 [-1.262, -0.152] (n=211) |
| difference+ | classical | -0.500 [-0.514, -0.489] (n=211) | -0.500 [-0.509, -0.491] (n=211) | -0.500 [-0.510, -0.491] (n=211) |
| difference- | qae | -1.080 [-2.386, 0.295] (n=110) | -1.011 [-2.383, 0.681] (n=110) | -0.521 [-1.353, 0.500] (n=110) |
| difference- | mlae_S32 | -0.883 [-1.441, 0.131] (n=110) | -0.552 [-1.018, -0.211] (n=110) | -0.195 [-0.999, 0.022] (n=110) |
| difference- | mlae_S64 | -0.967 [-1.419, -0.097] (n=110) | -0.773 [-1.319, -0.374] (n=110) | -0.460 [-1.437, -0.045] (n=110) |
| difference- | mlae_S128 | -0.999 [-1.261, -0.261] (n=110) | -0.944 [-1.145, -0.520] (n=110) | -0.760 [-1.315, -0.227] (n=110) |
| difference- | classical | -0.500 [-0.513, -0.482] (n=110) | -0.500 [-0.508, -0.486] (n=110) | -0.500 [-0.510, -0.487] (n=110) |

**Why the QAE 90 % quantile slope is read as a median over records.** For large N the exact QAE error is |a_hat - a| = 2 pi sqrt(a(1-a)) |delta| / N (1 + o(1)), where delta is the offset of the outcome from N phi, and the outcome law of delta depends only on f_m = frac(2^m theta / pi): q90 = 2 pi sqrt(a(1-a)) g(f_m) / N with g bounded. The exact values agree with this: for a = 0.2, q90 N is about proportional to f_m for small f_m (0.013 at f = 0.005, 0.40 at f = 0.16) and reaches about 4 at f = 0.6. At fixed a the factor g(f_m) therefore moves with m, so the slope of one amplitude over six values of m is alignment noise. The population slope is -1 (the median over records is the statistic to read; the per-record range is shown only to make that noise visible).

**Why the RMS is a secondary column.** For canonical QAE the tails are P(|delta| > x) of order 1/(pi^2 x), so P(|a_hat - a| > e) is of order 1/(N e) for 1/N << e << 1. The mean absolute error then behaves as (log N)/N (up to the alignment factor), with slope -1 in cost. The mean square, E(a_hat - a)^2 = integral of 2 e P(|a_hat - a| > e) de, is of order 1/N: the integrand 2 e P(|a_hat - a| > e) is about 2/(pi^2 N) for every e in (1/N, O(1)), so the mass is spread uniformly over error levels up to O(1) and is carried by outcomes of probability about 1/N and error of order a. It therefore scales as N^(-1/2) and is dominated by these heavy tails (for the QAE rows of m3b_v3_metrics.csv, sqrt(N) times the exact RMS is checkable directly and oscillates with m). A finite sample of 2000 or 4000 seeds almost never contains them, so a sampled RMS of QAE under-estimates it and its slope is noise; the exact RMS is reported as a secondary column for that reason.

## 4. Exact QAE metrics versus a large sampled estimate

Exact mean absolute error and q90 from the pmf, against 200000 sampled QAE runs (z = (sampled - exact) / standard error):

| a | m | exact mean abs | sampled mean abs | z | exact q90 | sampled q90 | exact RMS | sampled RMS |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 6 | 0.01772 | 0.01735 | -1.97 | 0.01653 | 0.01653 | 0.08691 | 0.0853 |
| 0.005 | 10 | 8.453e-05 | 8.831e-05 | 0.33 | 2.911e-05 | 2.911e-05 | 0.00464 | 0.005071 |
| 0.03 | 6 | 0.02424 | 0.02418 | -0.29 | 0.02904 | 0.02904 | 0.08745 | 0.08758 |
| 0.03 | 10 | 0.001188 | 0.00115 | -1.15 | 0.00133 | 0.00133 | 0.016 | 0.01476 |
| 0.2 | 6 | 0.04064 | 0.0405 | -0.83 | 0.0643 | 0.0643 | 0.08709 | 0.08636 |
| 0.2 | 10 | 0.000781 | 0.000764 | -1.00 | 0.0003082 | 0.0003082 | 0.008496 | 0.007666 |

## 5. Cost of the difference estimate relative to the stand-alone estimate

Targets on the 90 % quantile of |a_hat - a|: absolute 1e-3 and 1e-4 (T_sa = T_D = eps); relative 0.1 and 0.01 (T_sa = r p', T_D = r (a+ + a-), with a+ + a- = P(D != 0)). Cost ratio = cost(difference) / cost(stand-alone).

- One-sided moves use one estimate of the nonzero side at target T_D: ratio = cost_D / cost_sa, prediction sqrt(v_D) / T_D over sqrt(v_sa) / T_sa for quantum estimators (v = a(1-a)) and v_D / T_D^2 over v_sa / T_sa^2 for classical.
- Signed moves use two estimates of P(D = +1) and P(D = -1) at target T_D / 2 each: ratio = (cost_+ + cost_-) / cost_sa, prediction 2 (sqrt v+ + sqrt v-) / T_D over sqrt(v_sa) / T_sa (quantum); 4 (v+ + v-) / T_D^2 over v_sa / T_sa^2 (classical).
- Status: ok (cost measured), floor (a target is met at the smallest setting, so the cost is an upper bound and the row is not used), fail (target not reached within the grid). Medians and the ratio/prediction spread use the ok rows only, and the same rows for all three columns.
- QAE cost-to-target inherits the alignment factor g(f_m) of Section 3: the cost is the smallest m from which every larger m meets the target (suffix maximum over the error, cost resolved in factors of 2), so QAE ratios carry a multiplicative scatter. Read the QAE columns with that in mind.
- The claim 'ratio tracks rho_q' applies to class 1 (insertion) with absolute targets only: there the prediction for one-sided moves is exactly rho_q = sqrt(Var D) / sqrt(p'(1-p')). It is not claimed for classes 2 and 3 or for relative targets, whose predictions differ.

### Target: absolute 0.001

Each cell: median ratio / median prediction; median (ratio / prediction) [p10, p90]; ok / floor / fail counts.

| cell | qae | mlae_S32 | mlae_S64 | mlae_S128 | classical |
|---|---|---|---|---|---|
| cls1 / low | 0.872 / 0.88; 1.02 [0.279, 4.42]; 30 / 0 / 0 | 0.768 / 0.88; 0.951 [0.437, 1.99]; 30 / 0 / 0 | 0.838 / 0.88; 0.94 [0.47, 1.11]; 30 / 0 / 0 | 0.858 / 0.88; 0.977 [0.753, 1.02]; 30 / 0 / 0 | 0.796 / 0.774; 1 [0.909, 1.03]; 30 / 0 / 0 |
| cls1 / decision | 0.685 / 0.795; 1.24 [0.376, 3.22]; 30 / 0 / 0 | 0.745 / 0.795; 0.949 [0.67, 1.34]; 30 / 0 / 0 | 0.802 / 0.795; 0.993 [0.764, 1.21]; 30 / 0 / 0 | 0.799 / 0.795; 0.997 [0.771, 1.12]; 30 / 0 / 0 | 0.626 / 0.632; 1.01 [0.974, 1.06]; 30 / 0 / 0 |
| cls1 / high | 1.03 / 0.965; 1.08 [0.557, 3.18]; 30 / 0 / 0 | 0.959 / 0.965; 1.04 [0.901, 1.25]; 30 / 0 / 0 | 0.85 / 0.965; 1.01 [0.838, 1.21]; 30 / 0 / 0 | 0.908 / 0.965; 0.991 [0.825, 1.13]; 30 / 0 / 0 | 0.932 / 0.931; 1.01 [0.95, 1.04]; 30 / 0 / 0 |
| cls2 / low | 0.513 / 0.47; 0.921 [0.6, 2.64]; 30 / 0 / 0 | 0.358 / 0.47; 0.825 [0.449, 1.08]; 30 / 0 / 0 | 0.361 / 0.47; 0.874 [0.586, 0.974]; 30 / 0 / 0 | 0.354 / 0.47; 0.8 [0.675, 0.94]; 30 / 0 / 0 | 0.224 / 0.221; 1.01 [0.877, 1.07]; 30 / 0 / 0 |
| cls2 / decision | 0.479 / 0.584; 1.4 [0.324, 4.17]; 30 / 0 / 0 | 0.573 / 0.584; 0.965 [0.732, 1.34]; 30 / 0 / 0 | 0.557 / 0.584; 0.947 [0.796, 1.1]; 30 / 0 / 0 | 0.553 / 0.584; 0.946 [0.765, 1.03]; 30 / 0 / 0 | 0.19 / 0.205; 0.977 [0.921, 1.03]; 30 / 0 / 0 |
| cls2 / high | 0.778 / 0.61; 1.28 [0.492, 10.5]; 30 / 0 / 0 | 0.708 / 0.589; 1.1 [0.888, 4.23]; 28 / 0 / 2 | 0.572 / 0.61; 0.969 [0.808, 1.43]; 30 / 0 / 0 | 0.57 / 0.61; 0.97 [0.765, 1.1]; 30 / 0 / 0 | 0.204 / 0.21; 1 [0.949, 1.03]; 30 / 0 / 0 |
| cls3 / low | 0.382 / 0.371; 0.951 [0.416, 1.97]; 30 / 0 / 0 | 0.332 / 0.37; 0.9 [0.393, 1.03]; 29 / 0 / 1 | 0.332 / 0.37; 0.872 [0.626, 0.995]; 29 / 0 / 1 | 0.338 / 0.371; 0.891 [0.74, 0.955]; 30 / 0 / 0 | 0.136 / 0.137; 0.985 [0.869, 1.07]; 30 / 0 / 0 |
| cls3 / decision | 0.525 / 0.496; 1.08 [0.363, 7.16]; 30 / 0 / 0 | 0.532 / 0.496; 0.976 [0.838, 1.53]; 30 / 0 / 0 | 0.476 / 0.496; 0.939 [0.776, 1.08]; 30 / 0 / 0 | 0.452 / 0.496; 0.929 [0.705, 1.04]; 30 / 0 / 0 | 0.155 / 0.153; 1.01 [0.915, 1.05]; 30 / 0 / 0 |
| cls3 / high | 0.868 / 0.745; 1.01 [0.692, 5.42]; 30 / 0 / 0 | 0.994 / 0.745; 1.08 [0.82, 4.54]; 30 / 0 / 0 | 0.708 / 0.745; 0.976 [0.678, 1.18]; 30 / 0 / 0 | 0.733 / 0.745; 0.978 [0.904, 1.06]; 30 / 0 / 0 | 0.346 / 0.338; 0.994 [0.975, 1.04]; 30 / 0 / 0 |

### Target: absolute 0.0001

Each cell: median ratio / median prediction; median (ratio / prediction) [p10, p90]; ok / floor / fail counts.

| cell | qae | mlae_S32 | mlae_S64 | mlae_S128 | classical |
|---|---|---|---|---|---|
| cls1 / low | 0.953 / 0.88; 1.04 [0.216, 2.42]; 30 / 0 / 0 | 0.901 / 0.88; 0.981 [0.342, 2.14]; 20 / 0 / 10 | 0.883 / 0.894; 0.999 [0.383, 1.5]; 28 / 0 / 2 | 0.848 / 0.88; 0.988 [0.68, 1.2]; 30 / 0 / 0 | 0.79 / 0.774; 1 [0.954, 1.04]; 30 / 0 / 0 |
| cls1 / decision | 0.624 / 0.795; 1.01 [0.285, 4.25]; 30 / 0 / 0 | 0.913 / 0.782; 1.06 [0.922, 1.35]; 21 / 0 / 9 | 0.83 / 0.8; 1.01 [0.841, 1.12]; 29 / 0 / 1 | 0.789 / 0.79; 0.998 [0.823, 1.07]; 29 / 0 / 1 | 0.632 / 0.632; 1 [0.965, 1.04]; 30 / 0 / 0 |
| cls1 / high | 0.853 / 0.965; 0.997 [0.307, 3.62]; 30 / 0 / 0 | nan / nan; nan [nan, nan]; 0 / 0 / 30 | 0.958 / 0.982; 0.985 [0.933, 1.13]; 21 / 0 / 9 | 0.879 / 0.967; 0.986 [0.84, 1.16]; 29 / 0 / 1 | 0.924 / 0.931; 1.01 [0.975, 1.04]; 26 / 0 / 4 |
| cls2 / low | 0.54 / 0.47; 1.04 [0.465, 2.83]; 30 / 0 / 0 | 0.316 / 0.47; 0.703 [0.383, 1.53]; 17 / 0 / 13 | 0.487 / 0.47; 0.937 [0.65, 2.56]; 28 / 0 / 2 | 0.444 / 0.47; 0.985 [0.807, 1.13]; 30 / 0 / 0 | 0.224 / 0.221; 1 [0.966, 1.04]; 30 / 0 / 0 |
| cls2 / decision | 0.568 / 0.584; 1.06 [0.506, 3.47]; 30 / 0 / 0 | 0.574 / 0.34; 1.06 [0.83, 3.53]; 12 / 0 / 18 | 0.824 / 0.582; 1.06 [0.964, 1.9]; 25 / 0 / 5 | 0.583 / 0.582; 1.06 [0.976, 1.37]; 29 / 0 / 1 | 0.216 / 0.205; 1.01 [0.98, 1.04]; 30 / 0 / 0 |
| cls2 / high | 0.523 / 0.61; 1.11 [0.496, 4.08]; 30 / 0 / 0 | nan / nan; nan [nan, nan]; 0 / 0 / 30 | 0.584 / 0.566; 1.03 [0.925, 1.35]; 20 / 0 / 10 | 0.615 / 0.61; 1.01 [0.769, 1.21]; 28 / 0 / 2 | 0.203 / 0.203; 0.989 [0.974, 1.03]; 26 / 0 / 4 |
| cls3 / low | 0.58 / 0.371; 1.06 [0.34, 3.92]; 30 / 0 / 0 | 0.396 / 0.366; 0.956 [0.594, 3.7]; 17 / 0 / 13 | 0.447 / 0.37; 0.997 [0.724, 3.08]; 28 / 0 / 2 | 0.416 / 0.371; 1.01 [0.774, 1.84]; 30 / 0 / 0 | 0.139 / 0.137; 1 [0.955, 1.03]; 30 / 0 / 0 |
| cls3 / decision | 0.472 / 0.496; 0.977 [0.271, 2.86]; 30 / 0 / 0 | 0.597 / 0.312; 1.04 [0.968, 3.05]; 14 / 0 / 16 | 0.518 / 0.474; 1.05 [0.891, 1.77]; 27 / 0 / 3 | 0.484 / 0.496; 1.02 [0.925, 1.74]; 30 / 0 / 0 | 0.149 / 0.153; 0.992 [0.965, 1.02]; 30 / 0 / 0 |
| cls3 / high | 0.756 / 0.745; 1.2 [0.294, 4.25]; 30 / 0 / 0 | 1.39 / 1.11; 1.25 [1.25, 1.25]; 1 / 0 / 29 | 0.737 / 0.711; 1.05 [0.953, 1.53]; 19 / 0 / 11 | 0.754 / 0.745; 1.02 [0.92, 1.25]; 30 / 0 / 0 | 0.368 / 0.362; 1 [0.971, 1.04]; 27 / 0 / 3 |

### Target: relative 0.1 (x amplitude)

Each cell: median ratio / median prediction; median (ratio / prediction) [p10, p90]; ok / floor / fail counts.

| cell | qae | mlae_S32 | mlae_S64 | mlae_S128 | classical |
|---|---|---|---|---|---|
| cls1 / low | 1.41 / 1.14; 1.19 [0.283, 5.25]; 30 / 0 / 0 | 1.07 / 1.14; 0.991 [0.45, 3.72]; 28 / 0 / 2 | 1.06 / 1.13; 1 [0.522, 1.14]; 29 / 0 / 1 | 1.1 / 1.14; 1.01 [0.761, 1.29]; 30 / 0 / 0 | 1.35 / 1.3; 1.02 [0.963, 1.05]; 30 / 0 / 0 |
| cls1 / decision | 1.58 / 1.33; 0.921 [0.2, 3.85]; 28 / 2 / 0 | 1.2 / 1.3; 1.02 [0.807, 1.34]; 30 / 0 / 0 | 1.48 / 1.3; 1.01 [0.846, 1.26]; 30 / 0 / 0 | 1.48 / 1.3; 1.03 [0.754, 1.51]; 30 / 0 / 0 | 1.68 / 1.69; 0.991 [0.939, 1.03]; 30 / 0 / 0 |
| cls1 / high | 1.63 / 1.48; 0.962 [0.588, 2.07]; 26 / 4 / 0 | 1.3 / 1.56; 0.977 [0.718, 1.35]; 17 / 13 / 0 | 1.25 / 1.05; 1.01 [0.761, 1.18]; 5 / 25 / 0 | 1.05 / 1.01; 1.03 [1.03, 1.03]; 1 / 29 / 0 | 2.97 / 2.92; 0.999 [0.931, 1.03]; 30 / 0 / 0 |
| cls2 / low | 2.95 / 2.14; 1.21 [0.592, 4.24]; 30 / 0 / 0 | 2.1 / 2.14; 0.985 [0.568, 2.74]; 24 / 0 / 6 | 2.3 / 2.14; 1.01 [0.824, 1.85]; 30 / 0 / 0 | 2.17 / 2.14; 1.03 [0.812, 1.34]; 30 / 0 / 0 | 4.65 / 4.58; 0.995 [0.955, 1.02]; 30 / 0 / 0 |
| cls2 / decision | 8.7 / 8.6; 0.979 [0.421, 1.94]; 30 / 0 / 0 | 8.86 / 8.6; 1.06 [0.861, 2.16]; 28 / 0 / 2 | 10.2 / 8.65; 1.12 [0.903, 1.36]; 27 / 1 / 2 | 11.2 / 8.65; 1.21 [0.887, 1.49]; 29 / 1 / 0 | 41.7 / 42.3; 1.01 [0.967, 1.04]; 30 / 0 / 0 |
| cls2 / high | 15.6 / 18.8; 1.06 [0.413, 1.79]; 30 / 0 / 0 | 23.9 / 19.4; 1.31 [0.848, 4.83]; 14 / 15 / 1 | 12.1 / 12.7; 0.986 [0.745, 2.17]; 7 / 23 / 0 | 7.27 / 10.1; 0.717 [0.717, 0.717]; 1 / 29 / 0 | 242 / 242; 0.997 [0.959, 1.06]; 30 / 0 / 0 |
| cls3 / low | 2.93 / 2.75; 0.965 [0.113, 3.53]; 30 / 0 / 0 | 2.92 / 2.78; 1.05 [0.551, 6.99]; 24 / 0 / 6 | 2.91 / 2.74; 1.02 [0.877, 3.94]; 29 / 0 / 1 | 3.13 / 2.75; 1.05 [0.908, 2.19]; 30 / 0 / 0 | 7.5 / 7.54; 0.994 [0.956, 1.04]; 30 / 0 / 0 |
| cls3 / decision | 8.77 / 8.66; 0.987 [0.399, 2.72]; 30 / 0 / 0 | 8.72 / 8.11; 1.12 [0.834, 2.12]; 27 / 0 / 3 | 10.3 / 8.66; 1.08 [0.857, 1.27]; 30 / 0 / 0 | 11.2 / 8.66; 1.16 [0.846, 1.41]; 30 / 0 / 0 | 71.4 / 75.2; 0.999 [0.952, 1.04]; 30 / 0 / 0 |
| cls3 / high | 19.5 / 22; 0.84 [0.539, 3.52]; 27 / 3 / 0 | 20.6 / 15.2; 1.08 [0.719, 2.8]; 17 / 11 / 2 | 12.3 / 12.6; 0.875 [0.734, 1.17]; 10 / 20 / 0 | 9.9 / 12.6; 0.718 [0.694, 1.08]; 6 / 24 / 0 | 284 / 294; 1.01 [0.95, 1.09]; 29 / 1 / 0 |

### Target: relative 0.01 (x amplitude)

Each cell: median ratio / median prediction; median (ratio / prediction) [p10, p90]; ok / floor / fail counts.

| cell | qae | mlae_S32 | mlae_S64 | mlae_S128 | classical |
|---|---|---|---|---|---|
| cls1 / low | 1.12 / 1.14; 0.983 [0.142, 2.39]; 30 / 0 / 0 | 1.18 / 1.12; 1.11 [0.45, 1.72]; 17 / 0 / 13 | 1.26 / 1.12; 1.05 [0.513, 1.51]; 27 / 0 / 3 | 1.17 / 1.14; 0.999 [0.693, 1.15]; 30 / 0 / 0 | 1.3 / 1.3; 0.999 [0.974, 1.04]; 30 / 0 / 0 |
| cls1 / decision | 2.04 / 1.3; 1.03 [0.358, 3.53]; 30 / 0 / 0 | 1.31 / 1.28; 0.975 [0.717, 1.4]; 27 / 0 / 3 | 1.37 / 1.28; 1.01 [0.819, 1.29]; 29 / 0 / 1 | 1.36 / 1.3; 1.01 [0.949, 1.32]; 30 / 0 / 0 | 1.71 / 1.69; 1.01 [0.971, 1.05]; 30 / 0 / 0 |
| cls1 / high | 2.06 / 1.71; 1.03 [0.263, 3.17]; 30 / 0 / 0 | 1.53 / 1.71; 0.97 [0.821, 1.29]; 30 / 0 / 0 | 1.64 / 1.71; 1.02 [0.778, 1.18]; 30 / 0 / 0 | 1.62 / 1.71; 1.01 [0.839, 1.2]; 30 / 0 / 0 | 2.95 / 2.92; 1.01 [0.965, 1.05]; 30 / 0 / 0 |
| cls2 / low | 2.2 / 2.14; 0.998 [0.357, 3.85]; 30 / 0 / 0 | 2.2 / 2.2; 0.964 [0.457, 1.69]; 12 / 0 / 18 | 2.03 / 2.14; 0.944 [0.839, 1.3]; 26 / 0 / 4 | 2.13 / 2.14; 0.982 [0.871, 1.27]; 30 / 0 / 0 | 4.57 / 4.58; 0.998 [0.97, 1.03]; 30 / 0 / 0 |
| cls2 / decision | 9.99 / 8.6; 1.24 [0.483, 3.45]; 30 / 0 / 0 | 8.88 / 6.7; 1.13 [0.938, 2.17]; 13 / 0 / 17 | 7.67 / 8.44; 1.04 [0.876, 1.63]; 24 / 0 / 6 | 9.74 / 8.65; 1.04 [0.843, 1.38]; 29 / 0 / 1 | 41.9 / 42.3; 1 [0.947, 1.04]; 30 / 0 / 0 |
| cls2 / high | 27 / 18.8; 1.39 [0.508, 8.01]; 30 / 0 / 0 | 19.7 / 18.4; 1.16 [0.988, 1.93]; 18 / 0 / 12 | 21 / 18.5; 1.09 [0.963, 1.48]; 27 / 0 / 3 | 19.8 / 18.4; 1.05 [0.997, 1.39]; 28 / 0 / 2 | 238 / 242; 0.996 [0.958, 1.03]; 30 / 0 / 0 |
| cls3 / low | 2.22 / 2.75; 0.759 [0.259, 4.35]; 30 / 0 / 0 | 2.25 / 2.76; 0.935 [0.742, 1.08]; 4 / 0 / 26 | 2.93 / 2.79; 1.14 [0.789, 2]; 18 / 0 / 12 | 3.17 / 2.74; 1.05 [0.991, 2.11]; 25 / 0 / 5 | 7.44 / 7.54; 1.01 [0.967, 1.04]; 30 / 0 / 0 |
| cls3 / decision | 9.94 / 8.66; 1.16 [0.324, 3.59]; 30 / 0 / 0 | 10.3 / 7.57; 1.25 [0.961, 2.17]; 14 / 0 / 16 | 11.1 / 8.09; 1.07 [0.771, 1.63]; 24 / 0 / 6 | 9.37 / 8.22; 1.02 [0.775, 1.31]; 28 / 0 / 2 | 74.3 / 75.2; 0.992 [0.947, 1.04]; 30 / 0 / 0 |
| cls3 / high | 16.9 / 22.3; 0.781 [0.423, 2.37]; 30 / 0 / 0 | 19.1 / 15.1; 1.13 [0.92, 1.47]; 18 / 0 / 12 | 19 / 21.6; 1.06 [0.851, 1.21]; 25 / 0 / 5 | 24.6 / 22.3; 1.04 [0.869, 1.24]; 30 / 0 / 0 | 309 / 312; 0.995 [0.951, 1.04]; 30 / 0 / 0 |

Counts per cell and family (targets: 4) are in m3b_v3_ratio_summary.csv; every row is in m3b_v3_ratio.csv with its status.

**Class 1 against rho_q (absolute targets only).** For class 1 (insertion; P(D = -1) = 0 for every class-1 move in the source file) the predicted ratio is rho_q. The median of ratio / rho_q over the ok rows, for the three p' bands, is:

- qae: target 0.001: 1.02 to 1.24; target 0.0001: 1.00 to 1.04
- mlae_S32: target 0.001: 0.95 to 1.04; target 0.0001: 0.98 to 1.06
- mlae_S64: target 0.001: 0.94 to 1.01; target 0.0001: 0.98 to 1.01
- mlae_S128: target 0.001: 0.98 to 1.00; target 0.0001: 0.99 to 1.00
- classical: target 0.001: 1.00 to 1.01; target 0.0001: 1.00 to 1.01

Only class 1 is compared with rho_q. For classes 2 and 3 the signed prediction is 2(sqrt v+ + sqrt v-)/sqrt(v) (in [2, 2 sqrt 2] times rho_q) and is reported as the prediction, not as rho_q.

Figure m3b_v3_error.png: the 90 % quantile error against A-applications (classical samples), log-log, for three stand-alone amplitudes (about 0.005, 0.03, 0.2). The QAE curve is jagged because the exact 90 % quantile depends on the alignment f_m of Section 3. At a = 0.005025 the curve is flat at 1.8e-8 for m >= 15: theta/pi lies within about 4e-8 of the dyadic point k/2^15, so every grid from m = 15 up returns the same estimate sin^2(pi k / 2^15) and the error cannot fall below its distance from a. This is an exact property of the estimator at this amplitude, not a numerical floor.

## 6. MLAE with S = 1 and S = 8: not rerun (from v2)

v2 included S = 1 and S = 8. v3 keeps S in {32, 64, 128} only and does not rerun S = 1 or S = 8. The table is computed from m3b_v2_rms.csv (2000 seeds, the v2 run) over all v2 amplitude records (stand-alone and difference). S = 1 does not converge: its median 90 % error at J = 10 is 0.51, and only 17% of records reach 1e-3 at any J and 9% reach 1e-4. S = 8 is too slow for the 1e-4 target: 64% of records reach 1e-3 at J = 10 but only 19% reach 1e-4 at any J (S = 64: 98%). Neither S = 1 nor S = 8 is used in the cost comparison.

| family (v2) | records | median q90 at J = 10 | fraction with q90 <= 1e-3 at J = 10 | fraction with q90 <= 1e-3 at some J | fraction with q90 <= 1e-4 at some J |
|---|---|---|---|---|---|
| mlae_S1 | 183 | 0.505 | 0.08 | 0.17 | 0.09 |
| mlae_S8 | 183 | 0.000564 | 0.64 | 0.70 | 0.19 |
| mlae_S64 | 183 | 4.11e-05 | 1.00 | 1.00 | 0.98 |

## 7. Run

- validation: reused from v2 (no circuits rerun)
- simulation time: 3846 s with 2 workers
- total time: 3857 s (analysis and summary re-rendered afterwards)
- QAE: exact pmf, m = 3..22
- MLAE: J = 1..10, S in [32, 64, 128], 4000 seeds, grid 20001 points + golden-section
- classical: N = 2^4..2^26, 4000 seeds
- seeds: root SeedSequence(20261013); one spawned stream per amplitude, estimator and setting
- python: 3.11.17

## 8. Deviations from the brief

- Sample: 30 moves per class x band cell (9 cells, 270 moves, 588 unique amplitudes). Every cell had at least 30 eligible moves. The 200-hit rule applies to every nonzero amplitude of a move, so a move is excluded if any of its amplitudes is below it; both counts are in Section 2.
- One-sided moves are those with exactly one nonzero side. The brief gives P(D = -1) = 0; in classes 2 and 3 the nonzero side is sometimes P(D = -1), so the single estimate is taken on whichever side is nonzero.
- Relative targets were not specified in the brief beyond the example. The definitions used are T_sa = r p' and T_D = r (a+ + a-). For signed moves each of the two estimates has target T_D / 2, as in v2 for absolute targets.
- Grids were not reduced: QAE m = 3..22, MLAE J = 1..10 with S in {32, 64, 128}, classical N = 2^4..2^26. Sampled estimators use 4000 seeds; QAE is exact and has no seeds.
- Slopes use the same six-setting window for every family, as in v2. For QAE the per-amplitude slope is alignment noise (Section 3), so the median over records is the value to read; per-record spreads are shown.
- Step 1 was reused from v2 and not rerun, as the brief allows. The MLAE code was rewritten for speed (one matrix product into a reused buffer, refinement vectorised over seeds). The grid, argmax and golden-section refinement are unchanged, and on four test settings the new and previous outputs agree exactly (maximum difference 0.0).
- The ratio table reports rows at the floor and rows that fail as separate counts. They are excluded from the medians, which are computed on the same ok rows for ratio, prediction and their quotient.
- The exact QAE quantile is the smallest value with cdf >= q; the sampled quantile uses the inverted-cdf rule, the same definition on a sample.

