# QUASAR cert: simultaneous risk certification of region plans

alpha = 0.1, delta = 0.05, eps in [0.01, 0.003, 0.001, 0.0001]; 148 instances, 10518 routes. Groups = (region family, gate); d = routes in the group.

Note: p_j are Monte Carlo estimates from 20 000 common scenarios per instance (standard error up to about 0.0016 at p = 0.05), so gaps below about 3e-3 are not resolved.

Q = sum_j L_j q_cost_j (applications of A, A^dagger, model L); Q_ideal = sum_j L_j r_t (2 M_t + 1) for the stopping round t alone (no doubling, no earlier rounds), comparable with C; C = sum_j L_j n_j (classical route samples, model L); C_shared = sum over instances of max_j n_j * n_total(instance) (one whole-plan sample per instance). Q_unw and C_unw are the same sums without the L_j weights. k* is the best fixed k for the group and eps (oracle choice on the true risks). Ratios are per route, q_cost/c_n, and are independent of L_j.
Run time: 38.3 s.

| region | gate | eps | d | k* | Q | C | C_shared | Q/C | Q/C_shared | Q_ideal | Q_ideal/C | Q_unw | C_unw | Q floor share | median q/c | 90% q/c | share q<c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN | Det | 0.01 | 257 | 1 | 1.255e+07 | 1.699e+06 | 1.306e+07 | 7.382 | 0.961 | 6.477e+06 | 3.811 | 1.544e+06 | 2.413e+05 | 0.000 | 11.021 | 18.156 | 0.000 |
| VNN | Det | 0.003 | 257 | 1 | 1.255e+07 | 1.699e+06 | 1.306e+07 | 7.382 | 0.961 | 6.477e+06 | 3.811 | 1.544e+06 | 2.413e+05 | 0.000 | 11.021 | 18.156 | 0.000 |
| VNN | Det | 0.001 | 257 | 1 | 1.255e+07 | 1.699e+06 | 1.306e+07 | 7.382 | 0.961 | 6.477e+06 | 3.811 | 1.544e+06 | 2.413e+05 | 0.000 | 11.021 | 18.156 | 0.000 |
| VNN | Det | 0.0001 | 257 | 1 | 1.255e+07 | 1.699e+06 | 1.306e+07 | 7.382 | 0.961 | 6.477e+06 | 3.811 | 1.544e+06 | 2.413e+05 | 0.000 | 11.021 | 18.156 | 0.000 |
| VNN | SAA | 0.01 | 375 | 1 | 3.284e+07 | 6.931e+06 | 9.251e+06 | 4.739 | 3.550 | 1.678e+07 | 2.421 | 4.665e+06 | 9.529e+05 | 0.000 | 4.833 | 6.610 | 0.000 |
| VNN | SAA | 0.003 | 375 | 1 | 3.284e+07 | 6.931e+06 | 9.251e+06 | 4.739 | 3.550 | 1.678e+07 | 2.421 | 4.665e+06 | 9.529e+05 | 0.000 | 4.833 | 6.610 | 0.000 |
| VNN | SAA | 0.001 | 375 | 1 | 3.284e+07 | 6.931e+06 | 9.251e+06 | 4.739 | 3.550 | 1.678e+07 | 2.421 | 4.665e+06 | 9.529e+05 | 0.000 | 4.833 | 6.610 | 0.000 |
| VNN | SAA | 0.0001 | 375 | 1 | 3.284e+07 | 6.931e+06 | 9.251e+06 | 4.739 | 3.550 | 1.678e+07 | 2.421 | 4.665e+06 | 9.529e+05 | 0.000 | 4.833 | 6.610 | 0.000 |
| GD | Det | 0.01 | 4019 | 1 | 3.796e+08 | 1.045e+08 | 2.855e+09 | 3.634 | 0.133 | 1.938e+08 | 1.855 | 4.603e+07 | 1.382e+07 | 0.003 | 10.642 | 12.860 | 0.006 |
| GD | Det | 0.003 | 4019 | 1 | 4.072e+08 | 2.339e+08 | 1.081e+10 | 1.741 | 0.038 | 2.078e+08 | 0.889 | 5.014e+07 | 3.322e+07 | 0.000 | 10.642 | 12.860 | 0.006 |
| GD | Det | 0.001 | 4019 | 1 | 4.140e+08 | 4.013e+08 | 2.214e+10 | 1.032 | 0.019 | 2.112e+08 | 0.526 | 5.111e+07 | 5.725e+07 | 0.000 | 10.642 | 12.860 | 0.006 |
| GD | Det | 0.0001 | 4019 | 1 | 4.140e+08 | 4.013e+08 | 2.214e+10 | 1.032 | 0.019 | 2.112e+08 | 0.526 | 5.111e+07 | 5.725e+07 | 0.000 | 10.642 | 12.860 | 0.006 |
| GD | SAA | 0.01 | 5867 | 1 | 6.319e+08 | 1.330e+08 | 1.892e+08 | 4.751 | 3.340 | 3.198e+08 | 2.404 | 9.569e+07 | 1.968e+07 | 0.000 | 4.877 | 6.445 | 0.000 |
| GD | SAA | 0.003 | 5867 | 1 | 6.319e+08 | 1.330e+08 | 1.892e+08 | 4.751 | 3.340 | 3.198e+08 | 2.404 | 9.569e+07 | 1.968e+07 | 0.000 | 4.877 | 6.445 | 0.000 |
| GD | SAA | 0.001 | 5867 | 1 | 6.319e+08 | 1.330e+08 | 1.892e+08 | 4.751 | 3.340 | 3.198e+08 | 2.404 | 9.569e+07 | 1.968e+07 | 0.000 | 4.877 | 6.445 | 0.000 |
| GD | SAA | 0.0001 | 5867 | 1 | 6.319e+08 | 1.330e+08 | 1.892e+08 | 4.751 | 3.340 | 3.198e+08 | 2.404 | 9.569e+07 | 1.968e+07 | 0.000 | 4.877 | 6.445 | 0.000 |
