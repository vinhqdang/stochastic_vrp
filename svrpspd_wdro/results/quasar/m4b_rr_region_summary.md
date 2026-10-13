# QUASAR M4b repair-step statistics (task B, mode rr: ruin-and-recreate)

Steps: 2052 over 26 instances; alpha = 0.1, DELTA = 0.05, V_WC = 0.25; ratios are medians of per-step Nq/Nc; share Nq<Nc = share of steps where the quantum cost is below the classical one.

## Stop index s (eps-free)

| family | gate | steps | s=0 | s_q50 | s_q90 | s_max | M_med |
|---|---|---|---|---|---|---|---|
| VNN | Det | 1026 | 0.0799 | 10 | 50 | 315 | 107 |
| VNN | SAA | 1026 | 0 | 1 | 4 | 64 | 111 |
| VNN | ALL | 2052 | 0.04 | 2 | 32 | 315 | 109 |
| ALL | Det | 1026 | 0.0799 | 10 | 50 | 315 | 107 |
| ALL | SAA | 1026 | 0 | 1 | 4 | 64 | 111 |
| ALL | ALL | 2052 | 0.04 | 2 | 32 | 315 | 109 |

## eps = 0.01 (suffix _e2)

| family | gate | steps | k0 | k1 | k_q50 | k_q90 | k_max | band_med | Nq/Nc full wc | Nq/Nc full opt | Nq/Nc core | Nq/Nc crn core | share Nq<Nc full wc | share Nq<Nc full opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN | Det | 1026 | 0.0867 | 0.109 | 10 | 51 | 315 | 0 | 270 | 271 | 0.121 | 0.384 | 0 | 0 |
| VNN | SAA | 1026 | 0 | 0.716 | 1 | 5 | 64 | 0 | 313 | 465 | 0.626 | 0.747 | 0 | 0 |
| VNN | ALL | 2052 | 0.0434 | 0.413 | 2 | 35 | 315 | 0 | 313 | 392 | 0.313 | 0.561 | 0 | 0 |
| ALL | Det | 1026 | 0.0867 | 0.109 | 10 | 51 | 315 | 0 | 270 | 271 | 0.121 | 0.384 | 0 | 0 |
| ALL | SAA | 1026 | 0 | 0.716 | 1 | 5 | 64 | 0 | 313 | 465 | 0.626 | 0.747 | 0 | 0 |
| ALL | ALL | 2052 | 0.0434 | 0.413 | 2 | 35 | 315 | 0 | 313 | 392 | 0.313 | 0.561 | 0 | 0 |

## eps = 0.003 (suffix _e3x)

| family | gate | steps | k0 | k1 | k_q50 | k_q90 | k_max | band_med | Nq/Nc full wc | Nq/Nc full opt | Nq/Nc core | Nq/Nc crn core | share Nq<Nc full wc | share Nq<Nc full opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN | Det | 1026 | 0.0828 | 0.115 | 10 | 50.5 | 315 | 0 | 97.9 | 99.6 | 0.043 | 0.106 | 0 | 0 |
| VNN | SAA | 1026 | 0 | 0.729 | 1 | 4 | 64 | 0 | 114 | 161 | 0.211 | 0.298 | 0 | 0 |
| VNN | ALL | 2052 | 0.0414 | 0.422 | 2 | 33 | 315 | 0 | 114 | 139 | 0.106 | 0.193 | 0 | 0 |
| ALL | Det | 1026 | 0.0828 | 0.115 | 10 | 50.5 | 315 | 0 | 97.9 | 99.6 | 0.043 | 0.106 | 0 | 0 |
| ALL | SAA | 1026 | 0 | 0.729 | 1 | 4 | 64 | 0 | 114 | 161 | 0.211 | 0.298 | 0 | 0 |
| ALL | ALL | 2052 | 0.0414 | 0.422 | 2 | 33 | 315 | 0 | 114 | 139 | 0.106 | 0.193 | 0 | 0 |

## eps = 0.001 (suffix _e3)

| family | gate | steps | k0 | k1 | k_q50 | k_q90 | k_max | band_med | Nq/Nc full wc | Nq/Nc full opt | Nq/Nc core | Nq/Nc crn core | share Nq<Nc full wc | share Nq<Nc full opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN | Det | 1026 | 0.0799 | 0.115 | 10 | 50 | 315 | 0 | 21.8 | 34 | 0.0145 | 0.0359 | 0 | 0 |
| VNN | SAA | 1026 | 0 | 0.737 | 1 | 4 | 64 | 0 | 25.3 | 35.4 | 0.0459 | 0.126 | 0 | 0 |
| VNN | ALL | 2052 | 0.04 | 0.426 | 2 | 32 | 315 | 0 | 25.3 | 35.4 | 0.0325 | 0.0691 | 0 | 0 |
| ALL | Det | 1026 | 0.0799 | 0.115 | 10 | 50 | 315 | 0 | 21.8 | 34 | 0.0145 | 0.0359 | 0 | 0 |
| ALL | SAA | 1026 | 0 | 0.737 | 1 | 4 | 64 | 0 | 25.3 | 35.4 | 0.0459 | 0.126 | 0 | 0 |
| ALL | ALL | 2052 | 0.04 | 0.426 | 2 | 32 | 315 | 0 | 25.3 | 35.4 | 0.0325 | 0.0691 | 0 | 0 |

## Insertions in mode rr (new route share, median chosen rank)

| family | gate | steps | new route share | chosen rank median |
|---|---|---|---|---|
| VNN | Det | 1026 | 0.0799 | 10 |
| VNN | SAA | 1026 | 0 | 1 |
| VNN | ALL | 2052 | 0.04 | 2 |
| ALL | Det | 1026 | 0.0799 | 10 |
| ALL | SAA | 1026 | 0 | 1 |
| ALL | ALL | 2052 | 0.04 | 2 |
