# QUASAR M3a v3 (revision 3): interpretation layer on the BATON copula model

Supersedes the interpretation of m3a_v2_summary.md (its Tables B, D and E are not carried over; Table A is replaced by Section 1, which bands by p' and by distance to the gate level). Revision 3 changes: classical certificate floor in Section 3, quadrature study with equal-probability factor bins in Section 4b, bias by route length and a qualified b rule in Section 4a, instance-level intervals in Section 1a.

Scenarios: N = 200000 per instance, seed 7, rho = 0.6, the BATON copula; moves and routes are the cached v2 results (`m3a_v2_moves.csv.gz`, `m3a_v2_routes.csv`). Scenarios were recomputed for the end-load and level checks; the recomputed intra-move P(D = 1) and P(D = -1) agree with the cached values to a maximum absolute difference of 5.0e-06 (CSV rounding), which confirms that the draws are identical.

Definitions: p' = P(F' > Q) (new route), p = P(F > Q) (base), D = 1[F' > Q] - 1[F > Q], rho_q = sqrt(Var D) / sqrt(p'(1 - p')). Zeros in the sample are upper bounds: with no scenario showing D != 0 out of N = 200000, P(D != 0) <= 3/N = 1.5e-05 (95 %), which gives the plug-in upper bound on rho_q reported next to the medians. Route-bootstrap intervals resample routes (clusters, with replacement, 1000 replicates); instance-level intervals resample instances in the same way (two-stage clustering is not used; within a sampled instance all its routes are kept). The bootstrap median is the lower weighted median, the table quantiles are linear.

## 1. Decision bands of p' and of the distance to the gate level

Gate level: per-route failure target 0.10 (1 - ALPHA of dethloff_runner). Decision band p' in [0.02, 0.2]. Distances |p' - 0.10| in the second table.

### 1a. Bands of p'

| class | family | band | n | median rho_q [95% route-bootstrap CI] | median rho_q [95% instance-level CI] | p10 | p25 | p75 | p90 | Var=0 share (in-sample) | median rho_q upper bound (zeros at 3/N) | NaN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | <0.02 | 37207 | 0.835 [0.824, 0.847] | [0.818, 0.851] | 0.476 | 0.686 | 0.934 | 0.984 | 0.004 | 0.837 | 2374 |
| insertion | CON | 0.02-0.2 | 89690 | 0.897 [0.888, 0.906] | [0.875, 0.917] | 0.558 | 0.753 | 0.97 | 0.991 | 0.000 | 0.897 | 0 |
| insertion | CON | 0.2-0.5 | 62616 | 0.95 [0.941, 0.957] | [0.903, 0.963] | 0.624 | 0.795 | 0.989 | 0.997 | 0.000 | 0.95 | 0 |
| insertion | CON | >0.5 | 50873 | 1.02 [1.02, 1.03] | [1.01, 1.03] | 0.713 | 0.961 | 1.18 | 1.53 | 0.000 | 1.02 | 0 |
| insertion | SCA | <0.02 | 36876 | 0.826 [0.813, 0.839] | [0.808, 0.847] | 0.406 | 0.655 | 0.935 | 0.986 | 0.005 | 0.828 | 1688 |
| insertion | SCA | 0.02-0.2 | 87287 | 0.902 [0.893, 0.911] | [0.876, 0.923] | 0.528 | 0.751 | 0.969 | 0.99 | 0.000 | 0.902 | 0 |
| insertion | SCA | 0.2-0.5 | 62328 | 0.946 [0.937, 0.955] | [0.903, 0.963] | 0.597 | 0.796 | 0.99 | 0.997 | 0.000 | 0.946 | 0 |
| insertion | SCA | >0.5 | 51689 | 1.02 [1.02, 1.03] | [1.01, 1.03] | 0.74 | 0.966 | 1.19 | 1.54 | 0.000 | 1.02 | 0 |
| insertion | CMT | <0.02 | 360 | 0.749 [0.661, 0.821] | [0.661, 0.821] | 0.508 | 0.626 | 0.84 | 0.938 | 0.006 | 0.749 | 200 |
| insertion | CMT | 0.02-0.2 | 1077 | 0.58 [0.534, 0.627] | [0.524, 0.616] | 0.35 | 0.474 | 0.694 | 0.784 | 0.003 | 0.58 | 0 |
| insertion | CMT | 0.2-0.5 | 5172 | 0.36 [0.328, 0.405] | [0.326, 0.408] | 0.176 | 0.263 | 0.488 | 0.613 | 0.009 | 0.361 | 0 |
| insertion | CMT | >0.5 | 11391 | 0.596 [0.579, 0.621] | [0.559, 0.648] | 0.4 | 0.476 | 0.727 | 0.858 | 0.000 | 0.596 | 0 |
| insertion | city | <0.02 | 3623 | 0.912 [0.837, 1] | [0.832, 0.986] | 0.613 | 0.751 | 1 | 1 | 0.000 | 0.912 | 3586 |
| insertion | city | 0.02-0.2 | 42345 | 0.704 [0.7, 0.709] | [0.695, 0.713] | 0.503 | 0.604 | 0.795 | 0.856 | 0.000 | 0.704 | 0 |
| insertion | city | 0.2-0.5 | 9537 | 0.427 [0.411, 0.443] | [0.413, 0.449] | 0.284 | 0.355 | 0.529 | 0.711 | 0.000 | 0.427 | 0 |
| insertion | city | >0.5 | 18903 | 0.597 [0.591, 0.604] | [0.59, 0.607] | 0.458 | 0.515 | 0.693 | 0.782 | 0.000 | 0.597 | 0 |
| intra | CON | <0.02 | 20208 | 0.0818 [0.0728, 0.0907] | [0.072, 0.0963] | 0 | 0.0258 | 0.16 | 0.264 | 0.237 | 0.107 | 255 |
| intra | CON | 0.02-0.2 | 27231 | 0.127 [0.119, 0.135] | [0.116, 0.143] | 0.0282 | 0.0629 | 0.214 | 0.311 | 0.029 | 0.127 | 0 |
| intra | CON | 0.2-0.5 | 9125 | 0.15 [0.136, 0.165] | [0.134, 0.177] | 0.0353 | 0.0775 | 0.245 | 0.357 | 0.006 | 0.15 | 0 |
| intra | CON | >0.5 | 7071 | 0.175 [0.161, 0.191] | [0.155, 0.202] | 0.0503 | 0.0978 | 0.265 | 0.363 | 0.003 | 0.175 | 0 |
| intra | SCA | <0.02 | 17807 | 0.0798 [0.0696, 0.0906] | [0.0673, 0.0981] | 0 | 0 | 0.167 | 0.27 | 0.286 | 0.117 | 465 |
| intra | SCA | 0.02-0.2 | 27570 | 0.131 [0.124, 0.14] | [0.119, 0.148] | 0.0296 | 0.065 | 0.219 | 0.32 | 0.026 | 0.131 | 0 |
| intra | SCA | 0.2-0.5 | 9359 | 0.154 [0.139, 0.171] | [0.133, 0.181] | 0.0367 | 0.0818 | 0.246 | 0.354 | 0.007 | 0.154 | 0 |
| intra | SCA | >0.5 | 6134 | 0.166 [0.152, 0.189] | [0.147, 0.198] | 0.0434 | 0.0891 | 0.266 | 0.385 | 0.005 | 0.166 | 0 |
| intra | CMT | <0.02 | 74 | 0 [0, 0] | [0, 0] | 0 | 0 | 0 | 0 | 1.000 | 0.0533 | 1 |
| intra | CMT | 0.02-0.2 | 1257 | 0 [0, 0.0202] | [0, 0.015] | 0 | 0 | 0.0213 | 0.135 | 0.652 | 0.0149 | 0 |
| intra | CMT | 0.2-0.5 | 11346 | 0 [0, 0.00633] | [0, 0.0119] | 0 | 0 | 0.0668 | 0.256 | 0.557 | 0.00787 | 0 |
| intra | CMT | >0.5 | 4438 | 0.0586 [0.0195, 0.11] | [0.0184, 0.103] | 0 | 0 | 0.287 | 0.529 | 0.314 | 0.0588 | 0 |
| intra | city | <0.02 | 3251 | 0 [0, 0] | [0, 0] | 0 | 0 | 0 | 0 | 0.907 | 0.0393 | 825 |
| intra | city | 0.02-0.2 | 38069 | 0 [0, 0] | [0, 0] | 0 | 0 | 0 | 0.0222 | 0.765 | 0.0207 | 0 |
| intra | city | 0.2-0.5 | 27328 | 0.0158 [0.0135, 0.0185] | [0.0135, 0.018] | 0 | 0.00453 | 0.0392 | 0.0735 | 0.193 | 0.0158 | 0 |
| alt-position | CON | <0.02 | 37226 | 0.072 [0.066, 0.0774] | [0.0603, 0.0855] | 0 | 0 | 0.154 | 0.26 | 0.268 | 0.105 | 2374 |
| alt-position | CON | 0.02-0.2 | 89686 | 0.139 [0.134, 0.144] | [0.12, 0.157] | 0.0279 | 0.0676 | 0.233 | 0.342 | 0.028 | 0.139 | 0 |
| alt-position | CON | 0.2-0.5 | 62619 | 0.194 [0.187, 0.201] | [0.173, 0.208] | 0.0514 | 0.109 | 0.292 | 0.393 | 0.005 | 0.194 | 0 |
| alt-position | CON | >0.5 | 50855 | 0.232 [0.225, 0.239] | [0.212, 0.243] | 0.0717 | 0.139 | 0.333 | 0.427 | 0.002 | 0.232 | 0 |
| alt-position | SCA | <0.02 | 36866 | 0.0645 [0.0594, 0.0705] | [0.0525, 0.0773] | 0 | 0 | 0.143 | 0.251 | 0.295 | 0.102 | 1689 |
| alt-position | SCA | 0.02-0.2 | 87303 | 0.141 [0.135, 0.145] | [0.122, 0.156] | 0.0281 | 0.0687 | 0.236 | 0.354 | 0.028 | 0.141 | 0 |
| alt-position | SCA | 0.2-0.5 | 62307 | 0.193 [0.186, 0.2] | [0.172, 0.206] | 0.0495 | 0.108 | 0.294 | 0.4 | 0.005 | 0.193 | 0 |
| alt-position | SCA | >0.5 | 51703 | 0.236 [0.229, 0.243] | [0.218, 0.246] | 0.0721 | 0.14 | 0.339 | 0.434 | 0.003 | 0.236 | 0 |
| alt-position | CMT | <0.02 | 360 | 0 [0, 0] | [0, 0] | 0 | 0 | 0 | 0.259 | 0.839 | 0.132 | 200 |
| alt-position | CMT | 0.02-0.2 | 1070 | 0 [0, 0.0126] | [0, 0.00606] | 0 | 0 | 0.029 | 0.224 | 0.593 | 0.014 | 0 |
| alt-position | CMT | 0.2-0.5 | 5193 | 0.00447 [0, 0.0127] | [0, 0.0142] | 0 | 0 | 0.0922 | 0.285 | 0.493 | 0.00802 | 0 |
| alt-position | CMT | >0.5 | 11377 | 0.0208 [0.00895, 0.0463] | [0.00895, 0.0504] | 0 | 0 | 0.213 | 0.412 | 0.399 | 0.0208 | 0 |
| alt-position | city | <0.02 | 3623 | 0 [0, 0] | [0, 0] | 0 | 0 | 0 | 0 | 0.957 | 0.106 | 3586 |
| alt-position | city | 0.02-0.2 | 42350 | 0 [0, 0] | [0, 0] | 0 | 0 | 0.00868 | 0.0252 | 0.708 | 0.0161 | 0 |
| alt-position | city | 0.2-0.5 | 9524 | 0.00897 [0.00775, 0.011] | [0.00742, 0.011] | 0 | 0 | 0.0245 | 0.0559 | 0.306 | 0.00957 | 0 |
| alt-position | city | >0.5 | 18911 | 0.0196 [0.0173, 0.0222] | [0.0171, 0.022] | 0 | 0.00636 | 0.0458 | 0.0847 | 0.163 | 0.0196 | 0 |

### 1b. Bands of |p' - 0.10|

| class | family | band | n | median rho_q [95% route-bootstrap CI] | p10 | p25 | p75 | p90 | Var=0 share (in-sample) | median rho_q upper bound (zeros at 3/N) | NaN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | <0.02 | 18841 | 0.884 [0.872, 0.901] | 0.678 | 0.792 | 0.98 | 0.992 | 0.000 | 0.884 | 0 |
| insertion | CON | 0.02-0.1 | 108056 | 0.879 [0.873, 0.886] | 0.517 | 0.711 | 0.958 | 0.99 | 0.001 | 0.88 | 0 |
| insertion | CON | 0.1-0.4 | 62616 | 0.95 [0.941, 0.957] | 0.624 | 0.795 | 0.989 | 0.997 | 0.000 | 0.95 | 2374 |
| insertion | CON | >0.4 | 50873 | 1.02 [1.02, 1.03] | 0.713 | 0.961 | 1.18 | 1.53 | 0.000 | 1.02 | 0 |
| insertion | SCA | <0.02 | 17330 | 0.888 [0.877, 0.905] | 0.688 | 0.804 | 0.976 | 0.99 | 0.000 | 0.888 | 0 |
| insertion | SCA | 0.02-0.1 | 106833 | 0.879 [0.872, 0.887] | 0.464 | 0.695 | 0.96 | 0.989 | 0.002 | 0.88 | 0 |
| insertion | SCA | 0.1-0.4 | 62328 | 0.946 [0.937, 0.955] | 0.597 | 0.796 | 0.99 | 0.997 | 0.000 | 0.946 | 1688 |
| insertion | SCA | >0.4 | 51689 | 1.02 [1.02, 1.03] | 0.74 | 0.966 | 1.19 | 1.54 | 0.000 | 1.02 | 0 |
| insertion | CMT | <0.02 | 307 | 0.588 [0.364, 0.727] | 0.332 | 0.476 | 0.704 | 0.759 | 0.000 | 0.588 | 0 |
| insertion | CMT | 0.02-0.1 | 1130 | 0.618 [0.545, 0.723] | 0.4 | 0.5 | 0.757 | 0.862 | 0.004 | 0.619 | 0 |
| insertion | CMT | 0.1-0.4 | 5172 | 0.36 [0.328, 0.405] | 0.176 | 0.263 | 0.488 | 0.613 | 0.009 | 0.361 | 200 |
| insertion | CMT | >0.4 | 11391 | 0.596 [0.579, 0.621] | 0.4 | 0.476 | 0.727 | 0.858 | 0.000 | 0.596 | 0 |
| insertion | city | <0.02 | 10701 | 0.791 [0.785, 0.796] | 0.719 | 0.755 | 0.82 | 0.846 | 0.000 | 0.791 | 0 |
| insertion | city | 0.02-0.1 | 35267 | 0.673 [0.666, 0.682] | 0.486 | 0.581 | 0.777 | 0.894 | 0.000 | 0.673 | 0 |
| insertion | city | 0.1-0.4 | 9537 | 0.427 [0.411, 0.443] | 0.284 | 0.355 | 0.529 | 0.711 | 0.000 | 0.427 | 3586 |
| insertion | city | >0.4 | 18903 | 0.597 [0.591, 0.604] | 0.458 | 0.515 | 0.693 | 0.782 | 0.000 | 0.597 | 0 |
| intra | CON | <0.02 | 2892 | 0.151 [0.116, 0.194] | 0.0317 | 0.0713 | 0.247 | 0.355 | 0.017 | 0.151 | 0 |
| intra | CON | 0.02-0.1 | 44547 | 0.106 [0.0994, 0.112] | 0 | 0.0459 | 0.19 | 0.291 | 0.124 | 0.115 | 0 |
| intra | CON | 0.1-0.4 | 9125 | 0.15 [0.136, 0.165] | 0.0353 | 0.0775 | 0.245 | 0.357 | 0.006 | 0.15 | 255 |
| intra | CON | >0.4 | 7071 | 0.175 [0.161, 0.191] | 0.0503 | 0.0978 | 0.265 | 0.363 | 0.003 | 0.175 | 0 |
| intra | SCA | <0.02 | 3104 | 0.118 [0.101, 0.146] | 0.0284 | 0.0626 | 0.195 | 0.311 | 0.020 | 0.118 | 0 |
| intra | SCA | 0.02-0.1 | 42273 | 0.111 [0.104, 0.119] | 0 | 0.0457 | 0.199 | 0.304 | 0.136 | 0.125 | 0 |
| intra | SCA | 0.1-0.4 | 9359 | 0.154 [0.139, 0.171] | 0.0367 | 0.0818 | 0.246 | 0.354 | 0.007 | 0.154 | 465 |
| intra | SCA | >0.4 | 6134 | 0.166 [0.152, 0.189] | 0.0434 | 0.0891 | 0.266 | 0.385 | 0.005 | 0.166 | 0 |
| intra | CMT | <0.02 | 424 | 0 [0, 0.0128] | 0 | 0 | 0.0182 | 0.055 | 0.644 | 0.0128 | 0 |
| intra | CMT | 0.02-0.1 | 907 | 0 [0, 0.0344] | 0 | 0 | 0.0213 | 0.157 | 0.685 | 0.0162 | 0 |
| intra | CMT | 0.1-0.4 | 11346 | 0 [0, 0.00633] | 0 | 0 | 0.0668 | 0.256 | 0.557 | 0.00787 | 1 |
| intra | CMT | >0.4 | 4438 | 0.0586 [0.0195, 0.11] | 0 | 0 | 0.287 | 0.529 | 0.314 | 0.0588 | 0 |
| intra | city | <0.02 | 642 | 0.00816 [0, 0.0221] | 0 | 0 | 0.0221 | 0.0444 | 0.460 | 0.0141 | 0 |
| intra | city | 0.02-0.1 | 40678 | 0 [0, 0] | 0 | 0 | 0 | 0.0209 | 0.781 | 0.0208 | 0 |
| intra | city | 0.1-0.4 | 27328 | 0.0158 [0.0135, 0.0185] | 0 | 0.00453 | 0.0392 | 0.0735 | 0.193 | 0.0158 | 825 |
| alt-position | CON | <0.02 | 18863 | 0.155 [0.149, 0.162] | 0.0379 | 0.0812 | 0.248 | 0.357 | 0.011 | 0.155 | 0 |
| alt-position | CON | 0.02-0.1 | 108049 | 0.112 [0.107, 0.116] | 0 | 0.0457 | 0.207 | 0.315 | 0.114 | 0.123 | 0 |
| alt-position | CON | 0.1-0.4 | 62619 | 0.194 [0.187, 0.201] | 0.0514 | 0.109 | 0.292 | 0.393 | 0.005 | 0.194 | 2374 |
| alt-position | CON | >0.4 | 50855 | 0.232 [0.225, 0.239] | 0.0717 | 0.139 | 0.333 | 0.427 | 0.002 | 0.232 | 0 |
| alt-position | SCA | <0.02 | 17324 | 0.157 [0.151, 0.163] | 0.0388 | 0.0839 | 0.248 | 0.368 | 0.012 | 0.157 | 0 |
| alt-position | SCA | 0.02-0.1 | 106845 | 0.11 [0.105, 0.114] | 0 | 0.0435 | 0.208 | 0.325 | 0.122 | 0.123 | 0 |
| alt-position | SCA | 0.1-0.4 | 62307 | 0.193 [0.186, 0.2] | 0.0495 | 0.108 | 0.294 | 0.4 | 0.005 | 0.193 | 1689 |
| alt-position | SCA | >0.4 | 51703 | 0.236 [0.229, 0.243] | 0.0721 | 0.14 | 0.339 | 0.434 | 0.003 | 0.236 | 0 |
| alt-position | CMT | <0.02 | 312 | 0 [0, 0] | 0 | 0 | 0.01 | 0.179 | 0.712 | 0.0136 | 0 |
| alt-position | CMT | 0.02-0.1 | 1118 | 0 [0, 0.00905] | 0 | 0 | 0.0223 | 0.234 | 0.640 | 0.0292 | 0 |
| alt-position | CMT | 0.1-0.4 | 5193 | 0.00447 [0, 0.0127] | 0 | 0 | 0.0922 | 0.285 | 0.493 | 0.00802 | 200 |
| alt-position | CMT | >0.4 | 11377 | 0.0208 [0.00895, 0.0463] | 0 | 0 | 0.213 | 0.412 | 0.399 | 0.0208 | 0 |
| alt-position | city | <0.02 | 10701 | 0 [0, 0] | 0 | 0 | 0.0107 | 0.0303 | 0.650 | 0.0136 | 0 |
| alt-position | city | 0.02-0.1 | 35272 | 0 [0, 0] | 0 | 0 | 0 | 0.0221 | 0.751 | 0.0171 | 0 |
| alt-position | city | 0.1-0.4 | 9524 | 0.00897 [0.00775, 0.011] | 0 | 0 | 0.0245 | 0.0559 | 0.306 | 0.00957 | 3586 |
| alt-position | city | >0.4 | 18911 | 0.0196 [0.0173, 0.0222] | 0 | 0.00636 | 0.0458 | 0.0847 | 0.163 | 0.0196 | 0 |

## 2. End-load lemma

Lemma. For any reordering r' of the same customers, L_0 = sum of deliveries and L_L = sum of pickups are order-invariant. Let E = {sum d <= Q and sum p <= Q}. Outside E, one end load exceeds Q for every reordering, so both overflow indicators equal 1 and D = 0. On E, D != 0 forces max(F_r, F_r') > Q, so P(D != 0) <= P(E and (F_r > Q or F_r' > Q)). The lemma covers every reordering of the same customers, hence intra-route moves and alternative-position moves (class 3, where the two routes contain the same customers) alike; the table below checks intra-route moves only. LEMMA BOUND (valid): B_union = P(E and (F_r > Q or F_r' > Q)), with r' ranging over the sampled intra neighbours of the route; it bounds P(D != 0) of every such move. DIAGNOSTIC ONLY, NOT A VALID BOUND: B_spec = P(E and F_r > Q). Explained share = P(max(sum d, sum p) > Q) / P(F_r > Q).

| family | routes with intra moves | intra moves | observed Var=0 share | median B_spec | median B_union | pooled explained share | median per-route explained share | max P(D != 0) over moves | median per-route max P(D != 0) | routes with B_spec < route max | routes with B_union < route max | D != 0 with an end load > Q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CON | 1273 | 63890 | 0.092 | 0.00034 | 0.00481 | 0.953 | 0.983 | 0.157 | 0.0034 | 1110 | 0 | 0 |
| SCA | 1261 | 61335 | 0.104 | 0.000425 | 0.00552 | 0.954 | 0.982 | 0.145 | 0.00402 | 1093 | 0 | 0 |
| CMT | 91 | 17116 | 0.503 | 0.000805 | 0.134 | 0.932 | 0.998 | 0.448 | 0.0933 | 72 | 0 | 0 |
| city | 387 | 69473 | 0.549 | 1.5e-05 | 0.0002 | 0.997 | 1.000 | 0.0157 | 0.00012 | 344 | 0 | 0 |
| all | 3012 | 211814 | 0.279 | 0.00027 | 0.00453 | 0.959 | 0.989 | 0.448 | 0.00294 | 2619 | 0 | 0 |

Consistency check (not a test of the lemma, which is an identity): every intra move (211814 sampled moves over 3012 routes with intra moves; all 3049 routes evaluated) was recomputed scenario by scenario. Scenarios with D != 0 and an end load above Q: 0. Routes with B_union below their largest observed P(D != 0): 0. The zero count checks that the move reconstruction and the scenario draws reproduce the cached move code path.

B_spec is a diagnostic and is not a valid bound on P(D != 0): it omits scenarios where only the moved route overflows. The column 'routes with B_spec < route max' counts where it falls short of the observed largest P(D != 0). B_union is the lemma bound.

## 3. Query costs per class, both paradigms, with certified classical counts

Moves with p' in (0.02, 0.5). Stand-alone: p' estimated at eps (quantum: canonical AE, 5 x (2M + 1) applications; classical: max(z^2 p'(1-p') / eps^2, 3 / eps), z = 1.96). CRN: class 1 one amplitude P(D = 1) with variance max(Var D, 3/N); classes 2-3 quantum two-estimate P(D = 1) and P(D = -1) at eps/2 each, classical direct signed mean max(z^2 max(Var D, 3/N) / eps^2, 3/eps). The 3/eps term is the classical certificate floor: without it, a move with no event in the sample is charged a CLT count (58 samples at eps = 1e-3) that certifies nothing. eps is absolute (1e-3, 1e-4) or relative, eps_i = 0.1 p'_i per move. Sums are over moves; medians are over moves.

### 3a. Within-paradigm medians (p' in (0.02, 0.5))

| class | family | moves | median rho_q (in-sample) | median rho_c (in-sample) | median rho_q upper bound | Var=0 share |
|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 0.917 | 0.841 | 0.917 | 0.000 |
| insertion | SCA | 149612 | 0.92 | 0.846 | 0.92 | 0.000 |
| insertion | CMT | 6249 | 0.394 | 0.155 | 0.394 | 0.008 |
| insertion | city | 51882 | 0.674 | 0.455 | 0.674 | 0.000 |
| intra | CON | 36355 | 0.133 | 0.0176 | 0.133 | 0.023 |
| intra | SCA | 36929 | 0.136 | 0.0186 | 0.136 | 0.021 |
| intra | CMT | 12603 | 0 | 0 | 0.00798 | 0.566 |
| intra | city | 65397 | 0 | 0 | 0.0205 | 0.526 |
| alt-position | CON | 152296 | 0.161 | 0.026 | 0.161 | 0.019 |
| alt-position | SCA | 149608 | 0.162 | 0.0262 | 0.162 | 0.018 |
| alt-position | CMT | 6263 | 0 | 0 | 0.0111 | 0.510 |
| alt-position | city | 51874 | 0 | 0 | 0.0158 | 0.634 |

Reading. rho_q is the ratio of the CRN amplitude standard deviation to the stand-alone one, so the quantum query count of the CRN difference is about rho_q times the stand-alone count. For insertions (class 1, median rho_q by family in the band: CON 0.92, SCA 0.92, CMT 0.39, city 0.67) the value is near 0.9 only on CON/SCA; on CMT and city a substantial part of the variance is removed by the common draws. Intra-route and alternative-position moves reorder the same customers, so Var(D) is small: on CON/SCA the medians are 0.13-0.16, and on CMT/city they are zero in the sample for more than half of the moves. The zero is an in-sample value and is bounded by the upper-bound column.

### 3b. Cost sums (queries) by class and family, certified classical counts

Nq/Nq_sa and Nc/Nc_sa: within-paradigm CRN gain (CRN cost over stand-alone cost). Nq/Nc: quantum CRN over classical CRN (classical direct for classes 2-3) at the same target, on the sums; the last column is the median over moves of the per-move ratio Nq/Nc. The quantum counts count applications of A; the classical counts count scenario evaluations. These are different unit costs (see the reading below).

#### absolute eps = 1e-3

| class | family | moves | sum Nq_sa | sum Nq | Nq/Nq_sa | sum Nc_sa | sum Nc | Nc/Nc_sa | Nq/Nc (sums) | median per-move Nq/Nc |
|---|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 4.874e+09 | 4.211e+09 | 0.864 | 7.919e+10 | 6.093e+10 | 0.769 | 0.0691 | 0.0794 |
| insertion | SCA | 149612 | 4.804e+09 | 4.134e+09 | 0.861 | 7.820e+10 | 5.991e+10 | 0.766 | 0.069 | 0.0796 |
| insertion | CMT | 6249 | 2.448e+08 | 1.050e+08 | 0.429 | 5.235e+09 | 9.693e+08 | 0.185 | 0.108 | 0.131 |
| insertion | city | 51882 | 1.347e+09 | 9.085e+08 | 0.674 | 2.015e+10 | 8.187e+09 | 0.406 | 0.111 | 0.13 |
| intra | CON | 36355 | 1.051e+09 | 4.636e+08 | 0.441 | 1.406e+10 | 6.541e+08 | 0.047 | 0.709 | 1.71 |
| intra | SCA | 36929 | 1.056e+09 | 4.763e+08 | 0.451 | 1.423e+10 | 6.649e+08 | 0.047 | 0.716 | 1.71 |
| intra | CMT | 12603 | 4.984e+08 | 1.302e+08 | 0.261 | 1.109e+10 | 3.094e+08 | 0.028 | 0.421 | 1.71 |
| intra | city | 65397 | 1.887e+09 | 3.750e+08 | 0.199 | 3.123e+10 | 2.198e+08 | 0.007 | 1.71 | 1.71 |
| alt-position | CON | 152296 | 4.874e+09 | 2.527e+09 | 0.518 | 7.920e+10 | 4.644e+09 | 0.059 | 0.544 | 1.29 |
| alt-position | SCA | 149608 | 4.803e+09 | 2.511e+09 | 0.523 | 7.817e+10 | 4.709e+09 | 0.060 | 0.533 | 1.27 |
| alt-position | CMT | 6263 | 2.455e+08 | 6.732e+07 | 0.274 | 5.250e+09 | 1.621e+08 | 0.031 | 0.415 | 1.71 |
| alt-position | city | 51874 | 1.347e+09 | 2.770e+08 | 0.206 | 2.014e+10 | 1.588e+08 | 0.008 | 1.74 | 1.71 |

#### absolute eps = 1e-4

| class | family | moves | sum Nq_sa | sum Nq | Nq/Nq_sa | sum Nc_sa | sum Nc | Nc/Nc_sa | Nq/Nc (sums) | median per-move Nq/Nc |
|---|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 4.325e+10 | 3.870e+10 | 0.895 | 7.919e+12 | 6.093e+12 | 0.769 | 0.00635 | 0.00796 |
| insertion | SCA | 149612 | 4.240e+10 | 3.780e+10 | 0.891 | 7.820e+12 | 5.991e+12 | 0.766 | 0.00631 | 0.00787 |
| insertion | CMT | 6249 | 2.022e+09 | 1.057e+09 | 0.523 | 5.235e+11 | 9.691e+10 | 0.185 | 0.0109 | 0.013 |
| insertion | city | 51882 | 1.313e+10 | 8.955e+09 | 0.682 | 2.015e+12 | 8.187e+11 | 0.406 | 0.0109 | 0.0126 |
| intra | CON | 36355 | 9.199e+09 | 4.014e+09 | 0.436 | 1.406e+12 | 6.248e+10 | 0.044 | 0.0643 | 0.173 |
| intra | SCA | 36929 | 9.298e+09 | 4.095e+09 | 0.440 | 1.423e+12 | 6.359e+10 | 0.045 | 0.0644 | 0.167 |
| intra | CMT | 12603 | 4.057e+09 | 9.866e+08 | 0.243 | 1.109e+12 | 2.850e+10 | 0.026 | 0.0346 | 0.683 |
| intra | city | 65397 | 1.543e+10 | 1.910e+09 | 0.124 | 3.123e+12 | 6.233e+09 | 0.002 | 0.306 | 0.683 |
| alt-position | CON | 152296 | 4.325e+10 | 2.285e+10 | 0.528 | 7.920e+12 | 4.557e+11 | 0.058 | 0.0501 | 0.107 |
| alt-position | SCA | 149608 | 4.240e+10 | 2.275e+10 | 0.537 | 7.817e+12 | 4.624e+11 | 0.059 | 0.0492 | 0.106 |
| alt-position | CMT | 6263 | 2.027e+09 | 5.185e+08 | 0.256 | 5.250e+11 | 1.504e+10 | 0.029 | 0.0345 | 0.683 |
| alt-position | city | 51874 | 1.313e+10 | 1.247e+09 | 0.095 | 2.014e+12 | 2.494e+09 | 0.001 | 0.5 | 0.683 |

#### relative eps = 0.1 p'

| class | family | moves | sum Nq_sa | sum Nq | Nq/Nq_sa | sum Nc_sa | sum Nc | Nc/Nc_sa | Nq/Nc (sums) | median per-move Nq/Nc |
|---|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 3.950e+08 | 3.389e+08 | 0.858 | 5.835e+08 | 4.161e+08 | 0.713 | 0.815 | 1.26 |
| insertion | SCA | 149612 | 3.875e+08 | 3.299e+08 | 0.851 | 5.722e+08 | 4.034e+08 | 0.705 | 0.818 | 1.26 |
| insertion | CMT | 6249 | 9.653e+06 | 4.441e+06 | 0.460 | 6.467e+06 | 2.073e+06 | 0.321 | 2.14 | 5.07 |
| insertion | city | 51882 | 1.589e+08 | 1.139e+08 | 0.716 | 2.404e+08 | 1.074e+08 | 0.447 | 1.06 | 1.18 |
| intra | CON | 36355 | 1.212e+08 | 8.406e+07 | 0.694 | 2.180e+08 | 1.984e+07 | 0.091 | 4.24 | 5.65 |
| intra | SCA | 36929 | 1.243e+08 | 8.685e+07 | 0.699 | 2.251e+08 | 2.069e+07 | 0.092 | 4.2 | 5.53 |
| intra | CMT | 12603 | 1.797e+07 | 1.037e+07 | 0.577 | 1.054e+07 | 1.435e+06 | 0.136 | 7.23 | 9.89 |
| intra | city | 65397 | 2.256e+08 | 1.135e+08 | 0.503 | 3.971e+08 | 3.297e+07 | 0.083 | 3.44 | 3.74 |
| alt-position | CON | 152296 | 3.949e+08 | 2.908e+08 | 0.736 | 5.831e+08 | 5.712e+07 | 0.098 | 5.09 | 7.42 |
| alt-position | SCA | 149608 | 3.875e+08 | 2.872e+08 | 0.741 | 5.724e+08 | 5.678e+07 | 0.099 | 5.06 | 7.45 |
| alt-position | CMT | 6263 | 9.668e+06 | 5.678e+06 | 0.587 | 6.457e+06 | 7.813e+05 | 0.121 | 7.27 | 10.1 |
| alt-position | city | 51874 | 1.589e+08 | 7.697e+07 | 0.484 | 2.404e+08 | 2.033e+07 | 0.085 | 3.79 | 4.04 |

#### Reading of Section 3b

Unit costs and what the ratios mean. Nq counts applications of the amplitude-estimation unitary A. In the circuit one application of a stand-alone peak oracle is one coherent oracle call with T(L, c, m) Toffolis (DESIGN_M2.md, Section 'Toffoli count'). A difference oracle evaluates two route predicates on one scenario register, so one of its applications costs more Toffolis than a stand-alone one; the tables do not convert counts to Toffolis. Nc counts classical scenario evaluations, each a full route evaluation. A ratio Nq/Nc below 1 therefore says that the quantum count is smaller; it is not a runtime ratio.

Conservativeness of the quantum count. The count uses canonical amplitude estimation with the bound 2 pi sqrt(v)/M + pi^2/M^2, which holds with probability at least 8/pi^2, and R = 5 repetitions with the median (failure probability about 0.05, so about 95 % confidence). The bound is a worst case over the position of the phase on the grid. In a check on the exact QPE distribution at the same M, the 81 % quantile of the actual error was smaller than the bound by factors of 1.6, 2.0 and 4.6 in the three non-aligned cases checked (a = 0.05 with eps = 1e-2; a = 0.2 with eps = 1e-3; a = 0.2 with eps = 1e-2), and by much more when the phase falls on the grid. The factor depends on alignment and is not a general constant: the quantum counts here are conservative by 1.6-4.6x in the three cases checked, and no larger factor is claimed.

Absolute eps = 1e-3 (classical certified). Quantum-to-classical ratios of the sums: insertions 0.069 to 0.11 across families, intra moves 0.42 to 1.7, alternative-position moves 0.42 to 1.7. The city cells, which the CLT count made 7 to 19 times worse in the previous table, are now 1.7 (alternative) and 1.7 (intra). The city ratio is a ratio of two constants of the floor regime, not a property of the problem: a move with no event in the sample costs 2 R (2M + 1) = 5130 applications at eps = 1e-3 (R = 5, M = 256, two estimates), against the classical floor 3 / eps = 3000, and 5130 / 3000 = 1.71. At eps = 1e-4 the same count is 20490 against 30000 (0.68). Floor moves are a large share of city moves (Var(D) = 0 share 0.53 of city intra and 0.63 of city alternative moves), so the city cells mostly measure the floor. The honest statement is comparable cost at 1e-3 (the floor constants set the ratio) and a lower quantum count at 1e-4 in every class and family of this table, not a quantum penalty. At eps = 1e-4 the ratios are 0.0063 to 0.011 (insertions), 0.035 to 0.31 (intra) and 0.034 to 0.5 (alternative).

Within-paradigm CRN gain. Insertions gain little from CRN in the quantum count (Nq/Nq_sa 0.86 on CON) and in the classical count (Nc/Nc_sa 0.77). Signed moves gain a lot in both: quantum Nq/Nq_sa 0.44 and classical Nc/Nc_sa 0.047 on CON intra moves, because the stand-alone variance is much larger than Var(D).

Relative precision (eps = 0.1 p'). Here the classical direct estimator is cheaper on signed moves: Nq/Nc is 3.44 to 7.27 over the signed (intra and alternative-position) cells, so the quantum count is that many times the classical count in these sums (the unit costs differ, see above). The mechanism: a quantum count grows like sqrt(variance)/eps and a classical count like variance/eps^2, so the classical estimator wins when eps is large compared with the standard deviation of the indicators. Here eps_i = 0.1 p'_i is relative to the stand-alone p'_i of each move, not to E[D]. At eps = 0.1 p' this holds for the signed moves here, while at absolute eps = 1e-4 (where eps is small against those deviations) the quantum count is the lower one. Insertions are the exception in the relative block: Nq/Nc is 0.81 to 2.1 there. Recommendation: the absolute blocks (1e-3 and 1e-4) are the main comparison; the relative block answers a different question (precision relative to p'), and its signed-move ratios favour the classical direct estimator. Quote it only with that label.

### 3c. Convention for one-sided moves (classes 2-3, absolute eps = 1e-3)

Reading. The single-estimate convention lowers the quantum count of one-sided moves by the share shown in the table (the classical direct count is the same). It does not change any conclusion of 3b: the floor-dominated city cells stay near 1.4 to 1.7 at 1e-3 and the CON/SCA/CMT cells stay below 1 at 1e-3. The main convention is the one used in 3b.

Main tables charge a one-sided move (only one of P(D = 1), P(D = -1) is nonzero) two estimates at eps/2, the zero side at the rule-of-three floor. The M3b convention estimates only the nonzero side, at eps. Both are shown; the classical direct count is the same in both.

| class | family | moves | one-sided moves | sum Nq (main convention) | sum Nq (single nonzero-side estimate) | Nq/Nc (main) | Nq/Nc (single) |
|---|---|---|---|---|---|---|---|
| intra | CON | 36355 | 6809 | 4.636e+08 | 4.272e+08 | 0.709 | 0.653 |
| intra | SCA | 36929 | 6743 | 4.763e+08 | 4.402e+08 | 0.716 | 0.662 |
| intra | CMT | 12603 | 3794 | 1.302e+08 | 9.493e+07 | 0.421 | 0.307 |
| intra | city | 65397 | 18302 | 3.750e+08 | 3.030e+08 | 1.71 | 1.38 |
| alt-position | CON | 152296 | 24570 | 2.527e+09 | 2.373e+09 | 0.544 | 0.511 |
| alt-position | SCA | 149608 | 24338 | 2.511e+09 | 2.355e+09 | 0.533 | 0.5 |
| alt-position | CMT | 6263 | 2733 | 6.732e+07 | 4.160e+07 | 0.415 | 0.257 |
| alt-position | city | 51874 | 13099 | 2.770e+08 | 2.271e+08 | 1.74 | 1.43 |

## 4. Discretisation without quadrature error, and the quadrature study

### 4a. Paired level-only check (same continuous copula scenarios, nearest-level binning)

Bias = P(F_level > K) - P(F > Q) on the same N scenarios (no quadrature anywhere in this check); paired SE from the scenario-wise difference of the two indicators. Routes: up to 6 per family x base-p stratum (seeded). Relative bias only where P(F > Q) >= 0.01.

Routes per family x stratum (base p): CON: 0.005-0.05 6, 0.05-0.1 6, 0.1-0.3 6, >0.3 6, p<0.005 6; SCA: 0.005-0.05 6, 0.05-0.1 6, 0.1-0.3 6, >0.3 6, p<0.005 6; CMT: 0.005-0.05 2, 0.05-0.1 2, 0.1-0.3 5, >0.3 6, p<0.005 2; city: 0.005-0.05 6, 0.05-0.1 4, 0.1-0.3 6, >0.3 6, p<0.005 6. Total routes 105.

| rule | b | routes | median abs bias | max abs bias | median paired SE | median abs rel bias (p >= 0.01) | max abs rel bias (p >= 0.01) |
|---|---|---|---|---|---|---|---|
| design | 4 | 105 | 2.575e-03 | 3.728e-02 | 2.835e-04 | 0.0287 | 0.359 |
| design | 5 | 105 | 4.550e-04 | 1.184e-02 | 1.978e-04 | 0.00549 | 0.0842 |
| design | 6 | 105 | 1.250e-04 | 6.710e-03 | 1.389e-04 | 0.00133 | 0.0348 |
| nearest | 4 | 105 | 2.570e-03 | 3.728e-02 | 2.868e-04 | 0.0293 | 0.48 |
| nearest | 5 | 105 | 5.150e-04 | 1.184e-02 | 1.978e-04 | 0.00587 | 0.0842 |
| nearest | 6 | 105 | 1.150e-04 | 6.710e-03 | 1.389e-04 | 0.00117 | 0.0348 |

By family (nearest rule):

| family | b | routes | median abs bias | max abs bias | median abs rel bias (p >= 0.01) | max abs rel bias (p >= 0.01) |
|---|---|---|---|---|---|---|
| CON | 4 | 30 | 2.723e-03 | 3.728e-02 | 0.05 | 0.344 |
| CON | 5 | 30 | 6.100e-04 | 1.107e-02 | 0.00793 | 0.0735 |
| CON | 6 | 30 | 1.000e-04 | 4.360e-03 | 0.00161 | 0.0344 |
| SCA | 4 | 30 | 1.885e-03 | 3.277e-02 | 0.0222 | 0.48 |
| SCA | 5 | 30 | 5.325e-04 | 1.184e-02 | 0.00712 | 0.0842 |
| SCA | 6 | 30 | 1.825e-04 | 6.710e-03 | 0.00159 | 0.0348 |
| CMT | 4 | 17 | 4.135e-03 | 1.640e-02 | 0.0269 | 0.148 |
| CMT | 5 | 17 | 5.150e-04 | 9.080e-03 | 0.00711 | 0.0201 |
| CMT | 6 | 17 | 1.150e-04 | 5.800e-04 | 0.000428 | 0.00414 |
| city | 4 | 28 | 1.115e-03 | 2.141e-02 | 0.0246 | 0.126 |
| city | 5 | 28 | 3.475e-04 | 3.930e-03 | 0.0038 | 0.0328 |
| city | 6 | 28 | 9.000e-05 | 4.700e-04 | 0.000962 | 0.011 |

By route length (nearest rule; bands of the route length L, the number of customers on the route):

| b | L band | routes | median abs bias | max abs bias | median abs rel bias (p >= 0.01) | max abs rel bias (p >= 0.01) |
|---|---|---|---|---|---|---|
| 4 | <=8 | 53 | 1.700e-03 | 3.728e-02 | 0.0396 | 0.48 |
| 4 | 9-15 | 30 | 2.263e-03 | 1.640e-02 | 0.0305 | 0.154 |
| 4 | 16-25 | 21 | 7.950e-03 | 2.141e-02 | 0.0225 | 0.148 |
| 4 | >25 | 1 | 4.440e-03 | 4.440e-03 | 0.017 | 0.017 |
| 5 | <=8 | 53 | 4.850e-04 | 1.184e-02 | 0.00811 | 0.0842 |
| 5 | 9-15 | 30 | 4.475e-04 | 5.045e-03 | 0.00845 | 0.0328 |
| 5 | 16-25 | 21 | 7.950e-04 | 9.080e-03 | 0.00478 | 0.0201 |
| 5 | >25 | 1 | 7.000e-05 | 7.000e-05 | 0.000268 | 0.000268 |
| 6 | <=8 | 53 | 7.500e-05 | 6.710e-03 | 0.00157 | 0.0348 |
| 6 | 9-15 | 30 | 8.500e-05 | 1.860e-03 | 0.00158 | 0.00875 |
| 6 | 16-25 | 21 | 1.950e-04 | 1.515e-03 | 0.000797 | 0.011 |
| 6 | >25 | 1 | 1.050e-04 | 1.050e-04 | 0.000403 | 0.000403 |

Level rule (a choice, stated as such): median |rel bias| <= 1% and max |rel bias| <= 5% (p >= 0.01). Under this rule the recommended b is 6. The rule is met at b = 6 (median 0.12%, max 3.48%). At b = 5 the median meets the rule (0.59%) but the maximum does not (8.42%; 2 route(s) above 5 %, of length L = 5, 8). The worst relative cases are short routes with few large demands; the relative bias falls with L, while the median absolute bias is larger on the 16-25 band than on the short routes (see the length table). At b = 6 the largest absolute bias is 6.71e-03, so the level error is small in relative terms but not below 1e-3 absolute on every route. Cost of b = 6: 6 qubits per customer and per demand type (12 per customer for delivery and pickup), against 5 (10) at b = 5.

### 4b. Quadrature of the copula window probability

Method. The window probability P(F_r > Q) is the exact double sum over factor values sum_t sum_s w_t w_s P(F > Q | z_t, z_s), computed by the batched exact recursion (no Monte Carlo inside the quadrature; validated against classical.chance). Three factor grids per copula factor, with the same nine routes as before (three longest, six of mixed length, seeded; base p in (0.05, 0.95) for the mixed ones):
- GH n: Gauss-Hermite nodes, n = 16, 32, 64, 128.
- EMm 2^f: equal-probability bins, conditional pmf at the bin midpoint (the Revision 3 grid).
- EMa 2^f: equal-probability bins, conditional pmf averaged over each bin, P(l | t) = 2^f integral over bin t of P(l | z) phi(z) dz, with 32-point Gauss-Legendre in z per bin (outer bins cut at |z| = 9). The mixture over bins reproduces the marginal level pmf to about 1e-15 (unit test).
f = 3..7 at b = 5 and f = 3..5 at b = 6 (EMa 64 at b = 6 was not run for all routes: about four times the cost of EMa 32 per route, too slow for the time available). Gauss-Hermite and EMm at b = 6 are as before (up to 64).
Reference. No grid is the reference. For each route and b the two converging sequences are shown side by side: the largest Gauss-Hermite grid (GH 128 at b = 5, GH 64 at b = 6) and the largest bin-averaged grid (EMa 128 at b = 5, EMa 32 at b = 6). Their disagreement is the quadrature uncertainty indicator. Continuous Monte Carlo is the same N = 200000 common scenarios as the rest of M3a; it contains the level error of Section 4a and is not a quadrature check.

Table 4b-1, b = 5: absolute probability of each configuration per route (6 decimals; '-' not computed).

| route | family | L | p_MC (SE) | GH16 | GH32 | GH64 | GH128 | EMa8 | EMa16 | EMa32 | EMa64 | EMa128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CMT11X r2 | CMT | 28 | 0.6315 (0.0011) | 0.633211 | 0.633728 | 0.632329 | 0.631561 | 0.632399 | 0.631810 | 0.631623 | 0.631576 | 0.631565 |
| HCMC-100-3 r11 | city | 27 | 0.4606 (0.0011) | 0.487908 | 0.477270 | 0.473105 | 0.472719 | 0.472520 | 0.472391 | 0.472387 | 0.472408 | 0.472464 |
| CON3-4 r15 | CON | 22 | 0.6579 (0.0011) | 0.668088 | 0.654097 | 0.653016 | 0.653022 | 0.653887 | 0.653237 | 0.653075 | 0.653035 | 0.653025 |
| HCMC-400-1 r4 | city | 20 | 0.4616 (0.0011) | 0.480008 | 0.467847 | 0.464316 | 0.464122 | 0.464517 | 0.464209 | 0.464133 | 0.464116 | 0.464113 |
| HANOI-400-1 r32 | city | 18 | 0.4569 (0.0011) | 0.470010 | 0.457398 | 0.454790 | 0.454713 | 0.454974 | 0.454639 | 0.454557 | 0.454538 | 0.454542 |
| SCA3-1 r17 | SCA | 15 | 0.0625 (0.0005) | 0.064671 | 0.062362 | 0.062233 | 0.062231 | 0.050870 | 0.068765 | 0.066621 | 0.063481 | 0.062499 |
| SCA3-7 r7 | SCA | 14 | 0.3860 (0.0011) | 0.378284 | 0.385339 | 0.385576 | 0.385562 | 0.397628 | 0.387147 | 0.385892 | 0.385646 | 0.385583 |
| SCA8-1 r85 | SCA | 5 | 0.1694 (0.0008) | 0.164626 | 0.164652 | 0.164653 | 0.164653 | 0.168555 | 0.167357 | 0.165762 | 0.165013 | 0.164756 |
| CON8-7 r67 | CON | 3 | 0.0522 (0.0005) | 0.052857 | 0.052870 | 0.052870 | 0.052870 | 0.050548 | 0.052677 | 0.053002 | 0.053010 | 0.052982 |

Table 4b-2, b = 5: the two converging references side by side and their agreement (absolute probability; difference = GH128 minus EMa128; EMa increments show the convergence rate of the bin-averaged sequence).

| route | L | GH128 | EMa128 | GH128 - EMa128 | EMa128 - EMa64 | EMa64 - EMa32 |
|---|---|---|---|---|---|---|
| CMT11X r2 | 28 | 0.631561 | 0.631565 | -3.92e-06 | -1.17e-05 | -4.68e-05 |
| HCMC-100-3 r11 | 27 | 0.472719 | 0.472464 | +2.56e-04 | +5.59e-05 | +2.09e-05 |
| CON3-4 r15 | 22 | 0.653022 | 0.653025 | -3.41e-06 | -1.00e-05 | -4.02e-05 |
| HCMC-400-1 r4 | 20 | 0.464122 | 0.464113 | +9.42e-06 | -3.25e-06 | -1.75e-05 |
| HANOI-400-1 r32 | 18 | 0.454713 | 0.454542 | +1.71e-04 | +3.76e-06 | -1.90e-05 |
| SCA3-1 r17 | 15 | 0.062231 | 0.062499 | -2.68e-04 | -9.82e-04 | -3.14e-03 |
| SCA3-7 r7 | 14 | 0.385562 | 0.385583 | -2.10e-05 | -6.32e-05 | -2.46e-04 |
| SCA8-1 r85 | 5 | 0.164653 | 0.164756 | -1.03e-04 | -2.57e-04 | -7.49e-04 |
| CON8-7 r67 | 3 | 0.052870 | 0.052982 | -1.12e-04 | -2.82e-05 | +7.74e-06 |

Table 4b-1, b = 6: absolute probability of each configuration per route (6 decimals; '-' not computed).

| route | family | L | p_MC (SE) | GH16 | GH32 | GH64 | EMa8 | EMa16 | EMa32 |
|---|---|---|---|---|---|---|---|---|---|
| CMT11X r2 | CMT | 28 | 0.6315 (0.0011) | 0.635497 | 0.636087 | 0.635009 | 0.634703 | 0.634197 | 0.634021 |
| HCMC-100-3 r11 | city | 27 | 0.4606 (0.0011) | 0.481468 | 0.464422 | 0.457609 | 0.456920 | 0.456649 | 0.456609 |
| CON3-4 r15 | CON | 22 | 0.6579 (0.0011) | 0.676549 | 0.661463 | 0.659993 | 0.660778 | 0.660185 | 0.660037 |
| HCMC-400-1 r4 | city | 20 | 0.4616 (0.0011) | 0.480903 | 0.466993 | 0.462289 | 0.462333 | 0.462036 | 0.461963 |
| HANOI-400-1 r32 | city | 18 | 0.4569 (0.0011) | 0.472657 | 0.460178 | 0.457405 | 0.457562 | 0.457243 | 0.457165 |
| SCA3-1 r17 | SCA | 15 | 0.0625 (0.0005) | 0.065038 | 0.062780 | 0.062665 | 0.051439 | 0.069349 | 0.067243 |
| SCA3-7 r7 | SCA | 14 | 0.3860 (0.0011) | 0.377596 | 0.385130 | 0.385370 | 0.397566 | 0.386871 | 0.385679 |
| SCA8-1 r85 | SCA | 5 | 0.1694 (0.0008) | 0.166175 | 0.166186 | 0.166186 | 0.170120 | 0.168898 | 0.167316 |
| CON8-7 r67 | CON | 3 | 0.0522 (0.0005) | 0.052624 | 0.052636 | 0.052636 | 0.050307 | 0.052445 | 0.052767 |

Table 4b-2, b = 6: the two converging references side by side and their agreement (absolute probability; difference = GH64 minus EMa32; EMa increments show the convergence rate of the bin-averaged sequence).

| route | L | GH64 | EMa32 | GH64 - EMa32 | EMa32 - EMa16 | EMa16 - EMa8 |
|---|---|---|---|---|---|---|
| CMT11X r2 | 28 | 0.635009 | 0.634021 | +9.88e-04 | -1.77e-04 | -5.06e-04 |
| HCMC-100-3 r11 | 27 | 0.457609 | 0.456609 | +1.00e-03 | -4.06e-05 | -2.71e-04 |
| CON3-4 r15 | 22 | 0.659993 | 0.660037 | -4.41e-05 | -1.48e-04 | -5.93e-04 |
| HCMC-400-1 r4 | 20 | 0.462289 | 0.461963 | +3.26e-04 | -7.35e-05 | -2.96e-04 |
| HANOI-400-1 r32 | 18 | 0.457405 | 0.457165 | +2.40e-04 | -7.86e-05 | -3.19e-04 |
| SCA3-1 r17 | 15 | 0.062665 | 0.067243 | -4.58e-03 | -2.11e-03 | +1.79e-02 |
| SCA3-7 r7 | 14 | 0.385370 | 0.385679 | -3.09e-04 | -1.19e-03 | -1.07e-02 |
| SCA8-1 r85 | 5 | 0.166186 | 0.167316 | -1.13e-03 | -1.58e-03 | -1.22e-03 |
| CON8-7 r67 | 3 | 0.052636 | 0.052767 | -1.31e-04 | +3.22e-04 | +2.14e-03 |

Reading of the two references (b = 5). (i) Short, low-probability routes (CON8-7, SCA8-1, SCA3-1): the Gauss-Hermite sequence is stable from 32 nodes (GH 64 and GH 128 differ by 2e-6 on SCA3-1 and SCA8-1, and agree on CON8-7), while the bin-averaged sequence is not yet stable at 128 bins (EMa 128 minus GH 128 is +1.1e-4 on CON8-7; the EMa increments on SCA3-1 are -3.1e-3 then -9.8e-4). The earlier attribution of the short-route discrepancy to Gauss-Hermite is withdrawn: the Gauss-Hermite values are the converged ones there, and the bin-averaged sequence is the open item. (ii) Long routes (L >= 16): Gauss-Hermite 64 is not converged (GH 64 minus GH 128 is 7.7e-4 on CMT11X and 3.9e-4 on HCMC-100-3), while the bin-averaged grid is within 1e-4 of EMa 128 from 32 to 64 bins. The two references agree within 1e-5 to 1e-4 on CMT11X, CON3-4, HCMC-400-1 and HANOI-400-1, and disagree by 2.6e-4 on HCMC-100-3, which neither grid resolves. (iii) Heuristic, not established by the data: the conditional overflow step has width about sqrt((1 - rho) / (rho L)) (0.17 at L = 22, 0.15 at L = 28 for rho = 0.6), which is narrower than the Gauss-Hermite bulk spacing pi / sqrt(n) at 64 nodes (0.39), so the long-route Gauss-Hermite error is consistent with the step being unresolved; the same argument does not explain the short routes, where the step is wider.

Recommended bin count for the bin-averaged grid: f = 6 (64 bins) for long routes (L >= 16). On the five long routes, 32 bins are already within 7.7e-5 of EMa 128 (HCMC-100-3, the largest) and 64 bins within 5.6e-5 (HCMC-100-3 again), so f = 5 is the smallest grid that passes the 1e-4 tolerance there, and f = 6 lowers the largest gap; f = 6 is the recommendation for the register. The rule table above reports the smallest bin count within tolerance on all nine routes, which is not reached on the short low-probability routes (SCA3-1 and SCA8-1 are 9.8e-4 and 2.6e-4 from EMa 128 at 64 bins and still moving at 128 bins). For those routes Gauss-Hermite 64 to 128 is the converged rule, and no bin-averaged grid is recommended on the evidence here.\n
Recommendation of the bin count (b = 5, reference EMa 128, tolerance 0.0001):

| bin count 2^f | max |EMa - EMa128| all 9 routes | short routes (L <= 8) | long routes (L >= 16) |
|---|---|---|---|
| 2^3 = 8 | 1.20e-02 | 3.80e-03 | 8.62e-04 |
| 2^4 = 16 | 6.27e-03 | 2.60e-03 | 2.46e-04 |
| 2^5 = 32 | 4.12e-03 | 1.01e-03 | 7.68e-05 |
| 2^6 = 64 | 9.82e-04 | 2.57e-04 | 5.59e-05 |
| 2^7 = 128 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

Largest gap between the two references at b = 5 (GH 128 and EMa 128) over the nine routes: 2.68e-04. Within 0.0001 of EMa 128 on all nine routes, the smallest bin count is 2^7 = 128, but 2^7 is the reference itself, so this is not a convergence check: on the short low-probability routes the 2^6 deviation is 2.6e-4 and EMa 128 is still moving, so no bin count up to 2^6 is shown to meet the tolerance there.

### 4c. Factor-grid error on the CRN quantities P(D = +1), P(D = -1)

Method. For three routes (short, medium, long) and 6 intra and 6 alternative-position moves each (seeded, among moves with P(D != 0) > 0.001 in the continuous MC), P(D = +1) and P(D = -1) are computed in the discrete copula model at b = 5 on the grids EMa 8..64 and GH 64 and 128. The exact identity P(D = +1) = sum_ts w_t w_s P(D = +1 | t, s) holds on the discrete model, but no exact conditional computation of the joint of two routes is available in reasonable time, so each factor pair (t, s) is a conditional Monte Carlo with K = 4000 common draws (customers independent given the pair, pmfs of the discrete model). Both routes of a pair use the same draws. The standard error of each grid sum is written in Table 4c-2. The per-pair budget is K = 4000, not the 2e5 of the scenario MC, because the grids contain up to 16384 pairs per route. Per draw, P(D = +1) - P(D = -1) = P(F_b > Q) - P(F_a > Q) exactly, and the tail p_base of each base route is checked against the exact DP (chance_copula) in Table 4c-1.

Table 4c-1: per route and grid, the largest |deviation| over the 12 moves of P(D = +1) and P(D = -1), against the two converging references GH 128 and EMa 64 (the largest bin-averaged grid computed for every route); the base tail is the exact DP minus the conditional MC, in units of its standard error.

| route | role | L | grid | max |dP(D=+1)| vs GH128 | max |dP(D=-1)| vs GH128 | max |dP(D=+1)| vs EMa64 | max |dP(D=-1)| vs EMa64 | median SE of P(D=+1) | base p exact - MC (SE units) |
|---|---|---|---|---|---|---|---|---|---|
| CON8-7 r67 | short | 3 | EMa8 | 1.15e-03 | 7.99e-04 | 7.88e-04 | 5.82e-04 | 7.9e-05 | +0.25 |
| CON8-7 r67 | short | 3 | EMa16 | 6.86e-04 | 7.15e-04 | 3.23e-04 | 3.62e-04 | 3.8e-05 | +0.07 |
| CON8-7 r67 | short | 3 | EMa32 | 4.01e-04 | 3.49e-04 | 4.68e-05 | 7.92e-05 | 1.9e-05 | -0.02 |
| CON8-7 r67 | short | 3 | EMa64 | 3.63e-04 | 3.52e-04 | 0.00e+00 | 0.00e+00 | 9.3e-06 | -0.00 |
| CON8-7 r67 | short | 3 | GH64 | 4.64e-04 | 7.89e-04 | 3.26e-04 | 4.37e-04 | 6.0e-05 | -0.00 |
| CON8-7 r67 | short | 3 | GH128 | 0.00e+00 | 0.00e+00 | 3.63e-04 | 3.52e-04 | 4.3e-05 | +0.02 |
| SCA3-7 r7 | medium | 14 | EMa8 | 2.18e-03 | 4.97e-04 | 1.85e-03 | 4.78e-04 | 5.0e-05 | +0.15 |
| SCA3-7 r7 | medium | 14 | EMa16 | 5.01e-04 | 1.05e-04 | 1.70e-04 | 1.10e-04 | 2.4e-05 | +0.02 |
| SCA3-7 r7 | medium | 14 | EMa32 | 3.56e-04 | 9.56e-05 | 3.17e-05 | 5.58e-05 | 1.2e-05 | -0.01 |
| SCA3-7 r7 | medium | 14 | EMa64 | 3.31e-04 | 1.20e-04 | 0.00e+00 | 0.00e+00 | 6.1e-06 | +0.00 |
| SCA3-7 r7 | medium | 14 | GH64 | 3.28e-04 | 2.85e-04 | 3.58e-04 | 4.05e-04 | 5.3e-05 | -0.02 |
| SCA3-7 r7 | medium | 14 | GH128 | 0.00e+00 | 0.00e+00 | 3.31e-04 | 1.20e-04 | 3.7e-05 | +0.00 |
| CMT11X r2 | long | 28 | EMa8 | 1.96e-04 | 1.44e-04 | 8.53e-05 | 1.20e-04 | 3.8e-05 | -0.05 |
| CMT11X r2 | long | 28 | EMa16 | 1.40e-04 | 2.21e-04 | 1.28e-04 | 9.42e-05 | 2.0e-05 | -0.01 |
| CMT11X r2 | long | 28 | EMa32 | 1.34e-04 | 1.90e-04 | 6.39e-05 | 9.56e-05 | 9.9e-06 | -0.01 |
| CMT11X r2 | long | 28 | EMa64 | 1.77e-04 | 1.54e-04 | 0.00e+00 | 0.00e+00 | 4.9e-06 | -0.00 |
| CMT11X r2 | long | 28 | GH64 | 2.43e-04 | 1.31e-04 | 2.76e-04 | 1.79e-04 | 4.3e-05 | -0.06 |
| CMT11X r2 | long | 28 | GH128 | 0.00e+00 | 0.00e+00 | 1.77e-04 | 1.54e-04 | 3.3e-05 | -0.09 |

Table 4c-2: per move, P(D = +1) and P(D = -1) on each grid (6 decimals; standard error of the largest grid in brackets; the continuous MC of M3a for the same move in the last column, P(D=+1) / P(D=-1)).

Route short: CON8-7 r67 (L = 3).

| move | class | kind | +1 EMa8 | +1 EMa16 | +1 EMa32 | +1 EMa64 | +1 GH64 | +1 GH128 | -1 EMa8 | -1 EMa16 | -1 EMa32 | -1 EMa64 | -1 GH64 | -1 GH128 | continuous MC +1 / -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | relocate | 0.004020 | 0.003680 | 0.003553 | 0.003544 | 0.003572 | 0.003623 | 0.000078 | 0.000038 | 0.000031 | 0.000030 | 0.000032 | 0.000032 | 0.003695 / 0.000010 |
| 2 | 2 | relocate | 0.005254 | 0.004845 | 0.004635 | 0.004589 | 0.004614 | 0.004654 | 0.000707 | 0.000609 | 0.000572 | 0.000548 | 0.000489 | 0.000553 | 0.004510 / 0.000455 |
| 3 | 2 | swap | 0.001820 | 0.001654 | 0.001557 | 0.001519 | 0.001498 | 0.001484 | 0.000949 | 0.000854 | 0.000822 | 0.000787 | 0.000720 | 0.000772 | 0.001165 / 0.000695 |
| 4 | 2 | relocate | 0.000758 | 0.000632 | 0.000535 | 0.000514 | 0.000507 | 0.000510 | 0.000945 | 0.000858 | 0.000831 | 0.000797 | 0.000726 | 0.000780 | 0.000360 / 0.000695 |
| 5 | 3 | alt-position | 0.000145 | 0.000118 | 0.000119 | 0.000123 | 0.000138 | 0.000145 | 0.024094 | 0.024178 | 0.024544 | 0.024540 | 0.024103 | 0.024892 | 0.000110 / 0.025255 |
| 6 | 3 | alt-position | 0.000809 | 0.000734 | 0.000664 | 0.000676 | 0.000696 | 0.000676 | 0.012918 | 0.012438 | 0.012444 | 0.012523 | 0.012396 | 0.012567 | 0.000705 / 0.012520 |
| 7 | 3 | alt-position | 0.028645 | 0.029109 | 0.029394 | 0.029432 | 0.029331 | 0.029795 | 0.000035 | 0.000032 | 0.000030 | 0.000030 | 0.000023 | 0.000028 | 0.029900 / 0.000050 |
| 8 | 3 | alt-position | 0.000855 | 0.000859 | 0.000734 | 0.000700 | 0.000718 | 0.000676 | 0.002703 | 0.002293 | 0.002123 | 0.002121 | 0.002224 | 0.002044 | 0.000685 / 0.002265 |
| 9 | 3 | alt-position | 0.007402 | 0.007396 | 0.007468 | 0.007470 | 0.007144 | 0.007534 | 0.000285 | 0.000167 | 0.000180 | 0.000187 | 0.000168 | 0.000191 | 0.007580 / 0.000165 |
| 10 | 3 | alt-position | 0.001473 | 0.001427 | 0.001368 | 0.001370 | 0.001355 | 0.001432 | 0.000840 | 0.000889 | 0.000910 | 0.000899 | 0.000882 | 0.000860 | 0.001485 / 0.000895 |

Route medium: SCA3-7 r7 (L = 14).

| move | class | kind | +1 EMa8 | +1 EMa16 | +1 EMa32 | +1 EMa64 | +1 GH64 | +1 GH128 | -1 EMa8 | -1 EMa16 | -1 EMa32 | -1 EMa64 | -1 GH64 | -1 GH128 | continuous MC +1 / -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | swap | 0.001066 | 0.001342 | 0.001393 | 0.001388 | 0.001459 | 0.001482 | 0.000645 | 0.000758 | 0.000812 | 0.000803 | 0.000747 | 0.000835 | 0.001205 / 0.000645 |
| 2 | 2 | relocate | 0.000207 | 0.000271 | 0.000250 | 0.000247 | 0.000260 | 0.000295 | 0.001277 | 0.001396 | 0.001496 | 0.001441 | 0.001361 | 0.001497 | 0.000140 / 0.001025 |
| 3 | 2 | relocate | 0.016797 | 0.018478 | 0.018623 | 0.018648 | 0.018735 | 0.018979 | 0.000000 | 0.000002 | 0.000002 | 0.000002 | 0.000003 | 0.000000 | 0.017730 / 0.000005 |
| 4 | 2 | 2opt | 0.000168 | 0.000176 | 0.000192 | 0.000169 | 0.000165 | 0.000194 | 0.001840 | 0.002256 | 0.002329 | 0.002318 | 0.002242 | 0.002337 | 0.000170 / 0.002250 |
| 5 | 2 | 2opt | 0.000188 | 0.000240 | 0.000229 | 0.000231 | 0.000206 | 0.000268 | 0.001465 | 0.001787 | 0.001814 | 0.001783 | 0.001722 | 0.001809 | 0.000145 / 0.001760 |
| 6 | 2 | 2opt | 0.000324 | 0.000425 | 0.000428 | 0.000424 | 0.000429 | 0.000445 | 0.001746 | 0.002007 | 0.002051 | 0.001995 | 0.001958 | 0.002095 | 0.000320 / 0.001910 |
| 7 | 3 | alt-position | 0.000363 | 0.000288 | 0.000298 | 0.000291 | 0.000258 | 0.000254 | 0.001145 | 0.001222 | 0.001288 | 0.001273 | 0.001180 | 0.001327 | 0.000305 / 0.001265 |
| 8 | 3 | alt-position | 0.003441 | 0.003440 | 0.003507 | 0.003476 | 0.003366 | 0.003588 | 0.005055 | 0.005057 | 0.005082 | 0.005036 | 0.004988 | 0.004995 | 0.003490 / 0.005015 |
| 9 | 3 | alt-position | 0.000805 | 0.000698 | 0.000715 | 0.000713 | 0.000707 | 0.000688 | 0.014184 | 0.013767 | 0.013828 | 0.013860 | 0.013456 | 0.013741 | 0.000565 / 0.014130 |
| 10 | 3 | alt-position | 0.007164 | 0.007026 | 0.007151 | 0.007171 | 0.007299 | 0.007032 | 0.001098 | 0.001016 | 0.001044 | 0.001009 | 0.001064 | 0.001116 | 0.006955 / 0.000990 |
| 11 | 3 | alt-position | 0.002941 | 0.002870 | 0.002885 | 0.002859 | 0.002501 | 0.002830 | 0.000191 | 0.000155 | 0.000149 | 0.000153 | 0.000132 | 0.000126 | 0.003175 / 0.000165 |
| 12 | 3 | alt-position | 0.000484 | 0.000508 | 0.000515 | 0.000518 | 0.000548 | 0.000504 | 0.002957 | 0.003249 | 0.003163 | 0.003139 | 0.003250 | 0.003259 | 0.000520 / 0.003320 |

Route long: CMT11X r2 (L = 28).

| move | class | kind | +1 EMa8 | +1 EMa16 | +1 EMa32 | +1 EMa64 | +1 GH64 | +1 GH128 | -1 EMa8 | -1 EMa16 | -1 EMa32 | -1 EMa64 | -1 GH64 | -1 GH128 | continuous MC +1 / -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 2opt | 0.003387 | 0.003362 | 0.003444 | 0.003425 | 0.003149 | 0.003393 | 0.002445 | 0.002403 | 0.002379 | 0.002390 | 0.002434 | 0.002420 | 0.003275 / 0.002145 |
| 2 | 2 | swap | 0.002418 | 0.002299 | 0.002406 | 0.002427 | 0.002235 | 0.002357 | 0.000246 | 0.000252 | 0.000256 | 0.000262 | 0.000271 | 0.000312 | 0.002030 / 0.000205 |
| 3 | 2 | swap | 0.000008 | 0.000008 | 0.000013 | 0.000013 | 0.000010 | 0.000011 | 0.002699 | 0.002822 | 0.002769 | 0.002794 | 0.002615 | 0.002745 | 0.000000 / 0.002595 |
| 4 | 2 | relocate | 0.000707 | 0.000770 | 0.000776 | 0.000752 | 0.000717 | 0.000832 | 0.000734 | 0.000778 | 0.000734 | 0.000726 | 0.000754 | 0.000799 | 0.000815 / 0.000740 |
| 5 | 2 | relocate | 0.004684 | 0.004605 | 0.004642 | 0.004706 | 0.004443 | 0.004639 | 0.000000 | 0.000001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.004460 / 0.000000 |
| 6 | 2 | relocate | 0.002969 | 0.002793 | 0.002820 | 0.002883 | 0.002836 | 0.002773 | 0.000000 | 0.000001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.002755 / 0.000000 |
| 7 | 3 | alt-position | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.006582 | 0.006781 | 0.006606 | 0.006702 | 0.006606 | 0.006642 | 0.000000 / 0.005910 |
| 8 | 3 | alt-position | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.001363 | 0.001428 | 0.001354 | 0.001387 | 0.001277 | 0.001353 | 0.000000 / 0.001325 |
| 9 | 3 | alt-position | 0.000000 | 0.000003 | 0.000002 | 0.000002 | 0.000000 | 0.000000 | 0.004332 | 0.004479 | 0.004449 | 0.004413 | 0.004312 | 0.004259 | 0.000000 / 0.004305 |
| 10 | 3 | alt-position | 0.002445 | 0.002426 | 0.002420 | 0.002463 | 0.002409 | 0.002286 | 0.000043 | 0.000041 | 0.000042 | 0.000037 | 0.000039 | 0.000052 | 0.002670 / 0.000045 |
| 11 | 3 | alt-position | 0.000148 | 0.000156 | 0.000153 | 0.000151 | 0.000141 | 0.000170 | 0.001316 | 0.001388 | 0.001375 | 0.001368 | 0.001305 | 0.001337 | 0.000115 / 0.001220 |
| 12 | 3 | alt-position | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.008824 | 0.008986 | 0.008866 | 0.008892 | 0.009036 | 0.008968 | 0.000000 / 0.008660 |

## 5. Alternative-position moves by position distance

Distance |pos2 - pos| (1, 2-3, >3). Median rho_q with route-bootstrap CI; Var=0 share in-sample. Decision band: p' in [0.02, 0.2].

| distance | family | n (all p') | median rho_q [CI] | p10 | p90 | Var=0 share | n (decision band) | median rho_q [CI] (decision band) | Var=0 share (decision band) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CON | 89093 | 0.124 [0.12, 0.128] | 0.0144 | 0.309 | 0.075 | 33789 | 0.111 [0.105, 0.116] | 0.048 |
| 1 | SCA | 89274 | 0.122 [0.118, 0.126] | 0.0134 | 0.317 | 0.079 | 33432 | 0.11 [0.105, 0.115] | 0.047 |
| 1 | CMT | 2131 | 0 [0, 0] | 0 | 0.137 | 0.634 | 164 | 0 [0, 0] | 0.744 |
| 1 | city | 10615 | 0 [0, 0] | 0 | 0.0183 | 0.744 | 6622 | 0 [0, 0] | 0.844 |
| 2-3 | CON | 98454 | 0.193 [0.187, 0.199] | 0.036 | 0.408 | 0.044 | 35488 | 0.162 [0.154, 0.17] | 0.020 |
| 2-3 | SCA | 98454 | 0.196 [0.19, 0.203] | 0.0348 | 0.42 | 0.048 | 35020 | 0.166 [0.159, 0.173] | 0.019 |
| 2-3 | CMT | 3910 | 0 [0, 0] | 0 | 0.243 | 0.567 | 301 | 0 [0, 0.00614] | 0.724 |
| 2-3 | city | 18333 | 0 [0, 0] | 0 | 0.0265 | 0.658 | 11365 | 0 [0, 0] | 0.785 |
| >3 | CON | 52839 | 0.176 [0.169, 0.184] | 0.0438 | 0.392 | 0.035 | 20409 | 0.15 [0.143, 0.158] | 0.010 |
| >3 | SCA | 50451 | 0.175 [0.168, 0.183] | 0.041 | 0.396 | 0.039 | 18851 | 0.152 [0.145, 0.16] | 0.009 |
| >3 | CMT | 11959 | 0.0283 [0.0134, 0.055] | 0 | 0.434 | 0.374 | 605 | 0.00615 [0, 0.0298] | 0.488 |
| >3 | city | 45460 | 0.00793 [0.00634, 0.00952] | 0 | 0.0576 | 0.429 | 24363 | 0 [0, 0] | 0.635 |

