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

Reading. rho_q is the ratio of the CRN amplitude standard deviation to the stand-alone one, so the quantum query count of the CRN difference is about rho_q times the stand-alone count. Insertions keep most of their variance (rho_q near 0.9), because an inserted customer shifts the load of the whole suffix. Intra-route and alternative-position moves reorder the same customers, so Var(D) is small: on CON/SCA the medians are 0.13-0.16, and on CMT/city they are zero in the sample for more than half of the moves. The zero is an in-sample value and is bounded by the upper-bound column.

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

Conservativeness of the quantum count. The count uses canonical amplitude estimation with the bound 2 pi sqrt(v)/M + pi^2/M^2, which holds with probability at least 8/pi^2, and R = 5 repetitions with the median (failure probability about 0.05, so about 95 % confidence). The bound is a worst case over the position of the phase on the grid. In a check on the exact QPE distribution at the same M, the 81 % quantile of the actual error was smaller than the bound by factors of 1.6, 2.0 and 4.6 in the three non-aligned cases checked (a = 0.05 with eps = 1e-2; a = 0.2 with eps = 1e-3; a = 0.2 with eps = 1e-2), and by much more when the phase falls on the grid. The factor depends on alignment; about 2x is the conservative reading, so the quantum counts here are at least about 2x conservative.

Absolute eps = 1e-3 (classical certified). Quantum-to-classical ratios of the sums: insertions 0.069 to 0.11 across families, intra moves 0.42 to 1.7, alternative-position moves 0.42 to 1.7. The city cells, which the CLT count made 7 to 19 times worse in the previous table, are now 1.7 (alternative) and 1.7 (intra). Both paradigms are floor-dominated on CMT and city moves whose Var(D) is zero in the sample (0.53 of city intra moves and 0.63 of city alternative moves), so the honest statement is comparable cost at 1e-3 and quantum ahead at 1e-4, not a quantum penalty. At eps = 1e-4 the ratios are 0.0063 to 0.011 (insertions), 0.035 to 0.31 (intra) and 0.034 to 0.5 (alternative).

Within-paradigm CRN gain. Insertions gain little from CRN in the quantum count (Nq/Nq_sa 0.86 on CON) and in the classical count (Nc/Nc_sa 0.77). Signed moves gain a lot in both: quantum Nq/Nq_sa 0.44 and classical Nc/Nc_sa 0.047 on CON intra moves, because the stand-alone variance is much larger than Var(D).

Relative precision (eps = 0.1 p'). Here the classical direct estimator is cheaper on signed moves: Nq/Nc is 3.44 to 7.27 over the signed (intra and alternative-position) cells, so the quantum count is that many times the classical count in these sums (the unit costs differ, see above). The mechanism: a quantum count grows like sqrt(variance)/eps and a classical count like variance/eps^2, so the classical estimator wins when eps is large compared with the standard deviation of the indicators. At eps = 0.1 p' this holds for the signed moves here, while at absolute eps = 1e-4 (where eps is small against those deviations) the quantum count is the lower one. Insertions are the exception in the relative block: Nq/Nc is 0.81 to 2.1 there. The relative cells must be reported next to the absolute ones; the relative signed cells alone would read as a quantum win or loss depending on which one is quoted.

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

### 4b. Quadrature: Gauss-Hermite nodes against equal-probability factor bins

Method. The copula window probability P(F_r > Q) is the exact double sum over factor values sum_t sum_s w_t w_s P(F > Q | z_t, z_s), computed without Monte Carlo (batched exact recursion, validated against classical.chance). Two factor grids per copula factor: Gauss-Hermite nodes (GH n, n = 16, 32, 64, 128) and equal-probability bins (EM 2^f, f = 3..7): bin t has probability 1/2^f and node z_t = Phi^-1((t + 1/2) / 2^f), weights 1/2^f (Hadamard-prepared register). Nine routes: the three longest (CMT11X r2, HCMC-100-3 r11, CON3-4 r15) and six of mixed length (two each with L in 3-8, 9-15, 16-25, seeded). Continuous Monte Carlo is the same N = 200000 common scenarios as the rest of M3a. Deviations are measured from the finest equal-probability grid computed for the same route and b (the reference): EM 128 for b = 5 on all nine routes; EM 64 for b = 6 on all nine routes (a 128-point b = 6 pair grid costs about 55 ms per pair, about 15 min per route on one core, and was dropped for time). The reference is not a certified limit: its convergence is shown by the 32 to 64 to 128 sequence at b = 5.

Why Gauss-Hermite fails here. Given the delivery factor z, the conditional overflow indicator is a step in z. Its transition width is about sqrt((1 - rho) / (rho L)) (the spread of the L-term sum given z shrinks relative to the shift of its mean), which is 0.174 at L = 22 and 0.154 at L = 28 (rho = 0.6). Gauss-Hermite nodes in the bulk are spaced pi / sqrt(n): 0.785 at 16 nodes, 0.555 at 32 nodes, 0.393 at 64 nodes, 0.278 at 128 nodes. At 64 nodes the spacing (0.39) is more than twice the transition width, so the step is not resolved, and the quadrature error need not decrease monotonically in n (see Table 4b-1). Equal-probability bins are dense where the mass and the step lie: the central spacing is 2.5066 / 2^f, i.e. 0.313 at 2^3 bins, 0.157 at 2^4 bins, 0.078 at 2^5 bins, 0.039 at 2^6 bins, 0.020 at 2^7 bins. The number of bins across one transition width (about 0.17) is 0.6 at 2^3, 1.1 at 2^4, 2.2 at 2^5, 4.4 at 2^6, 8.9 at 2^7. The tails are coarser, but they carry little mass.

Table 4b-1, b = 5: deviation of each grid from the reference (absolute probability; '-' where the configuration was not computed for this route):

| route (instance r, id) | family | L | p_MC (SE) | ref | GH16 - ref | GH32 - ref | GH64 - ref | GH128 - ref | EM8 - ref | EM16 - ref | EM32 - ref | EM64 - ref | EM128 - ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CMT11X r2 | CMT | 28 | 0.6315 (0.0011) | EM128 | +1.65e-03 | +2.17e-03 | +7.68e-04 | -2.68e-08 | -1.58e-04 | -1.45e-08 | +2.50e-09 | +3.00e-10 | +0.00e+00 |
| HCMC-100-3 r11 | city | 27 | 0.4606 (0.0011) | EM128 | +1.55e-02 | +4.90e-03 | +7.36e-04 | +3.50e-04 | -3.80e-04 | -2.43e-04 | -1.35e-04 | -5.88e-05 | +0.00e+00 |
| CON3-4 r15 | CON | 22 | 0.6579 (0.0011) | EM128 | +1.51e-02 | +1.08e-03 | -5.75e-06 | -1.27e-07 | +2.48e-05 | +5.64e-06 | +1.12e-06 | +1.93e-07 | +0.00e+00 |
| HCMC-400-1 r4 | city | 20 | 0.4616 (0.0011) | EM128 | +1.59e-02 | +3.74e-03 | +2.09e-04 | +1.49e-05 | -1.03e-05 | -7.29e-06 | -4.95e-06 | -2.41e-06 | +0.00e+00 |
| HANOI-400-1 r32 | city | 18 | 0.4569 (0.0011) | EM128 | +1.55e-02 | +2.87e-03 | +2.60e-04 | +1.83e-04 | -2.63e-06 | -3.44e-06 | -2.94e-06 | -2.16e-06 | +0.00e+00 |
| SCA3-1 r17 | SCA | 15 | 0.0625 (0.0005) | EM128 | +2.42e-03 | +1.09e-04 | -2.04e-05 | -2.20e-05 | -3.81e-02 | -1.01e-02 | -8.48e-04 | +4.79e-05 | +0.00e+00 |
| SCA3-7 r7 | SCA | 14 | 0.3860 (0.0011) | EM128 | -7.28e-03 | -2.23e-04 | +1.40e-05 | -8.10e-09 | +3.18e-03 | +2.02e-04 | -7.44e-06 | -3.44e-08 | +0.00e+00 |
| SCA8-1 r85 | SCA | 5 | 0.1694 (0.0008) | EM128 | -1.01e-05 | +1.63e-05 | +1.64e-05 | +1.64e-05 | -1.47e-02 | -3.48e-03 | -6.63e-04 | -9.63e-05 | +0.00e+00 |
| CON8-7 r67 | CON | 3 | 0.0522 (0.0005) | EM128 | +5.75e-04 | +5.87e-04 | +5.87e-04 | +5.87e-04 | -1.54e-02 | -6.46e-03 | -2.52e-03 | -8.00e-04 | +0.00e+00 |

Table 4b-1, b = 6: deviation of each grid from the reference (absolute probability; '-' where the configuration was not computed for this route):

| route (instance r, id) | family | L | p_MC (SE) | ref | GH16 - ref | GH32 - ref | GH64 - ref | GH128 - ref | EM8 - ref | EM16 - ref | EM32 - ref | EM64 - ref | EM128 - ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CMT11X r2 | CMT | 28 | 0.6315 (0.0011) | EM64 | +1.54e-03 | +2.13e-03 | +1.05e-03 | - | -2.04e-04 | -2.04e-08 | +2.00e-10 | +0.00e+00 | - |
| HCMC-100-3 r11 | city | 27 | 0.4606 (0.0011) | EM64 | +2.49e-02 | +7.90e-03 | +1.08e-03 | - | -3.01e-04 | -1.68e-04 | -7.03e-05 | +0.00e+00 | - |
| CON3-4 r15 | CON | 22 | 0.6579 (0.0011) | EM64 | +1.66e-02 | +1.47e-03 | +4.79e-06 | - | +1.64e-05 | +4.17e-06 | +7.41e-07 | +0.00e+00 | - |
| HCMC-400-1 r4 | city | 20 | 0.4616 (0.0011) | EM64 | +1.90e-02 | +5.06e-03 | +3.54e-04 | - | -8.44e-06 | -3.85e-06 | -1.97e-06 | +0.00e+00 | - |
| HANOI-400-1 r32 | city | 18 | 0.4569 (0.0011) | EM64 | +1.55e-02 | +3.04e-03 | +2.68e-04 | - | -4.72e-07 | -1.22e-06 | -7.51e-07 | +0.00e+00 | - |
| SCA3-1 r17 | SCA | 15 | 0.0625 (0.0005) | EM64 | +2.26e-03 | +7.25e-07 | -1.14e-04 | - | -3.84e-02 | -1.04e-02 | -7.67e-04 | +0.00e+00 | - |
| SCA3-7 r7 | SCA | 14 | 0.3860 (0.0011) | EM64 | -7.76e-03 | -2.23e-04 | +1.73e-05 | - | +3.50e-03 | +1.67e-04 | -7.47e-06 | +0.00e+00 | - |
| SCA8-1 r85 | SCA | 5 | 0.1694 (0.0008) | EM64 | +1.05e-04 | +1.17e-04 | +1.17e-04 | - | -1.45e-02 | -3.44e-03 | -5.90e-04 | +0.00e+00 | - |
| CON8-7 r67 | CON | 3 | 0.0522 (0.0005) | EM64 | +1.38e-03 | +1.39e-03 | +1.39e-03 | - | -1.46e-02 | -5.66e-03 | -1.72e-03 | +0.00e+00 | - |

Table 4b-2: summary over the routes of each b (max and median of |deviation| from the reference; routes counted are those where the configuration was computed; the last column is the largest |p(config) - p_MC| over those routes, in units of the Monte Carlo standard error):

| b | configuration | routes | max |p - ref| | median |p - ref| | max |p - p_MC| / SE_MC |
|---|---|---|---|---|---|
| 5 | GH16 | 9 | 1.590e-02 | 7.279e-03 | 24.50 |
| 5 | GH32 | 9 | 4.901e-03 | 1.076e-03 | 14.96 |
| 5 | GH64 | 9 | 7.685e-04 | 2.092e-04 | 11.22 |
| 5 | GH128 | 9 | 5.875e-04 | 1.637e-05 | 10.87 |
| 5 | EM8 | 9 | 3.807e-02 | 3.804e-04 | 70.75 |
| 5 | EM16 | 9 | 1.009e-02 | 2.015e-04 | 19.04 |
| 5 | EM32 | 9 | 2.518e-03 | 7.440e-06 | 10.44 |
| 5 | EM64 | 9 | 7.996e-04 | 2.414e-06 | 10.51 |
| 5 | EM128 | 9 | 0.000e+00 | 0.000e+00 | 10.56 |
| 6 | GH16 | 9 | 2.494e-02 | 7.757e-03 | 18.72 |
| 6 | GH32 | 9 | 7.895e-03 | 1.474e-03 | 4.82 |
| 6 | GH64 | 9 | 1.391e-03 | 2.678e-04 | 3.80 |
| 6 | EM8 | 9 | 3.838e-02 | 3.014e-04 | 70.35 |
| 6 | EM16 | 9 | 1.039e-02 | 1.670e-04 | 18.63 |
| 6 | EM32 | 9 | 1.717e-03 | 7.468e-06 | 5.35 |
| 6 | EM64 | 9 | 0.000e+00 | 0.000e+00 | 3.94 |

Rule. The smallest grid whose max |p - ref| over the reference routes is at most 0.0001: equal-probability bins: 2^3 = 8: 3.81e-02, 2^4 = 16: 1.01e-02, 2^5 = 32: 2.52e-03, 2^6 = 64: 8.00e-04; Gauss-Hermite: 16: 1.59e-02, 32: 4.90e-03, 64: 7.68e-04, 128: 5.87e-04. No equal-probability grid up to 2^6 meets the tolerance. No Gauss-Hermite grid up to 128 nodes meets it (the 128-node value is itself compared with the equal-probability reference).

Within-tolerance counts at b = 5 (routes within 0.0001 of the EM 128 reference, of 9): equal-probability bins 2^3: 3, 2^4: 4, 2^5: 5, 2^6: 8; Gauss-Hermite 128 nodes: 6. Recommended factor register: f = 6 (64 equal-probability bins per factor) as the smallest grid that is within tolerance on all but at most one route; the exception is the shortest route (L = 3), where the equal-probability sequence converges slowly (about a factor 0.3 per doubling), so that route needs more bins to meet the tolerance. Gauss-Hermite does not converge to the reference at 128 nodes within the tolerance on the city routes and the shortest route.

Caveats. (i) Node count of the reference (EM 128) is not certified; the sequence 32 to 64 to 128 gives its convergence. (ii) The tolerance 1e-4 is a choice; the table gives the full deviations. (iii) At b = 6 the reference is EM 64 on every route, so the b = 6 deviations bound convergence only to 64 bins. (iii') The Monte Carlo column of Table 4b-2 compares the quadrature with the continuous model, so it contains the level (discretisation) error of Section 4a and is not a quadrature error. (iv) The factor-grid comparison is for the window probability of a single route; the level error of Section 4a is separate and is not reduced by finer factor grids.

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

