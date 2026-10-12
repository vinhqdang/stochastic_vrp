# QUASAR M3a v3: interpretation layer on the BATON copula model

Supersedes the interpretation of m3a_v2_summary.md (its Tables B, D and E are not carried over; Table A is replaced by Section 1, which bands by p' and by distance to the gate level).
Scenarios: N = 200000 per instance, seed 7, rho = 0.6, the BATON copula; moves and routes are the cached v2 results (`m3a_v2_moves.csv.gz`, `m3a_v2_routes.csv`). Scenarios were recomputed for the end-load and level checks; the recomputed intra-move P(D = 1) and P(D = -1) agree with the cached values to a maximum absolute difference of 5.0e-06 (CSV rounding), which confirms that the draws are identical.

Definitions: p' = P(F' > Q) (new route), p = P(F > Q) (base), D = 1[F' > Q] - 1[F > Q], rho_q = sqrt(Var D) / sqrt(p'(1 - p')). Zeros in the sample are upper bounds: with no scenario showing D != 0 out of N = 200000, P(D != 0) <= 3/N = 1.5e-05 (95 %), which gives the plug-in upper bound on rho_q reported next to the medians. Route-bootstrap intervals resample routes (clusters, with replacement, 1000 replicates); the bootstrap median is the lower weighted median, the table quantiles are linear.

## 1. Decision bands of p' and of the distance to the gate level

Gate level: per-route failure target 0.10 (1 - ALPHA of dethloff_runner). Decision band p' in [0.02, 0.2]. Distances |p' - 0.10| in the second table.

### 1a. Bands of p'

| class | family | band | n | median rho_q [95% route-bootstrap CI] | p10 | p25 | p75 | p90 | Var=0 share (in-sample) | median rho_q upper bound (zeros at 3/N) | NaN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | <0.02 | 37207 | 0.835 [0.824, 0.847] | 0.476 | 0.686 | 0.934 | 0.984 | 0.004 | 0.837 | 2374 |
| insertion | CON | 0.02-0.2 | 89690 | 0.897 [0.888, 0.906] | 0.558 | 0.753 | 0.97 | 0.991 | 0.000 | 0.897 | 0 |
| insertion | CON | 0.2-0.5 | 62616 | 0.95 [0.941, 0.957] | 0.624 | 0.795 | 0.989 | 0.997 | 0.000 | 0.95 | 0 |
| insertion | CON | >0.5 | 50873 | 1.02 [1.02, 1.03] | 0.713 | 0.961 | 1.18 | 1.53 | 0.000 | 1.02 | 0 |
| insertion | SCA | <0.02 | 36876 | 0.826 [0.813, 0.839] | 0.406 | 0.655 | 0.935 | 0.986 | 0.005 | 0.828 | 1688 |
| insertion | SCA | 0.02-0.2 | 87287 | 0.902 [0.893, 0.911] | 0.528 | 0.751 | 0.969 | 0.99 | 0.000 | 0.902 | 0 |
| insertion | SCA | 0.2-0.5 | 62328 | 0.946 [0.937, 0.955] | 0.597 | 0.796 | 0.99 | 0.997 | 0.000 | 0.946 | 0 |
| insertion | SCA | >0.5 | 51689 | 1.02 [1.02, 1.03] | 0.74 | 0.966 | 1.19 | 1.54 | 0.000 | 1.02 | 0 |
| insertion | CMT | <0.02 | 360 | 0.749 [0.661, 0.821] | 0.508 | 0.626 | 0.84 | 0.938 | 0.006 | 0.749 | 200 |
| insertion | CMT | 0.02-0.2 | 1077 | 0.58 [0.534, 0.627] | 0.35 | 0.474 | 0.694 | 0.784 | 0.003 | 0.58 | 0 |
| insertion | CMT | 0.2-0.5 | 5172 | 0.36 [0.328, 0.405] | 0.176 | 0.263 | 0.488 | 0.613 | 0.009 | 0.361 | 0 |
| insertion | CMT | >0.5 | 11391 | 0.596 [0.579, 0.621] | 0.4 | 0.476 | 0.727 | 0.858 | 0.000 | 0.596 | 0 |
| insertion | city | <0.02 | 3623 | 0.912 [0.837, 1] | 0.613 | 0.751 | 1 | 1 | 0.000 | 0.912 | 3586 |
| insertion | city | 0.02-0.2 | 42345 | 0.704 [0.7, 0.709] | 0.503 | 0.604 | 0.795 | 0.856 | 0.000 | 0.704 | 0 |
| insertion | city | 0.2-0.5 | 9537 | 0.427 [0.411, 0.443] | 0.284 | 0.355 | 0.529 | 0.711 | 0.000 | 0.427 | 0 |
| insertion | city | >0.5 | 18903 | 0.597 [0.591, 0.604] | 0.458 | 0.515 | 0.693 | 0.782 | 0.000 | 0.597 | 0 |
| intra | CON | <0.02 | 20208 | 0.0818 [0.0728, 0.0907] | 0 | 0.0258 | 0.16 | 0.264 | 0.237 | 0.107 | 255 |
| intra | CON | 0.02-0.2 | 27231 | 0.127 [0.119, 0.135] | 0.0282 | 0.0629 | 0.214 | 0.311 | 0.029 | 0.127 | 0 |
| intra | CON | 0.2-0.5 | 9125 | 0.15 [0.136, 0.165] | 0.0353 | 0.0775 | 0.245 | 0.357 | 0.006 | 0.15 | 0 |
| intra | CON | >0.5 | 7071 | 0.175 [0.161, 0.191] | 0.0503 | 0.0978 | 0.265 | 0.363 | 0.003 | 0.175 | 0 |
| intra | SCA | <0.02 | 17807 | 0.0798 [0.0696, 0.0906] | 0 | 0 | 0.167 | 0.27 | 0.286 | 0.117 | 465 |
| intra | SCA | 0.02-0.2 | 27570 | 0.131 [0.124, 0.14] | 0.0296 | 0.065 | 0.219 | 0.32 | 0.026 | 0.131 | 0 |
| intra | SCA | 0.2-0.5 | 9359 | 0.154 [0.139, 0.171] | 0.0367 | 0.0818 | 0.246 | 0.354 | 0.007 | 0.154 | 0 |
| intra | SCA | >0.5 | 6134 | 0.166 [0.152, 0.189] | 0.0434 | 0.0891 | 0.266 | 0.385 | 0.005 | 0.166 | 0 |
| intra | CMT | <0.02 | 74 | 0 [0, 0] | 0 | 0 | 0 | 0 | 1.000 | 0.0533 | 1 |
| intra | CMT | 0.02-0.2 | 1257 | 0 [0, 0.0202] | 0 | 0 | 0.0213 | 0.135 | 0.652 | 0.0149 | 0 |
| intra | CMT | 0.2-0.5 | 11346 | 0 [0, 0.00633] | 0 | 0 | 0.0668 | 0.256 | 0.557 | 0.00787 | 0 |
| intra | CMT | >0.5 | 4438 | 0.0586 [0.0195, 0.11] | 0 | 0 | 0.287 | 0.529 | 0.314 | 0.0588 | 0 |
| intra | city | <0.02 | 3251 | 0 [0, 0] | 0 | 0 | 0 | 0 | 0.907 | 0.0393 | 825 |
| intra | city | 0.02-0.2 | 38069 | 0 [0, 0] | 0 | 0 | 0 | 0.0222 | 0.765 | 0.0207 | 0 |
| intra | city | 0.2-0.5 | 27328 | 0.0158 [0.0135, 0.0185] | 0 | 0.00453 | 0.0392 | 0.0735 | 0.193 | 0.0158 | 0 |
| alt-position | CON | <0.02 | 37226 | 0.072 [0.066, 0.0774] | 0 | 0 | 0.154 | 0.26 | 0.268 | 0.105 | 2374 |
| alt-position | CON | 0.02-0.2 | 89686 | 0.139 [0.134, 0.144] | 0.0279 | 0.0676 | 0.233 | 0.342 | 0.028 | 0.139 | 0 |
| alt-position | CON | 0.2-0.5 | 62619 | 0.194 [0.187, 0.201] | 0.0514 | 0.109 | 0.292 | 0.393 | 0.005 | 0.194 | 0 |
| alt-position | CON | >0.5 | 50855 | 0.232 [0.225, 0.239] | 0.0717 | 0.139 | 0.333 | 0.427 | 0.002 | 0.232 | 0 |
| alt-position | SCA | <0.02 | 36866 | 0.0645 [0.0594, 0.0705] | 0 | 0 | 0.143 | 0.251 | 0.295 | 0.102 | 1689 |
| alt-position | SCA | 0.02-0.2 | 87303 | 0.141 [0.135, 0.145] | 0.0281 | 0.0687 | 0.236 | 0.354 | 0.028 | 0.141 | 0 |
| alt-position | SCA | 0.2-0.5 | 62307 | 0.193 [0.186, 0.2] | 0.0495 | 0.108 | 0.294 | 0.4 | 0.005 | 0.193 | 0 |
| alt-position | SCA | >0.5 | 51703 | 0.236 [0.229, 0.243] | 0.0721 | 0.14 | 0.339 | 0.434 | 0.003 | 0.236 | 0 |
| alt-position | CMT | <0.02 | 360 | 0 [0, 0] | 0 | 0 | 0 | 0.259 | 0.839 | 0.132 | 200 |
| alt-position | CMT | 0.02-0.2 | 1070 | 0 [0, 0.0126] | 0 | 0 | 0.029 | 0.224 | 0.593 | 0.014 | 0 |
| alt-position | CMT | 0.2-0.5 | 5193 | 0.00447 [0, 0.0127] | 0 | 0 | 0.0922 | 0.285 | 0.493 | 0.00802 | 0 |
| alt-position | CMT | >0.5 | 11377 | 0.0208 [0.00895, 0.0463] | 0 | 0 | 0.213 | 0.412 | 0.399 | 0.0208 | 0 |
| alt-position | city | <0.02 | 3623 | 0 [0, 0] | 0 | 0 | 0 | 0 | 0.957 | 0.106 | 3586 |
| alt-position | city | 0.02-0.2 | 42350 | 0 [0, 0] | 0 | 0 | 0.00868 | 0.0252 | 0.708 | 0.0161 | 0 |
| alt-position | city | 0.2-0.5 | 9524 | 0.00897 [0.00775, 0.011] | 0 | 0 | 0.0245 | 0.0559 | 0.306 | 0.00957 | 0 |
| alt-position | city | >0.5 | 18911 | 0.0196 [0.0173, 0.0222] | 0 | 0.00636 | 0.0458 | 0.0847 | 0.163 | 0.0196 | 0 |

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

Lemma. For any reordering r' of the same customers, L_0 = sum of deliveries and L_L = sum of pickups are order-invariant. Let E = {sum d <= Q and sum p <= Q}. On E, D != 0 forces max(F_r, F_r') > Q, so P(D != 0) <= P(E and (F_r > Q or F_r' > Q)). Outside E the overflow event is decided by an end load and is the same for every reordering, so D = 0 there. LEMMA BOUND (valid): B_union = P(E and (F_r > Q or F_r' > Q)), with r' ranging over the sampled intra neighbours of the route (the intra moves in the v2 cache); it bounds P(D != 0) of every such move. DIAGNOSTIC ONLY, NOT A VALID BOUND: B_spec = P(E and F_r > Q), the route-level quantity of the design note. Explained share = P(max(sum d, sum p) > Q) / P(F_r > Q).

| family | routes with intra moves | intra moves | observed Var=0 share | median B_spec | median B_union | pooled explained share | median per-route explained share | max P(D != 0) over moves | median per-route max P(D != 0) | routes with B_spec < route max | routes with B_union < route max | D != 0 with an end load > Q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CON | 1273 | 63890 | 0.092 | 0.00034 | 0.00481 | 0.953 | 0.983 | 0.157 | 0.0034 | 1110 | 0 | 0 |
| SCA | 1261 | 61335 | 0.104 | 0.000425 | 0.00552 | 0.954 | 0.982 | 0.145 | 0.00402 | 1093 | 0 | 0 |
| CMT | 91 | 17116 | 0.503 | 0.000805 | 0.134 | 0.932 | 0.998 | 0.448 | 0.0933 | 72 | 0 | 0 |
| city | 387 | 69473 | 0.549 | 1.5e-05 | 0.0002 | 0.997 | 1.000 | 0.0157 | 0.00012 | 344 | 0 | 0 |
| all | 3012 | 211814 | 0.279 | 0.00027 | 0.00453 | 0.959 | 0.989 | 0.448 | 0.00294 | 2619 | 0 | 0 |

Verification: every intra move (211814 sampled moves over 3012 routes with intra moves, all 3049 routes evaluated) was recomputed scenario by scenario. Scenarios with D != 0 and an end load above Q: 0. Routes with B_union below their largest observed P(D != 0): 0.

B_spec is a diagnostic and is not a valid bound on P(D != 0): it omits scenarios where only the moved route overflows. The column 'routes with B_spec < route max' counts where it falls short of the observed largest P(D != 0). B_union is the lemma bound.

## 3. Query costs per class, both paradigms, with the rule-of-three plug-in

Moves with p' in (0.02, 0.5). Stand-alone: p' estimated at eps (quantum: canonical AE, 5 x (2M + 1) applications; classical: z^2 p'(1-p') / eps^2, z = 1.96). CRN: class 1 one amplitude P(D = 1) with variance max(Var D, 3/N); classes 2-3 quantum two-estimate P(D = 1) and P(D = -1) at eps/2 each, classical direct signed mean z^2 max(Var D, 3/N) / eps^2. eps is absolute (1e-3, 1e-4) or relative, eps_i = 0.1 p'_i per move. Sums are over moves.

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

### 3b. Cost sums (queries) by class and family

Nq/Nq_sa and Nc/Nc_sa: within-paradigm CRN gain (CRN cost over stand-alone cost). Nq/Nc: quantum CRN over classical CRN (classical direct for classes 2-3) at the same target.

#### absolute eps = 1e-3

| class | family | moves | sum Nq_sa | sum Nq | Nq/Nq_sa | sum Nc_sa | sum Nc | Nc/Nc_sa | Nq/Nc |
|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 4.874e+09 | 4.211e+09 | 0.864 | 7.919e+10 | 6.093e+10 | 0.769 | 0.0691 |
| insertion | SCA | 149612 | 4.804e+09 | 4.134e+09 | 0.861 | 7.820e+10 | 5.991e+10 | 0.766 | 0.069 |
| insertion | CMT | 6249 | 2.448e+08 | 1.050e+08 | 0.429 | 5.235e+09 | 9.691e+08 | 0.185 | 0.108 |
| insertion | city | 51882 | 1.347e+09 | 9.085e+08 | 0.674 | 2.015e+10 | 8.187e+09 | 0.406 | 0.111 |
| intra | CON | 36355 | 1.051e+09 | 4.636e+08 | 0.441 | 1.406e+10 | 6.239e+08 | 0.044 | 0.743 |
| intra | SCA | 36929 | 1.056e+09 | 4.763e+08 | 0.451 | 1.423e+10 | 6.351e+08 | 0.045 | 0.75 |
| intra | CMT | 12603 | 4.984e+08 | 1.302e+08 | 0.261 | 1.109e+10 | 2.830e+08 | 0.026 | 0.46 |
| intra | city | 65397 | 1.887e+09 | 3.750e+08 | 0.199 | 3.123e+10 | 5.061e+07 | 0.002 | 7.41 |
| alt-position | CON | 152296 | 4.874e+09 | 2.527e+09 | 0.518 | 7.920e+10 | 4.555e+09 | 0.058 | 0.555 |
| alt-position | SCA | 149608 | 4.803e+09 | 2.511e+09 | 0.523 | 7.817e+10 | 4.621e+09 | 0.059 | 0.543 |
| alt-position | CMT | 6263 | 2.455e+08 | 6.732e+07 | 0.274 | 5.250e+09 | 1.495e+08 | 0.028 | 0.45 |
| alt-position | city | 51874 | 1.347e+09 | 2.770e+08 | 0.206 | 2.014e+10 | 1.437e+07 | 0.001 | 19.3 |

#### absolute eps = 1e-4

| class | family | moves | sum Nq_sa | sum Nq | Nq/Nq_sa | sum Nc_sa | sum Nc | Nc/Nc_sa | Nq/Nc |
|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 4.325e+10 | 3.870e+10 | 0.895 | 7.919e+12 | 6.093e+12 | 0.769 | 0.00635 |
| insertion | SCA | 149612 | 4.240e+10 | 3.780e+10 | 0.891 | 7.820e+12 | 5.991e+12 | 0.766 | 0.00631 |
| insertion | CMT | 6249 | 2.022e+09 | 1.057e+09 | 0.523 | 5.235e+11 | 9.691e+10 | 0.185 | 0.0109 |
| insertion | city | 51882 | 1.313e+10 | 8.955e+09 | 0.682 | 2.015e+12 | 8.187e+11 | 0.406 | 0.0109 |
| intra | CON | 36355 | 9.199e+09 | 4.014e+09 | 0.436 | 1.406e+12 | 6.239e+10 | 0.044 | 0.0643 |
| intra | SCA | 36929 | 9.298e+09 | 4.095e+09 | 0.440 | 1.423e+12 | 6.351e+10 | 0.045 | 0.0645 |
| intra | CMT | 12603 | 4.057e+09 | 9.866e+08 | 0.243 | 1.109e+12 | 2.830e+10 | 0.026 | 0.0349 |
| intra | city | 65397 | 1.543e+10 | 1.910e+09 | 0.124 | 3.123e+12 | 5.061e+09 | 0.002 | 0.377 |
| alt-position | CON | 152296 | 4.325e+10 | 2.285e+10 | 0.528 | 7.920e+12 | 4.555e+11 | 0.058 | 0.0502 |
| alt-position | SCA | 149608 | 4.240e+10 | 2.275e+10 | 0.537 | 7.817e+12 | 4.621e+11 | 0.059 | 0.0492 |
| alt-position | CMT | 6263 | 2.027e+09 | 5.185e+08 | 0.256 | 5.250e+11 | 1.495e+10 | 0.028 | 0.0347 |
| alt-position | city | 51874 | 1.313e+10 | 1.247e+09 | 0.095 | 2.014e+12 | 1.437e+09 | 0.001 | 0.868 |

#### relative eps = 0.1 p'

| class | family | moves | sum Nq_sa | sum Nq | Nq/Nq_sa | sum Nc_sa | sum Nc | Nc/Nc_sa | Nq/Nc |
|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 3.950e+08 | 3.389e+08 | 0.858 | 5.835e+08 | 4.154e+08 | 0.712 | 0.816 |
| insertion | SCA | 149612 | 3.875e+08 | 3.299e+08 | 0.851 | 5.722e+08 | 4.026e+08 | 0.704 | 0.819 |
| insertion | CMT | 6249 | 9.653e+06 | 4.441e+06 | 0.460 | 6.467e+06 | 1.969e+06 | 0.304 | 2.26 |
| insertion | city | 51882 | 1.589e+08 | 1.139e+08 | 0.716 | 2.404e+08 | 1.074e+08 | 0.447 | 1.06 |
| intra | CON | 36355 | 1.212e+08 | 8.406e+07 | 0.694 | 2.180e+08 | 7.586e+06 | 0.035 | 11.1 |
| intra | SCA | 36929 | 1.243e+08 | 8.685e+07 | 0.699 | 2.251e+08 | 8.335e+06 | 0.037 | 10.4 |
| intra | CMT | 12603 | 1.797e+07 | 1.037e+07 | 0.577 | 1.054e+07 | 3.482e+05 | 0.033 | 29.8 |
| intra | city | 65397 | 2.256e+08 | 1.135e+08 | 0.503 | 3.971e+08 | 2.371e+05 | 0.001 | 479 |
| alt-position | CON | 152296 | 3.949e+08 | 2.908e+08 | 0.736 | 5.831e+08 | 2.422e+07 | 0.042 | 12 |
| alt-position | SCA | 149608 | 3.875e+08 | 2.872e+08 | 0.741 | 5.724e+08 | 2.473e+07 | 0.043 | 11.6 |
| alt-position | CMT | 6263 | 9.668e+06 | 5.678e+06 | 0.587 | 6.457e+06 | 1.723e+05 | 0.027 | 33 |
| alt-position | city | 51874 | 1.589e+08 | 7.697e+07 | 0.484 | 2.404e+08 | 1.128e+05 | 0.000 | 683 |

## 4. Discretisation without quadrature error

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

Level rule: median |rel bias| <= 1% and max |rel bias| <= 5% (p >= 0.01) define the recommended b. Recommended b = 6. Max |bias| at the recommended b, over the worst of the paired routes, is still 6.71e-03, so the level error is small in relative terms but not below 1e-3 absolute everywhere.

### 4b. Gauss-Hermite convergence (3 long routes, nearest rule, nodes per factor)

p(n) is the copula discrete window probability with n Gauss-Hermite nodes per factor; p_MC is the continuous copula Monte Carlo on the same N scenarios (its SE is shown). Differences are to the 64-node value.

| instance | route | family | L | b | nodes | p(n) | p(n) - p(64) | p_MC | p_MC SE | p(n) - p_MC |
|---|---|---|---|---|---|---|---|---|---|---|
| CMT11X | 2 | CMT | 28 | 4 | 16 | 0.65743 | +0.00026 | 0.63148 | 0.00108 | +0.02595 |
| CMT11X | 2 | CMT | 28 | 4 | 32 | 0.65632 | -0.00086 | 0.63148 | 0.00108 | +0.02484 |
| CMT11X | 2 | CMT | 28 | 4 | 64 | 0.65717 | +0.00000 | 0.63148 | 0.00108 | +0.02569 |
| CMT11X | 2 | CMT | 28 | 5 | 16 | 0.63321 | +0.00088 | 0.63148 | 0.00108 | +0.00173 |
| CMT11X | 2 | CMT | 28 | 5 | 32 | 0.63373 | +0.00140 | 0.63148 | 0.00108 | +0.00225 |
| CMT11X | 2 | CMT | 28 | 5 | 64 | 0.63233 | +0.00000 | 0.63148 | 0.00108 | +0.00085 |
| CON3-4 | 15 | CON | 22 | 4 | 16 | 0.66938 | +0.01149 | 0.65791 | 0.00106 | +0.01147 |
| CON3-4 | 15 | CON | 22 | 4 | 32 | 0.65859 | +0.00069 | 0.65791 | 0.00106 | +0.00068 |
| CON3-4 | 15 | CON | 22 | 4 | 64 | 0.65790 | +0.00000 | 0.65791 | 0.00106 | -0.00001 |
| CON3-4 | 15 | CON | 22 | 5 | 16 | 0.66809 | +0.01507 | 0.65791 | 0.00106 | +0.01018 |
| CON3-4 | 15 | CON | 22 | 5 | 32 | 0.65410 | +0.00108 | 0.65791 | 0.00106 | -0.00381 |
| CON3-4 | 15 | CON | 22 | 5 | 64 | 0.65302 | +0.00000 | 0.65791 | 0.00106 | -0.00489 |
| HCMC-100-3 | 11 | city | 27 | 4 | 16 | 0.44667 | +0.01671 | 0.46060 | 0.00111 | -0.01393 |
| HCMC-100-3 | 11 | city | 27 | 4 | 32 | 0.43181 | +0.00184 | 0.46060 | 0.00111 | -0.02879 |
| HCMC-100-3 | 11 | city | 27 | 4 | 64 | 0.42996 | +0.00000 | 0.46060 | 0.00111 | -0.03064 |
| HCMC-100-3 | 11 | city | 27 | 5 | 16 | 0.48791 | +0.01480 | 0.46060 | 0.00111 | +0.02731 |
| HCMC-100-3 | 11 | city | 27 | 5 | 32 | 0.47727 | +0.00417 | 0.46060 | 0.00111 | +0.01667 |
| HCMC-100-3 | 11 | city | 27 | 5 | 64 | 0.47310 | +0.00000 | 0.46060 | 0.00111 | +0.01250 |

Node rule: the smallest node count whose max |p(n) - p(64)| over the three routes is <= 0.0001, checked at b = 5 (the quadrature check ran at b in {4, 5}; b = 6 was not run through Gauss-Hermite). Max |p(n) - p(64)|: 16 nodes 1.51e-02, 32 nodes 4.17e-03, 64 nodes 0.00e+00. No node count below 64 meets the tolerance, and 64 is the reference, so no node count is certified by these routes. 128 nodes would be needed to test 64.

Quadrature caveat: at b = 5, max |p(32) - p(64)| = 4.17e-03, so 32 nodes are not within 1e-3 of 64 on routes of length 22 to 28; 64 is the largest node count tested and its own convergence is not verified. The paired level check (Section 4a) does not depend on quadrature and is the basis for the b recommendation.

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

