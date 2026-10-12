# QUASAR M3a v2: coherent CRN statistics on the BATON copula model

Demand model: BATON's one-factor Gaussian copula (RHO = 0.6) on gamma marginals (cv = 0.3); deliveries and pickups carry independent factors, exactly as `dethloff_runner.make_scenarios`. Continuous model, no discretisation.
Common scenarios: N = 200000 per instance, seed 7 (the runner's SEED), shared by every route and move of that instance.

Instances: 73; unique routes: 3049 (from 3427 gate-route pairs; identical routes under several gates are evaluated once); moves: 1369458 (insertion 578822, intra-route 211814, alternative positions 578822).
Insertion-lemma violations: 0 (a violation aborts the run). Sum of per-instance times over all workers: 300.1 min, of which scenario draws 46.1 min (3 workers ran in parallel; instances cached before a code speed-up keep their earlier timings).

Caps: insertion candidates are a uniform sample of 200 per route (2106 of 3049 routes at the cap); intra-route neighbours are all for |r| <= 12 and a uniform sample of 200 otherwise (454 routes at the cap); alternative positions: one random second position per sampled insertion.

Definitions: D = 1[F' > Q] - 1[F > Q] over common scenarios; p = P(F > Q) (base), p' = P(F' > Q) (new route); rho_q = sqrt(Var D)/sqrt(p'(1-p')); rho_c = Var D/(p'(1-p')). Standard errors by the delta method on the per-scenario influence functions (stored per move in the CSV). Rows with p' in {0, 1} have NaN ratios and are excluded from quantiles; the NaN count is in the header of each table.

## A. Move class x family x base-p stratum

Quantiles of rho_q (median, p10, p25, p75, p90); n = moves with defined rho_q; frac Var=0 = share of moves with no scenario where D != 0 (over all moves in the cell).

| class | family | base p | n | median | p10 | p25 | p75 | p90 | frac Var=0 | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| insertion | CON | p<0.005 | 68340 | 0.9676 | 0.7072 | 0.8806 | 0.9936 | 0.9981 | 0.035 | 2374 |
| insertion | CON | 0.005-0.05 | 83742 | 0.907 | 0.5961 | 0.7805 | 0.9759 | 0.9948 | 0.000 | 0 |
| insertion | CON | 0.05-0.1 | 19328 | 0.8997 | 0.5702 | 0.7471 | 0.9835 | 1.023 | 0.000 | 0 |
| insertion | CON | 0.1-0.3 | 44012 | 0.8706 | 0.5109 | 0.6944 | 1.028 | 1.153 | 0.000 | 0 |
| insertion | CON | >0.3 | 24964 | 0.9557 | 0.4714 | 0.6842 | 1.388 | 1.845 | 0.000 | 0 |
| insertion | SCA | p<0.005 | 66176 | 0.9683 | 0.683 | 0.8728 | 0.9933 | 0.9985 | 0.028 | 1688 |
| insertion | SCA | 0.005-0.05 | 86400 | 0.9203 | 0.5803 | 0.7826 | 0.9783 | 0.9968 | 0.000 | 0 |
| insertion | SCA | 0.05-0.1 | 15348 | 0.8924 | 0.5286 | 0.7399 | 0.9784 | 1.025 | 0.000 | 0 |
| insertion | SCA | 0.1-0.3 | 46152 | 0.8744 | 0.4922 | 0.6889 | 1.03 | 1.174 | 0.000 | 0 |
| insertion | SCA | >0.3 | 24104 | 0.9799 | 0.4512 | 0.6812 | 1.404 | 1.923 | 0.000 | 0 |
| insertion | CMT | p<0.005 | 200 | 0.8207 | 0.599 | 0.7189 | 0.8992 | 0.9639 | 0.500 | 200 |
| insertion | CMT | 0.005-0.05 | 400 | 0.6928 | 0.4178 | 0.5449 | 0.7967 | 0.8791 | 0.005 | 0 |
| insertion | CMT | 0.05-0.1 | 400 | 0.5881 | 0.3328 | 0.4732 | 0.7268 | 0.7802 | 0.007 | 0 |
| insertion | CMT | 0.1-0.3 | 1000 | 0.575 | 0.3296 | 0.4587 | 0.6778 | 0.7635 | 0.000 | 0 |
| insertion | CMT | >0.3 | 16000 | 0.5269 | 0.2689 | 0.3966 | 0.6756 | 0.8178 | 0.003 | 0 |
| insertion | city | p<0.005 | 3008 | 0.9624 | 0.743 | 0.8593 | 1 | 1 | 0.544 | 3586 |
| insertion | city | 0.005-0.05 | 42000 | 0.7076 | 0.5064 | 0.6088 | 0.7987 | 0.8603 | 0.000 | 0 |
| insertion | city | 0.05-0.1 | 800 | 0.6547 | 0.4406 | 0.546 | 0.745 | 0.8057 | 0.000 | 0 |
| insertion | city | 0.1-0.3 | 2200 | 0.614 | 0.4058 | 0.5035 | 0.7106 | 0.7957 | 0.000 | 0 |
| insertion | city | >0.3 | 26400 | 0.5431 | 0.3627 | 0.4435 | 0.6559 | 0.7473 | 0.000 | 0 |
| insertion | all | p<0.005 | 137724 | 0.9677 | 0.6983 | 0.8762 | 0.9936 | 0.9985 | 0.056 | 7848 |
| insertion | all | 0.005-0.05 | 212542 | 0.8723 | 0.556 | 0.7146 | 0.9656 | 0.9933 | 0.000 | 0 |
| insertion | all | 0.05-0.1 | 35876 | 0.8879 | 0.5442 | 0.7279 | 0.9795 | 1.023 | 0.000 | 0 |
| insertion | all | 0.1-0.3 | 93364 | 0.8621 | 0.4931 | 0.6759 | 1.023 | 1.159 | 0.000 | 0 |
| insertion | all | >0.3 | 91468 | 0.6827 | 0.3666 | 0.4971 | 1.017 | 1.563 | 0.001 | 0 |
| intra | CON | p<0.005 | 11133 | 0.07673 | 0 | 0 | 0.1613 | 0.2748 | 0.334 | 255 |
| intra | CON | 0.005-0.05 | 23667 | 0.1077 | 0.01949 | 0.05124 | 0.1886 | 0.2864 | 0.075 | 0 |
| intra | CON | 0.05-0.1 | 4204 | 0.1369 | 0.02948 | 0.06448 | 0.2261 | 0.3306 | 0.030 | 0 |
| intra | CON | 0.1-0.3 | 14603 | 0.1427 | 0.03374 | 0.07369 | 0.2338 | 0.3354 | 0.011 | 0 |
| intra | CON | >0.3 | 10028 | 0.1671 | 0.04117 | 0.089 | 0.2622 | 0.3662 | 0.004 | 0 |
| intra | SCA | p<0.005 | 11137 | 0.06454 | 0 | 0 | 0.154 | 0.259 | 0.405 | 465 |
| intra | SCA | 0.005-0.05 | 21340 | 0.1184 | 0.02293 | 0.0558 | 0.2062 | 0.3113 | 0.062 | 0 |
| intra | SCA | 0.05-0.1 | 4425 | 0.1417 | 0.03278 | 0.07493 | 0.2253 | 0.3164 | 0.027 | 0 |
| intra | SCA | 0.1-0.3 | 14989 | 0.1365 | 0.03095 | 0.06972 | 0.2286 | 0.334 | 0.011 | 0 |
| intra | SCA | >0.3 | 8979 | 0.1688 | 0.04472 | 0.0914 | 0.264 | 0.3734 | 0.004 | 0 |
| intra | CMT | p<0.005 | 13 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1 |
| intra | CMT | 0.005-0.05 | 122 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| intra | CMT | 0.05-0.1 | 381 | 0 | 0 | 0 | 0 | 0 | 0.995 | 0 |
| intra | CMT | 0.1-0.3 | 1000 | 0.009506 | 0 | 0 | 0.03775 | 0.1668 | 0.436 | 0 |
| intra | CMT | >0.3 | 15599 | 0.004494 | 0 | 0 | 0.1246 | 0.3328 | 0.491 | 0 |
| intra | city | p<0.005 | 813 | 0 | 0 | 0 | 0 | 0 | 1.000 | 825 |
| intra | city | 0.005-0.05 | 38484 | 0 | 0 | 0 | 0 | 0.02139 | 0.782 | 0 |
| intra | city | 0.05-0.1 | 821 | 0 | 0 | 0 | 0.00861 | 0.01722 | 0.638 | 0 |
| intra | city | 0.1-0.3 | 2183 | 0.01004 | 0 | 0 | 0.033 | 0.05056 | 0.342 | 0 |
| intra | city | >0.3 | 26347 | 0.01567 | 0 | 0.00451 | 0.03932 | 0.07396 | 0.196 | 0 |
| intra | all | p<0.005 | 23096 | 0.0661 | 0 | 0 | 0.1535 | 0.2626 | 0.412 | 1546 |
| intra | all | 0.005-0.05 | 83613 | 0.02883 | 0 | 0 | 0.1221 | 0.2314 | 0.398 | 0 |
| intra | all | 0.05-0.1 | 9831 | 0.1187 | 0 | 0.04285 | 0.2088 | 0.3071 | 0.117 | 0 |
| intra | all | 0.1-0.3 | 32775 | 0.126 | 0.0161 | 0.05384 | 0.2191 | 0.3248 | 0.046 | 0 |
| intra | all | >0.3 | 60953 | 0.03601 | 0 | 0.006328 | 0.1391 | 0.2838 | 0.212 | 0 |
| alt-position | CON | p<0.005 | 13890 | 0.03827 | 0 | 0 | 0.1243 | 0.2339 | 0.550 | 2374 |
| alt-position | CON | 0.005-0.05 | 49232 | 0.09811 | 0 | 0.04192 | 0.1832 | 0.2876 | 0.100 | 0 |
| alt-position | CON | 0.05-0.1 | 28285 | 0.1346 | 0.02852 | 0.06707 | 0.2264 | 0.3384 | 0.024 | 0 |
| alt-position | CON | 0.1-0.3 | 60732 | 0.1742 | 0.04237 | 0.092 | 0.2704 | 0.3798 | 0.008 | 0 |
| alt-position | CON | >0.3 | 88247 | 0.2181 | 0.06401 | 0.1277 | 0.3176 | 0.415 | 0.002 | 0 |
| alt-position | SCA | p<0.005 | 14811 | 0 | 0 | 0 | 0.1082 | 0.2213 | 0.568 | 1689 |
| alt-position | SCA | 0.005-0.05 | 47567 | 0.0955 | 0.01084 | 0.0408 | 0.1806 | 0.2911 | 0.098 | 0 |
| alt-position | SCA | 0.05-0.1 | 28017 | 0.138 | 0.03111 | 0.0705 | 0.2294 | 0.3458 | 0.021 | 0 |
| alt-position | SCA | 0.1-0.3 | 58523 | 0.1741 | 0.04031 | 0.0922 | 0.2737 | 0.3884 | 0.008 | 0 |
| alt-position | SCA | >0.3 | 89261 | 0.2213 | 0.06436 | 0.1289 | 0.324 | 0.4233 | 0.003 | 0 |
| alt-position | CMT | p<0.005 | 191 | 0 | 0 | 0 | 0 | 0.4304 | 0.928 | 200 |
| alt-position | CMT | 0.005-0.05 | 212 | 0 | 0 | 0 | 0 | 0.1644 | 0.825 | 0 |
| alt-position | CMT | 0.05-0.1 | 366 | 0 | 0 | 0 | 0 | 0.2496 | 0.773 | 0 |
| alt-position | CMT | 0.1-0.3 | 1037 | 0.009192 | 0 | 0 | 0.07651 | 0.2513 | 0.425 | 0 |
| alt-position | CMT | >0.3 | 16194 | 0.01227 | 0 | 0 | 0.1755 | 0.3791 | 0.431 | 0 |
| alt-position | city | p<0.005 | 2286 | 0 | 0 | 0 | 0 | 0 | 0.998 | 3586 |
| alt-position | city | 0.005-0.05 | 9357 | 0 | 0 | 0 | 0 | 0.01716 | 0.819 | 0 |
| alt-position | city | 0.05-0.1 | 25818 | 0 | 0 | 0 | 0.007928 | 0.02254 | 0.737 | 0 |
| alt-position | city | 0.1-0.3 | 9605 | 0 | 0 | 0 | 0.01761 | 0.04517 | 0.505 | 0 |
| alt-position | city | >0.3 | 27342 | 0.0156 | 0 | 0.004491 | 0.03952 | 0.07486 | 0.206 | 0 |
| alt-position | all | p<0.005 | 31178 | 0 | 0 | 0 | 0.1076 | 0.2182 | 0.629 | 7849 |
| alt-position | all | 0.005-0.05 | 106368 | 0.08452 | 0 | 0.02833 | 0.1708 | 0.2782 | 0.164 | 0 |
| alt-position | all | 0.05-0.1 | 82486 | 0.07376 | 0 | 0.007503 | 0.1797 | 0.2934 | 0.250 | 0 |
| alt-position | all | 0.1-0.3 | 129897 | 0.1598 | 0.01957 | 0.072 | 0.2617 | 0.375 | 0.048 | 0 |
| alt-position | all | >0.3 | 221044 | 0.184 | 0.01224 | 0.07208 | 0.2987 | 0.4047 | 0.059 | 0 |

Classes: insertion = inter-route insertion (D in {0,1}); intra = relocate, swap or 2-opt inside the route (D in {-1,0,1}); alt-position = two positions of the same inserted customer (base = pos, new = pos2; base p is that insertion's p').

## B. Incremental feasibility totals (classes 1 and 2; base route once + increments)

Stand-alone: each candidate p' estimated at eps. Incremental: base p at eps/sqrt(2), each increment D at eps/sqrt(2) (signed D: two estimates at half that). Query counts: canonical AE, 5 x (2M+1) A-applications; classical z^2 var/eps^2, z = 1.96.

### eps = 0.001

| family | routes | stand-alone Nq | incremental Nq | Nq ratio | stand-alone Nc | incremental Nc | Nc ratio |
|---|---|---|---|---|---|---|---|
| CON | 1293 | 8.588e+09 | 1.056e+10 | 1.2301 | 1.385e+11 | 2.117e+11 | 1.5279 |
| SCA | 1275 | 8.471e+09 | 1.046e+10 | 1.2350 | 1.371e+11 | 2.113e+11 | 1.5404 |
| CMT | 91 | 1.393e+09 | 8.557e+08 | 0.6142 | 3.044e+10 | 1.495e+10 | 0.4912 |
| city | 390 | 4.061e+09 | 2.492e+09 | 0.6136 | 6.932e+10 | 3.08e+10 | 0.4443 |
| all | 3049 | 2.251e+10 | 2.437e+10 | 1.0826 | 3.754e+11 | 4.687e+11 | 1.2483 |

### eps = 0.01

| family | routes | stand-alone Nq | incremental Nq | Nq ratio | stand-alone Nc | incremental Nc | Nc ratio |
|---|---|---|---|---|---|---|---|
| CON | 1293 | 9.501e+08 | 1.03e+09 | 1.0842 | 1.385e+09 | 2.117e+09 | 1.5279 |
| SCA | 1275 | 9.365e+08 | 1.014e+09 | 1.0823 | 1.371e+09 | 2.113e+09 | 1.5404 |
| CMT | 91 | 1.71e+08 | 1.023e+08 | 0.5984 | 3.044e+08 | 1.495e+08 | 0.4912 |
| city | 390 | 4.702e+08 | 3.16e+08 | 0.6719 | 6.932e+08 | 3.08e+08 | 0.4443 |
| all | 3049 | 2.528e+09 | 2.462e+09 | 0.9740 | 3.754e+09 | 4.687e+09 | 1.2483 |

### eps = 0.0001

| family | routes | stand-alone Nq | incremental Nq | Nq ratio | stand-alone Nc | incremental Nc | Nc ratio |
|---|---|---|---|---|---|---|---|
| CON | 1293 | 7.504e+10 | 1.032e+11 | 1.3745 | 1.385e+13 | 2.117e+13 | 1.5279 |
| SCA | 1275 | 7.384e+10 | 1.025e+11 | 1.3884 | 1.371e+13 | 2.113e+13 | 1.5404 |
| CMT | 91 | 1.129e+10 | 8.112e+09 | 0.7187 | 3.044e+12 | 1.495e+12 | 0.4912 |
| city | 390 | 3.523e+10 | 2.242e+10 | 0.6364 | 6.932e+12 | 3.08e+12 | 0.4443 |
| all | 3049 | 1.954e+11 | 2.362e+11 | 1.2088 | 3.754e+13 | 4.687e+13 | 1.2483 |

## C. Sensitivity: rho = 0 (independent customers) on every 4th instance

Instances: 19 (CMT11X, CMT1X, CMT3X, CMT5X, CON3-2, CON3-6, CON8-0, CON8-4, CON8-8, HANOI-100-3, HCMC-100-2, NYC-100-1, PARIS-100-2, SCA3-2, SCA3-6, SCA8-0, SCA8-4, SCA8-8, SHANGHAI-100-3). Same seed; rho = 0 draws are independent per customer (dethloff_runner.sample_demands with rho = 0).

| class | base p | n (rho=0.6) | median rho_q (0.6) | n (rho=0) | median rho_q (0) | frac Var=0 (0.6) | frac Var=0 (0) |
|---|---|---|---|---|---|---|---|
| insertion | p<0.005 | 37411 | 0.9774 | 74338 | 0.9918 | 0.051 | 0.128 |
| insertion | 0.005-0.05 | 51148 | 0.8951 | 26108 | 0.9654 | 0.000 | 0.001 |
| insertion | 0.05-0.1 | 8056 | 0.8873 | 9232 | 0.9733 | 0.001 | 0.000 |
| insertion | 0.1-0.3 | 27100 | 0.8775 | 9092 | 1.008 | 0.000 | 0.000 |
| insertion | >0.3 | 21164 | 0.7364 | 17964 | 1.104 | 0.001 | 0.002 |
| intra | p<0.005 | 5145 | 0.07924 | 17466 | 0.1373 | 0.399 | 0.647 |
| intra | 0.005-0.05 | 17289 | 0.03946 | 5117 | 0.4059 | 0.351 | 0.070 |
| intra | 0.05-0.1 | 2434 | 0.09811 | 1545 | 0.2913 | 0.134 | 0.083 |
| intra | 0.1-0.3 | 8613 | 0.1237 | 1977 | 0.3462 | 0.054 | 0.197 |
| intra | >0.3 | 12253 | 0.04249 | 10742 | 0.06023 | 0.243 | 0.350 |
| alt-position | p<0.005 | 8462 | 0 | 35695 | 0.2236 | 0.610 | 0.493 |
| alt-position | 0.005-0.05 | 25260 | 0.09745 | 25073 | 0.3929 | 0.134 | 0.044 |
| alt-position | 0.05-0.1 | 19209 | 0.09049 | 10303 | 0.4239 | 0.214 | 0.012 |
| alt-position | 0.1-0.3 | 34784 | 0.1626 | 21004 | 0.4287 | 0.045 | 0.008 |
| alt-position | >0.3 | 57164 | 0.2019 | 44647 | 0.4082 | 0.048 | 0.090 |

Incremental totals on the same instances (eps = 1e-3):

| family | routes | Nq ratio (0.6) | Nq ratio (0) | Nc ratio (0.6) | Nc ratio (0) |
|---|---|---|---|---|---|
| CON | 345 / 345 | 1.2946 | 1.5788 | 1.6223 | 2.1497 |
| SCA | 334 / 334 | 1.2644 | 1.5537 | 1.6083 | 2.1102 |
| CMT | 25 / 25 | 0.5941 | 0.8894 | 0.4578 | 1.0309 |
| city | 77 / 77 | 0.6242 | 0.9323 | 0.4769 | 0.8535 |
| all | 781 / 781 | 1.1410 | 1.4325 | 1.3543 | 1.8336 |

## D. Discretisation check (copula discrete model vs continuous copula Monte Carlo)

Routes: 40 (two per family x base-p stratum where available); 16 Gauss-Hermite nodes per factor (256 node pairs); MC from the same 200000 scenarios. Bias = p_disc - p_mc with MC interval bias +- 1.96 se(p_mc). Relative bias only where p_mc >= 0.01. Rules: design (literal) for b = 4, 5, 6 (b = 6 only for routes with L <= 8); nearest (review variant) for b = 4, 5. Quadrature convergence on one long route is in table E.

| rule | family | b | base-p stratum | checks | median abs bias | max abs bias | median abs rel bias | intervals covering 0 |
|---|---|---|---|---|---|---|---|---|
| design | CMT | 4 | 0.005-0.05 | 2 | 0.001645 | 0.003155 | 0.072 | 1 / 2 |
| design | CMT | 4 | 0.05-0.1 | 2 | 0.001478 | 0.001589 | 0.02215 | 0 / 2 |
| design | CMT | 4 | 0.1-0.3 | 2 | 0.01612 | 0.02239 | 0.08707 | 0 / 2 |
| design | CMT | 4 | >0.3 | 2 | 0.01101 | 0.01633 | 0.02486 | 0 / 2 |
| design | CMT | 4 | p<0.005 | 2 | 4.935e-05 | 9.87e-05 | - | 1 / 2 |
| design | CMT | 5 | 0.005-0.05 | 2 | 0.0005948 | 0.001105 | 0.02521 | 1 / 2 |
| design | CMT | 5 | 0.05-0.1 | 2 | 0.002289 | 0.002586 | 0.03412 | 0 / 2 |
| design | CMT | 5 | 0.1-0.3 | 2 | 0.01285 | 0.02401 | 0.05363 | 0 / 2 |
| design | CMT | 5 | >0.3 | 2 | 0.01234 | 0.01815 | 0.02786 | 0 / 2 |
| design | CMT | 5 | p<0.005 | 2 | 5.211e-05 | 0.0001042 | - | 1 / 2 |
| design | CMT | 6 | 0.005-0.05 | 2 | 0.0005014 | 0.0009318 | 0.02127 | 1 / 2 |
| design | CMT | 6 | >0.3 | 1 | 0.006706 | 0.006706 | 0.01491 | 0 / 1 |
| design | CMT | 6 | p<0.005 | 2 | 5.109e-05 | 0.0001022 | - | 1 / 2 |
| design | CON | 4 | 0.005-0.05 | 2 | 0.001182 | 0.001435 | 0.08913 | 0 / 2 |
| design | CON | 4 | 0.05-0.1 | 2 | 0.01412 | 0.01926 | 0.1946 | 0 / 2 |
| design | CON | 4 | 0.1-0.3 | 2 | 0.004681 | 0.009165 | 0.03144 | 1 / 2 |
| design | CON | 4 | >0.3 | 2 | 0.00459 | 0.008068 | 0.01273 | 1 / 2 |
| design | CON | 4 | p<0.005 | 2 | 3.348e-05 | 4.751e-05 | - | 1 / 2 |
| design | CON | 5 | 0.005-0.05 | 2 | 0.0002516 | 0.0002846 | 0.01945 | 2 / 2 |
| design | CON | 5 | 0.05-0.1 | 2 | 0.001826 | 0.002047 | 0.02437 | 0 / 2 |
| design | CON | 5 | 0.1-0.3 | 2 | 0.007004 | 0.009028 | 0.04071 | 0 / 2 |
| design | CON | 5 | >0.3 | 2 | 0.003332 | 0.003705 | 0.008001 | 0 / 2 |
| design | CON | 5 | p<0.005 | 2 | 4.292e-05 | 4.878e-05 | - | 1 / 2 |
| design | CON | 6 | 0.005-0.05 | 1 | 6.516e-05 | 6.516e-05 | 0.004634 | 1 / 1 |
| design | CON | 6 | 0.05-0.1 | 2 | 0.0007422 | 0.0009076 | 0.01004 | 2 / 2 |
| design | CON | 6 | >0.3 | 1 | 0.004105 | 0.004105 | 0.008295 | 0 / 1 |
| design | CON | 6 | p<0.005 | 2 | 4.433e-05 | 4.741e-05 | - | 1 / 2 |
| design | SCA | 4 | 0.005-0.05 | 2 | 0.0005905 | 0.0008589 | 0.04398 | 1 / 2 |
| design | SCA | 4 | 0.05-0.1 | 2 | 0.00853 | 0.009204 | 0.09557 | 0 / 2 |
| design | SCA | 4 | 0.1-0.3 | 2 | 0.008039 | 0.01321 | 0.04455 | 0 / 2 |
| design | SCA | 4 | >0.3 | 2 | 0.04007 | 0.07423 | 0.07012 | 0 / 2 |
| design | SCA | 4 | p<0.005 | 2 | 0.0001449 | 0.0002897 | - | 0 / 2 |
| design | SCA | 5 | 0.005-0.05 | 2 | 0.0002226 | 0.0002753 | 0.01409 | 2 / 2 |
| design | SCA | 5 | 0.05-0.1 | 2 | 0.001475 | 0.002809 | 0.01802 | 1 / 2 |
| design | SCA | 5 | 0.1-0.3 | 2 | 0.003098 | 0.003193 | 0.01887 | 0 / 2 |
| design | SCA | 5 | >0.3 | 2 | 0.01127 | 0.01383 | 0.02456 | 0 / 2 |
| design | SCA | 5 | p<0.005 | 2 | 0.0001154 | 0.0002309 | - | 1 / 2 |
| design | SCA | 6 | 0.005-0.05 | 2 | 0.0001401 | 0.000146 | 0.007474 | 2 / 2 |
| design | SCA | 6 | 0.05-0.1 | 1 | 0.0001482 | 0.0001482 | 0.00152 | 1 / 1 |
| design | SCA | 6 | >0.3 | 1 | 0.0003514 | 0.0003514 | 0.0005791 | 1 / 1 |
| design | SCA | 6 | p<0.005 | 2 | 0.0001054 | 0.0002108 | - | 1 / 2 |
| design | city | 4 | 0.005-0.05 | 2 | 0.001097 | 0.001853 | 0.03507 | 1 / 2 |
| design | city | 4 | 0.05-0.1 | 2 | 0.002799 | 0.003769 | 0.03564 | 0 / 2 |
| design | city | 4 | 0.1-0.3 | 2 | 0.01096 | 0.01276 | 0.07777 | 0 / 2 |
| design | city | 4 | >0.3 | 2 | 0.02294 | 0.0339 | 0.05316 | 0 / 2 |
| design | city | 4 | p<0.005 | 2 | 3.843e-06 | 7.532e-06 | - | 1 / 2 |
| design | city | 5 | 0.005-0.05 | 2 | 0.001095 | 0.001926 | 0.03402 | 1 / 2 |
| design | city | 5 | 0.05-0.1 | 2 | 0.005239 | 0.005367 | 0.06818 | 0 / 2 |
| design | city | 5 | 0.1-0.3 | 2 | 0.001863 | 0.002264 | 0.013 | 0 / 2 |
| design | city | 5 | >0.3 | 2 | 0.01398 | 0.01657 | 0.03283 | 0 / 2 |
| design | city | 5 | p<0.005 | 2 | 3.591e-06 | 7.052e-06 | - | 1 / 2 |
| design | city | 6 | p<0.005 | 2 | 3.63e-06 | 7.134e-06 | - | 1 / 2 |
| nearest | CMT | 4 | 0.005-0.05 | 2 | 0.001747 | 0.003352 | 0.07652 | 1 / 2 |
| nearest | CMT | 4 | 0.05-0.1 | 2 | 0.001478 | 0.001589 | 0.02215 | 0 / 2 |
| nearest | CMT | 4 | 0.1-0.3 | 2 | 0.01612 | 0.02239 | 0.08707 | 0 / 2 |
| nearest | CMT | 4 | >0.3 | 2 | 0.01069 | 0.01573 | 0.02414 | 0 / 2 |
| nearest | CMT | 4 | p<0.005 | 2 | 4.935e-05 | 9.87e-05 | - | 1 / 2 |
| nearest | CMT | 5 | 0.005-0.05 | 2 | 0.0005872 | 0.001085 | 0.02478 | 1 / 2 |
| nearest | CMT | 5 | 0.05-0.1 | 2 | 0.002281 | 0.002586 | 0.03399 | 0 / 2 |
| nearest | CMT | 5 | 0.1-0.3 | 2 | 0.01285 | 0.02401 | 0.05363 | 0 / 2 |
| nearest | CMT | 5 | >0.3 | 2 | 0.01235 | 0.01817 | 0.02788 | 0 / 2 |
| nearest | CMT | 5 | p<0.005 | 2 | 4.849e-05 | 9.697e-05 | - | 1 / 2 |
| nearest | CON | 4 | 0.005-0.05 | 2 | 0.001222 | 0.001435 | 0.09244 | 0 / 2 |
| nearest | CON | 4 | 0.05-0.1 | 2 | 0.01412 | 0.01926 | 0.1946 | 0 / 2 |
| nearest | CON | 4 | 0.1-0.3 | 2 | 0.004392 | 0.008588 | 0.02949 | 1 / 2 |
| nearest | CON | 4 | >0.3 | 2 | 0.003925 | 0.007256 | 0.01104 | 1 / 2 |
| nearest | CON | 4 | p<0.005 | 2 | 3.201e-05 | 4.751e-05 | - | 1 / 2 |
| nearest | CON | 5 | 0.005-0.05 | 2 | 0.0002516 | 0.0002846 | 0.01945 | 2 / 2 |
| nearest | CON | 5 | 0.05-0.1 | 2 | 0.001826 | 0.002047 | 0.02437 | 0 / 2 |
| nearest | CON | 5 | 0.1-0.3 | 2 | 0.007013 | 0.009046 | 0.04077 | 0 / 2 |
| nearest | CON | 5 | >0.3 | 2 | 0.003312 | 0.003665 | 0.007961 | 0 / 2 |
| nearest | CON | 5 | p<0.005 | 2 | 4.275e-05 | 4.878e-05 | - | 1 / 2 |
| nearest | SCA | 4 | 0.005-0.05 | 2 | 0.0005905 | 0.0008589 | 0.04398 | 1 / 2 |
| nearest | SCA | 4 | 0.05-0.1 | 2 | 0.008634 | 0.009204 | 0.09685 | 0 / 2 |
| nearest | SCA | 4 | 0.1-0.3 | 2 | 0.008039 | 0.01321 | 0.04455 | 0 / 2 |
| nearest | SCA | 4 | >0.3 | 2 | 0.04006 | 0.07423 | 0.07008 | 0 / 2 |
| nearest | SCA | 4 | p<0.005 | 2 | 0.0001537 | 0.0003074 | - | 0 / 2 |
| nearest | SCA | 5 | 0.005-0.05 | 2 | 0.0002276 | 0.0002852 | 0.0146 | 2 / 2 |
| nearest | SCA | 5 | 0.05-0.1 | 2 | 0.001511 | 0.002756 | 0.01833 | 1 / 2 |
| nearest | SCA | 5 | 0.1-0.3 | 2 | 0.003098 | 0.003193 | 0.01887 | 0 / 2 |
| nearest | SCA | 5 | >0.3 | 2 | 0.01063 | 0.01257 | 0.02351 | 0 / 2 |
| nearest | SCA | 5 | p<0.005 | 2 | 0.0001229 | 0.0002459 | - | 1 / 2 |
| nearest | city | 4 | 0.005-0.05 | 2 | 0.001129 | 0.001917 | 0.03596 | 1 / 2 |
| nearest | city | 4 | 0.05-0.1 | 2 | 0.002795 | 0.003769 | 0.03559 | 0 / 2 |
| nearest | city | 4 | 0.1-0.3 | 2 | 0.01112 | 0.013 | 0.07882 | 0 / 2 |
| nearest | city | 4 | >0.3 | 2 | 0.02463 | 0.0369 | 0.05704 | 0 / 2 |
| nearest | city | 4 | p<0.005 | 2 | 2.943e-06 | 5.731e-06 | - | 1 / 2 |
| nearest | city | 5 | 0.005-0.05 | 2 | 0.001095 | 0.001926 | 0.03402 | 1 / 2 |
| nearest | city | 5 | 0.05-0.1 | 2 | 0.005213 | 0.005315 | 0.06782 | 0 / 2 |
| nearest | city | 5 | 0.1-0.3 | 2 | 0.001868 | 0.002275 | 0.01303 | 0 / 2 |
| nearest | city | 5 | >0.3 | 2 | 0.01398 | 0.01657 | 0.03283 | 0 / 2 |
| nearest | city | 5 | p<0.005 | 2 | 3.591e-06 | 7.052e-06 | - | 1 / 2 |

Overall by rule and b:

| rule | b | checks | median abs bias | max abs bias | median abs rel bias (p>=0.01) | max abs rel bias (p>=0.01) |
|---|---|---|---|---|---|---|
| design | 4 | 40 | 0.002363 | 0.07423 | 0.05703 | 0.2874 |
| design | 5 | 40 | 0.001959 | 0.02401 | 0.02308 | 0.09209 |
| design | 6 | 19 | 0.0001341 | 0.006706 | 0.007474 | 0.02127 |
| nearest | 4 | 40 | 0.002394 | 0.07423 | 0.05597 | 0.2874 |
| nearest | 5 | 40 | 0.001952 | 0.02401 | 0.02267 | 0.09209 |

## E. Gauss-Hermite quadrature convergence (one long CMT route, design rule)

| instance | route | L | b | Gauss-Hermite nodes per factor | p_disc | p_mc | p_disc - p_mc | MC se |
|---|---|---|---|---|---|---|---|---|
| CMT11Y | 1 | 22 | 4 | 10 | 0.13924 | 0.11159 | +0.02765 | 0.00070 |
| CMT11Y | 1 | 22 | 4 | 16 | 0.12144 | 0.11159 | +0.00985 | 0.00070 |
| CMT11Y | 1 | 22 | 4 | 24 | 0.11461 | 0.11159 | +0.00302 | 0.00070 |
| CMT11Y | 1 | 22 | 5 | 10 | 0.13905 | 0.11159 | +0.02746 | 0.00070 |
| CMT11Y | 1 | 22 | 5 | 16 | 0.11328 | 0.11159 | +0.00169 | 0.00070 |
| CMT11Y | 1 | 22 | 5 | 24 | 0.10861 | 0.11159 | -0.00298 | 0.00070 |
