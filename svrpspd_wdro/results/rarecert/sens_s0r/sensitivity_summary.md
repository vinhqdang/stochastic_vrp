# RARECERT R1 task B: sensitivity, coverage and the single-shift diagnostic

Gamma = 0.0001. Sensitivity: 30 routes (QUASAR order), 200,000 draws per SMIS, single shift and CE mixture; reference = exact DP (V = 256/512 Richardson) for gamma marginals with L <= 8, else CE mixture with 1,000,000 fresh draws (flagged). Errors are signed relative errors p/p_ref - 1 (medians of absolute values, q90 of absolute values). Wall time 47.3 min.

Rows: 120 route-setting pairs, 0 failed.

## Sensitivity by factor level

| factor | level | rows | ref dp / ce | varrho med | n_modes med | m_hat med | err SMIS med / q90 | err single med / q90 | err CE med / q90 | single share err<0 | RV SMIS med | RV single med | RV CE med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 30 | 25 / 5 | 0.365 | 2 | 0 | 0.27% / 0.77% | 0.24% / 0.82% | 0.26% / 0.82% | 0.60 | 4.52 | 4.23 | 4.58 |
| rho | rho=0.8 | 30 | 25 / 5 | 0.367 | 2 | 0 | 0.33% / 0.95% | 0.39% / 0.88% | 0.39% / 0.85% | 0.40 | 4.48 | 4.2 | 4.56 |
| cv | cv=0.5 | 30 | 25 / 5 | 0.365 | 3 | 5.29e-13 | 0.37% / 0.66% | 0.38% / 0.66% | 0.35% / 0.66% | 0.57 | 4.55 | 4.27 | 4.63 |
| pickup | s=2 | 30 | 25 / 5 | 0.182 | 2 | 0 | 0.40% / 0.77% | 0.34% / 0.66% | 0.28% / 0.61% | 0.57 | 4.51 | 4.23 | 4.57 |

Per route-setting detail (signed errors) is in sensitivity_routes.csv.

## Coverage of the nominal 95% CLT interval (base setting)

10 routes, gamma = 0.0001, 300 replications with N = 10,000 draws each; the proposal is built once per route. Coverage: share of replications with |p - ref| <= 1.96 p rse.

| route | L | ref | method | coverage | median rse | zero estimates |
|---|---|---|---|---|---|---|
| GD-DONGGUANSHI-2#9 | 7 | dp | smis | 0.943 | 0.021 | 0 |
| GD-DONGGUANSHI-2#9 | 7 | dp | single | 0.940 | 0.021 | 0 |
| GD-DEQINGXIAN#5 | 7 | dp | smis | 0.947 | 0.021 | 0 |
| GD-DEQINGXIAN#5 | 7 | dp | single | 0.957 | 0.021 | 0 |
| GD-CHAOYANGXIAN-1#53 | 5 | dp | smis | 0.930 | 0.021 | 0 |
| GD-CHAOYANGXIAN-1#53 | 5 | dp | single | 0.980 | 0.021 | 0 |
| GD-CHAOYANGXIAN-1#67 | 6 | dp | smis | 0.943 | 0.021 | 0 |
| GD-CHAOYANGXIAN-1#67 | 6 | dp | single | 0.970 | 0.021 | 0 |
| GD-BOLUOXIAN#50 | 11 | ce | smis | 0.950 | 0.021 | 0 |
| GD-BOLUOXIAN#50 | 11 | ce | single | 0.967 | 0.021 | 0 |

Pooled mean coverage, smis: 0.943 over 5 routes.

Pooled mean coverage, single: 0.963 over 5 routes.
