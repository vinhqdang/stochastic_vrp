# Timing on an idle machine (one core, one task at a time)

gamma = 0.0001, eps = 0.01, 95% confidence, N = 200,000 draws for the per-draw time; t = build + 1.96^2 RV / eps^2 x per-draw time (quadrature: its wall time).

| route | L | load1 | SMIS | CE mixture | single CE shift | GL shifted | quad (41, V 128/256) | quad (41, V 256/512) |
|---|---|---|---|---|---|---|---|---|
| VNN-DIENBIEN#2 | 10 | 1.21 | 3.40 | 22.61 | 3.08 | 8.71 | 41.2 | 221.6 |
| VNN-HAIDUONG#14 | 9 | 1.01 | 2.38 | 19.56 | 2.93 | 7.07 | 36.4 | 200.6 |
| VNN-HAIPHONG#3 | 10 | 1.00 | 2.81 | 19.51 | 220.65 | 11.55 | 40.3 | 219.3 |
| GD-BOLUOXIAN#50 | 11 | 1.01 | 2.71 | 53.67 | 3.15 | 7.62 | 45.1 | 245.0 |
| GD-DONGGUANSHI-2#9 | 7 | 1.09 | 1.82 | 23.94 | 2.02 | 5.02 | 28.9 | 154.5 |
| GD-DONGGUANSHI-4#48 | 6 | 1.05 | 1.61 | 20.11 | 1.59 | 3.77 | 23.6 | 133.2 |
| median | | | 2.55 | 21.36 | 3.01 | 7.34 | 38.4 | 210.0 |
