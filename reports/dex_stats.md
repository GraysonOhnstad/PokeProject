# Dex generator: stat-line model (BST × profile)

5-fold family cross-validation on the training pool (Gen 5 held out). Mean ± spread over folds. Phase-1 reference (shared heads, averaged prediction): stat MAE 21.2, BST MAE 85.

| Metric | text | text+aux | oracle-up+aux | pred-up+aux |
|---|---:|---:|---:|---:|
| BST rank correlation ↑ | 0.566 ± 0.083 | 0.575 ± 0.076 | 0.791 ± 0.059 | 0.571 ± 0.084 |
| BST 80% interval coverage (→ 80%) | 76.7% ± 2.5% | 75.5% ± 3.5% | 72.4% ± 3.9% | 51.6% ± 1.5% |
| BST MAE (expected) ↓ | 85 ± 3 | 85 ± 4 | 53 ± 4 | 85 ± 4 |
| Stat MAE (expected) ↓ | 21.1 ± 0.3 | 21.1 ± 0.2 | 17.7 ± 0.4 | 21.3 ± 0.2 |
| Realism KS: BST, sampled ↓ | 0.112 ± 0.039 | 0.099 ± 0.028 | 0.090 ± 0.016 | 0.149 ± 0.027 |
| Realism KS: BST, averaged ↓ | 0.280 ± 0.033 | 0.289 ± 0.032 | 0.134 ± 0.040 | 0.272 ± 0.030 |
| Realism KS: single stats, sampled ↓ | 0.069 ± 0.018 | 0.064 ± 0.029 | 0.067 ± 0.017 | 0.086 ± 0.012 |
| Realism KS: lopsidedness, sampled ↓ | 0.171 ± 0.079 | 0.133 ± 0.019 | 0.109 ± 0.029 | 0.119 ± 0.040 |
| Realism KS: lopsidedness, averaged ↓ | 0.790 ± 0.058 | 0.792 ± 0.043 | 0.648 ± 0.058 | 0.826 ± 0.045 |

KS: Kolmogorov–Smirnov distance between generated and real distributions (one entry per species). 'Averaged' scores the expected stat line, which is what a point-prediction model outputs.
