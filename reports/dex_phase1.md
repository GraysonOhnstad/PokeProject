# Dex generator, phase 1: entry fields from flavor text

5-fold family cross-validation over 10458 texts of 866 species (Gen 5 held out, not used). Mean over folds, each species weighted equally; ± is the spread across folds. Trained models stop early per field on held-out training families.

| Metric | prior | tfidf-1nn | minilm-1nn | minilm-knn | minilm-mlp | minilm-mlp+knn | minilm-ft | minilm-ft+knn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Types mAP ↑ | 0.084 ± 0.001 | 0.115 ± 0.004 | 0.158 ± 0.005 | 0.283 ± 0.006 | 0.310 ± 0.014 | 0.339 ± 0.007 | 0.351 ± 0.014 | 0.354 ± 0.011 |
| Types top-1 ↑ | 15.7% ± 2.6% | 22.2% ± 1.0% | 30.4% ± 1.1% | 38.3% ± 0.6% | 40.6% ± 1.4% | 42.5% ± 0.7% | 45.0% ± 2.5% | 45.9% ± 1.0% |
| Shape acc. ↑ | 21.0% ± 4.4% | 17.8% ± 1.9% | 20.0% ± 1.6% | 20.8% ± 1.8% | 22.4% ± 2.1% | 22.5% ± 2.1% | 23.1% ± 3.9% | 23.3% ± 2.8% |
| Color acc. ↑ | 14.3% ± 3.2% | 14.5% ± 1.2% | 16.8% ± 1.8% | 18.5% ± 2.2% | 21.5% ± 4.2% | 20.2% ± 1.9% | 19.8% ± 5.2% | 20.6% ± 2.1% |
| Egg group mAP ↑ | 0.091 ± 0.005 | 0.117 ± 0.004 | 0.130 ± 0.004 | 0.203 ± 0.012 | 0.240 ± 0.015 | 0.249 ± 0.016 | 0.254 ± 0.015 | 0.253 ± 0.011 |
| Height error × ↓ | 1.85 ± 0.06 | 2.10 ± 0.04 | 2.04 ± 0.07 | 1.83 ± 0.05 | 1.77 ± 0.05 | 1.77 ± 0.05 | 1.80 ± 0.05 | 1.77 ± 0.05 |
| Weight error × ↓ | 3.72 ± 0.25 | 5.00 ± 0.20 | 4.59 ± 0.26 | 3.58 ± 0.25 | 3.33 ± 0.17 | 3.33 ± 0.19 | 3.34 ± 0.25 | 3.29 ± 0.22 |
| Stat MAE ↓ | 22.9 ± 0.5 | 29.0 ± 0.6 | 28.3 ± 0.4 | 22.9 ± 0.4 | 21.2 ± 0.3 | 21.3 ± 0.4 | 21.8 ± 0.7 | 21.4 ± 0.3 |
| BST MAE ↓ | 98 ± 3 | 104 ± 7 | 105 ± 2 | 91 ± 2 | 85 ± 4 | 85 ± 3 | 87 ± 5 | 85 ± 3 |
| Ability recall@5 ↑ | 6.8% ± 1.5% | 7.5% ± 0.4% | 8.3% ± 0.6% | 17.2% ± 1.2% | 16.3% ± 2.2% | 18.7% ± 1.0% | 18.1% ± 2.4% | 19.4% ± 1.4% |

Height/weight error ×: typical factor off (1.50 = off by 50% either way). Ability recall@5: share of the true abilities among the top 5 guesses (Gen 3+ games).
