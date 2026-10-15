# Dex generator, phase 1: entry fields from flavor text

5-fold family cross-validation over 10458 texts of 866 species (Gen 5 held out, not used). Mean over folds, each species weighted equally; ± is the spread across folds.

| Metric | prior | tfidf-1nn | minilm-1nn | minilm-knn | minilm-mlp |
|---|---:|---:|---:|---:|---:|
| Types mAP ↑ | 0.084 ± 0.001 | 0.115 ± 0.004 | 0.158 ± 0.005 | 0.283 ± 0.006 | 0.269 ± 0.029 |
| Types top-1 ↑ | 15.7% ± 2.6% | 22.2% ± 1.0% | 30.4% ± 1.1% | 38.3% ± 0.6% | 37.3% ± 4.2% |
| Shape acc. ↑ | 21.0% ± 4.4% | 17.8% ± 1.9% | 20.0% ± 1.6% | 20.8% ± 1.8% | 23.6% ± 1.9% |
| Color acc. ↑ | 14.3% ± 3.2% | 14.5% ± 1.2% | 16.8% ± 1.8% | 18.5% ± 2.2% | 24.1% ± 3.4% |
| Egg group mAP ↑ | 0.091 ± 0.005 | 0.117 ± 0.004 | 0.130 ± 0.004 | 0.203 ± 0.012 | 0.236 ± 0.015 |
| Height error × ↓ | 1.85 ± 0.06 | 2.10 ± 0.04 | 2.04 ± 0.07 | 1.83 ± 0.05 | 1.75 ± 0.06 |
| Weight error × ↓ | 3.72 ± 0.25 | 5.00 ± 0.20 | 4.59 ± 0.26 | 3.58 ± 0.25 | 3.28 ± 0.22 |
| Stat MAE ↓ | 22.9 ± 0.5 | 29.0 ± 0.6 | 28.3 ± 0.4 | 22.9 ± 0.4 | 21.2 ± 0.3 |
| BST MAE ↓ | 98 ± 3 | 104 ± 7 | 105 ± 2 | 91 ± 2 | 84 ± 4 |
| Ability recall@5 ↑ | 6.8% ± 1.5% | 7.5% ± 0.4% | 8.3% ± 0.6% | 17.2% ± 1.2% | 19.6% ± 1.6% |

Height/weight error ×: typical factor off (1.50 = off by 50% either way). Ability recall@5: share of the true abilities among the top 5 guesses (Gen 3+ games).
