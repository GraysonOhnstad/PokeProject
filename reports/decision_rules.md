# Decision rules: exact match vs F1

Same final ensemble and predictions under four ways of turning probabilities into types. Thresholds tuned on 12533 out-of-fold validation images, then applied unchanged to the 161 Gen 5 test sprites. Post-hoc analysis added after the one-time test; no model changed.

| Rule | Val exact | Val F1 | Test exact | Test F1 | Test: predicts 2 types |
|---|---:|---:|---:|---:|---:|
| 2nd type if p ≥ 0.18 (tuned for F1; reported rule) | 11.7% | 0.385 | 16.1% | 0.445 | 69% |
| 2nd type if p ≥ 0.39 (tuned for exact match; baselines' protocol) | 19.1% | 0.351 | 28.0% | 0.444 | 5% |
| Never a second type | 18.2% | 0.345 | 26.7% | 0.437 | 0% |
| Most probable type combination (no tuning) | 19.3% | 0.362 | 25.5% | 0.434 | 18% |

For reference, the color histogram under the baselines' protocol: test exact 20.5%, predicting one type 100% of the time; 49% of the test Pokémon have two types.
Check: the reported rule reproduces the evaluation's test numbers (exact 0.161, F1 0.445).
