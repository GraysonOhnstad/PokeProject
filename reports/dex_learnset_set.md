# Dex generator: learnset model

5-fold family cross-validation (Gen 5 held out); true upstream fields as input. One entry per held-out species (its latest game). Mean ± spread over folds.

| Move ranking (mAP ↑) | |
|---|---:|
| Type prior (same era, shared type) | 0.271 ± 0.013 |
| Text 10-nearest-neighbor | 0.137 ± 0.007 |
| **Generator: move-set model scores** | 0.263 ± 0.006 |

| Realism (1 sample vs real) | |
|---|---:|
| KS: learnset length ↓ | 0.204 ± 0.066 |
| Mean length: real / sampled | 14.1 / 13.2 |
| KS: STAB share ↓ | 0.135 ± 0.023 |
| KS: highest level ↓ | 0.165 ± 0.030 |
| Physical share vs Atk/(Atk+SpA), rank corr.: real / sampled | 0.71 / 0.53 |
| NLL per token, all held-out entries | nan ± nan |

Examples: results/dex_learnset_set_examples.json.
