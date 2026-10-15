# Dex generator: learnset model

5-fold family cross-validation (Gen 5 held out); true upstream fields as input. One entry per held-out species (its latest game). Mean ± spread over folds.

| Move ranking (mAP ↑) | |
|---|---:|
| Type prior (same era, shared type) | 0.271 ± 0.013 |
| Text 10-nearest-neighbor | 0.137 ± 0.007 |
| **Generator** (16 samples) | 0.117 ± 0.014 |

| Realism (1 sample vs real) | |
|---|---:|
| KS: learnset length ↓ | 0.161 ± 0.038 |
| Mean length: real / sampled | 14.1 / 13.4 |
| KS: STAB share ↓ | 0.711 ± 0.040 |
| KS: highest level ↓ | 0.146 ± 0.036 |
| Physical share vs Atk/(Atk+SpA), rank corr.: real / sampled | 0.71 / 0.17 |
| NLL per token, all held-out entries | 3.456 ± 0.013 |

Examples: results/dex_learnset_examples.json.
