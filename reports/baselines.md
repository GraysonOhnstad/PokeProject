# Baselines

Val: 1978 images (held-out families, all media). Test: 161 images (primary subset: Gen 5, Black/White front sprites).

Second-type threshold tuned on val for exact match. Training images weighted so each form counts equally.

| model | train variant | split | exact match | top type correct | mAP | macro F1 | threshold |
|---|---|---|---|---|---|---|---|
| prior | with bw style | val | 0.092 | 0.152 | 0.085 | 0.015 | 0.13 |
| prior | with bw style | test | 0.050 | 0.118 | 0.083 | 0.012 | 0.13 |
| color_histogram | with bw style | val | 0.133 | 0.250 | 0.195 | 0.117 | 0.31 |
| color_histogram | with bw style | test | 0.211 | 0.366 | 0.273 | 0.150 | 0.31 |
| prior | without bw style | val | 0.092 | 0.152 | 0.085 | 0.015 | 0.13 |
| prior | without bw style | test | 0.050 | 0.118 | 0.083 | 0.012 | 0.13 |
| color_histogram | without bw style | val | 0.135 | 0.251 | 0.195 | 0.118 | 0.31 |
| color_histogram | without bw style | test | 0.205 | 0.360 | 0.273 | 0.146 | 0.31 |

## Color histogram (with BW style): per-type test results

| type | precision | recall | F1 | support |
|---|---|---|---|---|
| normal | 0.33 | 0.79 | 0.47 | 19 |
| fighting | 0.00 | 0.00 | 0.00 | 16 |
| flying | 0.12 | 0.05 | 0.07 | 19 |
| poison | 0.20 | 0.14 | 0.17 | 7 |
| ground | 0.00 | 0.00 | 0.00 | 11 |
| rock | 0.00 | 0.00 | 0.00 | 10 |
| bug | 0.00 | 0.00 | 0.00 | 18 |
| ghost | 0.00 | 0.00 | 0.00 | 11 |
| steel | 0.00 | 0.00 | 0.00 | 12 |
| fire | 0.88 | 0.44 | 0.58 | 16 |
| water | 0.27 | 0.79 | 0.40 | 19 |
| grass | 0.53 | 0.80 | 0.64 | 20 |
| electric | 0.00 | 0.00 | 0.00 | 11 |
| psychic | 0.43 | 0.19 | 0.26 | 16 |
| ice | 0.00 | 0.00 | 0.00 | 7 |
| dragon | 0.00 | 0.00 | 0.00 | 10 |
| dark | 0.50 | 0.06 | 0.11 | 16 |
| fairy | 0.00 | 0.00 | 0.00 | 2 |
