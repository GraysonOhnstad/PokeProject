# Run comparison (validation only)

Val: 1978 images from held-out families, all media. "BW val": the 98 of them that are Black/White sprites (the test style).

Epoch noise = standard deviation of val mAP over the last 5 epochs of the run, a rough yardstick for how much of a difference between runs is real.

| run | best epoch / ran | val mAP | val exact | val top type | epoch noise | ECE before → after | BW val mAP | BW val exact |
|---|---|---|---|---|---|---|---|---|
| color histogram (with bw style) | — | 0.195 | 0.133 | 0.250 | — | — | — | — |
| color histogram (without bw style) | — | 0.195 | 0.135 | 0.251 | — | — | — | — |
| resnet18_bw | 9 / 19 | 0.239 | 0.131 | 0.340 | 0.006 | 0.073 → 0.040 (T=2.22) | 0.284 | 0.143 |
| resnet18_nobw | 7 / 17 | 0.236 | 0.149 | 0.349 | 0.005 | 0.069 → 0.041 (T=2.01) | 0.320 | 0.173 |
| small_cnn_bw | 2 / 12 | 0.233 | 0.063 | 0.219 | 0.011 | 0.031 → 0.032 (T=1.17) | 0.279 | 0.051 |
| small_cnn_nobw | 8 / 18 | 0.233 | 0.119 | 0.258 | 0.006 | 0.043 → 0.043 (T=1.27) | 0.336 | 0.143 |
