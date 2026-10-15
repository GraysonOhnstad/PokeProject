# Dex generator: text generators (Flan-T5-small, one multi-task model)

Trained on folds 1-4, evaluated on fold 0 (held-out families); Gen 5 untouched. Examples: results/dex_textgen_examples.json. Name-origin data: Bulbapedia (CC BY-NC-SA 2.5).

| Task | Metric | Value |
|---|---|---:|
| Genus | exact match | 4.5% |
| Genus | baseline: copy nearest text's genus | 4.5% |
| Name concepts | a predicted word is a canon source word | 5.8% |
| Name from canon words | exact real name | 7.0% |
| Name from canon words | character similarity to real name | 0.71 |
| Name samples | pass filters (new, ≤ 12 chars, one word) | 97.2% |
| Flavor text | types mAP of a text→type kNN: generated | 0.235 |
| Flavor text | same, on the real texts (reference) | 0.300 |
