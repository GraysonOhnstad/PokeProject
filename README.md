# PokeProject: predicting Pokémon types from images

A PyTorch classifier that predicts a Pokémon's type(s) from a single image, evaluated on Pokémon it has never seen in any form: every species introduced in Generation 5 is held out, and the headline test is their Black/White sprites. Built for Skillspire's *AI and Machine Learning: Code, Train & Deploy* course.

A second, ongoing project in this repo, the **Dex generator**, turns a made-up flavor text into a full Pokédex entry (see [below](#dex-generator-work-in-progress)).

Design decisions and their rationale: [PROJECT_BRIEF.md](PROJECT_BRIEF.md). Roadmap: [docs/roadmap.md](docs/roadmap.md).

## Results

Final model: an ensemble of 5 ConvNeXt-Tiny models (ImageNet-pretrained, one per cross-validation fold), with auxiliary heads for egg group, shape and color and a set-valued output over the 171 possible type combinations, calibrated by temperature scaling. Chosen by a 5-fold family cross-validation sweep over three backbones ([runs/r3/decisions.md](runs/r3/decisions.md)).

**One-time test: 161 Gen 5 Black/White sprites** ([reports/eval_round3.md](reports/eval_round3.md)), 95% bootstrap CIs:

| Metric | Ensemble | Color histogram | Most-frequent prior |
|---|---|---|---|
| mAP | **0.424** (0.395–0.508) | 0.273 | 0.083 |
| Top predicted type is a true type | **52.2%** (44.7–60.2%) | 36.0% | 11.8% |
| Per-image F1 | **0.445** (0.391–0.501) | – | – |
| Both types exactly right | 16.1% (10.6–22.4%) | 20.5% | 5.0% |

The decision threshold was tuned for per-image F1, so the model names a second type often (69% of predictions vs 49% of real Pokémon); that maximizes F1 but costs exact-match accuracy, where the color histogram, which rarely guesses two types, scores higher.

Other findings: training with Black/White-style sprites of older Pokémon made no significant difference (all paired-bootstrap CIs include 0); the same Gen 5 Pokémon in other art styles score a little lower (F1 0.395); shiny colors (F1 0.311) and grayscale (0.200) hurt most, so color carries much of the signal. Calibration error on the test set: 0.013. Inference: 193 ms per image on CPU for the whole ensemble.

Model selection (validation, 10-model mean mAP): ConvNeXt-Tiny 0.326, ResNet-50 0.316, ResNet-18 0.307. Auxiliary heads helped every backbone; attention pooling and an earlier feature tap did not pass the pre-registered +0.01 rule (the earlier tap cut parameters by 55–75% and the train–validation gap sharply for a small loss in validation mAP: the last stage is mostly memorization).

## Run the app

The app is a Gradio page: upload an image (or pick a Gen 5 example) to get the predicted types, confidences and inference time.

1. Download `pokemon-type-classifier-app.zip` from this repo's [Releases](https://github.com/GraysonOhnstad/PokeProject/releases) and unzip it.
2. In the unzipped `pokemon-type-classifier` folder (Python 3.11):
   ```
   pip install -r requirements.txt
   pip install gradio
   python app.py --share
   ```
   `--share` prints a temporary public link; without it the app runs at http://127.0.0.1:7860. A CPU is enough.

## Reproduce

Install the environment from `requirements.txt` (CPU or CUDA command in its header). Each step reads the previous step's output; the training machine only needs `data/manifest.csv`, `data/species.csv` and `data/processed/`.

| Step | Command | Output |
|---|---|---|
| 1. Fetch data | `python scripts/fetch_pokeapi.py` | `data/raw/` (pinned commits) |
| 2. Manifest + family split | `python scripts/build_manifest.py` | `data/manifest.csv`, `data/species.csv` |
| 3. 96×96 images + masks | `python scripts/preprocess.py`, then `python scripts/make_upscaled.py` | `data/processed/` |
| 4. Baselines | `python scripts/baselines.py` | `reports/baselines.md` |
| 5. Round-3 sweep (GPU, ~1.5 days on an RTX 4090) | `python scripts/sweep3.py` | `runs/r3/` (resumable) |
| 6. Calibrate the winner | `python scripts/calibrate3.py` and `--bw` | `runs/r3/final_model.pt` |
| 7. One-time test | `python scripts/evaluate3.py` | `reports/eval_round3.md` |
| 8. Build the app | `python scripts/build_space.py --bundle runs/r3/final_model.pt` | `space/` |

Rounds 1–2 (`train.py`, `sweep.py`, `calibrate.py`, `evaluate.py`) are kept for the record; [docs/desktop_training.md](docs/desktop_training.md) is the step-by-step guide for the training machine.

## Dex generator (work in progress)

Type a made-up flavor text and get a full Pokédex entry: name, category, types, height and weight, stat line, abilities and level-up learnset (sprites, cries and a Pokédex voice are planned). Each field comes from its own model, conditioned on the ones before it; names are built from what the description is about, found by a local language model and looked up on Wikipedia. Status, results and the decision log: [docs/roadmap.md](docs/roadmap.md), [docs/dex_decisions.md](docs/dex_decisions.md), `reports/dex_*.md`.

```
python scripts/build_dex.py      # dataset
python scripts/dex_final.py      # train the entry models
python scripts/dex_generate.py "It hides in burrows on the reef and punches prey so fast that the water boils."
```

Naming uses Llama 3.1 8B Instruct (4-bit GGUF in `data/models/`, via llama-cpp-python) when present, and falls back to offline naming without it.

## Credits

- Data: [PokeAPI](https://pokeapi.co/) (data and sprites).
- Name origins: [Bulbapedia](https://bulbapedia.bulbagarden.net/), CC BY-NC-SA 2.5. Facts quoted in generated entries: [Wikipedia](https://en.wikipedia.org/), CC BY-SA 4.0.
- Built with Llama (Meta Llama 3.1 Community License). Word lists: WordNet, wordfreq.
- Pokémon and all related names and images are trademarks of Nintendo, Creatures Inc. and GAME FREAK inc. This is a non-commercial fan and student project, not affiliated with or endorsed by them.
