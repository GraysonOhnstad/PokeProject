# PokeProject
The "basic" is an image classifier for Gen V 'mons for a class project. The "full" is a tool to generate a whole Dex entry from flavor text.

Design decisions and rationale: [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

## Workflow

Each step reads the previous step's output. Install the environment from `requirements.txt` (CPU or CUDA command in its header).

| Step | Command | Output |
|---|---|---|
| 1. Fetch data | `python scripts/fetch_pokeapi.py` | `data/raw/` (~16.5 GB, pinned commits) |
| 2. Manifest + split | `python scripts/build_manifest.py` | `data/manifest.csv`, `data/species.csv`, `reports/manifest_report.md` |
| 3. 96×96 images + masks | `python scripts/preprocess.py` | `data/processed/96/`, `data/processed/96_mask/` |
| 4. Baselines | `python scripts/baselines.py` | `reports/baselines.md` |
| 5. Train | `python scripts/train.py --arch resnet18 [--no-bw-style]` | `runs/<name>/` |
| 6. Calibrate (val) | `python scripts/calibrate.py --run <name>` | `runs/<name>/calibration.json`, `reliability.png` |
| 7. Final evaluation (test) | `python scripts/evaluate.py --run <name>` | `reports/eval_<name>.md` |
| 8. Build / deploy app | `python scripts/build_space.py --run <name> [--push USER/SPACE]` | `space/` |

Training only needs steps 2–3's outputs, so a training machine can skip the raw download: copy `data/manifest.csv`, `data/species.csv` and `data/processed/`.

Planned runs: `--arch resnet18` and `--arch small_cnn`, each with and without `--no-bw-style`.

Run the app locally from a built Space folder: `cd space && python app.py`.
