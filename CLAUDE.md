# PokeProject: notes for Claude Code

Two projects in one repo, sharing data and evaluation:

1. **Type classifier** (course deliverable, done): image → Pokémon type(s). Final model: 5-fold ConvNeXt-Tiny ensemble, calibrated; one-time Gen 5 test mAP 0.424. Released as GitHub Release `v1.0`; the submitted state is tagged `course-final` (both on `6eb3817`). Results and app instructions: [README.md](README.md).
2. **Dex generator** (in progress): made-up flavor text (and later a browser doodle) → full Pokédex entry with a generated Gen V-style sprite.

Read first: [docs/roadmap.md](docs/roadmap.md) (plan and status), [PROJECT_BRIEF.md](PROJECT_BRIEF.md) (classifier design), [docs/dex_decisions.md](docs/dex_decisions.md) (Dex decisions D1–D55 with status), [runs/r3/decisions.md](runs/r3/decisions.md) (round 3 sweep).

## Where things stand (2026-10-27)

The work is moving from a Windows laptop to this machine (the "nextstep" hub, WSL2 Debian on the desktop with the RTX 4090). Roadmap sections 1–2 are done; **section 3 (shared image/text embedding) is next**, and it can't start until the setup below is finished.

### Setup still to do, in order

1. **Hub environment.** Not run yet: the user approves installs and enters the sudo password.
   ```bash
   sudo apt install tmux build-essential cmake   # tmux: runs survive a dropped SSH; the others build llama-cpp-python
   cd ~/code/pokeproject
   python3 -m venv .venv && . .venv/bin/activate
   pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Hub Python is 3.13; the laptop's was 3.11 (torch 2.14.1 supports both). Never install a Linux NVIDIA driver in WSL: the Windows driver (610.88, CUDA 13.3) is passed through, and the pip wheels carry their own CUDA runtime.
2. **Move `data/` and the weights in `runs/` from the laptop.** Both are gitignored, so the clone here has neither (`data/` is empty). On the laptop (`C:\Users\Grayson\Desktop\PokeProject`):

   | What | Size | Needed for |
   |---|---|---|
   | `data/processed`, `data/dex`, `data/manifest.csv`, `data/species.csv`, `data/nltk_data` | 1.2 GB | training (sections 3–4) |
   | `data/raw` | 15.4 GB | `evaluate*.py`, `build_space.py` (source sprites) |
   | `data/models` (Llama 3.1 8B and Qwen 2.5 7B, Q4_K_M GGUF) | 8.9 GB | naming only (`DEFAULT_LLM`); Qwen may be unused |
   | weights in `runs/` (`*.pt`, `*.safetensors`, `*.np[yz]`; round 3 is 2.8 GB) | 4.0 GB | round 3 comparison, Dex models |

   Plan: stream over the WireGuard tunnel from the laptop (`tar -cf - <paths> | ssh nextstep-hub "tar -xf - -C ~/code/pokeproject"`; the laptop has Windows OpenSSH and tar, no rsync). It needs the laptop's tunnel up and the user's go-ahead. Verify the file counts and sizes on arrival. `transfer/`, `space/` and `release/` don't need to move (`release/` is published as v1.0).
3. **GPU access for Claude Code here.** This session's sandbox blocks the GPU. Until the user allows GPU access for this project in the sandbox settings, or approves running training commands outside the sandbox, training can't be launched from here. Ask; don't work around it.
4. **Backup decision (the user's call).** The nightly restic backup excludes `pokeproject/{data,transfer,space,release}` but not `runs/`, so checkpoints would be backed up and copied to the laptop every night. Options: exclude the weight patterns above (final weights live on GitHub Releases), or keep them, since a lost sweep is expensive to rerun.
5. Once 1–3 work, rewrite [docs/desktop_training.md](docs/desktop_training.md), which still describes the old route (Windows desktop, zip bundle over a temporary SMB share).

### Before every GPU run
- **Ollama:** run `ollama stop qwen3:32b`. It shares the card and holds about 22 GB of the 24 GB VRAM while loaded (it unloads after 5 idle minutes).
- **Gaming:** the desktop is also used for games, so a game during a run competes for the GPU.
- **RAM:** WSL sees 15 GiB, so keep DataLoader workers modest (raising `memory=` in Windows `.wslconfig` is the user's option).
- **Disk:** the real headroom is ~102 GB on Windows C:, not the 932 GB `df` shows. The virtual disk grows and doesn't shrink on its own.
- **tmux:** start long runs inside it.

## Ground rules (agreed with the user)

- **Data separation:** no model's output is ever another model's training label; models meet only at inference.
- **Splits:** same evolution-family folds everywhere (`splits/cv_folds.csv`). Gen 5 is held out for measurement until the final refit (roadmap section 5).
- **Honest numbers:** every reported number comes from a model that never trained on what it's scored on. Never state a number that isn't in the project's outputs.
- **Propose, don't build:** propose each new piece and wait for the user's approval. A technical decision stays in place until it's shown to be ineffective.
- **Decision log:** decisions made without review go in [docs/dex_decisions.md](docs/dex_decisions.md) with a rationale, how to reverse it, and a status.
- **Comparisons:** a change is kept only when it beats the reference by the pre-registered margin (e.g. > 0.01 CV mAP for the classifier).
- **Commits:** messages are one descriptive line with the key numbers. The user pushes, and decides what goes to GitHub.

## Dex generator: the user's standing decisions

Most of these are in the roadmap and decision log; these are easy to miss:
- **Fields:**
  - Learnsets are level-up only, from all generations (no egg, TM/HM or tutor moves).
  - Moves and abilities come only from the existing inventory; signature moves and abilities are excluded until a model can create new ones.
  - Sprites are always Gen V (Black/White) style; the era is selectable where the data supports it.
- **Names** are generated from two halves: what the Pokémon *is* and its *defining* trait. They're combined as a portmanteau ("wordplay") or a pun. Names must cohere closely with the description.
  - Name origins used for training come only from official or community-canon sources (Bulbapedia), never guessed ones.
  - A name filter is on by default with a user toggle; slurs and hate terms are always blocked.
- **Voice:** the TTS Dex voice is a plausible "recast", never an imitation of the anime actor.
- **Hardware:** the app may require a GPU and internet, with a labeled offline fallback. There's no paid hosting: demos run on the user's machine behind a Gradio share link.

### Naming evaluation
- **Dev set:** `evals/dex_naming_dev.jsonl` (20 descriptions, written by Claude; usable for tuning). The user's blind ratings are in `evals/dex_naming_dev_ratings.csv`, scored by `scripts/dex_naming_score.py` into [reports/dex_naming_dev_ratings.md](reports/dex_naming_dev_ratings.md).
  - Results: shown name mean 2.60/5, best of the two 2.90, and wordplay and pun about even.
  - Better names when both halves were found (a post-hoc reading). The "defines" half (12/20 found) is the weak step.
- **Holdout:** `evals/dex_naming_holdout.jsonl` (10 descriptions, written by the user) is scored **only at the end**.
- **Blind rating:** done on a private claude.ai rating page, https://claude.ai/artifact/Y8ErkEhhwnBAkVAaXYbQBQ. Its store has the `ratings` collection, one document per description, `{A, B}`.
  - Don't show the user `reports/<set>.md` before they've rated: it reveals which mode made which name.

## The user and the presentation
- **The user:** ML-literate and wants precise, information-dense answers with real terminology. The project is portfolio work.
- **The deck:** a Claude Slides deck (https://claude.ai/artifact/6QSEexwSngHgTNRGSkNY4V), exported to .pptx for Google Slides.
  - Audience: ML-curious viewers; portfolio-grade, ~20 slides.
  - Mascots: Gastly, Squirtle, Magikarp and Eevee lines, with slide colors from the featured mascot.
  - Numbers on slides come only from project outputs.
