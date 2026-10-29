# PokeProject: notes for Claude Code

Two projects in one repo, sharing data and evaluation:

1. **Type classifier** (course deliverable, done): image → Pokémon type(s). Final model: 5-fold ConvNeXt-Tiny ensemble, calibrated; one-time Gen 5 test mAP 0.424. Released as GitHub Release `v1.0`; the submitted state is tagged `course-final` (both on `6eb3817`). Results and app instructions: [README.md](README.md).
2. **Dex generator** (in progress): made-up flavor text (and later a browser doodle) → full Pokédex entry with a generated Gen V-style sprite.

Read first: [docs/roadmap.md](docs/roadmap.md) (plan and status), [PROJECT_BRIEF.md](PROJECT_BRIEF.md) (classifier design), [docs/dex_decisions.md](docs/dex_decisions.md) (Dex decisions D1–D55 with status), [runs/r3/decisions.md](runs/r3/decisions.md) (round 3 sweep).

## Where things stand (2026-10-29)

The project now lives on this machine (the "nextstep" hub, WSL2 Debian on the desktop with the RTX 4090); the laptop session is closed. Roadmap sections 1–2 are done. `data/` and the `runs/` weights are on the hub and hash-verified (item 4). **After the checklist below, the house rules apply to this project and roadmap section 3 (shared image/text embedding) is next.**

### Hub setup: the single checklist, in order

Nothing below has been carried out unless it says done. "User" = Grayson; "Claude" = Claude Code on the hub, inside its sandbox unless the item says otherwise.

1. **Environment.**
   - *Involves:* apt packages, a Python 3.13 venv, the CUDA build of PyTorch, then the rest of `requirements.txt`:
     ```bash
     sudo apt install tmux build-essential cmake   # tmux: runs survive a dropped SSH; the others build llama-cpp-python
     cd ~/code/pokeproject
     python3 -m venv .venv && . .venv/bin/activate
     pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
     pip install -r requirements.txt
     python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
     ```
     `requirements.txt` is unpinned and was last installed on Python 3.11 (laptop); record here any package that fails or resolves oddly on 3.13. `llama-cpp-python` builds from source: record whether it builds. Never install a Linux NVIDIA driver in WSL: the Windows driver (610.88, CUDA 13.3) is passed through, and the pip wheels carry their own CUDA runtime.
   - *Who:* user runs the `sudo apt` line (sudo password). Claude can create the venv and run the pip installs (network: pypi.org, files.pythonhosted.org, download.pytorch.org), with approval. The final CUDA check has to run outside the sandbox (item 2), so the user runs it.
   - *Done when:* `torch.cuda.is_available()` is `True` and the device name is the RTX 4090, run outside the sandbox; install notes (3.13 issues, llama-cpp-python build result) recorded here.
2. **GPU access for Claude Code.** The sandbox blocks the GPU. Ask; don't work around it.
   - *Options:*
     - **(a) Allow the GPU in this project's sandbox settings.** Exposes the WSL GPU interface (`/dev/dxg` and the driver libraries in `/usr/lib/wsl/lib`) to sandboxed commands; filesystem and network limits stay as they are. Not yet confirmed that the sandbox settings support a device passthrough: check the Claude Code docs before deciding.
     - **(b) Approve training commands to run outside the sandbox.** Each approved command runs with the user's full permissions: everything the user can read (including `~/.ssh`, `~/private`) and unrestricted network. Approval is per command, so a long sweep is one approval, but its Python code then runs unconfined.
     - **(c) The user launches GPU jobs in tmux; Claude writes the commands and reads the logs in `runs/`.** Exposes nothing new; costs the user a step per run.
   - *Recommendation:* (a) if the settings support it, since it adds only the GPU; otherwise (c). (b) is the widest exposure for the least benefit.
   - *Who:* user decides and changes the settings; Claude can check the docs for (a).
   - *Done when:* the chosen route is recorded here and the item 1 CUDA check passes through it.
3. **Backup choice for `runs/` (user's call; decide before tonight's backup runs).** The nightly restic job excludes `pokeproject/{data,transfer,space,release}` but not `runs/`, which now holds 4.26 GB of weights (528 files: `*.pt`, `*.safetensors`, `*.np[yz]`), so they'd be backed up and copied to the laptop every night.
   - *Options:* exclude `runs/**/*.pt`, `*.safetensors`, `*.np[yz]` (final classifier weights are on GitHub Release v1.0); or keep backing them up (a lost round-3 sweep is ~1.5 days of 4090 time to rerun).
   - *Who:* user decides; the restic config lives in the nextstep project.
   - *Done when:* the choice is recorded here and, if excluding, the backup config is changed.
4. ~~**Data check.**~~ **Done 2026-10-29.** Every gitignored file under `data/` and `runs/` was sent over the tunnel in verified chunks; the hub session then re-ran the full check independently: 240,648 files, 0 mismatches. The five Dex files from the first, interrupted transfer (`runs/dex/gen5_heldout/{dex.pt,concepts.npz,words.npz}`, `runs/dex/smoke/{dex.pt,concepts.npz}`) aren't in any manifest; they pass a zip integrity check but weren't hash-compared with the laptop.
   - *Re-check:* `sha256sum --quiet -c transfer/hub/*.sha256` (prints nothing when every file matches; ~2 minutes). It reads every byte under `data/`, so running it is the user's call under the house rules.

   | Batch | Files | Size | What |
   |---|---|---|---|
   | train | 170,355 | 1.69 GB | `data/processed`, `data/dex`, `data/nltk_data`, `data/manifest.csv`, `data/species.csv`, `runs/dex*` weights |
   | weights | 522 | 3.83 GB | classifier weights and prediction dumps in `runs/` (round 3 included) |
   | raw | 69,765 | 16.50 GB | `data/raw` (source sprites, for `evaluate*.py` and `build_space.py`) |
   | models | 6 | 9.60 GB | `data/models` (Llama 3.1 8B and Qwen 2.5 7B, Q4_K_M GGUF; naming only, and Qwen may be unused) |

   `transfer/`, `space/` and `release/` were not moved (`release/` is published as v1.0).
5. **GPU smoke test** (after 1 and 2).
   - *Involves:* inside tmux, `ollama stop qwen3:32b`, then the round-1 smoke run from [docs/desktop_training.md](docs/desktop_training.md) section 5, adapted for Linux:
     ```bash
     python scripts/train.py --arch small_cnn --name smoke --limit 512 --epochs 2 --warmup 1
     rm -r runs/smoke
     ```
     Only `runs/smoke`: `runs/dex/smoke` is a different (Dex) run and stays.
   - *Who:* Claude or the user, depending on item 2.
   - *Done when:* the first line says `on cuda`, two epoch lines print, `runs/smoke` is written, and `runs/smoke` is then deleted.
6. **Rewrite [docs/desktop_training.md](docs/desktop_training.md) for the hub.**
   - *Involves:* SSH plus tmux, the Linux venv from item 1, the pre-run checklist (Ollama, gaming, 15 GiB RAM, ~71 GB real disk headroom on C:); drop the Windows desktop, zip bundle and SMB route.
   - *Who:* Claude drafts; the user approves.
   - *Done when:* every command in it has run as written on the hub (at least the smoke test).
7. **Old Windows-side clone** at `C:\Users\Grayson\Desktop\PokeProject` (old CUDA venv, round zips): now redundant.
   - *Involves:* deleting it, which is the user's call. WSL can't see C: (automount is off on purpose), so its size has to be measured on Windows, e.g. in PowerShell: `(Get-ChildItem C:\Users\Grayson\Desktop\PokeProject -Recurse -File -Force | Measure-Object Length -Sum).Sum / 1GB`.
   - *Who:* user.
   - *Done when:* deleted or kept on purpose, with the size it freed (or would free) recorded here.
8. **Privacy/security audit** against the house rules, once 1–7 are done.
   - *Involves:* what leaves the machine (Wikipedia lookups in naming, Hugging Face and PyTorch downloads, the Gradio `--share` link, GitHub pushes); what is committed (weights and data stay out of git); the claude.ai artifact links in this file; whether `data/raw` and `data/models` need any handling.
   - *Who:* user leads; Claude can prepare the inventory.
   - *Done when:* the user signs off. From then on the house rules apply here like in every migrated project, and roadmap section 3 is next.

### Before every GPU run
- **Ollama:** run `ollama stop qwen3:32b`. It shares the card and holds about 22 GB of the 24 GB VRAM while loaded (it unloads after 5 idle minutes).
- **Gaming:** the desktop is also used for games, so a game during a run competes for the GPU.
- **RAM:** WSL sees 15 GiB, so keep DataLoader workers modest (raising `memory=` in Windows `.wslconfig` is the user's option).
- **Disk:** the real headroom is ~71 GB on Windows C: (after the transfer; figure from the laptop session, 2026-10-29), not what `df` shows. The virtual disk grows and doesn't shrink on its own.
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
