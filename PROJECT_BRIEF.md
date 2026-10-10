# Pokémon Type Classifier — Project Brief

Handoff notes for building the image dataset and model. Decisions below were settled in planning; open choices are listed at the end.

## Goal

Train an image classifier that predicts a Pokémon's type(s) from a single image, evaluated on Pokémon it has never seen in any form, drawn in an art style it never trained on.

- **Held out:** every species that debuted in Gen 5 (156 species), removed from training in all media.
- **Primary test set:** the Black/White (Gen 5) front sprites of those species, 96×96 pixel art.
- **Training data:** all media (pixel sprites, 3D-model renders, icons, artwork) of every other species.
- **Output:** per-type confidence for all 18 types, decoded into one or two predicted types with a tuned threshold.

Why Gen 5: it introduced no new evolutions or pre-evolutions of older Pokémon, so holding it out is a clean evolution-family-level split. Only 3 of its 82 families gained members later (Gens 8–9); those later members must also be excluded from training. It is also the last and most detailed pixel-art generation.

## Course requirements (must all be met)

- Classifier, built in PyTorch, at least 2 layers.
- Justified choice of architecture, loss, optimizer.
- Early stopping and/or learning-rate scheduling.
- `DataLoader` with batching, shuffling (train), and transforms.
- Test-set evaluation with standard classification metrics.
- Inference timer: seconds for a single-image prediction.
- Gradio UI: upload image → predicted label(s) + inference time.
- Deploy to Hugging Face Spaces with `app.py`, model weights, `requirements.txt` (plus a class-name mapping file).
- Instructor has approved a custom (non-torchvision) dataset.

## Data sources

1. **PokeAPI data repo**: `github.com/PokeAPI/pokeapi`, CSVs in `data/v2/csv/`. Key tables:
   - `pokemon.csv` (id, species_id, is_default)
   - `pokemon_species.csv` (generation_id, evolution_chain_id, color/shape/habitat, legendary flags)
   - `pokemon_types.csv` (pokemon_id, type_id, slot); `types.csv`
   - `pokemon_egg_groups.csv`, `pokemon_stats.csv` (useful later)
2. **PokeAPI sprites repo**: `github.com/PokeAPI/sprites`, images under `sprites/pokemon/`.
   - The repo is very large. In the cloud session, a full clone timed out and a blobless clone fetched files too slowly. On a local machine a shallow clone, or a sparse checkout of `sprites/pokemon/` only, should work. Alternatively, download individual files from `raw.githubusercontent.com/PokeAPI/sprites/master/<path>`; this worked fast with parallel requests.
   - Skip `animated/` folders and `other/showdown/back*`.

### Filename conventions

- Numeric filenames (`6.png`, `10034.png`) are `pokemon.id`. IDs ≥ 10001 are alternate forms (megas, regional forms, etc.). Use **form-specific types** from `pokemon_types.csv` by `pokemon_id`, since e.g. Alolan forms differ from the base form.
- Suffixed filenames (`201-b.png`, `666-meadow.png`) are cosmetic variants (Unown letters, Vivillon patterns, etc.) with no type change. Skip them.

### Folder → medium mapping (numeric files only)

| Folder (under `sprites/pokemon/`) | Medium | Notes |
|---|---|---|
| `versions/generation-i/*/transparent/` | pixel, gen 1 | Use transparent copies; the non-transparent files duplicate them. `gray/` = original monochrome Game Boy (color-free test set). `yellow/gbc/` = Game Boy Color palette. |
| `versions/generation-ii/*/transparent/` | pixel, gen 2 | Same: transparent copies only. Includes `shiny/`. |
| `versions/generation-iii/{ruby-sapphire,emerald,firered-leafgreen}/` | pixel, gen 3 | |
| `versions/generation-iv/{diamond-pearl,platinum,heartgold-soulsilver}/` | pixel, gen 4 | Includes `female/` variants. |
| `versions/generation-v/black-white/` | pixel, gen 5 | **Primary test style.** Contains all species up to the ~1,343 IDs; check which entries beyond #649 are genuine game sprites vs. later additions. |
| `versions/generation-vi/{x-y,omega-ruby-alpha-sapphire}/` | 3D-model render | |
| `versions/generation-vii/{ultra-sun-ultra-moon,lets-go-pikachu-lets-go-eevee}/` | 3D-model render | |
| `versions/generation-ix/champions/` | 3D-model render | Provenance unverified; flag. |
| `versions/generation-{iii..viii}/icons/`, `.../sword-shield/icons/`, `.../legends-arceus/icons/`, `.../brilliant-diamond-shining-pearl/icons/`, `.../scarlet-violet/icons/` | icon | Tiny menu icons. |
| `other/home/` | 3D-model render (high-res) | Same 3D models as X/Y, ORAS, USUM renders → same source group. |
| `other/official-artwork/` | illustration | Hand-drawn. |
| `other/dream-world/` | vector illustration (SVG) | Needs rasterizing. |
| `other/showdown/` | GIF | Use first frame. Provenance varies by generation; flag. |
| top-level `sprites/pokemon/*.png` | default sprite | Likely duplicates Black/White for #1–649; later IDs of uncertain provenance. Flag; dedupe by hash. |

Variant subfolders throughout: `back/`, `shiny/`, `female/`, and combinations.

## Manifest schema (one row per image)

| Column | Meaning |
|---|---|
| `path` | Relative image path |
| `pokemon_id`, `species_id`, `species_name` | From `pokemon.csv` |
| `is_default_form` | False for megas, regional forms, etc. |
| `debut_generation` | Species' `generation_id` |
| `evolution_chain_id` | Family, for splitting |
| `type1`, `type2` | Current typings, form-specific; `type2` empty for single-typed |
| `medium` | pixel / 3d_render / icon / illustration / vector / gif / default_sprite |
| `game` | e.g. `black-white`, `home`, `official-artwork` |
| `sprite_generation` | Generation of the game/art, distinct from debut generation |
| `view` | front / back |
| `color_variant` | normal / shiny / gray / gbc |
| `is_female_variant` | bool |
| `source_group` | Groups near-identical sources, e.g. all 3D-model renders of one form share a group |
| `phash` | Perceptual hash for near-duplicate detection |
| `width`, `height` | Original size |
| `provenance_flag` | Set for champions, showdown, top-level default sprites |
| `split` | train / val / test / exclude |

Also produce a species-level table: types, egg groups, shape, color, habitat, stats, genus, debut gen, family, legendary/mythical. It's cheap and needed later.

## Split rules (deterministic, seeded)

1. **test:** every image of every species with `debut_generation == 5`.
2. **exclude:** every image of non-Gen-5 species that share an `evolution_chain_id` with a Gen 5 species (the 3 later-added family members).
3. **val:** about 15% of the remaining evolution families, chosen at random with a fixed seed. Consider stratifying so rare types appear.
4. **train:** everything else.
5. Default training filter: `view == front`, `color_variant == normal`. Back, shiny, gray, and gbc rows stay in the manifest for out-of-distribution tests.
6. **Primary test subset:** `split == test`, `game == black-white`, `view == front`, `color_variant == normal`.

### Required checks

- No `evolution_chain_id` appears in more than one of train/val/test.
- No near-duplicate `phash` crosses splits.
- Every row has `type1`; type distribution reported per split.
- Every image opens.
- Count images per species. Gen 1 species have far more images than Gen 9; plan per-species weighting or sampling.

## Modeling decisions already made

- **Labels:** 18 sigmoid outputs, binary cross-entropy loss (multi-label; dual types).
- **Decoding:** always predict the top type; add a second type only if its confidence clears a threshold tuned on validation. Never output zero or three types.
- **Calibration:** temperature scaling on validation; report reliability plots before and after.
- **Metrics:**
  - exact-match accuracy (type set fully correct)
  - per-type precision/recall/F1
  - mean average precision
  - baselines: most-frequent-type prior, color-histogram classifier
- **Validation must mirror test:** use held-out families for early stopping and threshold tuning, not random training images.
- **Pixel-art preprocessing:** alpha-composite onto a fixed background, nearest-neighbor resizing, horizontal flip OK. No color jitter, since color carries much of the type signal.

## Open decisions

- **Harmonizing media at 96×96:**
  - plain downscale of renders/artwork, or
  - pixel-art-style downscale with palette reduction, or
  - both, as an ablation.
- **Backbone:** small CNN from scratch vs. fine-tuned pretrained model.
- **Which flagged sources to include:** champions, showdown, top-level default sprites.
- **Out-of-distribution tests for the report:**
  - Gen 5 species in other media
  - shinies
  - monochrome Game Boy sprites

## Later (personal project)

Pokédex-entry generation from made-up flavor text. Planned architecture:

- an autoencoder over whole entries (types, stats, learnset, abilities)
- a text encoder mapping flavor text into the same latent space

Data: `pokemon_species_flavor_text.csv` (English = language_id 9), `pokemon_moves.csv`, `pokemon_abilities.csv`, `pokemon_stats.csv`. Split by evolution family; mask species names in text.
