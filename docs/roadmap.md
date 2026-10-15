# Roadmap: type classifier + Dex generator

Two projects that share data and evaluation but never each other's outputs.

**Ground rules**
- No model's output is ever another model's training label. Models train only on real sprites, texts and entries; they meet only at inference (one judging another's candidates).
- Same evolution-family folds everywhere (`splits/cv_folds.csv`); Gen 5 is held out for measurement until the final refit.
- Every reported number comes from a model that never trained on what it is scored on.

## 1. Course deliverable: image → type classifier
- [ ] Round 3 sweep (desktop, running): feature tap, attention, aux heads, set output, Optuna, seed re-runs.
- [ ] Calibrate the winner (`calibrate3.py`), score the Gen 5 test once (`evaluate3.py`).
- [ ] App bundle, GitHub Release with weights, Gradio share link; fill the deck's placeholders.
- [ ] Tag the submitted state (`course-final`) so it stays reproducible.

## 2. Dex text models (laptop, in progress)
Flavor text (+ era) in; each field conditioned on the ones before it:
text → types → genus, body (height, weight, shape, color, egg groups) → stat line → abilities → level-up learnset.
- [x] Per-game dataset: period-correct types, stats, abilities, level-up learnsets; texts masked by role.
- [x] Baselines and first trained heads (`dex_phase1.py`, `reports/dex_phase1.md`).
- [x] Fine-tuned text encoder: types mAP 0.354 vs 0.339 frozen; adopt for types (not yet in the generator).
- [x] Evolution stage, line length and position as extra targets for the stat model.
- [x] Stat line: BST distribution × stat-share Dirichlet conditioned on BST (`reports/dex_stats.md`).
- [x] Abilities from the existing inventory (`reports/dex_abilities.md`); learnset as a move-set model with real learn levels (`reports/dex_learnset_set.md`).
- [x] Genus and name generation v1 (`reports/dex_textgen.md`); names still the weakest field.
  Training data for name origins comes only from official or community-canon sources (Bulbapedia's per-species "Name origin" sections, CC BY-NC-SA, credited), never from automatic guessing.

- [x] Generator v1: text → full entry without sprite (`scripts/dex_generate.py`, samples in `reports/dex_samples.md`).

## 3. Shared embedding space (desktop, after section 1)
Image encoder (round 3's winning backbone) and text encoder trained together on real (sprite, text, entry) triples: contrastive alignment plus each side's own heads.
- [ ] Classifier: image-only at test time; kept only if it beats round 3 by > 0.01 CV mAP. Reported separately from the course result.
- [ ] Judge: sprite ↔ text match score, measured by retrieval on held-out families (rank a Gen 5 sprite's true text among candidates).
- [ ] Diagnostics: where text and image models disagree on real Pokémon (types that are described but not drawn, and the reverse).
- [ ] Entry heads that take an image, a text, or both: shared heads on the fused embeddings, trained with modality dropout (randomly removing the image or the text) so any combination works. Measured three ways on held-out families: image only, text only, both.
- [ ] Doodle readiness: doodles are prepared like sprites (crop, flatten, 96 px); training adds sketch-style augmentation (line art, few colors, no shading). Real check: a small set of hand-drawn Gen 5 Pokémon, drawn from memory, scored like the test set.

## 4. Generation (desktop)
- [ ] Sprite generator: diffusion LoRA on real sprites, conditioned on the entry and the shared embedding, and on the doodle when there is one (sketch-guided, e.g. a scribble ControlNet); snapped to Gen V conventions (96×96, limited palette, outline). Evaluated on Gen 5 texts by the judge and the classifier, never trained on their scores.
- [ ] Sprite-ify a doodle: the doodle is the diffusion starting point (img2img/SDEdit) with a "how much to redraw" strength slider; a sketch-guided ControlNet compared as the stricter-lines alternative.
- [ ] Sketchify transform (sprite → doodle-style drawing: wobbly outline, flat colors, no shading). Used for (1) the field-notes doodle shown beside each generated sprite, so a fake reads as fan-made at a glance, alongside a small "fan-made" label; (2) training pairs for sprite-ify (sketchified real sprite → real sprite); (3) the image encoder's doodle augmentation.
- [ ] Sample and pick: K candidate entries/sprites, ranked by text match, type/body agreement and plausibility; K kept modest so candidates can't be selected for fooling the judges.
- [ ] Cry: retrieval + pitch/time transform by size and type.
- [ ] Dex voice: open TTS voice with a device-style effect chain; a plausible recast, not an imitation.

## 5. Final refit: Gen 5 into training
Once every design choice above is fixed, retrain the Dex models, the judge and the sprite generator on all 1,025 species (Gen 5 adds ~15% of texts and its native Black/White sprites). Reported metrics stay the held-out ones from sections 2–4. The course classifier is not refit: its tab keeps the submitted, Gen-5-held-out model, so demoing it on Gen 5 sprites stays honest.

## 6. One app
One Gradio app, one process, models loaded lazily:
- **Type classifier** tab: one upload, two answers side by side: the course model (unchanged) and the improved model trained with Dex data (both Gen 5 held out), each with types, confidence, time and attention.
- **Pokédex** tab: one fake Pokédex, models refit with Gen 5.
  - Input: flavor text, a doodle drawn in the browser (Gradio sketchpad), or both; era selector. When there is a doodle, a slider under it sets how closely the sprite follows it.
  - Output: one entry (name, genus, types, stat bars, abilities, learnset, height/weight, flavor text, cry, read-aloud) whose picture has a sprite/doodle toggle.
    - Doodle view: the user's own doodle, or (text-only input) the sketchified generated sprite.
    - Sprite view: a purely generated sprite, or one sprite-ified from the doodle at the strength set on submission. "Regenerate" makes another version at a new strength or seed; every version is kept in a per-entry gallery (strength and seed recorded, favorite pinned as the entry's sprite, downloadable), so rerolling never loses a good one.
  - Flavor text: the user's own, or generated for doodle-only input (the genus/name language model fine-tuned to write Dex text from the entry and image embedding; species tokens filled with real names).

Honest about memorization: each model's genuine predictions are on Pokémon it never trained on.
- Classifier tab: an upload whose perceptual hash matches a training sprite is labeled "in the training set"; Gen 5 sprites and new drawings are genuine predictions.
- Pokédex tab: input text that closely matches a real entry gets a note naming that Pokémon.
- Originality checks on every generated entry, before sample-and-pick: name not an existing one; sprite not a near-copy of any real sprite (perceptual hash and shared-embedding similarity); stat line/learnset flagged when several fields match a real Pokémon exactly.

Runs on the desktop GPU behind a Gradio share link; on CPU, everything but sprite generation works (slowly), so the page says so instead of failing. Weights on GitHub Releases.

## 7. Later
- Evolution lines generated in the background for every entry: decide line length and the entry's position, then the other members conditioned on it (types gained or kept, profile carried through, learnsets extended, size and genus), with evolution methods and levels. Trained on real families only; kept for the single-entry fields only if it improves them on held-out families. The Pokédex can show the line, with members' sprites generated on demand.
- Signature moves and abilities.
