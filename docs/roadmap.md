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
- [ ] Fine-tuned text encoder (running).
- [ ] Stat line: BST distribution × stat-share Dirichlet conditioned on BST; judged by BST rank correlation, interval coverage and realism, not MAE.
- [ ] Abilities from the existing inventory; learnset generator (autoregressive over existing moves, with levels).
- [ ] Genus and name generation (concept extraction → pun; filters: not an existing name or real word, pronounceable, ≤ 12 characters).

## 3. Shared embedding space (desktop, after section 1)
Image encoder (round 3's winning backbone) and text encoder trained together on real (sprite, text, entry) triples: contrastive alignment plus each side's own heads.
- [ ] Classifier: image-only at test time; kept only if it beats round 3 by > 0.01 CV mAP. Reported separately from the course result.
- [ ] Judge: sprite ↔ text match score, measured by retrieval on held-out families (rank a Gen 5 sprite's true text among candidates).
- [ ] Diagnostics: where text and image models disagree on real Pokémon (types that are described but not drawn, and the reverse).

## 4. Generation (desktop)
- [ ] Sprite generator: diffusion LoRA on real sprites, conditioned on the entry and the shared embedding; snapped to Gen V conventions (96×96, limited palette, outline). Evaluated on Gen 5 texts by the judge and the classifier, never trained on their scores.
- [ ] Sample and pick: K candidate entries/sprites, ranked by text match, type/body agreement and plausibility; K kept modest so candidates can't be selected for fooling the judges.
- [ ] Cry: retrieval + pitch/time transform by size and type.
- [ ] Dex voice: open TTS voice with a device-style effect chain; a plausible recast, not an imitation.

## 5. Final refit: Gen 5 into training
Once every design choice above is fixed, retrain the Dex models, the judge and the sprite generator on all 1,025 species (Gen 5 adds ~15% of texts and its native Black/White sprites). Reported metrics stay the held-out ones from sections 2–4. The course classifier is not refit: its tab keeps the submitted, Gen-5-held-out model, so demoing it on Gen 5 sprites stays honest.

## 6. One app
One Gradio app, one process, models loaded lazily:
- **Type classifier** tab: the course model, unchanged (upload image → types, confidence, time, attention).
- **Pokédex** tab: flavor text + era → a Pokédex-style page (sprite, name, genus, types, stat bars, abilities, learnset, height/weight, cry, read-aloud).
- **Draw your own** tab (optional): upload a drawing → types and body from the image → the rest of the entry.

Honest about memorization: each model's genuine predictions are on Pokémon it never trained on.
- Classifier tab: an upload whose perceptual hash matches a training sprite is labeled "in the training set"; Gen 5 sprites and new drawings are genuine predictions.
- Pokédex tab: input text that closely matches a real entry gets a note naming that Pokémon.
- Originality checks on every generated entry, before sample-and-pick: name not an existing one; sprite not a near-copy of any real sprite (perceptual hash and shared-embedding similarity); stat line/learnset flagged when several fields match a real Pokémon exactly.

Runs on the desktop GPU behind a Gradio share link; on CPU, everything but sprite generation works (slowly), so the page says so instead of failing. Weights on GitHub Releases.

## 7. Later
Evolution lines (family generation, evolution method and level), then signature moves and abilities.
