# Dex generator: decisions made without review

Decisions I made while building, each with its rationale and how to reverse it. Newest at the bottom.
Status: **open** = waiting for your review; **kept** / **changed** once you've looked.

## Data

**D1. Entries use default forms only (no regional forms, Megas, Rotom forms).** *open*
Flavor text is per species; only 249 form-specific texts exist. Forms would add entries whose text is shared with the base form, blurring what the text predicts. Reverse: add `pokemon_form_flavor_text` rows and form entries in `build_dex.py`.

**D2. Learnsets are level-up only; tutor moves are excluded too.** *open*
You excluded egg and TM/HM moves as noise. Tutor moves are the same kind of noise (broad, type-driven lists that vary by game), so I excluded them as well. Moves learned on evolution (level 0) and at level 1 are kept.

**D3. Gen 1 entries use one Special value for both SpA and SpD.** *open*
That's what the games had. The alternative (modern SpD values in Gen 1 entries) would contradict the era.

**D4. Era = generation (1–9), not the individual game.** *open*
Games within a generation differ little in the fields we model, and generation keeps each era's data large enough to learn from. Pokémon introduced later in a generation's family (e.g. Pichu, added in Gen 2) count as part of their line in every era.

**D5. Each species counts equally in losses and metrics.** *open*
Gen 1 species have ~15 games of texts, Gen 9 species 1–3. Without weighting, models and scores would mostly reflect early generations.

## Text

**D6. Text encoder: all-MiniLM-L6-v2 (22M parameters).** *open*
Small enough to fine-tune on the laptop CPU (~30 min per fold) and serve fast; a standard sentence encoder. A larger encoder (e.g. bge-base, 110M) is a later comparison once the desktop is free.

**D7. For the frozen encoder, role tokens are rendered as words** (`<self>` → "this Pokémon", `<water fish pokémon>` → "water fish pokémon"). *open*
A frozen model never saw the tokens; plain words carry their meaning. The fine-tuned encoder uses the same rendering so the two are comparable.

## Training and evaluation

**D8. Early stopping per field, on 15% of the training families.** *kept (discussed)*

**D9. Evolution position as data: stage, line length, is-final.** *open*
Stage counts from the family's first member (babies are stage 1). Line length is the longest line in the family, so Pikachu's line is 3 because of Pichu.

## Models

**D10. Stat model: BST as a 6-component Gaussian mixture; profile as a Dirichlet over stat shares, conditioned on BST.** *kept (discussed)*
Sampled stats are rounded and clipped to 1–255. The mixture lets "godlike" text put its weight on the legendary cluster, not shift an average.

**D11. Extra stat-model targets: evolution stage, line length, is-final, rarity (normal / legendary / mythical).** *open*
They teach the trunk the cues that set power level. They are training targets only; the generator never needs them as inputs.

**D12. Chained models are trained on true upstream fields and evaluated two ways.** *open*
"Oracle" (true upstream) measures each step on its own; "pred" (upstream predicted from text) measures what the chain delivers. Training on true fields is the standard choice (teacher forcing); if the gap is large, training on predicted fields is the fix to try.

**D13. Abilities: three slot heads (primary; secondary or none; hidden or none, Gen 5+), sampled in order without repeats.** *open*
Matches how the games store abilities, so a generated entry always has a valid slot layout.

**D14. Learnset generator: small transformer (4 layers, d=256) writing (level, move) pairs in level order.** *open*
Constraints applied while sampling: grammar, non-decreasing levels, no repeated move, only moves that exist in the entry's era (a move the data shows in a game is always allowed there). Trained on every training entry from every game; text input is the species' mean text embedding, so games without English text still train it.

**D15. Learnset evaluation uses one entry per held-out species (its latest game) and 16 samples.** *open*
Sampling every entry would take hours on CPU; latest-game entries are the era the app will default to.

**D16. Name origins: Bulbapedia "Name origin" sections, fetched once through the MediaWiki API (21 requests, 2 s apart).** *open*
Source words are the terms Bulbapedia itself marks (italics, Wikipedia links) in the paragraphs about the English name. Some pages also mark related trivia (Snorlax: "The Simpsons"), so the first three are used as the name's concepts. 20 species have none (names that are plain words, e.g. Slowpoke, and the Paradox names). Credit Bulbapedia (CC BY-NC-SA 2.5) wherever the data is shown.

**D17. Text generators: one Flan-T5-small, four tasks by prefix (genus, concepts, name, flavor).** *open*
Small enough for CPU training overnight; one model shares what it learns about the Dex across tasks. A larger T5 is a desktop job later. Role tokens become [self], [water fish pokémon], since T5's vocabulary has no "<".

**D18. Text generators are evaluated on one split (train folds 1–4, test fold 0), not 5-fold.** *open*
Fine-tuning T5 five times would take most of a day on CPU. The fold-0 numbers are less certain than the 5-fold ones elsewhere; treat small differences as noise.

**D19. Name generation pipeline: concepts → 8 sampled names → filters (not an existing Pokémon name, 3–12 characters, one word).** *open*
No real-word filter yet (needs a dictionary); the judge and sample-and-pick come later.

## Results-driven decisions

**D20. Stat model for text-only input: the text+aux variant; the upstream-conditioned variant only when upstream fields are concrete.** *open*
5-fold results (`reports/dex_stats.md`): sampling gives realistic stat lines (lopsidedness KS 0.13 vs 0.79 for averaged lines). True types/body raise BST rank correlation from 0.58 to 0.79, but phase-1 *predicted* (averaged) upstream fields add nothing and shrink 80% intervals to 52% coverage: the model treats fuzzy averages as facts. In the generator, upstream fields will be sampled concrete values (or come from a doodle), which is the situation the upstream variant was trained for; a chain-level check by sampling is still to do. Extra targets (stage, rarity) are kept: small, consistent gains in rank correlation and lopsidedness realism.

**D21. Sketchify v1 is rule-based image processing, not a model.** *open*
Lines come from the silhouette plus the sprite's own locally dark pixels (outlines, eyes, mouths); fill is 5 k-means colors lightened toward white; lines and fill wobble with smooth noise and the fill is offset a few pixels. Deterministic per seed, instant, and needs no training data. Examples: `reports/figures/sketchify_examples.png`. A learned sketch style is possible later if real doodles look different.

**D22. Abilities stop early on recall@5 of held-out training families, with label smoothing 0.1.** *open*
Found while debugging: the slot losses bottom out by epoch ~3 while held-out ranking keeps improving to epoch 11+ (recall@5 0.07 → 0.16). A 284-way softmax grows confident faster than it ranks better, so loss stopped training far too early. Stopping on the metric we care about fixes that.

## Generator v1 (`poketype/dex_pipeline.py`, `scripts/dex_generate.py`)

**D23. v1 uses the frozen MiniLM encoder with the MLP + 10-NN blend for types and egg groups.** *open*
It was the best validated text model when the pipeline was built and needs no fine-tuned weights. If the fine-tuned encoder wins clearly, swap it in (it also needs per-field weights saved).

**D24. Discrete fields are sampled, not taken as the most likely value.** *open*
Types and egg groups are sampled over all 1- and 2-element sets (sigmoid probabilities combined as independent, then normalized), shape and color from their softmax, height and weight as the predicted log size plus correlated noise matching the validation error. Taking the most likely values makes every entry look average, the same collapse the stat model avoids.

**D25. Evolution stage, line length and rarity for a generated entry come from the stat model's extra heads.** *open*
The learnset generator needs them as inputs; the stat model already learned to predict them from text and body. Line length is sampled first, then the stage within it.

**D26. v1 picker: among 8 candidates, the one whose types, shape and color are most probable under the text, with a penalty when no valid name was found.** *open*
Types are scored by the text 10-NN alone, so the picker is partly independent of the model that sampled them. A placeholder until the shared-embedding judge exists; it favors coherence with the text over novelty.

**D27. v1 text generators come from the fold-0 experiment (trained on 80% of the training pool).** *open*
Retraining T5 on the whole pool costs another ~2 hours of CPU; the gain is modest. Do it on the desktop with the final refit.

**D28. Fine-tuned MiniLM is adopted for types only; other fields stay on frozen-encoder heads.** *open*
5-fold result: fine-tuned + kNN reaches types mAP 0.354 (top-1 45.9%) vs 0.339 (42.5%) for the frozen MLP + kNN, which clears the +0.01 keep rule. Egg groups, body, stats and abilities show no gain beyond noise. Best epochs are 2–4, so a final fine-tune is ~2 hours on CPU or minutes on the desktop. Not yet in the generator (v1 uses frozen embeddings; D23); planned for the desktop run with the final refit.

**D29. Name concepts come from the text, not from T5: one notable word from the text or genus plus one concept retrieved by meaning from the canon name-origin vocabulary.** *open*
On held-out families, T5-generated concepts matched a canon source word 5.8% of the time and were often unrelated ("rhyolite" for Caterpie). Notable words from the species' own texts and genus match 31%; retrieval proposes related words ("antenna, insect, moth" for Caterpie) and the two together match 34%. T5 still does the blending, which it does reasonably (Metapod and Sandshrew rebuilt exactly from their canon words; character similarity 0.71). Notable = rarest by document frequency in the Dex texts. The concept vocabulary only holds words of species the config trained on.

**D30. Generated names must not be English words** (filter: the encoder's 30k-word vocabulary). *open*
Without it "Cocoon", "Beetle" and "Molasses" passed as names.

**D31. Learnset model v2: the conditioning is added at every position, and a type-and-era move prior is added to the move logits in training and sampling.** *open*
v1 ranked moves worse than the prior alone (mAP 0.117 vs 0.271) and its learnsets ignored the Pokémon's types (STAB realism KS 0.71): it overfit sequences and stopped using a conditioning that entered at one position only. With the prior as a built-in starting point, the network only has to learn adjustments (ordering, levels, species-specific moves).

**D32. Learnset model v3: a move-set model replaces the sequence model; levels come from each move's real learn levels in that era; sampled at temperature 0.5.** *open*
5-fold results: move ranking mAP 0.263 (sequence v2: 0.165; type prior alone: 0.271), type-match realism (STAB KS) 0.087 at temperature 0.5 (sequence v2: 0.50), physical/special alignment with Atk vs SpA 0.55 (real learnsets: 0.71). The sequence model had too little data (~6k learnsets, many near-duplicates across games) to learn both which moves and their order. Remaining weakness: occasional odd picks (Seismic Toss for a caterpillar). Reports: `reports/dex_learnset_set.md`, `reports/dex_learnset_seq.md`.

**D33. Consistency rules after sampling.** *open*
No Eggs is never paired with another egg group; No Eggs plus a total ≥ 570 makes the entry legendary; legendaries and mythicals are single-stage; moves learned "on evolution" (level 0) only appear from stage 2 on. These are facts about the games that independently sampled fields could otherwise break.

**D34. Names: reject anything that starts a real Pokémon's name or is ≥ 85% similar to one.** *open*
"Jolte" (Jolteon) passed the earlier filters.

## Known weaknesses (v1)

- **Names** are the weakest field (good: Sagestone, Snowfly, Mansurk; weak: Doomtadulu, Joltt). Next: score the 8 sampled names (T5 likelihood, pronounceability) instead of taking the first that passes.
- **Genus** is sometimes off ("Light Pokémon" for a candle ghost is fine; "Spiral Pokémon" for a seedling is not).
- **Body fields** are sampled independently of each other (a humanoid with egg groups Ground and Monster). The shared-embedding stage, and doodles, will supply body information text lacks.
- **Abilities** can include signature abilities of specific Pokémon (Protosynthesis); handling signatures is a planned later step.
- **Stage and line length** are sampled from text cues, not from the sampled total, so a low-total Pokémon can be single-stage; real ones exist (Luvdisc), but it is more common than in the games.
