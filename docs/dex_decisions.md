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
