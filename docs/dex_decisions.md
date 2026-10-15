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
