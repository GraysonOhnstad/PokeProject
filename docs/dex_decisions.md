# Dex generator: decisions made without review

Decisions I made while building, each with its rationale and how to reverse it. Newest at the bottom.
Status: **kept** = waiting for your review; **kept** / **changed** once you've looked.

## Data

**D1. Entries use default forms only (no regional forms, Megas, Rotom forms).** *kept*
Flavor text is per species; only 249 form-specific texts exist. Forms would add entries whose text is shared with the base form, blurring what the text predicts. Reverse: add `pokemon_form_flavor_text` rows and form entries in `build_dex.py`.

**D2. Learnsets are level-up only; tutor moves are excluded too.** *kept*
You excluded egg and TM/HM moves as noise. Tutor moves are the same kind of noise (broad, type-driven lists that vary by game), so I excluded them as well. Moves learned on evolution (level 0) and at level 1 are kept.

**D3. Gen 1 entries use one Special value for both SpA and SpD.** *kept*
That's what the games had. The alternative (modern SpD values in Gen 1 entries) would contradict the era.

**D4. Era = generation (1–9), not the individual game.** *kept*
Games within a generation differ little in the fields we model, and generation keeps each era's data large enough to learn from. Pokémon introduced later in a generation's family (e.g. Pichu, added in Gen 2) count as part of their line in every era.

**D5. Each species counts equally in losses and metrics.** *kept*
Gen 1 species have ~15 games of texts, Gen 9 species 1–3. Without weighting, models and scores would mostly reflect early generations.

## Text

**D6. Text encoder: all-MiniLM-L6-v2 (22M parameters).** *kept*
Small enough to fine-tune on the laptop CPU (~30 min per fold) and serve fast; a standard sentence encoder. A larger encoder (e.g. bge-base, 110M) is a later comparison once the desktop is free.

**D7. For the frozen encoder, role tokens are rendered as words** (`<self>` → "this Pokémon", `<water fish pokémon>` → "water fish pokémon"). *kept*
A frozen model never saw the tokens; plain words carry their meaning. The fine-tuned encoder uses the same rendering so the two are comparable.

## Training and evaluation

**D8. Early stopping per field, on 15% of the training families.** *kept (discussed)*

**D9. Evolution position as data: stage, line length, is-final.** *kept*
Stage counts from the family's first member (babies are stage 1). Line length is the longest line in the family, so Pikachu's line is 3 because of Pichu.

## Models

**D10. Stat model: BST as a 6-component Gaussian mixture; profile as a Dirichlet over stat shares, conditioned on BST.** *kept (discussed)*
Sampled stats are rounded and clipped to 1–255. The mixture lets "godlike" text put its weight on the legendary cluster, not shift an average.

**D11. Extra stat-model targets: evolution stage, line length, is-final, rarity (normal / legendary / mythical).** *kept*
They teach the trunk the cues that set power level. They are training targets only; the generator never needs them as inputs.

**D12. Chained models are trained on true upstream fields and evaluated two ways.** *kept*
"Oracle" (true upstream) measures each step on its own; "pred" (upstream predicted from text) measures what the chain delivers. Training on true fields is the standard choice (teacher forcing); if the gap is large, training on predicted fields is the fix to try.

**D13. Abilities: three slot heads (primary; secondary or none; hidden or none, Gen 5+), sampled in order without repeats.** *kept*
Matches how the games store abilities, so a generated entry always has a valid slot layout.

**D14. Learnset generator: small transformer (4 layers, d=256) writing (level, move) pairs in level order.** *superseded by D32*
Constraints applied while sampling: grammar, non-decreasing levels, no repeated move, only moves that exist in the entry's era (a move the data shows in a game is always allowed there). Trained on every training entry from every game; text input is the species' mean text embedding, so games without English text still train it.

**D15. Learnset evaluation uses one entry per held-out species (its latest game) and 16 samples.** *kept*
Sampling every entry would take hours on CPU; latest-game entries are the era the app will default to.

**D16. Name origins: Bulbapedia "Name origin" sections, fetched once through the MediaWiki API (21 requests, 2 s apart).** *kept*
Source words are the terms Bulbapedia itself marks (italics, Wikipedia links) in the paragraphs about the English name. Some pages also mark related trivia (Snorlax: "The Simpsons"), so the first three are used as the name's concepts. 20 species have none (names that are plain words, e.g. Slowpoke, and the Paradox names). Credit Bulbapedia (CC BY-NC-SA 2.5) wherever the data is shown.

**D17. Text generators: one Flan-T5-small, four tasks by prefix (genus, concepts, name, flavor).** *kept*
Small enough for CPU training overnight; one model shares what it learns about the Dex across tasks. A larger T5 is a desktop job later. Role tokens become [self], [water fish pokémon], since T5's vocabulary has no "<".

**D18. Text generators are evaluated on one split (train folds 1–4, test fold 0), not 5-fold.** *kept*
Fine-tuning T5 five times would take most of a day on CPU. The fold-0 numbers are less certain than the 5-fold ones elsewhere; treat small differences as noise.

**D19. Name generation pipeline: concepts → 8 sampled names → filters (not an existing Pokémon name, 3–12 characters, one word).** *superseded by D35, D36, D39*
No real-word filter yet (needs a dictionary); the judge and sample-and-pick come later.

## Results-driven decisions

**D20. Stat model for text-only input: the text+aux variant; the upstream-conditioned variant only when upstream fields are concrete.** *kept*
5-fold results (`reports/dex_stats.md`): sampling gives realistic stat lines (lopsidedness KS 0.13 vs 0.79 for averaged lines). True types/body raise BST rank correlation from 0.58 to 0.79, but phase-1 *predicted* (averaged) upstream fields add nothing and shrink 80% intervals to 52% coverage: the model treats fuzzy averages as facts. In the generator, upstream fields will be sampled concrete values (or come from a doodle), which is the situation the upstream variant was trained for; a chain-level check by sampling is still to do. Extra targets (stage, rarity) are kept: small, consistent gains in rank correlation and lopsidedness realism.

**D21. Sketchify v1 is rule-based image processing, not a model.** *kept*
Lines come from the silhouette plus the sprite's own locally dark pixels (outlines, eyes, mouths); fill is 5 k-means colors lightened toward white; lines and fill wobble with smooth noise and the fill is offset a few pixels. Deterministic per seed, instant, and needs no training data. Examples: `reports/figures/sketchify_examples.png`. A learned sketch style is possible later if real doodles look different.

**D22. Abilities stop early on recall@5 of held-out training families, with label smoothing 0.1.** *kept*
Found while debugging: the slot losses bottom out by epoch ~3 while held-out ranking keeps improving to epoch 11+ (recall@5 0.07 → 0.16). A 284-way softmax grows confident faster than it ranks better, so loss stopped training far too early. Stopping on the metric we care about fixes that.

## Generator v1 (`poketype/dex_pipeline.py`, `scripts/dex_generate.py`)

**D23. v1 uses the frozen MiniLM encoder with the MLP + 10-NN blend for types and egg groups.** *kept*
It was the best validated text model when the pipeline was built and needs no fine-tuned weights. If the fine-tuned encoder wins clearly, swap it in (it also needs per-field weights saved).

**D24. Discrete fields are sampled, not taken as the most likely value.** *kept*
Types and egg groups are sampled over all 1- and 2-element sets (sigmoid probabilities combined as independent, then normalized), shape and color from their softmax, height and weight as the predicted log size plus correlated noise matching the validation error. Taking the most likely values makes every entry look average, the same collapse the stat model avoids.

**D25. Evolution stage, line length and rarity for a generated entry come from the stat model's extra heads.** *kept*
The learnset generator needs them as inputs; the stat model already learned to predict them from text and body. Line length is sampled first, then the stage within it.

**D26. v1 picker: among 8 candidates, the one whose types, shape and color are most probable under the text, with a penalty when no valid name was found.** *kept*
Types are scored by the text 10-NN alone, so the picker is partly independent of the model that sampled them. A placeholder until the shared-embedding judge exists; it favors coherence with the text over novelty.

**D27. v1 text generators come from the fold-0 experiment (trained on 80% of the training pool).** *kept*
Retraining T5 on the whole pool costs another ~2 hours of CPU; the gain is modest. Do it on the desktop with the final refit.

**D28. Fine-tuned MiniLM is adopted for types only; other fields stay on frozen-encoder heads.** *kept*
5-fold result: fine-tuned + kNN reaches types mAP 0.354 (top-1 45.9%) vs 0.339 (42.5%) for the frozen MLP + kNN, which clears the +0.01 keep rule. Egg groups, body, stats and abilities show no gain beyond noise. Best epochs are 2–4, so a final fine-tune is ~2 hours on CPU or minutes on the desktop. Not yet in the generator (v1 uses frozen embeddings; D23); planned for the desktop run with the final refit.

**D29. Name concepts come from the text, not from T5: one notable word from the text or genus plus one concept retrieved by meaning from the canon name-origin vocabulary.** *changed → D35* (too literal: names reused the text's own words)
On held-out families, T5-generated concepts matched a canon source word 5.8% of the time and were often unrelated ("rhyolite" for Caterpie). Notable words from the species' own texts and genus match 31%; retrieval proposes related words ("antenna, insect, moth" for Caterpie) and the two together match 34%. T5 still does the blending, which it does reasonably (Metapod and Sandshrew rebuilt exactly from their canon words; character similarity 0.71). Notable = rarest by document frequency in the Dex texts. The concept vocabulary only holds words of species the config trained on.

**D30. Generated names must not be English words** (filter: the encoder's 30k-word vocabulary). *kept*
Without it "Cocoon", "Beetle" and "Molasses" passed as names.

**D31. Learnset model v2: the conditioning is added at every position, and a type-and-era move prior is added to the move logits in training and sampling.** *superseded by D32*
v1 ranked moves worse than the prior alone (mAP 0.117 vs 0.271) and its learnsets ignored the Pokémon's types (STAB realism KS 0.71): it overfit sequences and stopped using a conditioning that entered at one position only. With the prior as a built-in starting point, the network only has to learn adjustments (ordering, levels, species-specific moves).

**D32. Learnset model v3: a move-set model replaces the sequence model; levels come from each move's real learn levels in that era; sampled at temperature 0.5.** *kept*
5-fold results: move ranking mAP 0.263 (sequence v2: 0.165; type prior alone: 0.271), type-match realism (STAB KS) 0.087 at temperature 0.5 (sequence v2: 0.50), physical/special alignment with Atk vs SpA 0.55 (real learnsets: 0.71). The sequence model had too little data (~6k learnsets, many near-duplicates across games) to learn both which moves and their order. Remaining weakness: occasional odd picks (Seismic Toss for a caterpillar). Reports: `reports/dex_learnset_set.md`, `reports/dex_learnset_seq.md`.

**D33. Consistency rules after sampling.** *kept*
No Eggs is never paired with another egg group; No Eggs plus a total ≥ 570 makes the entry legendary; legendaries and mythicals are single-stage; moves learned "on evolution" (level 0) only appear from stage 2 on. These are facts about the games that independently sampled fields could otherwise break.

**D34. Names: reject anything that starts a real Pokémon's name or is ≥ 85% similar to one.** *kept*
"Jolte" (Jolteon) passed the earlier filters.

## Known weaknesses (v1)

- **Names** are the weakest field (good: Sagestone, Snowfly, Mansurk; weak: Doomtadulu, Joltt). Next: score the 8 sampled names (T5 likelihood, pronounceability) instead of taking the first that passes.
- **Genus** is sometimes off ("Light Pokémon" for a candle ghost is fine; "Spiral Pokémon" for a seedling is not).
- **Body fields** are sampled independently of each other (a humanoid with egg groups Ground and Monster). The shared-embedding stage, and doodles, will supply body information text lacks.
- **Abilities** can include signature abilities of specific Pokémon (Protosynthesis); handling signatures is a planned later step.
- **Stage and line length** are sampled from text cues, not from the sampled total, so a low-total Pokémon can be single-stage; real ones exist (Luvdisc), but it is more common than in the games.

**D35. Name ideas are associations, not the text's own words.** *kept* (replaces D29, after your review)
Four pools, sampled with these weights: associations of the text's key words (35%), canon name-origin words near the text's meaning (30%), words near the whole text's meaning (25%), and the text's own words (10%, a fallback). No idea may share a root with a word in the text.
- Associations: each key word's nearest words by meaning plus WordNet relations (parts, broader and narrower kinds: candle → wick, lamp; rodent → chinchilla), ranked by fit to both the word and the whole text, so a rodent text keeps "rats" and drops "computer".
- Only nouns and descriptive adjectives qualify (WordNet's most common sense), which drops "slowly", "newly", "hesitated". A blocklist keeps inappropriate words out of ideas and names.
- Each candidate tries 3 idea pairs × 8 T5 blends and keeps the name that visibly contains a piece (≥ 3 letters) of both ideas, is a blend rather than two words glued together, and is 6–10 letters.
Examples (`reports/dex_samples.md`): Ratele (rat + electrical), Dusmanor (dusk + manor), Wanderlamp (wandering + lamps), Fitturnal (warmth + nocturnal). T5-small's blending is now the limit; a larger T5 on the desktop is the next step.

**D36. Names are blended by a rule-based splicer and ranked by a name-likeness scorer; T5's blends join the candidate pool.** *kept*
The splicer joins a prefix of one idea to a suffix of the other (both orders), merging shared letters at the seam (chandelier + lure → Chandelure) and preferring syllable edges, plus Pokémon-style respellings (c → k, ck → k). The scorer is two character 4-gram models, one trained on real Pokémon names and one on English words, plus bonuses for keeping ≥ 3 recognizable letters of each idea, the seam, and 6–10 letters, and a penalty for gluing two whole words. Held-out evaluation (`reports/dex_names.md`, 758 species, the scorer never saw their names): the real name is producible 46% of the time, ranked first 11.5% (Flan-T5 alone: 6.8%) and in the top 5 27%. T5's letter-level similarity is slightly higher (0.71 vs 0.66), and adding its blends to the pool didn't change the exact-match rate. Examples: Arcturnal, Chandelure, Beartic, Sandshrew rebuilt exactly from their source words; generated: Torchaunt, Electrat, Flucturnal, Lamight.

**D37. A learned ranker (logistic regression on the same features) was tried and rejected.** *kept*
Trained to rank real names above other splices, it put the real name first only 7.1% of the time (hand-weighted: 11.5%); it learned to favor short candidates. The code stays in `poketype/dex_names.py` for reference.

**D38. The blocklist matches inside words and accepts over-blocking.** *changed → D45*
"raping" appeared as a splice candidate ("Staraptor ← starling, raptor"). Blocked words of 4+ letters now match anywhere in a candidate, which also rejects harmless names containing them (e.g. anything with "grape" or "scrape"). For a generator whose output is shown to people, losing a few harmless names is the better failure. The same check filters name ideas.

**D39. Generated names must not be in WordNet either; name ideas exclude inflected verbs and pairs that share 5+ letters.** *kept*
The encoder's 30k-word list missed "haunches", "fluffing" and "ration". Ideas like "wandering" and "sinking" made half the names end in "-ing"; "twilight" + "lighting" produced "Lighlight".

## Name inspiration (after your mantis-shrimp / dragonfly feedback)

**D40. Names start from what the text describes, found by a local instruction model and verified on Wikipedia.** *kept*
Llama 3.1 8B Instruct (4-bit GGUF, ~4.9 GB, llama.cpp) proposes the real-world basis as Wikipedia titles; each must resolve to a real article (not a disambiguation or Pokémon page) whose introduction matches the text. A failed proposal is retried by its head noun ("Electric mouse" → "Mouse"). Tested on five made-up texts against Qwen2.5 0.5B/1.5B/3B/7B and Phi-3.5-mini; only Llama 3.1 8B got all five (mantis shrimp from "punches so fast the water boils", Meganeura for the dragonfly, will-o'-the-wisp for the candle ghost). Speed: ~3–8 s per answer on the laptop CPU, well under a second on a gaming GPU. License: Llama 3.1 Community License ("Built with Llama" credit). Offline WordNet-definition matching was tried first and failed the hard cases (mantis shrimp → reef fish).

**D41. Facts are quoted only from Wikipedia, never written by the model.** *kept*
Facts are the article's sentences with extremes, numbers, deep time, folklore or odd abilities, ranked by fit with the text; the one shown is the fact closest to the chosen name ideas. Shown with "Wikipedia: <article>, CC BY-SA 4.0". All responses are cached (`data/dex/wiki_cache/`, `data/dex/llm_cache/`).

**D42. A name pairs a basis word (the creature's own name, head noun first) with an inspired concept.** *kept*
Concepts: the model's name ingredients from the description, basis and facts (55%), proper names in the facts like Meganeura (15%), and the offline association/canon pools (15% each). The basis word may come from the text itself: the creature's name is welcome even when typed literally. Without the model or offline, the generator falls back to the offline pools (D35).

**D43. The category ("the X Pokémon") comes from the model when a basis is found, with 4 canon examples in the prompt; Flan-T5 is the fallback.** *kept*
"Balloon Pokémon" for a mantis shrimp broke the coherence; the inspired options are "Puncher", "Lantern", "Boulder", "Shock".

## Rulings (2026-10-15)

- **App requirements (D40): kept.** The full Pokédex may require a GPU and internet; without them it falls back to more limited naming, and the app says which mode it is in. No paid hosting: demos run on your own machine.
- **D24: kept, with an entry gallery.** Same text + seed gives the same entry; "reroll" adds a new entry to a per-text gallery instead of replacing one, like sprite versions.
- **D26: kept** until the shared-embedding judge exists.
- **D1, D2: kept** (no forms for now; revisit with evolution lines. Tutor moves excluded).

**D44. Signature moves and abilities are never given to a generated Pokémon.** *kept (your ruling)*
Signature = held by at most 2 evolution families, counting every form and every learn method (179 of 765 moves, 131 of 284 abilities; e.g. Sketch, Sacred Fire, Wonder Guard, Multitype). Group signatures like Protosynthesis (≈10 Paradox families) are not caught by this rule. Later: a tuned version of the abilities/moves model that can either pass a former signature to a new Pokémon or create a new one when appropriate.

**D45. Name filter in two tiers.** *kept (your ruling: filtered by default, user may disable)*
Always on, with no setting: slurs, hate terms, sexual violence. On by default, user can turn off (`--unfiltered-names`; an app toggle later): sexual and profane words. Both match anywhere inside a name, so hidden words (pen + island) are caught. Name ideas always use both tiers, so turning the filter off never steers names toward crude ideas; it only stops rejecting accidental ones. Bug fixed with this: T5's blends had joined the candidate pool without the filter.

**D46. 30% of name idea pairs ignore the basis and use the offline pools.** *kept*
Held-out evaluation (`reports/dex_inspire_eval.md`, 172 real Pokédex texts): a canon source word of the real name was among the offline ideas 25% of the time, among the inspired ideas 18%, and among both together 32%. A verified basis was found for 92% of texts. Real Pokédex texts often describe something other than the design's basis (Metapod's text reads as a tortoise or armadillo), so this measures coherence with canon only roughly; there was no sign of the model recognizing real Pokémon. Made-up descriptions written to evoke a creature are the case inspiration is built for.

## Naming v3 design (agreed 2026-10-15)

**D47. Two halves, each found the same way.** *kept (your design)*
Half A = what the Pokémon is (creature, object, legend); half B = what defines it (trait, behavior, element, theme). Each: (1) the words in the text that name it, located by the model and checked to occur in the text; (2) otherwise distilled from the text; (3) Wikipedia article and its most interesting facts; (4) word forms: exact, near meaning (synonyms, related terms, Latin/Greek and other-language words), near spelling. The offline pools are the fallback only.

**D48. No fixed order of the halves.** *kept (your question, checked against canon)*
Of 313 canon names whose first two source words contain exactly one creature word, the creature comes first in 152 (Pidgeot, Weedle, Zubat) and second in 161 (Bulbasaur, Charizard, Spearow). The splicer tries both orders and the scorer chooses.

**D49. Two naming modes, chosen automatically, with the alternatives viewable.** *kept (your ruling)*
Wordplay: a blend with recognizable pieces of both halves that is not a real word. Pun: a blend (or phrase) that deliberately sounds like an existing word, phrase or cultural reference fitting the creature (Slaking ≈ slacking, Seaking ≈ seeking, Patrat ≈ pack rat, Wynaut ≈ "why not"; Bulbapedia flags 43 canon names this way). Each entry keeps the best name of each mode; the better-scoring one is shown, and an override shows the others for comparison.

## Naming v3 build (2026-10-15)

**D50. Both halves come from one model call that quotes the text where it can.** *open*
JSON answer: for "is" and "defines", the words of the description that name it (kept only if they really occur in the text) and up to 3 Wikipedia title guesses. A mention is looked up first; long descriptive mentions are reduced to content words in base form ("punches prey with club-like arms" → punch, prey, club, arm). Each half keeps a second article only if it fits nearly as well as the first. With both halves found, a name pair is strictly "is" × "defines" (plus proper names from the facts); the offline pools are used only without inspiration (replaces D46's 30% mix).

**D51. Pun mode: sound-alike search over common English words and phrases, plus the model's own pun targets, judged by the model.** *open*
Targets: the 60,000 most common English words (wordfreq) and WordNet phrases made only of common words, minus Pokémon names; per entry, the model adds 10 fitting words or idioms. A candidate is a pun when its Metaphone key is ≤ 1 edit from the phrase's, its spelling similarity (Jaro-Winkler) is ≥ 0.85 (canon puns score ≥ 0.88), it keeps ≥ 3 letters of both halves, it isn't itself a word, and the phrase isn't one of its own ingredients (a different word from the same root, like "slacking" from "slack", is allowed). On 40 canon puns with two English source words, without the model (`reports/dex_puns_eval.md`): the real name is the top pun 17.5% of the time and in the top 5 35% (Slaking ~ slacking, Patrat ~ pack rat, Seaking, Snover ~ snow covered).

**D52. Automatic mode choice: the pun is shown only when the model endorses it and its phrase fits the description (cosine ≥ 0.25); otherwise the wordplay name. Both are kept and shown as alternatives.** *open*
Known weakness: the 8B model's pun judgments are lenient (it endorsed Ghoisp ~ "ghost", Snowfox ~ "snow goose" while the wordplay alternatives Wistly and Electiny read better). Your blind ratings on the test descriptions are the planned way to tune this.

**D53. Names are generated only for the chosen entry (8 idea pairs, both modes), not for all 8 candidates.** *open*
Naming became the slowest step; the candidate picker never needed names. Speed on the laptop CPU: ~40–60 s per entry with the model, mostly model calls (cached after the first time); on a gaming GPU this should be a few seconds.
