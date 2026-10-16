# Dex generator: name blending

Held-out families, 5 folds; 758 species whose first two canon source words are English (Bulbapedia, CC BY-NC-SA 2.5). The scorer never saw the held-out names.

Real name is among the splices (including respellings): 45.5%

| Ranking | Real name first | Real name in top 5 | Char. similarity of top pick |
|---|---:|---:|---:|
| Hand-weighted score | 14.5% | 29.3% | 0.68 |
| Learned ranker (trained on the other folds) | 7.9% | 28.0% | 0.67 |

Fold 0 only (148 species), against Flan-T5 trained on folds 1-4:

| Method | Exact | Char. similarity |
|---|---:|---:|
| Flan-T5 beam | 6.8% | 0.71 |
| Splicer + hand-weighted score | 16.2% | 0.68 |
| Splicer + learned ranker | 6.1% | 0.67 |
| Splicer + T5 candidates, scored | 15.5% | 0.67 |

Examples (real name ← source words: top picks):

- Masquerain ← masquerade, rain: marain, masquerain, rairade, rainde, rainerade
- Duskull ← dusk, skull: duskull, dull, dukull, dusull, duskll
- Shiinotic ← shiitake, shine: shishine, shinke, shiine, shike, shiishine
- Lumineon ← luminous, neon: lumineon, luminon, luneon, lumeon, lumieon
- Dracozolt ← volt, jolt: voltolt, voltjolt, joltolt, voljolt, voltlt
- Bellossom ← bell, belle: bele, belll, bellell, bellbell, bellebell
- Wimpod ← wimp, pod: wimpod, podimp, poimp, wipod, powimp
- Chikorita ← chicory, ita: chita, chicorita, chicita, chicorta, chikorita
- Cetitan ← cetacean, cetus: ceus, cetacetus, cetaceus, cetactus, cetatus
- Magmortar ← magma, mortar: magar, martar, mamortar, magortar, magmortar
- Morelull ← morel, lull: morull, morell, morelull, molull, morlull
- Dusclops ← dusk, cyclops: dusops, dusclops, dusklops, duskops, duclops
- Skarmory ← sky, armory: skori, skory, skarmori, skrmori, skry
- Bellsprout ← bell, sprout: sproull, bellout, sprell, bellut, sproubell
- Vespiquen ← vespa, queen: vesqueen, vespaqueen, queenspa, vespeen, veen
- Tapu Bulu ← tapu, bull: tapull, tapubull, tabull, tapbull, taull
- Typhlosion ← typhoon, explosion: exploon, explon, exphoon, exploshoon, typion
- Slowking ← slow, king: sling, sloking, kinglow, slowing, kilow
- Staraptor ← starling, raptor: startor, raling, stor, stator, raing
- Dachsbun ← dachshund, bun: bund, bunshund, dabun, bunchshund, dakun
