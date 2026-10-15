# Dex generator: name blending

Held-out families, 5 folds; 758 species whose first two canon source words are English (Bulbapedia, CC BY-NC-SA 2.5). The scorer never saw the held-out names.

Real name is among the splices (including respellings): 45.6%

| Ranking | Real name first | Real name in top 5 | Char. similarity of top pick |
|---|---:|---:|---:|
| Hand-weighted score | 11.5% | 26.8% | 0.66 |
| Learned ranker (trained on the other folds) | 7.1% | 27.3% | 0.67 |

Fold 0 only (148 species), against Flan-T5 trained on folds 1-4:

| Method | Exact | Char. similarity |
|---|---:|---:|
| Flan-T5 beam | 6.8% | 0.71 |
| Splicer + hand-weighted score | 11.5% | 0.66 |
| Splicer + learned ranker | 5.4% | 0.66 |
| Splicer + T5 candidates, scored | 10.1% | 0.66 |

Examples (real name ← source words: top picks):

- Masquerain ← masquerade, rain: marain, rairade, rade, raierade, masquerain
- Duskull ← dusk, skull: dull, duskull, dukull, dusull, duskll
- Shiinotic ← shiitake, shine: shishine, shinke, shiine, shihine, shike
- Lumineon ← luminous, neon: lumineon, luminon, luneon, lumieon, lumeon
- Dracozolt ← volt, jolt: vololt, voltolt, voltjolt, joltolt, jololt
- Bellossom ← bell, belle: bele, belll, bellell, bellbell, belell
- Wimpod ← wimp, pod: wimpod, podimp, poimp, wipod, powimp
- Chikorita ← chicory, ita: chita, chicorita, chicorta, chicita, chicoryta
- Cetitan ← cetacean, cetus: ceus, cetaceus, cetactus, cetatus, cetacetus
- Magmortar ← magma, mortar: magar, martar, mamortar, magortar, magmortar
- Morelull ← morel, lull: morull, morell, morelull, molull, morlull
- Dusclops ← dusk, cyclops: dusops, dusclops, dusklops, duskops, duclops
- Skarmory ← sky, armory: skori, skory, skarmori, skrmori, skry
- Bellsprout ← bell, sprout: sproull, bellout, sprell, bellut, sprbell
- Vespiquen ← vespa, queen: vesqueen, vespaqueen, queenspa, queespa, veen
- Tapu Bulu ← tapu, bull: tapull, tapubull, tabull, tapbull, taull
- Typhlosion ← typhoon, explosion: exploon, explon, exphoon, exploshoon, explophoon
- Slowking ← slow, king: sling, sloking, kinglow, slowing, kilow
- Staraptor ← starling, raptor: startor, stor, raling, stator, raing
- Dachsbun ← dachshund, bun: bund, bunshund, bunund, bunchshund, dabun
