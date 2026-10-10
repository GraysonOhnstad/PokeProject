# Manifest report

Seed 0, val fraction 0.15.

**Overall: all checks pass**

## Checks

- [x] No evolution family in more than one of train/val/test: 0 families span splits
- [x] Every row has type1: 0 missing
- [x] Every image opens: 0 failed
- [x] No exact phash duplicate crosses splits: 0 pairs
- Near-duplicate pairs across splits (phash distance 1-4): 0 (listed below; review, not automatically a failure)
- Unknown pokemon_id files dropped: 26 (0.png, back/0.png, back/shiny/0.png, other/home/0.png, shiny/0.png, versions/generation-iii/emerald/0.png, versions/generation-iii/emerald/back/0.png, versions/generation-iii/emerald/back/shiny/0.png, versions/generation-iii/emerald/shiny/0.png, versions/generation-iii/ruby-sapphire/back/0.png...)
- Val type coverage (worst type's share of its species in val): 0.121

## Scope

Rows dropped, applied in order:

| reason | rows | detail |
|---|---|---|
| back view | 25827 | default 2891, black-white 2891, ultra-sun-ultra-moon 2116, scarlet-violet 1892, omega-ruby-alpha-sapphire 1818, x-y 1764 |
| low-fidelity or unofficial source (icons, Showdown, top-level defaults) | 13260 | default 2892, showdown 2770, generation-viii-icons 1113, generation-vii-icons 974, scarlet-violet-icons 946, sword-shield-icons 849 |
| form with the same typing as another kept form of its species | 2265 | black-white 427, official-artwork 426, home 401, ultra-sun-ultra-moon 218, champions 162, sword-shield 148 |
| unofficial: form introduced after the game | 947 | black-white 947 |

Alternate forms kept (one per distinct typing within a species): 106. wormadam-sandy (bug/ground), wormadam-trash (bug/steel), shaymin-sky (grass/flying), rotom-heat (electric/fire), rotom-wash (electric/water), rotom-frost (electric/ice), rotom-fan (electric/flying), rotom-mow (electric/grass), castform-sunny (fire), castform-rainy (water), castform-snowy (ice), darmanitan-zen (fire/psychic), meloetta-pirouette (normal/fighting), charizard-mega-x (fire/dragon), pinsir-mega (bug/flying), gyarados-mega (water/dark), mewtwo-mega-x (psychic/fighting), ampharos-mega (electric/dragon), aggron-mega (steel), sceptile-mega (grass/dragon), altaria-mega (dragon/fairy), audino-mega (normal/fairy), groudon-primal (ground/fire), hoopa-unbound (psychic/dark), lopunny-mega (normal/fighting), rattata-alola (dark/normal), raticate-alola (dark/normal), raichu-alola (electric/psychic), sandshrew-alola (ice/steel), sandslash-alola (ice/steel), vulpix-alola (ice), ninetales-alola (ice/fairy), diglett-alola (ground/steel), dugtrio-alola (ground/steel), meowth-alola (dark), persian-alola (dark), geodude-alola (rock/electric), graveler-alola (rock/electric), golem-alola (rock/electric), grimer-alola (poison/dark), muk-alola (poison/dark), exeggutor-alola (grass/dragon), marowak-alola (fire/ghost), oricorio-pom-pom (electric/flying), oricorio-pau (psychic/flying), oricorio-sensu (ghost/flying), necrozma-dusk (psychic/steel), necrozma-dawn (psychic/ghost), necrozma-ultra (psychic/dragon), meowth-galar (steel), ponyta-galar (psychic), rapidash-galar (psychic/fairy), slowpoke-galar (psychic), slowbro-galar (poison/psychic), farfetchd-galar (fighting), weezing-galar (poison/fairy), mr-mime-galar (ice/psychic), articuno-galar (psychic/flying), zapdos-galar (fighting/flying), moltres-galar (dark/flying), slowking-galar (poison/psychic), corsola-galar (ghost), zigzagoon-galar (dark/normal), linoone-galar (dark/normal), darumaka-galar (ice), darmanitan-galar-standard (ice), darmanitan-galar-zen (ice/fire), yamask-galar (ground/ghost), stunfisk-galar (ground/steel), zacian-crowned (fairy/steel), zamazenta-crowned (fighting/steel), urshifu-rapid-strike (fighting/water), calyrex-ice (psychic/ice), calyrex-shadow (psychic/ghost), growlithe-hisui (fire/rock), arcanine-hisui (fire/rock), voltorb-hisui (electric/grass), electrode-hisui (electric/grass), typhlosion-hisui (fire/ghost), qwilfish-hisui (dark/poison), sneasel-hisui (fighting/poison), samurott-hisui (water/dark), lilligant-hisui (grass/fighting), zorua-hisui (normal/ghost), zoroark-hisui (normal/ghost), braviary-hisui (psychic/flying), sliggoo-hisui (steel/dragon), goodra-hisui (steel/dragon), avalugg-hisui (ice/rock), decidueye-hisui (grass/fighting), tauros-paldea-combat-breed (fighting), tauros-paldea-blaze-breed (fighting/fire), tauros-paldea-aqua-breed (fighting/water), wooper-paldea (poison/ground), ogerpon-wellspring-mask (grass/water), ogerpon-hearthflame-mask (grass/fire), ogerpon-cornerstone-mask (grass/rock), clefable-mega (fairy/flying), meganium-mega (grass/fairy), feraligatr-mega (water/dragon), barbaracle-mega (rock/fighting), chimecho-mega (psychic/steel), absol-mega-z (dark/ghost), staraptor-mega (fighting/flying), garchomp-mega-z (dragon), golisopod-mega (bug/steel)

## Split sizes

| split | families | species | forms | images | primary filter* |
|---|---|---|---|---|---|
| train | 389 | 726 | 807 | 20744 | 10280 |
| val | 69 | 140 | 152 | 3944 | 1938 |
| test | 82 | 156 | 169 | 2678 | 1409 |
| exclude | 3 | 3 | 3 | 33 | 17 |

*front view, normal colors, not a duplicate.

Primary test subset (test, black-white, front, normal, unflagged): 161 images of 158 forms / 156 species.

## Training variants (primary filter)

BW-style sources (`bw_style`): black-white.

| variant | train | val |
|---|---|---|
| with BW style | 10280 | 1938 |
| without BW style | 9783 | 1840 |

## Type distribution per split (species, by default-form types)

| type | train | val | test |
|---|---|---|---|
| water | 117 | 19 | 17 |
| normal | 96 | 18 | 17 |
| grass | 93 | 14 | 20 |
| flying | 73 | 18 | 18 |
| psychic | 73 | 14 | 15 |
| bug | 62 | 12 | 18 |
| poison | 62 | 14 | 7 |
| fairy | 54 | 8 | 2 |
| rock | 54 | 10 | 10 |
| fire | 53 | 13 | 15 |
| ground | 53 | 10 | 11 |
| electric | 51 | 7 | 11 |
| fighting | 50 | 8 | 15 |
| dark | 45 | 7 | 16 |
| ghost | 45 | 9 | 9 |
| dragon | 44 | 16 | 10 |
| steel | 44 | 8 | 12 |
| ice | 32 | 9 | 7 |

Dual-typed share: train 52%, val 53%, test 47%

## Images per medium and split (primary filter)

| medium | train | val | test | exclude |
|---|---|---|---|---|
| 3d_render | 4968 | 935 | 917 | 13 |
| illustration | 1149 | 212 | 169 | 3 |
| pixel | 3481 | 659 | 161 | 0 |
| vector | 682 | 132 | 162 | 1 |

## Images per species (train, primary filter)

| debut gen | species | median imgs | min | max |
|---|---|---|---|---|
| 1 | 131 | 25 | 21 | 47 |
| 2 | 80 | 18 | 16 | 33 |
| 3 | 111 | 14 | 12 | 52 |
| 4 | 91 | 14 | 12 | 85 |
| 6 | 64 | 8 | 5 | 14 |
| 7 | 81 | 5 | 3 | 20 |
| 8 | 76 | 5 | 3 | 14 |
| 9 | 92 | 4 | 4 | 15 |

## Provenance flags (all rows)

| flag | rows |
|---|---|
| champions | 544 |
