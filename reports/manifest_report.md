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
- Unofficial sprites dropped (form introduced after the game): 6644; by game: default 2703, black-white 2703, showdown 1232, generation-vii-icons 4, generation-iii-icons 2
- Val type coverage (worst type's share of its species in val): 0.121

## Split sizes

| split | families | species | forms | images | primary filter* |
|---|---|---|---|---|---|
| train | 389 | 726 | 988 | 47504 | 15992 |
| val | 69 | 140 | 174 | 8800 | 2882 |
| test | 82 | 156 | 185 | 6681 | 2610 |
| exclude | 3 | 3 | 4 | 69 | 30 |

*front view, normal colors, not a duplicate.

Primary test subset (test, black-white, front, normal, unflagged): 168 images of 165 forms / 156 species.

## Training variants (primary filter)

BW-style sources (`bw_style`): black-white, default, showdown.

| variant | train | val |
|---|---|---|
| with BW style | 15992 | 2882 |
| without BW style | 15073 | 2704 |

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
| 3d_render | 5514 | 996 | 981 | 17 |
| default_sprite | 0 | 1 | 10 | 0 |
| gif | 418 | 79 | 163 | 0 |
| icon | 4527 | 777 | 934 | 8 |
| illustration | 1326 | 234 | 185 | 4 |
| pixel | 3499 | 659 | 168 | 0 |
| vector | 708 | 136 | 169 | 1 |

## Images per species (train, primary filter)

| debut gen | species | median imgs | min | max |
|---|---|---|---|---|
| 1 | 131 | 35 | 28 | 176 |
| 2 | 80 | 26 | 23 | 53 |
| 3 | 111 | 22 | 18 | 82 |
| 4 | 91 | 23 | 19 | 139 |
| 6 | 64 | 12 | 8 | 48 |
| 7 | 81 | 10 | 6 | 92 |
| 8 | 76 | 8 | 5 | 26 |
| 9 | 92 | 5 | 5 | 21 |

## Provenance flags (all rows)

| flag | rows |
|---|---|
| default_sprite | 3080 |
| showdown | 1538 |
| champions | 706 |
