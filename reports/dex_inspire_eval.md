# Dex generator: name inspiration on held-out Pokédex texts

Fold 0, 172 species with English canon source words (Bulbapedia, CC BY-NC-SA 2.5); one text each, names masked. A hit: a canon source word of the real name (first 5 letters) is among the ideas.

Caveat: Llama 3.1 8B has read about Pokémon and may recognize a real entry; Pokémon titles are filtered, but these scores can be optimistic. Made-up texts have nothing to recall.

| Ideas | Hit rate |
|---|---:|
| Offline pools (literal, associations, context, canon vocabulary) | 25.0% |
| Basis words only | 8.1% |
| Basis words + model concepts | 14.0% |
| … + fact terms | 18.0% |
| Everything together | 32.0% |

A verified Wikipedia basis was found for 92% of texts.

Examples (real name ← canon words | basis | concepts | category options vs real):

- **Caterpie** ← caterpillar | Caterpillar, Ant | munch, odor, bristle, leaf, venom, silk | Devourer, Leaf Eater, Stinker vs Worm Pokémon
- **Metapod** ← metamorphosis, pod | Tortoise, Armadillo | tortus, carapax, armadillo, gigas, immobilitas, fortis | Tortoise, Armadillo, Shell vs Cocoon Pokémon
- **Butterfree** ← butterfly, free, patapata | Honey bee, Bumblebee | apis, mellifera, apiarius, mellivora, foragius, volans | Honey, Bee, Forager vs Butterfly Pokémon
- **Sandshrew** ← sand, shrew | Desert tortoise, Armadillo | desert, tortoise, ball, roll, hide, tough | Roller, Balladillo, Desert Ball vs Mouse Pokémon
- **Sandslash** ← sand, slash | Porcupine, Ball python | spike, ball, python, rodent, sharp, claw | Ball, Porcupine, Ball Python vs Mouse Pokémon
- **Nidoran♀** ← needle, nido | Porcupine, Adder | quill, tox, adder, rodent, horn, venom | Quill, Adder, Thorn vs Poison Pin Pokémon
- **Nidorina** ← needle, rhinoceros, ina | Porcupine, Hedgehog | porc, quill, longev, nervos, spina, rodent | Quill, Prickly, Spine vs Poison Pin Pokémon
- **Nidoqueen** ← queen, nido | Armadillo | protector, tackle, shield, mother, roll, ball | Roller, Armadillo, Tank vs Drill Pokémon
- **Clefairy** ← clef, fairy | Mountain goat | frost, crepus, lumin, nestor, cohort, cryos | Moon, Goat, Herder vs Fairy Pokémon
- **Clefable** ← clef, fable | Gerridae, Fairyfly | skim, aqua, volans, gerrid, fairy, buoy | Water Walker, Water Strider, Fairy Fly vs Fairy Pokémon
- **Oddish** ← odd, radish | Succulent plant, Eelgrass | xerophyte, rhizome, euryhaline, succus, terra, geotropism | Succulent, Burrowing, Rooted vs Weed Pokémon
- **Gloom** ← goo, glue, bloom | Skunk, Rafflesia | skunk, rafflesia, stench, odor, bloom, bloomate | Stinker, Skunk, Rafflesia vs Weed Pokémon
- **Vileplume** ← vile, plume, bloom | Rafflesia, Amorphophallus titanum | gigantia, toxica, spathe, spadix, carrion, putrida | Flower, Rafflesia, Carrion vs Flower Pokémon
- **Diglett** ← dig, let | Badger, Groundhog | burrow, tiller, soil, farm, root, digger | Farmer, Burrower, Tiller vs Mole Pokémon
- **Dugtrio** ← dug, trio | Ant, Prairie dog | triplet, sync, burrow, ant, prairie, dog | Triplet, Ant, Burrower vs Mole Pokémon
- **Bellsprout** ← bell, sprout | Limax maximus, Cone snail | flex, corrosive, limax, radula, venom, acidic | Slip, Slug, Acid vs Flower Pokémon
- **Weepinbell** ← weeping, bell | Chameleon, Anolidae | hook, chameleon, tree, branch, sleep, fall | Hook, Chameleon, Tree Sitter vs Flycatcher Pokémon
- **Victreebel** ← victory, tree, bell | Venus flytrap, Utricularia | vine, snap, carnivore, adapt, prey, swallow | Vine, Carnivore, Snaptrap vs Flycatcher Pokémon
- **Magnemite** ← magnet, mite, might | Bat, Richmond Flying Squirrels | aero, grav, lux, vol, bat, wing | Flying, Bat, Glider vs Magnet Pokémon
- **Magneton** ← magnet, ton, magnetron | Magnetotactic bacteria, Electric eel | magnet, bio, force, iron, shock, volt | Magnet, Electric, Shockwave vs Magnet Pokémon
- **Shellder** ← shell, shelter | Tortoise, Sea turtle | tongue, burrow, nocturn, shell, sea, tort | Turtle, Shell Dweller, Night Digger vs Bivalve Pokémon
- **Cloyster** ← clam, oyster, cloister | Dugong, Manatee | aquas, hydro, spina, sire, jeti, shelli | Siren, Sea Siren, Aquaslug vs Bivalve Pokémon
- **Krabby** ← crab, crabby | Sandpiper, Plover | aren, terr, squab, cosm, bill, emer | Sand Digger, Sandpiper, Shorebird vs River Crab Pokémon
- **Kingler** ← king crab, angler, fiddler crab | Tortoise, Manatee | brachy, tortus, manus | Giant, Claw, Tortoise vs Pincer Pokémon
- **Cubone** ← cub, bone | Orangutan, Gibbon | lun, pong, siam, syndact, orph, gibb | Orphan, Gibbon, Moon vs Lonely Pokémon
