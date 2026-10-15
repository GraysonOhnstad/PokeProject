"""Build the Pokedex-entry dataset: one entry per (species, game) and its flavor texts.

Each mainline game group with a level-up learnset gives one entry per default-form
species, with that game's types, base stats, EV yield, abilities and level-up
learnset (egg, tutor and TM moves left out). Past values come from PokeAPI's
*_past tables, which list only what changed and the last generation it applied to.
Flavor texts are cleaned, names are masked by role (poketype/dex_text.py), and each text is
paired with the entry from its own game.

    py scripts/build_dex.py

Writes data/dex/: species.csv (per-species fields), entries.jsonl (per game),
texts.csv (one row per unique text per game) and vocab.json.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.dex_text import Masker, clean_text, species_token  # noqa: E402

CSV = ROOT / "data" / "raw" / "csv"
OUT = ROOT / "data" / "dex"

STATS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]
SPECIAL = 9  # Gen 1's single Special stat
# Spin-offs and Japan-only releases with no English flavor text of their own.
SKIP_VERSION_GROUPS = {"colosseum", "xd", "red-green-japan", "blue-japan"}
LEVEL_UP = 1
SIGNATURE_FAMILIES = 2  # a move or ability held by this many evolution families or fewer is "signature"


def read(name):
    return pd.read_csv(CSV / f"{name}.csv")


def english(df, id_col, value_col="name"):
    return df[df.local_language_id == 9].set_index(id_col)[value_col]


def past_value(past, key, gen, current):
    """Value in force in `gen`: the earliest past record covering it, else `current`."""
    later = [(g, v) for g, v in past.get(key, []) if g >= gen]
    return min(later, key=lambda r: r[0])[1] if later else current


def evolution_positions(spec):
    """Per species: stage (1 = unevolved), line_length (stages in its family's longest line),
    is_final (nothing evolves from it). Babies count as stage 1, as in the games' own lines."""
    parent = {s: (None if pd.isna(p) else int(p)) for s, p in zip(spec.species_id, spec.evolves_from_species_id)}
    children = {}
    for s, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(s)

    def stage(s):
        return 1 if parent[s] is None else 1 + stage(parent[s])

    stages = {s: stage(s) for s in parent}
    longest = spec.assign(stage=spec.species_id.map(stages)).groupby("evolution_chain_id").stage.max()
    return pd.DataFrame({
        "stage": stages,
        "line_length": {s: int(longest[c]) for s, c in zip(spec.species_id, spec.evolution_chain_id)},
        "is_final": {s: s not in children for s in parent},
    }).rename_axis("species_id")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    species = read("pokemon_species")
    pokemon = read("pokemon")
    default = pokemon[pokemon.is_default == 1].set_index("species_id")
    vg = read("version_groups").set_index("id")
    vg = vg[~vg.identifier.isin(SKIP_VERSION_GROUPS)]
    versions = read("versions").set_index("id")
    names = english(read("pokemon_species_names"), "pokemon_species_id")
    genus = english(read("pokemon_species_names"), "pokemon_species_id", "genus")
    old = pd.read_csv(ROOT / "data" / "species.csv").set_index("species_id")
    folds = pd.read_csv(ROOT / "splits" / "cv_folds.csv").set_index("evolution_chain_id").fold
    test_chains = set(species[species.generation_id == 5].evolution_chain_id)

    # Per-species fields that don't change between games.
    sp = species.set_index("id")
    rows = []
    for sid, s in sp.iterrows():
        p = default.loc[sid]
        chain = s.evolution_chain_id
        rows.append({
            "species_id": sid, "pokemon_id": int(p.id), "name": names[sid], "genus": genus.get(sid),
            "debut_generation": s.generation_id, "evolution_chain_id": chain,
            "evolves_from_species_id": s.evolves_from_species_id,
            "height_dm": p.height, "weight_hg": p.weight,
            "color": old.color.get(sid), "shape": old["shape"].get(sid), "habitat": old.habitat.get(sid),
            "egg_group1": old.egg_group1.get(sid), "egg_group2": old.egg_group2.get(sid),
            "gender_rate": s.gender_rate, "capture_rate": s.capture_rate,
            "base_happiness": s.base_happiness, "hatch_counter": s.hatch_counter,
            "growth_rate_id": s.growth_rate_id, "is_baby": s.is_baby,
            "is_legendary": s.is_legendary, "is_mythical": s.is_mythical,
            "split": "test" if chain in test_chains else "train",
            "fold": -1 if chain in test_chains else int(folds[chain]),
        })
    spec = pd.DataFrame(rows)
    spec = spec.join(evolution_positions(spec), on="species_id")
    spec.to_csv(OUT / "species.csv", index=False)

    # Current values, plus past overrides keyed by the last generation they applied to.
    pid_of = spec.set_index("species_id").pokemon_id
    pids = set(pid_of)
    types_now = read("pokemon_types").query("pokemon_id in @pids")
    type_name = read("types").set_index("id").identifier
    types_now = types_now.sort_values("slot").groupby("pokemon_id").type_id.apply(list)
    types_past = {}
    for (pid, g), grp in read("pokemon_types_past").groupby(["pokemon_id", "generation_id"]):
        types_past.setdefault(pid, []).append((g, grp.sort_values("slot").type_id.tolist()))

    st = read("pokemon_stats").query("pokemon_id in @pids")
    stat_id = read("stats").set_index("identifier").id
    stats_now = {(r.pokemon_id, r.stat_id): (r.base_stat, r.effort) for r in st.itertuples()}
    stats_past = {}
    for r in read("pokemon_stats_past").itertuples():
        stats_past.setdefault((r.pokemon_id, r.stat_id), []).append((r.generation_id, (r.base_stat, r.effort)))

    ab = read("pokemon_abilities").query("pokemon_id in @pids")
    abil_now = {(r.pokemon_id, r.slot): r.ability_id for r in ab.itertuples()}
    abil_past = {}
    for r in read("pokemon_abilities_past").itertuples():
        a = None if pd.isna(r.ability_id) else int(r.ability_id)
        abil_past.setdefault((r.pokemon_id, r.slot), []).append((r.generation_id, a))

    pm = read("pokemon_moves")
    pm = pm[(pm.pokemon_move_method_id == LEVEL_UP) & pm.pokemon_id.isin(pids) & pm.version_group_id.isin(vg.index)]
    learnsets = {}
    for (pid, g), grp in pm.groupby(["pokemon_id", "version_group_id"]):
        grp = grp.sort_values(["level", "order"]).drop_duplicates(["level", "move_id"])
        learnsets[(pid, g)] = [[int(lv), int(m)] for lv, m in zip(grp.level, grp.move_id)]

    entries = []
    for (pid, g), learnset in learnsets.items():
        gen = int(vg.generation_id[g])
        t = past_value(types_past, pid, gen, types_now[pid])
        stats, effort = {}, {}
        for s in STATS:
            key = (pid, stat_id[s])
            stats[s], effort[s] = (int(x) for x in past_value(stats_past, key, gen, stats_now[key]))
        if gen == 1:  # one Special stat: both special stats get it
            special = past_value(stats_past, (pid, SPECIAL), gen, None)
            if special:
                stats["special-attack"] = int(special[0])
            stats["special-defense"] = stats["special-attack"]
        abilities = {}
        if gen >= 3:  # abilities from Gen 3, hidden abilities from Gen 5
            for slot in (1, 2, 3) if gen >= 5 else (1, 2):
                a = past_value(abil_past, (pid, slot), gen, abil_now.get((pid, slot)))
                if a is not None:
                    abilities[slot] = int(a)
        entries.append({
            "pokemon_id": int(pid), "species_id": int(default.index[default.id == pid][0]),
            "version_group_id": int(g), "version_group": vg.identifier[g], "generation": gen,
            "types": [type_name[x] for x in t], "stats": stats, "effort": effort,
            "abilities": {"1": abilities.get(1), "2": abilities.get(2), "hidden": abilities.get(3)},
            "learnset": learnset,
        })
    with open(OUT / "entries.jsonl", "w", encoding="utf-8") as f:
        for e in sorted(entries, key=lambda e: (e["species_id"], e["version_group_id"])):
            f.write(json.dumps(e) + "\n")

    # Flavor texts, cleaned and masked per family, paired with their game's entry.
    types_at = {(e["species_id"], e["version_group_id"]): e["types"] for e in entries}
    latest = {e["species_id"]: e["types"] for e in sorted(entries, key=lambda e: e["version_group_id"])}
    parent = {s: (None if pd.isna(p) else int(p)) for s, p in zip(spec.species_id, spec.evolves_from_species_id)}
    genus_of = dict(zip(spec.species_id, spec.genus))
    mask = Masker(dict(zip(spec.species_id, spec.name)), parent,
                  lambda s, g: species_token(types_at.get((s, g), latest[s]), genus_of[s]))
    ft = read("pokemon_species_flavor_text").query("language_id == 9")
    ft = ft.assign(version_group_id=ft.version_id.map(versions.version_group_id))
    ft = ft[ft.version_group_id.isin(vg.index)]
    ft = ft.assign(text=ft.flavor_text.map(clean_text))
    ft = ft.assign(text_masked=[mask(t, s, g) for s, g, t in zip(ft.species_id, ft.version_group_id, ft.text)])
    ft = ft.drop_duplicates(["species_id", "version_group_id", "text"])
    have = {(e["species_id"], e["version_group_id"]) for e in entries}
    ft = ft[[k in have for k in zip(ft.species_id, ft.version_group_id)]]
    ft = ft.merge(spec[["species_id", "split", "fold"]], on="species_id")
    ft = ft.assign(generation=ft.version_group_id.map(vg.generation_id),
                   version=ft.version_id.map(versions.identifier))
    ft[["species_id", "version", "version_group_id", "generation", "split", "fold", "text", "text_masked"]] \
        .to_csv(OUT / "texts.csv", index=False)

    move_names = english(read("move_names"), "move_id")
    ability_names = english(read("ability_names"), "ability_id")
    used_moves = sorted({m for e in entries for _, m in e["learnset"]})
    used_abil = sorted({a for e in entries for a in e["abilities"].values() if a})
    # Signature moves and abilities: held by at most SIGNATURE_FAMILIES evolution families, counting
    # every form and every learn method (so a move most Pokemon get by TM is never "signature")
    chain_of_pokemon = pokemon.set_index("id").species_id.map(spec.set_index("species_id").evolution_chain_id)
    all_moves = read("pokemon_moves")
    move_fams = all_moves.assign(chain=all_moves.pokemon_id.map(chain_of_pokemon)).groupby("move_id").chain.nunique()
    all_abil = read("pokemon_abilities")
    abil_fams = all_abil.assign(chain=all_abil.pokemon_id.map(chain_of_pokemon)).groupby("ability_id").chain.nunique()
    vocab = {"moves": {int(m): move_names[m] for m in used_moves},
             "abilities": {int(a): ability_names[a] for a in used_abil},
             "version_groups": {int(g): vg.identifier[g] for g in vg.index},
             "signature_moves": sorted(int(m) for m in used_moves if move_fams.get(m, 0) <= SIGNATURE_FAMILIES),
             "signature_abilities": sorted(int(a) for a in used_abil if abil_fams.get(a, 0) <= SIGNATURE_FAMILIES)}
    (OUT / "vocab.json").write_text(json.dumps(vocab, indent=1, ensure_ascii=False), encoding="utf-8")

    # Checks and summary.
    own = [re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", t, re.IGNORECASE)
           for n, t in zip(ft.species_id.map(names), ft.text_masked)]
    assert not any(own), f"{sum(map(bool, own))} texts still contain their own name"
    assert not set(ft[ft.split == "test"].species_id) & set(ft[ft.split == "train"].species_id)
    print(f"species {len(spec)} | entries {len(entries)} | texts {len(ft)} "
          f"(test {int((ft.split == 'test').sum())}) | moves {len(used_moves)} | abilities {len(used_abil)}")
    print("texts per generation:", ft.groupby("generation").size().to_dict())


if __name__ == "__main__":
    sys.exit(main())
