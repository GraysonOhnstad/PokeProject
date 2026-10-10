"""Build the image manifest, species table, and split report.

Reads data/raw/ (from fetch_pokeapi.py) and writes:
    data/manifest.csv          one row per image (schema in PROJECT_BRIEF.md)
    data/species.csv           one row per species
    reports/manifest_report.md checks, split sizes, type distributions

    py scripts/build_manifest.py [--seed 0] [--val-frac 0.15] [--workers 8]
"""

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.images import composite, load_rgba  # noqa: E402

RAW = ROOT / "data" / "raw"
CSV = RAW / "csv"
SPRITES = RAW / "sprites"
OUT = ROOT / "data"
REPORT = ROOT / "reports" / "manifest_report.md"

TEST_GENERATION = 5
NUM_TYPES = 18  # type ids 1-18; 19+ are stellar/unknown/shadow
NEAR_DUP_BITS = 4  # phash Hamming distance reported as a near-duplicate
DUAL_TOLERANCE = 0.02  # max gap between val's and the overall dual-typed species share
# Sources drawn in the test set's Black/White style; train with/without them as an ablation.
BW_STYLE_GAMES = {"black-white"}
# Menu icons are too small to help; Showdown and the top-level default sprites are
# unofficial or duplicate black-white.
EXCLUDED_MEDIA = {"icon", "gif", "default_sprite"}

ROMAN = {f"generation-{r}": i for i, r in
         enumerate(["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"], 1)}
MODIFIERS = {"back", "shiny", "female", "gray", "gbc", "transparent"}

# Latest version group each folder's sprites can legitimately cover. A sprite of a
# form introduced after this is a later, unofficial addition (e.g. fan-made BW-style
# sprites for Gen 6+), and is dropped from the manifest.
GAME_VERSION_GROUP = {
    "showdown": "black-2-white-2", "default": "black-2-white-2",  # BW-style sources
    "red-blue": "red-blue", "red-green-japan": "red-green-japan", "yellow": "yellow",
    "gold": "gold-silver", "silver": "gold-silver", "crystal": "crystal",
    "ruby-sapphire": "ruby-sapphire", "emerald": "emerald",
    "firered-leafgreen": "firered-leafgreen", "diamond-pearl": "diamond-pearl",
    "platinum": "platinum", "heartgold-soulsilver": "heartgold-soulsilver",
    "black-white": "black-2-white-2", "x-y": "x-y",
    "omega-ruby-alpha-sapphire": "omega-ruby-alpha-sapphire",
    "ultra-sun-ultra-moon": "ultra-sun-ultra-moon",
    "lets-go-pikachu-lets-go-eevee": "lets-go-pikachu-lets-go-eevee",
    "scarlet-violet": "the-indigo-disk", "champions": "champions",
    "generation-iii-icons": "emerald", "generation-iv-icons": "heartgold-soulsilver",
    "generation-v-icons": "black-2-white-2", "generation-vi-icons": "omega-ruby-alpha-sapphire",
    "generation-vii-icons": "ultra-sun-ultra-moon", "generation-viii-icons": "the-crown-tundra",
    "lets-go-pikachu-lets-go-eevee-icons": "lets-go-pikachu-lets-go-eevee",
    "sword-shield-icons": "the-crown-tundra",
    "brilliant-diamond-shining-pearl-icons": "brilliant-diamond-shining-pearl",
    "legends-arceus-icons": "legends-arceus", "scarlet-violet-icons": "the-indigo-disk",
}

# Sources whose copies of one form are near-identical share a source_group prefix.
# Priority decides which copy of an exact duplicate is kept (lower wins).
SOURCE_FAMILY_PRIORITY = {"pixel": 0, "3d": 1, "icon": 2, "artwork": 3,
                          "dream-world": 4, "showdown": 5, "default": 6}


def parse_path(path):
    """Folder/filename conventions -> manifest metadata (see PROJECT_BRIEF.md)."""
    parts = path.split("/")
    dirs, stem = parts[:-1], parts[-1].split(".")[0]
    row = {"path": path, "pokemon_id": int(stem), "sprite_generation": None, "provenance": []}

    if dirs[:1] == ["versions"]:
        gen, rest = ROMAN[dirs[1]], dirs[2:]
        row["sprite_generation"] = gen
        if "icons" in rest:
            i = rest.index("icons")
            row["game"] = f"{rest[0] if i else dirs[1]}-icons"
            row["medium"], family, mods = "icon", "icon", rest[i + 1:]
        else:
            row["game"], mods = rest[0], rest[1:]
            if gen <= 5:
                row["medium"], family = "pixel", f"pixel-g{gen}"
            else:
                row["medium"], family = "3d_render", "3d"
        if row["game"] == "champions":
            row["provenance"].append("champions")
    elif dirs[:1] == ["other"]:
        src = dirs[1]
        if src == "official-artwork":
            row["medium"], family = "illustration", "artwork"
            if dirs[2:3] == ["versions"]:
                row["game"] = f"official-artwork-{dirs[4]}"
                row["sprite_generation"], mods = ROMAN[dirs[3]], dirs[5:]
            else:
                row["game"], mods = "official-artwork", dirs[2:]
        elif src == "home":
            row.update(medium="3d_render", game="home", sprite_generation=8)
            family, mods = "3d", dirs[2:]
        elif src == "dream-world":
            row.update(medium="vector", game="dream-world", sprite_generation=5)
            family, mods = "dream-world", dirs[2:]
        elif src == "showdown":
            row.update(medium="gif", game="showdown")
            family, mods = "showdown", dirs[2:]
            row["provenance"].append("showdown")
        else:
            raise ValueError(f"unknown source folder: {path}")
    else:
        row.update(medium="default_sprite", game="default", sprite_generation=5)
        family, mods = "default", dirs
        row["provenance"].append("default_sprite")

    unknown = set(mods) - MODIFIERS
    if unknown:
        raise ValueError(f"unknown folder modifier {unknown}: {path}")
    row["view"] = "back" if "back" in mods else "front"
    row["color_variant"] = next((m for m in ("shiny", "gray", "gbc") if m in mods), "normal")
    row["is_female_variant"] = "female" in mods
    row["source_family"] = family
    return row


def analyze(path):
    """Open one image; return size, perceptual hash, pixel hash, blank flag."""
    try:
        im = load_rgba(SPRITES / path)
        alpha = np.asarray(im)[..., 3]
        # Hash the cropped subject: on the full canvas, small sprites hash mostly background.
        bbox = im.getchannel("A").getbbox()
        subject = im.crop(bbox) if bbox else im
        return {
            "path": path, "opens": True, "error": "",
            "width": im.width, "height": im.height,
            "phash": str(imagehash.phash(composite(subject))),
            "pixel_md5": hashlib.md5(im.tobytes() + str(im.size).encode()).hexdigest(),
            "is_blank": bool(alpha.max() == 0),
        }
    except Exception as e:  # report, don't crash: "every image opens" is a check
        return {"path": path, "opens": False, "error": repr(e)}


def load_tables():
    t = {name: pd.read_csv(CSV / f"{name}.csv", keep_default_na=False, na_values=[""])
         for name in ["pokemon", "pokemon_species", "pokemon_forms", "pokemon_types", "types",
                      "version_groups", "pokemon_species_names", "pokemon_egg_groups",
                      "egg_groups", "pokemon_stats", "stats", "pokemon_colors",
                      "pokemon_shapes", "pokemon_habitats"]}
    return t


def pokemon_table(t):
    """One row per pokemon_id (form) with form-specific types and form debut."""
    types = t["types"].set_index("id")["identifier"]
    pt = t["pokemon_types"].assign(type=lambda d: d["type_id"].map(types))
    by_slot = pt.pivot(index="pokemon_id", columns="slot", values="type")
    pk = t["pokemon"].rename(columns={"id": "pokemon_id", "identifier": "pokemon_name"})
    pk["type1"] = pk["pokemon_id"].map(by_slot[1])
    pk["type2"] = pk["pokemon_id"].map(by_slot.get(2, pd.Series(dtype=object)))
    sp = t["pokemon_species"].rename(columns={
        "id": "species_id", "identifier": "species_name", "generation_id": "debut_generation"})
    pk = pk.merge(sp[["species_id", "species_name", "debut_generation", "evolution_chain_id"]],
                  on="species_id")
    pk["is_default_form"] = pk["is_default"].astype(bool)
    vg_order = t["version_groups"].set_index("id")["order"]
    forms = t["pokemon_forms"].sort_values(["pokemon_id", "is_default"], ascending=[True, False])
    intro = forms.drop_duplicates("pokemon_id").set_index("pokemon_id")["introduced_in_version_group_id"]
    pk["form_debut_order"] = pk["pokemon_id"].map(intro).map(vg_order)
    # Keep every Mega/Primal form, each as its own entry with its own typing. Among the
    # other forms, keep one per distinct typing within a species: the default form, then
    # the earliest form with each new typing (regional forms, Rotom, Wormadam, ...).
    # Same-typed variants (G-Max, totems, costumes) are dropped.
    is_mega = forms.drop_duplicates("pokemon_id").set_index("pokemon_id")["is_mega"].astype(bool)
    pk["is_mega"] = pk["pokemon_id"].map(is_mega).fillna(False) | pk["pokemon_name"].str.endswith("-primal")
    pk["_types"] = pk[["type1", "type2"]].apply(lambda r: frozenset(r.dropna()), axis=1)
    rest = pk[~pk["is_mega"]].sort_values(["species_id", "is_default", "pokemon_id"], ascending=[True, False, True])
    pk["keep_form"] = pk["is_mega"] | ~rest.duplicated(["species_id", "_types"]).reindex(pk.index, fill_value=True)
    return pk.drop(columns="_types")


def species_table(t, pk):
    sp = t["pokemon_species"].rename(columns={
        "id": "species_id", "identifier": "species_name", "generation_id": "debut_generation"})
    names = t["pokemon_species_names"]
    names = names[names["local_language_id"] == 9].set_index("pokemon_species_id")  # English
    sp["name"] = sp["species_id"].map(names["name"])
    sp["genus"] = sp["species_id"].map(names["genus"])
    for col, table in [("color", "pokemon_colors"), ("shape", "pokemon_shapes"),
                       ("habitat", "pokemon_habitats")]:
        sp[col] = sp[f"{col}_id"].map(t[table].set_index("id")["identifier"])
    default = pk[pk["is_default_form"]].set_index("species_id")
    sp["type1"] = sp["species_id"].map(default["type1"])
    sp["type2"] = sp["species_id"].map(default["type2"])
    eg = t["pokemon_egg_groups"].assign(
        g=lambda d: d["egg_group_id"].map(t["egg_groups"].set_index("id")["identifier"]))
    eg = eg.groupby("species_id")["g"].apply(list)
    sp["egg_group1"] = sp["species_id"].map(eg.str[0])
    sp["egg_group2"] = sp["species_id"].map(eg.str[1])
    stats = t["pokemon_stats"].assign(
        stat=lambda d: d["stat_id"].map(t["stats"].set_index("id")["identifier"]))
    stats = stats.pivot(index="pokemon_id", columns="stat", values="base_stat")
    for s in ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]:
        sp[s] = sp["species_id"].map(default["pokemon_id"]).map(stats[s])
    sp["n_forms"] = sp["species_id"].map(pk[pk["keep_form"]].groupby("species_id").size())  # type-distinct
    cols = ["species_id", "species_name", "name", "genus", "debut_generation",
            "evolution_chain_id", "evolves_from_species_id", "type1", "type2",
            "egg_group1", "egg_group2", "color", "shape", "habitat",
            "hp", "attack", "defense", "special-attack", "special-defense", "speed",
            "is_baby", "is_legendary", "is_mythical", "n_forms"]
    return sp[cols]


def assign_splits(pk, seed, val_frac, n_candidates=20000):
    """Family-level split. Returns {evolution_chain_id: split} and the val-search score."""
    test_chains = set(pk.loc[pk["debut_generation"] == TEST_GENERATION, "evolution_chain_id"])
    split = {c: "test" for c in test_chains}
    # Gen 5 families' later members: their species is not Gen 5, so they get "exclude"
    # at the species level (below); the chain itself stays "test".
    rest = pk[~pk["evolution_chain_id"].isin(test_chains) & pk["is_default_form"]]
    counts = rest.melt(id_vars="evolution_chain_id", value_vars=["type1", "type2"]).dropna() \
        .groupby(["evolution_chain_id", "value"]).size().unstack(fill_value=0)
    dual = rest.assign(dual=rest["type2"].notna()).groupby("evolution_chain_id")["dual"].agg(["sum", "size"])
    dual = dual.reindex(counts.index).to_numpy()  # (dual-typed species, species) per family
    chains, counts = counts.index.to_numpy(), counts.to_numpy()  # species per (family, type)
    n_val = round(val_frac * len(chains))
    dual_share = dual[:, 0].sum() / dual[:, 1].sum()

    # Seeded random search: of n_candidates random family subsets whose dual-typed share
    # is within DUAL_TOLERANCE of the overall share (val tunes the second-type threshold),
    # keep the one whose worst-covered type has the largest share of its species in val.
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    for _ in range(n_candidates):
        pick = rng.choice(len(chains), n_val, replace=False)
        if abs(dual[pick, 0].sum() / dual[pick, 1].sum() - dual_share) > DUAL_TOLERANCE:
            continue
        score = (counts[pick].sum(0) / counts.sum(0)).min()
        if score > best_score:
            best, best_score = pick, score
    val = set(chains[best])
    for c in chains:
        split[c] = "val" if c in val else "train"
    return split, best_score


def near_dup_pairs(df, a, b, max_bits):
    """Pairs (row in split a, row in split b) with phash distance <= max_bits."""
    ha = df.loc[df["split"] == a, "phash_int"]
    hb = df.loc[df["split"] == b, "phash_int"]
    if ha.empty or hb.empty:
        return []
    av, bv = ha.to_numpy(np.uint64), hb.to_numpy(np.uint64)
    pairs = []
    for start in range(0, len(av), 256):
        chunk = av[start:start + 256, None]
        dist = np.bitwise_count(chunk ^ bv[None, :])
        for i, j in zip(*np.nonzero(dist <= max_bits)):
            pairs.append((ha.index[start + i], hb.index[j], int(dist[i, j])))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    t = load_tables()
    pk = pokemon_table(t)
    species = species_table(t, pk)
    vg_order = t["version_groups"].set_index("identifier")["order"]

    # --- paths -> metadata
    files = sorted(str(f.relative_to(SPRITES)).replace("\\", "/")
                   for f in SPRITES.rglob("*") if f.is_file() and not f.name.endswith(".part"))
    m = pd.DataFrame([parse_path(f) for f in files])
    unknown_ids = m[~m["pokemon_id"].isin(pk["pokemon_id"])]
    m = m[m["pokemon_id"].isin(pk["pokemon_id"])]
    m = m.merge(pk[["pokemon_id", "pokemon_name", "species_id", "species_name", "is_default_form",
                    "debut_generation", "evolution_chain_id", "type1", "type2",
                    "form_debut_order", "keep_form"]], on="pokemon_id", how="left")

    # --- scope: official, high-fidelity, front-view images of type-distinct forms
    game_order = m["game"].map(GAME_VERSION_GROUP).map(vg_order)
    dropped = {}
    for reason, mask in [
        ("back view", lambda d: d["view"] == "back"),
        ("low-fidelity or unofficial source (icons, Showdown, top-level defaults)",
         lambda d: d["medium"].isin(EXCLUDED_MEDIA)),
        ("non-Mega form with the same typing as another kept form of its species", lambda d: ~d["keep_form"]),
        ("unofficial: form introduced after the game",
         lambda d: game_order[d.index].notna() & (d["form_debut_order"] > game_order[d.index])),
    ]:
        hit = mask(m)
        dropped[reason] = m[hit]
        m = m[~hit]
    m["bw_style"] = m["game"].isin(BW_STYLE_GAMES)

    # --- open every image
    print(f"analyzing {len(m)} images...", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        stats = pd.DataFrame(pool.map(analyze, m["path"], chunksize=64))
    m = m.merge(stats, on="path")
    blank = m["is_blank"].fillna(False).astype(bool)
    m.loc[blank, "provenance"] = m.loc[blank, "provenance"].apply(lambda f: f + ["blank_image"])

    # Exact pixel duplicates of the same form: keep the copy from the highest-priority source.
    m["_prio"] = m["source_family"].str.replace(r"-g\d$", "", regex=True).map(SOURCE_FAMILY_PRIORITY)
    m = m.sort_values(["_prio", "path"])
    m["is_duplicate"] = m["pixel_md5"].notna() & m.duplicated(["pokemon_id", "pixel_md5"])
    m["source_group"] = m["source_family"] + ":" + m["pokemon_id"].astype(str)
    m["provenance_flag"] = m["provenance"].str.join(";")

    # --- splits
    chain_split, val_score = assign_splits(pk, args.seed, args.val_frac)
    m["split"] = m["evolution_chain_id"].map(chain_split)
    late_family = (m["split"] == "test") & (m["debut_generation"] != TEST_GENERATION)
    m.loc[late_family, "split"] = "exclude"
    species["split"] = species["evolution_chain_id"].map(chain_split)
    species.loc[(species["split"] == "test") & (species["debut_generation"] != TEST_GENERATION),
                "split"] = "exclude"

    cols = ["path", "pokemon_id", "pokemon_name", "species_id", "species_name", "is_default_form",
            "debut_generation", "evolution_chain_id", "type1", "type2", "medium", "game",
            "sprite_generation", "view", "color_variant", "is_female_variant", "source_group",
            "phash", "pixel_md5", "is_duplicate", "width", "height", "opens", "error",
            "provenance_flag", "bw_style", "split"]
    m = m.sort_values("path")[cols]
    m["sprite_generation"] = m["sprite_generation"].astype("Int64")
    for c in ["width", "height"]:
        m[c] = m[c].astype("Int64")
    m.to_csv(OUT / "manifest.csv", index=False)
    species.to_csv(OUT / "species.csv", index=False)

    write_report(m, species, unknown_ids, dropped, val_score, args)
    print(f"wrote data/manifest.csv ({len(m)} rows), data/species.csv, {REPORT.relative_to(ROOT)}")


def write_report(m, species, unknown_ids, dropped, val_score, args):
    lines = ["# Manifest report\n", f"Seed {args.seed}, val fraction {args.val_frac}.\n"]
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        lines.append(f"- [{'x' if passed else ' '}] {name}{': ' + detail if detail else ''}")

    lines.append("## Checks\n")
    chain_splits = m[m["split"] != "exclude"].groupby("evolution_chain_id")["split"].nunique()
    check("No evolution family in more than one of train/val/test", (chain_splits <= 1).all(),
          f"{(chain_splits > 1).sum()} families span splits")
    check("Every row has type1", m["type1"].notna().all(), f"{m['type1'].isna().sum()} missing")
    check("Every image opens", m["opens"].all(), f"{(~m['opens']).sum()} failed")

    usable = m[m["opens"] & ~m["provenance_flag"].str.contains("blank_image") & (m["split"] != "exclude")].copy()
    usable["phash_int"] = usable["phash"].apply(lambda h: np.uint64(int(h, 16)))
    cross = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        cross += [(a, b, *pr) for pr in near_dup_pairs(usable, a, b, NEAR_DUP_BITS)]
    exact = [c for c in cross if c[4] == 0]
    check("No exact phash duplicate crosses splits", not exact, f"{len(exact)} pairs")
    lines.append(f"- Near-duplicate pairs across splits (phash distance 1-{NEAR_DUP_BITS}): "
                 f"{len(cross) - len(exact)} (listed below; review, not automatically a failure)")
    lines.append(f"- Unknown pokemon_id files dropped: {len(unknown_ids)}"
                 + (f" ({', '.join(unknown_ids['path'].head(10))}{'...' if len(unknown_ids) > 10 else ''})"
                    if len(unknown_ids) else ""))
    lines.append(f"- Val type coverage (worst type's share of its species in val): {val_score:.3f}")
    lines.insert(2, f"**Overall: {'all checks pass' if ok else 'SOME CHECKS FAIL'}**\n")

    lines.append("\n## Scope\n")
    lines.append("Rows dropped, applied in order:\n\n| reason | rows | detail |\n|---|---|---|")
    for reason, d in dropped.items():
        top = d["game"].value_counts().head(6)
        lines.append(f"| {reason} | {len(d)} | " + ", ".join(f"{g} {n}" for g, n in top.items()) + " |")
    alt = m.loc[~m["is_default_form"]].drop_duplicates("pokemon_id").sort_values("pokemon_id")
    lines.append(f"\nAlternate forms kept (all Megas/Primals, plus one per distinct typing within a species): {len(alt)}. "
                 + ", ".join(f"{r.pokemon_name} ({r.type1}{'/' + r.type2 if pd.notna(r.type2) else ''})"
                             for r in alt.itertuples()))

    lines.append("\n## Split sizes\n")
    lines.append("| split | families | species | forms | images | primary filter* |")
    lines.append("|---|---|---|---|---|---|")
    primary = (m["view"] == "front") & (m["color_variant"] == "normal") & ~m["is_duplicate"]
    for s in ["train", "val", "test", "exclude"]:
        r = m[m["split"] == s]
        lines.append(f"| {s} | {r['evolution_chain_id'].nunique()} | {r['species_id'].nunique()} | "
                     f"{r['pokemon_id'].nunique()} | {len(r)} | {(primary & (m['split'] == s)).sum()} |")
    lines.append("\n*front view, normal colors, not a duplicate.\n")
    pt = m[(m["split"] == "test") & (m["game"] == "black-white") & primary
           & (m["provenance_flag"] == "")]
    lines.append(f"Primary test subset (test, black-white, front, normal, unflagged): "
                 f"{len(pt)} images of {pt['pokemon_id'].nunique()} forms / "
                 f"{pt['species_id'].nunique()} species.\n")

    lines.append("## Training variants (primary filter)\n")
    lines.append("BW-style sources (`bw_style`): " + ", ".join(sorted(BW_STYLE_GAMES)) + ".\n")
    lines.append("| variant | train | val |\n|---|---|---|")
    for name, keep_bw in [("with BW style", True), ("without BW style", False)]:
        v = m[primary & (keep_bw | ~m["bw_style"])]
        lines.append(f"| {name} | {(v['split'] == 'train').sum()} | {(v['split'] == 'val').sum()} |")
    lines.append("")

    lines.append("## Type distribution per split (species, by default-form types)\n")
    sp = species[species["split"].isin(["train", "val", "test"])]
    long = sp.melt(id_vars=["species_id", "split"], value_vars=["type1", "type2"]).dropna()
    dist = long.groupby(["value", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    lines.append("| type | train | val | test |\n|---|---|---|---|")
    for ty, r in dist.sort_values("train", ascending=False).iterrows():
        lines.append(f"| {ty} | {r['train']} | {r['val']} | {r['test']} |")
    dual = sp.groupby("split")["type2"].apply(lambda s: s.notna().mean())
    lines.append("\nDual-typed share: " + ", ".join(f"{s} {dual[s]:.0%}" for s in ["train", "val", "test"]))

    lines.append("\n## Images per medium and split (primary filter)\n")
    tab = m[primary].groupby(["medium", "split"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=["train", "val", "test", "exclude"], fill_value=0)
    lines.append("| medium | train | val | test | exclude |\n|---|---|---|---|---|")
    for med, r in tab.iterrows():
        lines.append(f"| {med} | " + " | ".join(str(v) for v in r) + " |")

    lines.append("\n## Images per species (train, primary filter)\n")
    per = m[primary & (m["split"] == "train")].groupby("species_id").size()
    gen = species.set_index("species_id")["debut_generation"]
    by_gen = per.groupby(per.index.map(gen)).agg(["count", "median", "min", "max"])
    lines.append("| debut gen | species | median imgs | min | max |\n|---|---|---|---|---|")
    for g, r in by_gen.iterrows():
        lines.append(f"| {g} | {r['count']:.0f} | {r['median']:.0f} | {r['min']:.0f} | {r['max']:.0f} |")

    lines.append("\n## Provenance flags (all rows)\n")
    flags = m["provenance_flag"].str.split(";").explode()
    flags = flags[flags != ""]
    lines.append("| flag | rows |\n|---|---|")
    for f, n in flags.value_counts().items():
        lines.append(f"| {f} | {n} |")

    if (~m["opens"]).any():
        lines.append("\n## Images that failed to open\n")
        for _, r in m[~m["opens"]].head(50).iterrows():
            lines.append(f"- `{r['path']}`: {r['error']}")

    if cross:
        lines.append(f"\n## Cross-split near-duplicates (first 100 of {len(cross)})\n")
        lines.append("| distance | a | b |\n|---|---|---|")
        for a, b, i, j, d in sorted(cross, key=lambda c: c[4])[:100]:
            lines.append(f"| {d} | {a}: `{m.at[i, 'path']}` ({m.at[i, 'species_name']}) | "
                         f"{b}: `{m.at[j, 'path']}` ({m.at[j, 'species_name']}) |")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
