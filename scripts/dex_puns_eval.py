"""Evaluate pun mode on canon puns: Pokemon whose Bulbapedia name origin calls the name a pun, play on
words, homophone or phrase reference, and whose first two source words are English.

For each, pun mode blends its two canon source words (scorer trained without that name) against
common English words and phrases, judging fit against its own Pokedex text. Reported: how often
the real name is among the top pun candidates, and the phrase it is matched to. No language model
is used, so this measures the sound-alike search and fit filter alone.

    py scripts/dex_puns_eval.py

Writes reports/dex_puns_eval.md.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dex_phase1 import embed  # noqa: E402
from poketype.dex import DEX, natural  # noqa: E402
from poketype.dex_names import NameScorer  # noqa: E402
from poketype.dex_puns import PunIndex, common_phrases, letters, puns  # noqa: E402
from poketype.dex_words import wordnet  # noqa: E402

PUN_NOTE = re.compile(r"\b(pun|play on|phrase|homophone|sounds like|reference to the)\b", re.I)


def main():
    from wordfreq import top_n_list
    species = pd.read_csv(DEX / "species.csv")
    texts = pd.read_csv(DEX / "texts.csv")
    origins = json.loads((DEX / "name_origins.json").read_text(encoding="utf-8"))
    names = set(letters(n) for n in species.name)
    index = PunIndex(p for p in common_phrases(wordnet().all_lemma_names(), top_n_list("en", 60000))
                     if letters(p) not in names)
    english = [w for w in top_n_list("en", 30000) if w.isalpha() and len(w) >= 3]
    cases = []
    for _, s in species.iterrows():
        o = origins[str(s.species_id)]
        comps = [c for c in o["components"] if re.fullmatch(r"[a-z]+", c)][:2]
        if o.get("english") and PUN_NOTE.search(o["english"]) and len(comps) == 2:
            cases.append((s.species_id, s["name"], comps, o["english"]))
    rows = []
    for sid, name, (a, b), note in cases:
        scorer = NameScorer(species.name[species.name != name], english)
        text = natural(texts[texts.species_id == sid].text_masked.iloc[0])
        tv = embed([text])[0]
        found = puns([(a, b)], scorer, index, lambda ps: embed(list(ps)) @ tv, min_fit=0.1)
        top = [(n, ph) for n, _, ph, _ in found[:5]]
        rank = next((i for i, f in enumerate(found) if f[0].lower() == name.lower()), None)
        rows.append({"name": name, "words": f"{a}, {b}", "rank": rank, "top": top,
                     "matched": found[rank][2] if rank is not None else None, "note": note[:160]})
        print(f"{name:12s} {a}+{b}: rank {rank} {top[:3]}", flush=True)
    df = pd.DataFrame(rows)
    top1, top5 = float((df["rank"] == 0).mean()), float(df["rank"].lt(5).mean())
    found_any = float(df["rank"].notna().mean())
    lines = ["# Dex generator: pun mode on canon puns", "",
             f"{len(df)} canon names that Bulbapedia calls a pun, play on words, homophone or phrase reference, "
             "with two English source words. Pun mode blends the source words and searches common English "
             "words and phrases that sound alike and fit the Pokémon's own Pokédex text; no language model.", "",
             "| Real name among pun candidates | Rate |", "|---|---:|",
             f"| Ranked first | {top1:.1%} |", f"| In the top 5 | {top5:.1%} |", f"| Anywhere | {found_any:.1%} |", "",
             "| Pokémon | Source words | Rank | Matched phrase | Top pun candidates |", "|---|---|---:|---|---|"]
    for _, r in df.iterrows():
        rank = "–" if pd.isna(r["rank"]) else int(r["rank"]) + 1
        lines.append(f"| {r['name']} | {r['words']} | {rank} | {r['matched'] or '–'} | "
                     + ", ".join(f"{n} ~ {ph}" for n, ph in r["top"][:3]) + " |")
    (ROOT / "reports" / "dex_puns_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
