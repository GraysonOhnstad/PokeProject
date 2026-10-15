"""Evaluate name inspiration on real Pokedex texts from held-out families (fold 0).

For each held-out species (its longest text, names masked), does a canon source word of its
real name (Bulbapedia) appear among the name ideas? Compared:
  offline   the generator's offline pools (literal, assoc, context, canon)
  inspired  basis words + model concepts + fact terms (dex_inspire, Llama 3.1 8B + Wikipedia)
Caveat: an 8B model has read about Pokemon, so on real entries it may recognize the Pokemon
behind a text; Pokemon titles are filtered, but these scores can still be optimistic. Made-up
texts, the real use, have nothing to recall.

    py scripts/dex_inspire_eval.py [--limit N]

Writes results/dex_inspire_eval.json and reports/dex_inspire_eval.md.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.dex import DEX, natural  # noqa: E402
from poketype.dex_pipeline import DEFAULT_LLM, DexGenerator, nameworthy  # noqa: E402
from poketype.dex_wiki import fact_terms  # noqa: E402


def stem(w):
    return w[:5] if len(w) > 5 else w


def hit(words, canon):
    cs = {stem(x) for c in canon for x in c.split()}
    return any(stem(x) in cs for w in words for x in w.lower().split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    gen = DexGenerator(ROOT / "runs" / "dex" / "gen5_heldout" / "dex.pt", ROOT / "runs" / "dex_textgen",
                       llm=DEFAULT_LLM)
    species = pd.read_csv(DEX / "species.csv").set_index("species_id")
    texts = pd.read_csv(DEX / "texts.csv")
    origins = json.loads((DEX / "name_origins.json").read_text(encoding="utf-8"))
    ev = species.index[species.fold == 0]
    rows = []
    t0 = time.time()
    for sid in ev[:args.limit] if args.limit else ev:
        canon = [c for c in origins[str(sid)]["components"][:3] if re.fullmatch(r"[a-z][a-z ]+", c)]
        if not canon:
            continue
        t = texts[texts.species_id == sid]
        text = natural(t.loc[t.text_masked.str.len().idxmax(), "text_masked"])
        emb = gen.embed(text)
        off = gen.name_ideas(text, emb)
        offline = off["literal"] + off["assoc"] + off["context"] + off["canon"]
        insp = gen.inspiration(text)
        terms = [w for w, _ in fact_terms([s for s, _ in insp["facts"]], nameworthy, gen.real_words)]
        inspired = insp["basis_words"] + insp["concepts"]
        rows.append({"name": species.name[sid], "canon": canon, "text": text,
                     "basis": [b[0] for b in insp["basis"]], "basis_words": insp["basis_words"],
                     "concepts": insp["concepts"], "genus_options": insp.get("genus", []),
                     "true_genus": species.genus[sid],
                     "hit_offline": hit(offline, canon), "hit_basis": hit(insp["basis_words"], canon),
                     "hit_inspired": hit(inspired, canon), "hit_inspired_facts": hit(inspired + terms, canon),
                     "hit_all": hit(offline + inspired + terms, canon)})
        print(f"{len(rows):3d} {species.name[sid]:<12} canon {canon} | basis {rows[-1]['basis']} | "
              f"hits off/inspired: {rows[-1]['hit_offline']}/{rows[-1]['hit_inspired']} "
              f"({(time.time() - t0) / len(rows):.0f} s each)", flush=True)
    df = pd.DataFrame(rows)
    res = {k: float(df[k].mean()) for k in ["hit_offline", "hit_basis", "hit_inspired", "hit_inspired_facts",
                                             "hit_all"]}
    res["n"] = len(df)
    res["has_basis"] = float(df.basis.str.len().gt(0).mean())
    (ROOT / "results" / "dex_inspire_eval.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1,
                                                                       ensure_ascii=False), encoding="utf-8")
    lines = ["# Dex generator: name inspiration on held-out Pokédex texts", "",
             f"Fold 0, {res['n']} species with English canon source words (Bulbapedia, CC BY-NC-SA 2.5); "
             "one text each, names masked. A hit: a canon source word of the real name (first 5 letters) "
             "is among the ideas.", "",
             "Caveat: Llama 3.1 8B has read about Pokémon and may recognize a real entry; Pokémon titles are "
             "filtered, but these scores can be optimistic. Made-up texts have nothing to recall.", "",
             "| Ideas | Hit rate |", "|---|---:|",
             f"| Offline pools (literal, associations, context, canon vocabulary) | {res['hit_offline']:.1%} |",
             f"| Basis words only | {res['hit_basis']:.1%} |",
             f"| Basis words + model concepts | {res['hit_inspired']:.1%} |",
             f"| … + fact terms | {res['hit_inspired_facts']:.1%} |",
             f"| Everything together | {res['hit_all']:.1%} |", "",
             f"A verified Wikipedia basis was found for {res['has_basis']:.0%} of texts.", "",
             "Examples (real name ← canon words | basis | concepts | category options vs real):", ""]
    for _, r in df.head(25).iterrows():
        lines.append(f"- **{r['name']}** ← {', '.join(r['canon'])} | {', '.join(r['basis']) or '—'} | "
                     f"{', '.join(r['concepts'][:6])} | {', '.join(r['genus_options'])} vs {r['true_genus']}")
    (ROOT / "reports" / "dex_inspire_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
