"""Generate a Pokedex entry from flavor text (command line).

    py scripts/dex_generate.py "It drifts through old mansions, lighting candles with its ghostly flame."
    py scripts/dex_generate.py --era 4 --candidates 8 --seed 3 "..."
    py scripts/dex_generate.py --json "..."          # machine-readable

Uses runs/dex/gen5_heldout/dex.pt (scripts/dex_final.py) and runs/dex_textgen (scripts/dex_textgen.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.dex_pipeline import DEFAULT_LLM, DexGenerator  # noqa: E402


def show(e):
    bar = lambda v: "#" * round(v / 10)  # noqa: E731
    lines = [f"{e['name']}  -  the {e['genus']}", f"Type: {' / '.join(t.title() for t in e['types'])}",
             f"Height {e['height_m']} m   Weight {e['weight_kg']} kg   Shape {e['shape']}   Color {e['color']}",
             f"Egg groups: {', '.join(e['egg_groups'])}   Era: Gen {e['era']}   "
             f"Evolution: stage {e['evolution']['stage']} of {e['evolution']['line_length']}   ({e['rarity']})", "",
             e["flavor_text"], ""]
    for k, v in e["stats"].items():
        lines.append(f"  {k:<16}{v:>4}  {bar(v)}")
    lines.append(f"  {'total':<16}{e['bst']:>4}")
    ab = e["abilities"]
    if ab:
        lines += ["", "Abilities: " + ", ".join(v for k, v in ab.items() if k != "hidden")
                  + (f"   Hidden: {ab['hidden']}" if "hidden" in ab else "")]
    lines += ["", "Level-up moves:"]
    lines += [f"  {'Evo' if lv == 0 else lv:>3}  {m}" for lv, m in e["learnset"]]
    insp = e.get("inspiration")
    if insp:
        lines += ["", f"Inspired by: {', '.join(insp['based_on'])}"]
        if insp["fact"]:
            lines.append(f'  "{insp["fact"]}" (Wikipedia: {insp["fact_source"]}, CC BY-SA 4.0)')
    lines += ["", f"(name built from: {e['name_concepts']})"]
    if e.get("name_ideas"):
        lines.append("(name ideas: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in e["name_ideas"].items() if v) + ")")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text")
    p.add_argument("--era", type=int, default=9)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bundle", default=str(ROOT / "runs" / "dex" / "gen5_heldout" / "dex.pt"))
    p.add_argument("--textgen", default=str(ROOT / "runs" / "dex_textgen"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="skip name inspiration (offline idea pools only)")
    args = p.parse_args()
    t0 = time.time()
    gen = DexGenerator(args.bundle, args.textgen, llm=None if args.no_llm else DEFAULT_LLM)
    t1 = time.time()
    entry, _ = gen.generate(args.text, era=args.era, k=args.candidates, seed=args.seed)
    if args.json:
        print(json.dumps(entry, indent=1, ensure_ascii=False))
    else:
        print(show(entry))
        print(f"\n(load {t1 - t0:.1f} s, generate {time.time() - t1:.1f} s)")


if __name__ == "__main__":
    main()
