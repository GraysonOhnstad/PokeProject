"""Evaluate naming on a set of made-up descriptions with intended answers.

Each line of the set (evals/dex_naming_dev.jsonl, or --set) has the description, the intended "is"
and "defines" answers (any one counts) and a style note. For each, the full generator runs; the
script checks whether each half was identified (an intended answer appears in that half's
Wikipedia titles, mention or words) and records both names.

It also writes a blind rating sheet: evals/<set>_ratings.csv lists each description with its names
in random order and no mode shown; rate each name 1-5 (5 = fits the description and sounds like a
real Pokemon). The key mapping names to modes is in evals/<set>_key.json.

    py scripts/dex_naming_eval.py [--set evals/dex_naming_dev.jsonl]

Writes reports/<set>.md and results/<set>.json.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.dex_pipeline import DEFAULT_LLM, DexGenerator  # noqa: E402


def matches(answers, half):
    """Does any intended answer appear in the half's titles, mention or words?"""
    if not half:
        return False
    haystack = " ".join([a[0] for a in half["articles"]] + [half["mention"]] + half["words"]).lower()
    return any(ans.lower() in haystack or ans.lower()[:5] in haystack.split() for ans in answers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default=str(ROOT / "evals" / "dex_naming_dev.jsonl"))
    args = p.parse_args()
    path = Path(args.set)
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gen = DexGenerator(ROOT / "runs" / "dex" / "gen5_heldout" / "dex.pt", ROOT / "runs" / "dex_textgen",
                       llm=DEFAULT_LLM)
    rows, rng = [], random.Random(0)
    for it in items:
        t0 = time.time()
        entry, _ = gen.generate(it["text"], era=9, k=8, seed=0)
        insp = gen.inspiration(it["text"]) if gen.inspiration else {"halves": {}}  # cached, so cheap
        halves = insp.get("halves", {})
        names = entry.get("names") or {}
        row = {"id": it["id"], "style": it.get("style", ""), "text": it["text"],
               "intended_is": it["is"], "intended_defines": it["defines"],
               "found_is": [a[0] for a in halves.get("is", {}).get("articles", [])],
               "found_defines": [a[0] for a in halves.get("defines", {}).get("articles", [])],
               "words_is": halves.get("is", {}).get("words", []),
               "words_defines": halves.get("defines", {}).get("words", []),
               "hit_is": matches(it["is"], halves.get("is")), "hit_defines": matches(it["defines"], halves.get("defines")),
               "genus": entry["genus"], "shown": entry["name"], "mode": entry.get("name_mode"),
               "wordplay": (names.get("wordplay") or {}).get("name"),
               "wordplay_from": (names.get("wordplay") or {}).get("from"),
               "pun": (names.get("pun") or {}).get("name"), "pun_phrase": (names.get("pun") or {}).get("sounds_like"),
               "pun_endorsed": (names.get("pun") or {}).get("endorsed"), "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"{it['id']} is {row['hit_is']!s:5} {row['found_is']} | defines {row['hit_defines']!s:5} "
              f"{row['found_defines']} | {row['wordplay']} / {row['pun']} ({row['pun_phrase']}) -> {row['shown']}",
              flush=True)

    stem = path.stem
    (ROOT / "results" / f"{stem}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    n = len(rows)
    hit_is, hit_def = sum(r["hit_is"] for r in rows), sum(r["hit_defines"] for r in rows)
    lines = [f"# Naming on {path.name}", "",
             f"{n} made-up descriptions with intended answers. A half counts as identified when an intended "
             "answer appears in its Wikipedia titles, mention or name words.", "",
             f"- \"Is\" identified: {hit_is}/{n}", f"- \"Defines\" identified: {hit_def}/{n}",
             f"- Pun shown (endorsed by the model): {sum(r['mode'] == 'pun' for r in rows)}/{n}", "",
             "| # | Style | Intended is / defines | Found is / defines | Wordplay | Pun (sounds like) | Shown |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['style']} | {r['intended_is'][0]} / {r['intended_defines'][0]} | "
                     f"{'✓' if r['hit_is'] else '✗'} {', '.join(r['found_is']) or '–'} / "
                     f"{'✓' if r['hit_defines'] else '✗'} {', '.join(r['found_defines']) or '–'} | "
                     f"{r['wordplay'] or '–'} ({r['wordplay_from'] or ''}) | "
                     f"{r['pun'] or '–'} ({r['pun_phrase'] or ''}) | {r['shown'] or '–'} |")
    (ROOT / "reports" / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # blind rating sheet: names shuffled per description, mode hidden
    key, sheet = {}, []
    for r in rows:
        cands = [(m, r[m]) for m in ("wordplay", "pun") if r[m]]
        rng.shuffle(cands)
        for k, (mode, name) in enumerate(cands):
            sheet.append({"id": r["id"], "description": r["text"], "option": "AB"[k], "name": name, "rating_1_to_5": ""})
            key[f"{r['id']}{'AB'[k]}"] = {"name": name, "mode": mode, "shown": name == r["shown"]}
    with open(ROOT / "evals" / f"{stem}_ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (ROOT / "evals" / f"{stem}_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
