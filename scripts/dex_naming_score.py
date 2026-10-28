"""Score the blind name ratings for a naming set against the hidden key.

Reads evals/<set>_ratings.csv (filled in by hand, 1-5 per name) and evals/<set>_key.json (written by
dex_naming_eval.py) and compares the two naming modes, the name the generator shows, and the best
of the two (what a perfect picker would get).

    py scripts/dex_naming_score.py [--set evals/dex_naming_dev.jsonl]

Writes reports/<set>_ratings.md.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default=str(ROOT / "evals" / "dex_naming_dev.jsonl"))
    args = p.parse_args()
    stem = Path(args.set).stem
    key = json.loads((ROOT / "evals" / f"{stem}_key.json").read_text(encoding="utf-8"))
    with open(ROOT / "evals" / f"{stem}_ratings.csv", encoding="utf-8", newline="") as f:
        sheet = list(csv.DictReader(f))
    missing = [r["id"] + r["option"] for r in sheet if not r["rating_1_to_5"].strip()]
    if missing:
        raise SystemExit(f"unrated: {', '.join(missing)}")

    per = {}  # id -> {mode: (name, rating)}, plus the shown name
    for r in sheet:
        k = key[r["id"] + r["option"]]
        d = per.setdefault(r["id"], {"shown": None})
        d[k["mode"]] = (k["name"], int(r["rating_1_to_5"]))
        if k["shown"]:
            d["shown"] = (k["name"], int(r["rating_1_to_5"]))
    ids = sorted(per)
    diff = [i for i in ids if "wordplay" in per[i] and "pun" in per[i] and per[i]["wordplay"][0] != per[i]["pun"][0]]
    wp = [per[i]["wordplay"][1] for i in diff]
    pun = [per[i]["pun"][1] for i in diff]
    shown = [per[i]["shown"][1] for i in ids]
    best = [max(v[1] for m, v in per[i].items() if m != "shown") for i in ids]
    picked_best = sum(per[i]["shown"][1] == b for i, b in zip(ids, best))
    n = len(ids)

    def share(xs, t=4):
        return f"{sum(x >= t for x in xs)}/{len(xs)}"

    lines = [f"# Blind name ratings on {stem}.jsonl", "",
             "Each name rated 1-5 by the project author without seeing its mode (5 = fits the description and "
             "sounds like a real Pokemon). When both modes produced the same name it was rated once.", "",
             "| | Mean | Rated 4+ | n |", "|---|---|---|---|",
             f"| Wordplay (portmanteau) | {mean(wp):.2f} | {share(wp)} | {len(diff)} |",
             f"| Pun | {mean(pun):.2f} | {share(pun)} | {len(diff)} |",
             f"| Shown name | {mean(shown):.2f} | {share(shown)} | {n} |",
             f"| Best of the two | {mean(best):.2f} | {share(best)} | {n} |", "",
             f"Over the {len(diff)} descriptions with two different names: wordplay rated higher on "
             f"{sum(a > b for a, b in zip(wp, pun))}, pun on {sum(b > a for a, b in zip(wp, pun))}, "
             f"tied on {sum(a == b for a, b in zip(wp, pun))}. "
             f"The shown name was the better-rated (or tied) one on {picked_best}/{n}.", "",
             "| # | Wordplay | Pun | Shown |", "|---|---|---|---|"]
    for i in ids:
        d = per[i]
        lines.append(f"| {i} | {d['wordplay'][0]} ({d['wordplay'][1]}) | {d['pun'][0]} ({d['pun'][1]}) | "
                     f"{d['shown'][0]} |")
    out = ROOT / "reports" / f"{stem}_ratings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:13]))


if __name__ == "__main__":
    main()
