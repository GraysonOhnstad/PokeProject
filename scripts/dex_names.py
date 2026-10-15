"""Evaluate name blending: given a held-out Pokemon's canon source words, is its real name produced
and ranked near the top?

For each of the 5 family folds, the name scorer is trained on the other folds' names (plus
English words); held-out species with at least two English-word canon source words are blended
from their first two words. On fold 0, Flan-T5's blends (trained on folds 1-4) are compared and
added to the candidates.

    py scripts/dex_names.py

Writes results/dex_names.json and reports/dex_names.md.
"""

import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.dex import DEX  # noqa: E402
from poketype.dex_names import LearnedRanker, NameScorer, rank, splice  # noqa: E402

ENGLISH = re.compile(r"[a-z]+")


def english_words():
    from transformers import AutoTokenizer
    vocab = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").get_vocab()
    return [w for w in vocab if w.isalpha() and len(w) >= 3]


def sim(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    species = pd.read_csv(DEX / "species.csv").set_index("species_id")
    origins = json.loads((DEX / "name_origins.json").read_text(encoding="utf-8"))
    eng = english_words()
    t5 = None
    rows = []
    for f in range(5):
        train_names = species.name[(species.fold >= 0) & (species.fold != f)]
        scorer = NameScorer(train_names, eng)
        train_ex = []
        for sid, s in species[(species.fold >= 0) & (species.fold != f)].iterrows():
            comps = [c for c in origins[str(sid)]["components"] if ENGLISH.fullmatch(c)][:2]
            if len(comps) == 2:
                train_ex.append((comps[0], comps[1], s["name"]))
        ranker = LearnedRanker(scorer).fit(train_ex)
        ev = species[species.fold == f]
        pairs = []
        for sid, s in ev.iterrows():
            comps = [c for c in origins[str(sid)]["components"] if ENGLISH.fullmatch(c)][:2]
            if len(comps) == 2:
                pairs.append((sid, s["name"], comps))
        t5_out = {}
        if f == 0:
            t5 = t5 or load_t5()
            if t5:
                tok, model = t5
                import torch
                with torch.no_grad():
                    src = [f"name | {a}, {b}" for _, _, (a, b) in pairs]
                    for i in range(0, len(src), 32):
                        enc = tok(src[i:i + 32], padding=True, return_tensors="pt")
                        beam = tok.batch_decode(model.generate(**enc, num_beams=4, max_new_tokens=12),
                                                skip_special_tokens=True)
                        torch.manual_seed(0)
                        samp = tok.batch_decode(model.generate(**enc, do_sample=True, top_p=0.9, max_new_tokens=12,
                                                               num_return_sequences=8), skip_special_tokens=True)
                        for k, (sid, _, _) in enumerate(pairs[i:i + 32]):
                            t5_out[sid] = (beam[k], samp[k * 8:(k + 1) * 8])
        for sid, real, (a, b) in pairs:
            ranked = [n.lower() for n, _ in rank(a, b, scorer)]
            r = ranked.index(real.lower()) if real.lower() in ranked else None
            learned = [n.lower() for n, _ in ranker.rank(a, b)]
            rl = learned.index(real.lower()) if real.lower() in learned else None
            row = {"fold": f, "name": real, "words": f"{a}, {b}", "in_splices": real.lower() in splice(a, b),
                   "rank": r, "top1": ranked[0] if ranked else "", "top1_sim": sim(ranked[0], real) if ranked else 0,
                   "rank_learned": rl, "learned_top1_sim": sim(learned[0], real) if learned else 0,
                   "top5": learned[:5]}
            if sid in t5_out:
                beam, samp = t5_out[sid]
                comb = [n.lower() for n, _ in rank(a, b, scorer, extra=[beam] + samp)]
                row.update({"t5_beam": beam, "t5_exact": beam.lower() == real.lower(), "t5_sim": sim(beam, real),
                            "comb_top1": comb[0], "comb_exact": comb[0] == real.lower(),
                            "comb_sim": sim(comb[0], real)})
            rows.append(row)
        print(f"fold {f}: {len(pairs)} species", flush=True)

    df = pd.DataFrame(rows)
    f0 = df[df.fold == 0]
    res = {
        "n": int(len(df)),
        "real_name_reachable": float(df.in_splices.mean()),
        "top1_exact": float((df["rank"] == 0).mean()),
        "top5": float(df["rank"].lt(5).mean()),
        "top1_char_sim": float(df.top1_sim.mean()),
        "learned_top1_exact": float((df["rank_learned"] == 0).mean()),
        "learned_top5": float(df["rank_learned"].lt(5).mean()),
        "learned_top1_char_sim": float(df.learned_top1_sim.mean()),
        "fold0_learned_exact": float((f0["rank_learned"] == 0).mean()),
        "fold0_learned_sim": float(f0.learned_top1_sim.mean()),
        "fold0_n": int(len(f0)),
        "fold0_splice_exact": float((f0["rank"] == 0).mean()),
        "fold0_splice_sim": float(f0.top1_sim.mean()),
    }
    if "t5_exact" in f0:
        res.update({"fold0_t5_exact": float(f0.t5_exact.mean()), "fold0_t5_sim": float(f0.t5_sim.mean()),
                    "fold0_comb_exact": float(f0.comb_exact.mean()), "fold0_comb_sim": float(f0.comb_sim.mean())})
    (ROOT / "results" / "dex_names.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1,
                                                                ensure_ascii=False), encoding="utf-8")
    examples = df.sample(min(20, len(df)), random_state=0)
    lines = ["# Dex generator: name blending", "",
             f"Held-out families, 5 folds; {res['n']} species whose first two canon source words are English "
             "(Bulbapedia, CC BY-NC-SA 2.5). The scorer never saw the held-out names.", "",
             f"Real name is among the splices (including respellings): {res['real_name_reachable']:.1%}", "",
             "| Ranking | Real name first | Real name in top 5 | Char. similarity of top pick |",
             "|---|---:|---:|---:|",
             f"| Hand-weighted score | {res['top1_exact']:.1%} | {res['top5']:.1%} | {res['top1_char_sim']:.2f} |",
             f"| Learned ranker (trained on the other folds) | {res['learned_top1_exact']:.1%} | "
             f"{res['learned_top5']:.1%} | {res['learned_top1_char_sim']:.2f} |", ""]
    if "fold0_t5_exact" in res:
        lines += [f"Fold 0 only ({res['fold0_n']} species), against Flan-T5 trained on folds 1-4:", "",
                  "| Method | Exact | Char. similarity |", "|---|---:|---:|",
                  f"| Flan-T5 beam | {res['fold0_t5_exact']:.1%} | {res['fold0_t5_sim']:.2f} |",
                  f"| Splicer + hand-weighted score | {res['fold0_splice_exact']:.1%} | "
                  f"{res['fold0_splice_sim']:.2f} |",
                  f"| Splicer + learned ranker | {res['fold0_learned_exact']:.1%} | {res['fold0_learned_sim']:.2f} |",
                  f"| Splicer + T5 candidates, scored | {res['fold0_comb_exact']:.1%} | {res['fold0_comb_sim']:.2f} |", ""]
    lines += ["Examples (real name ← source words: top picks):", ""]
    lines += [f"- {r['name']} ← {r['words']}: {', '.join(r['top5'])}" for _, r in examples.iterrows()]
    (ROOT / "reports" / "dex_names.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def load_t5():
    path = ROOT / "runs" / "dex_textgen"
    if not path.exists():
        return None
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    return AutoTokenizer.from_pretrained(path), AutoModelForSeq2SeqLM.from_pretrained(path).eval()


if __name__ == "__main__":
    main()
