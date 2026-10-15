"""Dex text generators: one Flan-T5 model, four tasks, selected by a prefix.

  genus     flavor text + types            -> genus ("Fish" for "Fish Pokemon")
  concepts  flavor text + genus + types    -> the name's source words, from Bulbapedia's
                                              canon name origins ("magic, carp")
  name      source words                   -> a name ("Magikarp")
  flavor    entry fields (era, types, genus, body, power) -> a flavor text

Role tokens are written as [self], [evolution], [water fish pokémon] (T5 has no "<").
Trained on folds 1-4 of the training pool and evaluated on fold 0 (held-out families);
Gen 5 is untouched. One split instead of five because fine-tuning T5 on CPU takes hours.

    py scripts/dex_textgen.py                 # train, evaluate, save to runs/dex_textgen/
    py scripts/dex_textgen.py --eval-only

Evaluation (fold 0):
  genus     exact match; vs copying the genus of the most similar training text
  concepts  share of species where a predicted word is among the canon source words
  name      from the canon source words: exact match and character similarity to the real
            name; share of sampled names passing the filters (new, <= 12 characters)
  flavor    do generated texts read as the intended types? Types mAP of a text->type kNN
            (trained on real training texts only) on generated vs real texts
"""

import argparse
import difflib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dex_phase1 import embed  # noqa: E402
from poketype.dex import DEX, STATS, load, natural, weighted_map  # noqa: E402

MODEL = "google/flan-t5-small"
OUT = ROOT / "runs" / "dex_textgen"
EVAL_FOLD = 0
MAX_NAME = 12


def bracketed(text):
    return re.sub(r"<([^<>]+)>", r"[\1]", text)


def describe(row, sp, entry):
    st = entry["stats"]
    bst = sum(st[k] for k in STATS)
    eggs = "/".join(e for e in (sp.egg_group1, sp.egg_group2) if isinstance(e, str))
    return (f"era: gen {entry['generation']} | types: {'/'.join(entry['types'])} | genus: {genus_word(sp.genus)} | "
            f"shape: {sp['shape']} | color: {sp.color} | egg: {eggs} | height: {sp.height_dm / 10:.1f} m | "
            f"weight: {sp.weight_hg / 10:.1f} kg | stage {sp.stage} of {sp.line_length} | total stats {bst}")


def genus_word(g):
    return g[: -len(" Pokémon")] if g.endswith(" Pokémon") else g


def build_examples(rows, species, entries, origins, idx, rng, texts_per_species=2):
    """(task, source, target, species_id) examples from text rows `idx`."""
    ex = []
    sub = rows.iloc[idx]
    for sid, grp in sub.groupby("species_id"):
        sp = species.loc[sid]
        types = "/".join(entries[(sid, grp.version_group_id.iloc[-1])]["types"])
        comps = origins.get(str(sid), {}).get("components", [])[:3]
        picks = grp.sample(min(texts_per_species, len(grp)), random_state=int(rng.integers(1 << 31)))
        for _, r in picks.iterrows():
            t = bracketed(r.text_masked)
            ex.append(("genus", f"genus | types: {types} | text: {t}", genus_word(sp.genus), sid))
            if comps:
                ex.append(("concepts", f"concepts | genus: {genus_word(sp.genus)} | types: {types} | text: {t}",
                           ", ".join(comps), sid))
        if comps:
            ex.append(("name", f"name | {', '.join(comps)}", sp["name"], sid))
        for _, r in grp.iterrows():
            e = entries[(sid, r.version_group_id)]
            ex.append(("flavor", f"flavor | {describe(r, sp, e)}", bracketed(r.text_masked), sid))
    return ex


def batches(ex, size, rng):
    order = rng.permutation(len(ex))
    for i in range(0, len(order), size):
        yield [ex[j] for j in order[i:i + size]]


def train(model, tok, rows, species, entries, origins, tr_idx, epochs=3, lr=3e-4, batch=16, log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for epoch in range(1, epochs + 1):
        ex = build_examples(rows, species, entries, origins, tr_idx, rng)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, b in enumerate(batches(ex, batch, rng)):
            enc = tok([x[1] for x in b], padding=True, truncation=True, max_length=160, return_tensors="pt")
            lab = tok([x[2] for x in b], padding=True, truncation=True, max_length=96, return_tensors="pt").input_ids
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
            if i % 200 == 0:
                log(f"    epoch {epoch} step {i}/{len(ex) // batch}  loss {tot / n:.3f}  ({(time.time() - t0) / 60:.1f} min)")
        log(f"  epoch {epoch}: {len(ex)} examples, mean loss {tot / n:.3f}, {(time.time() - t0) / 60:.1f} min")


@torch.no_grad()
def generate(model, tok, sources, n=1, sample=False, max_new=64, batch=32, seed=0):
    model.eval()
    torch.manual_seed(seed)
    out = []
    for i in range(0, len(sources), batch):
        enc = tok(sources[i:i + batch], padding=True, truncation=True, max_length=160, return_tensors="pt")
        kw = dict(do_sample=True, top_p=0.95, temperature=1.0) if sample else dict(num_beams=4)
        g = model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n, **kw)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return [out[i * n:(i + 1) * n] for i in range(len(sources))] if n > 1 else out


def evaluate(model, tok, rows, species, entries, origins, tr_idx, ev_idx):

    ev_rows = rows.iloc[ev_idx]
    one = ev_rows.groupby("species_id").head(1)  # one text per held-out species
    sids = one.species_id.to_numpy()
    res, examples = {}, {}

    # genus
    types = ["/".join(entries[(s, g)]["types"]) for s, g in zip(sids, one.version_group_id)]
    src = [f"genus | types: {t} | text: {bracketed(x)}" for t, x in zip(types, one.text_masked)]
    pred = generate(model, tok, src)
    truth = [genus_word(species.loc[s].genus) for s in sids]
    res["genus_exact"] = float(np.mean([p.strip().lower() == t.lower() for p, t in zip(pred, truth)]))
    texts = [natural(t) for t in rows.text_masked]
    emb = embed(texts)
    tr_pos = np.asarray(tr_idx)
    nn = tr_pos[np.argmax(emb[one.index.to_numpy()] @ emb[tr_pos].T, 1)]
    knn_genus = [genus_word(species.loc[rows.species_id.iloc[j]].genus) for j in nn]
    res["genus_exact_knn"] = float(np.mean([p.lower() == t.lower() for p, t in zip(knn_genus, truth)]))
    examples["genus"] = [{"species": species.loc[s]["name"], "true": t, "pred": p}
                         for s, t, p in list(zip(sids, truth, pred))[:25]]

    # concepts
    has = [i for i, s in enumerate(sids) if origins.get(str(s), {}).get("components")]
    src = [f"concepts | genus: {truth[i]} | types: {types[i]} | text: {bracketed(one.text_masked.iloc[i])}" for i in has]
    pred_c = generate(model, tok, src)
    hit = [bool({w.strip().lower() for w in p.split(",")} & set(origins[str(sids[i])]["components"]))
           for p, i in zip(pred_c, has)]
    res["concepts_hit"] = float(np.mean(hit))

    # names from canon source words, and from predicted concepts (the real pipeline)
    existing = {n.lower() for n in species["name"]}
    canon = [", ".join(origins[str(sids[i])]["components"][:3]) for i in has]
    real = [species.loc[sids[i]]["name"] for i in has]
    beam = generate(model, tok, [f"name | {c}" for c in canon], max_new=12)
    res["name_exact"] = float(np.mean([b.lower() == r.lower() for b, r in zip(beam, real)]))
    res["name_char_sim"] = float(np.mean([difflib.SequenceMatcher(None, b.lower(), r.lower()).ratio()
                                          for b, r in zip(beam, real)]))
    samples = generate(model, tok, [f"name | {c}" for c in pred_c], n=8, sample=True, max_new=12)
    ok = [[s for s in ss if s.lower() not in existing and 2 < len(s) <= MAX_NAME and " " not in s.strip()]
          for ss in samples]
    res["name_filter_pass"] = float(np.mean([len(o) / 8 for o in ok]))
    examples["names"] = [{"species": r, "canon_words": c, "from_canon": b, "pred_words": p, "new_names": o[:5]}
                         for r, c, b, p, o in list(zip(real, canon, beam, pred_c, ok))[:30]]

    # flavor text: generated for each held-out species' entry; judged by a text->type kNN on real texts
    src = [f"flavor | {describe(r, species.loc[r.species_id], entries[(r.species_id, r.version_group_id)])}"
           for _, r in one.iterrows()]
    gen_text = generate(model, tok, src, sample=True, max_new=80)
    _, targets, _ = load()
    T = targets["types"]
    g_emb = embed([natural(re.sub(r"\[([^\[\]]+)\]", r"<\1>", g)) for g in gen_text])
    sim_gen = g_emb @ emb[tr_pos].T
    sim_real = emb[one.index.to_numpy()] @ emb[tr_pos].T

    def knn_types(sim):
        top = np.argpartition(-sim, 9, axis=1)[:, :10]
        return T[tr_pos][top].mean(1)
    y = T[one.index.to_numpy()]
    w = np.ones(len(y))
    res["flavor_types_mAP_generated"] = weighted_map(y, knn_types(sim_gen), w)
    res["flavor_types_mAP_real"] = weighted_map(y, knn_types(sim_real), w)
    examples["flavor"] = [{"species": species.loc[r.species_id]["name"], "fields": s, "generated": g,
                           "real": bracketed(r.text_masked)}
                          for (_, r), s, g in list(zip(one.iterrows(), src, gen_text))[:25]]
    return res, examples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--smoke", action="store_true", help="tiny run to test the code path")
    args = p.parse_args()
    torch.set_num_threads(12)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    rows, _, _ = load()
    species = pd.read_csv(DEX / "species.csv").set_index("species_id")
    entries = {}
    for line in open(DEX / "entries.jsonl", encoding="utf-8"):
        e = json.loads(line)
        entries[(e["species_id"], e["version_group_id"])] = e
    origins = json.loads((DEX / "name_origins.json").read_text(encoding="utf-8"))
    fold = rows.fold.to_numpy()
    tr_idx = np.where((fold >= 0) & (fold != EVAL_FOLD))[0]
    ev_idx = np.where(fold == EVAL_FOLD)[0]
    if args.smoke:
        tr_idx, ev_idx, args.epochs = tr_idx[:150], ev_idx[:60], 1

    if args.eval_only:
        tok, model = AutoTokenizer.from_pretrained(OUT), AutoModelForSeq2SeqLM.from_pretrained(OUT)
    else:
        tok, model = AutoTokenizer.from_pretrained(MODEL), AutoModelForSeq2SeqLM.from_pretrained(MODEL)
        train(model, tok, rows, species, entries, origins, tr_idx, epochs=args.epochs)
        if args.smoke:
            res, _ = evaluate(model, tok, rows, species, entries, origins, tr_idx, ev_idx)
            print(res)
            return
        OUT.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
    random.seed(0)
    res, examples = evaluate(model, tok, rows, species, entries, origins, tr_idx, ev_idx)
    (ROOT / "results" / "dex_textgen.json").write_text(json.dumps(res, indent=1))
    (ROOT / "results" / "dex_textgen_examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False),
                                                                 encoding="utf-8")
    lines = ["# Dex generator: text generators (Flan-T5-small, one multi-task model)", "",
             "Trained on folds 1-4, evaluated on fold 0 (held-out families); Gen 5 untouched. "
             "Examples: results/dex_textgen_examples.json. Name-origin data: Bulbapedia (CC BY-NC-SA 2.5).", "",
             "| Task | Metric | Value |", "|---|---|---:|",
             f"| Genus | exact match | {res['genus_exact']:.1%} |",
             f"| Genus | baseline: copy nearest text's genus | {res['genus_exact_knn']:.1%} |",
             f"| Name concepts | a predicted word is a canon source word | {res['concepts_hit']:.1%} |",
             f"| Name from canon words | exact real name | {res['name_exact']:.1%} |",
             f"| Name from canon words | character similarity to real name | {res['name_char_sim']:.2f} |",
             f"| Name samples | pass filters (new, ≤ {MAX_NAME} chars, one word) | {res['name_filter_pass']:.1%} |",
             f"| Flavor text | types mAP of a text→type kNN: generated | {res['flavor_types_mAP_generated']:.3f} |",
             f"| Flavor text | same, on the real texts (reference) | {res['flavor_types_mAP_real']:.3f} |"]
    (ROOT / "reports" / "dex_textgen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
