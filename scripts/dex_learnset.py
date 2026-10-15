"""Learnset generator under 5-fold family cross-validation (Gen 5 held out).

Conditioning (true upstream fields, so this measures the learnset step itself): types, base
stats, era, evolution stage / line length / is-final, rarity, and the species' mean text
embedding. Trained on every training entry (every game), stopped early on held-out training
families.

Metrics, one entry per held-out species (its latest game), each species weighted equally:
  move mAP        per Pokemon: ranking of moves by how often 16 samples contain them, vs the true learnset;
                  compared with a type prior and a text nearest-neighbor baseline
  NLL / token     on all held-out entries
  realism (KS, 1 sample vs real): learnset length, STAB share (damaging moves of the Pokemon's
                  own types), highest level; plus physical-vs-special alignment: rank correlation
                  between physical share of damaging moves and Atk / (Atk + SpA), real vs sampled

    py scripts/dex_learnset.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dex_phase1 import embed  # noqa: E402
from poketype.dex import DEX, STATS, load, natural  # noqa: E402
from poketype.dex_learnset import LearnsetModel, Tokens, cond_vector, generate, move_scores, nll  # noqa: E402

CSV = ROOT / "data" / "raw" / "csv"
SAMPLES = 16


def load_entries():
    entries = [json.loads(line) for line in open(DEX / "entries.jsonl", encoding="utf-8")]
    species = pd.read_csv(DEX / "species.csv").set_index("species_id")
    return entries, species


def conditioning(entries, species, text_emb):
    rows = []
    for e in entries:
        s = species.loc[e["species_id"]]
        rows.append(cond_vector(e["types"], [e["stats"][k] for k in STATS], e["generation"], s.stage,
                                s.line_length, s.is_final, 2 if s.is_mythical else 1 if s.is_legendary else 0,
                                text_emb.get(e["species_id"])))
    return np.stack(rows)


def per_row_ap(y, scores, allowed):
    """Mean over rows of average precision of each row's move ranking (era-available moves only)."""
    return float(np.mean([average_precision_score(y[r, allowed[r]], scores[r, allowed[r]]) for r in range(len(y))]))


def entry_weights(sids):
    s = pd.Series(sids)
    return (1.0 / s.map(s.value_counts())).to_numpy(np.float32)


def train_model(cond, seqs, fit_ix, stop_ix, sids, n_tokens, epochs=60, patience=6, seed=0, log=print):
    torch.manual_seed(seed)
    model = LearnsetModel(n_tokens, cond.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    c = torch.tensor(cond)
    w_fit = torch.tensor(entry_weights(sids[fit_ix]))
    best, state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(fit_ix))
        for b in perm.split(64):
            ix = fit_ix[b.numpy()]
            tot, n = nll(model, c[ix], [seqs[i] for i in ix])
            loss = ((tot / n) * w_fit[b]).sum() / w_fit[b].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = mean_nll(model, c, seqs, stop_ix, sids)
        if v < best - 1e-4:
            best, state, wait = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            wait += 1
        log(f"    epoch {epoch:2d}  stop NLL/token {v:.3f}{'  *' if wait == 0 else ''}")
        if wait >= patience:
            break
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def mean_nll(model, c, seqs, ix, sids):
    model.eval()
    w = entry_weights(sids[ix])
    per = []
    for b in np.array_split(ix, max(1, len(ix) // 512)):
        tot, n = nll(model, c[b], [seqs[i] for i in b])
        per.append((tot / n).numpy())
    return float(np.average(np.concatenate(per), weights=w))


def realism(lines, entries, move_meta):
    """Per learnset: length, STAB share, highest level, physical share of damaging moves."""
    out = []
    for pairs, e in zip(lines, entries):
        dmg = [m for _, m in pairs if move_meta.damage[m] != "status"]
        out.append({
            "length": len(pairs),
            "stab": np.mean([move_meta.type[m] in e["types"] for m in dmg]) if dmg else 0.0,
            "max_level": max((lv for lv, _ in pairs), default=0),
            "physical": np.mean([move_meta.damage[m] == "physical" for m in dmg]) if dmg else 0.5,
        })
    return pd.DataFrame(out)


def main():
    torch.set_num_threads(8)
    entries, species = load_entries()
    rows, targets, _ = load()
    texts = [natural(t) for t in rows.text_masked]
    emb = embed(texts)
    text_emb = {s: emb[ix].mean(0) for s, ix in rows.groupby("species_id").indices.items()}

    mv = pd.read_csv(CSV / "moves.csv").set_index("id")
    type_name = pd.read_csv(CSV / "types.csv").set_index("id").identifier
    move_meta = pd.DataFrame({"type": mv.type_id.map(type_name),
                              "damage": mv.damage_class_id.map({1: "status", 2: "physical", 3: "special"}),
                              "gen": mv.generation_id})
    tokens = Tokens(sorted({m for e in entries for _, m in e["learnset"]}))
    move_gen = move_meta.gen.reindex(tokens.moves).to_numpy()
    n_moves = len(tokens.moves)

    sids = np.array([e["species_id"] for e in entries])
    fold_of = species.fold.to_dict()
    efold = np.array([fold_of[s] for s in sids])
    gens = np.array([e["generation"] for e in entries])
    allowed = move_gen[None, :] <= gens[:, None]
    bad = sum(not allowed[i, tokens.index[m]] for i, e in enumerate(entries) for _, m in e["learnset"])
    print(f"entries {len(entries)} | moves {n_moves} | learned before their generation: {bad}", flush=True)
    for i, e in enumerate(entries):  # never forbid a move the data shows in that game
        for _, m in e["learnset"]:
            allowed[i, tokens.index[m]] = True

    seqs = [tokens.encode(e["learnset"]) for e in entries]
    cond = conditioning(entries, species, text_emb)
    chain = species.evolution_chain_id.reindex(sids).to_numpy()
    truth = np.zeros((len(entries), n_moves), np.float32)
    for i, e in enumerate(entries):
        truth[i, [tokens.index[m] for _, m in e["learnset"]]] = 1

    results = []
    for f in range(5):
        t0 = time.time()
        tr = np.where((efold >= 0) & (efold != f))[0]
        ev = np.where(efold == f)[0]
        fams = np.unique(chain[tr])
        stop_fams = set(np.random.default_rng(0).choice(fams, int(round(0.15 * len(fams))), replace=False))
        is_stop = np.isin(chain[tr], list(stop_fams))
        fit_ix, stop_ix = tr[~is_stop], tr[is_stop]
        print(f"fold {f}: fit {len(fit_ix)} / stop {len(stop_ix)} / eval {len(ev)} entries", flush=True)
        model = train_model(cond, seqs, fit_ix, stop_ix, sids, tokens.size)

        # one entry per held-out species: its latest game
        latest = pd.Series(ev).groupby(sids[ev]).max().to_numpy()
        c = torch.tensor(cond[latest]).repeat_interleave(SAMPLES, 0)
        al = torch.tensor(allowed[latest]).repeat_interleave(SAMPLES, 0)
        gen = torch.Generator().manual_seed(f)
        out = generate(model, c, al, generator=gen)
        samples = [out[i * SAMPLES:(i + 1) * SAMPLES] for i in range(len(latest))]
        scores = move_scores(samples, tokens, n_moves)

        # baselines: type prior (same era, sharing a type) and text 10-NN (their latest learnsets)
        tr_latest = pd.Series(tr).groupby(sids[tr]).max().to_numpy()
        prior = np.zeros_like(scores)
        for r, i in enumerate(latest):
            same = tr[(gens[tr] == gens[i])]
            share = np.array([bool(set(entries[j]["types"]) & set(entries[i]["types"])) for j in same])
            pool = same[share] if share.any() else same
            prior[r] = np.average(truth[pool], axis=0, weights=entry_weights(sids[pool]))
        E_tr = np.stack([text_emb[s] for s in sids[tr_latest]])
        E_ev = np.stack([text_emb[s] for s in sids[latest]])
        nn_ix = np.argsort(-(E_ev @ E_tr.T), 1)[:, :10]
        knn = truth[tr_latest][nn_ix].mean(1)

        y, al_latest = truth[latest], allowed[latest]

        one = [tokens.decode(s[0]) for s in samples]
        real = [[(lv, m) for lv, m in entries[i]["learnset"]] for i in latest]
        R_gen, R_real = realism(one, [entries[i] for i in latest], move_meta), \
            realism(real, [entries[i] for i in latest], move_meta)
        atk = np.array([entries[i]["stats"]["attack"] / (entries[i]["stats"]["attack"]
                        + entries[i]["stats"]["special-attack"]) for i in latest])
        res = {
            "map_generator": per_row_ap(y, scores, al_latest),
            "map_type_prior": per_row_ap(y, prior, al_latest),
            "map_text_knn": per_row_ap(y, knn, al_latest),
            "nll_per_token": mean_nll(model, torch.tensor(cond), seqs, ev, sids),
            "ks_length": float(ks_2samp(R_gen.length, R_real.length)[0]),
            "ks_stab": float(ks_2samp(R_gen.stab, R_real.stab)[0]),
            "ks_max_level": float(ks_2samp(R_gen.max_level, R_real.max_level)[0]),
            "phys_align_real": float(spearmanr(R_real.physical, atk)[0]),
            "phys_align_gen": float(spearmanr(R_gen.physical, atk)[0]),
            "mean_length_real": float(R_real.length.mean()), "mean_length_gen": float(R_gen.length.mean()),
        }
        results.append(res)
        print(f"  fold {f} ({(time.time() - t0) / 60:.1f} min): " + ", ".join(f"{k} {v:.3f}" for k, v in res.items()),
              flush=True)
        if f == 0:
            ex = {species.loc[sids[i], "name"]: {"real": entries[i]["learnset"], "sampled": one[r]}
                  for r, i in enumerate(latest[:3])}
            (ROOT / "results" / "dex_learnset_examples.json").write_text(json.dumps(ex, indent=1))

    summary = {k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results])))
               for k in results[0]}
    (ROOT / "results" / "dex_learnset.json").write_text(json.dumps({"summary": summary, "folds": results}, indent=1))
    write_report(summary)


def write_report(s):
    f = lambda k, fmt="{:.3f}": fmt.format(s[k][0]) + " ± " + fmt.format(s[k][1])  # noqa: E731
    lines = ["# Dex generator: learnset model", "",
             "5-fold family cross-validation (Gen 5 held out); true upstream fields as input. "
             "One entry per held-out species (its latest game). Mean ± spread over folds.", "",
             "| Move ranking (mAP ↑) | |", "|---|---:|",
             f"| Type prior (same era, shared type) | {f('map_type_prior')} |",
             f"| Text 10-nearest-neighbor | {f('map_text_knn')} |",
             f"| **Generator** (16 samples) | {f('map_generator')} |", "",
             "| Realism (1 sample vs real) | |", "|---|---:|",
             f"| KS: learnset length ↓ | {f('ks_length')} |",
             f"| Mean length: real / sampled | {s['mean_length_real'][0]:.1f} / {s['mean_length_gen'][0]:.1f} |",
             f"| KS: STAB share ↓ | {f('ks_stab')} |",
             f"| KS: highest level ↓ | {f('ks_max_level')} |",
             f"| Physical share vs Atk/(Atk+SpA), rank corr.: real / sampled | "
             f"{s['phys_align_real'][0]:.2f} / {s['phys_align_gen'][0]:.2f} |",
             f"| NLL per token, all held-out entries | {f('nll_per_token')} |", "",
             "Examples: results/dex_learnset_examples.json."]
    (ROOT / "reports" / "dex_learnset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
