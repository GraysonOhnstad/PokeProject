"""Train every Dex component on the whole training pool and save them for the generator.

Components (each validated by its own experiment script):
  heads      text -> types, shape, color, egg groups, height, weight (per-field best weights),
             plus the training embeddings/targets for the 10-NN blend   (dex_phase1.py)
  stats      BST x profile model on text + upstream fields, with stage/rarity heads (dex_stats.py)
  abilities  slot heads on text + upstream + stats                       (dex_abilities.py)
  learnset   transformer generator                                       (dex_learnset.py)
  textgen    Flan-T5 for genus, name and flavor text: runs/dex_textgen/ (dex_textgen.py)

    py scripts/dex_final.py              # Gen 5 held out (the evaluated configuration)
    py scripts/dex_final.py --with-gen5  # final refit on every species, for the app

Writes runs/dex/<config>/dex.pt.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dex_abilities  # noqa: E402
import dex_learnset  # noqa: E402
from dex_phase1 import ENCODER, embed, one_hot, sizes, tensors, row_weights  # noqa: E402
from dex_stats import fit as fit_stats, upstream  # noqa: E402
from poketype.dex import GENERATIONS, load, natural  # noqa: E402
from poketype.dex_models import Heads, stopping_split, train  # noqa: E402


def log(msg):
    print(msg, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-gen5", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2-epoch run into runs/dex/smoke, to test the code path")
    args = p.parse_args()
    torch.set_num_threads(8)
    out_dir = ROOT / "runs" / "dex" / ("smoke" if args.smoke else "all" if args.with_gen5 else "gen5_heldout")
    ep = (lambda n: 2) if args.smoke else (lambda n: n)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, targets, vocabs = load()
    fold = rows.fold.to_numpy()
    pool = np.where((fold >= 0) | args.with_gen5)[0]
    sid = rows.species_id.to_numpy()
    texts = [natural(t) for t in rows.text_masked]
    emb = embed(texts)
    era = one_hot(rows.generation.to_numpy() - 1, len(GENERATIONS))
    X_text = np.concatenate([emb, era], 1).astype(np.float32)
    bundle = {"config": out_dir.name, "encoder": ENCODER,
              "vocabs": {k: v.items for k, v in vocabs.items()}}

    # 1. text heads, per-field best weights
    t0 = time.time()
    fit_ix, stop_ix = stopping_split(rows, pool)
    model = Heads(X_text.shape[1], sizes(vocabs))
    T = tensors(targets, sid, fit_ix)
    Xt = torch.tensor(X_text)
    _, best, states = train(model, lambda b: Xt[b], T, row_weights(sid, [fit_ix, stop_ix]), fit_ix, stop_ix,
                            np.array([], dtype=int), [{"params": model.parameters(), "lr": 1e-3, "weight_decay": 1e-2}],
                            epochs=ep(200), patience=25, keep_states=True)
    # residual spread of log height/weight on the stopping families, for sampling sizes
    model.load_state_dict(states["height"])
    with torch.no_grad():
        reg_h = model(Xt[stop_ix])["reg"][:, :2].numpy() * T["_sd"][:2] + T["_mu"][:2]
    model.load_state_dict(states["weight"])
    with torch.no_grad():
        reg_w = model(Xt[stop_ix])["reg"][:, :2].numpy() * T["_sd"][:2] + T["_mu"][:2]
    resid = np.c_[reg_h[:, 0] - targets["log_height"][stop_ix], reg_w[:, 1] - targets["log_weight"][stop_ix]]
    bundle["heads"] = {"d_in": X_text.shape[1], "sizes": sizes(vocabs), "states": states, "best_epoch": best,
                       "reg_mu": T["_mu"], "reg_sd": T["_sd"], "size_resid_cov": np.cov(resid.T)}
    bundle["knn"] = {"emb": emb[pool], "types": targets["types"][pool], "egg": targets["egg"][pool],
                     "species_id": sid[pool]}
    log(f"heads: best epochs {best} ({(time.time() - t0) / 60:.1f} min)")

    # 2. stat model on text + true upstream fields (+ stage/rarity heads)
    t0 = time.time()
    hw = np.stack([targets["log_height"], targets["log_weight"]], 1)
    mu_hw, sd_hw = hw[pool].mean(0), hw[pool].std(0)
    U = upstream(targets["types"], one_hot(targets["shape"], len(vocabs["shape"])),
                 one_hot(targets["color"], len(vocabs["color"])), targets["egg"],
                 targets["log_height"], targets["log_weight"], mu_hw, sd_hw)
    X_up = np.concatenate([X_text, U], 1)
    stats_model = fit_stats(X_up, targets, rows, pool, aux=True, epochs=ep(300))
    bundle["stats"] = {"d_in": X_up.shape[1], "state": stats_model.state_dict(), "mu_hw": mu_hw, "sd_hw": sd_hw}
    log(f"stats: done ({(time.time() - t0) / 60:.1f} min)")

    # 3. abilities on text + upstream + stat line
    t0 = time.time()
    dex_abilities.N_ABIL = len(vocabs["abilities"])
    A = dex_abilities.slot_targets(rows, dex_abilities.N_ABIL, vocabs["abilities"].index)
    cont = np.c_[targets["log_height"], targets["log_weight"], targets["stats"].sum(1)]
    mu_c, sd_c = cont[pool].mean(0), cont[pool].std(0)
    st = targets["stats"]
    U2 = np.concatenate([targets["types"], one_hot(targets["shape"], len(vocabs["shape"])),
                         one_hot(targets["color"], len(vocabs["color"])), targets["egg"],
                         st / st.sum(1, keepdims=True), (cont - mu_c) / sd_c], 1).astype(np.float32)
    X_ab = np.concatenate([X_text, U2], 1)
    ab_model = dex_abilities.fit(X_ab, A, rows, pool, targets["abilities"], epochs=ep(200))
    bundle["abilities"] = {"d_in": X_ab.shape[1], "n": dex_abilities.N_ABIL, "state": ab_model.state_dict(),
                           "mu_c": mu_c, "sd_c": sd_c}
    log(f"abilities: done ({(time.time() - t0) / 60:.1f} min)")

    # 4. learnset generator on every training entry (all games)
    t0 = time.time()
    entries, species = dex_learnset.load_entries()
    text_emb = {s: emb[ix].mean(0) for s, ix in rows.groupby("species_id").indices.items()}
    tokens = dex_learnset.Tokens(sorted({m for e in entries for _, m in e["learnset"]}))
    e_sid = np.array([e["species_id"] for e in entries])
    e_fold = species.fold.reindex(e_sid).to_numpy()
    e_pool = np.where((e_fold >= 0) | args.with_gen5)[0]
    chain = species.evolution_chain_id.reindex(e_sid).to_numpy()
    fams = np.unique(chain[e_pool])
    stop_fams = set(np.random.default_rng(0).choice(fams, int(round(0.15 * len(fams))), replace=False))
    is_stop = np.isin(chain[e_pool], list(stop_fams))
    cond = dex_learnset.conditioning(entries, species, text_emb)
    seqs = [tokens.encode(e["learnset"]) for e in entries]
    fit_e, stop_e = e_pool[~is_stop], e_pool[is_stop]
    if args.smoke:
        fit_e, stop_e = fit_e[:300], stop_e[:100]
    prior = dex_learnset.move_prior(entries, tokens.moves, fit_e)
    bias = np.stack([dex_learnset.bias_for(prior, e["types"], e["generation"]) for e in entries])
    ls_model = dex_learnset.train_model(cond, seqs, fit_e, stop_e, e_sid, tokens.size, bias, epochs=ep(60), log=log)
    mv = pd.read_csv(ROOT / "data" / "raw" / "csv" / "moves.csv").set_index("id")
    bundle["learnset"] = {"d_cond": cond.shape[1], "n_tokens": tokens.size, "moves": tokens.moves,
                          "move_gen": mv.generation_id.reindex(tokens.moves).tolist(),
                          "state": ls_model.state_dict(), "prior": prior}
    log(f"learnset: done ({(time.time() - t0) / 60:.1f} min)")

    torch.save(bundle, out_dir / "dex.pt")
    (out_dir / "info.json").write_text(json.dumps({"config": out_dir.name, "rows": int(len(pool)),
                                                   "heads_best_epoch": best}, indent=1))
    log(f"saved {out_dir / 'dex.pt'}")


if __name__ == "__main__":
    main()
