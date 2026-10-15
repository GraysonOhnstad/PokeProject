"""Phase 1 of the Dex generator: predict an entry's fields from its flavor text.

Compares, with 5-fold family cross-validation on the training pool (Gen 5 untouched):
  prior        the training average (what you get from no text at all)
  tfidf-1nn    copy the entry of the most similar training text (word TF-IDF)
  minilm-1nn   the same with frozen MiniLM sentence embeddings
  minilm-knn   average the 10 most similar training texts' entries, weighted by similarity
  minilm-mlp   heads on frozen MiniLM embeddings + era               (--train mlp)
  minilm-ft    MiniLM fine-tuned end to end with the same heads      (--train ft)
  <model>+knn  average of a trained model's predictions and minilm-knn

Trained models stop early per field on 15% of the training families, never on the fold
being scored. Results merge into results/dex_phase1.json, so methods can run separately.

    py scripts/dex_phase1.py                 # baselines + mlp
    py scripts/dex_phase1.py --train ft      # add fine-tuning (about an hour on CPU)
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.dex import DEX, GENERATIONS, load, natural, score, species_weights  # noqa: E402
from poketype.dex_models import FineTuned, Heads, stopping_split, train  # noqa: E402

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
FOLDS = range(5)
K = 10
PRED_KEYS = ["types", "shape", "color", "egg", "log_height", "log_weight", "stats", "abilities"]
RESULTS = ROOT / "results" / "dex_phase1.json"


def log(msg):
    print(msg, flush=True)


def embed(texts, model_name=ENCODER, batch=64):
    """L2-normalized mean-pooled sentence embeddings, cached by text content."""
    key = hashlib.sha1(("\n".join(texts) + model_name).encode()).hexdigest()[:10]
    cache = DEX / f"emb_{model_name.split('/')[-1]}_{key}.npy"
    if cache.exists():
        return np.load(cache)
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.save(cache, emb)
    return emb


def one_hot(ix, n):
    return np.eye(n, dtype=np.float32)[ix]


def as_scores(targets, vocabs):
    """Targets in the form predictions take, so neighbors' entries can be averaged."""
    t = dict(targets)
    t["shape"] = one_hot(targets["shape"], len(vocabs["shape"]))
    t["color"] = one_hot(targets["color"], len(vocabs["color"]))
    return t


def prior(T, sid, tr, n):
    w = species_weights(sid[tr])
    return {k: np.repeat(np.average(T[k][tr], axis=0, weights=w)[None], n, 0) for k in PRED_KEYS}


def neighbors(sim, T, tr, k):
    """Similarity-weighted average of the k most similar training rows' targets."""
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k] if k > 1 else sim.argmax(1)[:, None]
    w = np.clip(np.take_along_axis(sim, top, 1), 1e-6, None)
    w = w / w.sum(1, keepdims=True)
    return {key: np.einsum("nk,nk...->n...", w, T[key][tr][top]) for key in PRED_KEYS}


def blend(a, b):
    return {k: (a[k] + b[k]) / 2 for k in PRED_KEYS}


def tensors(targets, sid, fit):
    """Training targets as tensors; regression targets standardized on the `fit` rows."""
    reg = np.concatenate([targets["log_height"][:, None], targets["log_weight"][:, None], targets["stats"]], 1)
    mu, sd = reg[fit].mean(0), reg[fit].std(0)
    return {"types": torch.tensor(targets["types"]), "shape": torch.tensor(targets["shape"]),
            "color": torch.tensor(targets["color"]), "egg": torch.tensor(targets["egg"]),
            "reg": torch.tensor((reg - mu) / sd, dtype=torch.float32),
            "abil": torch.tensor(targets["abilities"]), "has": torch.tensor(targets["has_abilities"]).float(),
            "_mu": mu, "_sd": sd}


def row_weights(sid, groups):
    w = torch.zeros(len(sid))
    for g in groups:
        w[g] = torch.tensor(species_weights(sid[g]), dtype=torch.float32)
    return w


def sizes(vocabs):
    return {"types": 18, "shape": len(vocabs["shape"]), "color": len(vocabs["color"]),
            "egg": len(vocabs["egg"]), "reg": 8, "abil": len(vocabs["abilities"])}


def train_mlp(X, targets, vocabs, rows, tr, ev):
    sid = rows.species_id.to_numpy()
    fit, stop = stopping_split(rows, tr)
    model = Heads(X.shape[1], sizes(vocabs))
    Xt = torch.tensor(X)
    preds, best = train(model, lambda b: Xt[b], tensors(targets, sid, fit), row_weights(sid, [fit, stop]),
                        fit, stop, ev, [{"params": model.parameters(), "lr": 1e-3, "weight_decay": 1e-2}],
                        epochs=200, patience=25)
    log(f"    mlp best epochs: {best}")
    return preds


def train_ft(texts, era, targets, vocabs, rows, tr, ev, epochs=8, patience=3):
    from transformers import AutoModel, AutoTokenizer
    sid = rows.species_id.to_numpy()
    fit, stop = stopping_split(rows, tr)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    ids, mask, era_t = enc["input_ids"], enc["attention_mask"], torch.tensor(era)

    def inputs(b):
        n = int(mask[b].sum(1).max())
        return ids[b, :n], mask[b, :n], era_t[b]

    model = FineTuned(AutoModel.from_pretrained(ENCODER), era.shape[1], sizes(vocabs))
    groups = [{"params": model.encoder.parameters(), "lr": 5e-5, "weight_decay": 1e-2},
              {"params": model.heads.parameters(), "lr": 1e-3, "weight_decay": 1e-2}]
    t0 = time.time()
    preds, best = train(model, inputs, tensors(targets, sid, fit), row_weights(sid, [fit, stop]),
                        fit, stop, ev, groups, epochs=epochs, patience=patience, batch=32, log=log)
    log(f"    ft best epochs: {best} ({(time.time() - t0) / 60:.1f} min)")
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="*", default=["mlp"], choices=["mlp", "ft"])
    args = p.parse_args()
    torch.set_num_threads(12)
    rows, targets, vocabs = load()
    pool = rows.fold.to_numpy() >= 0
    sid_all = rows.species_id.to_numpy()
    T = as_scores(targets, vocabs)
    texts = [natural(t) for t in rows.text_masked]
    emb = embed(texts)
    era = one_hot(rows.generation.to_numpy() - 1, len(GENERATIONS))
    X_mlp = np.concatenate([emb, era], 1).astype(np.float32)

    from sklearn.feature_extraction.text import TfidfVectorizer
    results = {}
    fold = rows.fold.to_numpy()
    for f in FOLDS:
        tr, ev = np.where(pool & (fold != f))[0], np.where(fold == f)[0]
        sid = sid_all[ev]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit([texts[i] for i in tr])
        sim_tfidf = (tfidf.transform([texts[i] for i in ev]) @ tfidf.transform([texts[i] for i in tr]).T).toarray()
        sim_mini = emb[ev] @ emb[tr].T
        knn = neighbors(sim_mini, T, tr, K)
        preds = {"prior": prior(T, sid_all, tr, len(ev)), "tfidf-1nn": neighbors(sim_tfidf, T, tr, 1),
                 "minilm-1nn": neighbors(sim_mini, T, tr, 1), "minilm-knn": knn}
        log(f"fold {f}")
        if "mlp" in args.train:
            preds["minilm-mlp"] = train_mlp(X_mlp, targets, vocabs, rows, tr, ev)
            preds["minilm-mlp+knn"] = blend(preds["minilm-mlp"], knn)
        if "ft" in args.train:
            preds["minilm-ft"] = train_ft(texts, era, targets, vocabs, rows, tr, ev)
            preds["minilm-ft+knn"] = blend(preds["minilm-ft"], knn)
        for m, pr in preds.items():
            results.setdefault(m, []).append(score(pr, targets, ev, sid))
        log("  types mAP: " + "  ".join(f"{m} {r[-1]['types_mAP']:.3f}" for m, r in results.items()))

    summary = {m: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                   for k in rs[0]} for m, rs in results.items()}
    old = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"summary": {}, "folds": {}}
    old["summary"].update(summary)
    old["folds"].update(results)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(old, indent=1))
    write_report(old["summary"], rows, pool)


ORDER = ["prior", "tfidf-1nn", "minilm-1nn", "minilm-knn", "minilm-mlp", "minilm-mlp+knn",
         "minilm-ft", "minilm-ft+knn"]
LABELS = [("types_mAP", "Types mAP", "{:.3f}", "↑"), ("types_top1", "Types top-1", "{:.1%}", "↑"),
          ("shape_acc", "Shape acc.", "{:.1%}", "↑"), ("color_acc", "Color acc.", "{:.1%}", "↑"),
          ("egg_mAP", "Egg group mAP", "{:.3f}", "↑"), ("height_x", "Height error ×", "{:.2f}", "↓"),
          ("weight_x", "Weight error ×", "{:.2f}", "↓"), ("stat_mae", "Stat MAE", "{:.1f}", "↓"),
          ("bst_mae", "BST MAE", "{:.0f}", "↓"), ("abil_recall5", "Ability recall@5", "{:.1%}", "↑")]


def write_report(summary, rows, pool):
    methods = [m for m in ORDER if m in summary]
    lines = ["# Dex generator, phase 1: entry fields from flavor text", "",
             f"5-fold family cross-validation over {int(pool.sum())} texts of "
             f"{rows[pool].species_id.nunique()} species (Gen 5 held out, not used). "
             "Mean over folds, each species weighted equally; ± is the spread across folds. "
             "Trained models stop early per field on held-out training families.", "",
             "| Metric | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for key, label, fmt, better in LABELS:
        cells = [fmt.format(summary[m][key][0]) + " ± " + fmt.format(summary[m][key][1]) for m in methods]
        lines.append(f"| {label} {better} | " + " | ".join(cells) + " |")
    lines += ["", "Height/weight error ×: typical factor off (1.50 = off by 50% either way). "
              "Ability recall@5: share of the true abilities among the top 5 guesses (Gen 3+ games)."]
    (ROOT / "reports" / "dex_phase1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
