"""Phase 1 of the Dex generator: predict an entry's fields from its flavor text.

Compares, with 5-fold family cross-validation on the training pool (Gen 5 untouched):
  prior      the training average (what you get from no text at all)
  tfidf-1nn  copy the entry of the most similar training text (word TF-IDF)
  minilm-1nn the same with MiniLM sentence embeddings
  minilm-knn average the 10 most similar training texts' entries, weighted by similarity
  minilm-mlp a small network with one head per field on frozen MiniLM embeddings + era

    py scripts/dex_phase1.py

Writes results/dex_phase1.json and reports/dex_phase1.md.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.dex import DEX, GENERATIONS, load, natural, score, species_weights  # noqa: E402

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
FOLDS = range(5)
K = 10


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


def prior(T, tr, n):
    w = species_weights(ROWS.species_id.to_numpy()[tr])
    return {k: np.repeat(np.average(T[k][tr], axis=0, weights=w)[None], n, 0)
            for k in PRED_KEYS}


def neighbors(sim, T, tr, k):
    """Similarity-weighted average of the k most similar training rows' targets."""
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k] if k > 1 else sim.argmax(1)[:, None]
    w = np.clip(np.take_along_axis(sim, top, 1), 1e-6, None)
    w = w / w.sum(1, keepdims=True)
    return {key: np.einsum("nk,nk...->n...", w, T[key][tr][top]) for key in PRED_KEYS}


class Heads(nn.Module):
    def __init__(self, d_in, sizes, hidden=512, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in sizes.items()})

    def forward(self, x):
        h = self.trunk(x)
        return {k: head(h) for k, head in self.heads.items()}


def mlp(X, targets, vocabs, tr, ev, epochs=300, patience=25, seed=0):
    """Trains on rows `tr`, early-stops on `ev`; returns predictions for `ev`."""
    torch.manual_seed(seed)
    reg = np.concatenate([targets["log_height"][:, None], targets["log_weight"][:, None], targets["stats"]], 1)
    mu, sd = reg[tr].mean(0), reg[tr].std(0)
    data = {
        "x": torch.tensor(X), "types": torch.tensor(targets["types"]),
        "shape": torch.tensor(targets["shape"]), "color": torch.tensor(targets["color"]),
        "egg": torch.tensor(targets["egg"]), "reg": torch.tensor((reg - mu) / sd, dtype=torch.float32),
        "abil": torch.tensor(targets["abilities"]), "has": torch.tensor(targets["has_abilities"]).float(),
    }
    sid = ROWS.species_id.to_numpy()
    w_all = torch.zeros(len(X))
    w_all[tr] = torch.tensor(species_weights(sid[tr]), dtype=torch.float32)
    w_all[ev] = torch.tensor(species_weights(sid[ev]), dtype=torch.float32)
    sizes = {"types": 18, "shape": len(vocabs["shape"]), "color": len(vocabs["color"]),
             "egg": len(vocabs["egg"]), "reg": reg.shape[1], "abil": len(vocabs["abilities"])}
    net = Heads(X.shape[1], sizes)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    def loss_fn(ix):
        out, d, w = net(data["x"][ix]), {k: v[ix] for k, v in data.items()}, w_all[ix]
        per = (F.binary_cross_entropy_with_logits(out["types"], d["types"], reduction="none").mean(1)
               + F.cross_entropy(out["shape"], d["shape"], reduction="none")
               + F.cross_entropy(out["color"], d["color"], reduction="none")
               + F.binary_cross_entropy_with_logits(out["egg"], d["egg"], reduction="none").mean(1)
               + F.mse_loss(out["reg"], d["reg"], reduction="none").mean(1)
               + d["has"] * F.binary_cross_entropy_with_logits(out["abil"], d["abil"], reduction="none").sum(1)
               / 3)
        return (per * w).sum() / w.sum()

    tr_t, ev_t = torch.tensor(tr), torch.tensor(ev)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for b in tr_t[torch.randperm(len(tr_t))].split(256):
            opt.zero_grad()
            loss_fn(b).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(ev_t).item()
        if v < best - 1e-4:
            best, best_state, wait = v, {k: p.clone() for k, p in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(data["x"][ev_t])
    reg_pred = out["reg"].numpy() * sd + mu
    return {"types": torch.sigmoid(out["types"]).numpy(), "shape": out["shape"].softmax(1).numpy(),
            "color": out["color"].softmax(1).numpy(), "egg": torch.sigmoid(out["egg"]).numpy(),
            "log_height": reg_pred[:, 0], "log_weight": reg_pred[:, 1], "stats": reg_pred[:, 2:],
            "abilities": torch.sigmoid(out["abil"]).numpy()}


PRED_KEYS = ["types", "shape", "color", "egg", "log_height", "log_weight", "stats", "abilities"]
ROWS = None


def main():
    global ROWS
    torch.set_num_threads(8)
    rows, targets, vocabs = load()
    pool = rows.fold.to_numpy() >= 0
    ROWS = rows
    T = as_scores(targets, vocabs)
    texts = [natural(t) for t in rows.text_masked]

    t0 = time.time()
    emb = embed(texts)
    print(f"embeddings {emb.shape} ({time.time() - t0:.0f}s)", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    gen_1hot = one_hot(rows.generation.to_numpy() - 1, len(GENERATIONS))
    X_mlp = np.concatenate([emb, gen_1hot], 1).astype(np.float32)

    methods = ["prior", "tfidf-1nn", "minilm-1nn", "minilm-knn", "minilm-mlp"]
    results = {m: [] for m in methods}
    fold = rows.fold.to_numpy()
    for f in FOLDS:
        tr, ev = np.where(pool & (fold != f))[0], np.where(fold == f)[0]
        sid = rows.species_id.to_numpy()[ev]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit([texts[i] for i in tr])
        A, B = tfidf.transform([texts[i] for i in ev]), tfidf.transform([texts[i] for i in tr])
        sim_tfidf = (A @ B.T).toarray()
        sim_mini = emb[ev] @ emb[tr].T
        preds = {
            "prior": prior(T, tr, len(ev)),
            "tfidf-1nn": neighbors(sim_tfidf, T, tr, 1),
            "minilm-1nn": neighbors(sim_mini, T, tr, 1),
            "minilm-knn": neighbors(sim_mini, T, tr, K),
            "minilm-mlp": mlp(X_mlp, targets, vocabs, tr, ev),
        }
        for m in methods:
            results[m].append(score(preds[m], targets, ev, sid))
        print(f"fold {f}: " + "  ".join(f"{m} types mAP {results[m][-1]['types_mAP']:.3f}" for m in methods),
              flush=True)

    summary = {m: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                   for k in rs[0]} for m, rs in results.items()}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dex_phase1.json").write_text(json.dumps({"summary": summary, "folds": results}, indent=1))
    write_report(summary, rows, pool)


LABELS = [("types_mAP", "Types mAP", "{:.3f}", "↑"), ("types_top1", "Types top-1", "{:.1%}", "↑"),
          ("shape_acc", "Shape acc.", "{:.1%}", "↑"), ("color_acc", "Color acc.", "{:.1%}", "↑"),
          ("egg_mAP", "Egg group mAP", "{:.3f}", "↑"), ("height_x", "Height error ×", "{:.2f}", "↓"),
          ("weight_x", "Weight error ×", "{:.2f}", "↓"), ("stat_mae", "Stat MAE", "{:.1f}", "↓"),
          ("bst_mae", "BST MAE", "{:.0f}", "↓"), ("abil_recall5", "Ability recall@5", "{:.1%}", "↑")]


def write_report(summary, rows, pool):
    methods = list(summary)
    lines = ["# Dex generator, phase 1: entry fields from flavor text", "",
             f"5-fold family cross-validation over {int(pool.sum())} texts of "
             f"{rows[pool].species_id.nunique()} species (Gen 5 held out, not used). "
             "Mean over folds, each species weighted equally; ± is the spread across folds.", "",
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
