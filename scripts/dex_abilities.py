"""Abilities model under 5-fold family cross-validation (Gen 5 held out).

Three slot heads over the existing ability inventory: primary, secondary (or none) and hidden
(or none; Gen 5+ games only). Generating samples the slots in order, each different from the
ones before. Variants:
  text        text embedding + era
  oracle-up   + true types, body (shape, color, egg groups, size) and stat line
  pred-up     the oracle-up model fed phase-1 predictions of those fields
  text-knn    baseline: ability frequencies among the 10 most similar training texts

Metrics (Gen 3+ entries, species weighted equally): recall@5 of the ability set (comparable to
phase 1), primary top-1 / top-5, and hidden-ability top-5 (Gen 5+).

    py scripts/dex_abilities.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dex_phase1 import embed, one_hot, train_mlp  # noqa: E402
from poketype.dex import DEX, GENERATIONS, load, natural, species_weights  # noqa: E402
from poketype.dex_models import stopping_split  # noqa: E402

VARIANTS = ["text-knn", "text", "oracle-up", "pred-up"]


class AbilityModel(nn.Module):
    def __init__(self, d_in, n, hidden=512, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.slots = nn.ModuleDict({"primary": nn.Linear(hidden, n), "secondary": nn.Linear(hidden, n + 1),
                                    "hidden": nn.Linear(hidden, n + 1)})  # last class = none

    def forward(self, x):
        h = self.trunk(x)
        return {k: head(h) for k, head in self.slots.items()}


def slot_targets(rows, n_abil, index):
    entries = {}
    for line in open(DEX / "entries.jsonl", encoding="utf-8"):
        e = json.loads(line)
        entries[(e["species_id"], e["version_group_id"])] = e
    prim, sec, hid = [], [], []
    for s, g in zip(rows.species_id, rows.version_group_id):
        a = entries[(s, g)]["abilities"]
        prim.append(index[a["1"]] if a["1"] else -100)
        sec.append(index[a["2"]] if a["2"] else n_abil)
        hid.append(index[a["hidden"]] if a["hidden"] else n_abil)
    gen = rows.generation.to_numpy()
    sec = np.where(gen >= 3, sec, -100)
    hid = np.where(gen >= 5, hid, -100)
    return {"primary": np.array(prim), "secondary": sec, "hidden": hid}


def fit(X, T, rows, tr, epochs=300, patience=30, seed=0):
    torch.manual_seed(seed)
    sid = rows.species_id.to_numpy()
    fit_ix, stop_ix = stopping_split(rows, tr)
    fit_ix = fit_ix[T["primary"][fit_ix] >= 0]
    stop_ix = stop_ix[T["primary"][stop_ix] >= 0]
    Xt, Tt = torch.tensor(X), {k: torch.tensor(v) for k, v in T.items()}
    w = torch.zeros(len(X))
    for g in (fit_ix, stop_ix):
        w[g] = torch.tensor(species_weights(sid[g]), dtype=torch.float32)
    model = AbilityModel(X.shape[1], N_ABIL)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def loss(ix):
        out = model(Xt[ix])
        per = sum(F.cross_entropy(out[k], Tt[k][ix], ignore_index=-100, reduction="none") for k in out)
        return (per * w[ix]).sum() / w[ix].sum()

    best, state, wait = float("inf"), None, 0
    fit_t, stop_t = torch.tensor(fit_ix), torch.tensor(stop_ix)
    for _ in range(epochs):
        model.train()
        for b in fit_t[torch.randperm(len(fit_t))].split(256):
            opt.zero_grad()
            loss(b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss(stop_t).item()
        if v < best - 1e-4:
            best, state, wait = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def slot_probs(model, X):
    out = model(torch.tensor(X))
    return {k: v.softmax(1).numpy() for k, v in out.items()}


def metrics(p, T, set_targets, ev, sid):
    """p: slot probabilities for rows ev (primary (n, A); secondary/hidden (n, A+1))."""
    m = T["primary"][ev] >= 0
    w = species_weights(sid[ev][m])
    any_slot = np.maximum(p["primary"], np.maximum(p["secondary"][:, :-1], p["hidden"][:, :-1]))[m]
    top5 = np.argsort(-any_slot, 1)[:, :5]
    y = set_targets[ev][m]
    recall5 = np.take_along_axis(y, top5, 1).sum(1) / y.sum(1)
    prim = T["primary"][ev][m]
    rank_p = np.argsort(-p["primary"][m], 1)
    hm = T["hidden"][ev] >= 0
    hidden_true = T["hidden"][ev][hm]
    has_h = hidden_true < N_ABIL
    rank_h = np.argsort(-p["hidden"][hm][:, :-1], 1)[:, :5]
    return {
        "recall5": float(np.average(recall5, weights=w)),
        "primary_top1": float(np.average(rank_p[:, 0] == prim, weights=w)),
        "primary_top5": float(np.average((rank_p[:, :5] == prim[:, None]).any(1), weights=w)),
        "hidden_top5": float(np.average((rank_h[has_h] == hidden_true[has_h][:, None]).any(1),
                                        weights=species_weights(sid[ev][hm][has_h]))),
    }


N_ABIL = 0


def main():
    global N_ABIL
    torch.set_num_threads(4)
    rows, targets, vocabs = load()
    N_ABIL = len(vocabs["abilities"])
    pool, fold, sid = rows.fold.to_numpy() >= 0, rows.fold.to_numpy(), rows.species_id.to_numpy()
    T = slot_targets(rows, N_ABIL, vocabs["abilities"].index)
    texts = [natural(t) for t in rows.text_masked]
    emb = embed(texts)
    era = one_hot(rows.generation.to_numpy() - 1, len(GENERATIONS))
    X_text = np.concatenate([emb, era], 1).astype(np.float32)

    def up(types, shape, color, egg, lh, lw, stats, mu, sd):
        bst = stats.sum(1, keepdims=True)
        cont = (np.concatenate([lh[:, None], lw[:, None], bst], 1) - mu) / sd
        return np.concatenate([types, shape, color, egg, stats / bst, cont], 1).astype(np.float32)

    results = {v: [] for v in VARIANTS}
    for f in range(5):
        tr, ev = np.where(pool & (fold != f))[0], np.where(fold == f)[0]
        cont = np.c_[targets["log_height"], targets["log_weight"], targets["stats"].sum(1)]
        mu, sd = cont[tr].mean(0), cont[tr].std(0)
        U = up(targets["types"], one_hot(targets["shape"], len(vocabs["shape"])),
               one_hot(targets["color"], len(vocabs["color"])), targets["egg"],
               targets["log_height"], targets["log_weight"], targets["stats"], mu, sd)
        X_or = np.concatenate([X_text, U], 1)
        ph = train_mlp(X_text, targets, vocabs, rows, tr, ev)
        X_pred = np.concatenate([X_text[ev], up(ph["types"], ph["shape"], ph["color"], ph["egg"], ph["log_height"],
                                                ph["log_weight"], np.clip(ph["stats"], 1, None), mu, sd)], 1)

        sim = emb[ev] @ emb[tr].T
        top = np.argpartition(-sim, 9, axis=1)[:, :10]
        knn_set = targets["abilities"][tr][top].mean(1)
        knn = {"primary": knn_set, "secondary": np.c_[knn_set, np.zeros(len(ev))],
               "hidden": np.c_[knn_set, np.zeros(len(ev))]}
        results["text-knn"].append(metrics(knn, T, targets["abilities"], ev, sid))
        m_text = fit(X_text, T, rows, tr)
        results["text"].append(metrics(slot_probs(m_text, X_text[ev]), T, targets["abilities"], ev, sid))
        m_or = fit(X_or, T, rows, tr)
        results["oracle-up"].append(metrics(slot_probs(m_or, X_or[ev]), T, targets["abilities"], ev, sid))
        results["pred-up"].append(metrics(slot_probs(m_or, X_pred), T, targets["abilities"], ev, sid))
        print(f"fold {f}: " + "  ".join(f"{v} recall5 {results[v][-1]['recall5']:.3f}" for v in VARIANTS), flush=True)

    summary = {v: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}
               for v, rs in results.items()}
    (ROOT / "results" / "dex_abilities.json").write_text(json.dumps({"summary": summary, "folds": results}, indent=1))
    labels = [("recall5", "Ability-set recall@5 ↑"), ("primary_top1", "Primary ability top-1 ↑"),
              ("primary_top5", "Primary ability top-5 ↑"), ("hidden_top5", "Hidden ability top-5 ↑ (Gen 5+)")]
    lines = ["# Dex generator: abilities model", "",
             "5-fold family cross-validation (Gen 5 held out), Gen 3+ entries, species weighted equally. "
             "Phase-1 reference (shared heads): recall@5 16.3%; with kNN blend 18.7%.", "",
             "| Metric | " + " | ".join(VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
    for k, label in labels:
        lines.append(f"| {label} | " + " | ".join(f"{summary[v][k][0]:.1%} ± {summary[v][k][1]:.1%}" for v in VARIANTS) + " |")
    (ROOT / "reports" / "dex_abilities.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
