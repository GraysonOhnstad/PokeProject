"""Stat-line model (BST x profile) under 5-fold family cross-validation.

Variants:
  text            text embedding + era
  text+aux        + auxiliary heads for evolution stage, line length, is-final, rarity
  oracle-up+aux   + the entry's true upstream fields (types, shape, color, egg groups, size):
                  the ceiling when everything earlier in the chain is right
  pred-up+aux     the same model fed upstream fields predicted from the text (phase-1 heads):
                  what the chain actually delivers

Metrics (held-out families, species weighted equally):
  BST rank corr.   Spearman between predicted and true BST, per species ("wimpy vs godlike")
  BST 80% cover    share of true BSTs inside the predicted 10-90% range (ideal: 80%)
  BST / stat MAE   of the expected stat line, comparable to phase 1 (not the goal)
  realism (KS)     distance between sampled and real distributions of BST, of single stats and
                   of lopsidedness (spread of the 6 shares); 0 = identical. The expected
                   (averaged) line is scored too, to show the collapse sampling avoids.

    py scripts/dex_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dex_phase1 import embed, one_hot, train_mlp  # noqa: E402
from poketype.dex import GENERATIONS, load, natural, species_weights  # noqa: E402
from poketype.dex_models import stopping_split  # noqa: E402
from poketype.dex_stats import SCALE, StatModel, bst_quantiles, expected, losses, sample  # noqa: E402

VARIANTS = ["text", "text+aux", "oracle-up+aux", "pred-up+aux"]


def upstream(types, shape, color, egg, log_h, log_w, mu_hw, sd_hw):
    hw = (np.stack([log_h, log_w], 1) - mu_hw) / sd_hw
    return np.concatenate([types, shape, color, egg, hw], 1).astype(np.float32)


def fit(X, targets, rows, tr, aux, epochs=300, patience=30, seed=0):
    torch.manual_seed(seed)
    sid = rows.species_id.to_numpy()
    fit_ix, stop_ix = stopping_split(rows, tr)
    bst = targets["stats"].sum(1)
    d = {"bst": torch.tensor(bst / SCALE, dtype=torch.float32),
         "shares": torch.tensor(targets["stats"] / bst[:, None], dtype=torch.float32),
         "stage": torch.tensor(targets["stage"]), "line_length": torch.tensor(targets["line_length"]),
         "is_final": torch.tensor(targets["is_final"]), "rarity": torch.tensor(targets["rarity"])}
    Xt = torch.tensor(X)
    w = torch.zeros(len(X))
    for g in (fit_ix, stop_ix):
        w[g] = torch.tensor(species_weights(sid[g]), dtype=torch.float32)
    model = StatModel(X.shape[1], aux=aux)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def batch_loss(ix, main_only=False):
        out = model(Xt[ix], d["bst"][ix])
        per = losses(out, {k: v[ix] for k, v in d.items()})
        if main_only:
            per = {k: v for k, v in per.items() if k != "aux"}
        tot = sum(per.values())
        return (tot * w[ix]).sum() / w[ix].sum()

    best, state, wait = float("inf"), None, 0
    fit_t, stop_t = torch.tensor(fit_ix), torch.tensor(stop_ix)
    for _ in range(epochs):
        model.train()
        for b in fit_t[torch.randperm(len(fit_t))].split(256):
            opt.zero_grad()
            batch_loss(b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = batch_loss(stop_t, main_only=True).item()
        if v < best - 1e-4:
            best, state, wait = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return model.eval()


def lopsidedness(stats):
    s = stats / stats.sum(-1, keepdims=True)
    return s.std(-1)


def evaluate(model, X, targets, rows, ev, seed=0):
    torch.manual_seed(seed)
    sid = rows.species_id.to_numpy()[ev]
    w = species_weights(sid)
    true = targets["stats"][ev]
    true_bst = true.sum(1)
    xt = torch.tensor(X[ev])
    bst_mean, stats_mean = expected(model, xt)
    q = bst_quantiles(model, xt, seed=seed)
    lines, _ = sample(model, xt, n=1)
    lines = lines[:, 0]

    sp = pd.DataFrame(np.c_[sid, bst_mean, true_bst], columns=["sid", "pred", "true"]).groupby("sid").mean()
    first = pd.Series(np.arange(len(ev))).groupby(sid).first().to_numpy()  # one row per species
    return {
        "bst_spearman": float(spearmanr(sp.pred, sp.true)[0]),
        "bst_cover80": float(np.average((true_bst >= q[:, 0]) & (true_bst <= q[:, 1]), weights=w)),
        "bst_mae": float(np.average(np.abs(bst_mean - true_bst), weights=w)),
        "stat_mae": float(np.average(np.abs(stats_mean - true).mean(1), weights=w)),
        "ks_bst_sampled": float(ks_2samp(lines[first].sum(1), true_bst[first])[0]),
        "ks_bst_expected": float(ks_2samp(bst_mean[first], true_bst[first])[0]),
        "ks_stat_sampled": float(ks_2samp(lines[first].ravel(), true[first].ravel())[0]),
        "ks_lopsided_sampled": float(ks_2samp(lopsidedness(lines[first]), lopsidedness(true[first]))[0]),
        "ks_lopsided_expected": float(ks_2samp(lopsidedness(stats_mean[first]), lopsidedness(true[first]))[0]),
    }


def main():
    torch.set_num_threads(4)
    rows, targets, vocabs = load()
    pool = rows.fold.to_numpy() >= 0
    fold = rows.fold.to_numpy()
    texts = [natural(t) for t in rows.text_masked]
    emb = embed(texts)
    era = one_hot(rows.generation.to_numpy() - 1, len(GENERATIONS))
    X_text = np.concatenate([emb, era], 1).astype(np.float32)

    results = {v: [] for v in VARIANTS}
    for f in range(5):
        tr, ev = np.where(pool & (fold != f))[0], np.where(fold == f)[0]
        hw = np.stack([targets["log_height"], targets["log_weight"]], 1)
        mu_hw, sd_hw = hw[tr].mean(0), hw[tr].std(0)
        up_true = upstream(targets["types"], one_hot(targets["shape"], len(vocabs["shape"])),
                           one_hot(targets["color"], len(vocabs["color"])), targets["egg"],
                           targets["log_height"], targets["log_weight"], mu_hw, sd_hw)
        X_oracle = np.concatenate([X_text, up_true], 1)
        p = train_mlp(X_text, targets, vocabs, rows, tr, ev)  # phase-1 heads, same training folds
        X_pred = X_oracle.copy()
        X_pred[ev] = np.concatenate([X_text[ev], upstream(p["types"], p["shape"], p["color"], p["egg"],
                                                          p["log_height"], p["log_weight"], mu_hw, sd_hw)], 1)
        models = {"text": (X_text, False), "text+aux": (X_text, True), "oracle-up+aux": (X_oracle, True)}
        trained = {}
        for name, (X, aux) in models.items():
            trained[name] = fit(X, targets, rows, tr, aux)
            results[name].append(evaluate(trained[name], X, targets, rows, ev))
        results["pred-up+aux"].append(evaluate(trained["oracle-up+aux"], X_pred, targets, rows, ev))
        print(f"fold {f}: " + "  ".join(f"{v} rho {results[v][-1]['bst_spearman']:.3f}" for v in VARIANTS),
              flush=True)

    summary = {v: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                   for k in rs[0]} for v, rs in results.items()}
    (ROOT / "results" / "dex_stats.json").write_text(json.dumps({"summary": summary, "folds": results}, indent=1))
    write_report(summary)


LABELS = [("bst_spearman", "BST rank correlation ↑", "{:.3f}"), ("bst_cover80", "BST 80% interval coverage (→ 80%)", "{:.1%}"),
          ("bst_mae", "BST MAE (expected) ↓", "{:.0f}"), ("stat_mae", "Stat MAE (expected) ↓", "{:.1f}"),
          ("ks_bst_sampled", "Realism KS: BST, sampled ↓", "{:.3f}"), ("ks_bst_expected", "Realism KS: BST, averaged ↓", "{:.3f}"),
          ("ks_stat_sampled", "Realism KS: single stats, sampled ↓", "{:.3f}"),
          ("ks_lopsided_sampled", "Realism KS: lopsidedness, sampled ↓", "{:.3f}"),
          ("ks_lopsided_expected", "Realism KS: lopsidedness, averaged ↓", "{:.3f}")]


def write_report(summary):
    vs = list(summary)
    lines = ["# Dex generator: stat-line model (BST × profile)", "",
             "5-fold family cross-validation on the training pool (Gen 5 held out). Mean ± spread over folds. "
             "Phase-1 reference (shared heads, averaged prediction): stat MAE 21.2, BST MAE 85.", "",
             "| Metric | " + " | ".join(vs) + " |", "|---|" + "---:|" * len(vs)]
    for k, label, fmt in LABELS:
        lines.append(f"| {label} | " + " | ".join(fmt.format(summary[v][k][0]) + " ± " + fmt.format(summary[v][k][1])
                                                for v in vs) + " |")
    lines += ["", "KS: Kolmogorov–Smirnov distance between generated and real distributions (one entry per species). "
              "'Averaged' scores the expected stat line, which is what a point-prediction model outputs."]
    (ROOT / "reports" / "dex_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
