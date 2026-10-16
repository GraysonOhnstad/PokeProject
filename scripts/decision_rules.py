"""Decision rules for the final ensemble: how the choice of rule trades exact match against F1.

The model outputs a probability for every type and every type combination; a decision rule turns
that into one or two types. Compared on the same predictions:
  f1_threshold     2nd type if p >= tau, tau tuned for per-image F1 (the reported rule)
  exact_threshold  same, tau tuned for exact match (the protocol the baselines used)
  top_type_only    never a second type
  most_likely_set  the single most probable type combination (171 options, no tuning)
Thresholds are tuned on out-of-fold validation predictions (each image scored by the fold model
that never trained on its family), then applied unchanged to the test set. The baselines' exact
match was measured with exact_threshold, which for the color histogram predicts one type 100% of
the time; this script puts the ensemble under the same rule.

Added after the one-time test (2026-10-16) to explain the exact-match comparison; it changes no
model and is reported as a secondary, post-hoc analysis.

    py scripts/decision_rules.py           # compare the rules
    py scripts/decision_rules.py --apply   # also make the exact-match rule the bundle's default

Writes reports/decision_rules.md and results/decision_rules.json. With --apply, runs/r3/final_model.pt
gets threshold = the exact-match tau (the default display), threshold_f1 = the F1 tau (the app's
behind-the-scenes view) and decision_rules = both rules' test metrics with 95% bootstrap intervals.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate3 import loader  # noqa: E402
from evaluate3 import bootstrap, test_sets  # noqa: E402
from poketype.data import load_manifest, targets  # noqa: E402
from poketype.folds import load_folds  # noqa: E402
from poketype.inference import Predictor  # noqa: E402
from poketype.metrics import evaluate, tune_threshold  # noqa: E402
from train3 import RUNS, fold_rows  # noqa: E402


@torch.no_grad()
def set_probs_ensemble(pred, rows):
    """Probabilities of all type sets: mean of the fold models' logits / T, then softmax."""
    spec = {"size": pred.size, "input_size": pred.input_size, "mean": pred.mean, "std": pred.std}
    out = []
    for x, *_ in loader(rows, spec):
        z = torch.stack([m(x.to(pred.device))["logits"].float() for m in pred.models]).mean(0)
        out.append((z / pred.temperature).softmax(1).cpu())
    return torch.cat(out)


@torch.no_grad()
def set_probs_oof(pred, rows):
    """Each image scored by the fold model whose validation fold holds its family."""
    fold_of = rows["evolution_chain_id"].map(load_folds()).to_numpy()
    spec = {"size": pred.size, "input_size": pred.input_size, "mean": pred.mean, "std": pred.std}
    out = None
    for k, model in enumerate(pred.models):
        idx = np.flatnonzero(fold_of == k)
        if not len(idx):
            continue
        dl = loader(rows.iloc[idx].reset_index(drop=True), spec)
        z = torch.cat([model(x.to(pred.device))["logits"].float() for x, *_ in dl])
        p = (z / pred.temperature).softmax(1).cpu()
        if out is None:
            out = torch.zeros(len(rows), p.shape[1])
        out[idx] = p
    return out


def rules(set_p, membership, tau_f1, tau_exact):
    """Predictions (n, 18) multi-hot under each rule, plus the per-type marginals."""
    marg = (set_p @ membership).numpy()
    order = np.argsort(-marg, 1)
    top = np.zeros_like(marg)
    top[np.arange(len(marg)), order[:, 0]] = 1

    def with_second(tau):
        pred = top.copy()
        second = marg[np.arange(len(marg)), order[:, 1]] >= tau
        pred[np.arange(len(marg))[second], order[second, 1]] = 1
        return pred
    best_set = membership[set_p.argmax(1)].numpy()
    return marg, {"f1_threshold": with_second(tau_f1), "exact_threshold": with_second(tau_exact),
                  "top_type_only": top, "most_likely_set": best_set}


def score(y, pred):
    exact = float((pred == y).all(1).mean())
    tp = (pred * y).sum(1)
    f1 = float(np.mean(2 * tp / (pred.sum(1) + y.sum(1))))
    return {"exact": exact, "f1": f1, "predicts_dual": float((pred.sum(1) == 2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="make the exact-match rule the bundle's default")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torch.load(RUNS / "final_model.pt", map_location="cpu")
    pred = Predictor(bundle, device)
    membership = pred.models[0].membership.cpu()
    m = load_manifest()

    val = pd.concat([fold_rows(m, k, "val") for k in range(5)], ignore_index=True)
    y_val = targets(val)
    p_val = set_probs_oof(pred, val)
    marg_val = (p_val @ membership).numpy()
    tau_f1 = tune_threshold(y_val, marg_val, objective="f1")
    tau_exact = tune_threshold(y_val, marg_val, objective="exact")

    rows, _ = test_sets(m)["primary"]
    y_test = targets(rows)
    p_test = set_probs_ensemble(pred, rows)

    out = {"tau_f1": tau_f1, "tau_exact": tau_exact, "n_val": len(val), "n_test": len(rows), "rules": {}}
    _, val_rules = rules(p_val, membership, tau_f1, tau_exact)
    marg_test, test_rules = rules(p_test, membership, tau_f1, tau_exact)
    for name in val_rules:
        out["rules"][name] = {"val": score(y_val, val_rules[name]), "test": score(y_test, test_rules[name])}
    out["test_check"] = evaluate(y_test, marg_test, tau_f1)  # must match reports/eval_round3.md
    summary = {}
    for rule, tau in (("exact", tau_exact), ("f1", tau_f1)):
        ev = evaluate(y_test, marg_test, tau)
        summary[rule] = {"threshold": tau, "exact": ev["exact_match"], "f1": ev["sample_f1"],
                         "top1": ev["top1_in_types"], "mAP": ev["mAP"], "predicts_dual": ev["dual_predicted"],
                         "ci": bootstrap(y_test, marg_test, tau)}
    out["test_summary"] = summary
    (ROOT / "results" / "decision_rules.json").write_text(json.dumps(out, indent=1))

    labels = {"f1_threshold": f"2nd type if p ≥ {tau_f1:.2f} (tuned for F1; reported rule)",
              "exact_threshold": f"2nd type if p ≥ {tau_exact:.2f} (tuned for exact match; baselines' protocol)",
              "top_type_only": "Never a second type", "most_likely_set": "Most probable type combination (no tuning)"}
    lines = ["# Decision rules: exact match vs F1", "",
             "Same final ensemble and predictions under four ways of turning probabilities into types. "
             f"Thresholds tuned on {len(val)} out-of-fold validation images, then applied unchanged to the "
             f"{len(rows)} Gen 5 test sprites. Post-hoc analysis added after the one-time test; no model changed.", "",
             "| Rule | Val exact | Val F1 | Test exact | Test F1 | Test: predicts 2 types |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in out["rules"].items():
        lines.append(f"| {labels[name]} | {r['val']['exact']:.1%} | {r['val']['f1']:.3f} | "
                     f"{r['test']['exact']:.1%} | {r['test']['f1']:.3f} | {r['test']['predicts_dual']:.0%} |")
    lines += ["", "For reference, the color histogram under the baselines' protocol: test exact 20.5%, "
              "predicting one type 100% of the time; 49% of the test Pokémon have two types.",
              f"Check: the reported rule reproduces the evaluation's test numbers (exact "
              f"{out['test_check']['exact_match']:.3f}, F1 {out['test_check']['sample_f1']:.3f})."]
    ci = summary["exact"]["ci"]
    lines += ["", f"Exact-match rule on the test set, 95% bootstrap intervals: exact {summary['exact']['exact']:.1%} "
              f"[{ci['exact'][0]:.1%}, {ci['exact'][1]:.1%}], F1 {summary['exact']['f1']:.3f} "
              f"[{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]; top type and mAP do not depend on the rule."]
    (ROOT / "reports" / "decision_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if args.apply:
        bundle["threshold"], bundle["threshold_f1"] = tau_exact, tau_f1
        bundle["decision_rules"] = {"default": "exact", **summary}
        torch.save(bundle, RUNS / "final_model.pt")
        print(f"final_model.pt: default threshold {tau_exact:.2f} (exact match), behind-the-scenes {tau_f1:.2f} (F1)")


if __name__ == "__main__":
    main()
