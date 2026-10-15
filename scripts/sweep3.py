"""Round 3: unattended tuning with 5-fold family cross-validation, for each backbone.

    python scripts/sweep3.py

For each backbone (ResNet-18, ConvNeXt-Tiny, ResNet-50), every configuration is scored
on all 5 folds (each fold's model trains on the other 4 folds' families):
  score = mean of the 5 folds' best val mAP; F1/exact = from the pooled out-of-fold
  predictions with one second-type threshold tuned (for per-image F1) on all of them.

  Stage 0  baseline: round 2's final settings on this backbone
  Stage 1  attention pooling (fixed vs learned mix weight)   keep if mAP > current + 0.01
  Stage 2  auxiliary heads (egg group/shape/color, 0.1 / 0.5) keep if mAP > current + 0.01
  Stage 3  set-valued output (no pair term / boost-only pair term)
                                         keep if F1 > current + 0.01 and mAP >= current - 0.01
  Stage 4  Optuna (TPE, no pruning): N_TRIALS configurations of all continuous settings,
           structure fixed by stages 1-3; the stage-3 winner is trial 0
  Stage 5  top 3 Optuna configurations re-run with a second seed; winner = best mean of
           both seeds (10 models)
Then the backbones' winners are compared on that 10-model mean, and the overall winner
is also trained with Black/White-style sprites (both seeds) for the ablation.

Everything is cached on disk, so after any interruption just run the same command again.
Writes runs/r3/decisions.md (as it goes), runs/r3/results.jsonl, reports/round3_summary.md.
The test set is never read.
"""

import argparse
import json
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from poketype.data import load_manifest, targets  # noqa: E402
from poketype.folds import K  # noqa: E402
from poketype.metrics import evaluate, tune_threshold  # noqa: E402
from train3 import BASE, LR_DEFAULTS, RUNS, config_id, fold_rows, train_fold  # noqa: E402

BACKBONES = ["resnet18", "convnext_tiny", "resnet50"]
N_TRIALS = 20
MARGIN = 0.01
DECISIONS = RUNS / "decisions.md"
RESULTS = RUNS / "results.jsonl"
SUMMARY = ROOT / "reports" / "round3_summary.md"


class Progress:
    """Counts training runs and prints a rough time estimate."""

    def __init__(self):
        self.done, self.seconds = 0, 0.0

    def add(self, seconds):
        self.done += 1
        self.seconds += seconds

    def eta(self, remaining):
        if not self.done:
            return "?"
        hours = self.seconds / self.done * remaining / 3600
        return f"~{hours:.1f} h at {self.seconds / self.done / 60:.1f} min/run"


PROGRESS = Progress()
_VAL_TARGETS = {}


def log(text):
    print(text, flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def val_targets(m, fold):
    if fold not in _VAL_TARGETS:
        _VAL_TARGETS[fold] = targets(fold_rows(m, fold, "val"))
    return _VAL_TARGETS[fold]


def evaluate_config(cfg, label, m, workers):
    """Train (or load) all 5 folds of a configuration and score it."""
    cid = config_id(cfg)
    cdir = RUNS / cfg["backbone"] / cid
    if (cdir / "summary.json").exists():
        return json.loads((cdir / "summary.json").read_text())
    print(f"\n>>> {label} [{cid}]", flush=True)
    fold_maps, probs, ys = [], [], []
    for fold in range(K):
        t0 = time.time()
        cached = (cdir / f"fold{fold}" / "result.json").exists()
        res = train_fold(cfg, fold, cdir / f"fold{fold}", workers=workers, save_weights=cfg["seed"] == 1)
        if not cached:
            PROGRESS.add(time.time() - t0)
        fold_maps.append(res["val_mAP"])
        probs.append(np.load(cdir / f"fold{fold}" / "val_probs.npy"))
        ys.append(val_targets(m, fold))
        print(f"  fold {fold}: val mAP {res['val_mAP']:.3f} (epoch {res['best_epoch']})", flush=True)
    p, y = np.concatenate(probs), np.concatenate(ys)
    th = tune_threshold(y, p, objective="f1")
    ev = evaluate(y, p, th)
    summary = {"id": cid, "label": label, "cfg": cfg, "mAP": statistics.mean(fold_maps),
               "mAP_sd": statistics.stdev(fold_maps), "fold_mAPs": fold_maps, "oof_mAP": ev["mAP"],
               "f1": ev["sample_f1"], "exact": ev["exact_match"], "top1": ev["top1_in_types"],
               "dual_predicted": ev["dual_predicted"], "threshold": th}
    (cdir / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    return summary


def fmt(s):
    return (f"mAP {s['mAP']:.3f} (folds +/- {s['mAP_sd']:.3f}), F1 {s['f1']:.3f}, exact {s['exact']:.3f}, "
            f"top type {s['top1']:.3f}, predicts dual {s['dual_predicted']:.0%}")


def structural_stages(backbone, m, workers):
    cfg = dict(BASE, backbone=backbone)
    cur = evaluate_config(cfg, f"{backbone} baseline", m, workers)
    log(f"\n### {backbone}\n\n- Stage 0 baseline (round-2 settings): {fmt(cur)}")
    stages = [
        ("attention pooling", "map", {"attn_fixed": {"pool": "attn"},
                                      "attn_learned": {"pool": "attn", "learn_lambda": True}}),
        ("auxiliary heads", "map", {"aux0.1": {"aux_weight": 0.1}, "aux0.5": {"aux_weight": 0.5}}),
        ("set-valued output", "f1", {"set": {"head": "set"},
                                     "set_boost": {"head": "set", "pair_boost": True}}),
    ]
    for i, (title, rule, variants) in enumerate(stages, 1):
        results = {}
        for name, change in variants.items():
            variant = dict(cfg, **change)
            results[name] = (variant, evaluate_config(variant, f"{backbone} {name}", m, workers))
            log(f"  - Stage {i} {title}, {name}: {fmt(results[name][1])}")
        if rule == "map":
            ok = {n: r for n, r in results.items() if r[1]["mAP"] > cur["mAP"] + MARGIN}
            best = max(ok, key=lambda n: ok[n][1]["mAP"]) if ok else None
        else:
            ok = {n: r for n, r in results.items()
                  if r[1]["f1"] > cur["f1"] + MARGIN and r[1]["mAP"] >= cur["mAP"] - MARGIN}
            best = max(ok, key=lambda n: ok[n][1]["f1"]) if ok else None
        if best:
            cfg, cur = results[best]
            log(f"  - **Kept {best}.**")
        else:
            log("  - No variant cleared the rule; keeping current settings.")
    return cfg, cur


def optuna_stage(backbone, cfg, m, workers, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{(RUNS / backbone / 'optuna.db').as_posix()}"
    study = optuna.create_study(storage=storage, study_name=f"r3_{backbone}", direction="maximize",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8, multivariate=True))
    structure = {k: cfg[k] for k in ("pool", "learn_lambda", "head", "pair_boost")}
    aux_on = cfg["aux_weight"] > 0

    def trial_params(t):
        """Settings of a trial; a queued (not yet started) trial keeps them in system_attrs."""
        return t.params or getattr(t, "system_attrs", {}).get("fixed_params", {})

    def to_cfg(params):
        scale = params["lr_scale"]
        c = dict(cfg, lr=None if scale == 1.0 else scale * LR_DEFAULTS[backbone])  # 1.0 = the default
        for k in ("backbone_lr_mult", "weight_decay", "dropout", "mixup", "epochs", "bg_prob", "aug_strong",
                  "weight_alpha", "aux_weight", "attn_lambda", "pair_l2"):
            if k in params:
                c[k] = params[k]
        return c

    def objective(trial):
        params = {
            "lr_scale": trial.suggest_float("lr_scale", 0.1, 4.0, log=True),
            "backbone_lr_mult": trial.suggest_float("backbone_lr_mult", 0.05, 1.0, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 0.3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.7),
            "mixup": trial.suggest_float("mixup", 0.0, 1.0),
            "epochs": trial.suggest_int("epochs", 15, 60),
            "bg_prob": trial.suggest_float("bg_prob", 0.0, 0.8),
            "aug_strong": trial.suggest_categorical("aug_strong", [False, True]),
            "weight_alpha": trial.suggest_float("weight_alpha", 0.0, 1.0),
        }
        if aux_on:
            params["aux_weight"] = trial.suggest_float("aux_weight", 0.03, 1.0, log=True)
        if structure["pool"] == "attn" and not structure["learn_lambda"]:
            params["attn_lambda"] = trial.suggest_float("attn_lambda", 0.05, 1.0)
        if structure["pair_boost"]:
            params["pair_l2"] = trial.suggest_float("pair_l2", 1e-4, 0.1, log=True)
        s = evaluate_config(to_cfg(params), f"{backbone} optuna trial {trial.number}", m, workers)
        trial.set_user_attr("config_id", s["id"])
        shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}
        log(f"  - Optuna trial {trial.number}: {fmt(s)}  `{json.dumps(shown)}`")
        return s["mAP"]

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not study.trials:  # trial 0 = the structural stages' winner, so Optuna starts from it
        first = {"lr_scale": 1.0, "backbone_lr_mult": cfg["backbone_lr_mult"], "weight_decay": cfg["weight_decay"],
                 "dropout": cfg["dropout"], "mixup": cfg["mixup"], "epochs": cfg["epochs"], "bg_prob": cfg["bg_prob"],
                 "aug_strong": cfg["aug_strong"], "weight_alpha": cfg["weight_alpha"]}
        if aux_on:
            first["aux_weight"] = cfg["aux_weight"]
        if structure["pool"] == "attn" and not structure["learn_lambda"]:
            first["attn_lambda"] = cfg["attn_lambda"]
        if structure["pair_boost"]:
            first["pair_l2"] = cfg["pair_l2"]
        study.enqueue_trial(first)
        log(f"\n  Stage 4 Optuna: {n_trials} trials, structure {structure}")
    # A trial cut off by an interruption stays "running" in the study forever; queue its
    # settings once more (its finished folds are cached), unless they already completed
    # or are already queued.
    finished = {t.user_attrs.get("config_id") for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE}
    queued = {config_id(to_cfg(trial_params(t))) for t in study.trials
              if t.state == optuna.trial.TrialState.WAITING and trial_params(t)}
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING and t.params:
            cid = config_id(to_cfg(t.params))
            if cid not in finished and cid not in queued:
                study.enqueue_trial(t.params)
                queued.add(cid)
    remaining = n_trials - len(done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    done = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                  key=lambda t: t.value, reverse=True)
    return [(to_cfg(t.params), t.number, t.value) for t in done]


def confirm_stage(backbone, ranked, m, workers):
    log("\n  Stage 5: top 3 re-run with a second seed")
    seen, finalists = set(), []
    for c, number, value in ranked:
        if config_id(c) in seen:
            continue
        seen.add(config_id(c))
        s0 = evaluate_config(c, f"{backbone} optuna trial {number}", m, workers)
        s1 = evaluate_config(dict(c, seed=1), f"{backbone} optuna trial {number} seed 1", m, workers)
        score = (s0["mAP"] + s1["mAP"]) / 2
        finalists.append((score, c, s0, s1, number))
        log(f"  - trial {number}: seed 0 mAP {s0['mAP']:.3f}, seed 1 mAP {s1['mAP']:.3f} -> mean {score:.3f}; "
            f"F1 {s0['f1']:.3f} / {s1['f1']:.3f}")
        if len(finalists) == 3:
            break
    best = max(finalists, key=lambda f: f[0])
    log(f"  - **{backbone} winner: trial {best[4]}, 10-model mean mAP {best[0]:.3f}.**")
    for _, _, _, s1, _ in finalists:  # keep weights only for the winner's seed-1 fold models
        if s1["id"] != best[3]["id"]:
            for w in (RUNS / backbone / s1["id"]).glob("fold*/best.pt"):
                w.unlink()
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    p.add_argument("--trials", type=int, default=N_TRIALS)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    m = load_manifest()
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < 10:
        print(f"WARNING: only {free_gb:.1f} GB free on this drive; the sweep needs a few GB. "
              f"Free some space first (Ctrl+C to stop).", flush=True)
    log(f"\n## Round 3 sweep started {datetime.now():%Y-%m-%d %H:%M} (backbones: {', '.join(args.backbones)}; "
        f"{args.trials} Optuna trials each; 5-fold CV; no pruning)")
    planned = len(args.backbones) * (7 + args.trials + 3) * K + 2 * K
    winners = {}
    for i, bb in enumerate(args.backbones):
        cfg, _ = structural_stages(bb, m, args.workers)
        ranked = optuna_stage(bb, cfg, m, args.workers, args.trials)
        winners[bb] = confirm_stage(bb, ranked, m, args.workers)
        left = planned - (i + 1) * (7 + args.trials + 3) * K
        log(f"  ({PROGRESS.done} runs trained this session; remaining estimate {PROGRESS.eta(left)})")

    overall = max(winners, key=lambda b: winners[b][0])
    score, cfg, s0, s1, number = winners[overall]
    log("\n### Backbone comparison (10-model mean val mAP)\n")
    for bb, (sc, _, a, b, n) in sorted(winners.items(), key=lambda kv: -kv[1][0]):
        log(f"- {bb}: {sc:.3f} (trial {n}; F1 {a['f1']:.3f} / {b['f1']:.3f}; "
            f"top type {a['top1']:.3f} / {b['top1']:.3f})")
    log(f"\n**Overall winner: {overall}, trial {number}.** Ablation: same settings with Black/White-style sprites.")
    bw = [evaluate_config(dict(cfg, bw_style=True, seed=s), f"{overall} winner with BW style seed {s}", m, args.workers)
          for s in (0, 1)]
    log(f"- with BW style: mAP {statistics.mean(b['mAP'] for b in bw):.3f} vs {score:.3f} without; "
        f"F1 {statistics.mean(b['f1'] for b in bw):.3f} vs {(s0['f1'] + s1['f1']) / 2:.3f}")
    final = {"backbone": overall, "cfg": cfg, "config_ids": [s0["id"], s1["id"]],
             "bw_config_ids": [b["id"] for b in bw], "mean_mAP": score}
    (RUNS / "final.json").write_text(json.dumps(final, indent=1))
    SUMMARY.write_text(DECISIONS.read_text(encoding="utf-8"), encoding="utf-8")
    log(f"\nRound 3 sweep finished {datetime.now():%Y-%m-%d %H:%M}. Final configuration in runs/r3/final.json.")


if __name__ == "__main__":
    main()
