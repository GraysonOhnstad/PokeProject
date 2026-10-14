"""Train a type classifier with early stopping on validation mAP.

    py scripts/train.py --arch resnet18 --name resnet18_bw
    py scripts/train.py --arch small_cnn --no-bw-style --name small_cnn_nobw

Choices (see PROJECT_BRIEF.md):
  - 18 sigmoid outputs + binary cross-entropy: a Pokemon has one or two types, so each
    type is an independent yes/no; weighted per image so every form counts equally.
  - AdamW: robust default for both from-scratch and fine-tuned CNNs; decoupled weight
    decay regularizes without interfering with the adaptive step sizes.
  - Linear warmup then cosine decay of the learning rate; early stopping when val mAP
    has not improved for --patience epochs; the best-val-mAP checkpoint is kept.
  - Round-2 options against memorization / scale mismatch: --size (preprocessed
    resolution), --input-size (nearest-neighbor upscale before the model),
    --backbone-lr-mult (pretrained layers learn slower than the new head), --mixup
    (blend image pairs and their labels), --dropout.
  - Val is held-out evolution families, so early stopping and the second-type
    threshold are tuned on unseen Pokemon, mirroring the test. The test set is never
    touched here (see scripts/evaluate.py).

Writes runs/<name>/: best.pt (weights + config + threshold), log.csv (per epoch),
summary.json, val_probs.npy.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import (TYPES, PokeTypeDataset, eval_transform, form_weights,  # noqa: E402
                           load_manifest, select, targets, train_transform)
from poketype.metrics import evaluate, tune_threshold  # noqa: E402
from poketype.images import SIZE  # noqa: E402
from poketype.models import build, count_parameters, head_parameter_names, predict_proba  # noqa: E402

DEFAULTS = {  # per-architecture learning rate and weight decay
    "small_cnn": {"lr": 1e-3, "weight_decay": 5e-4},
    "resnet18": {"lr": 3e-4, "weight_decay": 1e-2},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=list(DEFAULTS), default="resnet18")
    p.add_argument("--name", help="run directory name (default: arch + variant)")
    p.add_argument("--no-bw-style", action="store_true", help="drop Black/White sprites from training")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    p.add_argument("--warmup", type=int, default=2, help="linear LR warmup epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--weight-alpha", type=float, default=1.0, help="form-balancing exponent (0 = off)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, help="use only N training images (smoke tests)")
    p.add_argument("--size", type=int, default=SIZE, help="preprocessed resolution to read (96 or 192)")
    p.add_argument("--input-size", type=int, help="nearest-neighbor resize to this before the model")
    p.add_argument("--backbone-lr-mult", type=float, default=1.0,
                   help="learning rate of pretrained layers as a fraction of the head's")
    p.add_argument("--mixup", type=float, default=0.0, help="mixup Beta(a, a) parameter; 0 = off")
    p.add_argument("--dropout", type=float, help="dropout before the output layer (default: architecture's)")
    return p.parse_args()


def make_scheduler(opt, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for k, v in DEFAULTS[args.arch].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    variant = "nobw" if args.no_bw_style else "bw"
    run = ROOT / "runs" / (args.name or f"{args.arch}_{variant}")
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    model, mean, std = build(args.arch, dropout=args.dropout)
    model.to(device)
    m = load_manifest()
    train_rows = select(m, "train", bw_style=not args.no_bw_style)
    if args.limit:
        train_rows = train_rows.sample(args.limit, random_state=args.seed).reset_index(drop=True)
    val_rows = select(m, "val")
    y_val = targets(val_rows)

    # Cache decoded images only at 96 px: Windows data-loader workers each get a copy.
    cache = args.size <= SIZE
    train_ds = PokeTypeDataset(train_rows, train_transform(mean, std, args.input_size),
                               form_weights(train_rows, args.weight_alpha), cache=cache, size=args.size)
    val_ds = PokeTypeDataset(val_rows, eval_transform(mean, std, args.input_size), cache=cache, size=args.size)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=amp, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    head = head_parameter_names(model)
    groups = [{"params": [p for n, p in model.named_parameters() if n in head], "lr": args.lr},
              {"params": [p for n, p in model.named_parameters() if n not in head],
               "lr": args.lr * args.backbone_lr_mult}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)  # group 0 (head) is logged
    sched = make_scheduler(opt, args.warmup * len(train_dl), args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=amp)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    print(f"{run.name}: {args.arch} ({count_parameters(model) / 1e6:.2f}M params) on {device}; "
          f"{len(train_ds)} train / {len(val_ds)} val images", flush=True)
    log_path = run / "log.csv"
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_mAP", "val_exact_match", "threshold", "seconds"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best, since_best = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y, w in train_dl:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(x), device=device)
                x = lam * x + (1 - lam) * x[perm]
                y = lam * y + (1 - lam) * y[perm]  # soft multi-label targets
                w = lam * w + (1 - lam) * w[perm]
            with torch.autocast(device.type, enabled=amp):
                logits = model(x)
            loss = (bce(logits.float(), y).mean(1) * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)

        probs = predict_proba(model, val_dl, device)
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.from_numpy(probs).clamp(1e-7, 1 - 1e-7), torch.from_numpy(y_val)))
        th = tune_threshold(y_val, probs)
        ev = evaluate(y_val, probs, th)
        row = [epoch, opt.param_groups[0]["lr"], total / n, val_loss, ev["mAP"], ev["exact_match"], th,
               time.time() - t0]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        improved = ev["mAP"] > best
        print(f"epoch {epoch:3d}  train loss {total / n:.4f}  val loss {val_loss:.4f}  "
              f"val mAP {ev['mAP']:.3f}  exact {ev['exact_match']:.3f}  ({row[-1]:.0f}s)"
              + ("  *" if improved else ""), flush=True)
        if improved:
            best, since_best = ev["mAP"], 0
            torch.save({"model": model.state_dict(), "arch": args.arch, "mean": mean, "std": std,
                        "size": args.size, "input_size": args.input_size, "dropout": args.dropout,
                        "types": TYPES, "threshold": th, "epoch": epoch, "args": vars(args)}, run / "best.pt")
            np.save(run / "val_probs.npy", probs)
            best_eval = ev
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"early stop: no val mAP improvement for {args.patience} epochs")
                break

    summary = {"run": run.name, "arch": args.arch, "bw_style": not args.no_bw_style,
               "best_epoch": torch.load(run / "best.pt", map_location="cpu")["epoch"],
               "epochs_run": epoch, "val": best_eval, "args": vars(args)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best val mAP {best:.3f} (epoch {summary['best_epoch']}), exact match "
          f"{best_eval['exact_match']:.3f}; saved {run.relative_to(ROOT)}/best.pt")


if __name__ == "__main__":
    main()
