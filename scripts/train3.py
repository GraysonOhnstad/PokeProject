"""Round 3: train one cross-validation fold of one TypeNet configuration.

Used by sweep3.py (which calls train_fold in-process); also runnable alone:

    python scripts/train3.py --fold 0 --set backbone=resnet18 pool=attn epochs=2 limit=512

Same training recipe as train.py (weighted BCE or set cross-entropy, AdamW, linear warmup
+ cosine decay, early stopping on val mAP), with TypeNet's options and family-level
K-fold cross-validation: the model trains on the other folds' families and is
validated on fold `fold`. The test set is never read. Results are cached on disk
(run_dir/result.json), so a finished fold is never retrained.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import (PokeTypeDataset, TrainTransform, aux_targets, aux_vocab, eval_transform,  # noqa: E402
                           form_weights, load_manifest, processed_path, targets)
from poketype.folds import K, load_folds  # noqa: E402
from poketype.metrics import evaluate  # noqa: E402
from poketype.typenet import IMAGENET_MEAN, IMAGENET_STD, TypeNet, type_targets_for_sets  # noqa: E402

RUNS = ROOT / "runs" / "r3"
LR_DEFAULTS = {"resnet18": 3e-4, "resnet50": 3e-4, "convnext_tiny": 1e-4}
TRAIN_EVAL_N = 2000  # training images scored after training, for the generalization gap

# Round 2's final settings: the starting point for every backbone.
BASE = dict(backbone="resnet18", pool="avg", attn_lambda=0.2, learn_lambda=False, head="sigmoid",
            pair_boost=False, pair_l2=0.01, aux_weight=0.0, lr=None, backbone_lr_mult=1.0,
            weight_decay=0.05, dropout=0.5, mixup=0.4, epochs=60, warmup=2, patience=10, batch_size=64,
            aug_strong=False, bg_prob=0.0, weight_alpha=1.0, size=96, input_size=192, bw_style=False,
            tap="final", seed=0, limit=None)


def config_id(cfg):
    """Short stable hash of a configuration (seed included; fold excluded)."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def fold_rows(m, fold, part, bw_style=True):
    """Training ("train") or validation ("val") rows for a fold, under the usual filter:
    normal colors, no exact duplicates; bw_style=False drops Black/White sprites (train only)."""
    fold_of = m["evolution_chain_id"].map(load_folds())
    pool = m["split"].isin(["train", "val"]) & ~m["is_duplicate"] & (m["color_variant"] == "normal")
    keep = pool & ((fold_of != fold) if part == "train" else (fold_of == fold))
    if part == "train" and not bw_style:
        keep &= ~m["bw_style"]
    return m[keep].reset_index(drop=True)


def save_robust(write, path, attempts=5):
    """Write via a temp file and swap into place, retrying (antivirus/sync tools on Windows
    can briefly lock fresh files)."""
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(1, attempts + 1):
        try:
            with open(tmp, "wb") as f:
                write(f)
            os.replace(tmp, path)
            return
        except (OSError, RuntimeError) as e:
            if attempt == attempts:
                raise
            print(f"  save of {path.name} failed ({e}); retrying", flush=True)
            time.sleep(2 * attempt)


def soft_targets(index, n, lam, perm):
    one_hot = F.one_hot(index, n).float()
    return one_hot if perm is None else lam * one_hot + (1 - lam) * one_hot[perm]


def train_fold(cfg, fold, run_dir, workers=4, quiet=False, save_weights=True):
    """Train on all folds but `fold`, early-stop on fold `fold`; return the result dict."""
    run_dir = Path(run_dir)
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**cfg, "fold": fold}, indent=1))
    seed = cfg["seed"] * 100 + fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    lr = cfg["lr"] or LR_DEFAULTS[cfg["backbone"]]

    m = load_manifest()
    species = pd.read_csv(ROOT / "data" / "species.csv")
    train_rows = fold_rows(m, fold, "train", cfg["bw_style"])
    if cfg["limit"]:
        train_rows = train_rows.sample(cfg["limit"], random_state=seed).reset_index(drop=True)
    val_rows = fold_rows(m, fold, "val")
    y_val = targets(val_rows)

    # Read the pre-enlarged copies when they exist (identical pixels, much faster loading).
    folder, upscale = cfg["size"], cfg["input_size"]
    if upscale and upscale == 2 * cfg["size"] and processed_path(train_rows["path"][0], f"{cfg['size']}x2").exists():
        folder, upscale = f"{cfg['size']}x2", None
    vocab = aux_vocab(species)
    aux_sizes = {k: len(v) for k, v in vocab.items()} if cfg["aux_weight"] > 0 else {}
    model = TypeNet(cfg["backbone"], pool=cfg["pool"], head=cfg["head"], pair_boost=cfg["pair_boost"],
                    aux_sizes=aux_sizes, dropout=cfg["dropout"], attn_lambda=cfg["attn_lambda"],
                    learn_lambda=cfg["learn_lambda"], tap=cfg["tap"]).to(device)

    tf = TrainTransform(IMAGENET_MEAN, IMAGENET_STD, upscale, strong=cfg["aug_strong"], bg_prob=cfg["bg_prob"])
    train_ds = PokeTypeDataset(train_rows, tf, form_weights(train_rows, cfg["weight_alpha"]), size=folder,
                               with_mask=cfg["bg_prob"] > 0,
                               aux=aux_targets(train_rows, species, vocab) if aux_sizes else None)
    val_ds = PokeTypeDataset(val_rows, eval_transform(IMAGENET_MEAN, IMAGENET_STD, upscale), size=folder)
    kw = dict(batch_size=cfg["batch_size"], num_workers=workers, pin_memory=amp, persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    head = model.head_parameter_names()
    groups = [{"params": [p for n, p in model.named_parameters() if n in head], "lr": lr},
              {"params": [p for n, p in model.named_parameters() if n not in head],
               "lr": lr * cfg["backbone_lr_mult"]}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    steps, warm = cfg["epochs"] * len(train_dl), cfg["warmup"] * len(train_dl)

    def lr_factor(s):  # linear warmup, then cosine decay to 0 at `epochs`
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler(enabled=amp)

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_mAP", "seconds"])
    if not quiet:
        print(f"  fold {fold}: {len(train_ds)} train / {len(val_ds)} val images, {cfg['backbone']} on {device}",
              flush=True)
    best, best_state, best_probs, best_epoch, since = -1.0, None, None, 0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in train_dl:
            x, y, w = (t.to(device, non_blocking=True) for t in batch[:3])
            aux = {k: v.to(device) for k, v in batch[3].items()} if len(batch) > 3 else {}
            perm, lam = None, 1.0
            if cfg["mixup"] > 0:
                lam = float(np.random.beta(cfg["mixup"], cfg["mixup"]))
                perm = torch.randperm(len(x), device=device)
                x = lam * x + (1 - lam) * x[perm]
                w = lam * w + (1 - lam) * w[perm]
            with torch.autocast(device.type, enabled=amp):
                out = model(x)
            logits = out["logits"].float()
            if cfg["head"] == "set":
                t = soft_targets(type_targets_for_sets(y), logits.shape[1], lam, perm)
                per = -(t * logits.log_softmax(1)).sum(1)
            else:
                t = y if perm is None else lam * y + (1 - lam) * y[perm]
                per = F.binary_cross_entropy_with_logits(logits, t, reduction="none").mean(1)
            if aux:
                egg = aux["egg"] if perm is None else lam * aux["egg"] + (1 - lam) * aux["egg"][perm]
                a = F.binary_cross_entropy_with_logits(out["aux"]["egg"].float(), egg, reduction="none").mean(1)
                for k in ("shape", "color"):
                    tk = soft_targets(aux[k], out["aux"][k].shape[1], lam, perm)
                    a = a + -(tk * out["aux"][k].float().log_softmax(1)).sum(1)
                per = per + cfg["aux_weight"] * a / 3
            loss = (per * w).mean()
            boost = model.pair_boost()
            if boost is not None:
                loss = loss + cfg["pair_l2"] * (boost ** 2).sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)

        model.eval()
        with torch.no_grad():
            probs = torch.cat([model.type_probs(model(xb.to(device))["logits"].float()).cpu()
                               for xb, *_ in val_dl]).numpy()
        val_map = evaluate(y_val, probs, 0.5)["mAP"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, opt.param_groups[0]["lr"], total / n, val_map, time.time() - t0])
        if not quiet:
            print(f"    epoch {epoch:3d}  train loss {total / n:.4f}  val mAP {val_map:.3f}  ({time.time() - t0:.0f}s)"
                  + ("  *" if val_map > best else ""), flush=True)
        if val_map > best:
            best, best_epoch, since = val_map, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs = probs
        else:
            since += 1
            if since >= cfg["patience"]:
                break

    # Generalization gap: the best checkpoint scored on a fixed sample of training images
    # (no augmentation) vs on validation. A large gap means capacity went into memorizing.
    model.load_state_dict(best_state)
    model.eval()
    sample = train_rows.sample(min(TRAIN_EVAL_N, len(train_rows)), random_state=0).reset_index(drop=True)
    sample_dl = DataLoader(PokeTypeDataset(sample, eval_transform(IMAGENET_MEAN, IMAGENET_STD, upscale), size=folder),
                           batch_size=cfg["batch_size"], num_workers=workers, pin_memory=amp)
    with torch.no_grad():
        train_probs = torch.cat([model.type_probs(model(xb.to(device))["logits"].float()).cpu()
                                 for xb, *_ in sample_dl]).numpy()
    train_map = evaluate(targets(sample), train_probs, 0.5)["mAP"]
    n_params = sum(p.numel() for p in model.parameters())

    ckpt = {"version": 3, "model": best_state, "model_cfg": model.cfg, "train_cfg": copy.deepcopy(cfg),
            "fold": fold, "epoch": best_epoch, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "size": cfg["size"], "input_size": cfg["input_size"]}
    if save_weights:  # the sweep keeps weights only for candidate ensemble members (disk space)
        save_robust(lambda f: torch.save(ckpt, f), run_dir / "best.pt")
    save_robust(lambda f: np.save(f, best_probs), run_dir / "val_probs.npy")
    result = {"fold": fold, "val_mAP": best, "best_epoch": best_epoch, "epochs_run": epoch,
              "train_mAP": train_map, "gap": train_map - best, "params": n_params,
              "n_val": len(val_rows),
              "val_paths_hash": hashlib.sha1("".join(val_rows["path"]).encode()).hexdigest()[:10]}
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    del model, opt, train_dl, val_dl
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_value(v):
    for cast in (json.loads,):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, required=True, choices=range(K))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override BASE settings")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    cfg = dict(BASE)
    for kv in args.set:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown setting {k}"
        cfg[k] = parse_value(v)
    run_dir = RUNS / cfg["backbone"] / config_id(cfg) / f"fold{args.fold}"
    print(json.dumps(train_fold(cfg, args.fold, run_dir, workers=args.workers), indent=1))


if __name__ == "__main__":
    main()
