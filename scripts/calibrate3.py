"""Round 3: calibrate a cross-validated ensemble on its out-of-fold predictions and
bundle it for evaluation and the app. Uses the non-test families only.

    python scripts/calibrate3.py                     # the sweep's winner (runs/r3/final.json)
    python scripts/calibrate3.py --bw                # the winner trained with BW-style sprites
    python scripts/calibrate3.py --config runs/r3/resnet18/<id> --name mine

Each fold model predicts the families it never trained on (its validation fold), so
together they give one honest prediction for every non-test Pokemon (~870 species).
On those out-of-fold predictions it fits one temperature (minimizing binary
cross-entropy of the per-type probabilities) and the second-type threshold (maximizing
per-image F1), and draws reliability plots before and after.

Writes runs/r3/<name>_model.pt (all fold models + calibration, what evaluate3.py and
the app load), <name>_calibration.json and <name>_reliability.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate import plot, reliability  # noqa: E402
from poketype.data import PokeTypeDataset, eval_transform, load_manifest, processed_path, targets  # noqa: E402
from poketype.folds import K  # noqa: E402
from poketype.metrics import evaluate, tune_threshold  # noqa: E402
from poketype.typenet import TypeNet  # noqa: E402
from train3 import RUNS, fold_rows  # noqa: E402


def load_fold_models(config_dir, device):
    """[(checkpoint dict, TypeNet)] for folds 0..K-1 of a configuration."""
    out = []
    for fold in range(K):
        path = Path(config_dir) / f"fold{fold}" / "best.pt"
        if not path.exists():
            sys.exit(f"missing {path}: weights are kept only for the sweep's ensemble candidates (seed-1 runs)")
        ckpt = torch.load(path, map_location="cpu")
        model = TypeNet(**ckpt["model_cfg"], pretrained=False)
        model.load_state_dict(ckpt["model"])
        out.append((ckpt, model.to(device).eval()))
    return out


def loader(rows, ckpt, transform_wrap=None, batch_size=128):
    """DataLoader over preprocessed images in the form the checkpoint expects (reads the
    pre-enlarged copies when available, which are identical to enlarging on the fly)."""
    size, upscale = ckpt["size"], ckpt["input_size"]
    folder = size
    if upscale == 2 * size and processed_path(rows["path"].iloc[0], f"{size}x2").exists():
        folder, upscale = f"{size}x2", None
    tf = eval_transform(ckpt["mean"], ckpt["std"], upscale)
    if transform_wrap:
        tf = transform_wrap(tf)
    return DataLoader(PokeTypeDataset(rows, tf, size=folder), batch_size=batch_size, num_workers=0)


@torch.no_grad()
def raw_logits(model, dl, device):
    return torch.cat([model(x.to(device))["logits"].float().cpu() for x, *_ in dl])


def oof_logits(models, m, device):
    """Out-of-fold raw logits and targets for every non-test validation image."""
    zs, ys, folds = [], [], []
    for fold, (ckpt, model) in enumerate(models):
        rows = fold_rows(m, fold, "val")
        zs.append(raw_logits(model, loader(rows, ckpt), device))
        ys.append(targets(rows))
        folds.append(np.full(len(rows), fold))
        print(f"  fold {fold}: {len(rows)} out-of-fold images", flush=True)
    return torch.cat(zs), np.concatenate(ys), np.concatenate(folds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="configuration directory (default: from runs/r3/final.json)")
    p.add_argument("--bw", action="store_true", help="use the winner's BW-style ablation configuration")
    p.add_argument("--name", help="output name (default: final or final_bw)")
    args = p.parse_args()
    if args.config:
        config_dir, name = Path(args.config), args.name or Path(args.config).name
    else:
        final = json.loads((RUNS / "final.json").read_text())
        ids = final["bw_config_ids"] if args.bw else final["config_ids"]
        config_dir = RUNS / final["backbone"] / ids[1]  # seed-1 runs keep their weights
        name = args.name or ("final_bw" if args.bw else "final")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_fold_models(config_dir, device)
    print(f"{name}: {len(models)} fold models from {config_dir} ({models[0][0]['model_cfg']['backbone']}, "
          f"head {models[0][0]['model_cfg']['head']}, pool {models[0][0]['model_cfg']['pool']})", flush=True)
    z, y, _ = oof_logits(models, load_manifest(), device)
    to_probs = models[0][1].cpu().type_probs  # logits are on the CPU

    def probs(t):
        return to_probs(z, t).numpy()

    def bce(t):
        q = np.clip(probs(t), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    t = float(np.exp(minimize_scalar(lambda lt: bce(np.exp(lt)), bounds=(-3, 3), method="bounded").x))
    before, after = probs(1.0), probs(t)
    th_before, th = tune_threshold(y, before, objective="f1"), tune_threshold(y, after, objective="f1")
    rel_b, rel_a = reliability(before, y), reliability(after, y)
    plot(rel_b, rel_a, t, RUNS / f"{name}_reliability.png")
    ev_b, ev_a = evaluate(y, before, th_before), evaluate(y, after, th)

    first = models[0][0]
    bundle = {"version": 3, "models": [c["model"] for c, _ in models], "model_cfg": first["model_cfg"],
              "train_cfg": first["train_cfg"], "temperature": t, "threshold": th, "size": first["size"],
              "input_size": first["input_size"], "mean": first["mean"], "std": first["std"],
              "config_dir": config_dir.relative_to(ROOT).as_posix(), "name": name}
    torch.save(bundle, RUNS / f"{name}_model.pt")
    cal = {"name": name, "config_dir": bundle["config_dir"], "n_oof": len(y), "temperature": t, "threshold": th,
           "threshold_uncalibrated": th_before, "bce_before": bce(1.0), "bce_after": bce(t),
           "ece_before": rel_b[3], "ece_after": rel_a[3], "oof_before": ev_b, "oof_after": ev_a}
    (RUNS / f"{name}_calibration.json").write_text(json.dumps(cal, indent=1))
    print(f"T = {t:.3f}; out-of-fold BCE {cal['bce_before']:.4f} -> {cal['bce_after']:.4f}; "
          f"ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}; F1 threshold {th_before:.2f} -> {th:.2f}")
    print(f"out-of-fold (n={len(y)}): mAP {ev_a['mAP']:.3f}, F1 {ev_a['sample_f1']:.3f}, "
          f"exact {ev_a['exact_match']:.3f}, top type {ev_a['top1_in_types']:.3f}, "
          f"predicts dual {ev_a['dual_predicted']:.0%} (actual {ev_a['dual_actual']:.0%})")
    print(f"wrote runs/r3/{name}_model.pt, {name}_calibration.json, {name}_reliability.png")


if __name__ == "__main__":
    main()
