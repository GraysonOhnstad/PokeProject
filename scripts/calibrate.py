"""Temperature scaling on validation, with reliability plots before and after.

Fits one temperature T (logits / T before the sigmoid) minimizing BCE on val, then
re-tunes the second-type threshold on the calibrated confidences. Temperature is
monotonic, so per-type rankings (and mAP) are unchanged; it makes confidences honest.
Uses val only.

    py scripts/calibrate.py --run resnet18_bw
Writes runs/<run>/calibration.json and runs/<run>/reliability.png.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import PokeTypeDataset, eval_transform, load_manifest, select, targets  # noqa: E402
from poketype.metrics import evaluate, tune_threshold  # noqa: E402
from poketype.models import build  # noqa: E402

N_BINS = 10


@torch.no_grad()
def val_logits(run, device):
    ckpt = torch.load(run / "best.pt", map_location="cpu")
    model, mean, std = build(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    rows = select(load_manifest(), "val")
    ds = PokeTypeDataset(rows, eval_transform(mean, std, ckpt.get("input_size")), size=ckpt.get("size", 96))
    dl = DataLoader(ds, batch_size=128, num_workers=4)
    z = torch.cat([model(x.to(device)).float().cpu() for x, *_ in dl]).numpy()
    return z, targets(rows), ckpt["threshold"]


def bce(z, y, t):
    p = np.clip(1 / (1 + np.exp(-z / t)), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def reliability(probs, y, n_bins=N_BINS):
    """Per-bin (mean confidence, observed frequency, count) over all 18 x n outputs,
    and the expected calibration error."""
    p, y = probs.ravel(), y.ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    conf, freq, count = [], [], []
    for b in range(n_bins):
        sel = bins == b
        count.append(int(sel.sum()))
        conf.append(float(p[sel].mean()) if sel.any() else np.nan)
        freq.append(float(y[sel].mean()) if sel.any() else np.nan)
    count = np.array(count)
    gaps = np.nan_to_num(np.abs(np.array(conf) - np.array(freq)))
    return np.array(conf), np.array(freq), count, float((gaps * count).sum() / count.sum())


def plot(before, after, t, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    edges = np.linspace(0, 1, N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    panels = [("Before (T = 1)", before), (f"After (T = {t:.2f})", after)]
    for ax, (title, (conf, freq, count, ece)) in zip(axes, panels):
        freq0, conf0 = np.nan_to_num(freq), np.nan_to_num(conf)
        ax.bar(centers, freq0, width=1 / N_BINS, edgecolor="white", color="#4a7fc1",
               label="observed: fraction where the type is present")
        ax.bar(centers, conf0 - freq0, bottom=freq0, width=1 / N_BINS, color="#d1495b", alpha=0.35,
               edgecolor="#d1495b", hatch="//", label="gap to the bin's mean confidence")
        ax.plot([0, 1], [0, 1], ls="--", color="#888", label="perfect calibration")
        for x, n, top in zip(centers, count, np.maximum(freq0, conf0)):
            ax.text(x, top + 0.015, f"n={n}", ha="center", fontsize=6, color="#333")
        ax.set(title=f"{title}   ECE {ece:.3f}", xlabel="predicted confidence", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("fraction where the type is present")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Validation reliability, all 18 type outputs pooled")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = ROOT / "runs" / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z, y, th_before = val_logits(run, device)
    res = minimize_scalar(lambda log_t: bce(z, y, np.exp(log_t)), bounds=(-3, 3), method="bounded")
    t = float(np.exp(res.x))
    p_before, p_after = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-z / t))
    th_after = tune_threshold(y, p_after)
    rel_before, rel_after = reliability(p_before, y), reliability(p_after, y)
    plot(rel_before, rel_after, t, run / "reliability.png")

    ev_before, ev_after = evaluate(y, p_before, th_before), evaluate(y, p_after, th_after)
    out = {"temperature": t, "threshold": th_after, "threshold_uncalibrated": th_before,
           "val_bce_before": bce(z, y, 1.0), "val_bce_after": bce(z, y, t),
           "ece_before": rel_before[3], "ece_after": rel_after[3],
           "val_exact_match_before": ev_before["exact_match"], "val_exact_match_after": ev_after["exact_match"]}
    (run / "calibration.json").write_text(json.dumps(out, indent=2))
    print(f"T = {t:.3f}; val BCE {out['val_bce_before']:.4f} -> {out['val_bce_after']:.4f}; "
          f"ECE {out['ece_before']:.4f} -> {out['ece_after']:.4f}; threshold {th_before:.2f} -> {th_after:.2f}; "
          f"exact match {ev_before['exact_match']:.3f} -> {ev_after['exact_match']:.3f}")
    print(f"wrote {run.relative_to(ROOT)}/calibration.json and reliability.png")


if __name__ == "__main__":
    main()
