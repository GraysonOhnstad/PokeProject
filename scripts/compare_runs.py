"""Compare training runs on validation only (for choosing a model; never reads test).

    py scripts/compare_runs.py resnet18_bw resnet18_nobw small_cnn_bw small_cnn_nobw

For each run: best epoch, val metrics at that epoch, the epoch-to-epoch noise in val
mAP, calibration (if calibrate.py has run), and the same metrics on the val images in
the Black/White style (unseen Pokemon in the test style; small, so noisy).
Writes reports/run_comparison.md.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import load_manifest, select, targets  # noqa: E402
from poketype.metrics import evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="reports/run_comparison.md", help="report path, relative to the repo")
    args = p.parse_args()

    val = select(load_manifest(), "val")
    y = targets(val)
    bw = (val["game"] == "black-white").to_numpy()
    base = json.loads((ROOT / "results" / "baselines.json").read_text())

    lines = ["# Run comparison (validation only)\n",
             f"Val: {len(val)} images from held-out families, all media. "
             f"\"BW val\": the {int(bw.sum())} of them that are Black/White sprites (the test style).\n",
             "Epoch noise = standard deviation of val mAP over the last 5 epochs of the run, "
             "a rough yardstick for how much of a difference between runs is real.\n",
             "| run | best epoch / ran | val mAP | val exact | val top type | epoch noise | ECE before -> after | "
             "BW val mAP | BW val exact |", "|---|---|---|---|---|---|---|---|---|"]
    for variant in ["with_bw_style", "without_bw_style"]:
        b = base[f"color_histogram/{variant}"]["val"]
        lines.append(f"| color histogram ({variant.replace('_', ' ')}) | - | {b['mAP']:.3f} | {b['exact_match']:.3f} | "
                     f"{b['top1_in_types']:.3f} | - | - | - | - |")
    for name in args.runs:
        run = ROOT / "runs" / name
        summary = json.loads((run / "summary.json").read_text())
        with open(run / "log.csv") as f:
            maps = [float(r["val_mAP"]) for r in csv.DictReader(f)]
        probs = np.load(run / "val_probs.npy")
        th = summary["val"]["threshold"]
        ev, ev_bw = evaluate(y, probs, th), evaluate(y[bw], probs[bw], th)
        cal_path = run / "calibration.json"
        cal = (lambda c: f"{c['ece_before']:.3f} -> {c['ece_after']:.3f} (T={c['temperature']:.2f})")(
            json.loads(cal_path.read_text())) if cal_path.exists() else "not run"
        lines.append(f"| {name} | {summary['best_epoch']} / {summary['epochs_run']} | {ev['mAP']:.3f} | "
                     f"{ev['exact_match']:.3f} | {ev['top1_in_types']:.3f} | {np.std(maps[-5:]):.3f} | {cal} | "
                     f"{ev_bw['mAP']:.3f} | {ev_bw['exact_match']:.3f} |")
    (ROOT / args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
