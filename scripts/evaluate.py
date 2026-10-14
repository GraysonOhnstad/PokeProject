"""Final evaluation of a trained (and ideally calibrated) run. The only script that
reads the test set: run it once a model is locked in, not to tune anything.

    py scripts/evaluate.py --run resnet18_bw

Evaluates:
  primary          Gen 5 Black/White front sprites (the headline number, with 95% CIs)
  test_other_media Gen 5 Pokemon as 3D renders, official artwork and Dream World art
  test_shiny       Gen 5 Black/White shiny sprites (unfamiliar colors)
  test_grayscale   the primary sprites converted to grayscale (no color at all)
  val_gameboy_gray real Game Boy monochrome sprites; these exist only for Gen 1, so they
                   come from val families (unseen in training, but used for tuning)
  val_gameboy_color  the same val Pokemon's Red/Blue/Yellow color sprites, for contrast
plus single-image inference time (CPU, end to end from the raw file).

Writes reports/eval_<run>.md, reports/eval_<run>_examples.png, results/eval_<run>.json.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import (PokeTypeDataset, eval_transform, load_image, load_manifest,  # noqa: E402
                           select, targets)
from poketype.images import load_rgba  # noqa: E402
from poketype.inference import Predictor  # noqa: E402
from poketype.labels import TYPES, decode  # noqa: E402
from poketype.metrics import evaluate, exact_match  # noqa: E402
from poketype.models import predict_proba  # noqa: E402

SPRITES = ROOT / "data" / "raw" / "sprites"
N_BOOT = 2000
N_TIMING = 50


def eval_sets(m):
    gen1_games = ["red-blue", "yellow"]
    val_gb = select(m, "val", color_variant="gray")
    val_gb = val_gb[val_gb["game"].isin(gen1_games)]
    val_color = select(m, "val")
    val_color = val_color[val_color["game"].isin(gen1_games)
                          & val_color["pokemon_id"].isin(val_gb["pokemon_id"])]
    other = select(m, "test")
    other = other[other["game"] != "black-white"]
    return {
        "primary": (select(m, "test", primary_test=True), False),
        "test_other_media": (other, False),
        "test_shiny": (select(m, "test", color_variant="shiny", primary_test=True), False),
        "test_grayscale": (select(m, "test", primary_test=True), True),
        "val_gameboy_gray": (val_gb, False),
        "val_gameboy_color": (val_color, False),
    }


def bootstrap_ci(y, probs, threshold, seed=0):
    """95% percentile intervals for exact match and mAP, resampling images."""
    rng = np.random.default_rng(seed)
    pred = decode(probs, threshold)
    em, ap = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        em.append(exact_match(y[i], pred[i]))
        present = y[i].sum(0) > 0
        ap.append(average_precision_score(y[i][:, present], probs[i][:, present], average="macro"))
    return {"exact_match": np.percentile(em, [2.5, 97.5]).tolist(), "mAP": np.percentile(ap, [2.5, 97.5]).tolist()}


def time_single_image(predictor, paths):
    """Seconds per prediction, one image at a time from the raw file (preprocessing +
    forward pass), after one warm-up call."""
    images = [load_rgba(SPRITES / p) for p in paths]
    predictor.predict(images[0])
    times = [predictor.predict(im)["seconds"] for im in images]
    return {"n": len(times), "mean_s": statistics.mean(times), "median_s": statistics.median(times),
            "max_s": max(times), "device": str(predictor.device), "threads": torch.get_num_threads()}


def example_grid(rows, probs, threshold, path, n=24, seed=0):
    pick = np.random.default_rng(seed).choice(len(rows), min(n, len(rows)), replace=False)
    pred = decode(probs, threshold)
    fig, axes = plt.subplots(4, 6, figsize=(12, 11.5))
    for ax, i in zip(axes.ravel(), pick):
        r = rows.iloc[i]
        truth = [t for t in (r["type1"], r["type2"]) if isinstance(t, str)]
        guess = [TYPES[k] for k in np.argsort(-probs[i]) if pred[i, k]]
        ok = set(truth) == set(guess)
        ax.imshow(load_image(r["path"]))
        ax.set_title(f"{r['species_name']}\ntrue: {'/'.join(truth)}\npred: {'/'.join(guess)}",
                     fontsize=8, color="#2a7d2a" if ok else "#b3261e")
        ax.axis("off")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("Primary test set: random sample (green = both types exactly right)")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = ROOT / "runs" / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor.from_run(run, device)
    ckpt = torch.load(run / "best.pt", map_location="cpu")
    mean, std = ckpt["mean"], ckpt["std"]
    calibrated = (run / "calibration.json").exists()

    m = load_manifest()
    results, primary_probs = {}, None
    for name, (rows, grayscale) in eval_sets(m).items():
        tf = eval_transform(mean, std)
        if grayscale:
            tf = T.Compose([T.Grayscale(num_output_channels=3), tf])
        dl = DataLoader(PokeTypeDataset(rows, tf), batch_size=128, num_workers=0)
        probs = predict_proba(predictor.model, dl, device, predictor.temperature)
        y = targets(rows)
        results[name] = evaluate(y, probs, predictor.threshold)
        if name == "primary":
            primary_probs, primary_rows = probs, rows
            results[name]["ci95"] = bootstrap_ci(y, probs, predictor.threshold)
        if name == "test_other_media":
            results[name]["by_medium"] = {
                med: {"n": int(sel.sum()), "exact_match": exact_match(y[sel], decode(probs[sel], predictor.threshold))}
                for med in sorted(rows["medium"].unique()) for sel in [(rows["medium"] == med).to_numpy()]}
        print(f"{name:18s} n={len(rows):5d}  exact {results[name]['exact_match']:.3f}  "
              f"mAP {results[name]['mAP']:.3f}", flush=True)

    cpu = Predictor.from_run(run, "cpu")
    results["inference_time"] = time_single_image(cpu, primary_rows["path"].iloc[:N_TIMING].tolist())
    print(f"single-image inference (CPU): median {results['inference_time']['median_s'] * 1000:.1f} ms")
    results["model"] = {"run": args.run, "arch": ckpt["arch"], "epoch": ckpt["epoch"], "calibrated": calibrated,
                        "temperature": predictor.temperature, "threshold": predictor.threshold,
                        "bw_style_in_training": not ckpt["args"].get("no_bw_style", False)}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"eval_{args.run}.json").write_text(json.dumps(results, indent=2))
    example_grid(primary_rows, primary_probs, predictor.threshold, ROOT / "reports" / f"eval_{args.run}_examples.png")
    write_report(args.run, results)
    print(f"wrote reports/eval_{args.run}.md")


def write_report(run, r):
    md = r["model"]
    pr = r["primary"]
    lines = [f"# Evaluation: {run}\n",
             f"{md['arch']}, best epoch {md['epoch']}, trained {'with' if md['bw_style_in_training'] else 'without'} "
             f"Black/White-style sprites; {'calibrated' if md['calibrated'] else 'NOT calibrated'} "
             f"(T = {md['temperature']:.2f}), second-type threshold {md['threshold']:.2f}.\n",
             "## Headline: primary test set\n",
             f"Gen 5 Black/White front sprites, {pr['n']} images, species never seen in training.\n",
             "| metric | value | 95% CI |", "|---|---|---|",
             f"| exact match (both types right) | {pr['exact_match']:.3f} | "
             f"{pr['ci95']['exact_match'][0]:.3f}-{pr['ci95']['exact_match'][1]:.3f} |",
             f"| top type is a true type | {pr['top1_in_types']:.3f} | |",
             f"| mAP | {pr['mAP']:.3f} | {pr['ci95']['mAP'][0]:.3f}-{pr['ci95']['mAP'][1]:.3f} |",
             f"| macro F1 | {pr['macro_f1']:.3f} | |",
             f"| predicted dual-typed / actually dual-typed | {pr['dual_predicted']:.2f} / {pr['dual_actual']:.2f} | |"]

    base_path = ROOT / "results" / "baselines.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        variant = "with_bw_style" if md["bw_style_in_training"] else "without_bw_style"
        lines += ["\n### Against the baselines (same training variant)\n",
                  "| model | exact match | top type correct | mAP |", "|---|---|---|---|"]
        for name in ["prior", "color_histogram"]:
            b = base[f"{name}/{variant}"]["test"]
            lines.append(f"| {name.replace('_', ' ')} | {b['exact_match']:.3f} | "
                         f"{b['top1_in_types']:.3f} | {b['mAP']:.3f} |")
        lines.append(f"| **{run}** | **{pr['exact_match']:.3f}** | "
                     f"**{pr['top1_in_types']:.3f}** | **{pr['mAP']:.3f}** |")

    lines += ["\n## Out-of-distribution checks\n", "| set | images | exact match | top type correct | mAP |",
              "|---|---|---|---|---|"]
    ood = ["primary", "test_other_media", "test_shiny", "test_grayscale", "val_gameboy_color", "val_gameboy_gray"]
    for name in ood:
        e = r[name]
        lines.append(f"| {name} | {e['n']} | {e['exact_match']:.3f} | {e['top1_in_types']:.3f} | {e['mAP']:.3f} |")
    lines.append("\nGen 5 in other media, by medium: " + ", ".join(
        f"{k} {v['exact_match']:.3f} (n={v['n']})" for k, v in r["test_other_media"]["by_medium"].items()))
    lines.append("\nThe Game Boy rows come from val families (Gen 1 only has monochrome sprites); compare gray "
                 "against color for the same Pokemon rather than against the test numbers.")

    lines += ["\n## Per-type results (primary test)\n", "| type | precision | recall | F1 | support |",
              "|---|---|---|---|---|"]
    for t in TYPES:
        e = pr["per_type"][t]
        lines.append(f"| {t} | {e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} | {e['support']} |")

    it = r["inference_time"]
    lines += ["\n## Inference time\n",
              f"Single image, end to end (preprocessing + forward pass; file already in memory), {it['device']}, "
              f"{it['threads']} threads, {it['n']} images after warm-up: median **{it['median_s'] * 1000:.1f} ms**, "
              f"mean {it['mean_s'] * 1000:.1f} ms, max {it['max_s'] * 1000:.1f} ms.\n",
              f"![examples](eval_{run}_examples.png)"]
    (ROOT / "reports" / f"eval_{run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
