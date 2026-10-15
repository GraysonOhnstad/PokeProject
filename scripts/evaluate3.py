"""Round 3 final evaluation: the one-time test of the locked-in ensemble (and its
Black/White-style ablation). The only round-3 script that reads the test set.

    python scripts/evaluate3.py
    (defaults: runs/r3/final_model.pt and, if present, runs/r3/final_bw_model.pt)

Test sets (Gen 5, never seen in training, full ensemble):
  primary           Black/White front sprites (headline, with 95% bootstrap CIs)
  test_other_media  Gen 5 as 3D renders, official artwork, Dream World art
  test_shiny        Black/White shiny sprites (unfamiliar colors)
  test_grayscale    the primary sprites with all color removed
Out-of-fold sets (each image predicted only by the fold model that never trained on
its family; Game Boy monochrome sprites exist only for Gen 1):
  oof_gameboy_gray / oof_gameboy_color   the same Gen 1 Pokemon in both forms
Plus: paired bootstrap of main vs BW-ablation, test-set reliability, single-image CPU
inference time, and example predictions with attention maps.

Writes reports/eval_round3.md (+ _examples.png, _reliability.png), results/eval_round3.json.
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
from torchvision import transforms as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate import plot, reliability  # noqa: E402
from calibrate3 import loader  # noqa: E402
from poketype.data import load_image, load_manifest, select, targets  # noqa: E402
from poketype.folds import load_folds  # noqa: E402
from poketype.images import load_rgba  # noqa: E402
from poketype.inference import Predictor  # noqa: E402
from poketype.labels import TYPES, decode  # noqa: E402
from poketype.metrics import evaluate, exact_match, sample_f1  # noqa: E402
from train3 import RUNS  # noqa: E402

SPRITES = ROOT / "data" / "raw" / "sprites"
N_BOOT = 2000
REPORT = ROOT / "reports" / "eval_round3.md"


def test_sets(m):
    other = select(m, "test")
    return {
        "primary": (select(m, "test", primary_test=True), False),
        "test_other_media": (other[other["game"] != "black-white"].reset_index(drop=True), False),
        "test_shiny": (select(m, "test", color_variant="shiny", primary_test=True), False),
        "test_grayscale": (select(m, "test", primary_test=True), True),
    }


def gameboy_sets(m):
    """Gen 1 Pokemon from the cross-validation pool, as monochrome and as color sprites."""
    pool = m[m["split"].isin(["train", "val"]) & ~m["is_duplicate"] & m["game"].isin(["red-blue", "yellow"])]
    gray = pool[pool["color_variant"] == "gray"]
    color = pool[(pool["color_variant"] == "normal") & pool["pokemon_id"].isin(gray["pokemon_id"])]
    return {"oof_gameboy_gray": gray.reset_index(drop=True), "oof_gameboy_color": color.reset_index(drop=True)}


@torch.no_grad()
def ensemble_probs(pred, rows, grayscale=False):
    wrap = (lambda tf: T.Compose([T.Grayscale(num_output_channels=3), tf])) if grayscale else None
    dl = loader(rows, {"size": pred.size, "input_size": pred.input_size, "mean": pred.mean, "std": pred.std}, wrap)
    return torch.cat([pred.batch(x)[0] for x, *_ in dl]).numpy()


@torch.no_grad()
def oof_probs(pred, rows):
    """Each image scored by the one fold model whose validation fold holds its family."""
    fold_of = rows["evolution_chain_id"].map(load_folds()).to_numpy()
    out = np.zeros((len(rows), len(TYPES)), np.float32)
    spec = {"size": pred.size, "input_size": pred.input_size, "mean": pred.mean, "std": pred.std}
    for k, model in enumerate(pred.models):
        idx = np.flatnonzero(fold_of == k)
        if len(idx):
            dl = loader(rows.iloc[idx].reset_index(drop=True), spec)
            z = torch.cat([model(x.to(pred.device))["logits"].float() for x, *_ in dl])
            out[idx] = model.type_probs(z / pred.temperature).cpu().numpy()
    return out


def bootstrap(y, probs, threshold, other=None, seed=0):
    """95% intervals for exact match, F1, top type and mAP, resampling images; with
    `other` (probs, threshold), the interval of the paired difference (this - other)."""
    rng = np.random.default_rng(seed)
    pred = decode(probs, threshold)
    opred = decode(other[0], other[1]) if other else None

    def stats(i, pr, pb):
        present = y[i].sum(0) > 0
        return [exact_match(y[i], pr[i]), sample_f1(y[i], pr[i]),
                float(y[i][np.arange(len(i)), pb[i].argmax(1)].mean()),
                average_precision_score(y[i][:, present], pb[i][:, present], average="macro")]
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        s = stats(i, pred, probs)
        if other:
            s = [a - b for a, b in zip(s, stats(i, opred, other[0]))]
        draws.append(s)
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return {k: [float(a), float(b)] for k, a, b in zip(["exact", "f1", "top1", "mAP"], lo, hi)}


def inference_time(bundle, paths, n=50):
    cpu = Predictor(bundle, "cpu")  # the app runs on CPU
    images = [load_rgba(SPRITES / p) for p in paths[:n]]
    cpu.predict(images[0])
    times = [cpu.predict(im)["seconds"] for im in images]
    return {"n": len(times), "median_s": statistics.median(times), "mean_s": statistics.mean(times),
            "max_s": max(times), "members": len(cpu.models), "threads": torch.get_num_threads()}


def example_grid(pred, rows, probs, path, n=12, seed=0):
    """Random primary-test sprites: prediction vs truth, and (attention models) where the
    model looked for its top predicted type."""
    pick = np.random.default_rng(seed).choice(len(rows), min(n, len(rows)), replace=False)
    dl = loader(rows.iloc[pick].reset_index(drop=True),
                {"size": pred.size, "input_size": pred.input_size, "mean": pred.mean, "std": pred.std})
    attention = torch.cat([a for a in (pred.batch(x)[1] for x, *_ in dl) if a is not None] or [torch.empty(0)])
    has_attn = attention.numel() > 0
    labels = decode(probs[pick], pred.threshold)
    cols = 6
    fig, axes = plt.subplots((len(pick) + cols - 1) // cols * (2 if has_attn else 1), cols,
                             figsize=(2.2 * cols, 3.0 * ((len(pick) + cols - 1) // cols) * (2 if has_attn else 1)))
    axes = np.atleast_2d(axes)
    for j, i in enumerate(pick):
        r = rows.iloc[i]
        truth = [t for t in (r["type1"], r["type2"]) if isinstance(t, str)]
        guess = [TYPES[k] for k in np.argsort(-probs[i]) if labels[j, k]]
        img = np.asarray(load_image(r["path"]))
        row, col = (j // cols) * (2 if has_attn else 1), j % cols
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(f"{r['species_name']}\ntrue: {'/'.join(truth)}\npred: {'/'.join(guess)}", fontsize=8,
                     color="#2a7d2a" if set(truth) == set(guess) else "#b3261e")
        ax.axis("off")
        if has_attn:
            top = int(np.argmax(probs[i]))
            amap = attention[j, top].numpy()
            ax2 = axes[row + 1, col]
            ax2.imshow(img)
            ax2.imshow(np.kron(amap / amap.max(), np.ones((img.shape[0] // amap.shape[0],) * 2)),
                       cmap="magma", alpha=0.55)
            ax2.set_title(f"looking for {TYPES[top]}", fontsize=8)
            ax2.axis("off")
    for ax in axes.ravel()[len(pick) * (2 if has_attn else 1):]:
        ax.axis("off")
    fig.suptitle("Primary test set: random sample" + (" (lower rows: attention for the top type)" if has_attn else ""))
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=110)
    plt.close(fig)


BUNDLES = {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=str(RUNS / "final_model.pt"))
    p.add_argument("--ablation", default=str(RUNS / "final_bw_model.pt"))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BUNDLES["main"] = torch.load(args.bundle, map_location="cpu")
    preds = {"main": Predictor(BUNDLES["main"], device)}
    if Path(args.ablation).exists():
        BUNDLES["bw"] = torch.load(args.ablation, map_location="cpu")
        preds["bw"] = Predictor(BUNDLES["bw"], device)
    m = load_manifest()

    results, cache = {}, {}
    for who, pred in preds.items():
        res = {}
        for name, (rows, gray) in test_sets(m).items():
            probs = ensemble_probs(pred, rows, gray)
            res[name] = evaluate(targets(rows), probs, pred.threshold)
            cache[(who, name)] = (rows, probs)
            if name == "test_other_media":
                y = targets(rows)
                res[name]["by_medium"] = {
                    med: {"n": int(sel.sum()), "f1": sample_f1(y[sel], decode(probs[sel], pred.threshold)),
                          "exact": exact_match(y[sel], decode(probs[sel], pred.threshold))}
                    for med in sorted(rows["medium"].unique()) for sel in [(rows["medium"] == med).to_numpy()]}
        for name, rows in gameboy_sets(m).items():
            res[name] = evaluate(targets(rows), oof_probs(pred, rows), pred.threshold)
        rows, probs = cache[(who, "primary")]
        res["primary"]["ci95"] = bootstrap(targets(rows), probs, pred.threshold)
        results[who] = res
        print(f"{who}: primary exact {res['primary']['exact_match']:.3f}, F1 {res['primary']['sample_f1']:.3f}, "
              f"top type {res['primary']['top1_in_types']:.3f}, mAP {res['primary']['mAP']:.3f}", flush=True)

    rows, probs = cache[("main", "primary")]
    y = targets(rows)
    if "bw" in preds:
        results["main_minus_bw_ci95"] = bootstrap(y, probs, preds["main"].threshold,
                                                  other=(cache[("bw", "primary")][1], preds["bw"].threshold))
    test_all = np.concatenate([cache[("main", n)][1] for n in ("primary", "test_other_media")])
    y_all = np.concatenate([targets(cache[("main", n)][0]) for n in ("primary", "test_other_media")])
    rel = reliability(test_all, y_all)
    plot(rel, rel, preds["main"].temperature, REPORT.with_name("eval_round3_reliability.png"))
    results["test_ece"] = rel[3]
    results["inference_time"] = inference_time(BUNDLES["main"], rows["path"].tolist())
    results["model"] = {k: BUNDLES["main"][k] for k in ("model_cfg", "train_cfg", "temperature", "threshold",
                                                       "config_dir")}
    example_grid(preds["main"], rows, probs, REPORT.with_name("eval_round3_examples.png"))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval_round3.json").write_text(json.dumps(results, indent=1, default=float))
    write_report(results)
    print(f"wrote {REPORT.relative_to(ROOT)}")


def write_report(r):
    main, mc = r["main"], r["model"]
    pr = main["primary"]
    cfg = mc["model_cfg"]
    lines = ["# Round 3 evaluation (test set)\n",
             f"Ensemble of the {mc['train_cfg'].get('backbone', cfg['backbone'])} cross-validation fold models "
             f"(`{mc['config_dir']}`): pool `{cfg['pool']}`, head `{cfg['head']}`"
             f"{' + pair boost' if cfg['pair_boost'] else ''}, auxiliary heads "
             f"{'on' if cfg['aux_sizes'] else 'off'}. Calibrated on out-of-fold predictions: T = "
             f"{mc['temperature']:.2f}, second-type threshold {mc['threshold']:.2f} (tuned for per-image F1).\n",
             "## Primary test: Gen 5 Black/White sprites\n",
             f"{pr['n']} images, species never seen in training in any form.\n",
             "| metric | value | 95% CI |", "|---|---|---|"]
    for key, label in [("exact_match", "exact match (both types right)"), ("sample_f1", "per-image F1"),
                       ("top1_in_types", "top type is a true type"), ("mAP", "mAP")]:
        ci = pr["ci95"][{"exact_match": "exact", "sample_f1": "f1", "top1_in_types": "top1", "mAP": "mAP"}[key]]
        lines.append(f"| {label} | {pr[key]:.3f} | {ci[0]:.3f}-{ci[1]:.3f} |")
    lines.append(f"| predicted dual / actual dual | {pr['dual_predicted']:.2f} / {pr['dual_actual']:.2f} | |")

    base = ROOT / "results" / "baselines.json"
    if base.exists():
        b = json.loads(base.read_text())
        lines += ["\n### Against the baselines\n", "| model | exact | F1 | top type | mAP |", "|---|---|---|---|---|"]
        for name in ["prior", "color_histogram"]:
            e = b[f"{name}/without_bw_style"]["test"]
            f1 = f"{e['sample_f1']:.3f}" if "sample_f1" in e else "-"  # baselines predate the F1 metric
            lines.append(f"| {name.replace('_', ' ')} | {e['exact_match']:.3f} | {f1} | "
                         f"{e['top1_in_types']:.3f} | {e['mAP']:.3f} |")
        lines.append(f"| **round-3 ensemble** | **{pr['exact_match']:.3f}** | **{pr['sample_f1']:.3f}** | "
                     f"**{pr['top1_in_types']:.3f}** | **{pr['mAP']:.3f}** |")

    if "bw" in r:
        bw, d = r["bw"]["primary"], r["main_minus_bw_ci95"]
        lines += ["\n## Ablation: training with Black/White-style sprites\n",
                  "| metric | without BW (main) | with BW | difference (main - BW), 95% CI |", "|---|---|---|---|"]
        for key, k2, label in [("exact_match", "exact", "exact"), ("sample_f1", "f1", "F1"),
                               ("top1_in_types", "top1", "top type"), ("mAP", "mAP", "mAP")]:
            lines.append(f"| {label} | {pr[key]:.3f} | {bw[key]:.3f} | {pr[key] - bw[key]:+.3f} "
                         f"({d[k2][0]:+.3f} to {d[k2][1]:+.3f}) |")
        lines.append("\nA CI that excludes 0 means the difference is unlikely to be chance "
                     "(images resampled together).")

    lines += ["\n## Out-of-distribution checks\n", "| set | model | images | exact | F1 | top type | mAP |",
              "|---|---|---|---|---|---|---|"]
    sets = ["primary", "test_other_media", "test_shiny", "test_grayscale", "oof_gameboy_color", "oof_gameboy_gray"]
    for name in sets:
        for who in [w for w in ("main", "bw") if w in r]:
            e = r[who][name]
            lines.append(f"| {name} | {who} | {e['n']} | {e['exact_match']:.3f} | {e['sample_f1']:.3f} | "
                         f"{e['top1_in_types']:.3f} | {e['mAP']:.3f} |")
    lines.append("\nGen 5 in other media (main), by medium: " + ", ".join(
        f"{k} F1 {v['f1']:.3f} (n={v['n']})" for k, v in main["test_other_media"]["by_medium"].items()))
    lines.append("\nGame Boy rows are Gen 1 Pokemon scored out-of-fold (by the fold model that never trained on "
                 "their family); compare gray against color for the same Pokemon, not against the test rows.")
    lines += ["\n## Per-type results (primary test, main)\n", "| type | precision | recall | F1 | support |",
              "|---|---|---|---|---|"]
    for t in TYPES:
        e = pr["per_type"][t]
        lines.append(f"| {t} | {e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} | {e['support']} |")
    it = r["inference_time"]
    lines += ["\n## Calibration and inference time\n",
              f"Test-set expected calibration error (primary + other media, all 18 outputs): {r['test_ece']:.3f} "
              "(plot: eval_round3_reliability.png).\n",
              f"Single image on CPU, end to end (preprocessing + all {it['members']} ensemble members), "
              f"{it['threads']} threads, {it['n']} images after warm-up: median **{it['median_s'] * 1000:.0f} ms**, "
              f"max {it['max_s'] * 1000:.0f} ms.\n", "![examples](eval_round3_examples.png)"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
