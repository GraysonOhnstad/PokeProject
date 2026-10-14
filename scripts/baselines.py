"""Baselines: most-frequent-type prior and a color-histogram classifier.

Both are trained on train, have their second-type threshold (and the histogram model's
regularization) tuned on val, and are scored on val and the primary test subset. Run
for both training variants (with / without Black/White-style sprites).

    py scripts/baselines.py
Writes reports/baselines.md and results/baselines.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import TYPES, form_weights, load_image, load_manifest, select, targets  # noqa: E402
from poketype.metrics import evaluate, tune_threshold  # noqa: E402

BINS = 8  # per RGB channel -> 512-dim joint histogram
REPORT = ROOT / "reports" / "baselines.md"
RESULTS = ROOT / "results" / "baselines.json"


def color_histogram(path):
    """Normalized joint RGB histogram over non-background (non-white) pixels."""
    px = np.asarray(load_image(path)).reshape(-1, 3)
    px = px[(px < 250).any(axis=1)]
    idx = (px // (256 // BINS)).astype(np.int64)
    h = np.bincount(idx[:, 0] * BINS * BINS + idx[:, 1] * BINS + idx[:, 2], minlength=BINS ** 3)
    return np.sqrt(h / max(h.sum(), 1))  # Hellinger: sqrt of proportions


def features(rows, cache):
    for p in rows["path"]:
        if p not in cache:
            cache[p] = color_histogram(p)
    return np.stack([cache[p] for p in rows["path"]])


def prior_model(y_train, w_train):
    freq = (y_train * w_train[:, None]).sum(0) / w_train.sum()
    return lambda n: np.tile(freq, (n, 1))


def fit_histogram_model(x_train, y_train, w_train, x_val, y_val, tolerance=0.002):
    """One logistic regression per type. Val mAP plateaus as C shrinks while predictions
    collapse toward the prior, so take the least-regularized C within `tolerance` of the
    best val mAP."""
    scaler = StandardScaler().fit(x_train)
    xt, xv = scaler.transform(x_train), scaler.transform(x_val)
    present = y_val.sum(0) > 0
    fits = []
    for c in [0.00001, 0.00003, 0.0001, 0.0003, 0.001]:
        models = [LogisticRegression(C=c, max_iter=2000).fit(xt, y_train[:, k], sample_weight=w_train)
                  for k in range(len(TYPES))]
        pv = np.column_stack([mdl.predict_proba(xv)[:, 1] for mdl in models])
        score = average_precision_score(y_val[:, present], pv[:, present], average="macro")
        print(f"    C={c}: val mAP {score:.3f}", flush=True)
        fits.append((c, score, models))
    best = max(s for _, s, _ in fits)
    c, _, models = max((f for f in fits if f[1] >= best - tolerance), key=lambda f: f[0])
    return c, lambda x: np.column_stack([mdl.predict_proba(scaler.transform(x))[:, 1] for mdl in models])


def main():
    m = load_manifest()
    val = select(m, "val")
    test = select(m, "test", primary_test=True)
    y_val, y_test = targets(val), targets(test)
    cache, results = {}, {}

    for variant, bw in [("with_bw_style", True), ("without_bw_style", False)]:
        train = select(m, "train", bw_style=bw)
        y_train, w_train = targets(train), form_weights(train)
        print(f"{variant}: {len(train)} train images", flush=True)

        prior = prior_model(y_train, w_train)
        pv, pt = prior(len(val)), prior(len(test))
        th = tune_threshold(y_val, pv)
        results[f"prior/{variant}"] = {"val": evaluate(y_val, pv, th), "test": evaluate(y_test, pt, th)}

        print("  color histogram: extracting features / fitting", flush=True)
        x_train, x_val, x_test = features(train, cache), features(val, cache), features(test, cache)
        c, predict = fit_histogram_model(x_train, y_train, w_train, x_val, y_val)
        pv, pt = predict(x_val), predict(x_test)
        th = tune_threshold(y_val, pv)
        results[f"color_histogram/{variant}"] = {
            "C": c, "val": evaluate(y_val, pv, th), "test": evaluate(y_test, pt, th)}

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    write_report(results, len(val), len(test))
    print(f"wrote {REPORT.relative_to(ROOT)} and {RESULTS.relative_to(ROOT)}")


def write_report(results, n_val, n_test):
    lines = ["# Baselines\n",
             f"Val: {n_val} images (held-out families, all media). "
             f"Test: {n_test} images (primary subset: Gen 5, Black/White front sprites).\n",
             "Second-type threshold tuned on val for exact match. "
             "Training images weighted so each form counts equally.\n",
             "| model | train variant | split | exact match | top type correct | mAP | macro F1 | threshold |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        model, variant = name.split("/")
        for split in ["val", "test"]:
            e = r[split]
            lines.append(f"| {model} | {variant.replace('_', ' ')} | {split} | {e['exact_match']:.3f} | "
                         f"{e['top1_in_types']:.3f} | {e['mAP']:.3f} | {e['macro_f1']:.3f} | {e['threshold']:.2f} |")
    best = results["color_histogram/with_bw_style"]["test"]["per_type"]
    lines += ["\n## Color histogram (with BW style): per-type test results\n",
              "| type | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for t in TYPES:
        e = best[t]
        lines.append(f"| {t} | {e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} | {e['support']} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
