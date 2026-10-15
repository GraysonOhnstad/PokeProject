"""Decoding per-type confidences into 1-2 types, and the evaluation metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from poketype.labels import TYPES, decode  # noqa: F401  (decode re-exported)


def exact_match(y, pred):
    return float((y.astype(np.int8) == pred).all(axis=1).mean())


def sample_f1(y, pred):
    """Mean per-image F1 between predicted and true type sets (partial credit: one of two
    types right on a dual-typed Pokemon scores 0.67)."""
    hits = (y.astype(np.int8) & pred).sum(1)
    return float((2 * hits / (y.sum(1) + pred.sum(1))).mean())


OBJECTIVES = {"exact": exact_match, "f1": sample_f1}


def tune_threshold(y, probs, objective="exact", grid=np.linspace(0.05, 0.95, 91)):
    """Second-type threshold maximizing the objective on these images (ties -> lowest).
    Rounds 1-2 tuned for exact match; round 3 onward tunes for per-image F1."""
    score = OBJECTIVES[objective]
    scores = [score(y, decode(probs, t)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, probs, threshold):
    """Metrics dict for multi-hot targets `y` and per-type confidences `probs`."""
    pred = decode(probs, threshold)
    present = y.sum(0) > 0  # AP is undefined for a type absent from this set
    p, r, f, support = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    top1 = y[np.arange(len(y)), probs.argmax(1)].mean()
    return {
        "n": len(y),
        "threshold": threshold,
        "exact_match": exact_match(y, pred),
        "sample_f1": sample_f1(y, pred),
        "top1_in_types": float(top1),  # top type is one of the true types
        "mAP": float(average_precision_score(y[:, present], probs[:, present], average="macro")),
        "macro_f1": float(f[present].mean()),
        "dual_predicted": float((pred.sum(1) == 2).mean()),
        "dual_actual": float((y.sum(1) == 2).mean()),
        "per_type": {t: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                         "support": int(support[i])} for i, t in enumerate(TYPES)},
    }
