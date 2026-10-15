"""Pokedex-entry dataset: one row per flavor text with its entry's targets, plus metrics.

Built by scripts/build_dex.py. Fields predicted from text in phase 1:

    types        18-way multi-label            mAP, top-1 accuracy
    shape, color single label                  accuracy
    egg groups   multi-label                   mAP
    height, weight  log-scale regression       typical ratio error (x1.5 = off by 50%)
    stats        6 base stats                  mean absolute error per stat; total (BST) error
    abilities    multi-label over the inventory   recall@5 (Gen 3+ entries only)

Metrics weight each species equally, however many texts it has.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from poketype.labels import TYPES

ROOT = Path(__file__).resolve().parent.parent
DEX = ROOT / "data" / "dex"
STATS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]
GENERATIONS = list(range(1, 10))
NATURAL = {"<self>": "this Pokémon", "<pre-evolution>": "its pre-evolution",
           "<evolution>": "its evolution", "<sibling evolution>": "a related evolution"}


def natural(text):
    """Masked text in plain words, for encoders that never saw the tokens."""
    for tok, words in NATURAL.items():
        text = text.replace(tok, words)
    return re.sub(r"<([^<>]+)>", r"\1", text)


class Vocab:
    def __init__(self, items):
        self.items = list(items)
        self.index = {x: i for i, x in enumerate(self.items)}

    def __len__(self):
        return len(self.items)

    def multi_hot(self, lists):
        y = np.zeros((len(lists), len(self)), dtype=np.float32)
        for r, xs in enumerate(lists):
            for x in xs:
                y[r, self.index[x]] = 1
        return y


def load():
    """(rows, targets, vocabs): one row per text in texts.csv, targets as arrays."""
    texts = pd.read_csv(DEX / "texts.csv")
    species = pd.read_csv(DEX / "species.csv").set_index("species_id")
    entries = {}
    with open(DEX / "entries.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            entries[(e["species_id"], e["version_group_id"])] = e
    ent = [entries[k] for k in zip(texts.species_id, texts.version_group_id)]
    sp = species.loc[texts.species_id]

    eggs = [[g for g in (a, b) if isinstance(g, str)] for a, b in zip(sp.egg_group1, sp.egg_group2)]
    abil = [[a for a in e["abilities"].values() if a] for e in ent]
    vocabs = {
        "types": Vocab(TYPES),
        "shape": Vocab(sorted(species["shape"].unique())),
        "color": Vocab(sorted(species["color"].unique())),
        "egg": Vocab(sorted({g for gs in eggs for g in gs})),
        "abilities": Vocab(sorted({a for a_s in abil for a in a_s})),
    }
    targets = {
        "types": vocabs["types"].multi_hot([e["types"] for e in ent]),
        "shape": np.array([vocabs["shape"].index[s] for s in sp["shape"]]),
        "color": np.array([vocabs["color"].index[c] for c in sp["color"]]),
        "egg": vocabs["egg"].multi_hot(eggs),
        "log_height": np.log(sp.height_dm.to_numpy(np.float64)),
        "log_weight": np.log(sp.weight_hg.to_numpy(np.float64)),
        "stats": np.array([[e["stats"][s] for s in STATS] for e in ent], dtype=np.float64),
        "abilities": vocabs["abilities"].multi_hot(abil),
        "has_abilities": np.array([e["generation"] >= 3 for e in ent]),
        # evolution position and rarity: 0-based classes
        "stage": sp.stage.to_numpy() - 1,
        "line_length": sp.line_length.to_numpy() - 1,
        "is_final": sp.is_final.to_numpy().astype(np.float32),
        "rarity": np.where(sp.is_mythical == 1, 2, np.where(sp.is_legendary == 1, 1, 0)),
    }
    rows = texts.assign(evolution_chain_id=sp.evolution_chain_id.to_numpy())
    return rows, targets, vocabs


def species_weights(species_ids):
    counts = pd.Series(species_ids).map(pd.Series(species_ids).value_counts()).to_numpy()
    return 1.0 / counts


def weighted_map(y, s, w):
    aps = [average_precision_score(y[:, j], s[:, j], sample_weight=w)
           for j in range(y.shape[1]) if y[:, j].any()]
    return float(np.mean(aps))


def score(pred, targets, idx, species_ids):
    """Metrics for predictions `pred` (dict of arrays aligned with `idx`)."""
    w = species_weights(species_ids)
    t = {k: v[idx] for k, v in targets.items()}
    out = {
        "types_mAP": weighted_map(t["types"], pred["types"], w),
        "types_top1": float(np.average(t["types"][np.arange(len(idx)), pred["types"].argmax(1)], weights=w)),
        "shape_acc": float(np.average(pred["shape"].argmax(1) == t["shape"], weights=w)),
        "color_acc": float(np.average(pred["color"].argmax(1) == t["color"], weights=w)),
        "egg_mAP": weighted_map(t["egg"], pred["egg"], w),
        "height_x": float(np.exp(np.average(np.abs(pred["log_height"] - t["log_height"]), weights=w))),
        "weight_x": float(np.exp(np.average(np.abs(pred["log_weight"] - t["log_weight"]), weights=w))),
        "stat_mae": float(np.average(np.abs(pred["stats"] - t["stats"]).mean(1), weights=w)),
        "bst_mae": float(np.average(np.abs(pred["stats"].sum(1) - t["stats"].sum(1)), weights=w)),
    }
    m = t["has_abilities"]
    top5 = np.argsort(-pred["abilities"][m], 1)[:, :5]
    hits = np.take_along_axis(t["abilities"][m], top5, 1).sum(1) / t["abilities"][m].sum(1)
    out["abil_recall5"] = float(np.average(hits, weights=w[m]))
    return out
