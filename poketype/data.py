"""Manifest selection, labels, and the PyTorch Dataset over preprocessed 96x96 images."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from poketype.images import BACKGROUND, SIZE

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
PROCESSED = ROOT / "data" / "processed" / str(SIZE)
MASKS = ROOT / "data" / "processed" / f"{SIZE}_mask"

# types.csv ids 1-18, in id order; index i is output unit i.
TYPES = ["normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
         "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark", "fairy"]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}


def load_manifest():
    return pd.read_csv(MANIFEST, low_memory=False)


def select(m, split, bw_style=True, color_variant="normal", primary_test=False):
    """Rows for one split under the default filter: one color variant, no exact duplicates.

    bw_style=False drops Black/White sprites (the with/without-BW-style ablation).
    primary_test=True restricts to the primary test subset (black-white, unflagged).
    """
    keep = (m["split"] == split) & ~m["is_duplicate"]
    if color_variant is not None:
        keep &= m["color_variant"] == color_variant
    if not bw_style:
        keep &= ~m["bw_style"]
    if primary_test:
        keep &= (m["game"] == "black-white") & m["provenance_flag"].isna()
    return m[keep].reset_index(drop=True)


def targets(rows):
    """Multi-hot (n, 18) float array of each row's types."""
    y = np.zeros((len(rows), len(TYPES)), dtype=np.float32)
    for i, (t1, t2) in enumerate(zip(rows["type1"], rows["type2"])):
        y[i, TYPE_INDEX[t1]] = 1
        if isinstance(t2, str):
            y[i, TYPE_INDEX[t2]] = 1
    return y


def form_weights(rows, alpha=1.0):
    """Per-image weights proportional to 1 / (images of that form)^alpha, mean 1.

    alpha=1 gives every form equal total weight (a Gen 9 form with 5 images counts as
    much as a Gen 1 form with 40); alpha=0 is unweighted.
    """
    counts = rows.groupby("pokemon_id")["path"].transform("size").to_numpy(dtype=np.float64)
    w = counts ** -alpha
    return (w / w.mean()).astype(np.float32)


def processed_path(path):
    return PROCESSED / Path(path).with_suffix(".png")


def mask_path(path):
    return MASKS / Path(path).with_suffix(".png")


def load_image(path):
    return Image.open(processed_path(path)).convert("RGB")


def load_mask(path):
    """Boolean (96, 96) array: True where the pixel is part of the Pokemon."""
    return np.asarray(Image.open(mask_path(path))) > 0


def train_transform(mean, std):
    # No color jitter: color carries much of the type signal. Flips and small shifts
    # only, with nearest-neighbor resampling and a background fill to match preprocessing.
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.05),
                       interpolation=T.InterpolationMode.NEAREST, fill=BACKGROUND),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(mean, std):
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


class PokeTypeDataset(Dataset):
    """Yields (image tensor, 18-dim multi-hot target, loss weight)."""

    def __init__(self, rows, transform, weights=None):
        self.paths = rows["path"].tolist()
        self.y = torch.from_numpy(targets(rows))
        self.w = torch.from_numpy(weights if weights is not None else np.ones(len(rows), np.float32))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_image(self.paths[i])), self.y[i], self.w[i]
