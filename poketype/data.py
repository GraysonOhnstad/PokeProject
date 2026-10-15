"""Manifest selection, labels, augmentation, and the PyTorch Dataset over preprocessed images."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from poketype.images import BACKGROUND, SIZE
from poketype.labels import TYPE_INDEX, TYPES  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
PROCESSED_ROOT = ROOT / "data" / "processed"  # <size>/ and <size>_mask/ per preprocessing size


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


def processed_path(path, size=SIZE):
    return PROCESSED_ROOT / str(size) / Path(path).with_suffix(".png")


def mask_path(path, size=SIZE):
    return PROCESSED_ROOT / f"{size}_mask" / Path(path).with_suffix(".png")


def load_image(path, size=SIZE):
    return Image.open(processed_path(path, size)).convert("RGB")


def load_mask(path, size=SIZE):
    """Boolean (size, size) array: True where the pixel is part of the Pokemon."""
    return np.asarray(Image.open(mask_path(path, size))) > 0


def _upscale(input_size):
    """Nearest-neighbor enlargement to the model's input size (e.g. 96 -> 192); keeps
    pixel art crisp and adds no detail."""
    return [T.Resize(input_size, interpolation=T.InterpolationMode.NEAREST)] if input_size else []


def train_transform(mean, std, input_size=None):
    # No color jitter: color carries much of the type signal. Flips and small shifts
    # only, with nearest-neighbor resampling and a background fill to match preprocessing.
    return T.Compose(_upscale(input_size) + [
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.05),
                       interpolation=T.InterpolationMode.NEAREST, fill=BACKGROUND),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(mean, std, input_size=None):
    return T.Compose(_upscale(input_size) + [T.ToTensor(), T.Normalize(mean, std)])


AUX_COLUMNS = {"egg": ["egg_group1", "egg_group2"], "shape": ["shape"], "color": ["color"]}


def aux_vocab(species):
    """Category lists for the auxiliary heads (egg groups multi-label; shape, color single)."""
    return {name: sorted(set(species[cols].stack().dropna())) for name, cols in AUX_COLUMNS.items()}


def aux_targets(rows, species, vocab):
    """Per-image auxiliary labels, looked up by species: {"egg": (n, k) multi-hot float,
    "shape": (n,) int, "color": (n,) int}. Used only as training targets, never as input."""
    sp = species.set_index("species_id")
    out = {}
    for name, cols in AUX_COLUMNS.items():
        index = {v: i for i, v in enumerate(vocab[name])}
        vals = sp.loc[rows["species_id"], cols].to_numpy()
        if name == "egg":
            t = np.zeros((len(rows), len(index)), np.float32)
            for i, row in enumerate(vals):
                for v in row:
                    if isinstance(v, str):
                        t[i, index[v]] = 1
        else:
            t = np.array([index[v[0]] for v in vals], np.int64)
        out[name] = t
    return out


class TrainTransform:
    """Round-3 training augmentation, applied jointly to the image and its subject mask:
    optional nearest-neighbor upscale, flip, small shift/scale (strong=True: larger), then
    optionally paint the background (mask == 0) a random solid color with probability
    bg_prob, so the model learns to ignore the background. Pokemon colors are never
    changed. Without bg_prob the mask is not needed."""

    def __init__(self, mean, std, input_size=None, strong=False, bg_prob=0.0):
        from torchvision.transforms import v2
        translate, scale = ((0.1, 0.1), (0.8, 1.1)) if strong else ((0.05, 0.05), (0.9, 1.05))
        self.geometric = v2.Compose(
            ([v2.Resize(input_size, interpolation=v2.InterpolationMode.NEAREST)] if input_size else [])
            + [v2.RandomHorizontalFlip(),
               v2.RandomAffine(degrees=0, translate=translate, scale=scale,
                               interpolation=v2.InterpolationMode.NEAREST, fill={"Image": BACKGROUND, "Mask": 0})])
        self.normalize = v2.Normalize(mean, std)
        self.bg_prob = bg_prob

    def __call__(self, im, mask=None):
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as TF
        image = tv_tensors.Image(TF.pil_to_tensor(im))
        m = tv_tensors.Mask(torch.from_numpy(mask.astype(np.uint8))[None] if mask is not None
                            else torch.ones(1, *image.shape[-2:], dtype=torch.uint8))
        image, m = self.geometric(image, m)
        x = image.float() / 255
        if self.bg_prob and torch.rand(()) < self.bg_prob:
            x = torch.where(m.bool(), x, torch.rand(3, 1, 1))
        return self.normalize(x)


class PokeTypeDataset(Dataset):
    """Yields (image tensor, 18-dim multi-hot target, loss weight[, aux targets dict]).

    folder: which preprocessed set to read (data/processed/<folder>/), e.g. 96, 192, or
    "96x2" (the 96 set pre-enlarged 2x, identical to enlarging on the fly but faster).
    with_mask: also load the subject mask and pass it to the transform (random backgrounds).
    """

    def __init__(self, rows, transform, weights=None, cache=False, size=SIZE, with_mask=False, aux=None):
        self.paths = rows["path"].tolist()
        self.y = torch.from_numpy(targets(rows))
        self.w = torch.from_numpy(weights if weights is not None else np.ones(len(rows), np.float32))
        self.transform = transform
        self.size = size
        self.with_mask = with_mask
        self.aux = {k: torch.from_numpy(v) for k, v in aux.items()} if aux else None
        # cache=True decodes every image up front; epochs then skip disk reads.
        self.images = [load_image(p, size) for p in self.paths] if cache else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = self.images[i] if self.images is not None else load_image(self.paths[i], self.size)
        x = self.transform(im, load_mask(self.paths[i], self.size)) if self.with_mask else self.transform(im)
        if self.aux is None:
            return x, self.y[i], self.w[i]
        return x, self.y[i], self.w[i], {k: v[i] for k, v in self.aux.items()}
