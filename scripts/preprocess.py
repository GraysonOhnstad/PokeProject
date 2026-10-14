"""Crop, flatten and resize every manifest image to 96x96 RGB, plus a subject mask.

Reads data/manifest.csv and writes, mirroring the data/raw/sprites/ layout:
    data/processed/96/<path>.png       model input (subject on white)
    data/processed/96_mask/<path>.png  subject mask (white = Pokemon pixel)
and reports/preprocess_samples.png: raw | processed | mask check (subject on magenta)
for a few images per medium, for eyeballing the result.

    py scripts/preprocess.py [--workers 8] [--force]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import mask_path, processed_path  # noqa: E402
from poketype.images import SIZE, composite, load_rgba, prepare_with_mask  # noqa: E402

SPRITES = ROOT / "data" / "raw" / "sprites"
SAMPLES = ROOT / "reports" / "preprocess_samples.png"


def process(args):
    path, medium, force = args
    dest, mdest = processed_path(path), mask_path(path)
    if dest.exists() and mdest.exists() and not force:
        return path, None
    try:
        rgb, mask = prepare_with_mask(load_rgba(SPRITES / path), pixel_art=medium == "pixel")
        for d in (dest, mdest):
            d.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dest)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mdest)
        return path, None
    except Exception as e:
        return path, repr(e)


def mask_check(path):
    """Processed image with non-subject pixels painted magenta."""
    a = np.asarray(Image.open(processed_path(path))).copy()
    a[np.asarray(Image.open(mask_path(path))) == 0] = (255, 0, 255)
    return Image.fromarray(a)


def sample_sheet(m, per_medium=5, seed=0):
    """One row per medium of (raw, processed, mask check) triples. Always includes the
    Red/Green sprites, which have no transparency and rely on the border flood fill."""
    rows = []
    normal = m[m["color_variant"] == "normal"]
    groups = list(normal.groupby("medium")) + [("red-green", normal[normal["game"] == "red-green-japan"])]
    for _, g in groups:
        triples = []
        for p in g.sample(min(per_medium, len(g)), random_state=seed)["path"]:
            raw = composite(load_rgba(SPRITES / p))
            raw.thumbnail((SIZE, SIZE))
            triples += [raw, Image.open(processed_path(p)), mask_check(p)]
        rows.append(triples)
    cell, gap = SIZE + 4, 12
    width = max(len(r) for r in rows) * cell + (per_medium - 1) * gap
    sheet = Image.new("RGB", (width, len(rows) * cell), (200, 200, 200))
    for y, r in enumerate(rows):
        for i, im in enumerate(r):
            x = i * cell + (i // 3) * gap
            sheet.paste(im, (x + (cell - im.width) // 2, y * cell + (cell - im.height) // 2))
    SAMPLES.parent.mkdir(exist_ok=True)
    sheet.save(SAMPLES)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="reprocess images that already exist")
    args = p.parse_args()

    m = pd.read_csv(ROOT / "data" / "manifest.csv", low_memory=False)
    print(f"processing {len(m)} images...", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = ((p, med, args.force) for p, med in zip(m["path"], m["medium"]))
        results = list(pool.map(process, jobs, chunksize=64))
    failed = [(p, e) for p, e in results if e]
    for p, e in failed[:20]:
        print(f"  FAILED {p}: {e}")
    print(f"done: {len(m) - len(failed)} ok, {len(failed)} failed")
    sample_sheet(m)
    print(f"wrote {SAMPLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
