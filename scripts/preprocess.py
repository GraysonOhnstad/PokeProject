"""Crop, flatten and resize every manifest image to 96x96 RGB.

Reads data/manifest.csv and writes data/processed/96/<path>.png (same folder layout
as data/raw/sprites/), plus reports/preprocess_samples.png: raw vs processed for a
few images per medium, for eyeballing the result.

    py scripts/preprocess.py [--workers 8] [--force]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.images import SIZE, composite, load_rgba, prepare  # noqa: E402

SPRITES = ROOT / "data" / "raw" / "sprites"
PROCESSED = ROOT / "data" / "processed" / str(SIZE)
SAMPLES = ROOT / "reports" / "preprocess_samples.png"


def processed_path(path):
    return PROCESSED / Path(path).with_suffix(".png")


def process(args):
    path, medium, force = args
    dest = processed_path(path)
    if dest.exists() and not force:
        return path, None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        prepare(load_rgba(SPRITES / path), pixel_art=medium == "pixel").save(dest)
        return path, None
    except Exception as e:
        return path, repr(e)


def sample_sheet(m, per_medium=6, seed=0):
    """Rows of (raw, processed) pairs, one row per medium."""
    rows = []
    for _, g in m[m["color_variant"] == "normal"].groupby("medium"):
        picks = g.sample(min(per_medium, len(g)), random_state=seed)["path"]
        pairs = []
        for p in picks:
            raw = composite(load_rgba(SPRITES / p))
            raw.thumbnail((SIZE, SIZE))
            pairs += [raw, Image.open(processed_path(p))]
        rows.append(pairs)
    cell, gap = SIZE + 4, 12
    width = max(len(r) for r in rows) * cell + (per_medium - 1) * gap
    sheet = Image.new("RGB", (width, len(rows) * cell), (200, 200, 200))
    for y, r in enumerate(rows):
        for i, im in enumerate(r):
            x = i * cell + (i // 2) * gap
            sheet.paste(im, (x + (cell - im.width) // 2, y * cell + (cell - im.height) // 2))
    SAMPLES.parent.mkdir(exist_ok=True)
    sheet.save(SAMPLES)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="reprocess images that already exist")
    args = p.parse_args()

    m = pd.read_csv(ROOT / "data" / "manifest.csv", low_memory=False)
    print(f"processing {len(m)} images to {PROCESSED.relative_to(ROOT)}...", flush=True)
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
