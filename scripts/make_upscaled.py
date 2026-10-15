"""Pre-enlarge a preprocessed image set by an integer factor (nearest-neighbor).

    python scripts/make_upscaled.py            # data/processed/96 -> data/processed/96x2

Identical, pixel for pixel, to enlarging on the fly during training (each pixel becomes a
factor x factor block), but reading the stored copies is several times faster on Windows.
Reads only data/processed/<size>/ and <size>_mask/, so the training machine can run it
without the raw sprites.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import mask_path, processed_path  # noqa: E402


def enlarge(args):
    path, size, factor = args
    for src, dst in [(processed_path(path, size), processed_path(path, f"{size}x{factor}")),
                     (mask_path(path, size), mask_path(path, f"{size}x{factor}"))]:
        if dst.exists():
            continue
        im = Image.open(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.resize((im.width * factor, im.height * factor), Image.NEAREST).save(dst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    paths = pd.read_csv(ROOT / "data" / "manifest.csv", low_memory=False)["path"]
    print(f"enlarging {len(paths)} images and masks {args.size} -> {args.size * args.factor} px...", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(enlarge, ((p, args.size, args.factor) for p in paths), chunksize=128))
    print(f"done: data/processed/{args.size}x{args.factor}/ and {args.size}x{args.factor}_mask/")


if __name__ == "__main__":
    main()
