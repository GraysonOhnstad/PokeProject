"""Assemble the Hugging Face Space folder for a trained run.

    py scripts/build_space.py --run resnet18_bw                # builds space/
    py scripts/build_space.py --run resnet18_bw --push USER/pokemon-types

space/ contains everything the Space needs: app.py, the inference-only part of the
poketype package, model.pt (weights + architecture + normalization + calibrated
temperature and threshold), classes.json (output index -> type name),
requirements.txt, README.md (Space metadata) and examples/ for the picker:
  - every primary test sprite (Gen 5, Black/White, front, normal colors)
  - Gen 5 Pokemon in other art styles (official artwork, HOME renders), one per species
--push uploads the folder to a Space (needs `hf auth login` first).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poketype.data import load_manifest, select  # noqa: E402
from poketype.images import load_rgba  # noqa: E402
from poketype.labels import TYPES  # noqa: E402

SPRITES = ROOT / "data" / "raw" / "sprites"
PACKAGE_FILES = ["__init__.py", "images.py", "labels.py", "models.py", "inference.py"]
OTHER_ART_GAMES = ["official-artwork", "home"]
EXAMPLE_MAX_SIDE = 256  # downscale large art in examples/ to keep the Space small

REQUIREMENTS = f"""--extra-index-url https://download.pytorch.org/whl/cpu
torch=={torch.__version__.split('+')[0]}
torchvision=={__import__('torchvision').__version__.split('+')[0]}
numpy
pillow
scipy
"""


def readme(gradio_version):
    return f"""---
title: Pokémon Type Classifier
emoji: 🔥
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.11"
app_file: app.py
pinned: false
---

Predicts a Pokémon's type(s) from a single image. Trained without any Generation 5
Pokémon; the example picker shows Gen 5 Pokémon the model has never seen.
"""


def display_name(row, species_names):
    name = species_names.get(row["species_id"], row["species_name"].title())
    if not row["is_default_form"]:
        form = row["pokemon_name"].removeprefix(row["species_name"]).strip("-").replace("-", " ")
        name = f"{name} ({form.title()})"
    return name


def write_examples(out, species_names):
    m = load_manifest()
    ex_dir = out / "examples"
    ex_dir.mkdir()
    index = {}

    def add(row, group):
        im = load_rgba(SPRITES / row["path"])
        if max(im.size) > EXAMPLE_MAX_SIDE:
            im.thumbnail((EXAMPLE_MAX_SIDE, EXAMPLE_MAX_SIDE))
        fname = f"{group}-{row['pokemon_id']:05d}.png"
        im.save(ex_dir / fname)
        index[fname] = {"name": display_name(row, species_names), "group": group,
                        "types": [t for t in (row["type1"], row["type2"]) if isinstance(t, str)]}

    sprites = select(m, "test", primary_test=True)
    sprites = sprites[~sprites["is_female_variant"]].sort_values("pokemon_id")
    for _, row in sprites.iterrows():
        add(row, "gen5_sprites")

    other = select(m, "test")
    other = other[other["game"].isin(OTHER_ART_GAMES) & other["is_default_form"]]
    other = other.sort_values(["species_id", "game"]).drop_duplicates("species_id")  # one per species
    for _, row in other.iterrows():
        add(row, "gen5_other_art")

    (ex_dir / "examples.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(sprites), len(other)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=str(ROOT / "space"))
    p.add_argument("--push", metavar="USER/SPACE", help="upload to this Hugging Face Space")
    args = p.parse_args()
    run, out = ROOT / "runs" / args.run, Path(args.out)

    if out.exists():
        shutil.rmtree(out)
    (out / "poketype").mkdir(parents=True)
    shutil.copy(ROOT / "app" / "app.py", out / "app.py")
    for f in PACKAGE_FILES:
        shutil.copy(ROOT / "poketype" / f, out / "poketype" / f)

    ckpt = torch.load(run / "best.pt", map_location="cpu")
    cal_path = run / "calibration.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    if not cal:
        print("WARNING: run is not calibrated; using T = 1 and the uncalibrated threshold")
    torch.save({"model": ckpt["model"], "arch": ckpt["arch"], "mean": ckpt["mean"], "std": ckpt["std"],
                "types": TYPES, "temperature": cal.get("temperature", 1.0),
                "threshold": cal.get("threshold", ckpt["threshold"]), "source_run": args.run},
               out / "model.pt")
    (out / "classes.json").write_text(json.dumps(TYPES, indent=1))
    (out / "requirements.txt").write_text(REQUIREMENTS)
    import gradio
    (out / "README.md").write_text(readme(gradio.__version__), encoding="utf-8")

    species = pd.read_csv(ROOT / "data" / "species.csv")
    n_sprites, n_other = write_examples(out, dict(zip(species["species_id"], species["name"])))
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"built {out} ({size_mb:.1f} MB): model from {args.run}, {n_sprites} sprite and {n_other} other-art examples")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push, repo_type="space",
                          commit_message=f"Deploy model from run {args.run}")
        print(f"pushed to https://huggingface.co/spaces/{args.push}")


if __name__ == "__main__":
    main()
