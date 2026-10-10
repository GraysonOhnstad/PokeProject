"""Fetch PokeAPI CSV tables and sprite images.

Standard library only. Both repos are pinned to the commit that is current on
the first run (recorded in data/raw/sources.json), so later runs fetch the same
files. Downloads resume: files already on disk with the expected size are skipped.

    py scripts/fetch_pokeapi.py              # CSVs + sprites
    py scripts/fetch_pokeapi.py --csv-only
    py scripts/fetch_pokeapi.py --refresh    # re-pin to the latest commits

Set GITHUB_TOKEN to raise the GitHub API rate limit (60/hour unauthenticated;
a full sprite listing takes roughly 20-40 calls).
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CSV_DIR = RAW / "csv"
SPRITE_DIR = RAW / "sprites"
SOURCES = RAW / "sources.json"
SPRITE_INDEX = RAW / "sprite_index.json"

DATA_REPO = "PokeAPI/pokeapi"
SPRITES_REPO = "PokeAPI/sprites"

CSV_TABLES = [
    # core: identity, forms, types
    "pokemon", "pokemon_species", "pokemon_forms", "pokemon_types",
    "pokemon_types_past", "types", "generations", "evolution_chains",
    "pokemon_evolution",
    # species-level table
    "pokemon_species_names", "pokemon_egg_groups", "egg_groups",
    "pokemon_stats", "stats", "pokemon_colors", "pokemon_shapes",
    "pokemon_habitats",
    # later project: Pokedex-entry generation
    "pokemon_species_flavor_text", "pokemon_abilities", "abilities",
    "pokemon_moves", "moves",
]

NUMERIC_FILE = re.compile(r"^\d+\.(png|svg|gif)$")
# Gen I/II games whose sprites also exist under transparent/; keep only those copies.
HAS_TRANSPARENT = {"red-blue", "yellow", "crystal", "gold", "silver"}


def http_get(url, token=None, retries=4):
    headers = {"User-Agent": "pokeproject-fetch"}
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 403 and "api.github.com" in url:
                sys.exit(f"GitHub API refused {url} (rate limit?). Set GITHUB_TOKEN or wait an hour.")
            if e.code == 404 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)


def api(path, token):
    return json.loads(http_get(f"https://api.github.com/repos/{path}", token))


def pin_commits(token, refresh):
    if SOURCES.exists() and not refresh:
        return json.loads(SOURCES.read_text())
    sources = {}
    for repo in (DATA_REPO, SPRITES_REPO):
        commit = api(f"{repo}/commits/master", token)
        sources[repo] = {"sha": commit["sha"], "date": commit["commit"]["committer"]["date"]}
    RAW.mkdir(parents=True, exist_ok=True)
    SOURCES.write_text(json.dumps(sources, indent=2))
    return sources


def fetch_csvs(sha):
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{DATA_REPO}/{sha}/data/v2/csv"
    for name in CSV_TABLES:
        dest = CSV_DIR / f"{name}.csv"
        if dest.exists():
            continue
        try:
            dest.write_bytes(http_get(f"{base}/{name}.csv"))
            print(f"  {name}.csv ({dest.stat().st_size / 1e6:.1f} MB)")
        except urllib.error.HTTPError as e:
            print(f"  {name}.csv FAILED: HTTP {e.code}")


def list_tree(tree_sha, prefix, token):
    """All blobs under a tree; descends level by level when a recursive listing is truncated."""
    t = api(f"{SPRITES_REPO}/git/trees/{tree_sha}?recursive=1", token)
    if not t["truncated"]:
        return [{"path": prefix + e["path"], "size": e["size"]}
                for e in t["tree"] if e["type"] == "blob"]
    t = api(f"{SPRITES_REPO}/git/trees/{tree_sha}", token)
    blobs = []
    for e in t["tree"]:
        if e["type"] == "blob":
            blobs.append({"path": prefix + e["path"], "size": e["size"]})
        elif e["type"] == "tree":
            blobs += list_tree(e["sha"], f"{prefix}{e['path']}/", token)
    return blobs


def sprite_index(sha, token, refresh):
    if SPRITE_INDEX.exists() and not refresh:
        return json.loads(SPRITE_INDEX.read_text())
    tree = api(f"{SPRITES_REPO}/git/trees/{sha}", token)
    sprites = next(e for e in tree["tree"] if e["path"] == "sprites")
    tree = api(f"{SPRITES_REPO}/git/trees/{sprites['sha']}", token)
    pokemon = next(e for e in tree["tree"] if e["path"] == "pokemon")
    blobs = list_tree(pokemon["sha"], "", token)
    SPRITE_INDEX.write_text(json.dumps(blobs))
    return blobs


def keep(path):
    parts = path.split("/")
    if not NUMERIC_FILE.match(parts[-1]):
        return False  # cosmetic variants (201-b.png) and non-image files
    if "animated" in parts:
        return False
    if path.startswith("other/showdown/back"):
        return False
    if (len(parts) > 2 and parts[0] == "versions"
            and parts[1] in ("generation-i", "generation-ii")
            and parts[2] in HAS_TRANSPARENT and "transparent" not in parts):
        return False
    return True


def fetch_sprites(sha, blobs, workers):
    todo = []
    for b in blobs:
        dest = SPRITE_DIR / b["path"]
        if not (dest.exists() and dest.stat().st_size == b["size"]):
            todo.append((b, dest))
    total_mb = sum(b["size"] for b in blobs) / 1e6
    print(f"  {len(blobs)} files kept ({total_mb:.0f} MB); {len(todo)} to download")
    base = f"https://raw.githubusercontent.com/{SPRITES_REPO}/{sha}/sprites/pokemon"

    def download(b, dest):
        data = http_get(f"{base}/{b['path']}")
        if len(data) != b["size"]:
            raise IOError(f"size mismatch: got {len(data)}, expected {b['size']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(download, b, dest): b["path"] for b, dest in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
            except Exception as e:
                failed.append((futures[f], repr(e)))
            if i % 2000 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {len(failed)} failed", flush=True)
    if failed:
        (RAW / "failed_downloads.txt").write_text("\n".join(f"{p}\t{e}" for p, e in failed))
        print(f"  {len(failed)} failures listed in data/raw/failed_downloads.txt; rerun to retry")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-only", action="store_true")
    p.add_argument("--refresh", action="store_true", help="re-pin to the latest commits and re-list sprites")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    token = os.environ.get("GITHUB_TOKEN")

    sources = pin_commits(token, args.refresh)
    print(f"pokeapi @ {sources[DATA_REPO]['sha'][:10]}, sprites @ {sources[SPRITES_REPO]['sha'][:10]}")

    print("CSVs:")
    fetch_csvs(sources[DATA_REPO]["sha"])
    if args.csv_only:
        return

    print("Sprites: listing tree")
    blobs = sprite_index(sources[SPRITES_REPO]["sha"], token, args.refresh)
    kept = [b for b in blobs if keep(b["path"])]
    print(f"  {len(blobs)} files in sprites/pokemon/")
    fetch_sprites(sources[SPRITES_REPO]["sha"], kept, args.workers)


if __name__ == "__main__":
    main()
