"""Generate entries for a fixed set of made-up flavor texts and write reports/dex_samples.md.

    py scripts/dex_samples.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dex_generate import show  # noqa: E402
from poketype.dex_pipeline import DexGenerator  # noqa: E402

PROMPTS = [
    (5, "It drifts through abandoned mansions at night, lighting old candles with its ghostly blue flame. "
        "Travelers who follow the light are never seen again."),
    (9, "This tiny rodent stores static electricity in its fluffy cheeks. When startled, it releases a jolt "
        "strong enough to stop a heart."),
    (9, "Ancient legends say it raised the oceans by weeping for a thousand years. Sailors still pray to it "
        "before a storm."),
    (4, "It sleeps curled up in snowdrifts. Its thick fur is covered in icicles that it shoots at anything "
        "that disturbs its rest."),
    (7, "A newly hatched seedling that toddles behind its parent. It shakes the leaf on its head to scatter "
        "pollen when it is happy."),
    (8, "Its shell is as hard as granite. It walks so slowly that moss and small trees grow on its back over "
        "the centuries."),
]


def main():
    gen = DexGenerator(ROOT / "runs" / "dex" / "gen5_heldout" / "dex.pt", ROOT / "runs" / "dex_textgen")
    out = ["# Dex generator v1: sample entries", "",
           "Made-up flavor texts; Gen-5-held-out bundle; no sprite yet. 8 candidates per text, one picked; seed 0. "
           "Regenerate with `py scripts/dex_samples.py`.", ""]
    for era, text in PROMPTS:
        t0 = time.time()
        entry, _ = gen.generate(text, era=era, k=8, seed=0)
        s = show(entry)
        print(s, f"\n({time.time() - t0:.1f} s)\n" + "-" * 60, flush=True)
        out += ["```", s, "```", ""]
    (ROOT / "reports" / "dex_samples.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
