"""Family-level K-fold cross-validation over all non-test evolution families (round 3).

Round 1-2 used one fixed validation set (~15% of families, 140 species). Round 3 pools
the train and val families and splits them into K folds; each model trains on K-1 folds
and validates on the held-out one, so every non-test Pokemon is validated on exactly
once. Test (Gen 5) and excluded families are never in any fold.

    python -m poketype.folds          # writes splits/cv_folds.csv (committed to the repo)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FOLDS_FILE = ROOT / "splits" / "cv_folds.csv"
K = 5


def make_folds(m, k=K, seed=0, n_candidates=20000):
    """{evolution_chain_id: fold}. Seeded random search for the assignment whose folds are
    most alike: smallest worst-case deviation of any type's (and the dual-typed) share of
    species per fold from the overall share."""
    pool = m[m["split"].isin(["train", "val"]) & m["is_default_form"]].drop_duplicates("species_id")
    counts = pool.melt(id_vars="evolution_chain_id", value_vars=["type1", "type2"]).dropna() \
        .groupby(["evolution_chain_id", "value"]).size().unstack(fill_value=0)
    dual = pool.assign(d=pool["type2"].notna()).groupby("evolution_chain_id")["d"].agg(["sum", "size"])
    feats = np.column_stack([counts.to_numpy(), dual.reindex(counts.index).to_numpy()])
    chains = counts.index.to_numpy()
    total = feats.sum(0)
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(n_candidates):
        assign = rng.permutation(len(chains)) % k
        per_fold = np.stack([feats[assign == f].sum(0) for f in range(k)])
        species = per_fold[:, -1:]  # species per fold
        shares = per_fold[:, :-1] / species  # type shares and dual share per fold
        score = np.abs(shares - total[:-1] / total[-1]).max()
        if score < best_score:
            best, best_score = assign, score
    return dict(zip(chains.tolist(), best.tolist())), best_score


def load_folds():
    f = pd.read_csv(FOLDS_FILE)
    return dict(zip(f["evolution_chain_id"], f["fold"]))


def main():
    from poketype.data import load_manifest
    m = load_manifest()
    folds, score = make_folds(m)
    FOLDS_FILE.parent.mkdir(exist_ok=True)
    pd.DataFrame(sorted(folds.items()), columns=["evolution_chain_id", "fold"]).to_csv(FOLDS_FILE, index=False)
    pool = m[m["split"].isin(["train", "val"]) & m["is_default_form"]].drop_duplicates("species_id")
    pool = pool.assign(fold=pool["evolution_chain_id"].map(folds))
    missing = set(m.loc[m["split"].isin(["train", "val"]), "evolution_chain_id"]) - set(folds)
    assert not missing, f"families without a fold: {sorted(missing)}"
    print(f"{len(folds)} families in {K} folds; worst type-share deviation {score:.3f}")
    print(pool.groupby("fold").agg(families=("evolution_chain_id", "nunique"), species=("species_id", "size"),
                                   dual=("type2", lambda s: round(s.notna().mean(), 2))).to_string())


if __name__ == "__main__":
    main()
