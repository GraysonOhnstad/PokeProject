"""Pun mode: names that deliberately sound like an existing word or phrase fitting the Pokemon.

Canon examples: Slaking (slack + king) sounds like "slacking", Seaking like "seeking", Patrat
(patrol + rat) like "pack rat", Watchog like "watchdog", Bewear like "beware", Wynaut like "why not".

    index = PunIndex(phrases)                         # WordNet words and phrases (+ per-entry extras)
    puns(pairs, scorer, index, fit, extra=llm_names)  # [(name, score, phrase)], best first

A candidate is a pun when its pronunciation key (Metaphone) is at most one edit from a phrase's
and it is spelled differently from that phrase (so it is a new name, not the word itself). The
phrase must fit the Pokemon: `fit` scores a phrase's meaning against the description.
"""

import re
from collections import defaultdict

import jellyfish

from poketype.dex_names import blocked, rank, recognizable


def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


def sound_key(s):
    return jellyfish.metaphone(letters(s)) if letters(s) else ""


class PunIndex:
    """Phrases bucketed by the first letter and length of their pronunciation key."""

    def __init__(self, phrases, min_len=4):
        self.buckets = defaultdict(list)
        self.size = 0
        for p in phrases:
            self.add(p, min_len)

    def add(self, phrase, min_len=4):
        flat = letters(phrase)
        key = sound_key(phrase)
        if len(flat) < min_len or len(key) < 2:
            return
        self.buckets[(key[0], len(key))].append((key, flat, phrase))
        self.size += 1

    def contains(self, name):
        key, flat = sound_key(name), letters(name)
        return any(f == flat for _, f, _ in self.buckets.get((key[:1], len(key)), ())) if key else False

    def near(self, name, max_dist=1, min_sim=0.85):  # canon puns score >= 0.88 (Patrat ~ pack rat)
        """Phrases that sound like `name`: [(phrase, key distance, spelling similarity)]. The sound
        key ignores vowels, so spelling similarity (which keeps them) must also be high."""
        key, flat = sound_key(name), letters(name)
        if len(key) < 2:
            return []
        out = []
        for n in range(len(key) - max_dist, len(key) + max_dist + 1):
            for k, f, phrase in self.buckets.get((key[0], n), ()):
                if f == flat:
                    continue  # the phrase itself is not a pun
                d = jellyfish.levenshtein_distance(key, k)
                if d <= max_dist:
                    sim = jellyfish.jaro_winkler_similarity(flat, f)
                    if sim >= min_sim:
                        out.append((phrase, d, sim))
        return out


def common_phrases(wordnet_lemmas, common_words):
    """Pun targets: common single words, and multi-word phrases made only of common words
    ("pack rat", "snow-covered"), not WordNet's taxonomy ("paridae")."""
    common = set(common_words)
    out = set(w for w in common if len(w) >= 3)
    for lemma in wordnet_lemmas:
        parts = re.split(r"[_\-\s]+", lemma.lower())
        if len(parts) == 1:
            if parts[0] in common:
                out.add(parts[0])
        elif all(p in common for p in parts):
            out.add(" ".join(parts))
    return sorted(out)


def puns(pairs, scorer, indexes, fit, extra=(), soft=True, min_fit=0.2, top=None):
    """Pun names from idea pairs [(a, b)] (their splices) plus `extra` candidate names (e.g. the
    model's own pun spellings), matched against one or more PunIndex objects. fit: callable
    list[str] -> array of fit scores for phrases. Score: the wordplay score of the splice + how
    closely and how fittingly it echoes a phrase. Returns [(name, score, phrase, "a, b")]."""
    indexes = indexes if isinstance(indexes, (list, tuple)) else [indexes]

    def near(name):
        return [m for ix in indexes for m in ix.near(name)]

    def known(name):
        return any(ix.contains(name) for ix in indexes)
    found = {}
    for a, b in pairs:
        for name, base in rank(a, b, scorer, soft=soft):
            if known(name):
                continue  # an existing word ("slacking") is the target, not a name
            for phrase, dist, sim in near(name):
                found.setdefault(name.lower(), []).append((phrase, dist, sim, base, a, b))
    for e in extra:
        name = letters(e)
        if not name or blocked(name, soft) or known(name):
            continue
        for phrase, dist, sim in near(name):
            found.setdefault(name, []).append((phrase, dist, sim, scorer(name), None, None))
    if not found:
        return []
    phrases = sorted({p for matches in found.values() for p, *_ in matches})
    fits = dict(zip(phrases, fit(phrases)))
    out = []
    for name, matches in found.items():
        best = None
        for phrase, dist, sim, base, a, b in matches:
            f = fits[phrase]
            if f < min_fit or (a and letters(phrase) in (letters(a), letters(b))):
                continue  # sounding like one of its own ingredients ("Tormancy" ~ "dormancy") is not a pun;
                # a different word from the same root is (Slaking, from "slack", ~ "slacking")
            pieces = min(recognizable(name, letters(a)), recognizable(name, letters(b))) if a else 3
            if pieces < 3:
                continue  # must still carry both halves ("Shion" keeps too little of shrimp + cavitation)
            s = base + 0.8 * f + 0.3 * (dist == 0) + 0.2 * sim + 0.1 * min(pieces, 3)
            if best is None or s > best[0]:
                best = (s, phrase, f"{a}, {b}" if a else "")
        if best:
            out.append((name.capitalize(), best[0], best[1], best[2]))
    out.sort(key=lambda t: -t[1])
    return out[:top] if top else out
