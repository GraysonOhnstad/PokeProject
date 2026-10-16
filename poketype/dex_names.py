"""Name blending: splice two idea words into candidate names and score how name-like they are.

splice("arctic", "nocturnal") -> [..., "arcturnal", ...]
    Joins a prefix of one word to a suffix of the other, in both orders. Seams are preferred
    where the words share letters (the shared run is kept once: chandelier + lure share "l"
    -> chandelure) or where a syllable ends and the next begins.

NameScorer: how much a string looks like a Pokemon name and reads as pronounceable English.
    Two smoothed character 4-gram models, one trained on real Pokemon names and one on English
    words; the score is their mean log-probability per character.

rank(a, b, scorer): every splice of a and b, scored by name-likeness, how recognizably each
    source word survives, and the seam.
"""

import math
import re
from collections import Counter

import numpy as np

VOWELS = set("aeiouy")

# Words a generated name must not contain, matched anywhere inside it (so "pen" + "island" is caught),
# with inflections listed. Two tiers:
#   HARD  slurs, hate and sexual violence: always blocked, no setting turns this off
#   SOFT  sexual and profane words: blocked by default; the user may turn this tier off
# Name ideas always use both tiers; the setting only decides whether an accidental hidden word in a
# finished name is rejected.
HARD_BLOCKED = set("""nigger negro fag faggot retard nazi hitler racist rape raped rapes raping rapist
genocide suicide""".split())
SOFT_BLOCKED = set("""sex sexy sexual orgasm porn porno nude naked erotic fetish slut whore penis vagina anus
anal boob nipple dick cock cunt fuck shit bitch bastard""".split())


SOFT_ENDINGS = ("ass", "butt", "poo", "turd")  # crude endings, default tier
INNOCENT_ENDINGS = ("glass", "grass", "brass", "class", "mass", "pass", "bass", "lass", "sass")


def blocked(name, soft=True, min_len=4):
    """True if `name` contains a blocked word (words shorter than min_len must match exactly).
    soft=False checks the always-on tier only."""
    n = re.sub(r"[^a-z]", "", name.lower())
    words = HARD_BLOCKED | SOFT_BLOCKED if soft else HARD_BLOCKED
    if soft and n.endswith(SOFT_ENDINGS) and not n.endswith(INNOCENT_ENDINGS):
        return True  # "Batass"
    return any((b in n) if len(b) >= min_len else (b == n) for b in words)


def respellings(name):
    """Common Pokemon-style respellings: c -> k (Magikarp), ck -> k, i <-> y at the end."""
    out = set()
    if "ck" in name:
        out.add(name.replace("ck", "k"))
    for m in re.finditer(r"c(?=[aou]|$)", name):
        out.add(name[:m.start()] + "k" + name[m.end():])
    if name.endswith("y"):
        out.add(name[:-1] + "i")
    elif name.endswith("i"):
        out.add(name[:-1] + "y")
    return out - {name}


def _boundaries(w):
    """Cut points after which a syllable plausibly ends: after a vowel run followed by a consonant,
    or between two consonants. Always includes the word's end."""
    cuts = set()
    for i in range(1, len(w)):
        if w[i - 1] in VOWELS and w[i] not in VOWELS:
            cuts.add(i)          # after a vowel run: "noc|"
        if w[i - 1] not in VOWELS and w[i] not in VOWELS:
            cuts.add(i)          # between consonants: "arc|tic", "noc|turnal"
    cuts.add(len(w))
    return cuts


def _starts(w):
    """Points where a syllable plausibly starts (the mirror of _boundaries); includes 0."""
    starts = {0}
    for i in range(1, len(w)):
        if w[i] not in VOWELS and w[i - 1] in VOWELS:
            starts.add(i)
        if w[i - 1] not in VOWELS and w[i] not in VOWELS:
            starts.add(i)
    return starts


def splice(a, b, min_len=4, max_len=12, soft=True):
    """Candidate blends of words a and b: {candidate: (seam overlap, syllable seam, kept a, kept b)}."""
    a, b = re.sub(r"[^a-z]", "", a.lower()), re.sub(r"[^a-z]", "", b.lower())
    out = {}
    for x, y in ((a, b), (b, a)):
        if len(x) < 2 or len(y) < 2:
            continue
        cuts, starts = _boundaries(x), _starts(y)
        for i in range(2, len(x) + 1):          # keep x[:i]
            for j in range(0, len(y) - 1):      # keep y[j:]
                # longest shared run at the seam: x[:i] ends with what y[j:] begins with, kept once
                # (electR O + R Odent -> electrodent pivots on "ro"); every length is checked, since a
                # two-letter pivot need not start with a one-letter match
                k = max((n for n in range(1, min(i, len(y) - j)) if x[i - n:i] == y[j:j + n]), default=0)
                cand = x[:i] + y[j + k:]
                if not (min_len <= len(cand) <= max_len) or cand in (x, y):
                    continue
                kept_x, kept_y = i / len(x), (len(y) - j) / len(y)
                feat = (k, (i in cuts and j in starts), kept_x if x == a else kept_y, kept_y if x == a else kept_x)
                if cand not in out or feat > out[cand]:
                    out[cand] = feat
    for cand, feat in list(out.items()):
        for r in respellings(cand):
            out.setdefault(r, feat)
    return {c: f for c, f in out.items() if not blocked(c, soft)}


class CharLM:
    """Character n-gram language model with interpolated absolute discounting."""

    def __init__(self, words, order=4, discount=0.75):
        self.order, self.d = order, discount
        self.counts = [Counter() for _ in range(order + 1)]   # counts[n][(history, char)]
        self.hist = [Counter() for _ in range(order + 1)]     # hist[n][history]
        self.types = [Counter() for _ in range(order + 1)]    # distinct continuations per history
        self.vocab = set("$")
        for w in words:
            s = "^" * (order - 1) + w + "$"
            self.vocab |= set(w)
            for i in range(order - 1, len(s)):
                for n in range(1, order + 1):
                    h = s[i - n + 1:i]
                    if self.counts[n][(h, s[i])] == 0:
                        self.types[n][h] += 1
                    self.counts[n][(h, s[i])] += 1
                    self.hist[n][h] += 1

    def prob(self, h, c):
        p = 1.0 / len(self.vocab)
        for n in range(1, self.order + 1):
            hh = h[len(h) - n + 1:] if n > 1 else ""
            total = self.hist[n][hh]
            if total:
                lam = self.d * self.types[n][hh] / total
                p = max(self.counts[n][(hh, c)] - self.d, 0) / total + lam * p
        return p

    def logprob(self, w):
        """Mean log-probability per character (including the end marker)."""
        s = "^" * (self.order - 1) + w + "$"
        lp = [math.log(self.prob(s[i - self.order + 1:i], s[i])) for i in range(self.order - 1, len(s))]
        return sum(lp) / len(lp)


class NameScorer:
    def __init__(self, pokemon_names, english_words):
        def clean(ws):
            return [c for c in (re.sub(r"[^a-z]", "", w.lower()) for w in ws) if c]
        self.poke = CharLM(clean(pokemon_names))
        self.eng = CharLM(clean(english_words))

    def __call__(self, name):
        n = re.sub(r"[^a-z]", "", name.lower())
        return 0.5 * self.poke.logprob(n) + 0.5 * self.eng.logprob(n)


def recognizable(name, word):
    """Longest run of letters `name` shares with `word` (how visibly the word survives)."""
    a, b = name.lower(), word.lower()
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


def rank(a, b, scorer, extra=(), top=None, soft=True):
    """Score every splice of a and b (plus `extra` candidates, e.g. T5 blends); best first.
    Returns [(name, score)]. Score: name-likeness + bonuses for keeping >= 3 recognizable letters
    of each word, for a seam on shared letters or syllable edges, and for a 6-10 letter length."""
    cands = splice(a, b, soft=soft)
    for e in extra:  # e.g. T5 blends: filtered like the splices
        e = re.sub(r"[^a-z]", "", e.lower())
        if e and e not in cands and not blocked(e, soft):
            cands[e] = (0, False, 0.5, 0.5)
    words = [re.sub(r"[^a-z]", "", w.lower()) for w in (a, b)]
    scored = []
    for c, (overlap, syllable, ka, kb) in cands.items():
        r = [min(recognizable(c, w), 5) for w in words]
        kept = min(min(recognizable(c, w) / len(w), 1.0) for w in words if w)  # how whole both words stay
        glued = all(w in c for w in words) and len(c) == sum(map(len, words))  # concatenated, no pivot
        s = (scorer(c)
             + 0.25 * (min(r) >= 3) + 0.05 * sum(r)
             + 0.2 * min(overlap, 3) + 0.15 * syllable      # pivots on shared letters (electro|rodent)
             + 0.4 * kept * (overlap > 0)                     # ... that keep both words nearly whole
             + 0.2 * (6 <= len(c) <= 10)
             - 0.3 * glued)
        scored.append((c.capitalize(), s))
    scored.sort(key=lambda t: -t[1])
    return scored[:top] if top else scored


def features(c, a, b, feat, scorer):
    """Feature vector for candidate c blended from a and b (see LearnedRanker)."""
    overlap, syllable, kept_a, kept_b = feat
    ra, rb = recognizable(c, a), recognizable(c, b)
    n = len(c)
    return [scorer.poke.logprob(c), scorer.eng.logprob(c), min(ra, 6), min(rb, 6), min(ra, rb) >= 3,
            min(overlap, 3), syllable, kept_a, kept_b, n, 6 <= n <= 10, abs(n - 8),
            any(len(w) >= 6 and w in c for w in (a, b)), c.startswith(a[:2]), c.startswith(b[:2])]


class LearnedRanker:
    """Logistic regression over `features`, trained to rank each real name above the other
    splices of its canon source words. Trained only on the families it is given."""

    def __init__(self, scorer):
        self.scorer, self.model = scorer, None

    @staticmethod
    def _clean(w):
        return re.sub(r"[^a-z]", "", w.lower())

    def fit(self, examples, negatives=40, seed=0):
        """examples: [(word_a, word_b, real_name)]; uses those whose real name is a splice."""
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(seed)
        X, y = [], []
        for a, b, real in examples:
            a, b, real = self._clean(a), self._clean(b), self._clean(real)
            cands = splice(a, b)
            if real not in cands:
                continue
            others = [c for c in cands if c != real]
            for c in [real] + list(rng.permutation(others)[:negatives]):
                X.append(features(c, a, b, cands[c], self.scorer))
                y.append(c == real)
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            np.array(X, dtype=float), np.array(y))
        return self

    def rank(self, a, b, top=None):
        a, b = self._clean(a), self._clean(b)
        cands = splice(a, b)
        if not cands:
            return []
        names = list(cands)
        X = np.array([features(c, a, b, cands[c], self.scorer) for c in names], dtype=float)
        s = self.model.decision_function(X)
        order = np.argsort(-s)[:top] if top else np.argsort(-s)
        return [(names[i].capitalize(), float(s[i])) for i in order]
