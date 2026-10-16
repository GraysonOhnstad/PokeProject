"""Word helpers shared by the Dex naming modules: WordNet access, which words can carry a name,
base forms, and rough same-word tests."""

import difflib

from poketype.dex import DEX
from poketype.dex_names import blocked

STOPWORDS = set("""about above after again against also among around because been before being below
between both could does doing done down during each either even ever every from further have having here
into itself just like many more most much must neither only onto other over same should some such than
that their them then there these they this those through under until very were what when where which
while whom whose will with within without would your""".split())


_WORDNET = None


def wordnet():
    """NLTK WordNet, from data/nltk_data (downloaded once)."""
    global _WORDNET
    if _WORDNET is None:
        import nltk
        nltk.data.path.insert(0, str(DEX.parent / "nltk_data"))
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", download_dir=str(DEX.parent / "nltk_data"), quiet=True)
            from nltk.corpus import wordnet as wn
        _WORDNET = wn
    return _WORDNET


def nameworthy(word):
    """Names are built from nouns and descriptive adjectives: the word's most common sense must
    be one (WordNet), so adverbs and verbs ("slowly", "hesitated") are left out. Blocked words
    never qualify."""
    if blocked(word):
        return False
    if word.endswith(("ing", "ed")) and base_form(word) != word:  # "wandering", but not "king", "seed"
        return False
    s = wordnet().synsets(word)
    return bool(s) and s[0].pos() in "nas"


def related_nouns(word, limit=8):
    """WordNet neighbors of a word's first noun sense: parts and substances (candle -> wick),
    broader kinds and narrower kinds (rodent -> chinchilla, lemming). Single words only."""
    s = wordnet().synsets(word, "n")
    if not s:
        return []
    x = s[0]
    out = []
    for r in (x.part_meronyms() + x.substance_meronyms() + x.member_holonyms() + x.hypernyms()
              + x.hyponyms() + x.part_holonyms()):
        for lemma in r.lemma_names()[:2]:
            w = lemma.lower()
            if w.isalpha() and w not in out:
                out.append(w)
    return out[:limit]


def shared_piece(name, word):
    """Length of the longest stretch of letters `name` shares with `word`."""
    a, b = name.lower(), word.lower().replace(" ", "")
    return difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b)).size


def base_form(w):
    """Dictionary base form for comparing words ("shot" and "shoots" -> "shoot")."""
    wn = wordnet()
    return wn.morphy(w, "v") or wn.morphy(w, "n") or w


def same_root(a, b):
    """Same-word test: shared first four letters, one starting the other, near-identical spelling,
    or the same base form."""
    a, b = a.lower(), b.lower()
    if a[:4] == b[:4] or a.startswith(b) or b.startswith(a):
        return True
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.8:  # shots / shoots
        return True
    return " " not in a + b and base_form(a) == base_form(b)
