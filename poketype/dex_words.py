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


# Established combining forms for concepts Pokemon names often carry (Electrode, Hydreigon, Pyroar,
# Cryogonal, Terrakion, Aerodactyl, Lumineon, Noctowl, Toxapex, Ferroseed): a word whose base form or
# start matches a key also yields its forms, so a blend can pivot on the familiar prefix (electrodent).
COMBINING = {
    "electr": ["electro"], "shock": ["electro"], "spark": ["electro"], "thunder": ["electro"], "volt": ["volt"],
    "water": ["aqua", "hydro"], "aqua": ["aqua"], "hydr": ["hydro"], "sea": ["mari", "thalasso"],
    "ocean": ["mari"], "fire": ["pyro", "igni"], "flame": ["pyro", "flam"], "pyr": ["pyro"], "burn": ["pyro"],
    "ice": ["cryo", "glacio"], "cold": ["cryo"], "frost": ["cryo"], "snow": ["nivi"], "freez": ["cryo"],
    "earth": ["terra", "geo"], "ground": ["terra", "geo"], "soil": ["terra"], "stone": ["litho", "petro"],
    "rock": ["litho", "petro"], "air": ["aero"], "wind": ["aero", "anemo"], "sky": ["aero", "celest"],
    "flight": ["aero"], "light": ["lumi", "photo"], "lumin": ["lumi"], "sun": ["helio", "sol"],
    "star": ["astro", "stell"], "moon": ["luna", "seleno"], "night": ["noct"], "dark": ["noct", "umbra"],
    "shadow": ["umbra", "skia"], "time": ["chrono"], "poison": ["toxi", "toxo"], "venom": ["toxi"],
    "toxic": ["toxi"], "mind": ["psycho", "psy"], "psych": ["psycho", "psy"], "dream": ["oneiro"],
    "iron": ["ferro"], "metal": ["metallo"], "steel": ["ferro"], "life": ["bio"], "plant": ["phyto"],
    "leaf": ["phyllo", "folio"], "flower": ["antho", "flora"], "tree": ["dendro"], "wood": ["xylo"],
    "sound": ["sono", "phono"], "music": ["melo"], "dragon": ["draco"], "ghost": ["spectro", "phantasm"],
    "spirit": ["spectro"], "fairy": ["fae"], "fight": ["pugil"], "punch": ["pugil"], "fast": ["tachy"],
    "speed": ["tachy"], "giant": ["giga", "mega"], "huge": ["mega", "giga"], "small": ["micro", "mini"],
    "tiny": ["micro", "mini"], "ancient": ["paleo", "archeo"], "fossil": ["paleo"], "insect": ["entomo"],
    "bug": ["entomo"], "bird": ["ornitho", "avi"], "fish": ["ichthyo", "pisci"], "snake": ["ophio", "serpen"],
}


def combining_forms(word):
    """Combining forms for a word: by its base form or its start ("electricity" -> electro)."""
    w, out = word.lower(), []
    b = base_form(w)
    for key, forms in COMBINING.items():
        if b == key or w.startswith(key):
            out += [f for f in forms if f != w]
    return list(dict.fromkeys(out))
