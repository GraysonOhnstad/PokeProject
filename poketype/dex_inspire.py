"""Name inspiration: the two halves of a name, what is interesting about each, and which words
can carry them.

    insp = Inspiration(llm_path, embed)          # llm_path: a GGUF instruction model, or None
    result = insp(text)
    result["halves"]["is"]       what the Pokemon is (creature, object, legend)
    result["halves"]["defines"]  what defines it (trait, behavior, element, theme)
        each: {"mention": words of the text naming it, or "",
               "articles": [(title, description, fit)], "facts": [(sentence, title)],
               "words": name ingredients, the half's own words first}
    result["genus"]              category options ("the X Pokemon")
    result["basis"], result["facts"], result["basis_words"], result["concepts"]: the same
        information flattened (half "is" as the basis, half "defines" as the concepts)

For each half:
  1. The model (llama.cpp) points to the words of the text that name it, if any; the words must
     really occur in the text. Otherwise it distills the half from the text. Either way it gives
     Wikipedia title guesses.
  2. Wikipedia verifies the guesses: the article must exist, not be a disambiguation or Pokemon
     page, and its introduction must relate to the text; a failed guess is retried by its head noun.
  3. Facts are mined from the verified articles (poketype/dex_wiki.py), only ever quoted, never
     written by the model.
  4. Name ingredients: the half's own words (mention, title head nouns), then the model's near-
     meaning forms (synonyms, Latin/Greek roots) informed by the facts.
The model also proposes phrases a name could sound like and judges pun candidates (dex_puns).
Without a model (or offline) the result is empty and the generator uses its offline idea pools.
Model answers and Wikipedia responses are cached on disk.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from poketype.dex_wiki import Wiki
from poketype.dex_words import STOPWORDS, base_form, wordnet

CACHE = Path(__file__).resolve().parent.parent / "data" / "dex" / "llm_cache"
FILLER = set("""that this with from into than they their them will have been were what when which species
genus family order group class suborder type kind like other also some most more about lights small large
long several various tailed winged carved common""".split())
POKEMON_WORDS = re.compile(r"pok[eé]mon|pikachu|list of", re.I)

HALVES_SYSTEM = (
    "You read the description of a new fictional creature and identify the two ideas its name should combine:\n"
    "\"is\": what the creature IS (a real animal, plant, object, phenomenon or legend);\n"
    "\"defines\": what DEFINES it (its most striking trait, behavior, ability, element or theme).\n"
    "For each give \"mention\": the exact words of the description that name it (copied verbatim), or \"\" if "
    "the description never names it; and \"guesses\": up to 3 English Wikipedia article titles for it, most "
    "specific and most likely first. Never name a Pokemon or any other fictional character. Answer in JSON.")
HALVES_SHOTS = [
    ("It squirts a cloud of black ink to escape, then changes the color of its skin to vanish against the rocks.",
     json.dumps({"is": {"mention": "", "guesses": ["Cuttlefish", "Octopus", "Squid"]},
                 "defines": {"mention": "changes the color of its skin",
                             "guesses": ["Chromatophore", "Active camouflage", "Camouflage"]}})),
    ("This little mantis shrimp boxes anything that comes near its burrow.",
     json.dumps({"is": {"mention": "mantis shrimp", "guesses": ["Mantis shrimp", "Stomatopoda"]},
                 "defines": {"mention": "boxes", "guesses": ["Boxing", "Punch (combat)", "Cavitation"]}})),
]
WORDS_SYSTEM = (
    "You help name a new fictional creature by blending two ideas. For each idea you get facts. List name "
    "ingredients for each: up to 6 short single words, the idea's plain English word first, then near-meaning "
    "forms (synonyms, Latin or Greek roots, a word from a striking fact), the kind of words creature names are "
    "blended from (for example 'char' and 'lizard', 'chandelier' and 'lure', 'bear' and 'arctic'). No names of "
    "existing fictional characters. Answer in JSON: {\"is\": [...], \"defines\": [...]}.")
GENUS_SYSTEM = (
    "You write the category line of a Pokedex entry for a new fictional creature, the 'X' in 'the X Pokemon'. "
    "It is one or two plain English words naming what the creature is or its most striking trait. Given the "
    "description and its two ideas, give 3 options separated by semicolons, best first, nothing else.")
GENUS_SHOTS = [  # canon examples: Bulbasaur, Magikarp, Chandelure, Beartic
    ("Description: A strange seed was planted on its back at birth.\nIs: Bulb\nDefines: Seed", "Seed; Bulb; Sprout"),
    ("Description: It is virtually worthless in terms of both power and speed.\nIs: Carp\nDefines: Splashing",
     "Fish; Carp; Splash"),
    ("Description: Its spooky flames burn the spirits of those it lures in.\nIs: Chandelier\nDefines: Lure",
     "Luring; Chandelier; Ghost Light"),
    ("Description: It freezes its breath to make fangs and claws of ice.\nIs: Polar bear\nDefines: Freezing",
     "Freezing; Ice Bear; Frost"),
]
PHRASES_SYSTEM = (
    "Pokemon names are sometimes puns: they sound like a common word, idiom, catchphrase or title that fits the "
    "creature (a lazy ape 'Slaking' sounds like 'slacking'; a curious one 'Wynaut' like 'why not'; a guard rodent "
    "'Patrat' like 'pack rat'; a dangerous bear 'Bewear' like 'beware'). For the creature described, list 10 "
    "common English words or short phrases its name could sound like that fit its nature. Answer in JSON: "
    "{\"phrases\": [...]}.")
JUDGE_SYSTEM = (
    "You judge pun names for a new fictional creature. Each candidate is a made-up name and the word or phrase it "
    "sounds like. A good pun is easy to hear in the name AND the phrase says something true and fitting about this "
    "creature, so the name gets a knowing smile (a lazy ape 'Slaking' ~ 'slacking'; a guard rodent 'Patrat' ~ "
    "'pack rat'). A bad pun sounds like a phrase unrelated to the creature (a dragonfly 'Winkatcher' ~ 'wing "
    "chair'). Be strict: most candidates are bad. Pick up to 3 good ones, best first, or none. Answer in JSON: "
    "{\"best\": [candidate numbers]}.")


class Inspiration:
    def __init__(self, llm_path, embed, wiki=None, english=None, n_threads=12, n_gpu_layers=-1):
        """embed: callable list[str] -> unit vectors (the generator's sentence encoder);
        english: callable word -> bool (is it an ordinary English word)."""
        self.embed, self.wiki = embed, wiki or Wiki()
        self.english = english or (lambda w: True)
        self.llm = None
        if llm_path and Path(llm_path).exists():
            from llama_cpp import Llama
            self.llm = Llama(model_path=str(llm_path), n_ctx=4096, n_threads=n_threads,
                             n_gpu_layers=n_gpu_layers, verbose=False)
            self.model_id = Path(llm_path).name
        CACHE.mkdir(parents=True, exist_ok=True)

    # ---- model calls ------------------------------------------------------------------------
    def _chat(self, system, shots, user, max_tokens=60, as_json=False):
        msgs = [{"role": "system", "content": system}]
        for q, a in shots:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        key = hashlib.sha1(json.dumps([self.model_id, msgs, as_json]).encode()).hexdigest()
        path = CACHE / f"{key}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            kw = {"response_format": {"type": "json_object"}} if as_json else {}
            out = self.llm.create_chat_completion(messages=msgs, max_tokens=max_tokens, temperature=0, **kw)
            text = out["choices"][0]["message"]["content"].strip()
            path.write_text(text, encoding="utf-8")
        if not as_json:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _split(answer):
        items = [re.sub(r"\(.*?\)", "", x).strip(" .'\"*-\n") for x in re.split(r"[;\n]", answer)]
        return [x for x in items if x and len(x) < 60]

    # ---- one half ---------------------------------------------------------------------------
    def articles(self, titles, text_vec, max_articles=2, min_fit=0.05):
        """Verified articles for title guesses: [(page, guess, fit)] best first. The model's own
        order counts for more than the fit, which only removes clearly unrelated articles."""
        pages = []
        for rank, title in enumerate(titles):
            if not title or POKEMON_WORDS.search(title):
                continue
            page = None
            # the guess itself, else its head noun ("Electric mouse" -> "Mouse", not a Pokemon redirect)
            for attempt in dict.fromkeys([title, title.split()[-1].capitalize()]):
                p = self.wiki.page(attempt)
                if p and not POKEMON_WORDS.search(p["title"] + " " + p["description"]):
                    page, title = p, attempt
                    break
            if page is None or any(p["title"] == page["title"] for p, _, _ in pages):
                continue
            fit = float(self.embed([page["intro"][:1000]])[0] @ text_vec)
            if fit >= min_fit:
                pages.append((page, title, fit - 0.08 * rank))
        pages = sorted(pages, key=lambda t: -t[2])[:max_articles]
        # a second article only if it fits nearly as well as the first (dragonfly, not dragonfly + "Wing")
        return [x for x in pages if x is pages[0] or x[2] >= 0.85 * pages[0][2]]

    def _half(self, spec, text, text_vec):
        mention = (spec.get("mention") or "").strip()
        if mention and mention.lower() not in text.lower():
            mention = ""  # the model must quote the text; anything else is a guess, not a mention
        guesses = [g for g in spec.get("guesses", []) if isinstance(g, str)]
        # a mention is looked up first: "mantis shrimp" -> its article, "boxes" -> its guesses
        arts = self.articles(([mention] if mention else []) + guesses, text_vec)
        facts = [(s, p["title"]) for p, _, _ in arts for s, _ in self.wiki.facts(p, text_vec, self.embed, top=3)]
        own = []
        for p, guess, _ in arts:  # each title's head noun first: "Galapagos giant tortoise" -> tortoise
            for src in (p["title"], guess):
                ws = words_of(src)
                own += ws[-1:] + ws[:-1]
        own = sorted(dict.fromkeys(own), key=lambda w: not self.english(w))
        # then the mention's content words in base form ("punches prey" -> punch, prey)
        own += [w for w in content_words(mention) if w not in own]
        return {"mention": mention, "articles": [(p["title"], p["description"], round(f, 3)) for p, _, f in arts],
                "facts": facts, "own": own[:4]}

    # ---- everything -------------------------------------------------------------------------
    def __call__(self, text):
        empty = {"halves": {}, "basis": [], "facts": [], "basis_words": [], "concepts": [], "genus": []}
        if self.llm is None:
            return empty
        text_vec = self.embed([text])[0]
        spec = self._chat(HALVES_SYSTEM, HALVES_SHOTS, text, max_tokens=160, as_json=True)
        halves = {k: self._half(spec.get(k, {}) if isinstance(spec.get(k), dict) else {}, text, text_vec)
                  for k in ("is", "defines")}
        if not halves["is"]["articles"] and not halves["defines"]["articles"]:
            return empty
        prompt = f"Description: {text}\n" + "\n".join(
            f"{'Is' if k == 'is' else 'Defines'}: {', '.join(a[0] for a in h['articles']) or h['mention'] or '?'}"
            + "".join(f"\n- {s}" for s, _ in h["facts"]) for k, h in halves.items())
        forms = self._chat(WORDS_SYSTEM, [], prompt, max_tokens=160, as_json=True)
        for k, h in halves.items():
            more = [w.lower() for w in forms.get(k, []) if isinstance(w, str) and re.fullmatch(r"[A-Za-z]{3,14}", w)]
            h["words"] = list(dict.fromkeys(h["own"] + more))[:8]
        short = "\n".join(line for line in prompt.split("\n") if not line.startswith("- "))
        genus = [g.title() for g in self._split(self._chat(GENUS_SYSTEM, GENUS_SHOTS, short, max_tokens=30))
                 if re.fullmatch(r"[A-Za-z]+( [A-Za-z]+)?", g) and not POKEMON_WORDS.search(g)]
        a, b = halves["is"], halves["defines"]
        return {"halves": halves, "genus": list(dict.fromkeys(genus))[:3],
                "basis": a["articles"] or b["articles"], "facts": a["facts"] + b["facts"],
                "basis_words": a["words"] or b["words"], "concepts": b["words"] if a["words"] else [],
                "prompt": short}

    def pun_phrases(self, insp):
        """Words and phrases a name for this creature could sound like (pun targets)."""
        if self.llm is None or not insp.get("prompt"):
            return []
        out = self._chat(PHRASES_SYSTEM, [], insp["prompt"], max_tokens=120, as_json=True)
        return [p.lower() for p in out.get("phrases", []) if isinstance(p, str) and 3 <= len(p) <= 30]

    def judge_puns(self, insp, candidates):
        """candidates: [(name, phrase)]; returns the indices the model picks, best first."""
        if self.llm is None or not candidates:
            return []
        lines = "\n".join(f"{i + 1}. {n} (sounds like \"{p}\")" for i, (n, p) in enumerate(candidates))
        out = self._chat(JUDGE_SYSTEM, [], insp["prompt"] + "\nCandidates:\n" + lines, max_tokens=40, as_json=True)
        picks = [int(x) - 1 for x in out.get("best", []) if str(x).isdigit()]
        return [i for i in dict.fromkeys(picks) if 0 <= i < len(candidates)]


def words_of(s):
    """Content words of a title or phrase, accents removed (Galapagos)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return [w for w in re.findall(r"[a-z]+", s) if len(w) >= 3 and w not in FILLER]


NUMBERS = set("one two three four five six seven eight nine ten hundred thousand".split())
FUNCTION = set("the and but for nor yet out off can make made nothing something anything all any".split())


def content_words(phrase):
    """Words of a mention that can carry a name, in base form: no function words or numbers,
    known to WordNet ("snatching gnats out of the air" -> snatch, gnat, air)."""
    wn = wordnet()
    out = []
    for w in words_of(phrase):
        if w in STOPWORDS or w in NUMBERS or w in FUNCTION or not wn.synsets(w):
            continue
        b = base_form(w)
        if b not in out:
            out.append(b)
    return out
