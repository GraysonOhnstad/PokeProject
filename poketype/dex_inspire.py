"""Name inspiration: what is this Pokemon based on, what is interesting about that, and which
words should its name be built from.

    insp = Inspiration(llm_path, embed)          # llm_path: a GGUF instruction model, or None
    result = insp(text)
    result["basis"]     verified Wikipedia articles the text most likely describes, best first
    result["facts"]     [(sentence, article title)]: interesting sentences from those articles
    result["basis_words"], result["concepts"]: name ingredients

Steps:
  1. A local instruction model (llama.cpp) proposes candidate bases as Wikipedia titles.
  2. Wikipedia verifies them: the article must exist, not be a disambiguation or Pokemon page,
     and its introduction must match the text's meaning; candidates are re-ranked by that match.
  3. Facts are mined from the verified articles (poketype/dex_wiki.py); they are only ever
     quoted from Wikipedia, never written by the model.
  4. The model proposes name ingredients that capture the basis and its facts (short English
     words or Latin/Greek roots); terms mined from the facts are added as a fallback.
Without a model (or offline), the result is empty and the generator uses its offline idea pools.
Model answers are cached on disk by prompt.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from poketype.dex_wiki import Wiki

CACHE = Path(__file__).resolve().parent.parent / "data" / "dex" / "llm_cache"
FILLER = set("""that this with from into than they their them will have been were what when which species
genus family order group class suborder type kind like other also some most more about lights small large
long several various tailed winged carved common""".split())
POKEMON_WORDS = re.compile(r"pok[eé]mon|pikachu|list of", re.I)

BASIS_SYSTEM = (
    "A fictional creature is described. Name the specific real-world animals, plants, objects, phenomena or "
    "legends it is most likely based on, as English Wikipedia article titles separated by semicolons, most likely "
    "first. Prefer the most specific match (a species or group, not a broad class). Never name a Pokemon or "
    "any other fictional character. Answer with titles only.")
BASIS_SHOTS = [
    ("It squirts a cloud of black ink to escape, then changes the color of its skin to vanish against the rocks.",
     "Cuttlefish; Octopus; Squid"),
    ("Its long ears hear predators a mile away. It thumps its huge hind feet to warn the others.",
     "Rabbit; Hare; Jackrabbit"),
]
CONCEPT_SYSTEM = (
    "You help name a new fictional creature. Given its description, what it is based on, and facts about that, "
    "list 8 short name ingredients: single English words or Latin/Greek roots that capture the creature and the "
    "most striking facts, the kind of words creature names are blended from (for example 'char' and 'lizard', "
    "'chandelier' and 'lure', 'bear' and 'arctic'). Do not use names of existing fictional characters. Answer "
    "with the words separated by semicolons, nothing else.")


GENUS_SYSTEM = (
    "You write the category line of a Pokedex entry for a new fictional creature, the 'X' in 'the X Pokemon'. "
    "It is one or two plain English words naming what the creature is or its most striking trait. Given the "
    "description, its real-world basis and facts, give 3 options separated by semicolons, best first, nothing else.")
GENUS_SHOTS = [  # canon examples: Bulbasaur, Magikarp, Chandelure, Beartic, Gengar
    ("Description: A strange seed was planted on its back at birth.\nBased on: Bulb; Dinosaur", "Seed; Bulb; Sprout"),
    ("Description: It is virtually worthless in terms of both power and speed.\nBased on: Carp", "Fish; Carp; Splash"),
    ("Description: Its spooky flames burn the spirits of those it lures in.\nBased on: Chandelier",
     "Luring; Chandelier; Ghost Light"),
    ("Description: It freezes its breath to make fangs and claws of ice.\nBased on: Polar bear",
     "Freezing; Ice Bear; Frost"),
]


class Inspiration:
    def __init__(self, llm_path, embed, wiki=None, english=None, n_threads=12, n_gpu_layers=-1):
        """embed: callable list[str] -> unit vectors (the generator's sentence encoder);
        english: callable word -> bool (is it an ordinary English word)."""
        self.embed, self.wiki = embed, wiki or Wiki()
        self.english = english or (lambda w: True)
        self.llm = None
        if llm_path and Path(llm_path).exists():
            from llama_cpp import Llama
            self.llm = Llama(model_path=str(llm_path), n_ctx=2048, n_threads=n_threads,
                             n_gpu_layers=n_gpu_layers, verbose=False)
            self.model_id = Path(llm_path).name
        CACHE.mkdir(parents=True, exist_ok=True)

    def _chat(self, system, shots, user, max_tokens=60):
        msgs = [{"role": "system", "content": system}]
        for q, a in shots:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        key = hashlib.sha1(json.dumps([self.model_id, msgs]).encode()).hexdigest()
        path = CACHE / f"{key}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        out = self.llm.create_chat_completion(messages=msgs, max_tokens=max_tokens, temperature=0)
        text = out["choices"][0]["message"]["content"].strip()
        path.write_text(text, encoding="utf-8")
        return text

    @staticmethod
    def _split(answer):
        items = [re.sub(r"\(.*?\)", "", x).strip(" .'\"*-\n") for x in re.split(r"[;\n]", answer)]
        return [x for x in items if x and len(x) < 60]

    def bases(self, text, text_vec, max_bases=2, min_fit=0.12):
        """Verified bases: [(page, proposed title, fit)] best first. The threshold only removes
        articles clearly unrelated to the text; the model's own order counts for more."""
        proposals = self._split(self._chat(BASIS_SYSTEM, BASIS_SHOTS, text))[:5]
        pages = []
        for rank, title in enumerate(proposals):
            if POKEMON_WORDS.search(title):
                continue
            page = None
            # the proposal itself, else its head noun ("Electric mouse" -> "Mouse", which isn't a redirect
            # to a Pokemon page)
            for attempt in dict.fromkeys([title, title.split()[-1].capitalize()]):
                p = self.wiki.page(attempt)
                if p and not POKEMON_WORDS.search(p["title"] + " " + p["description"]):
                    page, title = p, attempt
                    break
            if page is None:
                continue
            if any(p["title"] == page["title"] for p, _, _ in pages):
                continue
            fit = float(self.embed([page["intro"][:1000]])[0] @ text_vec)
            if fit >= min_fit:
                pages.append((page, title, fit - 0.08 * rank))   # earlier proposals are favored
        return sorted(pages, key=lambda t: -t[2])[:max_bases]

    def __call__(self, text):
        empty = {"basis": [], "facts": [], "basis_words": [], "concepts": [], "genus": []}
        if self.llm is None:
            return empty
        text_vec = self.embed([text])[0]
        bases = self.bases(text, text_vec)
        if not bases:
            return empty
        facts = []
        for page, _, _ in bases:
            facts += [(s, page["title"]) for s, _ in self.wiki.facts(page, text_vec, self.embed, top=3)]
        # the basis's own words: titles first (proposed, then resolved), then the article's short
        # description (so a species article still yields "mantis", "shrimp"); English before Latin

        def words_of(s):
            s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()  # Galápagos
            return [w for w in re.findall(r"[a-z]+", s) if len(w) >= 4 and w not in FILLER]
        titles = []
        for page, proposed, _ in bases:  # each title's head noun first: "Galapagos giant tortoise" -> tortoise
            for src in (page["title"], proposed):
                ws = words_of(src)
                titles += ws[-1:] + ws[:-1]
        described = [w for page, _, _ in bases for w in words_of(page["description"])]
        titles = sorted(dict.fromkeys(titles), key=lambda w: not self.english(w))
        # the creature itself (title words); description words ("raptorial", "winged") only as a fallback
        basis_words = titles[:6] or list(dict.fromkeys(described))[:4]
        prompt = (f"Description: {text}\nBased on: {', '.join(p['title'] for p, _, _ in bases)}\nFacts:\n"
                  + "\n".join(f"- {s}" for s, _ in facts))
        concepts = [c.lower() for c in self._split(self._chat(CONCEPT_SYSTEM, [], prompt, max_tokens=80))
                    if re.fullmatch(r"[A-Za-z]{3,14}", c)]
        genus = [g.title() for g in self._split(self._chat(GENUS_SYSTEM, GENUS_SHOTS, prompt, max_tokens=30))
                 if re.fullmatch(r"[A-Za-z]+( [A-Za-z]+)?", g) and not POKEMON_WORDS.search(g)]
        return {"basis": [(p["title"], p["description"], round(f, 3)) for p, _, f in bases], "facts": facts,
                "basis_words": basis_words, "concepts": list(dict.fromkeys(concepts)),
                "genus": list(dict.fromkeys(genus))[:3]}


def fit_scores(words, text_vec, embed):
    """How well each word fits the text's meaning (cosine)."""
    if not words:
        return np.zeros(0)
    return embed(list(words)) @ text_vec
