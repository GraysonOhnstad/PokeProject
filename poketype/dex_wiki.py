"""Wikipedia lookups for name inspiration: verify a proposed basis and mine its article for facts.

Every response is cached on disk (data/dex/wiki_cache/), so a basis is fetched once; with no
network the generator falls back to its offline idea pools. Article text is from Wikipedia
(CC BY-SA 4.0); credit it wherever a fact is shown.

    wiki = Wiki()
    page = wiki.page("Dragonfly")          # {"title", "description", "intro", "text"} or None
    wiki.facts(page, text_vec, embed)      # [(sentence, score)], most interesting and relevant first
"""

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "pokeproject-dex/0.1 (personal non-commercial project; Pokedex generator)"
CACHE = Path(__file__).resolve().parent.parent / "data" / "dex" / "wiki_cache"

# Sentences worth turning into a Pokemon: extremes, numbers, deep time, odd abilities, folklore.
INTERESTING = re.compile(
    r"\b(largest|smallest|fastest|strongest|longest|oldest|only|unique|unlike|extinct|prehistoric|fossil|"
    r"ancestor|ancestors|giant|million years|carboniferous|jurassic|cretaceous|permian|legend|myth|folklore|"
    r"believed|superstition|named after|venom|poison|camouflage|mimic|bioluminescent|glow|record|"
    r"can see|can detect|capable of|able to|up to|times|than any|world's|most)\b", re.I)


class Wiki:
    def __init__(self, pause=0.5, timeout=20):
        self.pause, self.timeout = pause, timeout
        CACHE.mkdir(parents=True, exist_ok=True)

    def _get(self, **params):
        params.update(format="json", formatversion=2)
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        url = API + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}),
                                        timeout=self.timeout) as r:
                data = json.loads(r.read())
        except Exception:
            return None
        path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(self.pause)
        return data

    def page(self, title):
        """The article for `title` (redirects followed), or None if missing or a disambiguation page."""
        data = self._get(action="query", prop="extracts|description|pageprops", explaintext=1,
                         exsectionformat="plain", titles=title, redirects=1)
        if not data:
            return None
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or "disambiguation" in pages[0].get("pageprops", {}):
            return None
        p = pages[0]
        text = p.get("extract") or ""
        text = text.split("\nSee also\n")[0].split("\nReferences\n")[0]
        intro = text.split("\n\n")[0]
        return {"title": p["title"], "description": p.get("description", ""), "intro": intro,
                "text": text[:40000]}

    @staticmethod
    def sentences(text):
        text = re.sub(r"\s+", " ", text)
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if 40 <= len(s) <= 300]

    def facts(self, page, text_vec, embed, top=5):
        """The article's most fact-like sentences, ranked by interestingness and fit with the
        Pokemon's description. embed: callable list[str] -> unit vectors."""
        sents = [s for s in self.sentences(page["text"]) if INTERESTING.search(s)]  # embed fact-like ones only
        if not sents:
            return []
        vecs = embed(sents)
        fit = vecs @ text_vec
        hits = [len(INTERESTING.findall(s)) + bool(re.search(r"\d", s)) for s in sents]
        score = [f + 0.15 * min(h, 3) for f, h in zip(fit, hits)]
        order = sorted(range(len(sents)), key=lambda i: -score[i])
        return [(sents[i], float(score[i])) for i in order[:top] if hits[i] > 0]


MONTHS = set("january february march april may june july august september october november december".split())


def fact_terms(sentences, nameworthy, real_words):
    """Concept candidates from fact sentences: proper and scientific names inside a sentence
    (Meganeura, Carboniferous) and name-worthy common nouns. Returns [(term, kind)]."""
    out, seen = [], set()
    for s in sentences:
        words = re.findall(r"[A-Za-z][a-z]+", s)
        for k, w in enumerate(words):
            low = w.lower()
            if low in seen or len(low) < 4 or low in MONTHS:
                continue
            if w[0].isupper() and k > 0 and low not in real_words:
                kind = "proper"         # mid-sentence capitalized and not an ordinary word
            elif low in real_words and nameworthy(low):
                kind = "common"
            else:
                continue
            seen.add(low)
            out.append((low, kind))
    return out
