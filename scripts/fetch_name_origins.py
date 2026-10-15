"""Fetch the "Name origin" section of every species' Bulbapedia page.

Uses the MediaWiki API, 50 pages per request with a pause between requests (~21 requests).
Text is from Bulbapedia (https://bulbapedia.bulbagarden.net), CC BY-NC-SA 2.5; credit it
wherever it is shown or used.

    py scripts/fetch_name_origins.py

Writes data/dex/name_origins.json: {species_id: {"name", "title", "wikitext", "text", "english",
"components"}}: "english" is the paragraph(s) about the English name and "components" the source
words Bulbapedia marks in it (italics and Wikipedia links), e.g. Chandelure -> chandelier, lure.
"""

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dex" / "name_origins.json"
API = "https://bulbapedia.bulbagarden.net/w/api.php"
UA = "pokeproject-dex/0.1 (personal non-commercial project; name-origin research)"
PAUSE = 2.0


def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def section(wikitext, heading="Name origin"):
    m = re.search(r"^(=+)\s*" + heading + r"\s*\1\s*$", wikitext, re.M)
    if not m:
        return None
    rest = wikitext[m.end():]
    end = re.search(r"^={2," + str(len(m.group(1))) + r"}[^=]", rest, re.M)
    return rest[:end.start() if end else len(rest)].strip()


def plain(wt):
    """Wikitext -> readable text: {{wp|x|y}} -> y or x, [[a|b]] -> b, drops other templates/markup."""
    t = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", wt, flags=re.S)
    t = re.sub(r"\{\{(?:wp|wiktionary|wt|tt|j|DL)\|([^{}|]*)(?:\|([^{}|]*))?[^{}]*\}\}",
               lambda m: m.group(2) or m.group(1), t)
    for _ in range(3):
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def english_part(wt, name):
    """Paragraphs of the section that explain the English name (they start with it)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", wt) if p.strip()]
    keep = [p for p in paras if plain(p).lower().startswith(name.lower())]
    return "\n\n".join(keep or paras[:1])


def components(wt_english, name):
    """Source words marked in the English-name paragraphs: ''italics'' and {{wp|...}} links."""
    t = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", wt_english, flags=re.S)
    links = []

    def link(m):
        parts = m.group(1).split("|")
        links.append(parts[-1])
        return parts[-1]
    t = re.sub(r"\{\{wp\|([^{}]+)\}\}", link, t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    found = re.findall(r"(?<!')''(?!')([^'\n]+?)(?<!')''(?!')", t) + links
    out = []
    for w in (html.unescape(x).strip().strip("-‑").lower() for x in found):
        if (w and w != name.lower() and w not in out and len(w.split()) <= 3
                and not re.search(r"[()\[\].,;:\"]", w)):
            out.append(w)
    return out


def main():
    species = pd.read_csv(ROOT / "data" / "dex" / "species.csv")
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [(int(s), n) for s, n in zip(species.species_id, species.name) if str(s) not in out]
    print(f"{len(todo)} pages to fetch", flush=True)
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        titles = {f"{n} (Pokémon)": s for s, n in chunk}
        data = get({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                    "titles": "|".join(titles), "redirects": 1, "format": "json", "formatversion": 2})
        q = data["query"]
        alias = {r["to"]: r["from"] for r in q.get("redirects", [])}
        alias.update({n["to"]: n["from"] for n in q.get("normalized", [])})
        for page in q["pages"]:
            title = page["title"]
            src = alias.get(title, title)
            src = alias.get(src, src)
            sid = titles.get(src) or titles.get(title)
            if sid is None or "revisions" not in page:
                continue
            wt = page["revisions"][0]["slots"]["main"]["content"]
            sec = section(wt)
            out[str(sid)] = {"name": dict(chunk)[sid], "title": title, "wikitext": sec}
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  {min(i + 50, len(todo))}/{len(todo)}", flush=True)
        time.sleep(PAUSE)
    for rec in out.values():  # derived fields, recomputed every run
        sec = rec["wikitext"]
        rec["text"] = plain(sec) if sec else None
        rec["english"] = plain(english_part(sec, rec["name"])) if sec else None
        rec["components"] = components(english_part(sec, rec["name"]), rec["name"]) if sec else []
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    missing = [n for s, n in zip(species.species_id, species.name) if not (out.get(str(s)) or {}).get("text")]
    print(f"done: {len(out)} pages, {len(missing)} without a name-origin section: {missing[:15]}")


if __name__ == "__main__":
    sys.exit(main())
