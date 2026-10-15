"""Flavor-text cleaning and name masking, shared by the dataset builder and the app.

Masking replaces every Pokemon name with a token for its role in the text:

    <self>              the species the entry describes
    <pre-evolution>     an earlier member of its line
    <evolution>         a later member of its line
    <sibling evolution> another branch of its line (e.g. Jolteon in a Vaporeon entry)
    <water fish pokémon> any other species, by its types and genus
                        (Magikarp: <water fish pokémon>; Kyogre: <water sea basin pokémon>)

A preceding "a"/"an" is kept and corrected for the token. Names of other species match
only when capitalized, so ordinary words that happen to be species names stay as they are.
"""

import re

SELF = "<self>"


def clean_text(t):
    t = t.replace("\xad\n", "").replace("\xad", "")
    t = re.sub(r"[\n\x0c]+", " ", t)
    t = t.replace("POKéMON", "Pokémon").replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\b[A-Z][A-Z'.]{2,}\b", lambda m: m.group(0).title(), t)  # Gen 1-3 ALL-CAPS names
    return re.sub(r"\s+", " ", t).strip()


def species_token(types, genus):
    """<types genus>, e.g. ["water"], "Fish Pokémon" -> <water fish pokémon>."""
    return "<" + "/".join(types) + " " + genus.lower() + ">"


class Masker:
    """names: {species_id: name}; parent: {species_id: evolves_from or None};
    describe: callable (species_id, version_group_id or None) -> token for another species."""

    def __init__(self, names, parent, describe):
        self.parent, self.describe = parent, describe
        self.by_name = {n.replace("’", "'").lower(): sid for sid, n in names.items()}
        alts = sorted(map(re.escape, self.by_name), key=len, reverse=True)
        self.pattern = re.compile(r"(?:\b(a|an|A|An) )?(?<!\w)(" + "|".join(alts) + r")(?!\w)",
                                  re.IGNORECASE)

    def ancestors(self, sid):
        out = []
        while self.parent.get(sid):
            sid = self.parent[sid]
            out.append(sid)
        return out

    def role(self, other, sid):
        if other == sid:
            return SELF
        if other in self.ancestors(sid):
            return "<pre-evolution>"
        if sid in self.ancestors(other):
            return "<evolution>"
        root = lambda s: (self.ancestors(s) or [s])[-1]  # noqa: E731
        if root(other) == root(sid):
            return "<sibling evolution>"
        return None

    def __call__(self, text, sid=None, version_group_id=None):
        """Mask `text` describing species `sid` (None for new, user-written text)."""
        def sub(m):
            article, name = m.group(1), m.group(2)
            other = self.by_name[name.lower()]
            token = self.role(other, sid) if sid is not None else None
            if token is None:
                if not name[0].isupper():
                    return m.group(0)  # lowercase: an ordinary word, not a name
                token = self.describe(other, version_group_id)
            if article:
                a = "an" if token[1] in "aeiou" else "a"
                return (a.capitalize() if article[0].isupper() else a) + " " + token
            return token
        return self.pattern.sub(sub, text)
