"""Text -> full Pokedex entry: every Dex model chained, with sample-and-pick.

    gen = DexGenerator("runs/dex/gen5_heldout/dex.pt", "runs/dex_textgen")
    entry = gen.generate("It drifts through old mansions, lighting candles with its ghostly flame.", era=5)

Order (each step conditioned on what came before): text -> types, body (shape, color, egg
groups, height, weight) -> genus -> stat line (+ evolution stage, line length, rarity) ->
abilities -> learnset -> name. K complete candidates are sampled and one is picked (`score`).
Every move and ability comes from the existing inventory; names never repeat a real one.
"""

import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from poketype.dex import DEX, STATS, natural
from poketype.dex_learnset import MoveSetModel, bias_for, cond_vector, sample_learnsets
from poketype.dex_models import AbilityModel, Heads
from poketype.dex_inspire import Inspiration
from poketype.dex_names import NameScorer, blocked, rank
from poketype.dex_wiki import fact_terms
from poketype.dex_stats import SCALE, StatModel
from poketype.dex_text import Masker, clean_text, species_token
from poketype.labels import TYPES

MAX_NAME = 12
DEFAULT_LLM = DEX.parent / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"  # name inspiration
PAIRS = 3  # idea pairs tried per candidate
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


TYPE_WORDS = set(TYPES) | {"pokémon", "pokemon", "self", "evolution"}
NAME_OK = re.compile(r"^[A-Z][a-zé'\-]*[a-zé]$")  # one capitalized word, like Farfetch'd or Porygon-Z


def bracketed(text):
    return re.sub(r"<([^<>]+)>", r"[\1]", text)


def set_logprobs(p, eps=1e-4):
    """Every 1- and 2-element set under independent sigmoid probabilities p (n,):
    returns (sets, normalized log-probabilities)."""
    p = np.clip(p, eps, 1 - eps)
    lo = np.log(p) - np.log1p(-p)
    n = len(p)
    sets = [(i,) for i in range(n)] + [(i, j) for i in range(n) for j in range(i + 1, n)]
    logit = np.array([lo[list(s)].sum() for s in sets])
    return sets, logit - np.logaddexp.reduce(logit)


def one_hot(i, n):
    v = np.zeros(n, np.float32)
    v[i] = 1
    return v


class DexGenerator:
    def __init__(self, bundle, textgen_dir, threads=8, llm=DEFAULT_LLM, filter_names=True):
        """llm: GGUF instruction model for name inspiration (None or missing file: offline idea pools).
        filter_names: reject names hiding sexual or profane words (slurs and hate terms are always
        rejected; see poketype/dex_names.py)."""
        self.filter_names = filter_names
        torch.set_num_threads(threads)
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
        b = torch.load(bundle, weights_only=False)
        self.b, self.v = b, b["vocabs"]
        self.enc_tok = AutoTokenizer.from_pretrained(b["encoder"])
        self.encoder = AutoModel.from_pretrained(b["encoder"]).eval()
        h = b["heads"]
        self.heads = {}
        for field, state in h["states"].items():
            m = Heads(h["d_in"], h["sizes"])
            m.load_state_dict(state)
            self.heads[field] = m.eval()
        self.stats = StatModel(b["stats"]["d_in"], aux=True)
        self.stats.load_state_dict(b["stats"]["state"])
        self.abil = AbilityModel(b["abilities"]["d_in"], b["abilities"]["n"])
        self.abil.load_state_dict(b["abilities"]["state"])
        ls = b["learnset"]
        self.moves = ls["moves"]
        self.move_gen = np.array(ls["move_gen"])
        self.learnset = MoveSetModel(ls["d_cond"], len(self.moves))
        self.learnset.load_state_dict(ls["state"])
        self.levels = {(int(k.split(":")[0]), int(k.split(":")[1])): np.array(v) for k, v in ls["levels"].items()}
        for m in (self.stats, self.abil, self.learnset):
            m.eval()
        self.t5_tok = AutoTokenizer.from_pretrained(textgen_dir)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(textgen_dir).eval()

        species = pd.read_csv(DEX / "species.csv").set_index("species_id")
        vocab = json.loads((DEX / "vocab.json").read_text(encoding="utf-8"))
        self.move_names = {int(k): v for k, v in vocab["moves"].items()}
        self.ability_names = {int(k): v for k, v in vocab["abilities"].items()}
        # signature moves and abilities (held by <= 2 families) are never given to a generated Pokemon
        sig_a, sig_m = set(vocab.get("signature_abilities", [])), set(vocab.get("signature_moves", []))
        self.signature_ability_ix = [i for i, a in enumerate(self.v["abilities"]) if int(a) in sig_a]
        self.signature_move_mask = np.array([int(m) in sig_m for m in self.b["learnset"]["moves"]])
        self.existing = {n.lower() for n in species["name"]}
        latest = {}
        for line in open(DEX / "entries.jsonl", encoding="utf-8"):
            e = json.loads(line)
            latest[e["species_id"]] = e["types"]
        genus = species.genus.to_dict()
        self.masker = Masker(species["name"].to_dict(),
                             {s: (None if pd.isna(p) else int(p)) for s, p in species.evolves_from_species_id.items()},
                             lambda s, g: species_token(latest[s], genus[s]))
        self._naming_data(species, bundle)
        self.inspiration = None
        if llm and Path(llm).exists():
            self.inspiration = Inspiration(llm, self.embed_many, english=lambda w: w in self.real_words,
                                           n_threads=threads)

    def _naming_data(self, species, bundle):
        """Name-concept vocabulary (canon source words of the species this config trained on, with
        embeddings), word rarity for picking notable words from a text, and a real-word list."""
        allowed = species.index if self.b["config"] == "all" else species.index[species.fold >= 0]
        origins = json.loads((DEX / "name_origins.json").read_text(encoding="utf-8"))
        vocab = sorted({c for s in allowed for c in origins[str(s)]["components"][:3]
                        if re.fullmatch(r"[a-z][a-z ]+", c)})
        cache = Path(bundle).parent / "concepts.npz"
        if cache.exists() and list(np.load(cache)["words"]) == vocab:
            vecs = np.load(cache)["vecs"]
        else:
            vecs = np.stack([self.embed(w) for w in vocab])
            np.savez(cache, words=np.array(vocab), vecs=vecs)
        self.concepts, self.concept_vecs = vocab, vecs
        texts = pd.read_csv(DEX / "texts.csv")
        texts = texts[texts.species_id.isin(allowed)]
        df = {}
        for t in texts.text_masked:
            for w in set(re.findall(r"[a-z]+", t.lower())):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log(len(texts) / c) for w, c in df.items()}
        self.real_words = {w for w in self.enc_tok.get_vocab() if w.isalpha() and len(w) > 2}
        # association vocabulary: the encoder's whole English words, embedded once and cached
        words = sorted(w for w in self.real_words if len(w) >= 4 and w not in STOPWORDS)
        cache = Path(bundle).parent / "words.npz"
        if cache.exists() and len(np.load(cache)["words"]) == len(words):
            vecs = np.load(cache)["vecs"].astype(np.float32)
        else:
            vecs = self.embed_many(words)
            np.savez(cache, words=np.array(words), vecs=vecs.astype(np.float16))
        self.words, self.word_vecs = words, vecs
        self.name_scorer = NameScorer(species["name"][allowed], words)

    # ---- text -------------------------------------------------------------------------------
    @torch.no_grad()
    def embed_many(self, texts, batch=256):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.enc_tok(texts[i:i + batch], padding=True, truncation=True, max_length=128,
                               return_tensors="pt")
            h = self.encoder(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed(self, text):
        enc = self.enc_tok([text], truncation=True, max_length=128, return_tensors="pt")
        h = self.encoder(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy()[0]

    @torch.no_grad()
    def t5(self, sources, n=1, max_new=24, seed=0):
        """Sampled T5 outputs: n per source."""
        torch.manual_seed(seed)
        enc = self.t5_tok(sources, padding=True, truncation=True, max_length=160, return_tensors="pt")
        g = self.t5_model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n, do_sample=True, top_p=0.9)
        out = self.t5_tok.batch_decode(g, skip_special_tokens=True)
        return [out[i * n:(i + 1) * n] for i in range(len(sources))]

    # ---- generation -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, text, era=9, k=8, seed=0):
        """One entry for `text` (era = generation 1-9); also returns all candidates."""
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        masked = self.masker(clean_text(text))
        emb = self.embed(natural(masked))
        x = np.concatenate([emb, one_hot(era - 1, 9)])[None].astype(np.float32)
        xt = torch.tensor(x)
        h = self.b["heads"]

        def head(field, key):
            return self.heads[field](xt)[key][0]
        p_types = torch.sigmoid(head("types", "types")).numpy()
        p_egg = torch.sigmoid(head("egg", "egg")).numpy()
        p_shape = head("shape", "shape").softmax(0).numpy()
        p_color = head("color", "color").softmax(0).numpy()
        reg_h = head("height", "reg").numpy() * h["reg_sd"] + h["reg_mu"]
        reg_w = head("weight", "reg").numpy() * h["reg_sd"] + h["reg_mu"]
        # 10-NN blend for types and egg groups (as validated: MLP + kNN)
        knn = self.b["knn"]
        sim = knn["emb"] @ emb
        top = np.argpartition(-sim, 9)[:10]
        w = np.clip(sim[top], 1e-6, None)
        knn_types = (w[:, None] * knn["types"][top]).sum(0) / w.sum()
        p_types = (p_types + knn_types) / 2
        p_egg = (p_egg + (w[:, None] * knn["egg"][top]).sum(0) / w.sum()) / 2

        type_sets, type_lp = set_logprobs(p_types)
        _, knn_lp = set_logprobs(knn_types)
        egg_sets, egg_lp = set_logprobs(p_egg)
        if "no-eggs" in self.v["egg"]:  # "No Eggs" (Undiscovered) is never paired with another group
            ne = self.v["egg"].index("no-eggs")
            egg_lp = np.where([len(s) == 2 and ne in s for s in egg_sets], -np.inf, egg_lp)
            egg_lp = egg_lp - np.logaddexp.reduce(egg_lp)
        cands = []
        for c in range(k):
            ti = rng.choice(len(type_sets), p=np.exp(type_lp))
            types = [TYPES[i] for i in type_sets[ti]]
            shape = int(rng.choice(len(p_shape), p=p_shape))
            color = int(rng.choice(len(p_color), p=p_color))
            eggs = list(egg_sets[rng.choice(len(egg_sets), p=np.exp(egg_lp))])
            z = rng.multivariate_normal(np.zeros(2), h["size_resid_cov"])
            log_h, log_w = reg_h[0] + z[0], reg_w[1] + z[1]
            cands.append({"types": types, "shape": shape, "color": color, "eggs": eggs,
                          "log_h": float(log_h), "log_w": float(log_w),
                          "score": float(knn_lp[ti] + np.log(p_shape[shape]) + np.log(p_color[color]))})
        self._stats(cands, x, rng)
        self._make_consistent(cands)
        self._abilities(cands, x, era, rng)
        self._learnsets(cands, emb, era, rng)
        insp = self.inspiration(natural(masked)) if self.inspiration else None
        self._genus_and_name(cands, masked, emb, rng, seed, insp)
        best = max(cands, key=lambda c: c["score"] + (0 if c["name"] else -5))
        return self._present(best, text, era), [self._present(c, text, era) for c in cands]

    def _upstream(self, c):
        nt, ns, nc, ne = 18, len(self.v["shape"]), len(self.v["color"]), len(self.v["egg"])
        t = np.zeros(nt, np.float32)
        t[[TYPES.index(x) for x in c["types"]]] = 1
        e = np.zeros(ne, np.float32)
        e[c["eggs"]] = 1
        return t, one_hot(c["shape"], ns), one_hot(c["color"], nc), e

    def _stats(self, cands, x, rng):
        s = self.b["stats"]
        U = []
        for c in cands:
            t, sh, co, e = self._upstream(c)
            hw = (np.array([c["log_h"], c["log_w"]]) - s["mu_hw"]) / s["sd_hw"]
            U.append(np.concatenate([x[0], t, sh, co, e, hw]))
        X = torch.tensor(np.stack(U).astype(np.float32))
        m = self.stats
        hid = m.trunk(X)
        logit, mu, log_sd = m.mixture(hid)
        comp = torch.multinomial(logit.softmax(1), 1)[:, 0]
        bst = (mu.gather(1, comp[:, None])[:, 0] + log_sd.exp().gather(1, comp[:, None])[:, 0] * torch.randn(len(X)))
        bst = bst.clamp(1.8, 7.8)
        shares = torch.distributions.Dirichlet(m.concentration(hid, bst)).sample()
        stats = (shares * bst[:, None] * SCALE).round().clamp(1, 255).int().numpy()
        aux = {k: m.aux_heads[k](hid) for k in ("stage", "line_length", "is_final", "rarity")}
        for i, c in enumerate(cands):
            c["stats"] = dict(zip(STATS, stats[i].tolist()))
            line = int(rng.choice(3, p=aux["line_length"][i].softmax(0).numpy())) + 1
            ps = aux["stage"][i, :line].softmax(0).numpy()
            c["stage"], c["line_length"] = int(rng.choice(line, p=ps)) + 1, line
            c["rarity"] = int(rng.choice(3, p=aux["rarity"][i].softmax(0).numpy()))

    def _make_consistent(self, cands):
        """Rules from the games that independently sampled fields can break: No Eggs with a
        legendary-level total means legendary; legendaries and mythicals are single-stage."""
        ne = self.v["egg"].index("no-eggs") if "no-eggs" in self.v["egg"] else None
        for c in cands:
            bst = sum(c["stats"].values())
            if ne is not None and c["eggs"] == [ne] and bst >= 570 and c["rarity"] == 0:
                c["rarity"] = 1
            if c["rarity"] > 0:
                c["stage"] = c["line_length"] = 1

    def _abilities(self, cands, x, era, rng):
        a = self.b["abilities"]
        X = []
        for c in cands:
            t, sh, co, e = self._upstream(c)
            st = np.array([c["stats"][k] for k in STATS], np.float32)
            cont = (np.array([c["log_h"], c["log_w"], st.sum()]) - a["mu_c"]) / a["sd_c"]
            X.append(np.concatenate([x[0], t, sh, co, e, st / st.sum(), cont]))
        out = self.abil(torch.tensor(np.stack(X).astype(np.float32)))
        n = a["n"]
        ids = self.v["abilities"]
        for i, c in enumerate(cands):
            c["abilities"] = {}
            if era < 3:
                continue
            used = set()
            for slot, allow_none in (("primary", False), ("secondary", True), ("hidden", True)):
                if slot == "hidden" and era < 5:
                    continue
                p = out[slot][i].softmax(0).numpy().copy()
                p[list(used)] = 0
                p[self.signature_ability_ix] = 0
                if not allow_none and len(p) > n:
                    p[n:] = 0
                j = int(rng.choice(len(p), p=p / p.sum()))
                if j < n:
                    used.add(j)
                    c["abilities"][slot] = int(ids[j])

    def _learnsets(self, cands, emb, era, rng):
        ls = self.b["learnset"]
        cond = np.stack([cond_vector(c["types"], [c["stats"][k] for k in STATS], era, c["stage"],
                                     c["line_length"], c["stage"] == c["line_length"], c["rarity"], emb)
                         for c in cands])
        bias = np.stack([bias_for(ls["prior"], c["types"], era) for c in cands])
        logits, size = self.learnset(torch.tensor(cond), torch.tensor(bias))
        allowed = np.tile((self.move_gen <= era) & ~self.signature_move_mask, (len(cands), 1))
        sets = sample_learnsets(logits.numpy(), size.numpy(), ls["size_sd"], allowed, self.moves, self.levels,
                                era, rng, ls["temperature"])
        for c, s in zip(cands, sets):  # moves "on evolution" (level 0) only for evolved Pokemon
            c["learnset"] = s if c["stage"] > 1 else sorted((max(lv, 1), m) for lv, m in s)

    def too_close(self, name, ratio=0.85):
        """True if `name` starts a real Pokemon's name or is nearly the same string."""
        n = name.lower()
        return any(e.startswith(n) or difflib.SequenceMatcher(None, n, e).ratio() >= ratio for e in self.existing)

    def notable_words(self, text, n=4):
        """The rarest content words of a text (by document frequency in the Dex texts)."""
        words = re.findall(r"[a-z]+", re.sub(r"<[^<>]*>", " ", text.lower()))
        words = [w for w in dict.fromkeys(words) if len(w) > 3 and w not in TYPE_WORDS and w in self.real_words]
        return sorted(words, key=lambda w: -self.idf.get(w, 8.0))[:n]

    def name_ideas(self, text, emb, insp=None, per_word=5, n_context=8, n_canon=6):
        """Pools of name ideas for a text, from the literal to the associated:
        basis    words of what the text describes (mantis, shrimp), from `insp` (dex_inspire)
        concepts name ingredients capturing the basis and its Wikipedia facts, from `insp`
        facts    scientific and proper names in those facts (meganeuropsis, carboniferous)
        literal  the text's rarest content words (also the genus), a low-weight fallback
        assoc    words nearest each literal word by meaning (candle -> lantern, torch)
        context  words nearest the whole text's meaning
        canon    canon name-origin words nearest the text's meaning
        Apart from the basis, no idea shares a root with a literal word, so names don't echo the text."""
        ideas = self._offline_ideas(text, emb, per_word, n_context, n_canon)
        ideas.update(basis=[], concepts=[], facts=[])
        if insp and insp["basis"]:
            ideas["basis"] = [w for w in insp["basis_words"] if nameworthy(w) or w not in self.real_words][:4]
            ideas["concepts"] = [w for w in insp["concepts"] if not blocked(w)
                                 and not any(same_root(w, b) for b in ideas["basis"])][:8]
            terms = fact_terms([s for s, _ in insp["facts"]], nameworthy, self.real_words)
            ideas["facts"] = [t for t, kind in terms if kind == "proper" and not blocked(t)][:6]
        return ideas

    def _offline_ideas(self, text, emb, per_word, n_context, n_canon):
        literal = [w for w in self.notable_words(text, n=6) if nameworthy(w)][:4]
        taken = list(literal)

        def fresh(w):
            return not any(same_root(w, t) for t in taken)
        assoc = []
        if literal:
            index = {w: i for i, w in enumerate(self.words)}
            for word, v in zip(literal, self.embed_many(literal)):
                near = [self.words[i] for i in np.argsort(-(self.word_vecs @ v))[:40]]
                rel = [w for w in related_nouns(word) if w in index]
                cand = [w for w in dict.fromkeys(rel + near) if fresh(w) and nameworthy(w)]
                # near the word AND fitting the whole text: a rodent text keeps "rats", drops "computer"
                score = [self.word_vecs[index[w]] @ (v + emb) for w in cand]
                picks = [cand[j] for j in np.argsort(score)[::-1][:per_word]]
                assoc += picks
                taken += picks
        context = [w for w in (self.words[i] for i in np.argsort(-(self.word_vecs @ emb))[:80])
                   if fresh(w) and nameworthy(w)][:n_context]
        taken += context
        canon = [w for w in (self.concepts[i] for i in np.argsort(-(self.concept_vecs @ emb))[:30])
                 if not any(same_root(w, t) for t in literal)][:n_canon]
        return {"literal": literal, "assoc": assoc, "context": context, "canon": canon}

    def _pick_pair(self, ideas, rng, weights=(("assoc", 0.35), ("canon", 0.3), ("context", 0.25), ("literal", 0.1))):
        """Two ideas from different pools, never sharing a root; literal words are rare picks.
        With a basis, the first idea is a basis word (earlier ones likelier) and the second mostly
        an inspired concept: the creature plus something striking about it."""
        if ideas.get("basis"):
            basis = ideas["basis"]
            wb = np.array([1 / (i + 1) ** 2 for i in range(len(basis))])  # the head noun dominates
            a = basis[rng.choice(len(basis), p=wb / wb.sum())]
            items, w = [], []
            for pool, weight in (("concepts", 0.55), ("facts", 0.15), ("assoc", 0.15), ("canon", 0.15)):
                ok = [x for x in ideas[pool] if not same_root(x, a) and shared_piece(x, a) < 5]
                for x in ok:
                    items.append(x)
                    w.append(weight / len(ok))
            if items:
                w = np.array(w) / sum(w)
                return f"{a}, {items[rng.choice(len(items), p=w)]}"
        items, w = [], []
        for pool, weight in weights:
            for x in ideas[pool]:
                items.append((x, pool))
                w.append(weight / len(ideas[pool]))
        if not items:
            return None
        w = np.array(w) / sum(w)
        a = items[rng.choice(len(items), p=w)]
        rest = [(i, x) for i, x in enumerate(items) if x[1] != a[1] and not same_root(x[0], a[0])
                and shared_piece(x[0], a[0]) < 5]  # twilight + lighting share too much to blend
        if not rest:
            return a[0]
        rw = np.array([w[i] for i, _ in rest])
        b = rest[rng.choice(len(rest), p=rw / rw.sum())][1]
        return f"{a[0]}, {b[0]}"

    def _genus_and_name(self, cands, masked, emb, rng, seed, insp=None):
        """Genus from T5. Name: two ideas from `name_ideas` (mostly associated words, not the
        text's own), blended by T5, then filtered (one capitalized word, 3-12 characters, not a
        real Pokemon or close to one, not an English word)."""
        txt = bracketed(masked)
        src = [f"genus | types: {'/'.join(c['types'])} | text: {txt}" for c in cands]
        genus = [" ".join(dict.fromkeys(g[0].strip().split())) for g in self.t5(src, seed=seed)]
        if insp and insp.get("genus"):  # the inspired category, first option likeliest; T5's is the fallback
            options = insp["genus"]
            p = np.array([1 / (i + 1) for i in range(len(options))])
            genus = [options[rng.choice(len(options), p=p / p.sum())] for _ in cands]
        pairs = []  # PAIRS idea pairs per candidate, 8 names each
        for c, g in zip(cands, genus):
            c["name_ideas"] = ideas = self.name_ideas(masked + " " + g, emb, insp)
            pairs += [self._pick_pair(ideas, rng) or g.lower() for _ in range(PAIRS)]
        names = self.t5([f"name | {w}" for w in pairs], n=8, max_new=12, seed=seed + 2)
        for i, (c, g) in enumerate(zip(cands, genus)):
            best = None
            for w, ns in zip(pairs[i * PAIRS:(i + 1) * PAIRS], names[i * PAIRS:(i + 1) * PAIRS]):
                parts = [s.strip().lower() for s in w.split(",")]
                if len(parts) < 2:
                    continue
                # splices of the two ideas plus T5's blends, ranked by name-likeness, recognizable
                # pieces of both ideas, seam and length (poketype/dex_names.py)
                for x, score in rank(parts[0], parts[1], self.name_scorer, extra=[x.strip() for x in ns],
                                     soft=self.filter_names):
                    if (NAME_OK.match(x) and 3 <= len(x) <= MAX_NAME and x.lower() not in self.existing
                            and x.lower() not in self.real_words and not wordnet().synsets(x.lower())
                            and not self.too_close(x)):
                        if best is None or score > best[0]:
                            best = (score, x, w)
                        break
            c["genus"] = g
            c["name"], c["concepts"] = (best[1], best[2]) if best else (None, pairs[i * PAIRS])
            c["inspiration"] = self._provenance(insp, c["concepts"], emb) if insp and insp["basis"] else None

    def _provenance(self, insp, pair, text_vec):
        """What the name draws on: the basis articles and the fact closest in meaning to the
        chosen ideas and the description (the main basis's facts slightly preferred)."""
        fact = None
        if insp["facts"]:
            sents = [s for s, _ in insp["facts"]]
            target = self.embed_many([pair])[0] + text_vec
            main = insp["basis"][0][0]
            score = self.embed_many(sents) @ target + np.array([0.05 * (t == main) for _, t in insp["facts"]])
            fact = insp["facts"][int(np.argmax(score))]
        return {"based_on": [b[0] for b in insp["basis"]],
                "fact": fact[0] if fact else None, "fact_source": fact[1] if fact else None}

    def _present(self, c, text, era):
        st = c["stats"]
        return {
            "name": c["name"] or "???", "genus": f"{c['genus']} Pokémon", "types": c["types"],
            "height_m": round(float(np.exp(c["log_h"])) / 10, 1), "weight_kg": round(float(np.exp(c["log_w"])) / 10, 1),
            "shape": self.v["shape"][c["shape"]], "color": self.v["color"][c["color"]],
            "egg_groups": [self.v["egg"][i] for i in c["eggs"]],
            "stats": st, "bst": int(sum(st.values())),
            "abilities": {k: self.ability_names.get(v, str(v)) for k, v in c["abilities"].items()},
            "learnset": [(lv, self.move_names.get(m, str(m))) for lv, m in c["learnset"]],
            "evolution": {"stage": c["stage"], "line_length": c["line_length"]},
            "rarity": ["normal", "legendary", "mythical"][c["rarity"]],
            "era": era, "flavor_text": text, "name_concepts": c["concepts"], "name_ideas": c.get("name_ideas"),
            "inspiration": c.get("inspiration"),
            "score": round(c["score"], 3),
        }
