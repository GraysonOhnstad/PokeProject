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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from poketype.dex import DEX, STATS, natural
from poketype.dex_learnset import MoveSetModel, bias_for, cond_vector, sample_learnsets
from poketype.dex_models import AbilityModel, Heads
from poketype.dex_stats import SCALE, StatModel
from poketype.dex_text import Masker, clean_text, species_token
from poketype.labels import TYPES

MAX_NAME = 12
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
    def __init__(self, bundle, textgen_dir, threads=8):
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

    def _naming_data(self, species, bundle):
        """Name-concept vocabulary (canon source words of the species this config trained on, with
        embeddings), word rarity for picking notable words from a text, and a real-word list."""
        from pathlib import Path
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

    # ---- text -------------------------------------------------------------------------------
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
        self._genus_and_name(cands, masked, emb, rng, seed)
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
        allowed = np.tile(self.move_gen <= era, (len(cands), 1))
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

    def _genus_and_name(self, cands, masked, emb, rng, seed):
        """Genus from T5. Name: one notable word from the text or genus plus one concept retrieved
        from the canon name-origin vocabulary by meaning, blended by T5, then filtered (one
        capitalized word, 3-12 characters, not a real Pokemon, not an English word)."""
        txt = bracketed(masked)
        src = [f"genus | types: {'/'.join(c['types'])} | text: {txt}" for c in cands]
        genus = [" ".join(dict.fromkeys(g[0].strip().split())) for g in self.t5(src, seed=seed)]
        retrieved = [self.concepts[i] for i in np.argsort(-(self.concept_vecs @ emb))[:6]]
        pairs = []
        for g in genus:
            own = self.notable_words(masked + " " + g) or [g.lower()]
            a = own[rng.integers(len(own))]
            b = next((r for r in rng.permutation(retrieved) if r[:4] != a[:4]), retrieved[0])
            pairs.append(f"{a}, {b}")
        names = self.t5([f"name | {w}" for w in pairs], n=8, max_new=12, seed=seed + 2)
        for c, g, w, ns in zip(cands, genus, pairs, names):
            parts = {s.strip().lower() for s in w.split(",")}
            ok = [x.strip() for x in ns if NAME_OK.match(x.strip()) and 3 <= len(x.strip()) <= MAX_NAME
                  and x.strip().lower() not in self.existing and x.strip().lower() not in parts
                  and x.strip().lower() not in self.real_words and not self.too_close(x.strip())]
            c["genus"], c["concepts"], c["name"] = g, w, (ok[0] if ok else None)

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
            "era": era, "flavor_text": text, "name_concepts": c["concepts"], "score": round(c["score"], 3),
        }
