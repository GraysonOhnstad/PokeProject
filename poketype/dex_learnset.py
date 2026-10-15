"""Learnset generator: a small transformer that writes a level-up learnset token by token.

Sequence: L(level) M(move) L M ... EOS, in level order (level 0 = learned on evolution).
A conditioning vector (types, stats, era, evolution position, text embedding) enters as a
prefix position. Generation enforces the grammar (level, move, level, ...), non-decreasing
levels, no repeated move, and only moves that exist in the entry's era, so every output
uses the existing move inventory and is well-formed by construction.
"""

import numpy as np
import torch
import torch.nn as nn

from poketype.labels import TYPES

PAD, EOS = 0, 1
LEVEL0 = 2               # token of level 0; level l -> LEVEL0 + l
N_LEVELS = 101           # 0..100
MOVE0 = LEVEL0 + N_LEVELS


def _one_hot(i, n):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1
    return v


def cond_vector(types, stats, generation, stage, line_length, is_final, rarity, emb):
    """Conditioning for one learnset. types: names; stats: 6 values in STATS order; rarity:
    0 normal, 1 legendary, 2 mythical; emb: mean text embedding or None."""
    st = np.asarray(stats, dtype=np.float32)
    t = np.zeros(18, dtype=np.float32)
    t[[TYPES.index(x) for x in types]] = 1
    return np.concatenate([
        t, st / 100, [st.sum() / 600], _one_hot(generation - 1, 9),
        _one_hot(stage - 1, 3), _one_hot(line_length - 1, 3), [float(is_final)], _one_hot(rarity, 3),
        emb if emb is not None else np.zeros(384, np.float32), [float(emb is not None)]]).astype(np.float32)


def move_prior(entries, moves, rows_ix, smoothing=1e-3):
    """Log-prior over moves per (generation, type): how often each move is in the level-up
    learnset of a Pokemon with that type in that generation (each species counted once per
    game). Built from entries `rows_ix` only. Returns array (10, 18, n_moves)."""
    index = {m: i for i, m in enumerate(moves)}
    counts = np.zeros((10, 18, len(moves)))
    n = np.zeros((10, 18, 1))
    for i in rows_ix:
        e = entries[i]
        present = [index[m] for _, m in e["learnset"] if m in index]
        for t in e["types"]:
            ti = TYPES.index(t)
            counts[e["generation"], ti, present] += 1
            n[e["generation"], ti] += 1
    return np.log(counts / np.maximum(n, 1) + smoothing).astype(np.float32)


def bias_for(prior, types, generation):
    """Move log-prior for one entry: the average over its types."""
    return prior[generation, [TYPES.index(t) for t in types]].mean(0)


class Tokens:
    def __init__(self, move_ids):
        self.moves = list(move_ids)
        self.index = {m: i for i, m in enumerate(self.moves)}
        self.size = MOVE0 + len(self.moves)

    def encode(self, learnset):
        out = []
        for level, move in learnset:
            out += [LEVEL0 + level, MOVE0 + self.index[move]]
        return out + [EOS]

    def decode(self, seq):
        pairs = []
        for a, b in zip(seq[0::2], seq[1::2]):
            if a == EOS or b < MOVE0:
                break
            pairs.append((int(a - LEVEL0), self.moves[b - MOVE0]))
        return pairs


class LearnsetModel(nn.Module):
    def __init__(self, n_tokens, d_cond, d=256, layers=4, heads=4, max_len=80, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(n_tokens, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.cond = nn.Sequential(nn.Linear(d_cond, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_tokens)

    def forward(self, cond, prev, move_bias=None):
        """cond (B, d_cond); prev (B, T) tokens so far; move_bias (B, n_moves) log-prior added to
        the move logits. Returns logits (B, T + 1, vocab): position t predicts token t (position 0
        predicts the first token). The conditioning is a prefix and is also added at every
        position, so the model cannot drift away from it over a long sequence."""
        c = self.cond(cond)[:, None]
        x = torch.cat([c, self.tok(prev)], 1)
        x = x + c + self.pos(torch.arange(x.shape[1]))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        logits = self.out(self.norm(self.body(x, mask=mask, is_causal=True)))
        if move_bias is not None:
            logits = torch.cat([logits[..., :MOVE0], logits[..., MOVE0:] + move_bias[:, None]], -1)
        return logits


@torch.no_grad()
def generate(model, cond, allowed, max_moves=40, temperature=1.0, top_p=0.95, generator=None, move_bias=None):
    """Sample learnsets. cond (B, d_cond); allowed (B, n_moves) bool: moves available in each
    entry's era. Returns a list of token lists."""
    model.eval()
    B, n_moves = allowed.shape
    vocab = model.out.out_features
    seq = torch.zeros(B, 0, dtype=torch.long)
    used = torch.zeros(B, n_moves, dtype=torch.bool)
    last_level = torch.zeros(B, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    for step in range(2 * max_moves + 1):
        logits = model(cond, seq, move_bias)[:, -1] / temperature
        ok = torch.zeros(B, vocab, dtype=torch.bool)
        if step % 2 == 0:  # a level, or the end (only after at least one move)
            lv = torch.arange(N_LEVELS)
            ok[:, LEVEL0:MOVE0] = lv[None] >= last_level[:, None]
            ok[:, EOS] = step > 0
        else:              # a move: available in the era and not used yet
            ok[:, MOVE0:] = allowed & ~used
        logits = logits.masked_fill(~ok, float("-inf"))
        probs = logits.softmax(-1)
        if top_p < 1:
            sp, si = probs.sort(-1, descending=True)
            cut = sp.cumsum(-1) - sp > top_p
            sp = sp.masked_fill(cut, 0)
            probs = torch.zeros_like(probs).scatter(-1, si, sp)
        nxt = torch.multinomial(probs, 1, generator=generator)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        if step % 2 == 0:
            done |= nxt == EOS
            is_level = (nxt >= LEVEL0) & (nxt < MOVE0)
            last_level = torch.where(is_level, nxt - LEVEL0, last_level)
        else:
            is_move = nxt >= MOVE0
            used[is_move.nonzero()[:, 0], (nxt[is_move] - MOVE0)] = True
        seq = torch.cat([seq, nxt[:, None]], 1)
        if done.all():
            break
    return [s[s != PAD].tolist() for s in seq]


def pad_batch(seqs, length=None):
    length = length or max(map(len, seqs))
    out = torch.full((len(seqs), length), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


def nll(model, cond, seqs, move_bias=None):
    """Per-sequence summed token NLL and token counts."""
    y = pad_batch(seqs)
    logits = model(cond, y[:, :-1], move_bias)
    lp = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, ignore_index=PAD, reduction="none")
    return lp.sum(1), (y != PAD).sum(1)


def move_scores(samples, tokens, n_moves):
    """Share of samples containing each move: (B, n_moves) from a list of S sample lists per row."""
    out = np.zeros((len(samples), n_moves), dtype=np.float32)
    for r, ss in enumerate(samples):
        for s in ss:
            for _, m in tokens.decode(s):
                out[r, tokens.index[m]] += 1
        out[r] /= max(len(ss), 1)
    return out


class MoveSetModel(nn.Module):
    """Learnset as a set: one score per move (a residual on the type-and-era prior) and a size.
    More data-efficient than writing the sequence; levels are assigned afterwards."""

    def __init__(self, d_cond, n_moves, hidden=512, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_cond, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.moves = nn.Linear(hidden, n_moves)
        self.size = nn.Linear(hidden, 1)
        nn.init.zeros_(self.moves.weight)  # start from the prior
        nn.init.zeros_(self.moves.bias)

    def forward(self, cond, move_bias):
        h = self.trunk(cond)
        return self.moves(h) + move_bias, self.size(h)[:, 0]


def level_table(entries, moves, rows_ix):
    """{(generation, move id): array of levels it is learned at} from entries `rows_ix`;
    generation 0 pools every generation."""
    table = {}
    for i in rows_ix:
        e = entries[i]
        for lv, m in e["learnset"]:
            table.setdefault((e["generation"], m), []).append(lv)
            table.setdefault((0, m), []).append(lv)
    return {k: np.array(v) for k, v in table.items()}


def sample_learnsets(logits, size_mean, size_sd, allowed, moves, levels, generation, rng, temperature=1.0):
    """Sample learnsets: size ~ N(predicted, sd); moves by Gumbel top-k on the scores (allowed
    moves only); each move's level drawn from the levels it is learned at in that generation
    (or any generation), then sorted. Returns lists of (level, move id)."""
    out = []
    for r in range(len(logits)):
        k = int(np.clip(round(size_mean[r] + rng.normal() * size_sd), 2, 40))
        g = logits[r] / temperature + rng.gumbel(size=logits.shape[1])
        g[~allowed[r]] = -np.inf
        pick = np.argsort(-g)[:k]
        pairs = []
        for j in pick:
            m = moves[j]
            pool = levels.get((generation, m), levels.get((0, m), np.array([1])))
            pairs.append((int(rng.choice(pool)), m))
        out.append(sorted(pairs))
    return out
