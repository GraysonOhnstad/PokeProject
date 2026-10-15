"""Stat-line model: BST distribution x stat-share profile.

    BST     mixture of K Gaussians (on BST / 100), from the input features
    profile Dirichlet over the 6 stat shares (HP, Atk, Def, SpA, SpD, Spe), from the features
            and the BST (the true one in training, a sampled one when generating)

A stat line is sampled as BST ~ mixture, shares ~ Dirichlet(BST), stats = round(BST x shares),
clipped to 1-255. Optional auxiliary heads (evolution stage, line length, is-final, rarity)
share the trunk, to teach it the cues that set a Pokemon's power level.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 100.0  # BST is modelled in hundreds


class StatModel(nn.Module):
    def __init__(self, d_in, k=6, hidden=512, dropout=0.3, aux=True):
        super().__init__()
        self.k, self.aux = k, aux
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.bst = nn.Linear(hidden, 3 * k)
        self.profile = nn.Sequential(nn.Linear(hidden + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 6))
        if aux:
            self.aux_heads = nn.ModuleDict({"stage": nn.Linear(hidden, 3), "line_length": nn.Linear(hidden, 3),
                                            "is_final": nn.Linear(hidden, 1), "rarity": nn.Linear(hidden, 3)})

    def mixture(self, h):
        logit, mu, log_sd = self.bst(h).chunk(3, dim=1)
        return logit, mu + 4.5, log_sd.clamp(-4, 1)  # centred near a typical BST of 450

    def concentration(self, h, bst):
        return F.softplus(self.profile(torch.cat([h, (bst[:, None] - 4.5)], 1))) + 1e-3

    def forward(self, x, bst):
        h = self.trunk(x)
        out = {"mix": self.mixture(h), "alpha": self.concentration(h, bst)}
        if self.aux:
            out.update({k: head(h) for k, head in self.aux_heads.items()})
        return out


def mixture_nll(mix, y):
    logit, mu, log_sd = mix
    logp = (-0.5 * ((y[:, None] - mu) / log_sd.exp()) ** 2 - log_sd - 0.5 * np.log(2 * np.pi))
    return -torch.logsumexp(F.log_softmax(logit, 1) + logp, 1)


def dirichlet_nll(alpha, shares):
    return -torch.distributions.Dirichlet(alpha).log_prob(shares.clamp_min(1e-4))


def losses(out, d, aux_weight=1.0):
    per = {"bst": mixture_nll(out["mix"], d["bst"]), "profile": dirichlet_nll(out["alpha"], d["shares"])}
    if "stage" in out and aux_weight > 0:
        per["aux"] = aux_weight * (F.cross_entropy(out["stage"], d["stage"], reduction="none")
                                   + F.cross_entropy(out["line_length"], d["line_length"], reduction="none")
                                   + F.binary_cross_entropy_with_logits(out["is_final"][:, 0], d["is_final"],
                                                                        reduction="none")
                                   + F.cross_entropy(out["rarity"], d["rarity"], reduction="none")) / 4
    return per


@torch.no_grad()
def sample(model, x, n=1, generator=None):
    """n sampled stat lines per row: array (rows, n, 6) of ints, plus sampled BSTs (rows, n)."""
    model.eval()
    h = model.trunk(x)
    logit, mu, log_sd = model.mixture(h)
    comp = torch.multinomial(logit.softmax(1), n, replacement=True, generator=generator)
    z = torch.randn(comp.shape, generator=generator)
    bst = mu.gather(1, comp) + log_sd.exp().gather(1, comp) * z  # (rows, n), in hundreds
    bst = bst.clamp(1.8, 7.8)
    lines = []
    for j in range(n):
        alpha = model.concentration(h, bst[:, j])
        shares = torch.distributions.Dirichlet(alpha).sample()
        lines.append((shares * bst[:, j:j + 1] * SCALE).round().clamp(1, 255))
    return torch.stack(lines, 1).numpy().astype(int), bst.numpy() * SCALE


@torch.no_grad()
def expected(model, x):
    """Mixture mean BST and expected stats (BST mean x Dirichlet mean shares)."""
    model.eval()
    h = model.trunk(x)
    logit, mu, _ = model.mixture(h)
    bst = (logit.softmax(1) * mu).sum(1)
    alpha = model.concentration(h, bst)
    stats = alpha / alpha.sum(1, keepdim=True) * bst[:, None] * SCALE
    return bst.numpy() * SCALE, stats.numpy()


@torch.no_grad()
def bst_quantiles(model, x, qs=(0.1, 0.9), n=400, seed=0):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    h = model.trunk(x)
    logit, mu, log_sd = model.mixture(h)
    comp = torch.multinomial(logit.softmax(1), n, replacement=True, generator=gen)
    draws = mu.gather(1, comp) + log_sd.exp().gather(1, comp) * torch.randn(comp.shape, generator=gen)
    return np.quantile(draws.numpy() * SCALE, qs, axis=1).T
