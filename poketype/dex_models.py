"""Text -> entry-field models for the Dex generator, and their training loop.

Two input modes share the same heads and loss:
  frozen     precomputed sentence embeddings (+ era one-hot) into an MLP trunk
  finetune   MiniLM itself, trained end to end, with the era one-hot appended after pooling

Early stopping is per field: each field keeps the predictions from the epoch where its own
loss on a stopping set (held-out training families) was lowest, and training ends when no
field has improved for `patience` epochs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FIELDS = ["types", "shape", "color", "egg", "height", "weight", "stats", "abilities"]


class Heads(nn.Module):
    def __init__(self, d_in, sizes, hidden=512, dropout=0.3, layers=2):
        super().__init__()
        trunk, d = [], d_in
        for _ in range(layers):
            trunk += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.heads = nn.ModuleDict({k: nn.Linear(d, n) for k, n in sizes.items()})

    def forward(self, x):
        h = self.trunk(x)
        return {k: head(h) for k, head in self.heads.items()}


class FineTuned(nn.Module):
    """Encoder + mean pooling + era one-hot + Heads."""

    def __init__(self, encoder, n_era, sizes, hidden=512, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.heads = Heads(encoder.config.hidden_size + n_era, sizes, hidden, dropout, layers=1)

    def forward(self, x):
        ids, mask, era = x
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        return self.heads(torch.cat([pooled, era], 1))


def field_losses(out, d):
    """Per-row loss for each field: {field: tensor (batch,)}."""
    bce = F.binary_cross_entropy_with_logits
    return {
        "types": bce(out["types"], d["types"], reduction="none").mean(1),
        "shape": F.cross_entropy(out["shape"], d["shape"], reduction="none"),
        "color": F.cross_entropy(out["color"], d["color"], reduction="none"),
        "egg": bce(out["egg"], d["egg"], reduction="none").mean(1),
        "height": F.mse_loss(out["reg"][:, 0], d["reg"][:, 0], reduction="none"),
        "weight": F.mse_loss(out["reg"][:, 1], d["reg"][:, 1], reduction="none"),
        "stats": F.mse_loss(out["reg"][:, 2:], d["reg"][:, 2:], reduction="none").mean(1),
        "abilities": d["has"] * bce(out["abil"], d["abil"], reduction="none").sum(1) / 3,
    }


def to_predictions(out, mu, sd):
    reg = out["reg"].numpy() * sd + mu
    return {"types": torch.sigmoid(out["types"]).numpy(), "shape": out["shape"].softmax(1).numpy(),
            "color": out["color"].softmax(1).numpy(), "egg": torch.sigmoid(out["egg"]).numpy(),
            "log_height": reg[:, 0], "log_weight": reg[:, 1], "stats": reg[:, 2:],
            "abilities": torch.sigmoid(out["abil"]).numpy()}


PRED_OF = {"types": ["types"], "shape": ["shape"], "color": ["color"], "egg": ["egg"],
           "height": ["log_height"], "weight": ["log_weight"], "stats": ["stats"], "abilities": ["abilities"]}


def train(model, inputs, targets, weights, tr, stop, ev, param_groups, epochs, patience,
          batch=256, log=None, seed=0):
    """Train on rows `tr`, early-stop each field on rows `stop`, predict rows `ev`.

    inputs: callable(index tensor) -> model input. targets: dict of tensors over all rows
    (with "reg" standardized; mu/sd passed in targets["_mu"], ["_sd"]). weights: tensor per row.
    Returns (predictions for `ev`, best epoch per field).
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(param_groups)
    steps = epochs * ((len(tr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in param_groups],
                                                total_steps=steps, pct_start=0.1)
    mu, sd = targets["_mu"], targets["_sd"]
    data = {k: v for k, v in targets.items() if not k.startswith("_")}

    def run(ix, chunk=512):
        outs = []
        for b in torch.as_tensor(ix).split(chunk):
            outs.append(model(inputs(b)))
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    def weighted(losses, ix):
        w = weights[ix]
        return {k: ((v * w).sum() / w.sum()).item() for k, v in losses.items()}

    best = {f: float("inf") for f in FIELDS}
    best_epoch = {f: 0 for f in FIELDS}
    preds = {}
    tr_t = torch.as_tensor(tr)
    for epoch in range(1, epochs + 1):
        model.train()
        for b in tr_t[torch.randperm(len(tr_t))].split(batch):
            losses = field_losses(model(inputs(b)), {k: v[b] for k, v in data.items()})
            w = weights[b]
            loss = sum((v * w).sum() / w.sum() for v in losses.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            stop_out = run(stop)
            stop_loss = weighted(field_losses(stop_out, {k: v[stop] for k, v in data.items()}), stop)
            improved = [f for f in FIELDS if stop_loss[f] < best[f] - 1e-4]
            if improved:
                ev_pred = to_predictions(run(ev), mu, sd)
                for f in improved:
                    best[f], best_epoch[f] = stop_loss[f], epoch
                    for k in PRED_OF[f]:
                        preds[k] = ev_pred[k]
        if log:
            log(f"    epoch {epoch:3d}  improved: {', '.join(improved) or '-'}")
        if epoch - max(best_epoch.values()) >= patience:
            break
    return preds, best_epoch


def stopping_split(rows, tr, frac=0.15, seed=0):
    """Split training rows into (fit, stop) by evolution family."""
    fams = np.unique(rows.evolution_chain_id.to_numpy()[tr])
    rng = np.random.default_rng(seed)
    stop_fams = set(rng.choice(fams, int(round(frac * len(fams))), replace=False))
    is_stop = np.isin(rows.evolution_chain_id.to_numpy()[tr], list(stop_fams))
    return tr[~is_stop], tr[is_stop]
