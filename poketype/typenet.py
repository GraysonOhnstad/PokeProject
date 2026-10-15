"""TypeNet (round 3): a pretrained backbone plus type-specific pooling and output heads.

    image -> backbone -> feature map (C x H x W)
             -> per-type scores, from whole-image average pooling, optionally plus
                class-specific residual attention (each type weights the regions where
                its own score is high; Zhu & Wu, "Residual Attention", ICCV 2021)
             -> output head:
                  "sigmoid": 18 independent yes/no outputs (binary cross-entropy)
                  "set":     softmax over the 171 valid answers (18 single types + 153
                             distinct pairs); a pair's score is the sum of its two type
                             scores + an image-dependent "has a second type" score +
                             optionally a learned non-negative pair boost
             -> optional auxiliary heads (egg groups, body shape, color), trained only
                as extra supervision and ignored at prediction time

Every head yields per-type probabilities (for a set head, the marginal probability that
the answer contains the type), so decoding, calibration and metrics work unchanged.
"""

import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from poketype.labels import TYPES

N = len(TYPES)
PAIRS = list(itertools.combinations(range(N), 2))  # 153 distinct pairs (i < j)
PAIR_INDEX = {p: k for k, p in enumerate(PAIRS)}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def set_membership():
    """(171, 18) 0/1 matrix: which types each valid answer contains."""
    m = torch.zeros(N + len(PAIRS), N)
    m[torch.arange(N), torch.arange(N)] = 1
    for k, (i, j) in enumerate(PAIRS):
        m[N + k, i] = m[N + k, j] = 1
    return m


def set_index(type_idx):
    """Index of the valid answer for a list of 1 or 2 type indices."""
    if len(type_idx) == 1:
        return type_idx[0]
    return N + PAIR_INDEX[tuple(sorted(type_idx))]


class Backbone(nn.Module):
    """Pretrained feature extractor returning a (B, C, H, W) feature map."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        if name in ("resnet18", "resnet50"):
            weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                       "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}[name] if pretrained else None
            net = getattr(models, name)(weights=weights)
            self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
            self.channels = net.fc.in_features
        elif name == "convnext_tiny":
            net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            # The classifier's channel-wise LayerNorm applies per location, so keep it on the map.
            self.body = nn.Sequential(net.features, net.classifier[0])
            self.channels = net.classifier[2].in_features
        else:
            raise ValueError(f"unknown backbone {name}")

    def forward(self, x):
        return self.body(x)


class TypeNet(nn.Module):
    def __init__(self, backbone="resnet18", pool="avg", head="sigmoid", pair_boost=False,
                 aux_sizes=None, dropout=0.5, attn_lambda=0.2, learn_lambda=False, pretrained=True):
        super().__init__()
        self.cfg = dict(backbone=backbone, pool=pool, head=head, pair_boost=pair_boost,
                        aux_sizes=aux_sizes or {}, dropout=dropout, attn_lambda=attn_lambda,
                        learn_lambda=learn_lambda)
        self.backbone = Backbone(backbone, pretrained)
        c = self.backbone.channels
        self.dropout = nn.Dropout(dropout)
        self.type_scores = nn.Conv2d(c, N, 1)  # a per-location score for each type
        if pool == "attn":
            init = torch.tensor(float(attn_lambda))
            self.attn_lambda = nn.Parameter(init) if learn_lambda else None
            self.register_buffer("fixed_lambda", init)
        if head == "set":
            self.dual = nn.Linear(c, 1)  # image-dependent "has a second type" score
            self.register_buffer("membership", set_membership())
            if pair_boost:
                # boost = softplus(w) >= 0, starting near 0: common pairs can earn a boost,
                # unseen or rare pairs stay at ~0 and are never pushed below their type scores
                self.pair_w = nn.Parameter(torch.full((len(PAIRS),), -5.0))
            self.register_buffer("pair_i", torch.tensor([i for i, _ in PAIRS]))
            self.register_buffer("pair_j", torch.tensor([j for _, j in PAIRS]))
        self.aux = nn.ModuleDict({k: nn.Linear(c, n) for k, n in (aux_sizes or {}).items()})

    def head_parameter_names(self):
        """Parameters added on top of the pretrained backbone (trained at the full LR)."""
        return {n for n, _ in self.named_parameters() if not n.startswith("backbone.")}

    def pair_boost(self):
        return F.softplus(self.pair_w) if self.cfg["head"] == "set" and self.cfg["pair_boost"] else None

    def forward(self, x):
        """Returns {"logits": (B, 18) or (B, 171), "aux": {name: (B, n)}, "attention": (B, 18, H, W) or None}."""
        fmap = self.backbone(x)
        pooled = fmap.mean((2, 3))
        smap = self.type_scores(self.dropout(fmap))  # (B, 18, H, W)
        flat = smap.flatten(2)
        z = flat.mean(2)  # whole-image average of each type's score map
        attention = None
        if self.cfg["pool"] == "attn":
            weights = flat.softmax(2)  # each type attends where its own score is high
            lam = self.attn_lambda if self.attn_lambda is not None else self.fixed_lambda
            z = z + lam * (weights * flat).sum(2)
            attention = weights.view_as(smap)
        if self.cfg["head"] == "set":
            d = self.dual(self.dropout(pooled))
            pairs = z[:, self.pair_i] + z[:, self.pair_j] + d
            boost = self.pair_boost()
            if boost is not None:
                pairs = pairs + boost
            logits = torch.cat([z, pairs], 1)
        else:
            logits = z
        aux = {k: head(self.dropout(pooled)) for k, head in self.aux.items()}
        return {"logits": logits, "aux": aux, "attention": attention}

    def type_probs(self, logits, temperature=1.0):
        """Per-type probabilities from the logits (marginals for a set head)."""
        if self.cfg["head"] == "set":
            return (logits / temperature).softmax(1) @ self.membership
        return torch.sigmoid(logits / temperature)


def type_targets_for_sets(y):
    """(B, 18) multi-hot -> (B,) index of the valid answer (for the set head's loss)."""
    return torch.tensor([set_index(torch.nonzero(row).flatten().tolist()) for row in y], device=y.device)
