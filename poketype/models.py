"""Model definitions. Every model maps a (B, 3, H, W) batch to (B, 18) type logits."""

import torch
from torch import nn
from torchvision import models

from poketype.labels import TYPES

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """From-scratch baseline CNN: four conv blocks (96 -> 6 px), global average pool,
    dropout, linear head."""

    def __init__(self, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, chans[1:])])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(widths[-1], len(TYPES)))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18(pretrained=True, dropout=0.2):
    """ImageNet ResNet-18 with its 1000-class layer replaced by an 18-type head."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, len(TYPES)))
    return net


# name -> (constructor, input normalization mean/std)
ARCHITECTURES = {
    "small_cnn": (SmallCNN, ((0.5,) * 3, (0.5,) * 3)),
    "resnet18": (resnet18, (IMAGENET_MEAN, IMAGENET_STD)),
}


def build(arch, pretrained=True, dropout=None):
    """Model plus its input mean/std. pretrained=False skips downloading ImageNet
    weights (for loading a trained checkpoint); dropout=None keeps the default."""
    ctor, (mean, std) = ARCHITECTURES[arch]
    kw = {} if dropout is None else {"dropout": dropout}
    if arch == "resnet18":
        kw["pretrained"] = pretrained
    return ctor(**kw), mean, std


def head_parameter_names(model):
    """Names of the parameters in the newly added output head (vs. the backbone)."""
    return {n for n, _ in model.named_parameters() if n.startswith(("fc.", "head."))}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict_proba(model, loader, device, temperature=1.0):
    """Sigmoid per-type confidences for every batch in `loader`, as a numpy array."""
    model.eval()
    out = []
    for x, *_ in loader:
        out.append(torch.sigmoid(model(x.to(device, non_blocking=True)).float() / temperature).cpu())
    return torch.cat(out).numpy()
