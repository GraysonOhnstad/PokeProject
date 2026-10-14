"""Single-image prediction, shared by the evaluation script and the Gradio app."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T

from poketype.images import prepare
from poketype.labels import TYPES, decode
from poketype.models import build


class Predictor:
    """Loads a checkpoint (weights, architecture, normalization, temperature, threshold)
    and predicts types for one image at a time."""

    def __init__(self, checkpoint, device="cpu"):
        ckpt = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu")
        self.device = torch.device(device)
        self.model, _, _ = build(ckpt["arch"], pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.threshold = float(ckpt["threshold"])
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(ckpt["mean"], ckpt["std"])])

    @classmethod
    def from_run(cls, run_dir, device="cpu"):
        """best.pt from a training run, with calibration.json applied if present."""
        run_dir = Path(run_dir)
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
        cal = run_dir / "calibration.json"
        if cal.exists():
            c = json.loads(cal.read_text())
            ckpt["temperature"], ckpt["threshold"] = c["temperature"], c["threshold"]
        return cls(ckpt, device)

    @torch.no_grad()
    def probs(self, image, pixel_art=None):
        """Per-type confidences (18,) for a PIL image, plus the preprocessed 96x96 image."""
        x = prepare(image, pixel_art=pixel_art)
        logits = self.model(self.to_tensor(x).unsqueeze(0).to(self.device)).float()
        return torch.sigmoid(logits / self.temperature)[0].cpu().numpy(), x

    def predict(self, image, pixel_art=None):
        """Predicted types, all confidences, the timed duration, and the model input."""
        start = time.perf_counter()
        p, processed = self.probs(image, pixel_art)
        pred = decode(p[None], self.threshold)[0]
        seconds = time.perf_counter() - start
        types = [TYPES[i] for i in np.argsort(-p) if pred[i]]
        return {"types": types, "confidences": dict(zip(TYPES, p.tolist())),
                "seconds": seconds, "processed": processed}
