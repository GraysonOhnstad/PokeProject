"""Single-image prediction, shared by the evaluation scripts and the Gradio app.

Two checkpoint kinds:
  - a single model from rounds 1-2 (train.py), with calibration.json applied;
  - a round-3 bundle (calibrate3.py): the K cross-validation fold models of one TypeNet
    configuration, one shared temperature and second-type threshold. The ensemble
    averages the members' temperature-scaled logits, then converts to per-type
    probabilities, which keeps it close to the members' calibration (averaging
    probabilities instead would make it systematically less confident).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T

from poketype.images import SIZE, prepare
from poketype.labels import TYPES, decode
from poketype.models import build


class Predictor:
    """Loads a checkpoint and predicts types for one image at a time (or a batch)."""

    def __init__(self, checkpoint, device="cpu"):
        ckpt = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu")
        self.device = torch.device(device)
        self.ensemble = ckpt.get("version") == 3
        if self.ensemble:
            from poketype.typenet import TypeNet
            self.models = []
            for state in ckpt["models"]:
                m = TypeNet(**ckpt["model_cfg"], pretrained=False)
                m.load_state_dict(state)
                self.models.append(m.to(self.device).eval())
        else:
            model, _, _ = build(ckpt["arch"], pretrained=False)
            model.load_state_dict(ckpt["model"])
            self.models = [model.to(self.device).eval()]
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.threshold = float(ckpt["threshold"])
        # optional second rule (F1-tuned threshold) and both rules' test metrics, for the app's details view
        self.threshold_f1 = ckpt.get("threshold_f1")
        self.decision_rules = ckpt.get("decision_rules")
        self.has_attention = (ckpt.get("model_cfg") or {}).get("pool") == "attn"
        self.size = int(ckpt.get("size", SIZE))  # preprocessing resolution the model was trained on
        self.input_size = ckpt.get("input_size")
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.to_tensor = T.Compose(
            ([T.Resize(self.input_size, interpolation=T.InterpolationMode.NEAREST)] if self.input_size else [])
            + [T.ToTensor(), T.Normalize(self.mean, self.std)])

    @property
    def model(self):
        """The single model of a round 1-2 checkpoint (first member of an ensemble)."""
        return self.models[0]

    @classmethod
    def from_run(cls, run_dir, device="cpu"):
        """best.pt from a round 1-2 training run, with calibration.json applied if present."""
        run_dir = Path(run_dir)
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
        cal = run_dir / "calibration.json"
        if cal.exists():
            c = json.loads(cal.read_text())
            ckpt["temperature"], ckpt["threshold"] = c["temperature"], c["threshold"]
        return cls(ckpt, device)

    @torch.no_grad()
    def batch(self, x):
        """Per-type probabilities (B, 18) and, for attention models, attention maps
        (B, 18, H, W) for a batch of normalized input tensors."""
        x = x.to(self.device)
        if not self.ensemble:
            return torch.sigmoid(self.models[0](x).float() / self.temperature).cpu(), None
        logits, attention = 0, 0
        for m in self.models:
            out = m(x)
            logits = logits + out["logits"].float() / self.temperature
            if out["attention"] is not None:
                attention = attention + out["attention"].float()
        n = len(self.models)
        probs = self.models[0].type_probs(logits / n)
        return probs.cpu(), (attention / n).cpu() if torch.is_tensor(attention) else None

    def probs(self, image, pixel_art=None):
        """Per-type confidences (18,), the preprocessed image, and attention maps (or None)."""
        x = prepare(image, pixel_art=pixel_art, size=self.size)
        p, attention = self.batch(self.to_tensor(x).unsqueeze(0))
        return p[0].numpy(), x, None if attention is None else attention[0].numpy()

    def predict(self, image, pixel_art=None):
        """Predicted types, all confidences, the timed duration, the model input, and
        attention maps for the predicted types (attention models only)."""
        start = time.perf_counter()
        p, processed, attention = self.probs(image, pixel_art)
        pred = decode(p[None], self.threshold)[0]
        seconds = time.perf_counter() - start
        types = [TYPES[i] for i in np.argsort(-p) if pred[i]]
        maps = None if attention is None else {t: attention[TYPES.index(t)] for t in types}
        types_f1 = None
        if self.threshold_f1 is not None:  # the same probabilities under the F1-tuned rule
            alt = decode(p[None], float(self.threshold_f1))[0]
            types_f1 = [TYPES[i] for i in np.argsort(-p) if alt[i]]
        return {"types": types, "types_f1": types_f1, "confidences": dict(zip(TYPES, p.tolist())),
                "seconds": seconds, "processed": processed, "attention": maps}
