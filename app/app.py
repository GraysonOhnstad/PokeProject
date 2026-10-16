"""Gradio app: upload or pick a Pokemon image -> predicted type(s) + inference time.

scripts/build_space.py assembles the app folder (this file, the poketype package,
model.pt, classes.json, examples/). Run it with `python app.py --share` for a temporary
public Gradio link.
"""

import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from poketype.inference import Predictor
from poketype.labels import TYPES

HERE = Path(__file__).resolve().parent
predictor = Predictor(HERE / "model.pt")
CLASSES = json.loads((HERE / "classes.json").read_text())
assert CLASSES == TYPES, "classes.json does not match the model's output order"
EXAMPLES = json.loads((HERE / "examples" / "examples.json").read_text())

INTRO = """# Pokémon Type Classifier
Predicts a Pokémon's type(s) from a single image. The model never saw any Pokémon introduced
in Generation 5 (Black/White) during training, in any art style, so the picker below shows
how it does on genuinely new Pokémon. Pick one, or upload your own image: a PNG with a
transparent background works best; a plain single-color background is removed automatically."""


def type_badges(types):
    return " / ".join(t.title() for t in types)


def attention_overlay(processed, amap):
    """Heat overlay of an attention map (coarse grid) on the model's input image."""
    img = np.asarray(processed.convert("RGB"), dtype=np.float32)
    a = np.asarray(Image.fromarray((amap / amap.max() * 255).astype(np.uint8)).resize(
        processed.size, Image.BILINEAR), dtype=np.float32)[..., None] / 255
    heat = np.array([255, 70, 0], np.float32)
    return Image.fromarray((img * (1 - 0.65 * a) + heat * 0.65 * a).astype(np.uint8))


def rules_table():
    """Both decision rules' test-set scores (Gen 5 Black/White sprites), from the model bundle."""
    rules = predictor.decision_rules
    if not rules:
        return ""
    rows = ["| Rule | Second type if ≥ | Both types right | Per-image F1 | Names two types |", "|---|---:|---:|---:|---:|"]
    for key, label in (("exact", "Exact match (shown above)"), ("f1", "F1")):
        r = rules[key]
        rows.append(f"| {label} | {r['threshold']:.2f} | {r['exact']:.1%} | {r['f1']:.3f} | {r['predicts_dual']:.0%} |")
    return "\n".join(rows)


DETAILS_INTRO = """Both rules use the same calibrated probabilities. The model always names its top type;
the rule decides when to add the runner-up. The **exact-match rule** (shown above) adds it only when
it is fairly likely, which gets the most Pokémon completely right. The **F1 rule** adds it at a much
lower bar, which earns more partial credit per image but gets fewer Pokémon completely right.
Test-set scores on the Gen 5 Black/White sprites:"""


def classify(image, key=""):
    if image is None:
        return None, "Upload an image or pick one below.", None, None, ""
    out = predictor.predict(image)
    lines = [f"### Predicted: {type_badges(out['types'])}",
             f"**Inference time:** {out['seconds']:.3f} s (preprocessing + model, CPU)"]
    ex = EXAMPLES.get(key or "")
    if ex:
        verdict = "correct" if set(out["types"]) == set(ex["types"]) else "not quite"
        lines.append(f"**Actual ({ex['name']}):** {type_badges(ex['types'])} — {verdict}")
    looked = None
    if out["attention"]:  # attention models: where it found evidence for its top type
        top = out["types"][0]
        looked = attention_overlay(out["processed"], out["attention"][top])
        lines.append(f"*The orange overlay shows where it found evidence for {top.title()}.*")
    details = ""
    if out["types_f1"]:
        details = f"**Under the F1 rule:** {type_badges(out['types_f1'])}"
        if ex:
            details += " — " + ("correct" if set(out["types_f1"]) == set(ex["types"]) else "not quite")
    return out["confidences"], "\n\n".join(lines), out["processed"], looked, details


with gr.Blocks(title="Pokémon Type Classifier") as demo:
    gr.Markdown(INTRO)
    key = gr.Textbox(visible=False)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", image_mode="RGBA", label="Pokémon image", sources=["upload", "clipboard"])
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            confidences = gr.Label(num_top_classes=5, label="Confidence per type")
            with gr.Row():
                processed = gr.Image(label="What the model sees", interactive=False, height=160)
                looked = gr.Image(label="Where it looked (top type)", interactive=False, height=160,
                                  visible=predictor.has_attention)
            with gr.Accordion("Behind the scenes: decision rules", open=False,
                              visible=predictor.threshold_f1 is not None):
                details = gr.Markdown()
                gr.Markdown(DETAILS_INTRO + "\n\n" + rules_table())
    outputs = [confidences, summary, processed, looked, details]

    for group, title in [("gen5_sprites", "Gen 5 sprites (never seen in training)"),
                         ("gen5_other_art", "The same new Pokémon in other art styles")]:
        items = [(f, e) for f, e in EXAMPLES.items() if e["group"] == group]
        gr.Examples(examples=[[str(HERE / "examples" / f), f] for f, _ in items], inputs=[image, key],
                    outputs=outputs, fn=classify, run_on_click=True, cache_examples=False,
                    example_labels=[e["name"] for _, e in items], visible_columns=[0],
                    examples_per_page=24, label=title, api_name=f"example_{group}")

    image.upload(lambda im: (*classify(im), ""), inputs=image, outputs=[*outputs, key], api_name="upload")
    button.click(classify, inputs=[image, key], outputs=outputs, api_name="predict")
    image.clear(lambda: (None, "", None, None, "", ""), outputs=[*outputs, key], api_name="clear")

if __name__ == "__main__":
    # --share: Gradio prints a temporary public link (the course deployment: code on GitHub + share link)
    demo.launch(share="--share" in sys.argv)
