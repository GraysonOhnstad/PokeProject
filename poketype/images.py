"""Image loading shared by the manifest builder and the training data pipeline."""

import io
from pathlib import Path

from PIL import Image

BACKGROUND = (255, 255, 255)


def render_svg(path):
    """Rasterize an SVG at its native size. Needs the resvg-py package."""
    import resvg_py
    png = resvg_py.svg_to_bytes(svg_path=str(path))
    return Image.open(io.BytesIO(bytes(png)))


def load_rgba(path):
    """Open any sprite (png, gif first frame, webp, svg) as an RGBA image."""
    path = Path(path)
    if path.suffix == ".svg":
        im = render_svg(path)
    else:
        im = Image.open(path)
        im.seek(0)  # first frame of animated GIF/WebP
    return im.convert("RGBA")


def composite(im, background=BACKGROUND):
    """Alpha-composite an RGBA image onto a solid background, returning RGB."""
    bg = Image.new("RGBA", im.size, background + (255,))
    return Image.alpha_composite(bg, im).convert("RGB")
