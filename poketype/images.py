"""Image loading shared by the manifest builder and the training data pipeline."""

import io
from pathlib import Path

from PIL import Image, ImageChops

BACKGROUND = (255, 255, 255)
SIZE = 96


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


def content_bbox(im, background=BACKGROUND):
    """Bounding box of the subject: non-transparent pixels, or non-background
    pixels for images without transparency (e.g. Red/Green sprites)."""
    alpha = im.getchannel("A")
    if alpha.getextrema()[0] < 255:
        return alpha.getbbox()
    bg = Image.new("RGB", im.size, background)
    return ImageChops.difference(im.convert("RGB"), bg).getbbox()


def prepare(im, pixel_art=None, size=SIZE, background=BACKGROUND):
    """Crop to the subject, flatten onto the background, and fit the longer side to
    `size` on a square canvas. Pixel art is resized nearest-neighbor to stay crisp;
    everything else uses Lanczos. With pixel_art=None (unknown source, e.g. an
    upload), images that need upscaling are treated as pixel art."""
    im = im.convert("RGBA")
    bbox = content_bbox(im, background)
    if bbox:
        im = im.crop(bbox)
    rgb = composite(im, background)
    scale = size / max(rgb.size)
    new_size = tuple(max(1, round(d * scale)) for d in rgb.size)
    if pixel_art is None:
        pixel_art = scale >= 1
    resample = Image.NEAREST if pixel_art else Image.LANCZOS
    rgb = rgb.resize(new_size, resample)
    canvas = Image.new("RGB", (size, size), background)
    canvas.paste(rgb, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas
