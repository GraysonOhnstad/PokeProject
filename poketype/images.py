"""Image loading and preprocessing shared by the data pipeline and the app."""

import io
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

BACKGROUND = (255, 255, 255)
SIZE = 96
BG_TOLERANCE = 24  # max per-channel difference from the border color, for opaque images
MIN_BLOB_FRACTION = 0.001  # detached subject blobs smaller than this share of the image are noise


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


def border_flood_mask(rgb, tolerance=BG_TOLERANCE, min_blob=MIN_BLOB_FRACTION):
    """Subject mask for an image without transparency: background is the region of
    the most common border color that is connected to the image border. 4-connectivity,
    so the fill cannot leak through diagonal gaps in a sprite's outline; white areas
    enclosed by the outline stay part of the subject. Tiny detached blobs (JPEG noise
    around edges) are dropped."""
    a = np.asarray(rgb, dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg_color = np.array(Counter(map(tuple, border)).most_common(1)[0][0], dtype=np.int16)
    near_bg = (np.abs(a - bg_color) <= tolerance).all(axis=2)
    labels, _ = ndimage.label(near_bg)  # default structure is 4-connected
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    subject = ~np.isin(labels, edge_labels[edge_labels > 0])
    blobs, n = ndimage.label(subject, structure=np.ones((3, 3)))
    sizes = ndimage.sum(subject, blobs, index=np.arange(1, n + 1))
    small = np.flatnonzero(sizes < min_blob * subject.size) + 1
    return subject & ~np.isin(blobs, small)


def with_subject_alpha(im):
    """RGBA image whose alpha marks the subject. Images with transparency keep their
    own alpha; opaque images (e.g. Red/Green sprites, JPG uploads) get a binary alpha
    from a border flood fill, which also removes a plain non-white background."""
    im = im.convert("RGBA")
    if im.getchannel("A").getextrema()[0] < 255:
        return im
    mask = border_flood_mask(im.convert("RGB"))
    im.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    return im


def prepare_with_mask(im, pixel_art=None, size=SIZE, background=BACKGROUND):
    """Crop to the subject, flatten onto the background, and fit the longer side to
    `size` on a square canvas. Returns (RGB image, boolean subject mask), both
    size x size. Pixel art is resized nearest-neighbor to stay crisp; everything else
    uses Lanczos. With pixel_art=None (unknown source, e.g. an upload), images that
    need upscaling are treated as pixel art."""
    im = with_subject_alpha(im)
    bbox = im.getchannel("A").getbbox()
    if bbox:
        im = im.crop(bbox)
    scale = size / max(im.size)
    new_size = tuple(max(1, round(d * scale)) for d in im.size)
    if pixel_art is None:
        pixel_art = scale >= 1
    resample = Image.NEAREST if pixel_art else Image.LANCZOS
    rgb = composite(im, background).resize(new_size, resample)
    alpha = im.getchannel("A").resize(new_size, Image.NEAREST if pixel_art else Image.BILINEAR)
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas = Image.new("RGB", (size, size), background)
    canvas.paste(rgb, offset)
    mask = Image.new("L", (size, size), 0)
    mask.paste(alpha, offset)
    return canvas, np.asarray(mask) >= 128


def prepare(im, pixel_art=None, size=SIZE, background=BACKGROUND):
    """Model input only; see prepare_with_mask."""
    return prepare_with_mask(im, pixel_art, size, background)[0]
