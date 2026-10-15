"""Sketchify: turn a sprite into a hand-doodled look.

Used for the Pokedex's doodle view of generated Pokemon, for doodle->sprite training pairs
(sketchified real sprite -> real sprite), and as doodle-style augmentation for the image encoder.

Steps: crop and enlarge; smooth away pixel stair-steps; flatten to a few lightened marker
colors (k-means, which also drops most shading); draw outlines where colors meet (thicker on
the silhouette); bend lines and fill with a smooth random displacement so they wobble, and
shift the fill a little off the lines, as a hurried colorer would.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans

PAPER = np.array([250, 248, 242], dtype=np.float32)
INK = np.array([40, 38, 45], dtype=np.float32)


def _smooth_field(shape, scale, amp, rng):
    """Low-frequency random displacement (pixels) of the given shape."""
    small = rng.normal(size=(max(2, shape[0] // scale), max(2, shape[1] // scale)))
    field = ndimage.zoom(small, (shape[0] / small.shape[0], shape[1] / small.shape[1]), order=3)
    return field[:shape[0], :shape[1]] * amp


def _warp(img, dy, dx, order=1):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]].astype(np.float32)
    coords = [yy + dy, xx + dx]
    if img.ndim == 2:
        return ndimage.map_coordinates(img, coords, order=order, mode="nearest")
    return np.stack([ndimage.map_coordinates(img[..., c], coords, order=order, mode="nearest")
                     for c in range(img.shape[2])], -1)


def sketchify(sprite, size=384, colors=5, wobble=3.0, offset=4, seed=0):
    """sprite: RGBA PIL image (transparent background). Returns an RGB PIL image."""
    rng = np.random.default_rng(seed)
    im = sprite.convert("RGBA")
    bbox = im.getchannel("A").getbbox()
    if bbox:
        im = im.crop(bbox)
    scale = (size * 0.8) / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.NEAREST)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0

    sigma = max(1.0, scale * 0.6)
    alpha = ndimage.gaussian_filter(a[..., 3], sigma) > 0.5
    rgb = np.stack([ndimage.gaussian_filter(a[..., c] * a[..., 3], sigma) for c in range(3)], -1)
    rgb /= np.maximum(ndimage.gaussian_filter(a[..., 3], sigma), 1e-3)[..., None]

    # the sprite's own near-black pixels (outlines, eyes, mouths) become the inner lines
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    raw_lum = a[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # pixels clearly darker than their surroundings: sprite outlines (often dark blue or brown,
    # not black), eyes and mouths
    weight = ndimage.gaussian_filter(a[..., 3], 4 * scale)
    local = ndimage.gaussian_filter(raw_lum * a[..., 3], 4 * scale) / np.maximum(weight, 1e-3)
    darkish = ((raw_lum < 0.3) | (raw_lum < local - 0.18)).astype(np.float32) * a[..., 3]
    dark = alpha & (ndimage.gaussian_filter(darkish, sigma * 0.7) > 0.45)

    # flat marker colors: k-means on the non-dark subject, then lighten; dark pixels take the
    # color of the nearest non-dark region so the fill has no holes
    fill_px = alpha & ~dark & (lum >= 0.15)
    if fill_px.sum() < 10:
        fill_px = alpha
    px = rgb[fill_px]
    k = min(colors, max(1, len(np.unique((px * 16).astype(int), axis=0))))
    km = KMeans(k, n_init=3, random_state=seed).fit(px[:: max(1, len(px) // 4000)])
    labels = np.full(alpha.shape, -1)
    labels[fill_px] = km.predict(px)
    labels = ndimage.median_filter(labels, size=max(3, int(sigma * 3)) | 1)
    _, (iy, ix) = ndimage.distance_transform_edt(labels < 0, return_indices=True)
    labels = labels[iy, ix]
    labels[~alpha] = -1
    palette = km.cluster_centers_ * 255
    palette = palette * 0.75 + 255 * 0.25  # lighter, like markers on paper

    # lines: the silhouette (slightly thicker) and the sprite's dark details
    outer = alpha ^ ndimage.binary_erosion(alpha, iterations=1)
    lw = max(1, size // 128)
    lines = ndimage.binary_dilation(outer, iterations=lw) | dark
    lab, n = ndimage.label(lines)  # drop specks
    if n:
        sizes = ndimage.sum(lines, lab, range(1, n + 1))
        lines &= np.isin(lab, 1 + np.nonzero(sizes >= (size // 40))[0])

    # wobble: lines and fill bend with smooth noise; fill also shifted off the lines
    dy, dx = (_smooth_field(alpha.shape, size // 6, wobble, rng) for _ in range(2))
    lines_w = _warp(lines.astype(np.float32), dy, dx) > 0.5
    shift = rng.normal(size=2) * offset
    fill_lab = _warp(labels.astype(np.float32), dy + shift[0], dx + shift[1], order=0).astype(int)

    out = np.tile(PAPER, (*alpha.shape, 1))
    for c in range(k):
        out[fill_lab == c] = palette[c]
    out[lines_w] = INK
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
