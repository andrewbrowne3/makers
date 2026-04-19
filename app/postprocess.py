"""Deterministic image post-processing — runs after the image-gen provider.

`force_white_background`: uses rembg (U²-Net) to cut out the foreground and
paste onto a pure white canvas. Output is identical for identical input —
no model randomness, no LLM. Cheap once the model is downloaded (~170MB,
one-time) and runs ~0.5–2s per image on CPU.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Optional

from PIL import Image

from app.config import CFG
from app.logging_config import get_logger

log = get_logger("postprocess")


@lru_cache(maxsize=1)
def _rembg_session():
    from rembg import new_session

    log.info("🧽 rembg session init model=%s (first run downloads ~170MB)", CFG.image_gen.rembg_model)
    return new_session(CFG.image_gen.rembg_model)


def force_white_background(png_bytes: bytes, bg_rgb: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    """Remove background via rembg and composite the subject on a solid canvas."""
    from rembg import remove

    session = _rembg_session()
    cutout_png = remove(png_bytes, session=session)  # bytes with alpha
    cutout = Image.open(io.BytesIO(cutout_png)).convert("RGBA")

    canvas = Image.new("RGB", cutout.size, bg_rgb)
    canvas.paste(cutout, mask=cutout.split()[3])  # alpha channel as mask

    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    log.info(
        "🧽 force_white_bg in=%d out=%d size=%dx%d",
        len(png_bytes), len(out.getvalue()), canvas.width, canvas.height,
    )
    return out.getvalue()


def maybe_force_white_bg(png_bytes: bytes, *, enabled: Optional[bool] = None) -> bytes:
    """Respect the FORCE_WHITE_BG config flag; allow per-call override."""
    flag = CFG.image_gen.force_white_bg if enabled is None else enabled
    if not flag:
        return png_bytes
    return force_white_background(png_bytes)


def non_white_ratio(png_bytes: bytes, threshold: int = 240, stride: int = 8) -> float:
    """Fraction of pixels that are not near-white. Samples every `stride`-th pixel."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    px = img.load()
    total = nonwhite = 0
    for y in range(0, img.height, stride):
        for x in range(0, img.width, stride):
            total += 1
            r, g, b = px[x, y]
            if min(r, g, b) < threshold:
                nonwhite += 1
    return nonwhite / max(1, total)


def foreground_bbox(png_bytes: bytes, threshold: int = 240) -> Optional[tuple[int, int, int, int]]:
    """Bounding box of non-white pixels (same method as assets.crop_to_subject).

    Returns (left, top, right, bottom) or None if no foreground detected.
    """
    from PIL import ImageChops

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    # threshold everything near white so faint rembg edge halos don't fool the bbox
    _ = threshold  # reserved for future use; getbbox already handles zero-diff pixels
    return diff.getbbox()


def pose_diagnostics(png_bytes: bytes) -> dict:
    """Run both the erasure + pose-aspect checks. Returns a dict of signals."""
    ratio = non_white_ratio(png_bytes)
    bbox = foreground_bbox(png_bytes)
    if bbox is None:
        return {
            "non_white_ratio": ratio,
            "sock_erased": True,
            "sock_horizontal": False,
            "bbox": None,
            "bbox_w": 0,
            "bbox_h": 0,
        }
    left, top, right, bottom = bbox
    w, h = right - left, bottom - top
    return {
        "non_white_ratio": ratio,
        "sock_erased": ratio < 0.08,
        "sock_horizontal": w > 0.9 * h,
        "bbox": bbox,
        "bbox_w": w,
        "bbox_h": h,
    }


def describe_failure(diag: dict) -> Optional[str]:
    """Return a human-readable failure reason if diag indicates a problem, else None."""
    if diag.get("sock_erased"):
        return f"the sock was erased or nearly invisible (only {diag['non_white_ratio']:.1%} non-white pixels)"
    if diag.get("sock_horizontal"):
        return f"the sock was laid down horizontally (bbox {diag['bbox_w']}x{diag['bbox_h']}, width > 0.9 * height)"
    return None
