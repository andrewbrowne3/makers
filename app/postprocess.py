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
