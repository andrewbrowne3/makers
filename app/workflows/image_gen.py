"""Deterministic image-gen workflow — NO LLM in the critical path.

For each requested mockup:
  1. Load the client logo (from filesystem path or URL).
  2. Load the mockup PNG by index.
  3. Call the image-gen provider with prompt + [logo, mockup] references.
  4. Upload the result to S3, return a presigned URL.

This path produces reliable, reproducible output. The router still decides
*whether* to take this path; the evaluator still scores the result. We only
removed the ReAct LLM from the sequencing step where it was hallucinating.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from app.assets import crop_to_subject, get_mockup
from app.logging_config import get_logger
from app.postprocess import maybe_force_white_bg
from app.providers.image_gen import get_image_gen
from app.s3 import get_s3

log = get_logger("workflow.image_gen")

DEFAULT_PROMPT = (
    "Replace the 'LogoHere' placeholder on this 3D sock mockup with the attached client logo. "
    "Preserve the knit fabric texture, 3D shading, sock ribbing, and original sock colors. "
    "Blend the logo so it looks woven or printed into the sock — not pasted flat on top. "
    "Output a single sock floating alone against a pure solid white (#FFFFFF) background. "
    "Do NOT add any of the following: display stand, sock form, mannequin, hanger, hook, "
    "wire frame, armature, box, pedestal, shadow, reflection, gradient, or studio floor. "
    "Nothing but the sock itself should be visible."
)


def _load_logo(*, logo_path: Optional[str], logo_url: Optional[str]) -> bytes:
    if logo_path:
        p = Path(logo_path)
        if not p.exists():
            raise FileNotFoundError(f"logo_path not found: {logo_path}")
        data = p.read_bytes()
        log.info("📁 logo loaded from disk path=%s bytes=%d", p, len(data))
        return data
    if logo_url:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            r = client.get(logo_url)
            r.raise_for_status()
        log.info("🌐 logo fetched url=%s bytes=%d", logo_url, len(r.content))
        return r.content
    raise ValueError("no logo_path or logo_url provided")


def _build_prompt(client_name: str, notes: Optional[str], colors: list[str]) -> str:
    extras = []
    if colors:
        extras.append(f"Client brand colors: {', '.join(colors)}.")
    if notes:
        extras.append(f"Designer notes: {notes}.")
    if extras:
        return DEFAULT_PROMPT + " " + " ".join(extras)
    return DEFAULT_PROMPT


def generate_one_mockup(
    *,
    client_name: str,
    logo_path: Optional[str],
    logo_url: Optional[str],
    mockup_index: int,
    colors: Optional[list[str]] = None,
    notes: Optional[str] = None,
    pre_crop: bool = True,
) -> dict:
    """Execute the workflow for one mockup. Returns dict with image_url + debug info."""
    colors = colors or []
    log.info("🧵 workflow start client=%s mock=%d pre_crop=%s", client_name, mockup_index, pre_crop)

    logo_bytes = _load_logo(logo_path=logo_path, logo_url=logo_url)

    mock = get_mockup(mockup_index)
    mock_bytes = mock.path.read_bytes()
    log.info("🧵 mockup loaded index=%d file=%s bytes=%d", mock.index, mock.path.name, len(mock_bytes))
    if pre_crop:
        mock_bytes = crop_to_subject(mock_bytes)

    prompt = _build_prompt(client_name, notes, colors)

    provider = get_image_gen()
    png = provider.generate(prompt, reference_images=[logo_bytes, mock_bytes])
    log.info("🧵 image generated provider=%s bytes=%d", provider.name, len(png))

    png = maybe_force_white_bg(png)

    s3 = get_s3()
    filename = f"{_slug(client_name)}_mock{mockup_index}.png"
    key = s3.upload_bytes(png, filename=filename, subpath="designs")
    url = s3.presigned_download(key)
    log.info("🧵 uploaded key=%s", key)

    return {
        "image_url": url,
        "s3_key": key,
        "prompt": prompt,
        "provider": f"{provider.name}:{provider.model}",
        "mockup_file": mock.path.name,
    }


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")
