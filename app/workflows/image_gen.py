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
from app.postprocess import describe_failure, maybe_force_white_bg, pose_diagnostics
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
    "Nothing but the sock itself should be visible. "
    "The sock MUST stand upright vertically with the cuff at the top and the toe pointing "
    "horizontally to one side, filling at least 70% of the image height. Do NOT tilt, "
    "rotate, lay the sock down on its side, show it at an angle, or zoom out so it appears "
    "tiny in the frame."
)

RETRY_ADDENDUM = (
    "\n\nCRITICAL RETRY: the previous attempt produced an invalid image "
    "({failure}). The output MUST show the entire sock standing upright, "
    "cuff at the top, filling most of the frame. Do not erase the sock, "
    "do not lay it horizontally, do not tilt it, do not show a mostly-empty canvas."
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
    retry_reason: Optional[str] = None,
    extra_prompt: Optional[str] = None,
) -> dict:
    """Execute the workflow for one mockup. Returns dict with image_url + debug info.

    If `retry_reason` is set, the caller is driving a retry (e.g. the orchestrator
    after an evaluator-flagged failure). The reason is prepended to the prompt
    and the workflow's internal bbox-retry is skipped (caller decides).
    """
    colors = colors or []
    log.info("🧵 workflow start client=%s mock=%d pre_crop=%s retry=%s",
             client_name, mockup_index, pre_crop, bool(retry_reason))

    logo_bytes = _load_logo(logo_path=logo_path, logo_url=logo_url)

    mock = get_mockup(mockup_index)
    mock_bytes = mock.path.read_bytes()
    log.info("🧵 mockup loaded index=%d file=%s bytes=%d", mock.index, mock.path.name, len(mock_bytes))
    if pre_crop:
        mock_bytes = crop_to_subject(mock_bytes)

    prompt = _build_prompt(client_name, notes, colors)
    if extra_prompt:
        prompt = prompt + " " + extra_prompt
    if retry_reason:
        prompt = prompt + RETRY_ADDENDUM.format(failure=retry_reason)

    provider = get_image_gen()
    if retry_reason:
        # caller-driven retry — single pass, no internal bbox-retry
        png = provider.generate(prompt, reference_images=[logo_bytes, mock_bytes])
        png = maybe_force_white_bg(png)
        diag = pose_diagnostics(png)
        diag["final_failure"] = describe_failure(diag)
        attempts = 1
    else:
        png, diag, attempts = _generate_with_retry(provider, prompt, [logo_bytes, mock_bytes])
    log.info("🧵 image generated provider=%s bytes=%d attempts=%d", provider.name, len(png), attempts)

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
        "diagnostics": diag,
        "attempts": attempts,
    }


def _generate_with_retry(provider, prompt: str, refs: list[bytes], max_retries: int = 1) -> tuple[bytes, dict, int]:
    """Generate, run pose diagnostics, retry once with stricter prompt if the
    output is erased or laid horizontally. Returns (final_png_bytes, diag, attempts)."""
    png = provider.generate(prompt, reference_images=refs)
    png = maybe_force_white_bg(png)
    diag = pose_diagnostics(png)
    attempts = 1
    fail = describe_failure(diag)
    if fail and attempts <= max_retries:
        log.warning("🔁 retry triggered: %s", fail)
        retry_prompt = prompt + RETRY_ADDENDUM.format(failure=fail)
        png2 = provider.generate(retry_prompt, reference_images=refs)
        png2 = maybe_force_white_bg(png2)
        diag2 = pose_diagnostics(png2)
        attempts = 2
        fail2 = describe_failure(diag2)
        if fail2:
            log.warning("🔁 retry still failing: %s — returning retry output anyway", fail2)
        png, diag = png2, diag2
    diag["final_failure"] = describe_failure(diag)
    return png, diag, attempts


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")
