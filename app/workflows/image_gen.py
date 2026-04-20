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

from app.assets import crop_to_subject, get_mockup, mockup_pose_profile
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
    "COMPOSITION RULE — the sock is the ONLY element in the image. Absolutely nothing else "
    "appears anywhere in the frame: "
    "   - NO floating or oversized text beside or behind the sock "
    "   - NO poster layouts, advertising compositions, promotional overlays, or marketing graphics "
    "   - NO brand wordmarks, typography, slogans, or logos rendered outside the sock's surface "
    "   - NO additional socks, product variants, or comparison shots "
    "   - NO display stand, sock form, mannequin, hanger, hook, wire frame, armature, box, pedestal "
    "   - NO shadow, reflection, gradient, studio floor, or background texture "
    "Every logo or typography element must live ON the sock fabric itself, never floating in empty space. "
    "POSE LOCK — match the reference mockup's pose exactly: same camera angle, same tilt, "
    "same toe direction, same sock-to-frame proportion. If the reference shows the toe "
    "pointing right, the output's toe must point right. If the reference fills ~35% of the "
    "frame, the output must fill the same. Do NOT zoom in, zoom out, rotate, mirror, or "
    "reframe the sock — treat the reference mockup's pose as a rigid constraint. Only the "
    "colors, patterns, and logo content change; orientation and composition stay identical."
)


def _pose_hint(profile: dict) -> str:
    """Build a short reference-specific pose addendum from a mockup's profile."""
    toe = profile.get("toe_side")
    aspect = profile.get("aspect") or 0.0
    fill = profile.get("frame_fill") or 0.0
    bits = []
    if toe:
        bits.append(f"toe pointing {toe}")
    if aspect > 0:
        bits.append(f"bbox aspect (height/width) ≈ {aspect:.2f}")
    if fill > 0:
        bits.append(f"frame fill ≈ {int(fill * 100)}%")
    if not bits:
        return ""
    return " Reference-specific constraints: " + ", ".join(bits) + "."

RETRY_ADDENDUM = (
    "\n\nCRITICAL RETRY (attempt 2): the previous attempt produced an invalid image "
    "({failure}). The output MUST show the entire sock standing upright, "
    "cuff at the top, filling most of the frame. Do not erase the sock, "
    "do not lay it horizontally, do not tilt it, do not show a mostly-empty canvas."
)

RETRY_ADDENDUM_ESCALATED = (
    "\n\nFINAL RETRY (attempt 3): the previous attempts ALL produced invalid images "
    "({failure}). Render ONLY the sock on pure white — nothing else. No floating text, "
    "no poster graphics, no brand overlays, no background elements, no additional objects. "
    "Match the reference mockup's exact pose and size. If you cannot do this cleanly, "
    "output a simple plain sock with the logo centered on the shin — err on the side of "
    "boring and clean over creative and wrong."
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

    source_profile = mockup_pose_profile(mockup_index)
    prompt = _build_prompt(client_name, notes, colors) + _pose_hint(source_profile)
    if extra_prompt:
        prompt = prompt + " " + extra_prompt
    if retry_reason:
        prompt = prompt + RETRY_ADDENDUM.format(failure=retry_reason)

    provider = get_image_gen()
    if retry_reason:
        # caller-driven retry — single pass, no internal bbox-retry
        png = provider.generate(prompt, reference_images=[logo_bytes, mock_bytes])
        png = maybe_force_white_bg(png)
        diag = pose_diagnostics(png, source_profile=source_profile)
        diag["final_failure"] = describe_failure(diag)
        attempts = 1
    else:
        png, diag, attempts = _generate_with_retry(
            provider, prompt, [logo_bytes, mock_bytes], source_profile=source_profile,
        )
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


EXTREME_FILL_DRIFT = 0.90  # 2× source fill → hard-fail territory


def _generate_with_retry(
    provider, prompt: str, refs: list[bytes], max_retries: int = 2,
    source_profile: Optional[dict] = None,
) -> tuple[bytes, dict, int]:
    """Generate, run pose diagnostics, retry up to `max_retries` times with escalating
    prompts if the output is erased, laid horizontally, or orientation drifts from the
    source. Returns (final_png_bytes, diag, attempts).

    API exceptions (e.g. 400 INVALID_ARGUMENT transient failures) also trigger a retry
    using the next escalation addendum."""
    current_png, current_diag, attempts = None, None, 0
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):  # initial + up to max_retries retries
        if attempt == 1:
            retry_prompt = prompt
        else:
            reason = (
                describe_failure(current_diag) if current_diag
                else f"the provider raised {type(last_error).__name__}: {str(last_error)[:200]}"
            )
            addendum = RETRY_ADDENDUM if attempt == 2 else RETRY_ADDENDUM_ESCALATED
            retry_prompt = prompt + addendum.format(failure=reason)
            log.warning("🔁 retry %d triggered: %s", attempt - 1, reason)
        try:
            png = provider.generate(retry_prompt, reference_images=refs)
            png = maybe_force_white_bg(png)
            diag = pose_diagnostics(png, source_profile=source_profile)
            current_png, current_diag = png, diag
            attempts = attempt
            if not describe_failure(diag):
                break
        except Exception as e:  # noqa: BLE001
            log.warning("🔁 provider raised on attempt %d: %s", attempt, type(e).__name__)
            last_error = e
            attempts = attempt
            continue
    if current_png is None:
        # Every attempt failed at the API level; re-raise the last error
        raise last_error if last_error else RuntimeError("all provider attempts failed")

    # Drop through to the original tail logic
    final_fail = describe_failure(current_diag)
    current_diag["final_failure"] = final_fail
    current_diag["critical_failure"] = bool(
        final_fail and current_diag.get("fill_drift", 0) > EXTREME_FILL_DRIFT
    )
    if final_fail:
        log.warning("🔁 still failing after %d attempts: %s (critical=%s)",
                    attempts, final_fail, current_diag["critical_failure"])
    return current_png, current_diag, attempts


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")
