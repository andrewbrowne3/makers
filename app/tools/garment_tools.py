"""Tools that agents can call via ACT: lines.

All tool signatures accept only keyword args (so the regex-based parser works).
Tools may optionally accept `_ctx` (dict) for shared request state.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import httpx
from PIL import Image

from app.assets import get_mockup, list_mockups
from app.logging_config import get_logger
from app.providers.image_gen import get_image_gen
from app.s3 import get_s3

log = get_logger("tools")


def fetch_reference(*, url: str, _ctx: Optional[dict[str, Any]] = None) -> str:
    """Download a reference image (e.g. the client logo) by URL."""
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.content
    if _ctx is not None:
        refs: list[bytes] = _ctx.setdefault("references", [])
        refs.append(data)
    return f"fetched {len(data)} bytes from {url}"


def load_local_image(*, path: str, _ctx: Optional[dict[str, Any]] = None) -> str:
    """Load a local image file (logo or mockup) into the reference set."""
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    data = p.read_bytes()
    if _ctx is not None:
        refs: list[bytes] = _ctx.setdefault("references", [])
        refs.append(data)
    return f"loaded {len(data)} bytes from {p}"


def load_mockup(*, index: str, _ctx: Optional[dict[str, Any]] = None) -> str:
    """Load one of the 10 Crew Mock PNGs as a reference image (index 1..10)."""
    try:
        mock = get_mockup(int(index))
    except (KeyError, ValueError) as e:
        return f"ERROR: {e}"
    data = mock.path.read_bytes()
    if _ctx is not None:
        refs: list[bytes] = _ctx.setdefault("references", [])
        refs.append(data)
    return f"loaded mockup #{mock.index} ({mock.path.name}, {len(data)} bytes)"


def list_available_mockups(*, _ctx: Optional[dict[str, Any]] = None) -> str:
    """List the mockup indices available on disk."""
    mocks = list_mockups()
    return ", ".join(f"#{m.index}:{m.path.name}" for m in mocks) or "no mockups found"


def gen_image(*, prompt: str, _ctx: Optional[dict[str, Any]] = None) -> str:
    """Generate a garment design image from a text prompt (+ any fetched references)."""
    refs = (_ctx or {}).get("references") or []
    provider = get_image_gen()
    png = provider.generate(prompt, reference_images=refs)
    if _ctx is not None:
        _ctx["last_image"] = png
    return f"generated {len(png)} bytes via {provider.name}:{provider.model}"


def upload_s3(*, filename: str = "design.png", subpath: str = "designs", _ctx: Optional[dict[str, Any]] = None) -> str:
    """Upload the most recently generated image to S3 and return a presigned URL."""
    data = (_ctx or {}).get("last_image")
    if not data:
        return "ERROR: no image in context — call gen_image first"
    s3 = get_s3()
    key = s3.upload_bytes(data, filename=filename, subpath=subpath)
    url = s3.presigned_download(key)
    if _ctx is not None:
        _ctx["s3_key"] = key
        _ctx["image_url"] = url
    return f"uploaded key={key} url={url}"


def pillow_composite(
    *,
    template_path: str,
    logo_url: str,
    x: str = "0",
    y: str = "0",
    width: str = "300",
    _ctx: Optional[dict[str, Any]] = None,
) -> str:
    """Deterministic Pillow composite: overlay a logo onto a template at given coords.

    Coords are strings because the regex tool parser only carries quoted values.
    """
    template = Image.open(template_path).convert("RGBA")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        r = client.get(logo_url)
        r.raise_for_status()
        logo = Image.open(io.BytesIO(r.content)).convert("RGBA")

    w = int(width)
    ratio = w / logo.width
    logo = logo.resize((w, int(logo.height * ratio)))

    canvas = template.copy()
    canvas.alpha_composite(logo, (int(x), int(y)))

    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    if _ctx is not None:
        _ctx["last_image"] = buf.getvalue()
    return f"composited template={template_path} logo={logo_url} at=({x},{y}) w={width}"
