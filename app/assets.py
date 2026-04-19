"""Filesystem asset catalogue for Executive Crew Socks.

Lists available mockup PNGs and TRAINING_DATA pairs so agents / tools
can reference them without hard-coding paths.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops

from app.config import CFG
from app.logging_config import get_logger

log = get_logger("assets")

MOCK_RE = re.compile(r"LogoHere_Crew_Mock_(\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Mockup:
    index: int
    path: Path


@dataclass(frozen=True)
class TrainingPair:
    client: str
    logo_paths: list[Path]
    output_paths: list[Path]


def list_mockups() -> list[Mockup]:
    root = Path(CFG.assets.mock_dir)
    if not root.exists():
        log.warning("⚠️  mock_dir missing: %s", root)
        return []
    out: list[Mockup] = []
    for p in sorted(root.iterdir()):
        m = MOCK_RE.search(p.name)
        if m:
            out.append(Mockup(index=int(m.group(1)), path=p))
    out.sort(key=lambda x: x.index)
    return out


def get_mockup(index: int) -> Mockup:
    mocks = {m.index: m for m in list_mockups()}
    if index not in mocks:
        raise KeyError(f"mockup {index} not found (available: {sorted(mocks)})")
    return mocks[index]


def crop_to_subject(png_bytes: bytes, padding_pct: float = 0.08) -> bytes:
    """Crop a mockup image to the tight bounding box of its subject (non-background pixels).

    Uses the top-left corner pixel as the reference background color. Pure Pillow,
    deterministic, no ML. Returns the original bytes if no subject is detected.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    bg_color = img.getpixel((0, 0))
    bg = Image.new("RGB", img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        log.info("✂️  crop_to_subject: no subject detected, returning original")
        return png_bytes

    w, h = img.size
    left, top, right, bottom = bbox
    pad_x = int((right - left) * padding_pct)
    pad_y = int((bottom - top) * padding_pct)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(w, right + pad_x)
    bottom = min(h, bottom + pad_y)

    cropped = img.crop((left, top, right, bottom))
    out = io.BytesIO()
    cropped.save(out, format="PNG")
    log.info(
        "✂️  crop_to_subject orig=%dx%d → %dx%d (bg=%s)",
        w, h, cropped.width, cropped.height, bg_color,
    )
    return out.getvalue()


def list_training_pairs() -> list[TrainingPair]:
    root = Path(CFG.assets.training_dir)
    if not root.exists():
        log.warning("⚠️  training_dir missing: %s", root)
        return []
    pairs: list[TrainingPair] = []
    for client_dir in sorted(root.iterdir()):
        if not client_dir.is_dir():
            continue
        logos_dir = client_dir / "logos and design assets"
        outputs_dir = client_dir / "output designs"
        logos = sorted(p for p in logos_dir.glob("*") if p.is_file()) if logos_dir.exists() else []
        outputs = sorted(p for p in outputs_dir.glob("*") if p.is_file()) if outputs_dir.exists() else []
        pairs.append(
            TrainingPair(
                client=client_dir.name.removeprefix("https_--").removesuffix("/"),
                logo_paths=logos,
                output_paths=outputs,
            )
        )
    return pairs
