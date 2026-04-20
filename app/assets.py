"""Filesystem asset catalogue for Executive Crew Socks.

Lists available mockup PNGs and TRAINING_DATA pairs so agents / tools
can reference them without hard-coding paths.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import lru_cache
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


def _pose_profile(png_bytes: bytes) -> dict:
    """Compute a pose profile: bbox, aspect ratio, toe_side (left|right), frame_fill.

    toe_side heuristic: within the bottom 40% of the sock bbox, count non-white pixels
    left vs right of the bbox's horizontal midpoint. The heavier side is where the toe
    (foot) extends — typical sock photos show the foot bending away from the ankle line.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return {
            "bbox": None, "bbox_w": 0, "bbox_h": 0,
            "aspect": 0.0, "toe_side": None, "frame_fill": 0.0,
        }
    left, top, right, bottom = bbox
    w, h = right - left, bottom - top

    # Count non-white pixels in the bottom 40% of the bbox, left vs right of center
    foot_top = top + int(h * 0.60)
    mid_x = left + w // 2
    diff_rgb = diff.convert("L")  # luminance difference; 0 = identical to white
    left_count = right_count = 0
    for y in range(foot_top, bottom, 4):
        for x in range(left, right, 4):
            if diff_rgb.getpixel((x, y)) > 15:  # noise floor
                if x < mid_x:
                    left_count += 1
                else:
                    right_count += 1
    toe_side = None
    if left_count + right_count > 20:  # enough signal to decide
        toe_side = "right" if right_count > left_count else "left"

    frame_fill = (w * h) / max(1, img.width * img.height)
    return {
        "bbox": bbox,
        "bbox_w": w,
        "bbox_h": h,
        "aspect": h / max(1, w),
        "toe_side": toe_side,
        "frame_fill": round(frame_fill, 3),
        "image_size": (img.width, img.height),
    }


@lru_cache(maxsize=16)
def mockup_pose_profile(index: int) -> dict:
    """Cached pose profile for one of the 10 source mockup templates."""
    mock = get_mockup(index)
    data = mock.path.read_bytes()
    profile = _pose_profile(data)
    log.info(
        "📐 mockup #%d profile: aspect=%.2f toe=%s fill=%.2f bbox=%sx%s",
        index, profile["aspect"], profile["toe_side"], profile["frame_fill"],
        profile["bbox_w"], profile["bbox_h"],
    )
    return profile


def pose_profile_from_bytes(png_bytes: bytes) -> dict:
    """Compute the pose profile of an arbitrary PNG (e.g. an output render)."""
    return _pose_profile(png_bytes)


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
