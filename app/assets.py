"""Filesystem asset catalogue for Executive Crew Socks.

Lists available mockup PNGs and TRAINING_DATA pairs so agents / tools
can reference them without hard-coding paths.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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
