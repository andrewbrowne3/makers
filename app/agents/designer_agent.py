"""Designer agent — proposes N creative directions for a given (client, logo, mockup).

Uses the local Gemma 4 E4B (multimodal) to look at the logo + mockup and suggest
different creative prompt addenda. Output is appended to the base workflow prompt,
so the base pose-lock / no-stands guardrails still apply.

Returns a list of Direction dicts with a short label and a prompt_addendum string.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from app.config import CFG
from app.logging_config import get_logger
from app.providers.llm import get_llm

log = get_logger("agent.designer")


DEFAULT_DIRECTIONS = [
    {
        "label": "classic",
        "prompt_addendum": (
            "Produce a classic, timeless design. Preserve the mockup's original colorway. "
            "Use subtle repeat patterns and a clean single logo patch placement near the shin."
        ),
    },
    {
        "label": "bold",
        "prompt_addendum": (
            "Produce a bold, high-contrast design. Amplify the brand's primary colors. "
            "Use a larger, more prominent logo and striking accent bands at the cuff and toe."
        ),
    },
    {
        "label": "minimalist",
        "prompt_addendum": (
            "Produce a minimalist design. Use a single small logo patch on the shin, "
            "plain solid-color sock body, and no repeat patterns. Restrained palette."
        ),
    },
    {
        "label": "playful",
        "prompt_addendum": (
            "Produce a playful design with a dense all-over logo repeat. "
            "Freely harmonize accent colors with the logo's palette."
        ),
    },
    {
        "label": "technical",
        "prompt_addendum": (
            "Produce a technical-athletic look with engineered-knit stripes, "
            "ribbed texture emphasis, and a small logo patch on the shin."
        ),
    },
]


DESIGNER_SYSTEM_PROMPT = """You are a senior sock designer for a B2B garment company.
Given a client logo and a sock mockup template, propose {n} distinct CREATIVE DIRECTIONS
for how the final sock should look. Each direction is a short label plus a one-sentence
prompt addendum describing the aesthetic. Cover different moods (e.g. classic, bold,
minimalist) so the client has real variety to choose from.

Return ONLY a JSON array of objects, no prose, no markdown fences:
[
  {{"label": "...", "prompt_addendum": "..."}},
  ...
]
Each label should be 1-2 words. Each prompt_addendum should be under 200 characters.
Do not repeat directions. Do not reference pose or background (the base prompt handles those)."""


@dataclass
class Direction:
    label: str
    prompt_addendum: str


def _parse_directions(raw: str) -> Optional[list[dict]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    if not isinstance(data, list):
        return None
    for d in data:
        if not isinstance(d, dict):
            continue
        label = str(d.get("label", "")).strip()
        add = str(d.get("prompt_addendum", "")).strip()
        if label and add:
            out.append({"label": label, "prompt_addendum": add})
    return out or None


def propose_directions(
    *,
    client_name: str,
    logo_bytes: bytes,
    mockup_bytes: bytes,
    n: int = 3,
) -> list[Direction]:
    """Ask the LLM to propose n creative directions. Falls back to DEFAULT_DIRECTIONS
    on parse failure or if the LLM returns fewer than n usable entries."""
    n = max(1, min(5, n))  # sensible bounds
    system = DESIGNER_SYSTEM_PROMPT.format(n=n)
    user_text = f"Client: {client_name}\nPropose {n} creative directions for this sock mockup."

    try:
        llm = get_llm()
        log.info("🎨 designer.propose n=%d client=%s", n, client_name)
        raw = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            images=[logo_bytes, mockup_bytes],
        )
        parsed = _parse_directions(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("designer llm call failed, using defaults: %s", e)
        parsed = None

    if not parsed or len(parsed) < n:
        log.info("🎨 designer falling back to built-in defaults (got %d, need %d)",
                 len(parsed or []), n)
        parsed = DEFAULT_DIRECTIONS[:n]

    return [Direction(label=d["label"], prompt_addendum=d["prompt_addendum"]) for d in parsed[:n]]


def should_use_designer(request_flag: Optional[bool]) -> bool:
    """Per-request flag overrides the env default."""
    if request_flag is not None:
        return bool(request_flag)
    return CFG.designer.enabled_by_default


def resolved_directions_count(request_count: Optional[int]) -> int:
    if request_count is not None:
        return max(1, min(5, int(request_count)))
    return CFG.designer.directions
