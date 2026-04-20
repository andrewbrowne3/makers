"""Evaluator — scores a rendered image on a rubric.

Uses the LLM's multimodal input (Gemma 4, Claude, GPT-4V) to look at the
image and return a structured score.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from app.config import CFG
from app.logging_config import get_logger
from app.providers.llm import get_llm

log = get_logger("agent.evaluator")

RUBRIC = """Score each criterion from 0.0 to 1.0 then return JSON (and nothing else — no prose, no code fences):
  logo_readable: the logo is crisp, legible, and recognizable
  placement_correct: the logo is in the expected garment region
  color_palette: the brand colors are faithfully rendered
  no_distortion: no warping, artifacts, or weird textures
  pose_vertical: the sock stands upright with the cuff at the top and the toe pointing sideways (0.0 = tilted/horizontal, 1.0 = perfectly upright)
  sock_visible: a recognizable, full-size sock fills the majority of the image (0.0 = missing/tiny/erased, 1.0 = fills frame)
  composition_pure: the sock is the ONLY subject — no floating text, no poster graphics, no brand wordmarks or logo art OUTSIDE the sock's surface, no additional objects anywhere in the frame (0.0 = image contains anything beyond the sock itself, 1.0 = sock is the lone element on pure white)
  overall: the average of all seven criteria above
  notes: ONE short sentence under 120 characters. Do not repeat words. No markdown.
Return exactly this JSON schema (no markdown fences):
{"logo_readable": 0.0, "placement_correct": 0.0, "color_palette": 0.0, "no_distortion": 0.0, "pose_vertical": 0.0, "sock_visible": 0.0, "composition_pure": 0.0, "overall": 0.0, "notes": "..."}
"""


class EvaluatorAgent:
    name = "evaluator_agent"

    def __init__(self) -> None:
        self.llm = get_llm()

    def score(self, image_bytes: bytes, brief: str) -> dict:
        messages = [
            {"role": "system", "content": "You are a QA evaluator for garment design mockups."},
            {"role": "user", "content": f"Brief: {brief}\n\n{RUBRIC}"},
        ]
        log.info("📋 evaluator.score brief_len=%d", len(brief))
        raw = self.llm.chat(messages, images=[image_bytes])
        return _extract_json(raw) or {"overall": 0.0, "notes": raw, "parse_error": True}


def _extract_json(text: str) -> Optional[dict]:
    """Robust JSON extraction that tolerates ```json fences, trailing prose,
    truncated repetition loops inside string fields, and greedy-match fallback."""
    # strip common markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # first try: greedy {...} match
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # second try: find the opening brace and attempt to parse progressively
    # smaller tails, stopping at the largest valid JSON
    start = cleaned.find("{")
    if start >= 0:
        # walk back from end, trim tokens from notes field if needed
        # try to fix common "notes" token repetition ("mockup mockup mockup...")
        # by truncating anything >200 chars in the notes field
        candidate = cleaned[start:]
        candidate = re.sub(
            r'("notes"\s*:\s*")([^"]{200,})(")',
            lambda m: m.group(1) + m.group(2)[:120].rstrip() + m.group(3),
            candidate,
            count=1,
        )
        # greedy close
        m2 = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except json.JSONDecodeError:
                pass

    return None


def build_evaluator() -> Optional[EvaluatorAgent]:
    if not CFG.evaluator.enabled:
        return None
    return EvaluatorAgent()
