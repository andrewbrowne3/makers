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

RUBRIC = """Score each criterion from 0.0 to 1.0 then return JSON:
  logo_readable: the logo is crisp, legible, and recognizable
  placement_correct: the logo is in the expected garment region
  color_palette: the brand colors are faithfully rendered
  no_distortion: no warping, artifacts, or weird textures
  pose_vertical: the sock stands upright with the cuff at the top and the toe pointing sideways — NOT tilted, rotated, or laid on its side (0.0 if tilted/horizontal, 1.0 if perfectly upright)
  sock_visible: a recognizable, full-size sock occupies the majority of the image — NOT erased, missing, tiny, or cropped (0.0 if the sock is missing or nearly invisible, 1.0 if it fills the frame well)
overall: the average of all six criteria above.
Return exactly this JSON schema:
  {"logo_readable": 0.0, "placement_correct": 0.0, "color_palette": 0.0, "no_distortion": 0.0, "pose_vertical": 0.0, "sock_visible": 0.0, "overall": 0.0, "notes": "..."}
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
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def build_evaluator() -> Optional[EvaluatorAgent]:
    if not CFG.evaluator.enabled:
        return None
    return EvaluatorAgent()
