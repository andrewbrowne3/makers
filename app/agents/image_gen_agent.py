"""Agent for the image_gen branch — applies a client logo to a sock mockup.

Flow per request: logo + mockup PNG → image-edit model replaces the
"LogoHere" placeholder region with the client's logo, preserving fabric
texture and 3D shading.
"""
from __future__ import annotations

from app.agents.base import ReactAgent
from app.providers.llm import get_llm
from app.tools.garment_tools import (
    fetch_reference,
    gen_image,
    list_available_mockups,
    load_local_image,
    load_mockup,
    upload_s3,
)


class ImageGenAgent(ReactAgent):
    name = "image_gen_agent"
    max_steps = 8
    min_tool_calls = 4  # logo load + mockup load + gen_image + upload_s3
    system_prompt = (
        "You generate Executive Crew Sock mockups by replacing the \"LogoHere\" placeholder "
        "on a 3D sock mockup PNG with a client's logo.\n\n"
        "You MUST call tools IN ORDER. Do NOT fabricate URLs. Do NOT call FINAL before upload_s3 returns.\n\n"
        "Required sequence (4 tool calls, exactly in this order):\n"
        "Step 1 — ACT: load_local_image(path=\"<logo_path>\")   (or fetch_reference(url=\"<logo_url>\") if url given)\n"
        "Step 2 — ACT: load_mockup(index=\"<N>\")\n"
        "Step 3 — ACT: gen_image(prompt=\"Replace the LogoHere placeholder on this sock with the attached client logo. Preserve the knit fabric texture, 3D shading, sock ribbing, and original colors. Blend the logo so it looks woven into the sock, not pasted on top.\")\n"
        "Step 4 — ACT: upload_s3(filename=\"<client>_mock<N>.png\")\n"
        "Step 5 — FINAL: <the exact presigned URL returned by upload_s3 — copy it verbatim from the OBSERVE line>\n\n"
        "Rules: use the EXACT path/URL given in the user message. Do NOT invent URLs, filenames, or paths. "
        "Wait for each OBSERVE before issuing the next ACT."
    )


def build_image_gen_agent() -> ImageGenAgent:
    return ImageGenAgent(
        llm=get_llm(),
        tools={
            "fetch_reference": fetch_reference,
            "load_local_image": load_local_image,
            "load_mockup": load_mockup,
            "list_available_mockups": list_available_mockups,
            "gen_image": gen_image,
            "upload_s3": upload_s3,
        },
    )
