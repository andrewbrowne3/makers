"""Image-generation provider abstraction — swap via IMAGE_GEN_PROVIDER env.

Providers all expose `.generate(prompt, reference_images=None) -> bytes` returning PNG bytes.

Defaults / candidates:
  - nano_banana  → Google Gemini 2.5 Flash Image
  - gpt_image_1  → OpenAI gpt-image-1
  - flux_kontext → Black Forest Labs (via fal/replicate)
  - stub         → local Pillow placeholder for dev without API keys
"""
from __future__ import annotations

import io
from typing import Optional, Protocol

from PIL import Image, ImageDraw, ImageFont

from app.config import CFG
from app.logging_config import get_logger

log = get_logger("image_gen")


class ImageGenProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, reference_images: Optional[list[bytes]] = None) -> bytes: ...


class StubGen:
    """Placeholder that renders the prompt as text on a canvas — for dev without API keys."""

    def __init__(self, model: str) -> None:
        self.name = "stub"
        self.model = model

    def generate(self, prompt: str, reference_images: Optional[list[bytes]] = None) -> bytes:
        log.info("🖼️  stub.generate prompt_len=%d refs=%d", len(prompt), len(reference_images or []))
        img = Image.new("RGB", (1024, 1024), color=(235, 235, 245))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        wrapped = _wrap(prompt, 50)
        draw.multiline_text((30, 30), f"[STUB IMAGE]\n\n{wrapped}", fill=(20, 20, 30), font=font, spacing=6)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


class NanoBananaGen:
    def __init__(self, model: str, api_key: str) -> None:
        self.name = "nano_banana"
        self.model = model
        self._api_key = api_key

    def generate(self, prompt: str, reference_images: Optional[list[bytes]] = None) -> bytes:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        parts: list = [types.Part.from_text(text=prompt)]
        for img in reference_images or []:
            parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))
        log.info("🖼️  nano_banana.generate model=%s prompt_len=%d refs=%d", self.model, len(prompt), len(reference_images or []))
        resp = client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
        )
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                return part.inline_data.data  # type: ignore[return-value]
        raise RuntimeError("nano_banana returned no image")


class GPTImage1Gen:
    def __init__(self, model: str, api_key: str) -> None:
        self.name = "gpt_image_1"
        self.model = model
        self._api_key = api_key

    def generate(self, prompt: str, reference_images: Optional[list[bytes]] = None) -> bytes:
        import base64 as _b64

        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        log.info("🖼️  gpt_image_1.generate model=%s prompt_len=%d refs=%d", self.model, len(prompt), len(reference_images or []))
        if reference_images:
            files = [("image[]", (f"ref{i}.png", img, "image/png")) for i, img in enumerate(reference_images)]
            resp = client.images.edit(model=self.model, prompt=prompt, image=files[0][1][1] if len(files) == 1 else [f[1][1] for f in files])  # type: ignore[arg-type]
        else:
            resp = client.images.generate(model=self.model, prompt=prompt)
        b64 = resp.data[0].b64_json  # type: ignore[index,union-attr]
        if not b64:
            raise RuntimeError("gpt_image_1 returned no b64 data")
        return _b64.b64decode(b64)


def get_image_gen() -> ImageGenProvider:
    cfg = CFG.image_gen
    if cfg.provider == "stub":
        return StubGen(cfg.model)
    if cfg.provider == "nano_banana":
        return NanoBananaGen(cfg.model, CFG.llm.google_api_key)
    if cfg.provider == "gpt_image_1":
        return GPTImage1Gen(cfg.model, CFG.llm.openai_api_key)
    if cfg.provider == "flux_kontext":
        raise NotImplementedError("flux_kontext provider not wired yet — add fal/replicate client")
    raise ValueError(f"Unknown IMAGE_GEN_PROVIDER: {cfg.provider}")


def _wrap(text: str, width: int) -> str:
    lines: list[str] = []
    for raw in text.splitlines() or [text]:
        if len(raw) <= width:
            lines.append(raw)
            continue
        cur = ""
        for word in raw.split(" "):
            if len(cur) + len(word) + 1 > width:
                lines.append(cur.rstrip())
                cur = word + " "
            else:
                cur += word + " "
        if cur:
            lines.append(cur.rstrip())
    return "\n".join(lines)
