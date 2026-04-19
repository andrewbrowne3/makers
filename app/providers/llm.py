"""LLM provider abstraction — swap via LLM_PROVIDER env.

Default: Ollama + gemma4:e4b. Alternates: Anthropic, OpenAI, Google.
All providers expose the same `.chat(messages, images=None) -> str` interface.
"""
from __future__ import annotations

import base64
from typing import Optional, Protocol

import httpx

from app.config import CFG
from app.logging_config import get_logger

log = get_logger("llm")


Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": "..."}


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[Message], images: Optional[list[bytes]] = None) -> str: ...


class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        self.name = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[Message], images: Optional[list[bytes]] = None) -> str:
        payload: dict = {"model": self.model, "messages": list(messages), "stream": False}
        if images:
            encoded = [base64.b64encode(img).decode("ascii") for img in images]
            if payload["messages"]:
                payload["messages"][-1] = {
                    **payload["messages"][-1],
                    "images": encoded,
                }
        log.info("🧠 ollama.chat model=%s msgs=%d images=%d", self.model, len(messages), len(images or []))
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return data.get("message", {}).get("content", "")


class AnthropicLLM:
    def __init__(self, model: str, api_key: str) -> None:
        from anthropic import Anthropic

        self.name = "anthropic"
        self.model = model
        self._client = Anthropic(api_key=api_key)

    def chat(self, messages: list[Message], images: Optional[list[bytes]] = None) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        convo = [m for m in messages if m["role"] != "system"]
        content: list = []
        if convo:
            tail = convo[-1]
            content.append({"type": "text", "text": tail["content"]})
            for img in images or []:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(img).decode("ascii"),
                        },
                    }
                )
            convo[-1] = {"role": tail["role"], "content": content}  # type: ignore[assignment]
        log.info("🧠 anthropic.chat model=%s msgs=%d images=%d", self.model, len(messages), len(images or []))
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system or None,  # type: ignore[arg-type]
            messages=convo,  # type: ignore[arg-type]
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAILLM:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.name = "openai"
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def chat(self, messages: list[Message], images: Optional[list[bytes]] = None) -> str:
        msgs: list[dict] = [dict(m) for m in messages]
        if images and msgs:
            last = msgs[-1]
            content: list[dict] = [{"type": "text", "text": last["content"]}]
            for img in images:
                b64 = base64.b64encode(img).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            msgs[-1] = {"role": last["role"], "content": content}  # type: ignore[assignment]
        log.info("🧠 openai.chat model=%s msgs=%d images=%d", self.model, len(messages), len(images or []))
        resp = self._client.chat.completions.create(model=self.model, messages=msgs)  # type: ignore[arg-type]
        return resp.choices[0].message.content or ""


def get_llm() -> LLMProvider:
    cfg = CFG.llm
    if cfg.provider == "ollama":
        return OllamaLLM(cfg.model, cfg.ollama_base_url)
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg.model, cfg.anthropic_api_key)
    if cfg.provider == "openai":
        return OpenAILLM(cfg.model, cfg.openai_api_key)
    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.provider}")
