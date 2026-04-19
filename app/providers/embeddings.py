"""Embedding provider — Ollama default (mxbai-embed-large), OpenAI alternate."""
from __future__ import annotations

from typing import Protocol

import httpx

from app.config import CFG
from app.logging_config import get_logger

log = get_logger("embeddings")


class EmbeddingProvider(Protocol):
    name: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbed:
    def __init__(self, model: str, base_url: str) -> None:
        self.name = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        with httpx.Client(timeout=60) as client:
            for t in texts:
                r = client.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": t})
                r.raise_for_status()
                out.append(r.json()["embedding"])
        log.info("🧬 ollama.embed model=%s n=%d dim=%d", self.model, len(texts), len(out[0]) if out else 0)
        return out


class OpenAIEmbed:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.name = "openai"
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        log.info("🧬 openai.embed model=%s n=%d", self.model, len(texts))
        return [d.embedding for d in resp.data]


def get_embedder() -> EmbeddingProvider:
    cfg = CFG.embed
    if cfg.provider == "ollama":
        return OllamaEmbed(cfg.model, CFG.llm.ollama_base_url)
    if cfg.provider == "openai":
        return OpenAIEmbed(cfg.model, CFG.llm.openai_api_key)
    raise ValueError(f"Unknown EMBED_PROVIDER: {cfg.provider}")
