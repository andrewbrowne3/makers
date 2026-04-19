"""Central config loaded from .env. All runtime knobs live here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_bool(key: str, default: bool = False) -> bool:
    raw = _get(key, "").lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on")


def _get_float(key: str, default: float) -> float:
    raw = _get(key, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


LLMProvider = Literal["ollama", "anthropic", "openai", "google"]
ImageGenProvider = Literal["nano_banana", "gpt_image_1", "flux_kontext", "stub"]
EmbedProvider = Literal["ollama", "openai"]


@dataclass(frozen=True)
class LLMConfig:
    provider: LLMProvider = _get("LLM_PROVIDER", "ollama")  # type: ignore[assignment]
    model: str = _get("LLM_MODEL", "gemma4:e4b")
    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434")
    anthropic_api_key: str = _get("ANTHROPIC_API_KEY")
    openai_api_key: str = _get("OPENAI_API_KEY")
    google_api_key: str = _get("GOOGLE_API_KEY")


@dataclass(frozen=True)
class ImageGenConfig:
    provider: ImageGenProvider = _get("IMAGE_GEN_PROVIDER", "nano_banana")  # type: ignore[assignment]
    model: str = _get("IMAGE_GEN_MODEL", "gemini-2.5-flash-image")
    force_white_bg: bool = _get_bool("FORCE_WHITE_BG", True)
    rembg_model: str = _get("REMBG_MODEL", "u2net")


@dataclass(frozen=True)
class EmbedConfig:
    provider: EmbedProvider = _get("EMBED_PROVIDER", "ollama")  # type: ignore[assignment]
    model: str = _get("EMBED_MODEL", "mxbai-embed-large")
    chroma_dir: str = _get("CHROMA_DIR", str(PROJECT_ROOT / "db" / "chroma_db"))
    route_threshold: float = _get_float("ROUTE_SCORE_THRESHOLD", 0.3)


@dataclass(frozen=True)
class S3Config:
    access_key: str = _get("AWS_ACCESS_KEY_ID")
    secret_key: str = _get("AWS_SECRET_ACCESS_KEY")
    bucket: str = _get("AWS_STORAGE_BUCKET_NAME")
    region: str = _get("AWS_S3_REGION_NAME", "us-east-1")
    key_prefix: str = _get("S3_KEY_PREFIX", "makersgarments")
    presign_expiry: int = int(_get_float("S3_PRESIGN_EXPIRY", 3600))


@dataclass(frozen=True)
class AssetsConfig:
    exec_root: str = _get("EXEC_ROOT", "/home/ab/Executive Crew Socks")
    mock_dir: str = _get("MOCK_DIR", "/home/ab/Executive Crew Socks/PNG_3D_Mockup_Exec")
    psd_dir: str = _get("PSD_DIR", "/home/ab/Executive Crew Socks/PHOTOSHOP_FILES_FOR_DESIGNERS_Exec")
    prod_dir: str = _get("PROD_DIR", "/home/ab/Executive Crew Socks/ProdFiles_Exec")
    training_dir: str = _get("TRAINING_DIR", "/home/ab/Executive Crew Socks/TRAINING_DATA")


@dataclass(frozen=True)
class EvaluatorConfig:
    enabled: bool = _get_bool("EVALUATOR_ENABLED", True)
    provider: str = _get("EVALUATOR_PROVIDER", "ollama")
    model: str = _get("EVALUATOR_MODEL", "gemma4:e4b")
    threshold: float = _get_float("EVALUATOR_THRESHOLD", 0.6)


@dataclass(frozen=True)
class AppConfig:
    port: int = int(_get("PORT", "8000"))
    log_level: str = _get("LOG_LEVEL", "INFO")
    log_dir: str = _get("LOG_DIR", str(PROJECT_ROOT / "logs"))
    llm: LLMConfig = field(default_factory=LLMConfig)
    image_gen: ImageGenConfig = field(default_factory=ImageGenConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    s3: S3Config = field(default_factory=S3Config)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)


CFG = AppConfig()
