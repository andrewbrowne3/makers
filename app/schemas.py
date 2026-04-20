"""Pydantic request/response shapes for the HTTP layer."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request payload: a client + their logo + which mockups to render."""

    client_name: str = Field(..., description="Client the design is for")
    logo_url: Optional[str] = Field(None, description="URL to the client logo image")
    logo_path: Optional[str] = Field(None, description="Local filesystem path to the client logo")
    garment_type: str = Field("sock", description="Currently only 'sock' is wired")
    mockups: Optional[list[int]] = Field(
        None,
        description="Which mockup indices (1..10) to render. Defaults to all 10.",
    )
    colors: list[str] = Field(default_factory=list, description="Hex or named brand colors")
    notes: Optional[str] = Field(None, description="Free-text designer notes")
    force_branch: Optional[str] = Field(
        None,
        description="Override the router — 'image_gen', 'evaluate', 'ask_question'",
    )
    run_label: Optional[str] = Field(
        None,
        description="Human-readable label for this run (e.g. 'Option-C batch', 'ui-adhoc')",
    )
    run_kind: Optional[str] = Field(
        "adhoc",
        description="Run category: 'adhoc' | 'batch' | 'regression' | 'ui'",
    )
    designer_mode: Optional[bool] = Field(
        None,
        description="If true, designer agent proposes N creative directions per mockup (3× cost). Overrides env default.",
    )
    designer_directions: Optional[int] = Field(
        None,
        description="How many creative variations per mockup when designer_mode is on (default from env).",
    )


class MockupResult(BaseModel):
    mockup_index: int
    image_url: Optional[str] = None
    evaluator_score: Optional[float] = None
    error: Optional[str] = None


class GenerateResponse(BaseModel):
    status: str
    branch: str
    router_score: Optional[float] = None
    results: list[MockupResult] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    image_gen_provider: str
    embed_provider: str
    bucket: str
