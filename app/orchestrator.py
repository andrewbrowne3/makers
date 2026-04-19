"""Top-level flow: router → (workflow | agent) → (optional) evaluator → response.

Image-gen runs as a deterministic workflow (no LLM in the sequencing path).
Evaluator + question agent remain LLM-driven where judgment is actually needed.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

from app.agents.evaluator_agent import build_evaluator
from app.agents.question_agent import build_question_agent
from app.assets import list_mockups
from app.config import CFG
from app.db import create_run, finalize_run, record_design, sha256_file, upsert_client, upsert_logo
from app.logging_config import get_logger
from app.router import RouteDecision, get_router
from app.schemas import GenerateRequest, GenerateResponse, MockupResult
from app.workflows.image_gen import generate_one_mockup

log = get_logger("orchestrator")


def _build_query(req: GenerateRequest) -> str:
    bits = [f"client={req.client_name}", f"garment={req.garment_type}"]
    if req.colors:
        bits.append(f"colors={','.join(req.colors)}")
    if req.notes:
        bits.append(f"notes={req.notes}")
    return " | ".join(bits)


def _run_evaluator(image_url: str, brief: str) -> tuple[Optional[float], dict]:
    if not CFG.evaluator.enabled:
        return None, {}
    ev = build_evaluator()
    if ev is None:
        return None, {}
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            data = client.get(image_url).content
        score = ev.score(data, brief=brief)
        overall = float(score.get("overall", 0.0))
        log.info("📋 evaluator overall=%.3f", overall)
        return overall, score
    except Exception as e:  # noqa: BLE001
        log.exception("evaluator failed: %s", e)
        return None, {"error": str(e)}


def _resolve_mockup_indices(req: GenerateRequest) -> list[int]:
    if req.mockups:
        return req.mockups
    return [m.index for m in list_mockups()]


def _register_logo(client_id: int, req: GenerateRequest) -> Optional[int]:
    """Register the logo being used. Returns the logo row id, or None if no local file."""
    from pathlib import Path as _P

    if req.logo_path:
        p = _P(req.logo_path)
        if not p.exists():
            return None
        sha = sha256_file(p)
        return upsert_logo(client_id=client_id, filename=p.name, sha256=sha, storage_path=str(p))
    if req.logo_url:
        # we don't download-to-register here; the workflow already fetches.
        # callers who want full tracking should set logo_path.
        return None
    return None


def handle_generate(req: GenerateRequest) -> GenerateResponse:
    query = _build_query(req)

    if req.force_branch:
        decision = RouteDecision(branch=req.force_branch, score=1.0, matched_example="force_branch")
        log.info("🎯 force_branch=%s", req.force_branch)
    else:
        decision = get_router().classify(query)
        log.info("🎯 route=%s score=%.3f", decision.branch, decision.score)

    details: dict[str, Any] = {
        "router": {"branch": decision.branch, "score": decision.score, "match": decision.matched_example}
    }

    if decision.branch == "image_gen":
        indices = _resolve_mockup_indices(req)
        log.info("🧵 image_gen indices=%s", indices)
        client_id = upsert_client(req.client_name)
        logo_id = _register_logo(client_id, req)
        import json as _json
        run_id = create_run(
            label=req.run_label or f"{req.client_name} × {len(indices)} mock(s)",
            kind=req.run_kind or "adhoc",
            client_id=client_id,
            request_json=_json.dumps(req.model_dump(), ensure_ascii=False),
        )
        details["run_id"] = run_id
        results: list[MockupResult] = []
        per_mockup: list[dict] = []

        for idx in indices:
            try:
                workflow_out = generate_one_mockup(
                    client_name=req.client_name,
                    logo_path=req.logo_path,
                    logo_url=req.logo_url,
                    mockup_index=idx,
                    colors=req.colors,
                    notes=req.notes,
                )
                url = workflow_out["image_url"]
                brief = f"client={req.client_name} mockup={idx} prompt={workflow_out['prompt']}"
                score, ev_data = _run_evaluator(url, brief=brief)

                record_design(
                    client_id=client_id,
                    logo_id=logo_id,
                    mockup_index=idx,
                    s3_key=workflow_out["s3_key"],
                    prompt=workflow_out["prompt"],
                    provider=workflow_out["provider"],
                    evaluator=ev_data or None,
                    source="ai_generated",
                    attempts=int(workflow_out.get("attempts", 1)),
                    run_id=run_id,
                )

                results.append(
                    MockupResult(
                        mockup_index=idx,
                        image_url=url,
                        evaluator_score=score,
                        error=None,
                    )
                )
                per_mockup.append({
                    "mockup_index": idx,
                    "mockup_file": workflow_out["mockup_file"],
                    "provider": workflow_out["provider"],
                    "prompt": workflow_out["prompt"],
                    "evaluator": ev_data,
                })
            except Exception as e:  # noqa: BLE001
                log.exception("workflow failed for mockup=%d: %s", idx, e)
                results.append(MockupResult(mockup_index=idx, error=str(e)))
                per_mockup.append({"mockup_index": idx, "error": str(e)})

        details["mockups"] = per_mockup
        details["client_id"] = client_id
        details["logo_id"] = logo_id
        finalize_run(run_id)
        return GenerateResponse(
            status="ok",
            branch=decision.branch,
            router_score=decision.score,
            results=results,
            details=details,
        )

    if decision.branch == "evaluate":
        details["note"] = "standalone evaluate branch not yet wired"
        return GenerateResponse(status="not_implemented", branch=decision.branch, router_score=decision.score, details=details)

    # ask_question fallback
    q_agent = build_question_agent()
    result = q_agent.run(f"Client: {req.client_name}\nNotes: {req.notes or ''}")
    details["agent"] = {"answer": result.get("answer"), "steps": result.get("steps")}
    details["trace"] = result.get("trace", [])
    return GenerateResponse(
        status="needs_clarification",
        branch=decision.branch,
        router_score=decision.score,
        details=details,
    )
