"""Top-level flow: router → (workflow | agent) → (optional) evaluator → response.

Image-gen runs as a deterministic workflow (no LLM in the sequencing path).
Evaluator + question agent remain LLM-driven where judgment is actually needed.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

from app.agents.designer_agent import Direction, propose_directions, resolved_directions_count, should_use_designer
from app.agents.evaluator_agent import build_evaluator
from app.agents.question_agent import build_question_agent
from app.assets import get_mockup, list_mockups
from app.config import CFG
from app.db import create_run, finalize_run, record_design, sha256_file, upsert_client, upsert_logo
from app.logging_config import get_logger
from app.router import RouteDecision, get_router
from app.schemas import GenerateRequest, GenerateResponse, MockupResult
from app.workflows.image_gen import generate_one_mockup


COST_PER_RENDER = 0.039  # NB Pro @ 1024

log = get_logger("orchestrator")


def _build_query(req: GenerateRequest) -> str:
    bits = [f"client={req.client_name}", f"garment={req.garment_type}"]
    if req.colors:
        bits.append(f"colors={','.join(req.colors)}")
    if req.notes:
        bits.append(f"notes={req.notes}")
    return " | ".join(bits)


def _directions_for(
    req: GenerateRequest, designer_on: bool, n: int, mockup_index: int,
) -> list[Optional[Direction]]:
    """If designer is on, ask the designer for n directions. Otherwise return a
    single None sentinel meaning 'render once, no extra prompt'."""
    if not designer_on or n <= 1:
        return [None]
    # Load logo + mockup bytes for the designer's multimodal input
    from pathlib import Path as _P
    import httpx as _httpx

    if req.logo_path and _P(req.logo_path).exists():
        logo_bytes = _P(req.logo_path).read_bytes()
    elif req.logo_url:
        with _httpx.Client(timeout=30, follow_redirects=True) as c:
            logo_bytes = c.get(req.logo_url).content
    else:
        logo_bytes = b""
    try:
        mock_bytes = get_mockup(mockup_index).path.read_bytes()
    except Exception:  # noqa: BLE001
        mock_bytes = b""
    try:
        dirs = propose_directions(
            client_name=req.client_name,
            logo_bytes=logo_bytes,
            mockup_bytes=mock_bytes,
            n=n,
        )
        return list(dirs)
    except Exception as e:  # noqa: BLE001
        log.warning("designer failed, single render: %s", e)
        return [None]


def _render_and_score(
    req: GenerateRequest,
    idx: int,
    *,
    extra_prompt: Optional[str],
    direction_label: Optional[str],
) -> dict:
    """Render one variant (optionally with extra_prompt), score with evaluator,
    auto-retry once if pose/visibility fails. Returns a flat dict."""
    workflow_out = generate_one_mockup(
        client_name=req.client_name,
        logo_path=req.logo_path,
        logo_url=req.logo_url,
        mockup_index=idx,
        colors=req.colors,
        notes=req.notes,
        extra_prompt=extra_prompt,
    )
    url = workflow_out["image_url"]
    brief = f"client={req.client_name} mockup={idx} direction={direction_label or 'default'}"
    score, ev_data = _run_evaluator(url, brief=brief)

    total_attempts = int(workflow_out.get("attempts", 1))
    reason = _evaluator_retry_reason(ev_data or {})
    if reason:
        log.warning("🔁 evaluator-driven retry (dir=%s): %s", direction_label or "default", reason)
        workflow_out = generate_one_mockup(
            client_name=req.client_name,
            logo_path=req.logo_path,
            logo_url=req.logo_url,
            mockup_index=idx,
            colors=req.colors,
            notes=req.notes,
            extra_prompt=extra_prompt,
            retry_reason=reason,
        )
        url = workflow_out["image_url"]
        score, ev_data = _run_evaluator(url, brief=brief)
        total_attempts += int(workflow_out.get("attempts", 1))

    return {
        "image_url": url,
        "s3_key": workflow_out["s3_key"],
        "prompt": workflow_out["prompt"],
        "provider": workflow_out["provider"],
        "score": score,
        "evaluator": ev_data,
        "attempts": total_attempts,
        "direction_label": direction_label,
    }


def _evaluator_retry_reason(ev_data: dict) -> Optional[str]:
    """Translate evaluator scores into a retry reason string, if warranted."""
    if not ev_data:
        return None
    pose = ev_data.get("pose_vertical")
    vis = ev_data.get("sock_visible")
    try:
        if pose is not None and float(pose) < 0.5:
            return f"evaluator flagged the sock as tilted or laid horizontally (pose_vertical={pose})"
        if vis is not None and float(vis) < 0.5:
            return f"evaluator flagged the sock as missing or too small (sock_visible={vis})"
    except (TypeError, ValueError):
        pass
    return None


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
        designer_on = should_use_designer(req.designer_mode)
        n_directions = resolved_directions_count(req.designer_directions) if designer_on else 1
        run_id = create_run(
            label=req.run_label or f"{req.client_name} × {len(indices)} mock(s){' · designer' if designer_on else ''}",
            kind=req.run_kind or "adhoc",
            client_id=client_id,
            request_json=_json.dumps(req.model_dump(), ensure_ascii=False),
        )
        details["run_id"] = run_id
        details["designer_mode"] = designer_on
        details["n_directions"] = n_directions
        results: list[MockupResult] = []
        per_mockup: list[dict] = []
        spent_usd = 0.0
        cost_cap = CFG.designer.cost_cap_usd if designer_on else None

        for idx in indices:
            if cost_cap is not None and spent_usd >= cost_cap:
                log.warning("💰 cost cap reached ($%.2f >= $%.2f), skipping mock#%d and remaining",
                            spent_usd, cost_cap, idx)
                results.append(MockupResult(mockup_index=idx, error=f"cost cap ${cost_cap:.2f} reached"))
                per_mockup.append({"mockup_index": idx, "error": "cost_cap_reached"})
                continue

            directions = _directions_for(req, designer_on, n_directions, idx)
            variants: list[dict] = []

            for d in directions:
                try:
                    out = _render_and_score(
                        req,
                        idx,
                        extra_prompt=d.prompt_addendum if d else None,
                        direction_label=d.label if d else None,
                    )
                    spent_usd += COST_PER_RENDER * int(out["attempts"])
                    record_design(
                        client_id=client_id,
                        logo_id=logo_id,
                        mockup_index=idx,
                        s3_key=out["s3_key"],
                        prompt=out["prompt"],
                        provider=out["provider"],
                        evaluator=out["evaluator"] or None,
                        source="ai_generated",
                        attempts=out["attempts"],
                        run_id=run_id,
                        direction_label=d.label if d else None,
                    )
                    variants.append(out)
                except Exception as e:  # noqa: BLE001
                    log.exception("variant failed mock=%d dir=%s: %s", idx, d.label if d else "default", e)
                    variants.append({"error": str(e), "direction_label": d.label if d else None})

            # Pick hero by evaluator score (highest wins; ties go to first)
            scored = [(i, v.get("score")) for i, v in enumerate(variants) if v.get("score") is not None]
            hero_idx = max(scored, key=lambda x: x[1])[0] if scored else 0
            hero = variants[hero_idx] if variants else None

            if hero and hero.get("s3_key"):
                from app.db import mark_hero
                mark_hero(run_id=run_id, mockup_index=idx, s3_key=hero["s3_key"])

            results.append(
                MockupResult(
                    mockup_index=idx,
                    image_url=(hero or {}).get("image_url"),
                    evaluator_score=(hero or {}).get("score"),
                    error=None if hero and hero.get("image_url") else "no variant succeeded",
                )
            )
            per_mockup.append({
                "mockup_index": idx,
                "variants": [
                    {
                        "direction_label": v.get("direction_label"),
                        "image_url": v.get("image_url"),
                        "score": v.get("score"),
                        "attempts": v.get("attempts"),
                        "error": v.get("error"),
                    }
                    for v in variants
                ],
                "hero_direction": variants[hero_idx].get("direction_label") if variants else None,
            })

        details["mockups"] = per_mockup
        details["client_id"] = client_id
        details["logo_id"] = logo_id
        details["total_cost_usd"] = round(spent_usd, 4)
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
