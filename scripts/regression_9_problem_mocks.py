"""Regression on the 9 problem mocks from the Option-C batch, with the new
pose-lock prompt + auto-retry + expanded evaluator rubric.

Usage: python -m scripts.regression_9_problem_mocks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.assets import list_training_pairs
from app.logging_config import setup_logging
from app.orchestrator import handle_generate
from app.schemas import GenerateRequest

PROBLEMS = [
    ("communitycoffee.com",    [1, 2]),
    ("courtreserve.com",       [1, 2, 3]),
    ("harrisonhydragen.com",   [2, 3]),
    ("labviva.com",            [1, 2]),
]

RASTER = {".png", ".jpg", ".jpeg", ".webp"}


def _find_logo(client_key: str):
    for p in list_training_pairs():
        name = p.client.removeprefix("https_--").removeprefix("www.")
        if name == client_key:
            r = [lp for lp in p.logo_paths if lp.suffix.lower() in RASTER]
            if r:
                return r[0]
    return None


def main() -> int:
    setup_logging()
    rows = []
    for client, mockups in PROBLEMS:
        logo = _find_logo(client)
        if not logo:
            print(f"❌ {client}: no raster logo, skipping")
            continue
        req = GenerateRequest(
            client_name=client,
            logo_path=str(logo),
            logo_url=None,
            garment_type="sock",
            mockups=mockups,
            colors=[],
            notes=None,
            force_branch="image_gen",
            run_label=f"regression-9 · {client} × {len(mockups)}",
            run_kind="regression",
        )
        resp = handle_generate(req)
        for r in resp.results:
            print(f"  {client} mock#{r.mockup_index}: eval={r.evaluator_score} url_ok={bool(r.image_url)} err={r.error}")
            rows.append({
                "client": client,
                "mockup": r.mockup_index,
                "image_url": r.image_url,
                "evaluator_score": r.evaluator_score,
                "error": r.error,
            })

    scores = [r["evaluator_score"] for r in rows if r["evaluator_score"] is not None]
    avg = sum(scores) / max(1, len(scores))
    print(f"\n=== regression avg={avg:.3f} (n={len(scores)}) ===")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
