"""Run the 5-client test batch described in the plan.

Uses handle_generate() directly — no uvicorn needed. Every render auto-persists
to the SQLite DB via the orchestrator.

Usage: python -m scripts.test_batch_multi_client
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

TESTS = [
    # (client_domain_key, mockup_indices, logo_index_in_folder)
    ("680thefan.com",          [3, 4, 5, 6, 7, 8, 9, 10], 0),
    ("jarrettwalker.com",      [1, 2, 3],                 0),
    ("communitycoffee.com",    [1, 2, 3],                 1),  # PNG variant, not JPG
    ("courtreserve.com",       [1, 2, 3],                 0),
    ("harrisonhydragen.com",   [1, 2, 3],                 0),
    ("labviva.com",            [1, 2, 3],                 0),
    ("mspc.cpa",               [1],                       0),
]


RASTER_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _find_logo(client_key: str, logo_idx: int) -> Path | None:
    pairs = list_training_pairs()
    for p in pairs:
        name = p.client.removeprefix("https_--").removeprefix("www.")
        if name == client_key:
            rasters = [lp for lp in p.logo_paths if lp.suffix.lower() in RASTER_EXTS]
            if not rasters:
                return None
            return rasters[min(logo_idx, len(rasters) - 1)]
    return None


def main() -> int:
    setup_logging()
    summary = []
    for client_key, mockups, logo_idx in TESTS:
        logo = _find_logo(client_key, logo_idx)
        if not logo:
            print(f"❌ {client_key}: no logo found, skipping")
            continue
        print(f"\n=== {client_key} — mockups={mockups} — logo={logo.name} ===")
        req = GenerateRequest(
            client_name=client_key,
            logo_path=str(logo),
            logo_url=None,
            garment_type="sock",
            mockups=mockups,
            colors=[],
            notes=None,
            force_branch="image_gen",
            run_label=f"multi-client batch · {client_key} × {len(mockups)}",
            run_kind="batch",
        )
        resp = handle_generate(req)
        for r in resp.results:
            print(f"  mock#{r.mockup_index}: url={'OK' if r.image_url else 'ERR'} eval={r.evaluator_score} err={r.error}")
            summary.append({
                "client": client_key,
                "mockup": r.mockup_index,
                "image_url": r.image_url,
                "evaluator_score": r.evaluator_score,
                "error": r.error,
            })

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
