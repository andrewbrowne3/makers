"""Tier 2 batch: fill coverage matrix (mocks 4-10 for under-tested clients)
plus re-run the 2 known-bad mockups with the new evaluator-driven retry.

Budget target: ~$2.40 base, up to ~$2.70 with retries.
Every request becomes its own run (grouped by client × mockup-set) so runs
remain differentiable in the gallery.
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

COVERAGE_TESTS = [
    ("ajw-inc.com",            [3, 4, 5, 6, 7, 8, 9, 10], 0),
    ("armorous.com",           [3, 4, 5, 6, 7, 8, 9, 10], 0),
    ("jarrettwalker.com",      [4, 5, 6, 7, 8, 9, 10],    0),
    ("communitycoffee.com",    [4, 5, 6, 7, 8, 9, 10],    1),  # PNG variant
    ("courtreserve.com",       [4, 5, 6, 7, 8, 9, 10],    0),
    ("harrisonhydragen.com",   [4, 5, 6, 7, 8, 9, 10],    0),
    ("labviva.com",            [4, 5, 6, 7, 8, 9, 10],    0),
    ("mspc.cpa",               [2, 3, 4, 5, 6, 7, 8, 9, 10], 0),
]

REGRESSION_TESTS = [
    # the 2 cases that exposed bugs last round
    ("courtreserve.com",       [1], 0),
    ("communitycoffee.com",    [2], 1),
]

RASTER = {".png", ".jpg", ".jpeg", ".webp"}


def _find_logo(client_key: str, idx: int):
    for p in list_training_pairs():
        name = p.client.removeprefix("https_--").removeprefix("www.")
        if name == client_key:
            r = [lp for lp in p.logo_paths if lp.suffix.lower() in RASTER]
            if r:
                return r[min(idx, len(r) - 1)]
    return None


def run(tests: list[tuple[str, list[int], int]], kind: str, label_prefix: str) -> list[dict]:
    out = []
    for client, mockups, logo_idx in tests:
        logo = _find_logo(client, logo_idx)
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
            run_label=f"{label_prefix} · {client} × {len(mockups)}",
            run_kind=kind,
        )
        resp = handle_generate(req)
        for r in resp.results:
            print(f"  [{kind}] {client} mock#{r.mockup_index}: eval={r.evaluator_score} url={'OK' if r.image_url else 'ERR'}")
            out.append({
                "kind": kind,
                "client": client,
                "mockup": r.mockup_index,
                "image_url": r.image_url,
                "evaluator_score": r.evaluator_score,
                "error": r.error,
            })
    return out


def main() -> int:
    setup_logging()
    rows = []
    print("\n=== REGRESSION (2 mocks) ===")
    rows += run(REGRESSION_TESTS, kind="regression", label_prefix="tier2-regression")
    print("\n=== COVERAGE FILL ===")
    rows += run(COVERAGE_TESTS, kind="batch", label_prefix="tier2-coverage")

    scores = [r["evaluator_score"] for r in rows if r["evaluator_score"] is not None]
    avg = sum(scores) / max(1, len(scores))
    print(f"\n=== SUMMARY ===  renders={len(rows)}  scored={len(scores)}  avg={avg:.3f}")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
