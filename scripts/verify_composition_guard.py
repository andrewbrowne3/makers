"""Verify the 4-layer composition integrity guard.

Test 1: non-designer mode on 680thefan mock#1 (the Run #15 failure case).
Test 2: designer mode × 3 variants on the same.

Both tests report: attempts, critical_failure, composition_pure score, fill drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import setup_logging
from app.orchestrator import handle_generate
from app.schemas import GenerateRequest

LOGO = "/home/ab/Executive Crew Socks/TRAINING_DATA/https_--680thefan.com/logos and design assets/680-the-fan-logo-vertical1200x1200-300x300.png"


def run(label: str, designer_mode: bool) -> dict:
    print(f"\n=== {label} (designer_mode={designer_mode}) ===")
    req = GenerateRequest(
        client_name="680thefan.com",
        logo_path=LOGO,
        logo_url=None,
        mockups=[1],
        force_branch="image_gen",
        designer_mode=designer_mode,
        run_label=label,
        run_kind="adhoc",
    )
    resp = handle_generate(req)
    print(f"run_id={resp.details.get('run_id')}  cost=${resp.details.get('total_cost_usd', 0):.3f}")
    for m in resp.details.get("mockups", []):
        for v in m.get("variants", []):
            ev = v.get("evaluator") or {}
            print(
                f"  [{v.get('direction_label','default')}] "
                f"attempts={v.get('attempts')} "
                f"score={v.get('score')} "
                f"comp_pure={ev.get('composition_pure')} "
                f"pose={ev.get('pose_vertical')} "
                f"visible={ev.get('sock_visible')}"
            )
    return resp.model_dump()


def main() -> int:
    setup_logging()
    out = {}
    out["non_designer"] = run("comp-guard verify · non-designer", designer_mode=False)
    out["designer"] = run("comp-guard verify · designer", designer_mode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
