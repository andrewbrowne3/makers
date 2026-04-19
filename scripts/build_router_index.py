"""Rebuild the router's Chroma index from the seed corpus.

Usage: python -m scripts.build_router_index
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import setup_logging
from app.router import get_router


def main() -> int:
    setup_logging()
    n = get_router().rebuild()
    print(f"✅ Rebuilt router index with {n} examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
