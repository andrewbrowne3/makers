"""Structured logging: emoji-prefixed stdout + rotating file log.

Log every router decision, agent step, provider call. We want loud logs
so we can see where things go wrong.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from app.config import CFG

_CONFIGURED = False


def setup_logging() -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("makersgarments")
    if _CONFIGURED:
        return root

    Path(CFG.log_dir).mkdir(parents=True, exist_ok=True)

    root.setLevel(CFG.log_level.upper())
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.handlers.RotatingFileHandler(
        Path(CFG.log_dir) / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"makersgarments.{name}")
