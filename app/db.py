"""SQLite persistence layer — stdlib only.

Tables:
  clients  — one row per client name
  logos    — one row per unique logo file (sha256-dedup'd); may belong to a client
  designs  — one row per generated (or baseline) design; versioned per (logo, mockup)

Schema auto-created on first connection. Safe to import from any module.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from app.config import CFG, PROJECT_ROOT
from app.logging_config import get_logger

log = get_logger("db")

DB_PATH = Path(CFG.embed.chroma_dir).parent / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS logos (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id    INTEGER REFERENCES clients(id) ON DELETE SET NULL,
    filename     TEXT NOT NULL,
    sha256       TEXT NOT NULL UNIQUE,
    storage_path TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_logos_client ON logos(client_id);

CREATE TABLE IF NOT EXISTS designs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    logo_id          INTEGER REFERENCES logos(id) ON DELETE SET NULL,
    client_id        INTEGER REFERENCES clients(id) ON DELETE SET NULL,
    mockup_index     INTEGER,
    s3_key           TEXT,
    local_path       TEXT,
    prompt           TEXT,
    provider         TEXT,
    evaluator_score  REAL,
    evaluator_notes  TEXT,
    evaluator_json   TEXT,
    source           TEXT NOT NULL DEFAULT 'ai_generated',
    version          INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_designs_client ON designs(client_id);
CREATE INDEX IF NOT EXISTS idx_designs_logo   ON designs(logo_id);
CREATE INDEX IF NOT EXISTS idx_designs_key    ON designs(s3_key);
"""


@dataclass
class ClientRow:
    id: int
    name: str
    created_at: str


@dataclass
class LogoRow:
    id: int
    client_id: Optional[int]
    filename: str
    sha256: str
    storage_path: Optional[str]
    created_at: str


@dataclass
class DesignRow:
    id: int
    logo_id: Optional[int]
    client_id: Optional[int]
    mockup_index: Optional[int]
    s3_key: Optional[str]
    local_path: Optional[str]
    prompt: Optional[str]
    provider: Optional[str]
    evaluator_score: Optional[float]
    evaluator_notes: Optional[str]
    evaluator_json: Optional[str]
    source: str
    version: int
    created_at: str


def _row(cur: sqlite3.Cursor, row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    return {d[0]: row[i] for i, d in enumerate(cur.description)}


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
    finally:
        con.close()


def _init_schema() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


_init_schema()
log.info("💾 SQLite ready path=%s", DB_PATH)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upsert_client(name: str) -> int:
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO clients (name) VALUES (?)", (name,))
        row = con.execute("SELECT id FROM clients WHERE name = ?", (name,)).fetchone()
        return int(row["id"])


def upsert_logo(*, client_id: Optional[int], filename: str, sha256: str, storage_path: Optional[str]) -> int:
    with connect() as con:
        existing = con.execute("SELECT id FROM logos WHERE sha256 = ?", (sha256,)).fetchone()
        if existing:
            return int(existing["id"])
        cur = con.execute(
            "INSERT INTO logos (client_id, filename, sha256, storage_path) VALUES (?, ?, ?, ?)",
            (client_id, filename, sha256, storage_path),
        )
        return int(cur.lastrowid)


def _next_version(con: sqlite3.Connection, logo_id: Optional[int], mockup_index: Optional[int]) -> int:
    row = con.execute(
        "SELECT COALESCE(MAX(version), 0) AS v FROM designs WHERE logo_id IS ? AND mockup_index IS ?",
        (logo_id, mockup_index),
    ).fetchone()
    return int(row["v"]) + 1


def record_design(
    *,
    client_id: Optional[int],
    logo_id: Optional[int],
    mockup_index: Optional[int],
    s3_key: Optional[str],
    local_path: Optional[str] = None,
    prompt: Optional[str] = None,
    provider: Optional[str] = None,
    evaluator: Optional[dict[str, Any]] = None,
    source: str = "ai_generated",
) -> int:
    ev_score = None
    ev_notes = None
    ev_json = None
    if evaluator:
        ev_score = evaluator.get("overall")
        if isinstance(ev_score, (int, float)):
            ev_score = float(ev_score)
        else:
            ev_score = None
        ev_notes = evaluator.get("notes")
        ev_json = json.dumps(evaluator, ensure_ascii=False)

    with connect() as con:
        version = _next_version(con, logo_id, mockup_index)
        cur = con.execute(
            """
            INSERT INTO designs
                (logo_id, client_id, mockup_index, s3_key, local_path, prompt,
                 provider, evaluator_score, evaluator_notes, evaluator_json, source, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (logo_id, client_id, mockup_index, s3_key, local_path, prompt,
             provider, ev_score, ev_notes, ev_json, source, version),
        )
        design_id = int(cur.lastrowid)
        log.info("💾 design recorded id=%d client=%s logo=%s mock=%s v=%d source=%s",
                 design_id, client_id, logo_id, mockup_index, version, source)
        return design_id


def list_clients() -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT id, name, created_at FROM clients ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def get_client_by_name(name: str) -> Optional[dict]:
    with connect() as con:
        row = con.execute("SELECT id, name, created_at FROM clients WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


def list_logos_for_client(client_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT id, client_id, filename, sha256, storage_path, created_at "
            "FROM logos WHERE client_id = ? ORDER BY created_at DESC",
            (client_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_designs(
    *,
    client_id: Optional[int] = None,
    logo_id: Optional[int] = None,
    mockup_index: Optional[int] = None,
    source: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    where = []
    args: list[Any] = []
    if client_id is not None:
        where.append("client_id = ?")
        args.append(client_id)
    if logo_id is not None:
        where.append("logo_id = ?")
        args.append(logo_id)
    if mockup_index is not None:
        where.append("mockup_index = ?")
        args.append(mockup_index)
    if source is not None:
        where.append("source = ?")
        args.append(source)
    sql = "SELECT * FROM designs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with connect() as con:
        rows = con.execute(sql, args).fetchall()
        return [dict(r) for r in rows]
