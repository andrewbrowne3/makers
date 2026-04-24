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
    retry_of         INTEGER REFERENCES designs(id) ON DELETE SET NULL,
    attempts         INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_designs_client ON designs(client_id);
CREATE INDEX IF NOT EXISTS idx_designs_logo   ON designs(logo_id);
CREATE INDEX IF NOT EXISTS idx_designs_key    ON designs(s3_key);

CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    label          TEXT,
    kind           TEXT NOT NULL DEFAULT 'adhoc',
    client_id      INTEGER REFERENCES clients(id) ON DELETE SET NULL,
    request_json   TEXT,
    status         TEXT NOT NULL DEFAULT 'running',
    n_renders      INTEGER NOT NULL DEFAULT 0,
    n_retries      INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    avg_score      REAL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
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
        # Idempotent column migrations for DBs created before these columns existed
        for stmt in (
            "ALTER TABLE designs ADD COLUMN retry_of INTEGER REFERENCES designs(id) ON DELETE SET NULL",
            "ALTER TABLE designs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE designs ADD COLUMN run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL",
            "ALTER TABLE designs ADD COLUMN direction_label TEXT",
            "ALTER TABLE designs ADD COLUMN is_hero INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                con.execute(stmt)
            except sqlite3.OperationalError:
                pass


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
    retry_of: Optional[int] = None,
    attempts: int = 1,
    run_id: Optional[int] = None,
    direction_label: Optional[str] = None,
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
        # Ensure the new columns exist on databases that were created before the migration
        for stmt in (
            "ALTER TABLE designs ADD COLUMN retry_of INTEGER REFERENCES designs(id) ON DELETE SET NULL",
            "ALTER TABLE designs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE designs ADD COLUMN run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL",
        ):
            try:
                con.execute(stmt)
            except sqlite3.OperationalError:
                pass

        version = _next_version(con, logo_id, mockup_index)
        cur = con.execute(
            """
            INSERT INTO designs
                (logo_id, client_id, mockup_index, s3_key, local_path, prompt,
                 provider, evaluator_score, evaluator_notes, evaluator_json,
                 source, version, retry_of, attempts, run_id, direction_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (logo_id, client_id, mockup_index, s3_key, local_path, prompt,
             provider, ev_score, ev_notes, ev_json, source, version,
             retry_of, attempts, run_id, direction_label),
        )
        design_id = int(cur.lastrowid)
        log.info("💾 design recorded id=%d run=%s client=%s mock=%s v=%d source=%s attempts=%d",
                 design_id, run_id, client_id, mockup_index, version, source, attempts)
        return design_id


# ── Runs ──────────────────────────────────────────────────────────────

COST_PER_RENDER_USD = 0.039  # Nano Banana Pro @ 1024


def create_run(
    *,
    label: Optional[str] = None,
    kind: str = "adhoc",
    client_id: Optional[int] = None,
    request_json: Optional[str] = None,
) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO runs (label, kind, client_id, request_json) VALUES (?, ?, ?, ?)",
            (label, kind, client_id, request_json),
        )
        run_id = int(cur.lastrowid)
    log.info("🏃 run created id=%d kind=%s label=%s client=%s", run_id, kind, label, client_id)
    return run_id


def finalize_run(run_id: int, *, status: str = "completed") -> None:
    with connect() as con:
        stats = con.execute(
            """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(attempts), 0) AS attempts,
                   AVG(evaluator_score) AS avg_score
            FROM designs
            WHERE run_id = ? AND source = 'ai_generated'
            """,
            (run_id,),
        ).fetchone()
        n = int(stats["n"] or 0)
        attempts_total = int(stats["attempts"] or 0)
        avg_score = float(stats["avg_score"]) if stats["avg_score"] is not None else None
        cost = round(attempts_total * COST_PER_RENDER_USD, 4)
        n_retries = max(0, attempts_total - n)
        con.execute(
            """
            UPDATE runs
               SET status = ?, n_renders = ?, n_retries = ?,
                   total_cost_usd = ?, avg_score = ?,
                   completed_at = datetime('now')
             WHERE id = ?
            """,
            (status, n, n_retries, cost, avg_score, run_id),
        )
    log.info("🏁 run finalized id=%d n=%d retries=%d cost=$%.3f avg=%s", run_id, n, n_retries, cost, avg_score)


def list_runs(limit: int = 100) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """
            SELECT r.*, c.name AS client_name
              FROM runs r LEFT JOIN clients c ON c.id = r.client_id
             ORDER BY r.created_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> Optional[dict]:
    with connect() as con:
        row = con.execute(
            "SELECT r.*, c.name AS client_name FROM runs r LEFT JOIN clients c ON c.id=r.client_id WHERE r.id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None


def get_design_by_id(design_id: int) -> Optional[dict]:
    """Return a single design row (joined with its run) or None if missing."""
    with connect() as con:
        row = con.execute(
            """
            SELECT d.*, r.kind AS run_kind, r.label AS run_label
              FROM designs d
              LEFT JOIN runs r ON r.id = d.run_id
             WHERE d.id = ?
            """,
            (design_id,),
        ).fetchone()
        return dict(row) if row else None


def get_logo_by_id(logo_id: int) -> Optional[dict]:
    """Return a single logo row or None."""
    with connect() as con:
        row = con.execute("SELECT * FROM logos WHERE id = ?", (logo_id,)).fetchone()
        return dict(row) if row else None


def list_templates_used_by_client(client_id: int) -> list[dict]:
    """Distinct mockup_index values that this client's AI renders used, with counts."""
    with connect() as con:
        rows = con.execute(
            """
            SELECT mockup_index, COUNT(*) AS render_count
              FROM designs
             WHERE client_id = ?
               AND source = 'ai_generated'
               AND mockup_index IS NOT NULL
             GROUP BY mockup_index
             ORDER BY mockup_index
            """,
            (client_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_hero(*, run_id: int, mockup_index: int, s3_key: str) -> None:
    """Set is_hero=1 on the design matching (run_id, mockup_index, s3_key); clear
    the flag on sibling variants of the same mockup within the run."""
    with connect() as con:
        con.execute(
            "UPDATE designs SET is_hero=0 WHERE run_id=? AND mockup_index=?",
            (run_id, mockup_index),
        )
        con.execute(
            "UPDATE designs SET is_hero=1 WHERE run_id=? AND mockup_index=? AND s3_key=?",
            (run_id, mockup_index, s3_key),
        )
    log.info("🏆 hero set run=%d mock=%d key=%s", run_id, mockup_index, s3_key)


def list_designs_for_run(run_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """
            SELECT d.*, r.kind AS run_kind, r.label AS run_label
              FROM designs d LEFT JOIN runs r ON r.id = d.run_id
             WHERE d.run_id = ?
             ORDER BY d.mockup_index, d.version
            """,
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


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
        where.append("d.client_id = ?")
        args.append(client_id)
    if logo_id is not None:
        where.append("d.logo_id = ?")
        args.append(logo_id)
    if mockup_index is not None:
        where.append("d.mockup_index = ?")
        args.append(mockup_index)
    if source is not None:
        where.append("d.source = ?")
        args.append(source)
    sql = """
        SELECT d.*, r.kind AS run_kind, r.label AS run_label
          FROM designs d
          LEFT JOIN runs r ON r.id = d.run_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY d.created_at DESC LIMIT ?"
    args.append(limit)
    with connect() as con:
        rows = con.execute(sql, args).fetchall()
        return [dict(r) for r in rows]
