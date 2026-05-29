"""
Persistence for targets, scans, findings, artifacts, and API keys.

Three backing stores:
  SQLite        — structured records (targets, scans, findings). File: data/ltraas.db
  JSON files    — raw engine output per job. Directory: data/artifacts/
  In-memory dict — API key vault. Keys are fetched just-in-time and never
                   written to disk.

No other module writes SQL directly — everything goes through these functions.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .types import Target, ScanDefinition, Finding


DB_PATH = Path("data/ltraas.db")
ARTIFACTS_DIR = Path("data/artifacts")


def init_db() -> None:
    """Create tables on first run. Idempotent."""
    DB_PATH.parent.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS targets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            endpoint_url TEXT NOT NULL,
            system_type TEXT NOT NULL,
            purpose TEXT NOT NULL,
            auth_ref TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS findings (
            id TEXT PRIMARY KEY,
            scan_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            category_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            engine_source TEXT NOT NULL,
            probe_text TEXT NOT NULL,
            target_response TEXT NOT NULL,
            judge_verdict TEXT NOT NULL,
            judge_reasoning TEXT NOT NULL,
            runs INTEGER NOT NULL,
            successes INTEGER NOT NULL,
            success_rate REAL NOT NULL,
            plugin_id TEXT,
            cluster_id TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    # Migration: add plugin_id column if it was created before this field existed.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(findings)")}
    if "plugin_id" not in cols:
        conn.execute("ALTER TABLE findings ADD COLUMN plugin_id TEXT")
        conn.commit()
    conn.close()


def _conn():
    return sqlite3.connect(DB_PATH)


def save_target(t: Target) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO targets VALUES (?,?,?,?,?,?)",
            (t.id, t.name, t.endpoint_url, t.system_type, t.purpose, t.auth_ref),
        )


def get_target(target_id: str) -> Target | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM targets WHERE id=?", (target_id,)).fetchone()
        return Target(*row) if row else None


def save_scan(scan_id: str, target_id: str, name: str, status: str, created_at: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?)",
            (scan_id, target_id, name, status, created_at),
        )


def update_scan_status(scan_id: str, status: str) -> None:
    with _conn() as c:
        c.execute("UPDATE scans SET status=? WHERE id=?", (status, scan_id))


def save_finding(f: Finding) -> None:
    with _conn() as c:
        c.execute(
            """INSERT INTO findings
               (id, scan_id, target_id, category_id, severity, engine_source,
                probe_text, target_response, judge_verdict, judge_reasoning,
                runs, successes, success_rate, plugin_id, cluster_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f.id, f.scan_id, f.target_id, f.category_id, f.severity,
                f.engine_source, f.probe_text, f.target_response,
                f.judge_verdict, f.judge_reasoning,
                f.runs, f.successes, f.success_rate,
                f.plugin_id, f.cluster_id, f.created_at,
            ),
        )


def findings_for_scan(scan_id: str) -> list[Finding]:
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM findings WHERE scan_id=?", (scan_id,)).fetchall()
        return [Finding(**dict(r)) for r in rows]


def save_artifact(scan_id: str, job_id: str, data: Any) -> str:
    """Drop a worker's raw output to disk. Returns the path."""
    path = ARTIFACTS_DIR / f"{scan_id}_{job_id}.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


_VAULT: dict[str, str] = {}


def vault_put(ref: str, secret: str) -> None:
    """Store a secret under the given reference key."""
    _VAULT[ref] = secret


def vault_get(ref: str) -> str:
    """Retrieve a secret by its reference key. Raises KeyError if not found."""
    return _VAULT[ref]
