"""SQLite-based caching for LLM extraction and judge results."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class JudgeCache:
    """Thread-safe SQLite cache for LLM judge results and extracted profiles.

    Keyed by (product_id, query_hash) for judge results and by product_id
    for extraction profiles.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._local = threading.local()
        # Initialise schema on the creating thread
        self._init_schema(self._get_conn())

    # ------------------------------------------------------------------
    # Connection management (one per thread for thread safety)
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """\
            CREATE TABLE IF NOT EXISTS judge_cache (
                product_id TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                cached_at  TEXT NOT NULL,
                PRIMARY KEY (product_id, query_hash)
            );
            CREATE TABLE IF NOT EXISTS extraction_cache (
                product_id TEXT PRIMARY KEY,
                domain     TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                summary    TEXT NOT NULL,
                cached_at  TEXT NOT NULL
            );
            """
        )

    # ------------------------------------------------------------------
    # Judge result cache
    # ------------------------------------------------------------------
    def get_judgment(self, product_id: str, query_hash: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT result_json FROM judge_cache "
            "WHERE product_id = ? AND query_hash = ?",
            (product_id, query_hash),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_judgment(
        self, product_id: str, query_hash: str, result: dict
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO judge_cache "
            "(product_id, query_hash, result_json, cached_at) VALUES (?, ?, ?, ?)",
            (
                product_id,
                query_hash,
                json.dumps(result),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Extraction profile cache
    # ------------------------------------------------------------------
    def get_extraction(self, product_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT profile_json, summary FROM extraction_cache "
            "WHERE product_id = ?",
            (product_id,),
        ).fetchone()
        if row is None:
            return None
        return {"profile": json.loads(row[0]), "summary": row[1]}

    def put_extraction(
        self, product_id: str, domain: str, profile: dict, summary: str
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO extraction_cache "
            "(product_id, domain, profile_json, summary, cached_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                product_id,
                domain,
                json.dumps(profile),
                summary,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            "DELETE FROM judge_cache; DELETE FROM extraction_cache;"
        )

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
