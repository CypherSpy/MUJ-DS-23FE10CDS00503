"""SQLite response cache for LLM outputs."""

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path


class ResponseCache:
    """Thread-safe SQLite response cache for LLM API calls."""

    def __init__(self, db_path: str = ".cache/llm_cache.sqlite", enabled: bool = True):
        self.db_path = Path(db_path)
        self.enabled = enabled
        self._lock = threading.Lock()

        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS responses (
                            key TEXT PRIMARY KEY,
                            raw TEXT NOT NULL,
                            input_tokens INTEGER,
                            output_tokens INTEGER,
                            created_at TEXT NOT NULL
                        )
                        """
                    )
            finally:
                conn.close()

    @staticmethod
    def make_key(
        provider: str,
        model: str,
        prompt_version: str,
        mode: str,
        temperature: float,
        text: str,
    ) -> str:
        """Generate deterministic SHA-256 cache key based on request parameters."""
        payload = json.dumps(
            [provider, model, prompt_version, mode, float(temperature), text],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[str, int | None, int | None] | None:
        """Fetch raw cached response by key.

        Returns (raw_text, input_tokens, output_tokens) if found and valid JSON, else None.
        """
        if not self.enabled:
            return None

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT raw, input_tokens, output_tokens FROM responses WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                raw, in_tok, out_tok = row
                # Verify raw text is valid JSON / parseable
                try:
                    # Quick JSON validation check
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        return None
                except Exception:  # noqa: BLE001
                    # Ignore invalid/corrupted cache entries
                    return None

                return (raw, in_tok, out_tok)
            finally:
                conn.close()

    def set(
        self,
        key: str,
        raw: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Store or overwrite raw response in cache."""
        if not self.enabled:
            return

        now_str = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO responses (key, raw, input_tokens, output_tokens, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (key, raw, input_tokens, output_tokens, now_str),
                    )
            finally:
                conn.close()

    def clear(self) -> None:
        """Clear all cached responses."""
        if not self.enabled:
            return
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("DELETE FROM responses")
            finally:
                conn.close()
