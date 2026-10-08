"""Immutable raw-object storage plus normalized research tables."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def default_data_root() -> str:
    """RESEARCH_DATA_ROOT if set, else <project folder>/data_lake, whatever the working directory.

    Before 2026-10-02 the default was the relative path "data_lake", so a run started from another
    folder (VS Code's Run button starts in VS Code's own folder) wrote a second, separate database.
    """
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("RESEARCH_DATA_ROOT") or os.path.join(project, "data_lake")


class DataStore:
    """Owns the isolated research data lake.

    Raw provider responses are content-addressed and never overwritten.  The
    normalized database is convenient for research queries and can always be
    rebuilt from the raw layer.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None):
        root = default_data_root() if root is None else root
        self.root = Path(root).expanduser().resolve()
        self.raw_root = self.root / "raw"
        self.db_path = self.root / "research.sqlite3"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection for one unit of work: commit on success, roll back
        on error, and always close. sqlite3's own ``with conn:`` only commits;
        the unclosed handle keeps the file locked on Windows."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    source_url TEXT,
                    content_type TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(source, dataset, sha256)
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    source TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    checkpoint_key TEXT NOT NULL,
                    checkpoint_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(source, dataset, checkpoint_key)
                );

                CREATE TABLE IF NOT EXISTS macro_observations (
                    source TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    observation_date TEXT NOT NULL,
                    value REAL,
                    realtime_start TEXT NOT NULL DEFAULT '',
                    realtime_end TEXT NOT NULL DEFAULT '',
                    retrieved_at TEXT NOT NULL,
                    raw_object_id INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(source, series_id, observation_date, realtime_start),
                    FOREIGN KEY(raw_object_id) REFERENCES raw_objects(id)
                );

                CREATE TABLE IF NOT EXISTS alpha_earnings (
                    symbol TEXT NOT NULL,
                    period_type TEXT NOT NULL,
                    fiscal_date_ending TEXT NOT NULL,
                    reported_date TEXT NOT NULL DEFAULT '',
                    report_time TEXT NOT NULL DEFAULT '',
                    reported_eps REAL,
                    estimated_eps REAL,
                    surprise REAL,
                    surprise_percentage REAL,
                    retrieved_at TEXT NOT NULL,
                    raw_object_id INTEGER,
                    raw_json TEXT NOT NULL,
                    PRIMARY KEY(symbol, period_type, fiscal_date_ending, reported_date),
                    FOREIGN KEY(raw_object_id) REFERENCES raw_objects(id)
                );

                CREATE TABLE IF NOT EXISTS alpha_listings (
                    as_of_date TEXT NOT NULL,
                    state TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    exchange TEXT,
                    asset_type TEXT,
                    ipo_date TEXT,
                    delisting_date TEXT,
                    status TEXT,
                    retrieved_at TEXT NOT NULL,
                    raw_object_id INTEGER,
                    raw_json TEXT NOT NULL,
                    PRIMARY KEY(as_of_date, state, symbol),
                    FOREIGN KEY(raw_object_id) REFERENCES raw_objects(id)
                );

                CREATE TABLE IF NOT EXISTS news_articles (
                    provider_code TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    published_at TEXT,
                    first_received_at TEXT NOT NULL,
                    last_received_at TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    body TEXT,
                    article_type INTEGER,
                    raw_json TEXT NOT NULL,
                    PRIMARY KEY(provider_code, article_id)
                );

                CREATE TABLE IF NOT EXISTS news_article_symbols (
                    provider_code TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    PRIMARY KEY(provider_code, article_id, symbol),
                    FOREIGN KEY(provider_code, article_id)
                        REFERENCES news_articles(provider_code, article_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS news_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    assessed_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    source_status TEXT NOT NULL,
                    status TEXT NOT NULL,
                    event_ids_json TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    latest_event_at TEXT,
                    event_type TEXT,
                    scope TEXT,
                    direction TEXT,
                    severity INTEGER,
                    horizon TEXT,
                    novelty TEXT,
                    relevance REAL,
                    confidence REAL,
                    headline TEXT,
                    catalyst TEXT,
                    surprise_summary TEXT,
                    reason TEXT,
                    evidence_article_ids_json TEXT NOT NULL DEFAULT '[]',
                    market_context_json TEXT NOT NULL DEFAULT '{}',
                    raw_response TEXT,
                    error TEXT,
                    latency_ms INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_news_assessments_symbol_time
                    ON news_assessments(symbol, assessed_at);

                CREATE TABLE IF NOT EXISTS premarket_news_decisions (
                    decision_id TEXT PRIMARY KEY,
                    assessment_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    quote_bid REAL,
                    quote_ask REAL,
                    quote_mid REAL,
                    spread_pct REAL,
                    previous_close REAL,
                    gap_pct REAL,
                    position_size_multiplier REAL,
                    submitted_at TEXT,
                    entry_price REAL,
                    exit_at TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl_pct REAL,
                    price_0930 REAL,
                    return_0930 REAL,
                    price_1000 REAL,
                    return_1000 REAL,
                    price_1030 REAL,
                    return_1030 REAL,
                    price_close REAL,
                    return_close REAL,
                    FOREIGN KEY(assessment_id) REFERENCES news_assessments(assessment_id)
                );

                CREATE INDEX IF NOT EXISTS idx_premarket_decisions_symbol_time
                    ON premarket_news_decisions(symbol, decided_at);

                CREATE TABLE IF NOT EXISTS core_news_shadow (
                    shadow_id TEXT PRIMARY KEY,
                    assessment_id TEXT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    signal_at TEXT NOT NULL,
                    reference_price REAL,
                    would_block INTEGER NOT NULL,
                    block_reason TEXT,
                    score REAL,
                    impact TEXT,
                    assessment_status TEXT,
                    price_30m REAL,
                    return_30m REAL,
                    price_60m REAL,
                    return_60m REAL,
                    price_close REAL,
                    return_close REAL
                );

                CREATE INDEX IF NOT EXISTS idx_core_news_shadow_symbol_time
                    ON core_news_shadow(symbol, signal_at);
                """
            )

    def save_raw(
        self,
        *,
        source: str,
        dataset: str,
        content: bytes,
        extension: str,
        source_url: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        digest = hashlib.sha256(content).hexdigest()
        now = dt.datetime.now(dt.timezone.utc)
        extension = extension.lstrip(".") or "bin"
        relative = Path(source) / dataset / f"{now:%Y}" / f"{now:%m}" / f"{digest}.{extension}"
        destination = self.raw_root / relative

        with self._lock:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
                    tmp.write(content)
                    temp_name = tmp.name
                os.replace(temp_name, destination)

            with self.connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO raw_objects
                        (source, dataset, retrieved_at, sha256, relative_path,
                         source_url, content_type, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        dataset,
                        utc_now(),
                        digest,
                        str(Path("raw") / relative),
                        source_url,
                        content_type,
                        json.dumps(metadata or {}, sort_keys=True),
                    ),
                )
                row = conn.execute(
                    "SELECT id FROM raw_objects WHERE source=? AND dataset=? AND sha256=?",
                    (source, dataset, digest),
                ).fetchone()
                return int(row["id"])

    def set_checkpoint(self, source: str, dataset: str, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                    (source, dataset, checkpoint_key, checkpoint_value, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, dataset, checkpoint_key) DO UPDATE SET
                    checkpoint_value=excluded.checkpoint_value,
                    updated_at=excluded.updated_at
                """,
                (source, dataset, key, value, utc_now()),
            )

    def get_checkpoint(self, source: str, dataset: str, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT checkpoint_value FROM checkpoints
                   WHERE source=? AND dataset=? AND checkpoint_key=?""",
                (source, dataset, key),
            ).fetchone()
        return None if row is None else str(row["checkpoint_value"])

    def upsert_macro(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = list(rows)
        if not payload:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO macro_observations
                    (source, series_id, observation_date, value, realtime_start,
                     realtime_end, retrieved_at, raw_object_id, metadata_json)
                VALUES (:source, :series_id, :observation_date, :value,
                        :realtime_start, :realtime_end, :retrieved_at,
                        :raw_object_id, :metadata_json)
                ON CONFLICT(source, series_id, observation_date, realtime_start)
                DO UPDATE SET
                    value=excluded.value,
                    realtime_end=excluded.realtime_end,
                    retrieved_at=excluded.retrieved_at,
                    raw_object_id=excluded.raw_object_id,
                    metadata_json=excluded.metadata_json
                """,
                payload,
            )
        return len(payload)

    def upsert_news_headlines(self, items: Iterable[dict[str, Any]]) -> int:
        payload = list(items)
        if not payload:
            return 0
        now = utc_now()
        with self.connect() as conn:
            for item in payload:
                provider = str(item.get("provider_code") or "UNKNOWN")
                article_id = str(item.get("article_id") or "").strip()
                if not article_id:
                    article_id = hashlib.sha256(
                        f"{provider}|{item.get('published_at')}|{item.get('headline')}".encode()
                    ).hexdigest()
                conn.execute(
                    """
                    INSERT INTO news_articles
                        (provider_code, article_id, published_at, first_received_at,
                         last_received_at, headline, body, article_type, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider_code, article_id) DO UPDATE SET
                        published_at=COALESCE(excluded.published_at, news_articles.published_at),
                        last_received_at=excluded.last_received_at,
                        headline=excluded.headline,
                        raw_json=excluded.raw_json
                    """,
                    (
                        provider,
                        article_id,
                        item.get("published_at"),
                        now,
                        now,
                        str(item.get("headline") or ""),
                        item.get("body"),
                        item.get("article_type"),
                        json.dumps(item, sort_keys=True, default=str),
                    ),
                )
                for symbol in item.get("symbols", []):
                    conn.execute(
                        """INSERT OR IGNORE INTO news_article_symbols
                           (provider_code, article_id, symbol) VALUES (?, ?, ?)""",
                        (provider, article_id, str(symbol).upper()),
                    )
        return len(payload)

    def update_news_body(
        self,
        provider_code: str,
        article_id: str,
        body: str,
        article_type: int | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE news_articles
                   SET body=?, article_type=?, last_received_at=?
                   WHERE provider_code=? AND article_id=?""",
                (body, article_type, utc_now(), provider_code, article_id),
            )

    def record_news_assessment(self, row: dict[str, Any]) -> None:
        """Persist one model assessment, including errors and model provenance."""
        defaults = {
            "latest_event_at": None,
            "event_type": None,
            "scope": None,
            "direction": None,
            "severity": None,
            "horizon": None,
            "novelty": None,
            "relevance": None,
            "confidence": None,
            "headline": None,
            "catalyst": None,
            "surprise_summary": None,
            "reason": None,
            "evidence_article_ids_json": "[]",
            "market_context_json": "{}",
            "raw_response": None,
            "error": None,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        payload = {**defaults, **row}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO news_assessments (
                    assessment_id, symbol, workflow, assessed_at, model_id,
                    prompt_version, source_status, status, event_ids_json,
                    event_count, latest_event_at, event_type, scope, direction,
                    severity, horizon, novelty, relevance, confidence, headline,
                    catalyst, surprise_summary, reason, evidence_article_ids_json,
                    market_context_json, raw_response, error, latency_ms,
                    input_tokens, output_tokens
                ) VALUES (
                    :assessment_id, :symbol, :workflow, :assessed_at, :model_id,
                    :prompt_version, :source_status, :status, :event_ids_json,
                    :event_count, :latest_event_at, :event_type, :scope, :direction,
                    :severity, :horizon, :novelty, :relevance, :confidence, :headline,
                    :catalyst, :surprise_summary, :reason, :evidence_article_ids_json,
                    :market_context_json, :raw_response, :error, :latency_ms,
                    :input_tokens, :output_tokens
                )
                """,
                payload,
            )

    def record_premarket_decision(self, row: dict[str, Any]) -> None:
        defaults = {
            "quote_bid": None,
            "quote_ask": None,
            "quote_mid": None,
            "spread_pct": None,
            "previous_close": None,
            "gap_pct": None,
            "position_size_multiplier": None,
        }
        payload = {**defaults, **row}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO premarket_news_decisions (
                    decision_id, assessment_id, symbol, decided_at, action,
                    status, reason, quote_bid, quote_ask, quote_mid, spread_pct,
                    previous_close, gap_pct, position_size_multiplier
                ) VALUES (
                    :decision_id, :assessment_id, :symbol, :decided_at, :action,
                    :status, :reason, :quote_bid, :quote_ask, :quote_mid, :spread_pct,
                    :previous_close, :gap_pct, :position_size_multiplier
                )
                """,
                payload,
            )

    def record_core_news_shadow(self, row: dict[str, Any]) -> None:
        defaults = {
            "assessment_id": None,
            "reference_price": None,
            "block_reason": None,
            "score": None,
            "impact": None,
            "assessment_status": None,
        }
        payload = {**defaults, **row}
        payload["would_block"] = 1 if payload.get("would_block") else 0
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO core_news_shadow (
                    shadow_id, assessment_id, symbol, signal, signal_at,
                    reference_price, would_block, block_reason, score, impact,
                    assessment_status
                ) VALUES (
                    :shadow_id, :assessment_id, :symbol, :signal, :signal_at,
                    :reference_price, :would_block, :block_reason, :score, :impact,
                    :assessment_status
                )
                """,
                payload,
            )

    def update_core_news_shadow(self, shadow_id: str, **updates: Any) -> None:
        allowed = {
            "price_30m",
            "return_30m",
            "price_60m",
            "return_60m",
            "price_close",
            "return_close",
        }
        payload = {key: value for key, value in updates.items() if key in allowed}
        if not payload:
            return
        assignments = ", ".join(f"{key}=?" for key in payload)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE core_news_shadow SET {assignments} WHERE shadow_id=?",
                [*payload.values(), shadow_id],
            )

    def update_premarket_decision(self, decision_id: str, **updates: Any) -> None:
        allowed = {
            "action",
            "status",
            "reason",
            "submitted_at",
            "entry_price",
            "exit_at",
            "exit_price",
            "exit_reason",
            "pnl_pct",
            "price_0930",
            "return_0930",
            "price_1000",
            "return_1000",
            "price_1030",
            "return_1030",
            "price_close",
            "return_close",
        }
        payload = {key: value for key, value in updates.items() if key in allowed}
        if not payload:
            return
        assignments = ", ".join(f"{key}=?" for key in payload)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE premarket_news_decisions SET {assignments} WHERE decision_id=?",
                [*payload.values(), decision_id],
            )
