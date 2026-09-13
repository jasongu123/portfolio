"""Point-in-time earnings state for the existing equity strategy.

This module is deliberately broker- and model-agnostic.  IBKR/WSH, a news
collector, or a historical replay process may append events and structured
assessments to the store.  The trading strategy only sees a small state
machine and an entry-only embargo.  No method in this module can place an
order.

The feature is dormant unless explicitly enabled by ``ibkr2.py``.  Legacy
``earnings_dates.json`` records can be imported as conservative, all-day
events, but records with a confirmed BEFORE/AFTER MARKET classification are
preferred.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")

EMBARGO = "EMBARGO"
WAITING_FOR_RELEASE = "WAITING_FOR_RELEASE"
WAITING_FOR_CALL = "WAITING_FOR_CALL"
HOLD = "HOLD"
ELIGIBLE_LONG = "ELIGIBLE_LONG"
ELIGIBLE_SHORT = "ELIGIBLE_SHORT"
RELEASED = "RELEASED"

TIME_OF_DAY_VALUES = {
    "BEFORE_MARKET",
    "DURING_MARKET",
    "AFTER_MARKET",
    "UNSPECIFIED",
}
DOCUMENT_TYPES = {"release", "call_prepared", "call_qa", "call_complete"}
DIRECTIONS = {"bullish", "bearish", "mixed", "neutral", "unknown"}
MARKET_CONFIRMATIONS = {"confirmed", "contradicted", "unavailable"}


def _aware(value: dt.datetime | str) -> dt.datetime:
    """Return an aware Eastern datetime; reject ambiguous historical times."""

    parsed = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(parsed, dt.datetime):
        raise TypeError(f"expected datetime, got {type(value).__name__}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    return parsed.astimezone(EASTERN)


def _date(value: dt.date | dt.datetime | str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _storage_time(value: dt.datetime | str) -> str:
    """Canonical UTC string so SQLite ordering remains correct across DST."""

    return _aware(value).astimezone(dt.timezone.utc).isoformat()


def _previous_weekday(value: dt.date) -> dt.date:
    value -= dt.timedelta(days=1)
    while value.weekday() >= 5:
        value -= dt.timedelta(days=1)
    return value


def _next_weekday(value: dt.date) -> dt.date:
    value += dt.timedelta(days=1)
    while value.weekday() >= 5:
        value += dt.timedelta(days=1)
    return value


@dataclass(frozen=True)
class EarningsEvent:
    event_id: str
    symbol: str
    report_date: dt.date
    time_of_day: str = "UNSPECIFIED"
    status: str = "UNCONFIRMED"
    scheduled_at: dt.datetime | None = None
    release_at: dt.datetime | None = None
    call_at: dt.datetime | None = None
    source: str = "manual"
    source_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        symbol = self.symbol.strip().upper()
        time_of_day = self.time_of_day.strip().upper()
        if not symbol:
            raise ValueError("earnings event requires a symbol")
        if time_of_day not in TIME_OF_DAY_VALUES:
            raise ValueError(f"unsupported earnings time_of_day: {time_of_day}")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "time_of_day", time_of_day)
        object.__setattr__(self, "report_date", _date(self.report_date))
        for name in ("scheduled_at", "release_at", "call_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _aware(value))


@dataclass(frozen=True)
class EarningsAssessment:
    assessment_id: str
    event_id: str
    symbol: str
    document_type: str
    available_at: dt.datetime
    assessed_at: dt.datetime
    direction: str
    confidence: float
    market_confirmation: str = "unavailable"
    fields: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[Mapping[str, Any], ...] = ()
    missing_required_fields: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    parser_version: str = "earnings-v1"
    model_id: str = "unknown"
    source_url: str | None = None
    source_hash: str | None = None

    def __post_init__(self):
        symbol = self.symbol.strip().upper()
        document_type = self.document_type.strip().lower()
        direction = self.direction.strip().lower()
        market_confirmation = self.market_confirmation.strip().lower()
        if not symbol:
            raise ValueError("earnings assessment requires a symbol")
        if document_type not in DOCUMENT_TYPES:
            raise ValueError(f"unsupported document_type: {document_type}")
        if direction not in DIRECTIONS:
            raise ValueError(f"unsupported direction: {direction}")
        if market_confirmation not in MARKET_CONFIRMATIONS:
            raise ValueError(
                f"unsupported market_confirmation: {market_confirmation}"
            )
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.event_id:
            raise ValueError("earnings assessment requires event_id")
        if not isinstance(self.fields, Mapping):
            raise ValueError("fields must be an object")
        if not isinstance(self.evidence, tuple):
            object.__setattr__(self, "evidence", tuple(self.evidence))
        if any(not isinstance(item, Mapping) for item in self.evidence):
            raise ValueError("every evidence item must be an object")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "document_type", document_type)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "market_confirmation", market_confirmation)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "available_at", _aware(self.available_at))
        object.__setattr__(self, "assessed_at", _aware(self.assessed_at))


def assessment_from_mapping(payload: Mapping[str, Any]) -> EarningsAssessment:
    """Validate a parser result before it can reach the strategy state machine."""

    required = {
        "event_id",
        "symbol",
        "document_type",
        "available_at",
        "direction",
        "confidence",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("missing assessment fields: " + ", ".join(missing))
    now = payload.get("assessed_at") or dt.datetime.now(EASTERN)
    source_text = payload.get("source_text")
    source_hash = payload.get("source_hash")
    if source_hash is None and isinstance(source_text, str):
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return EarningsAssessment(
        assessment_id=str(payload.get("assessment_id") or uuid.uuid4()),
        event_id=str(payload["event_id"]),
        symbol=str(payload["symbol"]),
        document_type=str(payload["document_type"]),
        available_at=payload["available_at"],
        assessed_at=now,
        direction=str(payload["direction"]),
        confidence=float(payload["confidence"]),
        market_confirmation=str(payload.get("market_confirmation", "unavailable")),
        fields=dict(payload.get("fields") or {}),
        evidence=tuple(payload.get("evidence") or ()),
        missing_required_fields=tuple(payload.get("missing_required_fields") or ()),
        contradictions=tuple(payload.get("contradictions") or ()),
        parser_version=str(payload.get("parser_version") or "earnings-v1"),
        model_id=str(payload.get("model_id") or "unknown"),
        source_url=payload.get("source_url"),
        source_hash=source_hash,
    )


class EarningsEventStore:
    """Append-oriented SQLite journal with short-lived Windows-safe connections."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS earnings_events (
                    event_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    report_date TEXT NOT NULL,
                    time_of_day TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scheduled_at TEXT,
                    release_at TEXT,
                    call_at TEXT,
                    source TEXT NOT NULL,
                    source_payload_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_earnings_events_symbol_date
                    ON earnings_events(symbol, report_date);

                CREATE TABLE IF NOT EXISTS earnings_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    assessed_at TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    market_confirmation TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    missing_required_fields_json TEXT NOT NULL,
                    contradictions_json TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    source_url TEXT,
                    source_hash TEXT,
                    FOREIGN KEY(event_id) REFERENCES earnings_events(event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_earnings_assessments_visible
                    ON earnings_assessments(symbol, available_at, assessed_at);

                CREATE TABLE IF NOT EXISTS earnings_state_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    symbol TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_earnings_state_symbol_time
                    ON earnings_state_log(symbol, observed_at);
                """
            )

    def upsert_event(self, event: EarningsEvent):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO earnings_events (
                    event_id, symbol, report_date, time_of_day, status,
                    scheduled_at, release_at, call_at, source,
                    source_payload_json, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    symbol=excluded.symbol,
                    report_date=excluded.report_date,
                    time_of_day=excluded.time_of_day,
                    status=excluded.status,
                    scheduled_at=excluded.scheduled_at,
                    release_at=COALESCE(excluded.release_at, earnings_events.release_at),
                    call_at=COALESCE(excluded.call_at, earnings_events.call_at),
                    source=excluded.source,
                    source_payload_json=excluded.source_payload_json,
                    observed_at=excluded.observed_at
                """,
                (
                    event.event_id,
                    event.symbol,
                    event.report_date.isoformat(),
                    event.time_of_day,
                    event.status,
                    event.scheduled_at.isoformat() if event.scheduled_at else None,
                    event.release_at.isoformat() if event.release_at else None,
                    event.call_at.isoformat() if event.call_at else None,
                    event.source,
                    json.dumps(dict(event.source_payload), sort_keys=True),
                    dt.datetime.now(EASTERN).isoformat(),
                ),
            )

    def record_assessment(self, assessment: EarningsAssessment):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO earnings_assessments (
                    assessment_id, event_id, symbol, document_type,
                    available_at, assessed_at, direction, confidence,
                    market_confirmation, fields_json, evidence_json,
                    missing_required_fields_json, contradictions_json,
                    parser_version, model_id, source_url, source_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment.assessment_id,
                    assessment.event_id,
                    assessment.symbol,
                    assessment.document_type,
                    _storage_time(assessment.available_at),
                    _storage_time(assessment.assessed_at),
                    assessment.direction,
                    assessment.confidence,
                    assessment.market_confirmation,
                    json.dumps(dict(assessment.fields), sort_keys=True),
                    json.dumps(list(assessment.evidence), sort_keys=True),
                    json.dumps(list(assessment.missing_required_fields)),
                    json.dumps(list(assessment.contradictions)),
                    assessment.parser_version,
                    assessment.model_id,
                    assessment.source_url,
                    assessment.source_hash,
                ),
            )

    def events(self, symbol: str | None = None) -> list[EarningsEvent]:
        query = "SELECT * FROM earnings_events"
        params: tuple[Any, ...] = ()
        if symbol:
            query += " WHERE symbol=?"
            params = (symbol.strip().upper(),)
        query += " ORDER BY report_date, symbol"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._event_from_row(row) for row in rows]

    def latest_assessment(
        self,
        symbol: str,
        visible_at: dt.datetime,
        *,
        event_id: str | None = None,
    ) -> EarningsAssessment | None:
        """Return only assessments whose source was available by ``visible_at``."""

        query = """
            SELECT * FROM earnings_assessments
            WHERE symbol=? AND available_at<=? AND assessed_at<=?
        """
        params: list[Any] = [
            symbol.strip().upper(),
            _storage_time(visible_at),
            _storage_time(visible_at),
        ]
        if event_id:
            query += " AND event_id=?"
            params.append(event_id)
        query += " ORDER BY available_at DESC, assessed_at DESC LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return self._assessment_from_row(row) if row else None

    def assessment(self, assessment_id: str) -> EarningsAssessment | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM earnings_assessments WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
        return self._assessment_from_row(row) if row else None

    def log_state(
        self,
        *,
        symbol: str,
        observed_at: dt.datetime,
        state: str,
        reason: str,
        event_id: str | None,
    ):
        with self.connect() as conn:
            previous = conn.execute(
                """
                SELECT event_id, state, reason FROM earnings_state_log
                WHERE symbol=? ORDER BY id DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()
            if previous and (
                previous["event_id"], previous["state"], previous["reason"]
            ) == (event_id, state, reason):
                return
            conn.execute(
                """
                INSERT INTO earnings_state_log
                    (event_id, symbol, observed_at, state, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    symbol,
                    _storage_time(observed_at),
                    state,
                    reason,
                ),
            )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EarningsEvent:
        return EarningsEvent(
            event_id=row["event_id"],
            symbol=row["symbol"],
            report_date=row["report_date"],
            time_of_day=row["time_of_day"],
            status=row["status"],
            scheduled_at=row["scheduled_at"],
            release_at=row["release_at"],
            call_at=row["call_at"],
            source=row["source"],
            source_payload=json.loads(row["source_payload_json"] or "{}"),
        )

    @staticmethod
    def _assessment_from_row(row: sqlite3.Row) -> EarningsAssessment:
        return EarningsAssessment(
            assessment_id=row["assessment_id"],
            event_id=row["event_id"],
            symbol=row["symbol"],
            document_type=row["document_type"],
            available_at=row["available_at"],
            assessed_at=row["assessed_at"],
            direction=row["direction"],
            confidence=row["confidence"],
            market_confirmation=row["market_confirmation"],
            fields=json.loads(row["fields_json"] or "{}"),
            evidence=tuple(json.loads(row["evidence_json"] or "[]")),
            missing_required_fields=tuple(
                json.loads(row["missing_required_fields_json"] or "[]")
            ),
            contradictions=tuple(json.loads(row["contradictions_json"] or "[]")),
            parser_version=row["parser_version"],
            model_id=row["model_id"],
            source_url=row["source_url"],
            source_hash=row["source_hash"],
        )


class EarningsEngine:
    """Read-only strategy interface plus observer journaling.

    ``shadow_signal`` is informational.  ``ibkr2.py`` never routes it to
    ``execute_trade``.  The only live behavior this engine can authorize is an
    explicit, entry-only embargo.
    """

    def __init__(
        self,
        store: EarningsEventStore,
        *,
        enabled: bool = False,
        embargo_enabled: bool = False,
        legacy_dates_path: str | Path | None = None,
        confidence_floor: float = 0.75,
    ):
        self.store = store
        self.enabled = bool(enabled)
        self.embargo_enabled = bool(embargo_enabled) and self.enabled
        self.confidence_floor = float(confidence_floor)
        if not 0 <= self.confidence_floor <= 1:
            raise ValueError("confidence_floor must be between 0 and 1")
        if legacy_dates_path:
            self.import_legacy_dates(legacy_dates_path)
        self._reload_events()

    def _reload_events(self):
        self._events_by_symbol: dict[str, list[EarningsEvent]] = {}
        for event in self.store.events():
            self._events_by_symbol.setdefault(event.symbol, []).append(event)

    def import_legacy_dates(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for raw_symbol, dates in payload.items():
            symbol = str(raw_symbol).strip().upper()
            for raw_date in dates:
                report_date = _date(raw_date)
                self.store.upsert_event(
                    EarningsEvent(
                        event_id=f"legacy:{symbol}:{report_date.isoformat()}",
                        symbol=symbol,
                        report_date=report_date,
                        time_of_day="UNSPECIFIED",
                        status="UNCONFIRMED",
                        source="earnings_dates.json",
                        source_payload={"raw_date": str(raw_date)},
                    )
                )

    def ingest_event(self, event: EarningsEvent):
        self.store.upsert_event(event)
        self._reload_events()

    def ingest_assessment(self, payload: Mapping[str, Any]) -> EarningsAssessment:
        assessment = assessment_from_mapping(payload)
        known_ids = {
            event.event_id
            for event in self._events_by_symbol.get(assessment.symbol, ())
        }
        if assessment.event_id not in known_ids:
            raise ValueError(
                f"assessment references unknown event_id {assessment.event_id!r}"
            )
        self.store.record_assessment(assessment)
        return assessment

    def record_market_confirmation(
        self,
        assessment_id: str,
        confirmation: str,
        observed_at: dt.datetime,
        *,
        market_fields: Mapping[str, Any] | None = None,
    ) -> EarningsAssessment:
        """Append a timestamped market-reaction revision to a parser result."""

        original = self.store.assessment(assessment_id)
        if original is None:
            raise ValueError(f"unknown assessment_id {assessment_id!r}")
        confirmation = confirmation.strip().lower()
        if confirmation not in MARKET_CONFIRMATIONS:
            raise ValueError(f"unsupported market confirmation: {confirmation}")
        fields = dict(original.fields)
        fields["market_observation"] = dict(market_fields or {})
        fields["parent_assessment_id"] = original.assessment_id
        revision = EarningsAssessment(
            assessment_id=str(uuid.uuid4()),
            event_id=original.event_id,
            symbol=original.symbol,
            document_type=original.document_type,
            available_at=_aware(observed_at),
            assessed_at=_aware(observed_at),
            direction=original.direction,
            confidence=original.confidence,
            market_confirmation=confirmation,
            fields=fields,
            evidence=original.evidence,
            missing_required_fields=original.missing_required_fields,
            contradictions=original.contradictions,
            parser_version=original.parser_version,
            model_id=original.model_id,
            source_url=original.source_url,
            source_hash=original.source_hash,
        )
        self.store.record_assessment(revision)
        return revision

    def event_near(self, symbol: str, now: dt.datetime) -> EarningsEvent | None:
        now = _aware(now)
        candidates = self._events_by_symbol.get(symbol.strip().upper(), ())
        if not candidates:
            return None
        nearby = [
            event
            for event in candidates
            if abs((event.report_date - now.date()).days) <= 4
        ]
        if not nearby:
            return None
        # A confirmed WSH record and its BEFORE/AFTER classification must win
        # over a legacy date-only row for the same symbol/date.
        return min(
            nearby,
            key=lambda event: (
                abs((event.report_date - now.date()).days),
                event.time_of_day == "UNSPECIFIED",
                event.status != "CONFIRMED",
            ),
        )

    @staticmethod
    def embargo_window(event: EarningsEvent) -> tuple[dt.datetime, dt.datetime]:
        report_date = event.report_date
        if event.time_of_day == "AFTER_MARKET":
            start = dt.datetime.combine(report_date, dt.time(15, 45), EASTERN)
            end = dt.datetime.combine(
                _next_weekday(report_date), dt.time(10, 30), EASTERN
            )
        elif event.time_of_day == "BEFORE_MARKET":
            start = dt.datetime.combine(
                _previous_weekday(report_date), dt.time(15, 45), EASTERN
            )
            end = dt.datetime.combine(report_date, dt.time(10, 30), EASTERN)
        elif event.time_of_day == "DURING_MARKET":
            start = dt.datetime.combine(report_date, dt.time(9, 30), EASTERN)
            end = dt.datetime.combine(report_date, dt.time(16, 0), EASTERN)
        else:
            # A date without a confirmed time is too ambiguous to infer a
            # prior-day close.  Conservatively block new entries on that date.
            start = dt.datetime.combine(report_date, dt.time.min, EASTERN)
            end = dt.datetime.combine(report_date, dt.time.max, EASTERN)
        return start, end

    def should_block_entry(
        self, symbol: str, now: dt.datetime
    ) -> tuple[bool, str]:
        if not self.enabled or not self.embargo_enabled:
            return False, ""
        event = self.event_near(symbol, now)
        if event is None:
            return False, ""
        current = _aware(now)
        start, end = self.embargo_window(event)
        if start <= current <= end:
            return True, (
                f"earnings embargo {event.time_of_day} "
                f"({event.report_date.isoformat()}, {event.status})"
            )
        return False, ""

    def state_at(self, symbol: str, now: dt.datetime) -> tuple[str, str, str | None]:
        if not self.enabled:
            return RELEASED, "earnings observer disabled", None
        current = _aware(now)
        event = self.event_near(symbol, current)
        if event is None:
            return RELEASED, "no nearby earnings event", None
        start, end = self.embargo_window(event)
        latest = self.store.latest_assessment(
            symbol, current, event_id=event.event_id
        )

        if current < start:
            return RELEASED, "earnings window has not started", event.event_id
        if latest is None:
            if event.release_at and current >= event.release_at:
                return WAITING_FOR_RELEASE, "release available; parser pending", event.event_id
            if current <= end:
                return EMBARGO, "scheduled earnings entry embargo", event.event_id
            return HOLD, "event passed without a point-in-time assessment", event.event_id

        if event.call_at and latest.document_type == "release":
            if current < event.call_at:
                reason = "release parsed; scheduled call still pending"
            else:
                reason = "call began; call assessment pending"
            return WAITING_FOR_CALL, reason, event.event_id
        if latest.contradictions:
            return HOLD, "assessment contains contradictions", event.event_id
        if latest.missing_required_fields:
            return HOLD, "assessment is missing required fields", event.event_id
        if latest.confidence < self.confidence_floor:
            return HOLD, "assessment confidence below threshold", event.event_id
        if latest.market_confirmation != "confirmed":
            return HOLD, "market reaction has not confirmed the assessment", event.event_id
        if latest.direction == "bullish":
            return ELIGIBLE_LONG, "shadow long candidate", event.event_id
        if latest.direction == "bearish":
            return ELIGIBLE_SHORT, "shadow short candidate", event.event_id
        return HOLD, f"non-directional assessment: {latest.direction}", event.event_id

    def shadow_signal(self, symbol: str, now: dt.datetime) -> str:
        state, _, _ = self.state_at(symbol, now)
        if state == ELIGIBLE_LONG:
            return "BUY"
        if state == ELIGIBLE_SHORT:
            return "SELL"
        return "HOLD"

    def on_timer(self, now: dt.datetime, symbols: Iterable[str]):
        """Journal state transitions; never fetch data and never submit orders."""

        if not self.enabled:
            return
        for symbol in symbols:
            state, reason, event_id = self.state_at(symbol, now)
            if event_id is not None:
                self.store.log_state(
                    symbol=symbol.strip().upper(),
                    observed_at=now,
                    state=state,
                    reason=reason,
                    event_id=event_id,
                )


__all__ = [
    "EarningsAssessment",
    "EarningsEngine",
    "EarningsEvent",
    "EarningsEventStore",
    "assessment_from_mapping",
    "EMBARGO",
    "WAITING_FOR_RELEASE",
    "WAITING_FOR_CALL",
    "HOLD",
    "ELIGIBLE_LONG",
    "ELIGIBLE_SHORT",
    "RELEASED",
]
