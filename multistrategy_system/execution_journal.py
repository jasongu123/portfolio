"""Best-effort, observe-only journal for live order execution costs.

The journal is deliberately isolated from order submission.  Public write
methods catch persistence and validation errors, expose the last failure via
``status()``, and return ``False`` rather than raising into a trading callback.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS order_intents (
    client_id INTEGER NOT NULL,
    order_id INTEGER NOT NULL,
    strategy TEXT,
    sleeve TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    decision_timestamp TEXT NOT NULL,
    bid REAL,
    ask REAL,
    mid REAL,
    quoted_spread_bps REAL,
    order_type TEXT,
    limit_price REAL,
    predicted_commission REAL,
    predicted_spread_cost REAL,
    predicted_slippage_cost REAL,
    predicted_impact_cost REAL,
    predicted_total_cost REAL,
    status TEXT NOT NULL DEFAULT 'recorded',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (client_id, order_id)
);

CREATE TABLE IF NOT EXISTS execution_costs (
    exec_id TEXT PRIMARY KEY,
    client_id INTEGER,
    order_id INTEGER,
    perm_id INTEGER,
    symbol TEXT,
    side TEXT,
    shares REAL,
    fill_price REAL,
    fill_timestamp TEXT,
    arrival_mid REAL,
    signed_slippage_bps REAL,
    signed_slippage_dollars REAL,
    commission REAL,
    realized_total_cost REAL,
    status TEXT NOT NULL DEFAULT 'pending_execution',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_execution_costs_order
    ON execution_costs(client_id, order_id);
"""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _timestamp(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _number(value: Any, field_name: str, *, optional: bool = False) -> float | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{field_name} is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_number(value: Any, field_name: str) -> float:
    result = _number(value, field_name)
    assert result is not None
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _side(value: Any) -> str:
    normalized = str(value).strip().upper()
    try:
        return {
            "BUY": "BUY",
            "BOT": "BUY",
            "SELL": "SELL",
            "SLD": "SELL",
        }[normalized]
    except KeyError as exc:
        raise ValueError("side must be BUY, SELL, BOT, or SLD") from exc


def _text(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _slippage(
    side: str,
    shares: float,
    fill_price: float,
    arrival_mid: float | None,
) -> tuple[float | None, float | None]:
    if arrival_mid is None or arrival_mid <= 0:
        return None, None
    direction = 1.0 if side == "BUY" else -1.0
    signed_price_difference = direction * (fill_price - arrival_mid)
    return (
        signed_price_difference / arrival_mid * 10_000.0,
        signed_price_difference * shares,
    )


class ExecutionCostJournal:
    """Thread-safe SQLite journal that never participates in order decisions."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._healthy = False
        self._last_error: str | None = None

        try:
            if self.db_path != ":memory:":
                resolved = Path(self.db_path).expanduser()
                resolved.parent.mkdir(parents=True, exist_ok=True)
                self.db_path = str(resolved)
            self._conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate_schema(self._conn)
            self._conn.commit()
            self._healthy = True
        except Exception as exc:  # The observer must not block live trading.
            self._remember_error("initialize", exc)
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._healthy

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def status(self) -> dict[str, Any]:
        """Return health information without touching the database."""

        with self._lock:
            return {
                "healthy": self._healthy,
                "last_error": self._last_error,
                "db_path": self.db_path,
            }

    def close(self) -> None:
        """Close the journal; repeated calls are harmless."""

        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception as exc:
                    self._remember_error("close", exc)
                finally:
                    self._conn = None
            self._healthy = False

    def __enter__(self) -> ExecutionCostJournal:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def record_order_intent(
        self,
        client_id: int,
        order_id: int,
        strategy: str | None = None,
        sleeve: str | None = None,
        symbol: str | None = None,
        side: str | None = None,
        qty: float | None = None,
        decision_timestamp: Any = None,
        bid: float | None = None,
        ask: float | None = None,
        mid: float | None = None,
        quoted_spread_bps: float | None = None,
        order_type: str | None = None,
        limit_price: float | None = None,
        predicted_commission: float | None = None,
        predicted_spread_cost: float | None = None,
        predicted_slippage_cost: float | None = None,
        predicted_impact_cost: float | None = None,
        predicted_total_cost: float | None = None,
        **aliases: Any,
    ) -> bool:
        """Record the market state and predicted costs at the order decision.

        The first observation for a ``(client_id, order_id)`` pair is retained.
        Repeated callbacks are therefore idempotent.  Common short aliases are
        accepted to make integration with callback objects less brittle.
        """

        try:
            decision_timestamp = self._take_alias(
                decision_timestamp, aliases, "decision_ts"
            )
            qty = self._take_alias(qty, aliases, "quantity")
            predicted_spread_cost = self._take_alias(
                predicted_spread_cost,
                aliases,
                "predicted_spread",
                "predicted_spread_dollars",
            )
            predicted_slippage_cost = self._take_alias(
                predicted_slippage_cost,
                aliases,
                "predicted_slippage",
                "predicted_slippage_dollars",
            )
            predicted_impact_cost = self._take_alias(
                predicted_impact_cost,
                aliases,
                "predicted_impact",
                "predicted_impact_dollars",
            )
            predicted_total_cost = self._take_alias(
                predicted_total_cost,
                aliases,
                "predicted_total",
                "predicted_total_dollars",
            )
            if aliases:
                raise TypeError(f"unexpected fields: {', '.join(sorted(aliases))}")

            normalized_client_id = int(client_id)
            normalized_order_id = int(order_id)
            normalized_symbol = _text(symbol, "symbol").upper()
            normalized_side = _side(side)
            normalized_qty = _positive_number(qty, "qty")
            normalized_timestamp = _timestamp(
                decision_timestamp, "decision_timestamp"
            )
            normalized_bid = _number(bid, "bid", optional=True)
            normalized_ask = _number(ask, "ask", optional=True)
            normalized_mid = _number(mid, "mid", optional=True)
            if (
                normalized_mid is None
                and normalized_bid is not None
                and normalized_ask is not None
            ):
                normalized_mid = (normalized_bid + normalized_ask) / 2.0

            normalized_spread_bps = _number(
                quoted_spread_bps, "quoted_spread_bps", optional=True
            )
            if (
                normalized_spread_bps is None
                and normalized_bid is not None
                and normalized_ask is not None
                and normalized_mid is not None
                and normalized_mid > 0
            ):
                normalized_spread_bps = (
                    (normalized_ask - normalized_bid) / normalized_mid * 10_000.0
                )

            predicted_values = [
                _number(predicted_commission, "predicted_commission", optional=True),
                _number(
                    predicted_spread_cost, "predicted_spread_cost", optional=True
                ),
                _number(
                    predicted_slippage_cost,
                    "predicted_slippage_cost",
                    optional=True,
                ),
                _number(
                    predicted_impact_cost, "predicted_impact_cost", optional=True
                ),
            ]
            normalized_total = _number(
                predicted_total_cost, "predicted_total_cost", optional=True
            )
            if normalized_total is None and any(
                value is not None for value in predicted_values
            ):
                normalized_total = sum(value or 0.0 for value in predicted_values)

            normalized_strategy = None if strategy is None else str(strategy)
            normalized_sleeve = None if sleeve is None else str(sleeve)
            normalized_order_type = (
                None if order_type is None else str(order_type).strip().upper()
            )
            normalized_limit = _number(
                limit_price, "limit_price", optional=True
            )
            now = _utc_now()

            with self._lock:
                conn = self._require_connection()
                with conn:
                    conn.execute(
                        """
                        INSERT INTO order_intents (
                            client_id, order_id, strategy, sleeve, symbol, side,
                            qty, decision_timestamp, bid, ask, mid,
                            quoted_spread_bps, order_type, limit_price,
                            predicted_commission, predicted_spread_cost,
                            predicted_slippage_cost, predicted_impact_cost,
                            predicted_total_cost, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                  ?, ?, ?, ?, ?)
                        ON CONFLICT(client_id, order_id) DO NOTHING
                        """,
                        (
                            normalized_client_id,
                            normalized_order_id,
                            normalized_strategy,
                            normalized_sleeve,
                            normalized_symbol,
                            normalized_side,
                            normalized_qty,
                            normalized_timestamp,
                            normalized_bid,
                            normalized_ask,
                            normalized_mid,
                            normalized_spread_bps,
                            normalized_order_type,
                            normalized_limit,
                            predicted_values[0],
                            predicted_values[1],
                            predicted_values[2],
                            predicted_values[3],
                            normalized_total,
                            now,
                            now,
                        ),
                    )
                    stored = conn.execute(
                        """
                        SELECT symbol, mid
                        FROM order_intents
                        WHERE client_id = ? AND order_id = ?
                        """,
                        (normalized_client_id, normalized_order_id),
                    ).fetchone()
                    if stored is not None:
                        self._reconcile_order_executions(
                            conn,
                            normalized_client_id,
                            normalized_order_id,
                            stored["symbol"],
                            stored["mid"],
                            now,
                        )
            self._remember_success()
            return True
        except Exception as exc:  # The observer must not block live trading.
            self._remember_error("record_order_intent", exc)
            return False

    def record_execution(
        self,
        exec_id: str,
        client_id: int,
        order_id: int,
        symbol: str,
        side: str,
        shares: float,
        fill_price: float,
        fill_time: Any = None,
        perm_id: int | None = None,
        **aliases: Any,
    ) -> bool:
        """Record one fill and calculate its signed arrival-price slippage.

        Positive slippage is adverse for either side: a buy above the arrival
        mid or a sell below it.  Each ``exec_id`` is one immutable fill, so
        separate execution IDs naturally represent partial fills.
        """

        try:
            fill_time = self._take_alias(fill_time, aliases, "fill_timestamp")
            if aliases:
                raise TypeError(f"unexpected fields: {', '.join(sorted(aliases))}")

            normalized_exec_id = _text(exec_id, "exec_id")
            normalized_client_id = int(client_id)
            normalized_order_id = int(order_id)
            normalized_perm_id = None if perm_id is None else int(perm_id)
            normalized_symbol = _text(symbol, "symbol").upper()
            normalized_side = _side(side)
            normalized_shares = _positive_number(shares, "shares")
            normalized_fill = _positive_number(fill_price, "fill_price")
            normalized_fill_time = _timestamp(fill_time, "fill_time")
            now = _utc_now()

            with self._lock:
                conn = self._require_connection()
                with conn:
                    existing = conn.execute(
                        "SELECT * FROM execution_costs WHERE exec_id = ?",
                        (normalized_exec_id,),
                    ).fetchone()
                    if existing is not None and existing["fill_price"] is not None:
                        # IBKR can replay execution callbacks.  Keep the first
                        # fill payload so aggregate costs cannot be doubled or
                        # silently rewritten by a conflicting duplicate.
                        if (
                            existing["perm_id"] is None
                            and normalized_perm_id is not None
                        ):
                            conn.execute(
                                """
                                UPDATE execution_costs
                                SET perm_id = ?, updated_at = ?
                                WHERE exec_id = ?
                                """,
                                (normalized_perm_id, now, normalized_exec_id),
                            )
                        self._remember_success()
                        return True

                    intent = conn.execute(
                        """
                        SELECT symbol, mid
                        FROM order_intents
                        WHERE client_id = ? AND order_id = ?
                        """,
                        (normalized_client_id, normalized_order_id),
                    ).fetchone()
                    symbol_matches = (
                        intent is not None
                        and str(intent["symbol"]).strip().upper() == normalized_symbol
                    )
                    arrival_mid = intent["mid"] if symbol_matches else None
                    slippage_bps, slippage_dollars = _slippage(
                        normalized_side,
                        normalized_shares,
                        normalized_fill,
                        arrival_mid,
                    )
                    commission = None if existing is None else existing["commission"]
                    realized_total = (
                        (slippage_dollars or 0.0) + (commission or 0.0)
                    )
                    status = (
                        "recorded"
                        if slippage_dollars is not None
                        else "missing_order_intent"
                        if intent is None
                        else "order_identity_mismatch"
                        if not symbol_matches
                        else "missing_arrival_mid"
                    )
                    error = (
                        None
                        if status == "recorded"
                        else "No matching order intent"
                        if intent is None
                        else (
                            "Order intent symbol "
                            f"{intent['symbol']} does not match execution symbol "
                            f"{normalized_symbol}"
                        )
                        if not symbol_matches
                        else "Matching order intent has no positive mid"
                    )

                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO execution_costs (
                                exec_id, client_id, order_id, perm_id, symbol, side,
                                shares, fill_price, fill_timestamp, arrival_mid,
                                signed_slippage_bps, signed_slippage_dollars,
                                commission, realized_total_cost, status, error,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                      ?, ?, ?, ?)
                            """,
                            (
                                normalized_exec_id,
                                normalized_client_id,
                                normalized_order_id,
                                normalized_perm_id,
                                normalized_symbol,
                                normalized_side,
                                normalized_shares,
                                normalized_fill,
                                normalized_fill_time,
                                arrival_mid,
                                slippage_bps,
                                slippage_dollars,
                                commission,
                                realized_total,
                                status,
                                error,
                                now,
                                now,
                            ),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE execution_costs
                            SET client_id = ?, order_id = ?, perm_id = ?,
                                symbol = ?, side = ?, shares = ?, fill_price = ?,
                                fill_timestamp = ?,
                                arrival_mid = ?, signed_slippage_bps = ?,
                                signed_slippage_dollars = ?,
                                realized_total_cost = ?, status = ?, error = ?,
                                updated_at = ?
                            WHERE exec_id = ?
                            """,
                            (
                                normalized_client_id,
                                normalized_order_id,
                                normalized_perm_id,
                                normalized_symbol,
                                normalized_side,
                                normalized_shares,
                                normalized_fill,
                                normalized_fill_time,
                                arrival_mid,
                                slippage_bps,
                                slippage_dollars,
                                realized_total,
                                status,
                                error,
                                now,
                                normalized_exec_id,
                            ),
                        )
            self._remember_success()
            return True
        except Exception as exc:  # The observer must not block live trading.
            self._remember_error("record_execution", exc)
            return False

    def record_commission(self, exec_id: str, commission: float) -> bool:
        """Attach a commission before or after its execution callback arrives."""

        try:
            normalized_exec_id = _text(exec_id, "exec_id")
            normalized_commission = _number(commission, "commission")
            assert normalized_commission is not None
            now = _utc_now()

            with self._lock:
                conn = self._require_connection()
                with conn:
                    existing = conn.execute(
                        "SELECT * FROM execution_costs WHERE exec_id = ?",
                        (normalized_exec_id,),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO execution_costs (
                                exec_id, commission, realized_total_cost,
                                status, created_at, updated_at
                            ) VALUES (?, ?, ?, 'pending_execution', ?, ?)
                            """,
                            (
                                normalized_exec_id,
                                normalized_commission,
                                normalized_commission,
                                now,
                                now,
                            ),
                        )
                    else:
                        realized_total = (
                            (existing["signed_slippage_dollars"] or 0.0)
                            + normalized_commission
                        )
                        conn.execute(
                            """
                            UPDATE execution_costs
                            SET commission = ?, realized_total_cost = ?,
                                updated_at = ?
                            WHERE exec_id = ?
                            """,
                            (
                                normalized_commission,
                                realized_total,
                                now,
                                normalized_exec_id,
                            ),
                        )
            self._remember_success()
            return True
        except Exception as exc:  # The observer must not block live trading.
            self._remember_error("record_commission", exc)
            return False

    def fetch_order_intent(
        self, client_id: int, order_id: int
    ) -> dict[str, Any] | None:
        """Return one intent as a plain dictionary, or ``None`` on failure."""

        try:
            with self._lock:
                conn = self._require_connection()
                row = conn.execute(
                    """
                    SELECT * FROM order_intents
                    WHERE client_id = ? AND order_id = ?
                    """,
                    (int(client_id), int(order_id)),
                ).fetchone()
            self._remember_success()
            return None if row is None else dict(row)
        except Exception as exc:
            self._remember_error("fetch_order_intent", exc)
            return None

    def fetch_execution(self, exec_id: str) -> dict[str, Any] | None:
        """Return one execution as a plain dictionary, or ``None`` on failure."""

        try:
            with self._lock:
                conn = self._require_connection()
                row = conn.execute(
                    "SELECT * FROM execution_costs WHERE exec_id = ?",
                    (_text(exec_id, "exec_id"),),
                ).fetchone()
            self._remember_success()
            return None if row is None else dict(row)
        except Exception as exc:
            self._remember_error("fetch_execution", exc)
            return None

    def get_order_summary(self, client_id: int, order_id: int) -> dict[str, Any]:
        """Aggregate all partial fills for a composite order identity."""

        empty = {
            "client_id": int(client_id),
            "order_id": int(order_id),
            "fill_count": 0,
            "filled_shares": 0.0,
            "signed_slippage_dollars": 0.0,
            "commission": 0.0,
            "realized_total_cost": 0.0,
        }
        try:
            with self._lock:
                conn = self._require_connection()
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS fill_count,
                           COALESCE(SUM(shares), 0.0) AS filled_shares,
                           COALESCE(SUM(signed_slippage_dollars), 0.0)
                               AS signed_slippage_dollars,
                           COALESCE(SUM(commission), 0.0) AS commission,
                           COALESCE(SUM(realized_total_cost), 0.0)
                               AS realized_total_cost
                    FROM execution_costs AS e
                    INNER JOIN order_intents AS i
                      ON i.client_id = e.client_id
                     AND i.order_id = e.order_id
                     AND UPPER(i.symbol) = UPPER(e.symbol)
                    WHERE e.client_id = ? AND e.order_id = ?
                      AND e.fill_price IS NOT NULL
                    """,
                    (int(client_id), int(order_id)),
                ).fetchone()
            self._remember_success()
            if row is not None:
                empty.update(dict(row))
            return empty
        except Exception as exc:
            self._remember_error("get_order_summary", exc)
            return empty

    def _reconcile_order_executions(
        self,
        conn: sqlite3.Connection,
        client_id: int,
        order_id: int,
        symbol: str,
        arrival_mid: float | None,
        updated_at: str,
    ) -> None:
        rows = conn.execute(
            """
            SELECT * FROM execution_costs
            WHERE client_id = ? AND order_id = ? AND UPPER(symbol) = UPPER(?)
              AND fill_price IS NOT NULL
            """,
            (client_id, order_id, symbol),
        ).fetchall()
        for row in rows:
            slippage_bps, slippage_dollars = _slippage(
                row["side"], row["shares"], row["fill_price"], arrival_mid
            )
            realized_total = (
                (slippage_dollars or 0.0) + (row["commission"] or 0.0)
            )
            status = "recorded" if slippage_dollars is not None else "missing_arrival_mid"
            error = (
                None
                if slippage_dollars is not None
                else "Matching order intent has no positive mid"
            )
            conn.execute(
                """
                UPDATE execution_costs
                SET arrival_mid = ?, signed_slippage_bps = ?,
                    signed_slippage_dollars = ?, realized_total_cost = ?,
                    status = ?, error = ?, updated_at = ?
                WHERE exec_id = ?
                """,
                (
                    arrival_mid,
                    slippage_bps,
                    slippage_dollars,
                    realized_total,
                    status,
                    error,
                    updated_at,
                    row["exec_id"],
                ),
            )

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Apply additive migrations required by older journal databases."""

        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(execution_costs)")
        }
        if "perm_id" not in columns:
            conn.execute("ALTER TABLE execution_costs ADD COLUMN perm_id INTEGER")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_execution_costs_perm
            ON execution_costs(perm_id)
            """
        )

    def _require_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("execution journal is unavailable or closed")
        return self._conn

    @staticmethod
    def _take_alias(value: Any, aliases: dict[str, Any], *names: str) -> Any:
        for name in names:
            if name in aliases:
                alias_value = aliases.pop(name)
                if value is None:
                    value = alias_value
        return value

    def _remember_success(self) -> None:
        with self._lock:
            self._healthy = self._conn is not None
            if self._healthy:
                self._last_error = None

    def _remember_error(self, operation: str, exc: Exception) -> None:
        message = f"{operation}: {type(exc).__name__}: {exc}"
        with self._lock:
            self._healthy = False
            self._last_error = message
        logger.warning("Execution journal %s", message)
