"""Read-only summary of the observe-only execution cost journal."""

from __future__ import annotations

import argparse
from contextlib import closing
import datetime as dt
from pathlib import Path
import sqlite3
from typing import Any, Sequence


DEFAULT_DB_PATH = Path(__file__).resolve().with_name("trading_system.db")
DEFAULT_DAYS = 30
_REQUIRED_TABLES = {"order_intents", "execution_costs"}


def _positive_days(value: Any) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("days must be a positive integer") from exc
    if days <= 0:
        raise argparse.ArgumentTypeError("days must be a positive integer")
    return days


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    """Open an existing SQLite database without permitting writes."""

    uri = f"{db_path.expanduser().resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _cutoff_timestamp(days: int, now: dt.datetime | None) -> str:
    reference = now or dt.datetime.now(dt.timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=dt.timezone.utc)
    else:
        reference = reference.astimezone(dt.timezone.utc)
    return (reference - dt.timedelta(days=days)).isoformat()


def summarize_execution_costs(
    db_path: str | Path = DEFAULT_DB_PATH,
    days: int = DEFAULT_DAYS,
    *,
    now: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Return recent fill-cost aggregates grouped by sleeve, strategy, symbol.

    Missing database files and databases without the journal tables are treated
    as empty journals. SQLite errors from malformed or incompatible databases
    are left to the CLI to turn into a clean diagnostic.
    """

    try:
        window_days = _positive_days(days)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc

    path = Path(db_path).expanduser()
    if not path.is_file():
        return []

    cutoff = _cutoff_timestamp(window_days, now)

    with closing(_read_only_connection(path)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not _REQUIRED_TABLES.issubset(tables):
            return []

        rows = connection.execute(
            """
            WITH fills AS (
                SELECT
                    COALESCE(NULLIF(TRIM(i.sleeve), ''), '(unknown)') AS sleeve,
                    COALESCE(NULLIF(TRIM(i.strategy), ''), '(unknown)') AS strategy,
                    UPPER(TRIM(i.symbol)) AS symbol,
                    ABS(e.shares) * e.fill_price AS filled_notional,
                    e.signed_slippage_bps AS signed_slippage_bps,
                    COALESCE(e.commission, 0.0) AS commission,
                    COALESCE(e.realized_total_cost, 0.0) AS realized_total_cost,
                    CASE
                        WHEN e.arrival_mid IS NULL OR e.arrival_mid <= 0 THEN 1
                        ELSE 0
                    END AS missing_arrival_mid
                FROM execution_costs AS e
                INNER JOIN order_intents AS i
                  ON i.client_id = e.client_id
                 AND i.order_id = e.order_id
                 AND UPPER(TRIM(i.symbol)) = UPPER(TRIM(e.symbol))
                WHERE e.fill_price IS NOT NULL
                  AND e.shares IS NOT NULL
                  AND COALESCE(
                        julianday(e.fill_timestamp),
                        julianday(e.created_at)
                      ) >= julianday(?)
            )
            SELECT
                sleeve,
                strategy,
                symbol,
                COUNT(*) AS fill_count,
                COALESCE(SUM(filled_notional), 0.0) AS filled_notional,
                CASE
                    WHEN SUM(CASE
                        WHEN signed_slippage_bps IS NOT NULL
                        THEN filled_notional ELSE 0.0 END) > 0
                    THEN SUM(CASE
                        WHEN signed_slippage_bps IS NOT NULL
                        THEN signed_slippage_bps * filled_notional ELSE 0.0 END)
                         / SUM(CASE
                            WHEN signed_slippage_bps IS NOT NULL
                            THEN filled_notional ELSE 0.0 END)
                    ELSE NULL
                END AS weighted_signed_slippage_bps,
                COALESCE(SUM(commission), 0.0) AS commissions,
                COALESCE(SUM(realized_total_cost), 0.0) AS realized_total_cost,
                COALESCE(SUM(missing_arrival_mid), 0) AS missing_arrival_mid_count
            FROM fills
            GROUP BY sleeve, strategy, symbol
            ORDER BY sleeve, strategy, symbol
            """,
            (cutoff,),
        ).fetchall()

    return [dict(row) for row in rows]


def count_quarantined_execution_fills(
    db_path: str | Path = DEFAULT_DB_PATH,
    days: int = DEFAULT_DAYS,
    *,
    now: dt.datetime | None = None,
) -> int:
    """Count recent fills lacking an exact client/order/symbol intent match."""

    try:
        window_days = _positive_days(days)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc

    path = Path(db_path).expanduser()
    if not path.is_file():
        return 0
    cutoff = _cutoff_timestamp(window_days, now)

    with closing(_read_only_connection(path)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not _REQUIRED_TABLES.issubset(tables):
            return 0
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM execution_costs AS e
            LEFT JOIN order_intents AS i
              ON i.client_id = e.client_id
             AND i.order_id = e.order_id
             AND UPPER(TRIM(i.symbol)) = UPPER(TRIM(e.symbol))
            WHERE e.fill_price IS NOT NULL
              AND e.shares IS NOT NULL
              AND i.client_id IS NULL
              AND COALESCE(
                    julianday(e.fill_timestamp),
                    julianday(e.created_at)
                  ) >= julianday(?)
            """,
            (cutoff,),
        ).fetchone()
    return 0 if row is None else int(row[0])


def format_report(
    rows: Sequence[dict[str, Any]],
    *,
    days: int,
    db_path: str | Path,
    quarantined_fill_count: int = 0,
) -> str:
    """Render grouped results as a compact fixed-width text table."""

    title = f"Execution costs for the last {days} day(s): {Path(db_path)}"
    headers = (
        "SLEEVE",
        "STRATEGY",
        "SYMBOL",
        "FILLS",
        "FILLED NOTIONAL",
        "SIGNED SLIPPAGE BPS",
        "COMMISSIONS",
        "REALIZED TOTAL COST",
        "MISSING MID",
    )
    rendered_rows = []
    for row in rows:
        weighted_bps = row["weighted_signed_slippage_bps"]
        rendered_rows.append(
            (
                str(row["sleeve"]),
                str(row["strategy"]),
                str(row["symbol"]),
                str(int(row["fill_count"])),
                f"{float(row['filled_notional']):,.2f}",
                "n/a" if weighted_bps is None else f"{float(weighted_bps):.2f}",
                f"{float(row['commissions']):,.2f}",
                f"{float(row['realized_total_cost']):,.2f}",
                str(int(row["missing_arrival_mid_count"])),
            )
        )

    widths = [
        max([len(headers[index]), *(len(row[index]) for row in rendered_rows)])
        for index in range(len(headers))
    ]

    def render(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) if index < 3 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    separator = "  ".join("-" * width for width in widths)
    quarantine = (
        f"Quarantined unmatched/mismatched fills: {quarantined_fill_count}"
    )
    return "\n".join(
        (
            title,
            quarantine,
            render(headers),
            separator,
            *(render(row) for row in rendered_rows),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the read-only execution-cost journal."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"journal database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--days",
        type=_positive_days,
        default=DEFAULT_DAYS,
        help=f"recent calendar-day window (default: {DEFAULT_DAYS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = summarize_execution_costs(args.db, args.days)
        quarantined = count_quarantined_execution_fills(args.db, args.days)
    except (OSError, sqlite3.Error) as exc:
        print(f"Execution journal could not be read: {exc}")
        return 0

    if not rows:
        print(
            f"No execution fills found with an exact intent match in the last "
            f"{args.days} day(s) at {args.db}. "
            f"Quarantined unmatched/mismatched fills: "
            f"{quarantined}."
        )
        return 0

    print(
        format_report(
            rows,
            days=args.days,
            db_path=args.db,
            quarantined_fill_count=quarantined,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
