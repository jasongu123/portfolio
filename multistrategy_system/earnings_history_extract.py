"""Resumable IBKR extraction for earnings-event windows only.

This is intentionally much smaller than downloading continuous multi-year
minute history.  Each job requests five calendar days ending after the event,
with ``useRTH=0``, then stores one immutable CSV per symbol/event/data type.

Examples (TWS paper port):

    py -3 earnings_history_extract.py --port 7497 --dry-run
    py -3 earnings_history_extract.py --port 7497 --years 3
    py -3 earnings_history_extract.py --port 7497 --types TRADES BID ASK

No order API is used anywhere in this program.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


SCRIPT_DIR = Path(__file__).resolve().parent
EASTERN = ZoneInfo("America/New_York")
DEFAULT_DATE_FILE = SCRIPT_DIR / "earnings_dates.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "earnings_history"
DEFAULT_CLIENT_ID = 221
VALID_DATA_TYPES = {"TRADES", "MIDPOINT", "BID", "ASK"}
FATAL_CODES = {162, 165, 200, 300, 321, 354, 366, 420}
BENIGN_CODES = {2104, 2106, 2107, 2108, 2158, 2100, 2150}


def _next_weekday(value: dt.date) -> dt.date:
    value += dt.timedelta(days=1)
    while value.weekday() >= 5:
        value += dt.timedelta(days=1)
    return value


@dataclass(frozen=True)
class HistoryJob:
    symbol: str
    report_date: dt.date
    data_type: str
    output_path: Path

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.report_date.isoformat()}:{self.data_type}"

    @property
    def end_datetime(self) -> str:
        end_date = _next_weekday(self.report_date)
        return f"{end_date:%Y%m%d} 20:00:00 US/Eastern"


def load_legacy_events(path: str | Path) -> list[tuple[str, dt.date]]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    events = []
    for raw_symbol, raw_dates in payload.items():
        symbol = str(raw_symbol).strip().upper()
        for raw_date in raw_dates:
            events.append((symbol, dt.date.fromisoformat(str(raw_date)[:10])))
    return sorted(set(events), key=lambda item: (item[1], item[0]))


def plan_jobs(
    events: Iterable[tuple[str, dt.date]],
    output_dir: str | Path,
    *,
    data_types: Sequence[str] = ("TRADES", "MIDPOINT"),
    symbols: Iterable[str] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> list[HistoryJob]:
    output_dir = Path(output_dir)
    selected = {value.strip().upper() for value in symbols or ()}
    normalized_types = tuple(value.strip().upper() for value in data_types)
    unsupported = sorted(set(normalized_types) - VALID_DATA_TYPES)
    if unsupported:
        raise ValueError("unsupported data types: " + ", ".join(unsupported))
    jobs = []
    for symbol, report_date in events:
        if selected and symbol not in selected:
            continue
        if start and report_date < start:
            continue
        if end and report_date > end:
            continue
        for data_type in normalized_types:
            path = (
                output_dir
                / "raw"
                / symbol
                / report_date.isoformat()
                / f"{data_type.lower()}_1m.csv"
            )
            jobs.append(HistoryJob(symbol, report_date, data_type, path))
    return jobs


def estimated_minutes(job_count: int, pace_seconds: float) -> float:
    return max(0, job_count) * max(0.0, pace_seconds) / 60.0


class Manifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    job_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    report_date TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    output_path TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def completed(self, job: HistoryJob) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM extraction_jobs WHERE job_key=?", (job.key,)
            ).fetchone()
        return bool(row and row["status"] == "complete" and job.output_path.exists())

    def record(self, job: HistoryJob, status: str, row_count: int, message: str = ""):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO extraction_jobs (
                    job_key, symbol, report_date, data_type, status,
                    row_count, output_path, message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    status=excluded.status,
                    row_count=excluded.row_count,
                    output_path=excluded.output_path,
                    message=excluded.message,
                    updated_at=excluded.updated_at
                """,
                (
                    job.key,
                    job.symbol,
                    job.report_date.isoformat(),
                    job.data_type,
                    status,
                    row_count,
                    str(job.output_path),
                    str(message)[:1000],
                    dt.datetime.now(EASTERN).isoformat(),
                ),
            )


class HistoricalClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = threading.Event()
        self.next_request_id = 1
        self._done: dict[int, threading.Event] = {}
        self._bars: dict[int, list[dict]] = {}
        self._errors: dict[int, str] = {}

    def nextValidId(self, orderId):
        self.connected.set()

    def error(self, reqId, code, msg, misc=None):
        if code in BENIGN_CODES:
            return
        message = f"IBKR {code}: {msg}"
        if reqId in self._done:
            self._errors[reqId] = message
            if code in FATAL_CODES:
                self._done[reqId].set()
        else:
            print(message)

    def historicalData(self, reqId, bar):
        timestamp = self._parse_bar_time(bar.date)
        self._bars.setdefault(reqId, []).append(
            {
                "datetime": timestamp.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "wap": float(bar.wap),
                "count": int(bar.barCount),
            }
        )

    def historicalDataEnd(self, reqId, start, end):
        event = self._done.get(reqId)
        if event:
            event.set()

    @staticmethod
    def _parse_bar_time(value) -> dt.datetime:
        raw = str(value).strip()
        if raw.isdigit():
            return dt.datetime.fromtimestamp(int(raw), tz=dt.timezone.utc).astimezone(
                EASTERN
            )
        for fmt in ("%Y%m%d %H:%M:%S %Z", "%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S"):
            try:
                parsed = dt.datetime.strptime(raw, fmt)
                return parsed.replace(tzinfo=EASTERN)
            except ValueError:
                continue
        raise ValueError(f"unrecognized IBKR bar timestamp: {raw!r}")

    def fetch(self, job: HistoryJob, timeout: float = 90.0) -> tuple[list[dict], str]:
        req_id = self.next_request_id
        self.next_request_id += 1
        self._done[req_id] = threading.Event()
        self._bars[req_id] = []

        contract = Contract()
        contract.symbol = job.symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        self.reqHistoricalData(
            req_id,
            contract,
            job.end_datetime,
            "5 D",
            "1 min",
            job.data_type,
            0,
            2,
            False,
            [],
        )
        finished = self._done[req_id].wait(timeout=timeout)
        if not finished:
            self.cancelHistoricalData(req_id)
            self._errors[req_id] = f"timeout after {timeout:.0f}s"
        rows = sorted(self._bars.get(req_id, ()), key=lambda row: row["datetime"])
        error = self._errors.get(req_id, "")
        self._done.pop(req_id, None)
        self._bars.pop(req_id, None)
        self._errors.pop(req_id, None)
        return rows, error


def write_rows(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    fields = ["datetime", "open", "high", "low", "close", "volume", "wap", "count"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_jobs(
    jobs: Sequence[HistoryJob],
    *,
    host: str,
    port: int,
    client_id: int,
    pace_seconds: float,
    timeout: float,
    overwrite: bool,
) -> tuple[int, int, int]:
    if not jobs:
        return 0, 0, 0
    output_root = jobs[0].output_path.parents[3]
    manifest = Manifest(output_root / "extraction_manifest.sqlite3")
    client = HistoricalClient()
    client.connect(host, port, client_id)
    thread = threading.Thread(target=client.run, daemon=True)
    thread.start()
    if not client.connected.wait(timeout=10):
        client.disconnect()
        raise RuntimeError(
            f"could not connect to TWS/IB Gateway on {host}:{port}; "
            "confirm API access and the paper/live port"
        )

    completed = skipped = failed = 0
    try:
        for index, job in enumerate(jobs, start=1):
            if not overwrite and manifest.completed(job):
                skipped += 1
                print(f"[{index}/{len(jobs)}] SKIP {job.key}")
                continue
            print(f"[{index}/{len(jobs)}] FETCH {job.key}", flush=True)
            manifest.record(job, "running", 0)
            rows, error = client.fetch(job, timeout=timeout)
            if rows:
                write_rows(job.output_path, rows)
                manifest.record(job, "complete", len(rows), error)
                completed += 1
                print(f"  saved {len(rows)} bars -> {job.output_path}")
            else:
                message = error or "IBKR returned no bars"
                manifest.record(job, "failed", 0, message)
                failed += 1
                print(f"  FAILED: {message}")
            if index < len(jobs):
                time.sleep(max(0.0, pace_seconds))
    finally:
        client.disconnect()
    return completed, skipped, failed


def _parse_date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-file", default=str(DEFAULT_DATE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--types", nargs="+", default=["TRADES", "MIDPOINT"])
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--years", type=int)
    parser.add_argument("--pace-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    events = load_legacy_events(args.date_file)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if args.years is not None:
        if args.years <= 0:
            raise SystemExit("--years must be positive")
        cutoff = dt.date.today() - dt.timedelta(days=365 * args.years)
        start = max(filter(None, (start, cutoff)), default=cutoff)
    if args.max_events is not None:
        if args.max_events <= 0:
            raise SystemExit("--max-events must be positive")
        filtered_events = [
            event
            for event in events
            if (not args.symbols or event[0] in {s.upper() for s in args.symbols})
            and (not start or event[1] >= start)
            and (not end or event[1] <= end)
        ][: args.max_events]
    else:
        filtered_events = events
    jobs = plan_jobs(
        filtered_events,
        args.output_dir,
        data_types=args.types,
        symbols=args.symbols,
        start=start,
        end=end,
    )
    unique_events = len({(job.symbol, job.report_date) for job in jobs})
    print(
        f"Planned {len(jobs)} requests for {unique_events} earnings events; "
        f"pacing-only estimate {estimated_minutes(len(jobs), args.pace_seconds):.1f} minutes."
    )
    print("Every request uses 1-minute bars with useRTH=0; no orders are possible.")
    if args.dry_run:
        for job in jobs[:20]:
            print(f"  {job.key} -> {job.output_path}")
        if len(jobs) > 20:
            print(f"  ... {len(jobs) - 20} more requests")
        return 0
    completed, skipped, failed = run_jobs(
        jobs,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        pace_seconds=args.pace_seconds,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )
    print(f"Complete: {completed}; skipped: {skipped}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

