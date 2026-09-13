"""Command-line entry point for the isolated research data collectors."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from .alpha_vantage import AlphaVantageCollector
from .macro import DEFAULT_FRED_SERIES, TREASURY_DATASETS, MacroCollector
from .storage import DataStore


DEFAULT_EQUITY_SYMBOLS = (
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "NVDA",
    "GOOG",
    "MSFT",
    "META",
    "JPM",
    "WMT",
    "UNH",
    "XOM",
)


def _symbols(comma_separated: str | None, filename: str | None) -> list[str]:
    values: list[str] = []
    if comma_separated:
        values.extend(comma_separated.split(","))
    if filename:
        for line in Path(filename).read_text().splitlines():
            values.extend(line.split(","))
    return list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))


def _print_result(result: object) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def run_macro(args: argparse.Namespace) -> None:
    collector = MacroCollector(DataStore(args.root))
    result: dict[str, int] = {}
    if not args.skip_treasury:
        result["treasury"] = collector.collect_treasury_range(
            args.start_year, args.end_year, args.treasury_datasets
        )
    if not args.skip_fred:
        result["fred"] = collector.collect_fred(args.fred_series)
    if not args.skip_nyfed:
        result["nyfed_sofr"] = collector.collect_nyfed_sofr(args.nyfed_start)
    _print_result(result)


def run_alpha_earnings(args: argparse.Namespace) -> None:
    symbols = _symbols(args.symbols, args.symbols_file) or list(DEFAULT_EQUITY_SYMBOLS)
    collector = AlphaVantageCollector(
        DataStore(args.root), request_pause_seconds=args.pause_seconds
    )
    _print_result(collector.collect_earnings_many(symbols, max_calls=args.max_calls))


def run_alpha_listings(args: argparse.Namespace) -> None:
    collector = AlphaVantageCollector(DataStore(args.root))
    states: Iterable[str] = ("active", "delisted") if args.state == "both" else (args.state,)
    _print_result(
        {state: collector.collect_listings(args.date, state=state) for state in states}
    )


def run_ibkr_news(args: argparse.Namespace) -> None:
    from .ibkr_news import IBKRNewsCollector

    symbols = _symbols(args.symbols, args.symbols_file) or list(DEFAULT_EQUITY_SYMBOLS)
    collector = IBKRNewsCollector(
        DataStore(args.root), host=args.host, port=args.port, client_id=args.client_id
    )
    collector.connect_and_start(timeout=args.connection_timeout)
    try:
        while True:
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            result = collector.collect_symbols(
                symbols,
                lookback_days=args.lookback_days,
                max_results=args.max_results,
                include_bodies=not args.headlines_only,
            )
            _print_result({"started_at": started, "symbols": result})
            if args.interval <= 0:
                break
            time.sleep(args.interval)
    finally:
        collector.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("RESEARCH_DATA_ROOT", "data_lake"),
        help="raw/SQLite data root (default: data_lake)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    macro = subparsers.add_parser("macro", help="collect Treasury, FRED, and NY Fed data")
    macro.add_argument("--start-year", type=int, default=2007)
    macro.add_argument("--end-year", type=int, default=dt.date.today().year)
    macro.add_argument(
        "--treasury-datasets",
        nargs="+",
        choices=tuple(TREASURY_DATASETS),
        default=["nominal", "real"],
    )
    macro.add_argument("--fred-series", nargs="+", default=list(DEFAULT_FRED_SERIES))
    macro.add_argument("--nyfed-start", default="2018-04-03")
    macro.add_argument("--skip-treasury", action="store_true")
    macro.add_argument("--skip-fred", action="store_true")
    macro.add_argument("--skip-nyfed", action="store_true")
    macro.set_defaults(func=run_macro)

    earnings = subparsers.add_parser("alpha-earnings", help="collect earnings histories")
    earnings.add_argument("--symbols", help="comma-separated tickers")
    earnings.add_argument("--symbols-file", help="newline- or comma-separated ticker file")
    earnings.add_argument("--max-calls", type=int, default=25)
    earnings.add_argument("--pause-seconds", type=float, default=1.0)
    earnings.set_defaults(func=run_alpha_earnings)

    listings = subparsers.add_parser(
        "alpha-listings", help="collect point-in-time active/delisted listings"
    )
    listings.add_argument("--date", default=dt.date.today().isoformat())
    listings.add_argument("--state", choices=("active", "delisted", "both"), default="both")
    listings.set_defaults(func=run_alpha_listings)

    news = subparsers.add_parser("ibkr-news", help="archive IBKR headlines and article bodies")
    news.add_argument("--host", default="127.0.0.1")
    news.add_argument("--port", type=int, default=7497, help="7497 paper; commonly 7496 live")
    news.add_argument("--client-id", type=int, default=91)
    news.add_argument("--symbols", help="comma-separated tickers")
    news.add_argument("--symbols-file", help="newline- or comma-separated ticker file")
    news.add_argument("--lookback-days", type=int, default=3)
    news.add_argument("--max-results", type=int, default=100)
    news.add_argument("--headlines-only", action="store_true")
    news.add_argument("--connection-timeout", type=float, default=10.0)
    news.add_argument(
        "--interval",
        type=int,
        default=0,
        help="repeat every N seconds; zero performs one collection",
    )
    news.set_defaults(func=run_ibkr_news)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"collection failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
