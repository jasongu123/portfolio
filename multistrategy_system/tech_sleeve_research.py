"""Cost-aware research entry point for the isolated technology sleeve.

This script deliberately does not place orders or download data.  It runs the
same stock backtester used by the core equity system on a disjoint technology
universe.  Obtain IBKR one-minute and daily files first, then evaluate the
sleeve standalone before setting ``TECH_SLEEVE_ENABLED=true``.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from tech_sleeve_live import TECH_SYMBOLS


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "backtest_data"


def missing_data_files(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    symbols=TECH_SYMBOLS,
) -> list[str]:
    root = Path(data_dir)
    missing: list[str] = []
    for symbol in symbols:
        for suffix in ("1m.csv", "daily.csv"):
            path = root / f"{symbol}_{suffix}"
            if not path.exists():
                missing.append(str(path))
    return missing


def run_research(
    start_date: str,
    end_date: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    long_only: bool = False,
):
    missing = missing_data_files(data_dir)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Tech-sleeve research data is incomplete:\n"
            f"{formatted}\n\n"
            "With TWS paper connected, fetch data in a separate process:\n"
            "  py -3 -c \"from ibkr_backtest import extract_ibkr_data; "
            "extract_ibkr_data(['INTC','AMD','AVGO','QCOM','MU'], "
            "years_back=6, timeframe='1 min')\""
        )

    # Keep the heavy backtester import out of module import so file checks and
    # unit tests do not initialize plotting/IBKR dependencies.
    from ibkr_backtest import StrategyBacktester

    backtester = StrategyBacktester(
        symbols=list(TECH_SYMBOLS),
        start_date=start_date,
        end_date=end_date,
        ml_mode="rf",
        use_stock_strategies=True,
        stock_strategy_mode="hybrid",
    )
    backtester.load_from_csv(data_dir=str(data_dir))
    backtester.run_backtest(long_only=long_only)
    backtester.print_report()
    return backtester


def _parse_args(argv=None):
    today = dt.date.today()
    parser = argparse.ArgumentParser(
        description="Backtest the disjoint INTC/AMD/AVGO/QCOM/MU tech sleeve."
    )
    parser.add_argument(
        "--start",
        default=str(today.replace(year=today.year - 3)),
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default=str(today),
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing IBKR CSV files.",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Disable short entries for this research run.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Only check whether all required CSV files exist.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    missing = missing_data_files(args.data_dir)
    if args.check_data:
        if missing:
            print("Missing tech-sleeve data:")
            for path in missing:
                print(f"  {path}")
            return 1
        print("Tech-sleeve data files are present.")
        return 0

    try:
        run_research(
            args.start,
            args.end,
            data_dir=args.data_dir,
            long_only=args.long_only,
        )
        return 0
    except FileNotFoundError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

