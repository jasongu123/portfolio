"""
Deterministic event-calendar helpers for ML features.

No external data sources needed — all dates are hardcoded annually. Update the
date lists each January with the new year's published schedule.

Features this module supports:
  - OPEX flag (3rd Friday) and day-after-OPEX
  - Days-to-FOMC announcement
  - Days-to-CPI release
  - Days-to-NFP (nonfarm payrolls) release

Sources for updates:
  - FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
  - CPI:  bls.gov/schedule/news_release/cpi.htm
  - NFP:  bls.gov/schedule/news_release/empsit.htm
"""

import datetime
from typing import Iterable

# -----------------------------------------------------------------------------
# Hardcoded event dates. Update annually.
# Format: 'YYYY-MM-DD'.
# -----------------------------------------------------------------------------

FOMC_DATES = [
    # 2024
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
    '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-10',
    # 2026 (announced schedule)
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09',
]

CPI_DATES = [
    # 2024
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15',
    '2024-06-12', '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10',
    '2024-11-13', '2024-12-11',
    # 2025
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13',
    '2025-06-11', '2025-07-15', '2025-08-12', '2025-09-11', '2025-10-15',
    '2025-11-13', '2025-12-10',
    # 2026 (BLS schedule)
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-15', '2026-05-13',
    '2026-06-10', '2026-07-15', '2026-08-12', '2026-09-10', '2026-10-15',
    '2026-11-13', '2026-12-10',
]

NFP_DATES = [
    # 2024
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05', '2024-05-03',
    '2024-06-07', '2024-07-05', '2024-08-02', '2024-09-06', '2024-10-04',
    '2024-11-01', '2024-12-06',
    # 2025
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04', '2025-05-02',
    '2025-06-06', '2025-07-03', '2025-08-01', '2025-09-05', '2025-10-03',
    '2025-11-07', '2025-12-05',
    # 2026 (first Friday of each month — refine when BLS publishes schedule)
    '2026-01-09', '2026-02-06', '2026-03-06', '2026-04-03', '2026-05-01',
    '2026-06-05', '2026-07-02', '2026-08-07', '2026-09-04', '2026-10-02',
    '2026-11-06', '2026-12-04',
]

# Per-symbol EARNINGS dates — the #1 stock-specific gap event. POPULATE these
# (hardcode quarterly, exactly like the macro dates above) for the live universe.
# Sources: company IR / Nasdaq earnings calendar, or pull once via an earnings API
# (yfinance Ticker.earnings_dates, Finnhub/FMP free tier). ETFs have no earnings.
# Empty list = no earnings block for that symbol (the macro blocks still apply).
EARNINGS_DATES = {
    'SPY': [], 'QQQ': [], 'IWM': [],                 # ETFs — leave empty
    'AAPL': [], 'MSFT': [], 'NVDA': [], 'GOOG': [], 'META': [],
    'JPM': [], 'WMT': [], 'UNH': [], 'XOM': [],
}

# Auto-load pulled earnings dates if present (pull_earnings.py writes earnings_dates.json).
try:
    import json as _json
    import os as _os
    _ef = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'earnings_dates.json')
    if _os.path.exists(_ef):
        with open(_ef) as _f:
            for _sym, _dates in _json.load(_f).items():
                EARNINGS_DATES[_sym] = list(_dates)
except Exception as _e:
    print(f"[event_calendar] could not load earnings_dates.json: {_e}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_date(d) -> datetime.date:
    """Coerce input (date, datetime, str, pandas Timestamp) to date."""
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, str):
        return datetime.datetime.strptime(d[:10], '%Y-%m-%d').date()
    # pandas Timestamp duck-types as having .date()
    if hasattr(d, 'date'):
        return d.date()
    raise TypeError(f"Cannot coerce {type(d).__name__} to date: {d!r}")


def is_opex_day(d) -> bool:
    """True iff `d` is a monthly OPEX day (3rd Friday of any month)."""
    dt = _to_date(d)
    return dt.weekday() == 4 and 15 <= dt.day <= 21


def is_day_after_opex(d) -> bool:
    """True iff `d` is the trading day after a monthly OPEX (Monday after 3rd Fri).

    Note: this is a calendar-Monday check, not a "next trading day" check,
    so it won't fire if Monday is a holiday. Good enough for an ML feature.
    """
    dt = _to_date(d)
    if dt.weekday() != 0:  # Monday
        return False
    prev_friday = dt - datetime.timedelta(days=3)
    return is_opex_day(prev_friday)


def days_to_event(d, event_dates: Iterable[str], cap: int = 30) -> int:
    """Days from `d` until the next event in `event_dates`.

    Returns 0 if today is an event day. Returns `cap` if no event in `cap` days
    (or no future events in the list at all).

    `event_dates` is an iterable of 'YYYY-MM-DD' strings.
    """
    today = _to_date(d)
    future = [_to_date(s) for s in event_dates if _to_date(s) >= today]
    if not future:
        return cap
    delta = (min(future) - today).days
    return min(delta, cap)


def days_since_event(d, event_dates: Iterable[str], cap: int = 30) -> int:
    """Days since the most recent past event. Returns `cap` if none within `cap`."""
    today = _to_date(d)
    past = [_to_date(s) for s in event_dates if _to_date(s) <= today]
    if not past:
        return cap
    delta = (today - max(past)).days
    return min(delta, cap)


# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------

def days_to_fomc(d, cap: int = 30) -> int:
    return days_to_event(d, FOMC_DATES, cap=cap)


def days_to_cpi(d, cap: int = 30) -> int:
    return days_to_event(d, CPI_DATES, cap=cap)


def days_to_nfp(d, cap: int = 30) -> int:
    return days_to_event(d, NFP_DATES, cap=cap)


def is_fomc_day(d) -> bool:
    today = _to_date(d)
    return any(_to_date(s) == today for s in FOMC_DATES)


def is_cpi_day(d) -> bool:
    today = _to_date(d)
    return any(_to_date(s) == today for s in CPI_DATES)


def is_nfp_day(d) -> bool:
    today = _to_date(d)
    return any(_to_date(s) == today for s in NFP_DATES)


def is_quad_witch(d) -> bool:
    """True iff `d` is a quarterly quadruple-witching expiration (3rd Friday of
    Mar/Jun/Sep/Dec) — the high-volume / pinning OPEX that distorts intraday action."""
    dt = _to_date(d)
    return dt.month in (3, 6, 9, 12) and dt.weekday() == 4 and 15 <= dt.day <= 21


# -----------------------------------------------------------------------------
# ENTRY FILTER — the deterministic, backtestable replacement for the LLM news gate
# -----------------------------------------------------------------------------
_FOMC_BLOCK_FROM = datetime.time(14, 0)     # 2pm announcement -> close = chop zone


def should_block_entry(symbol, dt):
    """Deterministic event filter for NEW ENTRIES (never call this to block exits).

    Returns (block: bool, reason: str). `dt` = ET datetime (tz-aware or naive).
    Blocks: the symbol's earnings day; FOMC afternoon; quad-witch expiration.

    EVIDENCE-BASED — validated on the equity book 2023-25 (event_filter_test.py), NOT
    assumed. Whole-day no-rf Sharpe by event type:
        FOMC       -1.8            -> BLOCK the afternoon chop
        quad-witch -9.0 (27% hit)  -> BLOCK
        CPI        +3.5 (the book's BEST event day!)  -> do NOT block
        NFP        ~neutral                            -> do NOT block
    We only avoid events the data shows are harmful. The CPI result is exactly why we
    backtest before wiring: a naive "block all macro events" filter would have thrown
    away the book's most profitable day. (Small n -> treat as directional.)
    Every trigger is a KNOWN scheduled date -> zero look-ahead, fully backtestable.
    """
    d = _to_date(dt)
    t = dt.time() if hasattr(dt, 'time') else datetime.time(12, 0)

    if any(_to_date(s) == d for s in EARNINGS_DATES.get(symbol, [])):
        return True, f"{symbol} reports earnings today"
    if is_fomc_day(d) and t >= _FOMC_BLOCK_FROM:
        return True, "FOMC announcement window (>=14:00 ET)"
    if is_quad_witch(d):
        return True, "quad-witch expiration"
    return False, ""


def event_flags(symbol, d):
    """All event flags for a date — handy for logging or as ML features."""
    return {
        'earnings': any(_to_date(s) == _to_date(d) for s in EARNINGS_DATES.get(symbol, [])),
        'fomc': is_fomc_day(d), 'cpi': is_cpi_day(d), 'nfp': is_nfp_day(d),
        'opex': is_opex_day(d), 'quad_witch': is_quad_witch(d),
        'days_to_fomc': days_to_fomc(d), 'days_to_cpi': days_to_cpi(d), 'days_to_nfp': days_to_nfp(d),
    }


if __name__ == '__main__':
    # Smoke test
    today = datetime.date.today()
    print(f"Today:           {today} ({today.strftime('%A')})")
    print(f"  is_opex_day:        {is_opex_day(today)}")
    print(f"  is_day_after_opex:  {is_day_after_opex(today)}")
    print(f"  is_quad_witch:      {is_quad_witch(today)}")
    print(f"  days_to_fomc:       {days_to_fomc(today)}")
    print(f"  is_fomc_day:        {is_fomc_day(today)}")
    print(f"  is_cpi_day:         {is_cpi_day(today)}")
    print(f"  is_nfp_day:         {is_nfp_day(today)}")

    # Filter demo on known event dates (block reasons)
    print("\nshould_block_entry() on known events:")
    fomc_pm = datetime.datetime(2025, 1, 29, 14, 30)   # FOMC day, 2:30pm ET
    cpi_am = datetime.datetime(2025, 1, 15, 9, 45)      # CPI day, 9:45am ET
    qw = datetime.datetime(2025, 3, 21, 11, 0)          # quad-witch Fri
    calm = datetime.datetime(2025, 1, 22, 11, 0)        # ordinary day
    for label, dt in [('FOMC 2:30pm', fomc_pm), ('CPI 9:45am', cpi_am),
                      ('quad-witch', qw), ('calm day', calm)]:
        print(f"  AAPL {label:14s} -> {should_block_entry('AAPL', dt)}")
