"""
Pull historical + upcoming EARNINGS dates for the equity universe and write
earnings_dates.json, which event_calendar.py auto-loads into EARNINGS_DATES.

The earnings block is the #1 stock-specific event in the deterministic filter, and
the one piece that needs an external source. Run where yfinance works (your machine /
VM). ETFs (SPY/QQQ/IWM) have no earnings and are skipped.

Usage:  python3.12 pull_earnings.py
"""

import os
import json
import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOG', 'META', 'JPM', 'WMT', 'UNH', 'XOM',
    # Disjoint technology candidate.
    'INTC', 'AMD', 'AVGO', 'QCOM', 'MU',
]
START, END = '2019-01-01', '2027-12-31'   # warmup/history + upcoming schedule

out = {}
for s in STOCKS:
    got = False
    for attempt in range(3):
        try:
            df = yf.Ticker(s).get_earnings_dates(limit=40)   # ~10y of quarters + upcoming
            if df is not None and len(df):
                dates = sorted({pd.Timestamp(ix).date().isoformat() for ix in df.index})
                dates = [d for d in dates if START <= d <= END]
                out[s] = dates
                print(f"  {s:5s}: {len(dates)} dates  {dates[0]}..{dates[-1]}", flush=True)
                got = True
                break
        except Exception as e:
            print(f"  {s:5s}: attempt {attempt+1} failed: {e}", flush=True)
        import time
        time.sleep(5)
    if not got:
        print(f"  {s:5s}: GAVE UP (rate-limited?) — leave for a later rerun", flush=True)

if out:
    with open(os.path.join(SCRIPT_DIR, 'earnings_dates.json'), 'w') as f:
        json.dump(out, f, indent=2)
    total = sum(len(v) for v in out.values())
    print(f"\nWrote earnings_dates.json: {len(out)} symbols, {total} dates. "
          f"event_calendar.py auto-loads it on import.")
else:
    print("\nNo earnings pulled (yfinance rate-limited?). Rerun in a few minutes, or fill "
          "EARNINGS_DATES manually in event_calendar.py.")
