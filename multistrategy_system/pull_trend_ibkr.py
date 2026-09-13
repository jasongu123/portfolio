"""
Extract the BREADTH assets for trend sleeve v2 from IBKR (NOT yfinance).

Adds 9 liquid ETFs that the original trend_cache.csv (9 assets) doesn't have, to
widen the trend engine across sectors — the breadth is where trend's Sharpe lives:

  Equity  : IWM (US small-cap), EWJ (Japan)
  Commod. : SLV (silver), USO (oil), DBA (agriculture)
  FX      : FXE (euro), FXY (yen), FXB (pound)
  Rates   : SHY (1-3y Treasury)

Pulls daily ADJUSTED_LAST (total-return) bars — one series per asset, which is all
trend needs (sign of momentum + realized vol). Writes trend_extra.csv; trend_follow_v2.py
merges it with trend_cache.csv. Matches ibkr2.py: native ibapi, STK/SMART/USD, 7496/7497.

Usage:
    python3.12 pull_trend_ibkr.py          # live port 7496
    python3.12 pull_trend_ibkr.py 7497     # paper port
"""

import os
import sys
import time
import threading
import datetime
import pandas as pd

from ibapi.client import EClient, Contract
from ibapi.wrapper import EWrapper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7496
CLIENT_ID = 78                      # distinct from pull_carry_ibkr (77) and ibkr2.py
TICKERS = ['IWM', 'EWJ', 'SLV', 'USO', 'DBA', 'FXE', 'FXY', 'FXB', 'SHY']
BAR = '1 day'
DURATIONS = ['20 Y', '15 Y', '10 Y']
PACE_SECONDS = 11
REQ_TIMEOUT = 60
FATAL_CODES = {162, 165, 200, 300, 321, 354, 366, 420}
BENIGN_CODES = {2104, 2106, 2107, 2108, 2158, 2100, 2150}


class HistClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.bars = {}
        self.done = {}
        self.next_id = 1
        self.connected_ok = threading.Event()

    def nextValidId(self, orderId):
        self.connected_ok.set()

    def error(self, reqId, code, msg, misc=None):
        if code in BENIGN_CODES:
            return
        print(f"  [err req {reqId}] {code}: {msg}")
        if reqId in self.done and code in FATAL_CODES:
            self.done[reqId].set()

    def historicalData(self, reqId, bar):
        ds = str(bar.date).strip()[:8]
        try:
            d = datetime.datetime.strptime(ds, '%Y%m%d').date()
        except ValueError:
            return
        self.bars.setdefault(reqId, {})[d] = float(bar.close)

    def historicalDataEnd(self, reqId, start, end):
        if reqId in self.done:
            self.done[reqId].set()

    def fetch(self, symbol, what='ADJUSTED_LAST'):
        c = Contract()
        c.symbol = symbol
        c.secType = 'STK'
        c.exchange = 'SMART'
        c.currency = 'USD'
        for dur in DURATIONS:
            rid = self.next_id
            self.next_id += 1
            ev = threading.Event()
            self.done[rid] = ev
            self.bars[rid] = {}
            self.reqHistoricalData(rid, c, '', dur, BAR, what, 1, 1, False, [])
            ev.wait(timeout=REQ_TIMEOUT)
            data = self.bars.get(rid, {})
            if data:
                return pd.Series(data).sort_index()
            time.sleep(PACE_SECONDS)
        return pd.Series(dtype=float)


def main():
    app = HistClient()
    app.connect('127.0.0.1', PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()
    if not app.connected_ok.wait(timeout=8):
        print(f"\nNOT CONNECTED on 127.0.0.1:{PORT}. Is TWS/IB Gateway logged in? "
              f"Live=7496, Paper=7497. If '326 client id in use', change CLIENT_ID.")
        app.disconnect()
        sys.exit(1)
    print(f"Connected on port {PORT}. Pulling {len(TICKERS)} breadth ETFs (ADJUSTED_LAST)...\n")

    cols = {}
    for sym in TICKERS:
        s = app.fetch(sym)
        time.sleep(PACE_SECONDS)
        if len(s):
            cols[sym] = s
            print(f"  {sym:5s}: {len(s):>5} bars | {s.index[0]}..{s.index[-1]}", flush=True)
        else:
            print(f"  {sym:5s}: MISSING", flush=True)
    app.disconnect()

    if not cols:
        sys.exit("No data returned. Check market-data permissions / TWS login.")
    df = pd.DataFrame(cols).sort_index()
    df.index.name = 'Date'
    df.to_csv(os.path.join(SCRIPT_DIR, 'trend_extra.csv'))
    print(f"\nSaved trend_extra.csv: {df.shape[1]} tickers, {len(df)} days, "
          f"{df.index[0]}..{df.index[-1]}")
    if df.shape[1] < len(TICKERS):
        print(f"WARNING: missing {[t for t in TICKERS if t not in df.columns]} — rerun to fill.")


if __name__ == '__main__':
    main()
