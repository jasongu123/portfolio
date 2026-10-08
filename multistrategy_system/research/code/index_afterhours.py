"""
Where does an S&P 500 addition's announcement jump happen? — RESEARCH ONLY, a MEASUREMENT (no decision).
2026-10-03, after T-95 found +3.79% from the announcement-day close to the next close (A0 -> A1), which
daily data cannot split. S&P usually publishes after the close (~17:15 New York time); IBKR allows
trading 16:00-20:00 and 04:00-09:30. Any trade built on this needs its own pre-registered test.

For each 2020+ addition in results/index_announce/events.csv, fetch 1-minute TRADES bars including
extended hours for A0 and A1 (stock and SPY), then take the last trade at or before each checkpoint:
  A0 16:00 (close), 17:00, 17:30, 18:00, 20:00 | A1 04:00(+), 08:00, 09:00, 09:29, 09:30 open, 10:00, 16:00
Reported per checkpoint (stock minus SPY over the same span, mean and t by announcement date):
  done   = move from the A0 close to the checkpoint (how much of the jump has happened)
  left   = move from the checkpoint to the A1 close: what buying at that moment and selling at the
           A1 close would have earned, before spreads (A1 close -> E-1 close is T-95's +0.14%)
Also: the median minute of the largest 1-minute move after 16:00 on A0 (when the release hit).

Read-only: historical data only, own client ID 97, no orders, no account calls.
Usage: python3 index_afterhours.py [--port 7497]   (TWS or Gateway logged in; ~20 min with pacing)
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'index_announce')
CACHE = os.path.join(OUT, 'afterhours_bars')
CLIENT_ID = 97
NY = 'America/New_York'
CHECKS = [('A0', '16:00'), ('A0', '17:00'), ('A0', '17:30'), ('A0', '18:00'), ('A0', '20:00'),
          ('A1', '04:05'), ('A1', '08:00'), ('A1', '09:00'), ('A1', '09:29'), ('A1', '09:30'), ('A1', '10:00'), ('A1', '16:00')]


class App(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.bars, self.done, self.err = {}, {}, {}

    def nextValidId(self, oid):
        self.ready.set()

    def error(self, reqId, *a):
        # ibapi signatures differ by version (an errorTime int was added); take the small int as the code
        ints = [x for x in a if isinstance(x, int) and x < 100000]
        strs = [x for x in a if isinstance(x, str)]
        code, msg = (ints[0] if ints else 0), (strs[0] if strs else '')
        if code in (2104, 2106, 2158, 2107, 2108, 2119, 2100):
            return
        if reqId in self.done:
            self.err[reqId] = f"{code} {msg}"
            self.done[reqId].set()

    def historicalData(self, reqId, bar):
        self.bars.setdefault(reqId, []).append((int(bar.date), bar.open, bar.close))

    def historicalDataEnd(self, reqId, *a):
        self.done[reqId].set()


def stock(sym):
    c = Contract()
    c.symbol, c.secType, c.exchange, c.currency = sym.replace('-', ' '), 'STK', 'SMART', 'USD'
    return c


def fetch(app, sym, a1, rid):
    path = os.path.join(CACHE, f"{sym}_{a1:%Y%m%d}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    app.done[rid] = threading.Event()
    end = f"{a1:%Y%m%d} 20:00:00 US/Eastern"
    app.reqHistoricalData(rid, stock(sym), end, '2 D', '1 min', 'TRADES', 0, 2, False, [])
    app.done[rid].wait(90)
    time.sleep(10.5)                                    # pacing: <= 60 requests per 10 minutes
    rows = app.bars.pop(rid, [])
    if not rows:
        print(f"  {sym} {a1:%Y-%m-%d}: no data ({app.err.get(rid, 'timeout')})", flush=True)
        return None
    df = pd.DataFrame(rows, columns=['t', 'open', 'close'])
    df.index = pd.to_datetime(df.t, unit='s', utc=True).dt.tz_convert(NY).dt.tz_localize(None)
    df = df[['open', 'close']]
    df.to_csv(path)
    return df


def price_at(df, day, hhmm):
    t = pd.Timestamp(f"{day:%Y-%m-%d} {hhmm}")
    if hhmm == '09:30':                                 # the regular open: first bar at 09:30
        x = df.loc[t:t + pd.Timedelta(minutes=1)]
        return float(x.open.iloc[0]) if len(x) else np.nan
    if hhmm == '16:00':
        t = t - pd.Timedelta(minutes=1)                 # the 15:59 bar closes at 16:00
    x = df.loc[:t]
    x = x[x.index.normalize() == pd.Timestamp(day)] if day is not None else x
    return float(x.close.iloc[-1]) if len(x) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=7497)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    ev = pd.read_csv(os.path.join(OUT, 'events.csv'), parse_dates=['announce', 'effective'])
    ev = ev[(ev.leg == 'ADD') & (ev.announce >= '2020-01-01')].reset_index(drop=True)
    px = pd.read_csv(os.path.join(HERE, 'backtest_data', 'sp500_prices_long.csv'), index_col=0, parse_dates=True).sort_index()
    days = px.index
    ev['A1'] = [days[i] for i in ev.a1]
    ev['Em1'] = [days[i - 1] for i in ev.i]

    app = App()
    app.connect('127.0.0.1', a.port, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()
    if not app.ready.wait(15):
        sys.exit("could not connect to TWS/Gateway (is the API enabled on this port?)")
    rid = 1
    spy = {}
    rows = []
    for k, r in ev.iterrows():
        for sym in (r.symbol, 'SPY'):
            if sym == 'SPY' and r.A1 in spy:
                continue
            df = fetch(app, sym, r.A1, rid)
            rid += 1
            if sym == 'SPY':
                spy[r.A1] = df
            else:
                bars = df
        if bars is None or spy.get(r.A1) is None:
            continue
        s, m = bars, spy[r.A1]
        rec = {'symbol': r.symbol, 'announce': r.announce, 'A1': r.A1}
        for d, hhmm in CHECKS:
            day = r.announce if d == 'A0' else r.A1
            rec[f'{d} {hhmm}'] = (price_at(s, day, hhmm), price_at(m, day, hhmm))
        ah = s[(s.index > pd.Timestamp(f"{r.announce:%Y-%m-%d} 16:00")) & (s.index <= pd.Timestamp(f"{r.announce:%Y-%m-%d} 20:00"))]
        jump = ah.close.pct_change().abs()
        rec['biggest_move_minute'] = jump.idxmax().strftime('%H:%M') if jump.notna().any() else None
        rec['afterhours_minutes_traded'] = int(len(ah))
        rows.append(rec)
        if k % 10 == 0:
            print(f"  {k + 1}/{len(ev)} {r.symbol} {r.announce:%Y-%m-%d}", flush=True)
    app.disconnect()

    out = []
    for rec in rows:
        base_s, base_m = rec['A0 16:00']
        a1s, a1m = rec['A1 16:00']
        row = {'symbol': rec['symbol'], 'announce': rec['announce'], 'biggest_move_minute': rec['biggest_move_minute'],
               'afterhours_minutes_traded': rec['afterhours_minutes_traded']}
        for d, hhmm in CHECKS:
            ps, pm = rec[f'{d} {hhmm}']
            row[f'done {d} {hhmm}'] = (ps / base_s - 1) - (pm / base_m - 1)
            row[f'left {d} {hhmm}'] = (a1s / ps - 1) - (a1m / pm - 1)
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUT, 'afterhours_points.csv'), index=False)

    lines = [f"S&P 500 additions 2020+ with extended-hours bars: {len(res)} of {len(ev)}; stock minus SPY, % "
             f"(t by announcement date)", f"{'checkpoint':14s} {'done since A0 close':>22s} {'left to A1 close':>20s}"]
    g = lambda col: res.groupby('announce')[col].mean().dropna()
    for d, hhmm in CHECKS:
        x, y = g(f'done {d} {hhmm}'), g(f'left {d} {hhmm}')
        lines.append(f"{d + ' ' + hhmm:14s} {x.mean() * 100:+10.2f} ({x.mean() / x.std() * np.sqrt(len(x)):+5.1f}) "
                     f"{y.mean() * 100:+11.2f} ({y.mean() / y.std() * np.sqrt(len(y)):+5.1f})")
    lines.append("\nA1 close -> E-1 close: +0.14% (t 0.2), from T-95 (daily, vs equal-weight members)")
    lines.append(f"Largest 1-minute after-hours move on A0, median time: {pd.Series(res.biggest_move_minute).dropna().sort_values().iloc[len(res) // 2] if len(res) else 'n/a'}; "
                 f"after-hours minutes with trades on A0: median {res.afterhours_minutes_traded.median():.0f} of 240")
    report = '\n'.join(lines)
    print(report)
    open(os.path.join(OUT, 'afterhours_report.txt'), 'w').write(report + '\n')


if __name__ == '__main__':
    main()
