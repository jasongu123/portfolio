"""
Small/mid-cap universe feasibility (2026-10-03) for FAILURE_AUTOPSY.md hypothesis H1: do the patterns
that died in the S&P 500 since 2020 still work in less-watched stocks? This file only checks DATA;
it computes no returns.

1. Point-in-time membership of the S&P MidCap 400 and SmallCap 600 from 2020-01-01, rebuilt backward
   from Wikipedia's current constituent lists and change tables (undo each change in reverse order).
   Completeness check: the rebuilt count should stay near 400 / 600; undo mismatches (an "added"
   ticker not in the set, usually a later ticker change or a missing row) are counted.
2. --ibkr-check: can IBKR supply daily prices for names that LEFT the indexes (acquired, delisted,
   moved)? Without them a backtest only sees survivors. Read-only, client ID 98.

  python3 smallcap_universe.py                 -> backtest_data/sp400_600_membership.csv + summary
  python3 smallcap_universe.py --ibkr-check    (TWS/Gateway logged in) -> coverage of leavers
"""
import argparse
import os
import random
import re
import threading
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(BD, 'sp400_600_membership.csv')
PAGES = {'SP400': 'List_of_S%26P_400_companies', 'SP600': 'List_of_S%26P_600_companies'}
START = pd.Timestamp('2020-01-01')
UA = {'User-Agent': 'ibkr research script (personal, non-commercial)'}


def _clean(x):
    x = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', x)
    return re.sub(r'<ref.*?(</ref>|/>)', '', x, flags=re.S).strip()


def _tables(s):
    return [s[m.start():s.find('\n|}', m.start())] for m in re.finditer(r'^\{\|', s, flags=re.M)]


def _rows(seg):
    out = []
    for blk in seg.split('\n|-')[1:]:
        if blk.lstrip().startswith('!'):
            continue
        txt = re.sub(r'^\|', '', ' '.join(l.strip() for l in blk.strip().splitlines()).strip())
        out.append([_clean(c) for c in re.split(r'\|\||\s\|(?!\|)\s', txt)])
    return out


def rebuild(index, wikitext):
    t = _tables(wikitext)
    current = {m.replace('.', '-') for m in re.findall(r'\{\{\s*\w*Symbol\s*\|\s*([A-Z0-9.\-]+)', t[0])}
    changes = []
    for r in _rows(t[1]):
        if len(r) < 5:
            continue
        d = pd.to_datetime(r[0], errors='coerce')
        if pd.notna(d):
            changes.append((d, r[1].replace('.', '-').strip(), r[3].replace('.', '-').strip()))
    changes.sort(key=lambda x: x[0], reverse=True)
    # spells: ticker -> list of [start, end); walk back from today
    open_since = {s: None for s in current}               # member now; start unknown yet
    spells, mismatches = [], 0
    for d, added, removed in changes:
        if d < START:
            break
        if added:
            if added in open_since:
                spells.append((added, d, open_since.pop(added)))
            else:
                mismatches += 1
        if removed:
            open_since[removed] = d
    for s, end in open_since.items():
        spells.append((s, START, end))
    df = pd.DataFrame(spells, columns=['symbol', 'start', 'end'])
    df['index'] = index
    return df, len(current), mismatches


def check_ibkr(df, port, n=60):
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper

    class App(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.ready, self.bars, self.done = threading.Event(), {}, {}

        def nextValidId(self, oid):
            self.ready.set()

        def error(self, reqId, *a):
            if reqId in self.done:
                self.done[reqId].set()

        def historicalData(self, reqId, bar):
            self.bars.setdefault(reqId, []).append(bar.date)

        def historicalDataEnd(self, reqId, *a):
            self.done[reqId].set()

    leavers = df[df.end.notna()].drop_duplicates('symbol')
    random.seed(7)
    sample = leavers.sample(min(n, len(leavers)), random_state=7)
    app = App()
    app.connect('127.0.0.1', port, 98)
    threading.Thread(target=app.run, daemon=True).start()
    if not app.ready.wait(15):
        raise SystemExit('could not connect to TWS/Gateway')
    rows = []
    for k, r in enumerate(sample.itertuples()):
        c = Contract()
        c.symbol, c.secType, c.exchange, c.currency = r.symbol.replace('-', ' '), 'STK', 'SMART', 'USD'
        app.done[k] = threading.Event()
        end = (r.end - pd.Timedelta(days=1)).strftime('%Y%m%d 16:00:00 US/Eastern')
        app.reqHistoricalData(k, c, end, '6 M', '1 day', 'TRADES', 1, 1, False, [])
        app.done[k].wait(30)
        bars = app.bars.get(k, [])
        rows.append({'symbol': r.symbol, 'index': r.index, 'left': r.end.date(), 'bars': len(bars),
                     'last_bar': bars[-1] if bars else None})
        time.sleep(10.5)
    app.disconnect()
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, 'results', 'smallcap_leaver_coverage.csv'), index=False)
    ok = (res.bars >= 60).mean()
    print(f"IBKR daily bars for the 6 months before leaving, {len(res)} sampled leavers: {ok:.0%} have >= 60 bars")
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ibkr-check', action='store_true')
    ap.add_argument('--port', type=int, default=7497)
    a = ap.parse_args()
    parts = []
    for index, page in PAGES.items():
        text = requests.get(f'https://en.wikipedia.org/w/index.php?title={page}&action=raw', headers=UA, timeout=60).text
        df, n_now, mism = rebuild(index, text)
        parts.append(df)
        at_start = ((df.start <= START) & (df.end.isna() | (df.end > START))).sum()
        print(f"{index}: {n_now} members now; rebuilt {at_start} members on {START:%Y-%m-%d}; "
              f"{df.symbol.nunique()} distinct names since; {df.end.notna().sum()} spells ended (left); "
              f"{mism} undo mismatches")
    mem = pd.concat(parts, ignore_index=True)
    mem.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    if a.ibkr_check:
        check_ibkr(mem, a.port)


if __name__ == '__main__':
    main()
