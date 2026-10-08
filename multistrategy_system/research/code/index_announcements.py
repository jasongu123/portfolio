"""
S&P 500 change ANNOUNCEMENT dates (2026-10-03), for a pre-registered follow-up to index_rebalance.py
(T-86/87 failed because only effective dates were known, so the run-up into inclusion could not be traded).

Source: Wikipedia "Historical components of the S&P 500" change table. Most rows cite the S&P Dow Jones
Indices press release (press.spglobal.com/YYYY-MM-DD-..., spglobal.com/.../announcements/YYYYMMDD-...,
or PR Newswire) with its date. Announcement date = the earliest such date within 90 days before the
effective date.

Checked by hand against known announcements: SMCI 2024-03-01, PLTR/DELL 2024-09-06, UBER 2023-12-01,
COIN 2025-05-12, DDOG 2025-07-02, XYZ 2025-07-18, HOOD/APP 2025-09-05 all match. Known exception: TSLA was
first announced 2020-11-16, but the cited release is the 2020-12-11 one with the final details, so for
such cases the date is LATE. A late date delays entry (conservative); it never gives look-ahead.
S&P releases normally come after the close, so a trade can enter at the NEXT trading day at the earliest.

  python3 index_announcements.py     -> backtest_data/sp500_announcements.csv (effective, added, announce)
"""
import os
import re

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'backtest_data', 'sp500_announcements.csv')
URL = 'https://en.wikipedia.org/w/index.php?title=Historical_components_of_the_S%26P_500&action=raw'
UA = {'User-Agent': 'ibkr research script (personal, non-commercial)'}


def _ref_dates(block, named):
    bodies = re.findall(r'<ref[^>/]*>(.*?)</ref>', block, flags=re.S)
    bodies += [named.get(n.strip(), '') for n in re.findall(r'<ref name\s*=\s*"?([^">/]+?)"?\s*/>', block)]
    out = []
    for b in bodies:
        if not re.search(r'S&P|spglobal|spdji|prnewswire|Standard & Poor|join', b, re.I):
            continue
        m = re.search(r'\|\s*date\s*=\s*([^|}]+)', b)
        if m:
            d = pd.to_datetime(m.group(1).strip(), errors='coerce')
            if pd.notna(d):
                out.append(d)
        m = re.search(r'press\.spglobal\.com/(\d{4}-\d{2}-\d{2})', b) or re.search(r'announcements/(\d{8})', b)
        if m:
            out.append(pd.to_datetime(m.group(1)))
    return out


def parse(wikitext):
    tbl = wikitext[wikitext.index('id="changes"'):]
    tbl = tbl[:tbl.index('\n|}')]
    named = {m.group(1).strip(): m.group(2)
             for m in re.finditer(r'<ref name\s*=\s*"?([^">/]+?)"?\s*>(.*?)</ref>', tbl, flags=re.S)}
    rows = []
    for blk in tbl.split('\n|-')[2:]:
        cells = [c.strip() for c in re.split(r'\n\|\||\n\|(?!\|)', '\n' + blk.strip()) if c.strip()]
        if len(cells) < 3:
            continue
        eff = pd.to_datetime(re.sub(r'\[\[|\]\]', '', cells[0]), errors='coerce')
        if pd.isna(eff):
            continue
        ds = [d for d in _ref_dates(blk, named) if eff - pd.Timedelta(days=90) <= d <= eff]
        rows.append({'effective': eff.date(), 'added': re.sub(r'[\[\]]', '', cells[1]),
                     'announce': min(ds).date() if ds else None})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = parse(requests.get(URL, headers=UA, timeout=60).text)
    df.to_csv(OUT, index=False)
    recent = df[pd.to_datetime(df.effective) >= '2020-01-01']
    lag = (pd.to_datetime(recent.effective) - pd.to_datetime(recent.announce)).dt.days
    print(f"{len(df)} changes, {df.announce.notna().sum()} with an announcement date; 2020+: "
          f"{recent.announce.notna().mean():.1%} covered, median {lag.median():.0f} calendar days before the effective date")
