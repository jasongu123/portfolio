"""
Earnings press-release text from SEC EDGAR, for the distillation project (2026-10-02).

Every 8-K with item 2.02 (Results of Operations) carries the company's earnings press release,
usually as exhibit 99.1. This module builds the filing index with accession numbers, downloads
releases and keeps their narrative text (tables removed: the numbers are in the financial
statements; the language is what the models read).

  python3 earnings_text.py --index              filing index -> backtest_data/sec_8k_202_index.csv
  python3 earnings_text.py --sample 2500        stratified sample (by year, seed 42) + download text
  python3 earnings_text.py --all                download every release (hours; resumable)

Text goes to backtest_data/earnings_text/<accession>.txt; backtest_data/earnings_text_meta.csv
records the source exhibit and length. Read-only on EDGAR; respects the SEC rate limit.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from short_interest_splice import SEC_UA, SEC_TICKERS, SEC_SLEEP  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
INDEX = os.path.join(BD, 'sec_8k_202_index.csv')
SUBS_DIR = os.path.join(BD, 'sec_submissions')
TEXT_DIR = os.path.join(BD, 'earnings_text')
META = os.path.join(BD, 'earnings_text_meta.csv')
SAMPLE = os.path.join(BD, 'earnings_text_sample.csv')
SAMPLE_V2 = {'train2': (os.path.join(BD, 'earnings_text_train2.csv'), 1500, 43),
             'test2': (os.path.join(BD, 'earnings_text_test2.csv'), 500, 44)}
TEST3 = (os.path.join(BD, 'earnings_text_test3.csv'), 500, 45)
MAX_CHARS = 40000


def _get(url, as_json=True, retries=4):
    for i in range(retries):
        try:
            r = requests.get(url, headers=SEC_UA, timeout=60)
            if r.status_code == 200:
                return r.json() if as_json else r.text
            if r.status_code == 404:
                return None
            time.sleep(2 * (i + 1))
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None


def build_index():
    os.makedirs(SUBS_DIR, exist_ok=True)
    uni = pd.read_csv(os.path.join(BD, 'sp500_universe_2020_2026.csv')).iloc[:, 0].astype(str).tolist()
    tick = _get(SEC_TICKERS)
    cik_of = {v['ticker'].upper().replace('.', '-'): int(v['cik_str']) for v in tick.values()}
    rows = []
    for k, sym in enumerate(uni):
        cik = cik_of.get(sym.upper().replace('.', '-'))
        if cik is None:
            continue
        cache = os.path.join(SUBS_DIR, f'{cik}.json')
        if os.path.exists(cache):
            blocks = json.load(open(cache))
        else:
            sub = _get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json')
            time.sleep(SEC_SLEEP)
            if sub is None:
                continue
            blocks = [sub['filings']['recent']]
            for f in sub['filings'].get('files', []):
                more = _get(f"https://data.sec.gov/submissions/{f['name']}")
                time.sleep(SEC_SLEEP)
                if more:
                    blocks.append(more)
            json.dump(blocks, open(cache, 'w'))
        for b in blocks:
            for form, acc, doc, fdate, accept, items in zip(b.get('form', []), b.get('accessionNumber', []),
                                                            b.get('primaryDocument', []), b.get('filingDate', []),
                                                            b.get('acceptanceDateTime', []), b.get('items', [])):
                if form == '8-K' and '2.02' in str(items).split(','):
                    rows.append({'symbol': sym, 'cik': cik, 'accession': acc, 'primary_doc': doc,
                                 'filing_date': fdate, 'acceptance': accept})
        if k % 100 == 0:
            print(f"  {k}/{len(uni)} {sym}: {len(rows)} filings", flush=True)
    df = pd.DataFrame(rows).drop_duplicates('accession').sort_values(['filing_date', 'symbol'])
    df.to_csv(INDEX, index=False)
    print(f"index: {len(df)} item-2.02 8-Ks, {df.symbol.nunique()} symbols, {df.filing_date.min()}..{df.filing_date.max()}")


def _documents(full):
    for d in re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', full, flags=re.S | re.I):
        m = re.search(r'<TYPE>\s*([^\s<]+)', d, flags=re.I)
        t = re.search(r'<TEXT>(.*?)</TEXT>', d, flags=re.S | re.I)
        yield (m.group(1).upper() if m else ''), (t.group(1) if t else d)


def narrative(raw):
    """Plain narrative text: HTML tables dropped, numeric-heavy lines dropped, whitespace normalised."""
    from bs4 import BeautifulSoup
    if re.search(r'<(html|body|p|div|table)\b', raw[:5000], flags=re.I):
        soup = BeautifulSoup(raw, 'lxml')
        for t in soup.find_all(['table', 'script', 'style']):
            t.decompose()
        text = soup.get_text('\n')
    else:
        text = raw
    keep = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if len(line) < 3:
            continue
        letters = sum(c.isalpha() for c in line)
        if letters / max(len(line), 1) < 0.5:
            continue
        keep.append(line)
    return '\n'.join(keep)[:MAX_CHARS]


def fetch_text(row):
    path = os.path.join(TEXT_DIR, f"{row.accession}.txt")
    if os.path.exists(path):
        return None
    acc = row.accession.replace('-', '')
    full = _get(f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{acc}/{row.accession}.txt", as_json=False)
    time.sleep(SEC_SLEEP)
    if full is None:
        return {'accession': row.accession, 'source': 'missing', 'chars': 0}
    docs = list(_documents(full))
    ex = sorted([(t, body) for t, body in docs if t.startswith('EX-99')], key=lambda x: x[0])
    src, body = (ex[0] if ex else (docs[0] if docs else ('none', '')))
    text = narrative(body)
    with open(path + '.part', 'w') as fh:
        fh.write(text)
    os.replace(path + '.part', path)
    return {'accession': row.accession, 'source': src, 'chars': len(text)}


def _save_meta(meta):
    if meta:
        new = pd.DataFrame(meta)
        if os.path.exists(META):
            new = pd.concat([pd.read_csv(META), new]).drop_duplicates('accession', keep='last')
        new.to_csv(META, index=False)


def fetch_many(df, label):
    """Resumable: a release already on disk is skipped. Each text is written to a temporary file and
    renamed, so an interrupted run (e.g. a restart) never leaves a half-written text; the metadata
    is saved every 500 releases."""
    os.makedirs(TEXT_DIR, exist_ok=True)
    meta = []
    t0 = time.time()
    for k, row in enumerate(df.itertuples()):
        m = fetch_text(row)
        if m:
            meta.append(m)
        if k % 200 == 0:
            print(f"  {label}: {k}/{len(df)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if len(meta) >= 500:
            _save_meta(meta)
            meta = []
    _save_meta(meta)


def sample(n, seed=42):
    idx = pd.read_csv(INDEX)
    idx['year'] = idx.filing_date.str[:4].astype(int)
    idx = idx[idx.year >= 2005]
    per = max(1, n // idx.year.nunique())
    s = (idx.groupby('year', group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=seed))
         .sample(frac=1, random_state=seed).head(n))
    s.to_csv(SAMPLE, index=False)
    return s


def sample_v2():
    """Version 2 sets (2026-10-02): a fresh 500-release test set and 1,500 extra training releases,
    stratified by year 2005-2026, disjoint from each other and from the version-1 sample."""
    idx = pd.read_csv(INDEX)
    idx['year'] = idx.filing_date.str[:4].astype(int)
    used = set(pd.read_csv(SAMPLE).accession)
    out = {}
    for name in ('test2', 'train2'):                      # test first, so it is drawn from the untouched pool
        path, n, seed = SAMPLE_V2[name]
        pool = idx[(idx.year >= 2005) & ~idx.accession.isin(used)]
        per = int(np.ceil(n / pool.year.nunique()))
        s = (pool.groupby('year', group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=seed))
             .sample(frac=1, random_state=seed).head(n))
        s.to_csv(path, index=False)
        used |= set(s.accession)
        out[name] = s
    return out


def sample_v3():
    """Version 3 test set (2026-10-02): 500 releases stratified by year 2005-2026 (seed 45), disjoint
    from the v1 sample and both v2 sets. Version 3's training pool is those three earlier sets."""
    idx = pd.read_csv(INDEX)
    idx['year'] = idx.filing_date.str[:4].astype(int)
    used = set(pd.read_csv(SAMPLE).accession)
    for path, _, _ in SAMPLE_V2.values():
        used |= set(pd.read_csv(path).accession)
    path, n, seed = TEST3
    pool = idx[(idx.year >= 2005) & ~idx.accession.isin(used)]
    per = int(np.ceil(n / pool.year.nunique()))
    s = (pool.groupby('year', group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=seed))
         .sample(frac=1, random_state=seed).head(n))
    s.to_csv(path, index=False)
    return s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', action='store_true', help='draw the version-2 test and extra training sets and download them')
    ap.add_argument('--v3', action='store_true', help='draw the version-3 test set and download it')
    ap.add_argument('--index', action='store_true')
    ap.add_argument('--sample', type=int)
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    if a.index:
        build_index()
    if a.sample:
        fetch_many(sample(a.sample), 'sample')
    if a.v2:
        for name, df in sample_v2().items():
            fetch_many(df, name)
    if a.v3:
        fetch_many(sample_v3(), 'test3')
    if a.all:
        fetch_many(pd.read_csv(INDEX), 'all')
    if os.path.exists(META):
        m = pd.read_csv(META)
        ok = m[m.chars > 0]
        print(f"texts: {len(m)} ({(m.source.str.startswith('EX-99')).mean():.0%} from EX-99); "
              f"narrative chars median {ok.chars.median():.0f}, p90 {ok.chars.quantile(0.9):.0f} "
              f"(~{ok.chars.median() / 4:.0f} tokens median)")
