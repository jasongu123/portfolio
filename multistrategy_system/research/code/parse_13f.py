"""Reduce SEC Form 13F structured datasets to institutional holdings per stock per quarter for
the S&P 500 universe. Original 13F-HR filings only (no amendments, no notices); share holdings
only (no puts/calls, no principal amounts); filings made more than 60 days after the quarter end are
dropped (the deadline is 45 days; late filings would otherwise be counted as if available on time). Output: results/inst_holdings_sp500.csv.gz with
period (quarter end), symbol, inst_shares (sum over managers), n_managers, filed_first,
filed_median, filed_last (filing dates, for availability).
Point-in-time fix (2026-10-11, audit finding 11): crowded_spaces.py uses a quarter's aggregate from
the first month-end at least 46 days after the quarter end. Only filings published strictly BEFORE
that first-use date (the last trading day of that month) are aggregated; later filings are left
out, so every holding in the aggregate was public when first used. Usage: python3 parse_13f.py"""
import glob, io, os, zipfile
import numpy as np
import pandas as pd

def point_in_time(allp, month_ends, lag_days=46):
    """Keep only manager filings published strictly before the quarter's first use: the first
    month-end (a trading day in month_ends) at least lag_days after the quarter end."""
    avail = allp.period + pd.Timedelta(days=lag_days)
    pos = month_ends.searchsorted(avail)
    first_use = pd.Series(month_ends[np.minimum(pos, len(month_ends) - 1)], index=allp.index).where(pos < len(month_ends))
    first_use = first_use.fillna(avail + pd.offsets.BMonthEnd(0))
    return allp[allp.filed < first_use]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
D13 = os.path.join(SCRIPT_DIR, 'backtest_data', 'sec13f')
OUT = os.path.join(SCRIPT_DIR, 'results', 'inst_holdings_sp500.csv.gz')
def main():
    cmap = pd.read_csv(os.path.join(SCRIPT_DIR, 'backtest_data', 'cusip_map_sp500.csv'), dtype=str)
    cusips = dict(zip(cmap.cusip, cmap.symbol))

    parts = []
    for f in sorted(glob.glob(os.path.join(D13, '*_form13f.zip'))):
        z = zipfile.ZipFile(f)
        name = {os.path.basename(n).upper(): n for n in z.namelist()}       # one archive nests its tables in a folder
        sub = pd.read_csv(io.BytesIO(z.read(name['SUBMISSION.TSV'])), sep='\t', dtype=str)
        sub = sub[sub.SUBMISSIONTYPE == '13F-HR'][['ACCESSION_NUMBER', 'FILING_DATE', 'CIK', 'PERIODOFREPORT']]
        chunks = []
        with z.open(name['INFOTABLE.TSV']) as fh:
            for ch in pd.read_csv(fh, sep='\t', dtype=str, usecols=['ACCESSION_NUMBER', 'CUSIP', 'SSHPRNAMT', 'SSHPRNAMTTYPE', 'PUTCALL'], chunksize=500_000):
                ch = ch[ch.CUSIP.isin(cusips) & (ch.SSHPRNAMTTYPE == 'SH') & ch.PUTCALL.isna()]
                chunks.append(ch[['ACCESSION_NUMBER', 'CUSIP', 'SSHPRNAMT']])
        it = pd.concat(chunks)
        it['shares'] = pd.to_numeric(it.SSHPRNAMT, errors='coerce')
        it = it.merge(sub, on='ACCESSION_NUMBER', how='inner')
        it['symbol'] = it.CUSIP.map(cusips)
        it['period'] = pd.to_datetime(it.PERIODOFREPORT, format='%d-%b-%Y')
        it['filed'] = pd.to_datetime(it.FILING_DATE, format='%d-%b-%Y')
        lag = (it.filed - it.period).dt.days
        late = it[(lag > 60) | (lag < 0)]
        print(f"  dropping {len(late)} rows filed > 60 days after (or before) the period end "
              f"({late.shares.sum() / max(it.shares.sum(), 1):.1%} of shares); periods now used from {it[lag.between(0, 60)].period.min().date()}", flush=True)
        it = it[lag.between(0, 60)]
        # one row per manager x period x symbol (sum classes/CUSIPs); if a manager filed twice, keep the latest
        per = it.groupby(['CIK', 'period', 'symbol', 'filed'], as_index=False)['shares'].sum()
        per = per.sort_values('filed').drop_duplicates(['CIK', 'period', 'symbol'], keep='last')
        parts.append(per)
        print(f"{os.path.basename(f)}: {len(sub)} filings, {len(per)} manager-holdings rows", flush=True)

    allp = pd.concat(parts)
    allp = allp.sort_values('filed').drop_duplicates(['CIK', 'period', 'symbol'], keep='last')   # across archives
    # point in time: keep only filings published before the quarter's first use (crowded_spaces LAG_DAYS = 46)
    days = pd.read_csv(os.path.join(SCRIPT_DIR, 'backtest_data', 'sp500_prices_long.csv'), index_col=0,
                       parse_dates=True, usecols=[0]).index
    month_ends = pd.DatetimeIndex(sorted(pd.Series(days).groupby([days.year, days.month]).max().values))
    n0, s0 = len(allp), allp.shares.sum()
    allp = point_in_time(allp, month_ends)
    print(f"point-in-time: dropped {n0 - len(allp)} manager-holdings filed on or after the quarter's first use "
          f"({1 - allp.shares.sum() / s0:.2%} of shares)", flush=True)
    agg = allp.groupby(['period', 'symbol']).agg(inst_shares=('shares', 'sum'), n_managers=('CIK', 'nunique'),
                                                filed_first=('filed', 'min'), filed_median=('filed', 'median'), filed_last=('filed', 'max')).reset_index()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(agg)} rows, {agg.symbol.nunique()} symbols, periods {agg.period.min().date()}..{agg.period.max().date()}")


if __name__ == '__main__':
    main()
