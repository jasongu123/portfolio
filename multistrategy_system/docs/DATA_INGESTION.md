# Research data ingestion

This pipeline collects free research inputs without changing `ibkr2.py` or the
live strategy database. Every provider response is saved immutably under
`data_lake/raw/`; normalized rows are upserted into
`data_lake/research.sqlite3`.

`data_lake/` is ignored by Git because it can become large. API credentials are
read from environment variables and are never written to the raw archive.

## What is automated

| Collector | Data | Credential | Practical use |
|---|---|---|---|
| US Treasury | Nominal and real daily curves | None | Curve, duration, inflation, and defensive regimes |
| FRED | Credit spreads, curve spreads, real yields, policy rates | None | Credit/risk regime features |
| New York Fed | SOFR history | None | Funding/liquidity regime features |
| Alpha Vantage | Earnings histories | Free API key | Earnings-event filters and event sleeves |
| Alpha Vantage | Active and delisted listings by date | Free API key | Point-in-time universes and survivorship controls |
| IBKR | Historical headlines and article bodies | TWS/Gateway + account news access | Forward news archive and shadow-mode filter research |

All commands use only the Python standard library except the IBKR command,
which uses the project's existing `ibapi` installation.

## Commands

Run commands from the repository root with the same Python used by the trading
project.

Backfill all default macro series from 2007 through today:

```bash
python3 -m data_ingestion.run macro --start-year 2007
```

For a quick daily refresh, requesting only the current Treasury year is enough;
the FRED and NY Fed calls remain idempotent:

```bash
python3 -m data_ingestion.run macro --start-year 2026
```

The default FRED set is:

- `BAMLH0A0HYM2`: high-yield option-adjusted spread
- `BAMLC0A0CM`: investment-grade corporate option-adjusted spread
- `BAA10Y`: long-history Baa corporate spread over the 10-year Treasury
- `NFCI`: Chicago Fed National Financial Conditions Index
- `STLFSI4`: St. Louis Fed Financial Stress Index
- `T10Y2Y` and `T10Y3M`: Treasury curve slopes
- `T10YIE`: 10-year breakeven inflation
- `DFII10`: 10-year real yield
- `SOFR` and `DFF`: overnight funding and policy rates

Override it with, for example:

```bash
python3 -m data_ingestion.run macro --skip-treasury --skip-nyfed \
  --fred-series VIXCLS NFCI STLFSI4
```

Get a free Alpha Vantage key, export it in the shell, then collect the current
12-symbol equity universe. The command deliberately defaults to no more than 25
calls per run to stay within the commonly applicable free daily allowance.

```bash
export ALPHAVANTAGE_API_KEY='your-key'
python3 -m data_ingestion.run alpha-earnings
python3 -m data_ingestion.run alpha-listings --date 2026-07-11 --state both
```

Custom tickers can be supplied as a comma-separated value or a text file:

```bash
python3 -m data_ingestion.run alpha-earnings --symbols AAPL,NVDA,MSFT
python3 -m data_ingestion.run alpha-earnings --symbols-file research_symbols.txt
```

With paper TWS or IB Gateway running, API socket access enabled, and a unique
client id, archive the equity universe's last three days of news:

```bash
python3 -m data_ingestion.run ibkr-news --port 7497 --client-id 91
```

Use `--headlines-only` if article bodies are unavailable. For a continuously
running forward archive, `--interval 1800` repeats the pull every 30 minutes.
IBKR limits a historical-news request to 300 headlines; `--max-results` is
therefore capped at 300 by the collector. The archive deduplicates articles
that appear under more than one symbol.

## Storage and querying

The normalized SQLite tables are:

- `macro_observations`
- `alpha_earnings`
- `alpha_listings`
- `news_articles`
- `news_article_symbols`
- `raw_objects` and `checkpoints`

Example research query:

```bash
sqlite3 data_lake/research.sqlite3 \
  "SELECT observation_date, value FROM macro_observations
   WHERE source='fred' AND series_id='BAMLH0A0HYM2'
   ORDER BY observation_date DESC LIMIT 10;"
```

The raw layer is the audit trail. If a parser or feature definition changes,
the normalized database can be rebuilt later without re-downloading data.
FRED macro histories can be revised. The archive preserves each response from
the day this collector starts, but a historically honest vintage backtest may
still require ALFRED vintages rather than today's revised history.

## Suggested cadence

- Macro: once after the US close on trading days.
- Earnings: rotate through the relevant stock universe daily within the API
  allowance; increase coverage only if the account's allowance permits it.
- Listings: save active and delisted snapshots weekly or monthly, and around
  universe reconstitutions.
- IBKR news: every 30–60 minutes while TWS/Gateway is available. More frequent
  polling is useful only after the archive has shown that freshness matters.

Keep the collectors in shadow mode first. The next research step is to derive
features from these stored observations and test them out of sample before any
feature is allowed to veto or resize live trades.
