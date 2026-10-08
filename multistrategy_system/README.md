# multistrategy_system

A systematic trading system for US equities and ETFs on the Interactive Brokers TWS API. It runs
in paper mode by default and is organised as independent sleeves that share one cost model, one
execution journal, and one event calendar.

The live engine and its strategy class are kept private. The cost model, execution journal,
event and earnings layers, ingestion package, trend sleeve, news filter, backtester, and the tests
they depend on are here.

## What is here

| Area | Files | Notes |
|---|---|---|
| Trend sleeve | `trend_follow_v2.py`, `trend_follow_v3.py`, `trend_sleeve_live.py`, `pull_trend_ibkr.py`, `check_trend_positions.py`, `combine_book_v2.py`, `rates_attribution.py` | Monthly cross-asset time-series momentum: sign of 3, 6 and 12-month returns, volatility-targeted. `docs/RATES_ATTRIBUTION.md` and `docs/TREND_V3.md` cover attribution and the v3 refinements. |
| Tech sleeve | `tech_sleeve_live.py`, `tech_sleeve_research.py` | Disabled-by-default candidate sleeve. See `docs/TECH_SLEEVE.md`. |
| Costs and execution | `trading_costs.py`, `execution_journal.py`, `execution_cost_report.py` | Shared cost model, observe-only execution journal, and the report that reconciles modelled against realised costs. `docs/COST_MODEL.md`. |
| Events | `event_calendar.py`, `earnings_events.py`, `earnings_parser.py`, `earnings_history_extract.py`, `earnings_replay.py`, `pull_earnings.py`, `earnings_dates.json` | Point-in-time event plumbing and the earnings embargo layer. `docs/EARNINGS_LAYER.md`. |
| News | `premarket_news.py`, `news_analyzer.py` | Paper-only premarket news filter. Deterministic policy is separate from model classification. `docs/PREMARKET_NEWS.md`. |
| Data ingestion | `data_ingestion/` | Alpha Vantage, FRED and Treasury macro series, and IBKR news into an immutable SQLite store. `docs/DATA_INGESTION.md`. |
| Research | `ibkr_backtest.py`, `deflated_sharpe.py` | Backtest engine that replays the live logic (its IBKR data-fetch paths expect the private engine; the cost and replay paths do not), and a deflated-Sharpe check for multiple testing. |

`docs/PORTFOLIO.md` is the overview: architecture, the validated book, results, and known limits.

The research trail that produced these choices is in [`research/`](research/README.md): a project
summary, the registry of 97 pre-registered trials (most of them failures, each with its likely
cause), one write-up per study, and the research scripts.

## Running

Python 3.12 with `pandas`, `numpy`, `scipy`, `scikit-learn`, `arch`, and `ibapi` (the TWS API
client from Interactive Brokers, not on PyPI). Tests need `pytest` and do not need a running TWS:

```bash
python -m pytest tests
```

Configuration is by environment variable, never in source:

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `news_analyzer.py` |
| `ALPHAVANTAGE_API_KEY` | `data_ingestion/alpha_vantage.py` |
| `RESEARCH_DATA_ROOT` | ingestion store location, default `data_lake` |

Every sleeve here runs in paper mode unless overridden by environment variable. Notional budgets and caps in the live sleeves
are paper-trading figures.
