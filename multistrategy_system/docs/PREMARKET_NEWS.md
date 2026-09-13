# Premarket news paper sleeve

The Claude integration is an isolated premarket event sleeve. It observes news
from the previous regular close through 9:30 a.m. ET and can submit paper orders
between 7:00 and 9:30. It does not alter the validated core strategy by default.

## Default controls

```text
NEWS_OBSERVER_ENABLED=true
PREMARKET_NEWS_PAPER_ENABLED=true
NEWS_CORE_VETO_ENABLED=false
NEWS_RF_OVERRIDE_ENABLED=false
NEWS_POSITION_SIZING_ENABLED=false
```

The premarket sleeve also requires `LIVE_TRADING = False`. The code refuses to
enable it on the live port even if the environment flag is set.

Set the API key before starting `ibkr2.py`:

```bash
export ANTHROPIC_API_KEY='your-key'
python3 ibkr2.py
```

The default classifier is the pinned `claude-haiku-4-5-20251001`, chosen for a
small, frequent classification workload. Override it without editing source:

```bash
export ANTHROPIC_NEWS_MODEL='claude-sonnet-4-5-20250929'
```

## Decision pipeline

1. At 7:00 a.m., request news since the previous weekday's 4:00 p.m. close.
2. After the first scan, process only unseen `(provider, article_id)` events.
3. Use IBKR article text when the subscription permits it; otherwise classify
   the headline. DuckDuckGo is used only if IBKR news is unavailable, not merely
   because IBKR has no new event.
4. Claude returns a schema-constrained classification: event type, scope,
   direction, severity, novelty, relevance, confidence, horizon and evidence.
5. Deterministic policy—not Claude—decides paper eligibility.

The initial paper policy requires:

- bullish or bearish direction;
- severity 3;
- new event or material update;
- relevance and confidence of at least 0.65;
- a real bid/ask quote with spread no wider than 1%; and
- no premarket gap of at least 0.5% in the opposite direction.

Market-wide catalysts are expressed only through `SPY`, `QQQ`, or `IWM`, rather
than creating the same trade independently in every stock.

Eligible trades use the sleeve's existing 0.5× position-size multiplier. Limit
prices are capped around the midpoint by the 0.3% execution buffer. After an
order is submitted, any unfilled remainder is cancelled and the position is
reloaded from IBKR. Stops, targets and exits operate only on the reconciled
position. Long positions exit with `SELL`; short positions exit with `BUY`.

## Research journal

Every successful or failed model assessment is written to:

```text
data_lake/research.sqlite3 → news_assessments
```

Every deterministic decision is written to:

```text
data_lake/research.sqlite3 → premarket_news_decisions
```

The journal includes model and prompt versions, event IDs, source status,
latency, token usage, quote, spread, previous close, gap, order lifecycle, exit
P&L, and raw forward returns at 9:30, 10:00, 10:30 and 16:00.

Example review query:

```bash
sqlite3 -header -column data_lake/research.sqlite3 "
SELECT d.decided_at, d.symbol, a.event_type, a.direction, a.severity,
       d.action, d.status, d.gap_pct, d.pnl_pct, d.return_close
FROM premarket_news_decisions d
JOIN news_assessments a USING (assessment_id)
ORDER BY d.decided_at DESC;
"
```

These thresholds are paper hypotheses, not validated parameters. The journal is
the evidence needed to compare event categories and continuation horizons before
any part of this sleeve is considered for live trading.
