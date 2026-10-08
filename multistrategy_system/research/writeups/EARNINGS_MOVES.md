# Big moves on earnings days versus other days (2026-10-02, T-2026-10-02-84, -85)

Research only: `earnings_moves.py` (pre-registered in its docstring before any return was
computed). Data: IBKR daily closes, point-in-time S&P 500 members, and 46,576 SEC 8-K item-2.02
(earnings) filings for 567 companies with acceptance times (`backtest_data/sec_8k_202.csv`).
Results in `results/earnings_moves/`.

Before the first run, a check of known release times showed that EDGAR acceptance times are UTC,
not Eastern as first assumed: after-close filings cluster at 20:00–21:59 UTC. The code was fixed to
convert to New York time before any result existed. Item-2.02 tagging starts in August 2004, so the
context window starts in 2005.

**Setup.**
- **Event:** a stock-specific move (stock minus the equal-weight member return) of at least 3
  standard deviations of its own last 63 days.
- **Earnings event:** an earnings reaction day falls within one day of the move.
- **Trades:** entered at the next day's close and hedged with the market.
  - NR fades non-earnings moves for 5 days.
  - ED follows earnings moves for 20 days.
- **Costs:** 6 bp per side, plus 0.5% a year borrow on shorts.

| Leg (2020–Sep 2026) | Net Sharpe | Gross Sharpe | Gross t | Last 24 months | Verdict |
|---|---|---|---|---|---|
| NR: fade big non-earnings moves, 5 days | −0.46 | −0.06 | −0.15 | −1.20 | FAIL |
| ED: follow big earnings moves, 20 days | −0.49 | −0.17 | −0.44 | +0.04 | FAIL |

**Event study** (mean signed stock-specific return; negative = reversal; t computed by date, so
each day counts once):

| | Events (dates) | t+2..6 | t by date | t+2..21 | t by date |
|---|---|---|---|---|---|
| 2020–26, non-earnings | 8,736 (1,343) | −0.56% | −0.29 | −0.59% | −1.35 |
| 2020–26, earnings | 5,171 (1,032) | +0.08% | +0.58 | +0.05% | −0.12 |
| 2005–19, non-earnings (context) | 11,440 (3,064) | −0.23% | −2.25 | −0.35% | −2.06 |
| 2005–19, earnings (context) | 7,608 (2,176) | +0.03% | +0.83 | +0.13% | +0.03 |

**Reading.**
1. **The apparent reversal is a handful of market-wide days, not a stock-level effect.**
   - Counted event by event, non-earnings moves "reverse" with t = −7.6.
   - Counted by date it is t = −0.3. The events cluster on a few days: 9 November 2020 (the vaccine
     rotation, 210 events), March 2020 (about 170 a day) and 3 April 2025 (tariffs, 191).
   - On those days the "stock-specific" moves are really sector and style rotations, which then
     partly unwound.
   - The calendar-time trading leg, which is what could actually be traded, earns nothing before costs.
2. **Some reversal existed in 2005–19** (t about −2.2 by date, in a survivorship-biased sample) **and
   is gone since 2020.** That fits other evidence that short-horizon reversal in large caps has been
   competed away.
3. **No post-earnings drift in large caps since 2020** (t near 0). That is consistent with the
   literature reporting that the drift has shrunk sharply in large, liquid stocks.
4. **Every pre-registered rule fails for both legs.** The legs are uncorrelated with the existing
   sleeves (|corr| ≤ 0.09), but there is nothing to diversify with.

**Methodological lesson worth keeping.** Event-level t-statistics overstate significance when
events cluster in time. Calendar-time portfolios, or date-clustered t-statistics, are the honest
test. The script now reports both.
