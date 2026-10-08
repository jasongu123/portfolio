# S&P 500 additions and deletions (2026-10-02, T-2026-10-02-86, -87)

Research only: `index_rebalance.py` (pre-registered in its docstring before any return was
computed). Data: IBKR daily closes; changes and effective dates from the S&P 500 change log.
Results in `results/index_rebalance/`.

| Trade (2020+, entry close E−1, 20 days, hedged, net) | Trades (dates) | Mean trade | t by date | Net Sharpe | Verdict |
|---|---|---|---|---|---|
| ADD: short additions after inclusion | 114 (78) | +1.51% | 1.25 | 0.21 | FAIL |
| DEL: long deletions after removal | 65 (36) | −2.05% | −1.15 | −0.43 | FAIL |

Event study of the stock's own stock-specific return (t by effective date):

| | Run-up E−11..E−2 | E−1 (funds trade) | E..E+4 | E..E+19 | E..E+59 |
|---|---|---|---|---|---|
| Additions 2020+ (114) | **+4.31% (+4.2)** | −0.08% | −1.04% (−1.6) | −1.88% (−1.4) | +0.44% |
| Additions 2005–19, survivor-biased (148) | +1.04% (+2.4) | −0.12% | −0.06% | −0.65% | −0.45% |
| Deletions 2020+ (65) | −2.37% (−1.1) | +0.44% | −1.17% | −3.01% (−1.1) | −1.26% |

**Reading.**
1. **The inclusion effect is alive before the effective date:** +4.3% stock-specific in the two
   weeks before inclusion since 2020 (t 4.2), larger than in 2005–19. That window contains the
   announcement, usually about five trading days before the effective date and released after the
   close, so much of the gain is probably an overnight jump that cannot be captured. **Whether any
   of it is tradable needs announcement dates, which this data does not have.**
2. **The reversal after inclusion is real in direction but too small and noisy to trade:**
   −1.9% over 20 days, t −1.4. After costs, shorting additions earns +1.5% per trade with t 1.25
   and Sharpe 0.21.
3. **Deletions do not rebound; they keep falling.** Stocks are removed after losing value, and the
   weakness continues. There is no liquidity-provision payoff here.

**Next, if pursued:** announcement dates from S&P Dow Jones Indices press releases (or the
change log's cited references). Then a pre-registered test of buying at the first open after the
announcement and selling at the E−1 close.

## Follow-up with announcement dates (T-2026-10-03-95): FAIL

**Setup.** Announcement dates came from S&P's press releases, cited in Wikipedia's historical change
table (`index_announcements.py`; 98% coverage for 2020 onward). The trade bought each addition at
the close of the first trading day after the announcement, hedged it with the market, and sold to
the index funds at the close of E-1. `index_announce_trial.py`, pre-registered before any return was
computed.

**Where the 4.3% run-up actually happens** (106 additions, 2020 onward; stock-specific return, t by
announcement date):

| Window | Return | t |
|---|---|---|
| 10 days up to the announcement (before the release) | +2.29% | 2.4 |
| First day after the announcement (the jump; untradable) | +3.79% | 5.5 |
| From that close to E-1 (the trade) | +0.14% | 0.2 |
| E-1 alone | −0.09% | −0.2 |
| 20 days after inclusion | −1.17% | −0.8 |

The trade averaged +0.02% (t 0.03) and had a Sharpe of 0.03. All four rules failed.

**Mechanism.** The effect is real but fully priced in two places:
- **Before the announcement.** Traders anticipate likely additions, because S&P's eligibility rules
  (size, profitability, liquidity) are public.
- **At the first open after it.**

By the time anyone can trade on the news, nothing is left. In 2005–2019 the same window was −0.81%
(t −3.1). This is the same lesson as guidance revisions (T-93): large caps price public news within
a day.

**Context only, not a finding.** Shorting deletions over the same window had a Sharpe of 0.65 and t
of 1.36 on 52 trades. That was not the registered question, and there is no fresh sample to confirm
it (only 8 priced deletions before 2020).

**What is left.** The only open part is *predicting* additions before the announcement, using S&P's
published eligibility rules: market cap, GAAP profitability (free in SEC XBRL data) and liquidity.
The +2.3% pre-release drift shows some traders already do this.

## Where the announcement jump happens: extended-hours minute bars (measurement, 2026-10-03)

`index_afterhours.py` pulled IBKR 1-minute bars including extended hours for all 106 additions
since 2020 (106 of 106 returned data). Returns are stock minus SPY; t is by announcement date.

| Checkpoint | Done since the 4 pm close | Left until the next 4 pm close |
|---|---|---|
| 5:00 pm (before the release) | +0.20% | +3.63% |
| **5:30 pm** | **+3.57% (t 8.6)** | **+0.27% (t 0.4)** |
| 6:00 pm | +3.85% | −0.01% |
| 8:00 pm | +3.79% | +0.04% |
| 9:00 am (premarket) | +3.96% | −0.22% |
| 9:30 am (open) | +3.56% | +0.28% |
| Next close | +3.82% | — |

**The jump happens in the first minutes after S&P's ~5:15 pm release.** The median time of the
largest one-minute after-hours move is **17:16**. By 5:30 pm, 93% of the move is done. Anyone buying
from then on earns about nothing, even before after-hours spreads.

Super Micro (March 2024), the hand-checked example, drifted up for hours, but it was an outlier.
Capturing the jump means trading within a minute of the release, which is a speed contest against
firms with direct feeds.

**No trade test was registered,** because the measurement shows nothing left at any reachable time.
The index-addition line is closed. The only remaining angle is predicting additions before the
release.
