# Residual stat arb after Avellaneda & Lee (T-2026-10-07-97): FAIL

**What it is.** The "hedge fund version" of pairs trading, as published by Avellaneda & Lee
(Quantitative Finance, 2010).
- Each S&P 500 stock is compared with its best-matching sector ETF over 60 days.
- The part of its move the sector doesn't explain (the residual) is modelled as mean-reverting.
- The stock is traded when its residual is stretched: short above +1.25 standard deviations, long
  below −1.25, with the ETF as the hedge.
- About 170 positions are open at once, roughly 3,400 trades a year, each held about two weeks.
- Pre-registered in `statarb_al.py`.
- Changes from the paper:
  - the stock's best-correlated sector SPDR replaces its industry ETF;
  - execution is at the next close, not the signal close;
  - costs are 6 bp per side, not 5 bp.

**Results** (Sharpe; net = after costs):

| Execution | Period | Before costs | After costs |
|---|---|---|---|
| **Next close (the decision)** | **2020–2026** | **0.49** | **−0.09** |
| Next close | 2005–2019 | 0.70 | −0.21 |
| Same close (the paper's timing) | 2020–2026 | 0.61 | 0.07 |
| Same close | 2005–2019 | 1.16 | 0.30 |

Rules P1, P2 and P4 failed. The last 24 months (+1.0%) and the correlation with SPY (+0.20) passed.

**What it shows.**
1. **The signal is real before costs.** It had a Sharpe of 1.16 at the paper's timing on 2005–19,
   close to the paper's 1.1 for 1997–2007.
2. **A one-day delay costs about a third of it.** 1.16 fell to 0.70, because much of the reversion
   happens on the next day.
3. **Trading costs take the rest.** At about 3,400 trades a year, 12 bp per round trip is roughly
   equal to the gross edge.
4. **The edge has shrunk further since 2020.** Gross 0.49 at next-close execution.
5. **It is the same lesson as the literature and T-96.** The idea still exists. What makes it pay is
   executing within the day at institutional costs, which is a scale-and-execution business, not
   an idea an individual can trade.

**Status.** Large-cap stat arb is closed for this project. The small/mid-cap version remains part
of FAILURE_AUTOPSY.md H1. Spreads there are wider, so the cost hurdle is higher.
