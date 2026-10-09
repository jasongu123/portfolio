# Systematic Trading Research on Interactive Brokers — Project Summary

*As of 9 October 2026 (numbers corrected after an external code audit). Paper trading only.*

## 1. In one paragraph

Since mid-2026 I have built and run a multi-strategy systematic trading system on Interactive Brokers (paper account), deployed on a Windows cloud VM, together with a research program that tests every idea under rules fixed in advance. The system trades a cross-asset trend-following sleeve, an intraday ETF momentum sleeve based on a published paper, and a stock swing book, plus two paper candidates. The research record holds **97 pre-registered trials**, each logged with its pass/fail rule before any result was seen. Most ideas failed, and each failure is written up with its likely cause. The most important results were diagnostic: taking my own engine apart showed where its profit really came from, and a faithful replication of the source paper showed that my own modifications had turned a working strategy into a losing one. After the research, the live system was restructured to its three working parts; that book backtests at a Sharpe of 1.70 (0.98 after a 5% cash rate) with a −5.8% maximum drawdown over 2020–2025, in-sample for the restructure decisions and now under live forward test. The latest project taught a small free model to read earnings press releases the way a large language model does: it learned to label them as well as Claude Haiku, and its labels identified real price-moving news, but that news was fully priced within a day.

## 2. What runs live (paper)

| Sleeve | What it does | Status |
|---|---|---|
| Cross-asset trend | Time-series momentum on 16 ETFs (equities, Treasuries, commodities, currencies, real estate), 3/6/12-month signal ensemble, rebalanced monthly at month-end | Validated |
| ETF intraday momentum | SPY, QQQ, IWM on the rules of Zarattini, Aziz & Barbon (2024), "Beat the Market": a trade when price leaves its normal intraday range (the 14-day average move from the open), decisions every 30 minutes, trailing stop at the range or VWAP, flat by the close, size targeting 2% daily volatility with a 4× cap | Forward paper test from Oct 2026 |
| Stock swing book | 21 large-cap stocks; multi-day long positions on momentum with ATR trailing stops | Running; treated as risk-managed market exposure, not alpha (see 5.3) |
| RFS (VOO) and intramonth sector rotation | Paper candidates from earlier research | Candidates only |
| Premarket news | An LLM (Claude) classifies overnight news per stock (event type, direction, severity, novelty); half-size paper trades on high-conviction events | Collecting data |

**Engineering:**
- **Live engine:** about 9,400 lines of Python on the IBKR API.
- **Backtester:** a separate minute-bar backtester of about 6,400 lines.
- **Tests:** 195 automated tests.
- **Execution-cost journal:** records every order's decision-time midpoint and the fill, so slippage is measured, not assumed.
- **Deployment:** checksummed deployment bundles to the VM, with version checks before install.
- **Sleeve separation:** each sleeve runs on its own IBKR client ID, with no overlapping symbols.
- **Safety rule:** entry controls can never block an exit.

## 3. Research method

- **Pre-registration.** Every trial's pass rule is written into the script's docstring and the trial registry before the first run.
- **Walk-forward everything.** Models are retrained only on past data, and portfolio weights are walk-forward. No full-sample weights, and no same-day data that would not have been known.
- **Realistic costs.** $0.0035 per share commission plus 1 bp per side for stocks, with sensitivity runs at other cost levels.
- **Recency and concentration rules.** These were added after older Sharpe-only rules were passed by strategies whose profit came from a few early years:
  - the last two years must be profitable;
  - no single two-year block may supply more than 75% of the profit.
- **Multiple-testing awareness.** Deflated Sharpe ratio on the headline, with the trial count tracked.
- **Conventions kept separate.** Sharpe both with a 5% cash rate subtracted ("rf") and without it ("no-rf"), never mixed. Drawdown is measured from initial capital.

## 4. Headline numbers (backtest, 2 Jan 2020 to 15 Dec 2025)

| | Sharpe, rf | Sharpe, no-rf | Max drawdown | Return a year |
|---|---|---|---|---|
| **Restructured book: trend + ETF momentum (paper rules) + stock swing book**, walk-forward inverse-vol weights | **0.98** | **1.70** | **−5.8%** | +12.2% |
| Old book (original engine + trend), same method and window | 0.44 | 1.04 | −8.2% | +8.7% |
| Trend sleeve alone | 0.41 | 0.81 | −22.2% | +9.7% |
| ETF momentum alone (paper rules) | 0.42 | 0.81 | −15.5% | +9.9% |
| Stock swing book alone | 0.79 | 1.19 | −13.1% | +15.0% |

**Why the book beats every sleeve.** The three sleeves barely move together: daily correlations
are −0.12, −0.02 and +0.01. The walk-forward weights average about one third each, so one sleeve's
bad stretch is mostly offset by the others. Every year was positive on rf Sharpe; 2025 was the
weakest (+0.04). The last 24 months came in at 1.50 no-rf.

**Limits on this headline:**
- **Not out of sample.** The restructure was chosen using these same years: dropping mean reversion
  and adopting the paper's ETF rules came from the split test (5.4, 5.5). The ETF rules were
  published in 2024 and not tuned here, but the decision to use them came after seeing this data.
  The live forward tests are the real check; expect them to be weaker.
- **The ETF sleeve is cost-sensitive.** These figures assume 0.5 bp a side, and its Sharpe roughly
  halves at 1 bp.
- **The stock list carries hindsight** (5.3).
- **The swing book adds nothing at book level.** Replacing it with an equal-weight basket of the same
  stocks gives the same book (1.71 no-rf).

All figures are after the audit fixes in 5.11; before them the old book read 0.51 / 1.13.

## 5. Key findings

### 5.1 The trend sleeve earns at month-end because it supplies liquidity
I expected month-end, the most crowded execution window, to be the worst day to rebalance. Testing all 21 possible rebalance days showed the opposite: month-end was the best and early-month the worst. The likely mechanism is that pension and fund rebalancing at month-end trades against recent trends, and a trend follower is paid for taking the other side.

Related results:
- **Treasuries:** the Treasury ETFs add about a fifth of the sleeve's return for an eighth of its risk, and most of its crisis protection (2022: +30% with them, +22% without).
- **More ETFs:** expanding from 16 to 46 ETFs *lowered* the Sharpe. The additions earned about half as much per slot, and currencies lost money.

### 5.2 Taking the equity engine apart
I tagged every trade by the rule that opened it. Results over 2020–25 (24 stocks, gated, corrected engine), as % of capital:

| Component | Contribution |
|---|---|
| Swing holdings | +80.6% |
| ETF intraday momentum | −2.7% |
| Stock intraday mean reversion | −8.2% (it lost money even before costs) |
| Gap trades | never fired: the entry window was 9:31–9:35, but the backtester acts only from 10:00 |

### 5.3 The swing book reduces risk; it does not add return
The swing-only stock book (21 stocks, corrected engine, 2 Jan 2020 to 15 Dec 2025) compared with SPY and with equal-weight baskets of the same stocks:

| | Return a year | Volatility | Sharpe (no-rf) | Sharpe (rf) | Max drawdown | Last 2 years (Sharpe) |
|---|---|---|---|---|---|---|
| Swing book | 14.8% | 12.4% | 1.19 | 0.79 | **−13.1%** | 1.27 |
| SPY, buy and hold | 16.2% | 20.8% | 0.78 | 0.54 | −33.7% | 1.27 |
| Same 21 stocks, equal weight, fully invested | 25.2% | 22.3% | 1.13 | 0.90 | −30.8% | 1.74 |
| Same basket at the swing book's average exposure (71% invested) | 17.8% | 15.8% | 1.13 | 0.81 | −22.4% | 1.74 |
| Same basket at the swing book's *daily* exposure | 16.6% | 12.5% | **1.32** | **0.92** | −17.6% | 1.94 |

What this shows:
- **Against SPY.** The swing book had a higher Sharpe (1.19 versus 0.78), 40% less volatility, a drawdown of −13% instead of −34%, and a beta to SPY of only 0.37. Most of that advantage comes from **which stocks are in the book**, though: the same 21 stocks simply held in equal weight also beat SPY (1.13). Those names were chosen in 2025–26, partly by their current size, so the stock list carries hindsight.
- **Risk.** The swing book cut the maximum drawdown by about 40% (−13% versus −22%), for 3 points less return a year. Risk-adjusted, it roughly ties the basket: +0.06 no-rf Sharpe, −0.02 rf.
- **Source of the benefit.** The benefit comes from *how much* it is invested day to day, not *which* stocks it holds. Copying only its daily exposure onto a passive basket gives a higher Sharpe than the swing book itself (1.32 versus 1.19).
- **Timing as insurance.** The timing works like crash insurance (2022 Sharpe −0.23 versus −0.55), paid for with lagging rallies (2024–25).
- **Verdict.** It failed the pre-registered test of "timing is an edge", which needed +0.10 Sharpe and a better last two years.

### 5.4 Replicating the source paper exposed my own modifications
The paper's exact rules on SPY/QQQ/IWM, 2020–25, at 0.5 bp per side: **Sharpe 0.81** (passed all three rules). My engine's version: **−0.89**. I added each of my changes back one at a time:

| Change from the paper | Effect on Sharpe |
|---|---|
| Deciding every minute instead of every 30 minutes | −1.20 (4× the trades; gross edge 3.4 → 0.9 bp per trade) |
| ML gate | −0.38 |
| Range twice as wide | −0.43 |
| VIX ≥ 17 filter | −0.19 |
| RSI size multiplier | −0.32 |

Limits of this result:
- 2025 was a losing year.
- It is cost-sensitive: in the original run, 1 bp per side cut the Sharpe to about 0.5 and 2 bp made it negative.

That is why the live forward test also checks real fill quality.

### 5.5 Machine learning: a trade filter
I built a random-forest gate trained walk-forward on 33 features. It screened **intraday** entries (stock mean reversion and ETF momentum) and blocked those whose predicted win probability was below a threshold. Swing entries never pass through it.
- **It helped the original engine:** no-rf Sharpe 0.96 → 1.07 on the 24-name book (21 stocks plus SPY, QQQ and IWM), 2020–25.
- **Where the help came from (measured).** The same 24-name engine was run with the gate on and off, on the corrected engine, 2020–25. Each day's profit was tagged by the strategy that earned it, as percentage points of account return:

| Strategy | Gate off | Gate on |
|---|---|---|
| ETF intraday momentum (the engine's modified version) | −8.0 | −2.7 |
| Stock intraday mean reversion | −10.7 | −8.2 |
| Swing holdings (not gated) | +80.7 | +80.6 |

All of the gate's value was shrinking the losses of two intraday strategies that lost money with or without it.
- **Removing the losing strategy beat filtering it.** This was a separate split test on the same 21 stocks, the same dates (2 Jan 2020 to 15 Dec 2025) and the corrected engine, with its own $100k:
  - the engine as it was (mean reversion + swing + gate): Sharpe 1.08, last two years 1.03;
  - swing only, with no mean reversion and no gate: **1.19**, last two years 1.27.
  - With mean reversion gone, the stock side has no intraday trades left for a gate to screen, so "swing + gate" would be identical to swing only.
- **It hurt once the ETF strategy was fixed.** On the paper-rule ETF strategy the gate cut Sharpe from 0.81 to 0.43. One likely reason: momentum profits come from a few large winners, which are the hardest trades to predict, so the filter blocks many of them. I removed the gate.
- **Model type did not matter.** Random forest, logistic regression, LightGBM and calibrated LightGBM gave portfolios 98–99% correlated, within 0.1 Sharpe of each other. A linear model tied the forest.
- **One model per stock tied one pooled model** (1.19 versus 1.22, 2021–25).
- **Why there was a ceiling:** about 200 noisy labels per fit, and a target close to random. The limit was the data, not the algorithm.
- **Which runs are corrected.** The gate-model comparisons (model type, per-stock versus pooled, 0.96 → 1.07) ran on the engine before the audit fixes and have not been rerun. The gate on/off table and the split test above use the corrected engine.

### 5.6 Machine learning: ranking S&P 500 stocks
Ridge regression and LightGBM on 17 price-based features, monthly decile long-short, walk-forward, out of sample 2021–Sept 2026:
- Sharpe −0.32 and −0.39. A textbook composite with no ML scored −0.30.
- The information coefficient was about 0.

Every input was a published price pattern that earlier trials had already found dead on large caps since 2020. Combining dead signals does not revive them. The next step would be new information (company financials, analyst revisions, news text), not a new model. Section 5.8 tests the text route.

### 5.7 Machine learning: forecasting volatility
Volatility, unlike returns, is persistent and predictable, so it is a fairer target for ML. I tested whether models with 19 engineered inputs beat HAR, the industry-standard benchmark (a linear regression on three volatility horizons), at forecasting next month's volatility. The loss was QLIKE, which scores both how well a model ranks assets and the level of its forecasts.

**Test 1: S&P 500 stocks.** About 480 stocks a month, 2021–2026, 32,000 forecasts, walk-forward.

| Model | Error vs HAR | Ranking (rank corr) |
|---|---|---|
| HAR (benchmark) | — | 0.64 |
| Ridge regression, 19 inputs | 8% lower (t 2.6) | 0.69 |
| LightGBM, 19 inputs (registered candidate) | 4% higher | 0.71 (best) |

LightGBM failed the pre-registered rule, even though it ranked stocks best. Its forecast level was too low, mostly in 2021–22, for two reasons:
- tree models cannot extrapolate beyond the range they were trained on, and those years included the 2020 spike;
- I had corrected each model's level using its in-sample fit, which flatters a flexible model.

A follow-up that measured the correction on held-back data narrowed the gap but still failed. The ridge model's gain was only a diagnostic, so I did not claim it.

**Test 2: fresh data.** 48 cross-asset ETFs (equities, bonds, commodities, currencies, real estate), 2012–2026, never used for this question. Pass rules were fixed in advance, with the significance bar raised to t ≥ 2.24 because two models were tested.

| Model | Error vs HAR | Years better | t | Verdict |
|---|---|---|---|---|
| Ridge regression, 19 inputs | 10% lower | 12 of 15 | 2.16 | Fail on significance only |
| LightGBM, 19 inputs | 11% lower, mostly from 2020 | 7 of 15 | 0.92 | Fail |

The ridge model's improvement replicated across two independent datasets and was better in the last two years. It narrowly missed the bar I set, so I treat it as promising, not proven. What generalised was the information (longer-horizon, market-wide and downside volatility), not the nonlinearity: a regularised linear model used it as well as a boosted tree, and more consistently.

**Test 3: sizing the trend sleeve with the forecast.** This failed. Against the 60-day volatility the sleeve already uses, the forecast's error was only 0.7% lower; HAR, the benchmark it had beaten, is worse than the 60-day rule on these ETFs. The forecast also cut positions faster when volatility spiked, which is exactly when trend trades pay: in 2020 the sleeve made +7.1% instead of +12.6%, and Sharpe fell from 0.68 to 0.62. Lesson: test against the rule actually in use, and remember that better forecasts are not automatically better sizing.

### 5.8 Machine learning: teaching a small model to read earnings releases
**Why.** Price-based ML had failed (5.6) because every input was already priced. Text is new information. A large language model can read an earnings release and judge it, but calling it on tens of thousands of documents costs money for every run. So I used **knowledge distillation**:
- **Teacher:** Claude labels a few thousand releases once.
- **Student:** FinBERT, a 110-million-parameter BERT pre-trained on financial text, is fine-tuned on my laptop's GPU to copy those labels. It then labels everything for free.

**Data.** 46,359 SEC 8-K earnings filings (item 2.02) for S&P 500 companies, 2005–2026. The press-release text was kept and the tables removed (median about 16,000 characters).

**Three versions**, each with its pass rule written down before any label existed:

| Version | What changed | Result |
|---|---|---|
| v1 | Claude Haiku labels 2,467 releases ($3.94); the student reads the first 512 tokens | Fail. Guidance 58% (needed 75%); hedging no better than always guessing "medium". In 44% of releases guidance first appears after the student's reading limit. |
| v2 | The student reads the same 12,000 characters as the teacher, in 8 pooled chunks; 4,000 labels; a fresh test set | Fail on guidance only (66.5%). Key finding: **two Claude models (Sonnet 5.5 and Haiku) agreed on guidance only 63% of the time and on hedging 48%.** "Raised versus maintained versus initiated" depends on last quarter's guidance, which one release usually does not contain. The bar was above what the label itself supported. |
| v3 | Only labels the text can answer (guidance given yes/no; revision only if explicitly stated; names a specific headwind yes/no); Sonnet 5.5 as teacher ($25); the bar set from the teachers' own agreement *before* training: "label as well as Haiku, within 3 points" | Teacher agreement rose to 84–92%. The student **beat Haiku** on guidance revision (89.3%) and cautionary language (84.2%). It missed guidance-given by 1 release of 494, beat/miss by 3, and tone clearly. Recorded as a fail; the near-misses are disclosed as within sampling noise. |

**Trading test.** Only the field that passed cleanly, guidance revision, went forward, in a new pre-registered test with a stricter t bar of 2.5. That bar was set because this was the fourth look at the data and the field was chosen after it passed.
- **The student labelled 40,967 releases** in about 1.5 hours.
- **The labels are right.** On the reaction day, "raised guidance" stocks beat the market by **1.0%** and "lowered guidance" stocks lost **3.8%**.
- **There is no drift afterwards.** From the next day, the 20-day stock-specific return was −0.1% for raised and +0.1% for lowered.
- **The trade fails.** Long raised and short lowered, entered after the reaction day, earned a net Sharpe of 0.26 (t 0.70), against 0.5 needed. Most of the profit came from one two-year stretch, and 2005–2019 was negative.
- **A plain keyword search did as well (0.40).** For this question, regular expressions were enough.

**What I take from it:**
- The student must see what the teacher saw.
- Measure how often two strong teachers agree with each other *before* setting accuracy targets. That agreement is the real ceiling.
- Ask for labels that are observable in one document.
- Large caps absorb explicit guidance news within a day. Capturing it would mean trading within seconds of an after-hours release, which is a different, much faster business than a daily strategy.

### 5.9 Pairs trading and statistical arbitrage: replications of two published methods
An earlier version traded 12 related large-cap pairs and failed when one relationship broke (AMD against MSFT/QQQ drifted 5.2 standard deviations, −33%). I tested whether the problem was too few pairs, then tested the design hedge funds use. Both tests used the whole S&P 500 with point-in-time membership, 2020 to Sep 2026, net of costs, under pre-registered rules.

**Pairs, distance method** (Gatev, Goetzmann & Rouwenhorst, 2006). Each month the closest pairs over the past year were traded for six months, opening at a 2-sigma gap and closing when it crossed zero.

| Pairs | 20 | 50 | 100 | 200 |
|---|---|---|---|---|
| Sharpe after costs, 2020+ | 0.02 | 0.16 | **0.22** (registered) | 0.27 |

- **More pairs help only by smoothing.** One broken pair matters less, but the average pair earns little more than its costs (Sharpe 0.46 before costs, t 0.71 after).
- **Most positions never converge.** 57% were still open after six months (mean −5.1%); the 43% that converged averaged +7.3%.
- **The profit comes from crises** (2008 +5.6%, 2009 +10.2%, 2020 +5.9%). 2005–19 was already weak (0.14). This matches Do & Faff (2010), who show the method's returns declining since the 1960s but holding up in turbulent markets.

**Residual statistical arbitrage** (Avellaneda & Lee, 2010). Each stock is traded against its sector ETF when the part of its move the sector doesn't explain is stretched. About 170 positions are open at a time, roughly 3,400 trades a year.

| Execution | Period | Sharpe before costs | Sharpe after costs |
|---|---|---|---|
| Next close (registered) | 2020–2026 | 0.49 | **−0.09** |
| Same close as the signal (the paper's timing) | 2005–2019 | **1.16** | 0.30 |
| Same close as the signal | 2020–2026 | 0.61 | 0.07 |

- **The replication reproduces the paper.** At the paper's timing it earned 1.16 before costs, against their 1.1 for 1997–2007.
- **Speed and cost decide it.** Trading a day later removes about a third of the edge (1.16 → 0.70 on 2005–19), and at 3,400 trades a year, retail costs take the rest. Trading at the signal close is optimistic, since the signal needs that close. The realistic version is a signal at 3:45 pm with market-on-close orders, which lands between the two rows and still fails.

**Conclusion.** Neither method is discredited, but the simple versions are competed away on large caps. What remains is an execution business: thousands of stocks, trading within the day, at institutional costs.

### 5.10 Published anomalies tested out of sample since 2020: all failed

| Idea | Result |
|---|---|
| Firm issuance and short-interest signals ("Taking Sides", McLean, Pontiff & Reilly 2025) | Failed |
| Same-calendar-month seasonality | Reversed in large caps |
| Institutional crowding of anomaly portfolios | Failed |
| Short-interest and bearish-positioning market timing | Failed |
| Intramonth sector rotation | Passed in-sample; replication on 3 other universes found the effect dead since 2024, so it was retired |
| Big moves on earnings days continue, other big moves reverse (Chan 2003) | Failed both ways since 2020. An apparent reversal (event-level t −7.6) was clustering on a few market days; clustered by date, t −0.3 |
| S&P 500 index additions and deletions | Failed. Additions rise about 6% around inclusion, but with announcement dates from S&P's press releases the move splits into +2.3% before the release (others anticipating it), +3.8% within minutes of the ~5:15 pm release (biggest one-minute move at 5:16 pm, measured with after-hours minute bars), and +0.1% afterwards. Nothing is left to trade |

### 5.11 Errors I caught in my own work
- **Same-day VIX look-ahead.** The backtester used the day's VIX close for intraday decisions. Corrected to per-minute VIX; the headline fell from 0.54 to 0.51 rf.
- **Stock-split detection.** It missed TSLA's 2020 split and invented a WFC split. Fixed by pinning splits to short-interest settlement data.
- **Training window before the data existed.** The live model trained on a window starting before the VIX data began. Fixed by moving to a calendar-based window, plus a data-coverage check that refuses to train on incomplete data.
- **An external code audit (October 2026)** found ten implementation defects in the backtester and research scripts. Examples:
  - swing positions sized with volatility from years later;
  - a close-of-day trade changing the same morning's position size;
  - training labels simulated against price bands that had not been computed yet;
  - each yearly run starting without its history.

  I reproduced every finding, fixed each with a regression test that fails on the old code, and reran everything affected without changing any strategy rule. Every pre-registered decision held. The headline book fell from 1.13 to 1.04 no-rf.
- **The news pipeline never worked.** Every LLM call was rejected because the output schema used range constraints the API does not support. Found it with a diagnostic probe and fixed it.

## 6. What changed live because of the research (October 2026)
- ETFs switched to the paper's exact rules, with a 6-month forward paper test. Pass rules: profitable after commissions, typical fill within 1 bp of the decision midpoint, and an early stop at the backtest's 5th-percentile drawdown.
- Stock mean reversion, gap trades and an untested first-30-minute strategy were switched off.
- The ML gate and its multi-hour startup training were switched off.
- Every change is behind a switch, so it can be undone without code changes.
- The result is three nearly uncorrelated sleeves: trend, ETF momentum on the paper's rules, and the stock swing book (section 4).

## 7. Honest limitations
- **Paper trading only.** IBKR declined futures permission, so the system uses ETFs instead.
- **Short backtest.** Minute data starts in late 2019, so most backtests cover 2020–2025: about six years with two crises.
- **Survivorship.** 56 of 625 historical S&P 500 members have no price data (mostly delisted).
- **The equity side is largely market exposure.** The swing book's design is an open question.
- **Weak recent year.** 2025 was weak for most components.
- **LLM labels may carry hindsight.** Claude was trained on data from after many of the releases it labelled. The prompt forbids outside knowledge, but that cannot be verified. Labels of explicitly worded facts ("we raised guidance") leave hindsight little room, and the trading test used 2020 onward.

## About this folder

- `TRIAL_REGISTRY.md`: every pre-registered trial, with its pass rule and result.
- `FAILURE_AUTOPSY.md`: the failures grouped by cause.
- `writeups/`: one report per study.
- `code/`: the research scripts behind them.

The scripts are published for reading. They depend on local price and filing caches, IBKR data
pulls and the private live engine, so they do not run on their own. The live engine, data and
trained models are not included.
