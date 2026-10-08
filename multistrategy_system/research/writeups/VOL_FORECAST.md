# Volatility forecasting with ML, S&P 500 stocks (2026-10-01, T-2026-10-01-76..80)

Research only: `vol_forecast_trial.py` (pre-registered in its docstring before any forecast was
scored). Results in `results/vol_forecast/` (T-79) and `results/vol_forecast_holdout/` (T-80).

**Question.** Can ML forecast each stock's volatility over the next month better than HAR, the
standard model? Setup: point-in-time S&P 500 members, about 480 stocks a month, 19 price-based
features, monthly walk-forward refits on 36 months, out of sample 2021-01..2026-07 (67 months,
31,972 stock-months). Loss: QLIKE on variance (judges the forecast's level as well as its ranking).

| Model | QLIKE (T-79) | vs HAR | QLIKE (T-80, holdout calibration) | vs HAR | Rank corr | Realised vol, lowest / highest forecast decile |
|---|---|---|---|---|---|---|
| Random walk (last 21 days) | 0.350 | +31% | 0.345 | +30% | 0.57 | 20.6% / 45.3% |
| HAR (benchmark) | 0.268 | — | 0.265 | — | 0.64 | 19.6% / 47.9% |
| Ridge, 19 features | 0.246 | **−8.1%** (t −2.6) | 0.247 | **−6.8%** | 0.69 | 19.0% / 49.3% |
| LightGBM, 19 features | 0.279 | +4.2% | 0.274 | +3.1% | **0.71** | **18.8% / 50.3%** |

**Verdict as registered: FAIL, twice.**
- **T-79:** LightGBM vs HAR, QLIKE +4.2%, t −0.6, 3/6 years. Only V4 (last 24 months) passes.
- **T-80** (holdout calibration, registered after T-79's diagnosis and reusing the same months):
  QLIKE +3.1%, t −0.5; V3 (4/6 years) and V4 pass, V1 and V2 fail.

**Reading.**
1. **LightGBM ranks stocks by future volatility better than every other model** (rank
   correlation 0.71 vs HAR 0.64; widest gap between its calmest and wildest deciles). It fails on
   the *level* of its forecasts, mostly in 2021–22. Trees cannot extrapolate beyond the range they
   were trained on, and the 2018–20 training windows contained the COVID spike. From 2023 it beats
   HAR in every year.
2. **The richer features are what help.** The linear model on the same 19 features beat HAR by
   7–8% QLIKE (t −2.6) and in 5 of 6 years. The main drivers are 6-month vol, market-wide vol and
   downside vol. This was a diagnostic, not the registered candidate, so it is not claimed as a pass.
3. **Calibration matters as much as ranking.** The in-sample residual variance flatters a flexible
   model's level correction (LightGBM forecasts sat near the median of realised variance, mean
   log(F/RV²) +0.03 against HAR's +0.26). Measuring it out of sample helped but did not close the gap.

**Next, if pursued:** a fresh pre-registered test of the ridge model (and LightGBM on
level-robust terms) on data not yet used, e.g. the 46-ETF, 20-year panel in
`trend_breadth_prices.csv`, where a pass would also feed the ETF and trend position sizing.

## Fresh-data test on ETFs (T-2026-10-01-81, -82)

`vol_forecast_etf.py`, pre-registered before scoring: 48 cross-asset ETFs (the trend sleeve's 16,
SPY and the 30 breadth ETFs), 2006–2026, never used for volatility forecasting. Expanding window,
holdout calibration fixed in advance for every model, out of sample 2012-01..2026-07 (175 months,
8,220 ETF-months). Both candidates were tested against HAR, with the t threshold raised to 2.24
(Bonferroni for two candidates).

| Model | QLIKE | vs HAR | t | Years better | Last 24 months | Rank corr | Verdict |
|---|---|---|---|---|---|---|---|
| Random walk | 0.322 | +4.6% | | | | 0.896 | reference |
| HAR | 0.308 | — | | | 0.301 | 0.908 | benchmark |
| **Ridge, 19 features** | 0.277 | **−10.1%** | **2.16** | **12/15** | 0.287 | 0.914 | **FAIL, on V2 only** (needs 2.24) |
| LightGBM, 19 features | 0.274 | −11.2% | 0.92 | 7/15 | 0.289 | 0.915 | FAIL (V2, V3) |

**Reading.**
1. **The ridge model replicated.** Fresh data, a different asset class, the same direction and
   size: −10.1% on ETFs, against −7% to −8% on stocks. It was better in 12 of 15 years and over
   the last 24 months. It missed only the significance bar I raised for multiple testing
   (t 2.16 against 2.24; it would pass the conventional 2.0). As registered, it is a FAIL.
2. **LightGBM's average is carried by one year.** In 2020 its QLIKE was 0.96 against HAR's 1.58,
   and that one year drives its −11%. It was better in only 7 of 15 years, with t 0.9: not reliable.
3. **What generalises is the information, not the nonlinearity.** Longer-horizon vol,
   market-wide vol and downside vol add to HAR's three horizons in both datasets. A linear model
   uses them as well as a boosted tree does, and more steadily.

**Status.** Ridge is the strongest result of this research line: consistent across two
independent datasets, but not significant at the pre-set bar. Further confirmation can only come
from new data (live forward use) or a sizing trial with its own pre-registered economic test.

## Using the forecast to size the trend sleeve (T-2026-10-03-94): FAIL

**Setup.** The live trend sleeve (16 ETFs) was rerun with one change: each ETF's size used the ridge
forecast of next month's volatility instead of the 60-day trailing volatility. Window 2012-02 to
2026-08. `vol_sizing_trial.py`, pre-registered before any run. The baseline reproduces the cached
V1_trail returns exactly.

| Sizing | Sharpe (no-rf) | Last 24 months | Max drawdown | 2020 | Vol error | Monthly portfolio-vol spread |
|---|---|---|---|---|---|---|
| 60-day trailing (live) | 0.68 | 0.88 | −20.4% | +12.6% | 0.117 | 0.047 |
| Ridge forecast | 0.62 | 0.77 | −21.4% | +7.1% | 0.116 | 0.042 |
| HAR | 0.66 | 0.85 | −20.9% | +10.1% | 0.128 | 0.043 |
| LightGBM | 0.64 | 0.80 | −21.4% | +5.9% | 0.138 | 0.041 |

All four rules failed. The volatility error was only 0.7% lower (t 0.11).

**Why.**
1. **The yardstick was wrong.**
   - T-81's ridge beat HAR by 10%.
   - On these ETFs, though, HAR is *worse* than the live 60-day window (error 0.128 vs 0.117).
   - Beating a weak benchmark did not mean beating the rule already in use.
2. **For a trend follower, faster de-risking is costly.**
   - The forecasts react faster to volatility spikes, so they cut size just as the crisis trends
     start to pay. In 2020 the sleeve earned +7.1% instead of +12.6%; LightGBM, the fastest-reacting,
     earned +5.9%.
   - The more responsive the forecast, the lower the Sharpe.
3. **The only gain was steadier risk.** Month-to-month portfolio volatility varied about 10% less.
   That is not worth 0.06 of Sharpe.

**Lessons.**
- Test a forecast against the rule actually in use, not only a textbook benchmark.
- Better volatility forecasts are not automatically better sizing. What matters is how the sizing
  interacts with when the strategy earns its money.
