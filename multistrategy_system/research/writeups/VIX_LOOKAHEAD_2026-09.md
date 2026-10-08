# Same-day VIX look-ahead in the backtester, corrected (2026-09-24)

`ibkr_backtest.py` read the trade date's own VIX close — printed at 16:15 ET, after every
intraday decision that day — in five places: the ETF entry gate (`vix_min = 17`, part of the
validated FULL-12 gated hybrid) and everything keyed off it that day (HMM action, danger zone,
swing entries and exits), the VIX-adaptive parameters, and the ML features `vix_level`,
`vix_term_slope` and `vix3m_ratio`. The headline equity sleeve and book were built on it.

This is a correction, not a candidate: no adoption gate. It was registered before any rerun that
the corrected series replaces the headline whatever the direction; and, before any intraday
result existed, that the per-minute series (below) is the headline because it is how the live
system trades.

## The fix, in two steps

| Gate mode | What the ETF gate sees | Status |
|---|---|---|
| same-day (old) | the day's 16:15 close, at 09:45 | look-ahead; withdrawn |
| prior close | the previous day's close, all day | causal; blind to moves during the day; conservative bound |
| **per-minute** | **the VIX at each decision minute** (`backtest_data/VIX_1m.csv`) | **causal; replicates live** (`ibkr2.py` calls `get_current_vix()` at every signal); **headline** |

The ML features use the prior day's close in every mode: identical to `ibkr2.py`'s
`_rf_vix_features` (live and training), verified by a parity test. The per-minute gate reads the
1-min VIX bar that closes at the same instant as the equity bar being decided on; the position
changes on the next bar. A blocked new entry or reversal goes flat; holds and exits are never
blocked. Swing decisions (close of day) read the 15:59 VIX bar.

Checks: the old code rerun today reproduces the 2026-09-15 caches bit for bit (2024, 251 days,
max difference 0.0), so every change below is the VIX fix and nothing else; the edited code in
prior-close mode reproduces its own earlier prior-close run bit for bit. VIX 1-min data: 616,653
bars, every session in the daily file (none missing, no duplicates), last bar matching the daily
close (median difference 0.00); fetched from IBKR with `fetch_vix_intraday.py`.

## Equity sleeve, 2020–2025 (FULL-12 gated hybrid, `combine_book` config unchanged)

rf = flat 5%/yr; MaxDD from initial capital.

| Gate | rf-Sharpe | no-rf | worst year (rf) | MaxDD | vol | CAGR | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same-day (old headline) | 0.44 | 0.91 | −1.28 (2022) | −14.9% | 10.4% | +9.4% | +0.40 | +1.43 | −1.28 | +1.39 | +0.71 | +0.47 |
| prior close | 0.15 | 0.62 | −1.49 (2022) | −17.8% | 10.6% | +6.2% | +0.21 | +1.13 | −1.49 | +1.21 | +0.38 | −0.20 |
| **per-minute (headline)** | **0.40** | **0.88** | **−1.43 (2022)** | **−17.1%** | **10.4%** | **+9.0%** | +0.46 | +1.36 | −1.43 | +1.31 | +0.68 | +0.54 |

Annual return, same-day → prior close → per-minute: 2020 +10.4 → +7.3 → +11.4%; 2021 +17.7 →
+15.0 → +16.9%; 2022 −9.3 → −11.5 → −11.0%; 2023 +20.0 → +17.9 → +19.0%; 2024 +10.9 → +8.1 →
+10.6%; 2025 +8.6 → +2.8 → +9.0%.

## Headline book, 2020–2025 (walk-forward inverse-vol, equity + trend as live)

The trend sleeve (rf 0.42 / no-rf 0.83) does not use VIX and is unchanged.

| Book | rf-Sharpe | no-rf | worst year (rf) | MaxDD | vol | CAGR | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same-day (old headline) | 0.54 | 1.16 | −0.02 (2022) | −7.4% | 8.1% | +9.5% | +0.79 | +0.78 | −0.02 | +0.21 | +0.68 | +0.76 |
| prior close | 0.34 | 0.96 | −0.17 (2022) | −8.5% | 8.1% | +7.8% | +0.66 | +0.62 | −0.17 | +0.05 | +0.48 | +0.30 |
| **per-minute (headline)** | **0.51** | **1.13** | **−0.16 (2022)** | **−7.3%** | **8.1%** | **+9.2%** | +0.82 | +0.74 | −0.16 | +0.14 | +0.67 | +0.79 |

Rigor (no-rf daily Sharpe): book PSR(SR > 0) 99.7%, PSR(SR > 0.5) 93.4% (was 94.4%); equity
sleeve 98.3% / 81.8% (was 98.6% / 84.0%). Deflated Sharpe of the book, trial dispersion 0.61:
67.9% at N = 9 (was 70.1%), 15.2% at N = 100 (was 16.4%), 26.5% at N = 48 (the registry has 48
rows now; the old 30.9% was at N = 42).

## Mechanism

- **The gate carried the look-ahead, not the model.** 2024 attribution (causal gate, ML features
  left on the same-day close): 0.71 → 0.18 rf. Making the ML features causal as well moves it to
  0.38; the two steps are not additive and one year is thin evidence for the size of the second.
- **Why same-day flattered:** the gate flips on only 2–29 days a year, but about half are days VIX
  rose through 17 during the session — often strong trend days, when ETF momentum pays. Knowing
  the close meant trading exactly those days from 09:45.
- **Why the prior close was too harsh:** on 30–51 days a year (2021, 2023–2025) VIX crosses 17
  during the session. A live gate trades the part of the day after the cross; the prior close
  shut the whole day. Example, 18 Dec 2024 (Fed day): prior close 15.87; VIX 15.3 at 10:30, 16.7
  at 14:30, 18.4 at 15:00, 24.1 at 15:59, 27.6 at 16:15 — the old backtest traded all day, prior
  close not at all, per-minute from 14:46 (VIX 17.25).
- **Net:** a real-time gate recovers most of what the look-ahead appeared to earn. The per-minute
  sleeve is below the old one in 2021–2024 and above it in 2020 and 2025; the loss concentrates
  in 2022 (worst year −1.28 → −1.43) and in the drawdown (−14.9% → −17.1%).

## What changes in the claims

- Headline book: **rf 0.51 / no-rf 1.13**, worst year −0.16 (2022), MaxDD −7.3%, positive in 5 of
  6 years (replaces 0.54 / 1.16 / −0.02 / −7.4%).
- Equity sleeve: **rf 0.40 / no-rf 0.88** (replaces 0.44 / 0.91); MaxDD −17.1% (replaces −14.9%).
- 2025 equity: **+0.54 rf** (replaces +0.47). The intraday edge is weaker than 2021–2023 but not
  gone; the prior-close −0.20 was an artefact of the blunt gate.

## Not rerun

Other results built on the same engine used the same-day gate or features: the equity-breadth
book (T-2026-09-16-47/48: all stocks, so the ETF gate does not apply, but the ML VIX features
did), equity_trend_gate, feature_sweep. They have not been re-verified.

## Files

- `ibkr_backtest.py`: `_prior_vix`, `_prior_daily_close`, `vix_gate_mode` ('intraday' default,
  prior close where no 1-min VIX), `_vix_at`, `_etf_vix_gate_open`.
- `fetch_vix_intraday.py`: 1-min VIX from TWS (read-only, client 96) into `backtest_data/VIX_1m.csv`.
- `vix_causal_rerun.py`: the reruns (`--gate intraday`); outputs in `results/vixcausal/` and
  `results/vixintraday/`. `equity_returns_{2020..2025}.csv` now hold the per-minute series; the
  same-day caches are kept as `legacy_caches/equity_returns_sameday_vix_{year}.csv`, the old book
  series as `results/headline_2026-09_daily_sameday_vix.csv`.
- `tests/test_backtest_vix_causal.py`: 9 tests, including a source scan for any same-day VIX read
  and ML-feature parity with `ibkr2.py`.
