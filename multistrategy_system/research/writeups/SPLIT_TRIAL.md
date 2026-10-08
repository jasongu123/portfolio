# ETF / single-stock split trial (2026-10-01, T-2026-10-01-59..71)

> **Corrected 2026-10-02** (AUDIT_RESPONSE_2026-10-01.md, results/split_trial_v2/): every decision is unchanged. E0 (paper rules, no HAR, corrected engine) 0.81, last 2y 0.39; departures: every minute −1.20, ML gate −0.38, band ×2 −0.43, VIX −0.19, RSI −0.32; as-is E6 −0.89. Stocks: S1 1.19 vs S0 1.08; S3 −1.11; S1 vs passive S4 1.19 vs 1.13 (needed 1.23), last 2y 1.27 vs 1.74. The tables below are the original run.

Research only: `split_trial.py` (pre-registered in its docstring before any run; variants in a
subclass, `ibkr_backtest.py` and `ibkr2.py` untouched). Results in `results/split_trial/`
(per arm-year pickles with position logs, summary.csv, report.txt). 72 runs, 34 minutes on the new Mac.
Each sleeve on its own $100k, 2020-01..2025-12-15, per-minute VIX, $0.0035/share (min $0.35) +
the stated slippage a side.

## ETF sleeve (SPY, QQQ, IWM; intraday only)

| Arm | Rules | Sharpe (no-rf) | rf | MaxDD | Last 2y | 2025 | Positions/yr | Gross bps | Net bps |
|---|---|---|---|---|---|---|---|---|---|
| E0p | paper rules, paper costs (0.03 bp) | 1.17 | 0.78 | −14% | 0.76 | 0.09 | 662 | 3.4 | 3.1 |
| **E0** | **paper rules, 0.5 bp a side** | **0.83** | 0.45 | −16% | 0.42 | −0.24 | 662 | 3.4 | 2.2 |
| E1 | E0 + decisions every minute | −0.47 | −0.83 | −44% | −0.40 | −1.25 | 2,887 | 0.8 | −0.4 |
| E2 | E0 + band ×2 | 0.32 | −0.22 | −16% | 0.41 | −0.56 | 299 | 2.4 | 1.2 |
| E3 | E0 + VIX ≥ 17 gate | 0.56 | 0.13 | −15% | 0.09 | −0.30 | 475 | 3.2 | 2.0 |
| E4 | E0 + ML gate | 0.22 | −0.52 | −7% | 0.07 | −0.29 | 205 | 2.1 | 0.9 |
| E5 | E0 + RSI size multiplier | 0.58 | 0.23 | −14% | 0.21 | −0.24 | 662 | 2.9 | 1.6 |
| E6 | the engine as is (all of the above, 1 bp) | −0.32 | −1.05 | −24% | 0.22 | −0.23 | 378 | 1.4 | −0.8 |

E0 net by symbol 2020–25: SPY +42.5k, QQQ +26.1k, IWM −6.1k. 2025 was negative for all but IWM.
Cost sensitivity (E0 rules): 1 bp a side → Sharpe 0.47 (last 2y 0.06); 2 bp → −0.24.
Daily correlation of E0 with the swing book −0.11, with the trend sleeve −0.02.

**E-A: PASS** (0.83 ≥ 0.50; last 2y +0.42; best 24 months 58% of profit). Every departure from
the paper lowers Sharpe; deciding every minute alone turns +0.83 into −0.47 (4× the trades,
gross edge 3.4 → 0.8 bps). The engine's ETF side never tested the paper; the paper itself works on
this data, but 2025 was negative and the edge needs execution near 0.5 bp a side.

## Stock sleeve (21 single stocks, 1 bp a side)

| Arm | Rules | Sharpe | rf | MaxDD | Last 2y | Last 15m | 2022 | 2025 |
|---|---|---|---|---|---|---|---|---|
| S0 | as is: mean reversion + swing + ML gate | 1.17 | 0.80 | −12% | 1.20 | 0.13 | 0.14 | 0.39 |
| **S1** | **swing only** | **1.26** | 0.88 | −15% | 1.34 | 0.21 | 0.27 | 0.37 |
| S2 | mean reversion only | −0.87 | −1.97 | −24% | −0.67 | −0.52 | −1.53 | −0.12 |
| S3 | paper momentum rule on stocks | −1.20 | −2.04 | −41% | −1.78 | −1.47 | −0.80 | −1.33 |
| S4 | passive: 21 names equal weight at 69.6% invested | 1.13 | 0.81 | −22% | 1.76 | 1.26 | −0.57 | 1.36 |

- **S-A: ACCEPT, drop mean reversion** (S1 1.26 ≥ S0 1.17; last 2y 1.34 ≥ 1.15).
- **S-B: FAIL** — the paper's momentum rule does not work on single stocks (−1.20).
- **S-C: NO EDGE** — swing beats passive over 2020–25 (1.26 vs 1.13 + 0.10) and in drawdown
  (−15% vs −22%), but not in the last 2 years (1.34 vs 1.76). As registered, the stock sleeve is a
  stock holding with stops: it gives up return in rising markets (last 15 months 0.21 vs 1.26)
  and saves it in falling ones (2022: 0.27 vs −0.57).

## What this implies (no live change made)

1. ETF intraday: rebuild on the paper's rules (30-minute decisions, band ×1, no VIX/ML/RSI) as a
   forward paper test, with realised execution cost measured against the 0.5 bp assumption.
2. Stock intraday: switch mean reversion off; live gap trades (never tested) off as well.
3. With both, the ML gate gates nothing: the paper rules have no gate and stocks have no intraday.
4. The swing book is the open design question: stop-loss timing is crash insurance paid for with
   lagging rallies; the trend sleeve may already provide that insurance.

## Live change (2026-10-01, approved by the user; vm_sync_2026-10-01.zip, not yet installed)

`ibkr2.py` switches (environment variables undo each): `ETF_PAPER_RULES=true` (decisions at
10:00..15:30 every 30 minutes with 5 minutes' grace, band ×1 without HAR, no VIX filter, no 0.3%
quick stop, no RSI/news size multipliers, size from 14-day daily-return vol, no ETF swing
entries), `ETF_INTRADAY_ENABLED=true` (forward-test stop switch), `STOCK_INTRADAY_ENABLED=false`
(mean reversion and gap entries), `EARLY_SESSION_ENABLED=false` (first-30-minute entries, never
backtested), `EQUITY_RF_GATE_ENABLED=false` (gate and startup training). Exits of open positions
run as before. Backup: `legacy_caches/code_backups/ibkr2_2026-10-01.py`. Tests:
`tests/test_restructure_2026_10.py`, `tests/test_etf_forward_report.py` (Mac 195 OK; bundle 192 OK).

Known live/backtest differences left in place: live cannot reverse long→short within one
decision (exit, then re-enter at the next half hour); the ETF end-of-day exit is at 15:50, not
the close; the swing book is still evaluated every minute live but at the close in the backtest.

Forward test T-2026-10-01-72 (`etf_forward_report.py`): 126 sessions; G1 net Sharpe > 0; G2
median slippage ≤ 1.0 bp against the decision midpoint; G3 early stop at E0's 5th-percentile
drawdown (21 sessions −5.0%, 63 −8.3%, 126 −13.1%). T-49 restarts on the install date.
