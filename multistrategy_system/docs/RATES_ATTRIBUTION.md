# Rates-bucket attribution for the trend sleeve (2026-09-11)

Research only. `rates_attribution.py` reads the same price caches as `trend_follow_v3.py` and
does not touch the live sleeve, `trend_follow_v2.py`, or any headline series. Outputs under
`results/rates_attribution_*.csv`.

**Question.** What do the Treasury ETFs (SHY, IEF, TLT) add to the cross-asset trend sleeve?

**Answer.** About a fifth of the return for an eighth of the risk, plus most of the crisis hedge.
Removing them lowers the sleeve's rf-Sharpe by 0.05, deepens its drawdown, cuts the 2022 year
from +30% to +22%, and drops the book's worst year from 0.43 to 0.06. As a standalone strategy
the bucket is worthless (rf-Sharpe 0.03); its value is that it is nearly uncorrelated with the
rest of the sleeve (+0.15) and negatively correlated with SPY (−0.27). No change to the live
universe.

## Conventions

Research baseline V1_trail (v2 signals, corrected lookback and rebalance stamps, trailing
504-day vol scalar capped at 3.0, 2 bp cost model, drawdown mark at initial capital). The full
window starts 2008-01-01 (all 16 signals live); the eval window is 2020–2025. rf-Sharpe (flat
5%/yr) is the headline; no-rf Sharpe is labelled where shown. Legacy `trend_follow_v2` rows are
printed by the script for reference; they tell the same story with slightly different levels.
The full-sleeve row reproduces the V1_trail line in `TREND_V3.md` exactly.

SHY prices in the cache start 2017-08, so the Rates bucket is IEF + TLT until 2018-08 and three
assets after. No borrow or financing cost is modelled on short TLT (project convention).

## View 1: in-run decomposition (the attribution)

One backtest of the full sleeve; each day's return is split into target weight × asset return
and summed by sector. The pieces add up to the sleeve's gross return (the residual, +0.22%/yr,
is the cost model).

Full window 2008–2026, gross of costs:

| Sector | Assets | Return contribution /yr | Share of return | Share of risk | Share of gross exposure | Avg net exposure | Corr with rest |
|---|---|---|---|---|---|---|---|
| Commodity | 5 | +2.7% | 37% | 34% | 20% | +0.03 | +0.28 |
| Equity | 3 | +1.6% | 22% | 24% | 13% | +0.13 | +0.31 |
| FX | 4 | +1.3% | 17% | 25% | 36% | +0.00 | +0.27 |
| **Rates** | 3 | **+1.4%** | **19%** | **12%** | 27% | **+0.23** | **+0.15** |
| Real estate | 1 | +0.4% | 5% | 5% | 4% | +0.06 | +0.21 |
| Total | 16 | +7.4% | 100% | 100% | 100% | +0.45 | |

Eval window 2020–2025: Rates +1.8%/yr, 17% of return, 13% of risk, 34% of gross exposure.

Return share divided by risk share is the cleanest "efficiency" reading: Rates 1.6, Commodity
1.1, Real estate 1.0, Equity 0.9, FX 0.7. FX carries the most gross exposure (four low-vol
currency ETFs, each vol-targeted up) and delivers the least per unit of risk.

Yearly Rates contribution (sum of daily contributions):

| | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rates | +3.0 | −2.2 | +1.0 | +5.3 | −0.1 | −0.1 | +2.2 | +1.0 | +1.4 | −0.7 | −2.8 | +9.1 | +5.4 | −1.0 | +7.6 | −1.4 | −0.9 | +1.2 | −1.4 |
| Sleeve | +11.6 | −3.3 | +6.1 | +7.9 | −10.3 | +15.8 | +11.2 | +10.0 | +0.4 | +10.0 | +0.2 | +6.5 | +12.9 | +8.0 | +27.4 | −6.9 | +8.1 | +13.3 | +6.9 |

Positive in all three crisis years, from opposite sides of the trade: long duration in 2008 and
2020, short duration in 2022. That is the trend argument for holding the bucket, and it is the
only sector with a negative SPY correlation (−0.27; rest of sleeve −0.06). The bucket loses
money in reversal years (2009, 2018, 2023) and is negative so far in 2026.

Inside the bucket (full window):

| Asset | Return contribution /yr | Share of risk | Avg abs weight | Avg net weight | Days active |
|---|---|---|---|---|---|
| SHY | +0.61% | 1.1% | 0.198 | +0.117 | 1,964 |
| IEF | +0.57% | 5.4% | 0.223 | +0.095 | 4,630 |
| TLT | +0.24% | 5.3% | 0.091 | +0.021 | 4,630 |

**SHY is a cash proxy, not a trend position.** Its vol is so low that the per-asset vol target
pins it at the 3.0× leverage cap (3.0 / 16 = 0.19, the observed average weight), it contributes
1% of the risk, and its return is mostly the T-bill yield it has earned since 2018. Under the
project's flat-5% rf convention that yield is charged back, so SHY adds to no-rf Sharpe and
almost nothing to rf-Sharpe. It is harmless, but it is one of sixteen 1/n slots spent on cash.
The bucket's directional content is IEF and TLT. The average net exposure of +0.23 shows the
sleeve was long duration most of the sample, which is the 2009–2020 bond bull market; the 2022
short is the evidence the signal also works on the other side.

## View 2: re-run counterfactuals (A/B/C)

The sleeve rebuilt with and without the bucket. B is a different portfolio (13 assets, 1/n and
the vol scalar re-set), so it answers "what if I had never held Treasuries", not "subtract the
contribution".

| Run | rf-Sharpe | no-rf | CAGR | Vol | MDD | Worst yr (rf-SR) | SPY corr | 2008 | 2020 | 2022 | Eval rf-SR | Eval MDD | Book rf-SR | Book worst yr | Book MDD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A full sleeve | 0.19 | 0.63 | 6.7% | 11.3% | −20.4% | −1.87 | −0.13 | +11% | +13% | +30% | 0.45 | −20.4% | 0.66 | 0.43 | −6.6% |
| B without Rates | 0.14 | 0.57 | 6.1% | 11.7% | −23.2% | −1.88 | −0.06 | +9% | +9% | +22% | 0.40 | −19.8% | 0.61 | 0.06 | −7.0% |
| C Rates alone | 0.03 | 0.43 | 4.7% | 12.3% | −25.7% | −2.24 | −0.27 | +15% | +17% | +30% | 0.10 | −20.5% | 0.45 | 0.02 | −6.5% |

A vs B on identical dates: rf-Sharpe +0.05, MDD 2.8 points shallower, correlation between A
and B 0.96. Correlation between B and C is +0.16, which is why a standalone 0.03 rf-Sharpe still
lifts the combination.

Book effect (risk-parity with the 2020–2025 equity sleeve, headline convention): rf-Sharpe 0.66
with Rates, 0.61 without; the worst-year book Sharpe falls from 0.43 to 0.06. That last number is
the one that matters for the book's stated goal of no losing years.

## Caveats

- Attribution is gross of the 2 bp cost model; costs are small and shown as the residual.
- No borrow or financing on short TLT. The 2022 short-duration gain would be a little smaller
  with a realistic financing charge; the sign does not change.
- The Rates sample is mostly a bond bull market. The bucket's long-side return is not a forecast;
  the 2022 short is the only large out-of-regime observation.
- 1/n with a cash-like asset at the leverage cap is a design wrinkle inherited from v2 and shared
  by the live sleeve. Replacing SHY (with a second duration point or a sixteenth diversifier) is a
  design change that would need its own trial; not done here.

## Registry

Diagnostic runs logged as T-2026-09-11-17 (B, sleeve without Rates) and T-2026-09-11-18 (C,
Rates alone) in `TRIAL_REGISTRY.md`. Neither is a promotion candidate; both were examined.
