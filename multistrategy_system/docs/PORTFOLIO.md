# Systematic Multi-Sleeve Trading Research

A retail quant research program built to an institutional standard: rigorous validation,
honest failure analysis, realistic cost modeling, and live (paper) risk controls. The
deliverable is not a magic strategy — it's **demonstrated research judgment**: a small,
robust, every-year-positive book plus a documented log of strategies *killed* with named
mechanisms.

> **Thesis:** Most retail "strategies" are overfit backtests their authors believe. This
> project's value is the opposite discipline — killing my own ideas when the evidence
> says to, refusing to claim wins the data can't support, and knowing *why* each failure
> failed. Five of six candidate sleeves were rejected. That hit rate is the point.

---

## 1. The validated book

**Configuration:** an intraday equity sleeve (Zarattini noise-area momentum + 60-min
mean-reversion, ML-gated) combined with a **cross-asset trend sleeve** (time-series
momentum, 18 ETFs, 3/6/12-month ensemble) at risk-parity.

**Validated over 6 years (2020–2025) including two crises — COVID-2020 and the 2022 bear:**

| 2020–2025 | rf-Sharpe | worst year | MaxDD | ann. vol |
|---|---|---|---|---|
| Equity sleeve alone | 0.54 | −1.28 (2022) | −14.9% | 11.7% |
| Trend sleeve alone (v2) | 0.33 | −1.90 (2023) | −22.1% | 12.7% |
| **Combined book (risk-parity)** | **0.60** | **−0.07 (2022)** | **−7.4%** | **8.8%** |

**The combined book beats both of its components on every risk metric** — higher Sharpe,
a worst year of essentially flat (−0.07) versus deeply negative for either sleeve alone,
and half the drawdown. Positive in 5 of 6 years; the one non-positive year (2022, the
worst stock+bond bear in decades) was a rounding error from zero. That is the
diversification thesis, quantified: the *book* is the product, not either strategy.

Per-year rf-Sharpe (\* = crisis year):

| | 2020\* | 2021 | 2022\* | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| equity | 0.40 | 1.44 | **−1.28** | 1.36 | 1.28 | 0.14 |
| trend v2 | 0.56 | 0.25 | **+0.83** | −1.90 | 0.60 | 0.59 |
| **combined** | 0.68 | 0.91 | **−0.07** | 0.39 | 1.13 | 0.47 |

The 2022 row is the thesis working: the equity book bleeds in a grinding bear, trend
hedges it (+0.83), and the book survives flat. 2023 is the same complementarity in
reverse (trend's whipsaw year, carried by equity).

**Sharpe convention is stated everywhere.** Two are used: `no-rf` (`mean/std·√252`) and
`rf` (subtracts a 5%/yr risk-free, ddof=0). They differ by ~2–3× when returns are small;
conflating them once cost real analysis time, so every figure is labeled.

---

## 2. Research methodology (the gate)

Every candidate sleeve had to clear the same bar before it could join the book:

1. **In-sample formation / out-of-sample trading** — parameters and pair/feature selection
   on a training window; performance measured only on held-out data.
2. **Positive Sharpe across regimes** — not one good backtest, but consistency across years
   *including* crisis years (2008 / 2020 / 2022).
3. **Survives realistic cost** — slippage modeled as bps-of-notional, stress-tested 0.1→5bp.
4. **Low correlation** to the existing book (a sleeve that just re-expresses equity beta adds
   nothing).

This is deliberately strict. It is *why* most candidates failed — which is the intended
outcome of a real research process.

---

## 3. Strategy scoreboard — 6 tested, 1 kept, 5 killed

| Sleeve | Verdict | Named mechanism |
|---|---|---|
| **Cross-asset trend** | ✅ **kept** | low-turnover, +crisis alpha in slow bears, −0.13 SPY corr |
| Overnight drift | ❌ | levered beta, cost-fragile |
| Short-term reversal | ❌ | money-loser out-of-sample (rf −0.82) |
| Betting-against-beta | ❌ | marginal / unstable (rf −0.33) |
| Stat-arb (cointegration pairs) | ❌ | **de-cointegration** — AMD~MSFT/QQQ spread ran to 5.2σ (−33%); z-stops made it *worse*; structural |
| **Carry (cross-asset)** | ❌ | **short-volatility** — see §4 |

Each rejection has a mechanism you can name and defend in an interview. "I added carry and
Sharpe went up" is a worse answer than "I tested carry, killed it with P&L attribution, and
can tell you exactly why it doesn't diversify a long-risk book."

The INTC/AMD/AVGO/QCOM/MU technology tilt is a **new, untested candidate**, not
part of this scoreboard or the validated book. Its live-paper wiring is disabled
until `tech_sleeve_research.py` clears the same gate above.

The **RFS newsy-month equity timing rule** is another standalone candidate, now implemented
in `rfs_newsy_month.py`. Its author-data reproduction matches the paper's roughly 0.44
Sharpe, but the shorter local ETF-era evidence is mixed; it is not in the scoreboard or
live book until a frozen walk-forward and book-correlation gate is complete. See
`RFS_NEWSY_MONTH.md`.

---

## 4. Key findings

**Short-vol vs long-vol is the organizing principle.** Carry — the classic CTA "second leg"
— *failed by construction*. P&L attribution showed it shorts SPY/GLD (lowest *income* yield
but highest *total* return) and goes long credit/duration that crashes together. Every fix
that made it profitable did so by becoming long-credit beta (SPY corr +0.6); the only fix
that lowered correlation killed the return. **No variant was both profitable and uncorrelated.**
Conclusion: carry is a short-vol premium — it pays you to take crash risk, the *mirror image*
of trend. For a long-risk book, only **long-vol / crisis-alpha** sleeves diversify.

**Trend hedges *slow* crises, not *fast* ones — verified empirically.** The April 2025 tariff
shock (S&P −19%, VIX 17→52) sits inside the data. During it: equity book −5.1%, **trend v1
−15%** (caught long into a V-shaped shock that snapped back before a monthly rebalance could
re-short), trend v2 −6%. So the book's resilience came from the *equity* sleeve's intraday
de-risking, not from trend. This produced a sharper, more honest thesis than "trend = crisis
hedge":

| Crisis type | What hedges it | Verified |
|---|---|---|
| Fast shock (Apr 2025: SPY −19%, VIX 52) | equity sleeve de-risks intraday (−5.1%) | ✅ in-sample event study |
| Slow grinding bear (2022) | trend sleeve (+0.83 Sharpe while equity −1.28) | ✅ **book-level, 2022 backtest** |
| COVID crash+recovery (2020) | both sleeves positive; combined 0.68 | ✅ book-level, 2020 backtest |
| Catastrophic overnight gap | *(open — a long-vol tail hedge would fill this)* | — |

The crisis map is now closed on live data for both tested regimes: **each sleeve carries
the crisis type the other can't.**

**Cost-fragility is the real killer of the intraday edge.** A dead-code bug was modeling
slippage as a flat $0.001/share (~0.02bp on a $500 stock); fixed to bps-of-notional, the
2025 edge decays from rf 0.37 (0.1bp) → 0.14 (1bp) → −0.12 (2bp). High-turnover equity alpha
dies at 1–2bp. The trend sleeve survives because it's monthly / low-turnover — which is also
why higher-frequency strategies are the *wrong* direction for a retail trader.

**The deepening lesson (anti-overfitting) — posed, then resolved by data.** Widening trend
to 18 assets + a 3/6/12-month ensemble clearly improved the *standalone* sleeve (Sharpe
0.47→0.57, MaxDD −29.9%→−22.7%). But on the original 3-year (crisis-free, bull) window,
the *simpler* v1 made the better book — so the choice was **deliberately left open** rather
than decided on an uninformative sample. Extending the equity book to 2020 settled it:
over 6 years including two crises, **equity+v2 beats equity+v1 on every metric** (Sharpe
0.60 vs 0.51, worst year −0.07 vs −0.52, MaxDD −7.4% vs −10.0%) — v2's crisis alpha, which
was invisible in the bull window, is exactly what shows up in 2020/2022. The methodological
point survived contact with the data: *a crisis hedge cannot be evaluated on a crisis-free
window.*

---

## 5. Risk management (live, paper)

- **`validate_order`** enforces a position-size cap (50% of NetLiq) and a daily-loss circuit
  breaker, wired into the automated entry path (entries only — exits are never blocked).
- **Sleeve isolation:** the live trend sleeve runs on its own IBKR connection, a disjoint
  symbol universe (no shared positions with the equity book), and a background thread, so it
  cannot destabilize the intraday engine. Its live sizing now applies the research-calibrated
  `2.95` portfolio scalar before a `$120k` per-name cap and `$600k` gross cap; the scalar is
  full-history rather than walk-forward, and the gross cap binds in the latest cached snapshot.
- **Technology candidate:** exact tickers are disjoint from core equity, entries are bounded
  at 2% of NetLiq per name and 10% estimated gross, and cost records use `sleeve='tech'`.
  It shares the serialized equity connection and remains off by default and in real-money mode.
- **Earnings candidate:** a disabled-by-default, entry-only embargo now sits in the shared
  core/tech execution path. The accompanying parser and replay journal are shadow-only and
  cannot place an earnings order; see `EARNINGS_LAYER.md`.

---

## 6. Statistical rigor (deflated_sharpe.py)

The book is stress-tested against the two questions a quant audience always asks —
short-sample reliability and multiple-testing selection:

- **Probabilistic Sharpe Ratio** (Bailey–López de Prado, adjusts for skew/kurtosis; n=1,477
  days ≈ 5.9y): combined book **P(true Sharpe > 0) = 99.7%**, **P(true Sharpe > 0.5) = 94.4%**
  (up from 97.9% / 88.4% on the original 3-year sample — doubling the history *and* adding
  two crises tightened, not weakened, the claim; the no-rf SR held at 1.17 vs 1.21).
- **Recorded Deflated Sharpe sensitivity** (haircut for ~6–24 sleeve trials):
  **88% / 77% / 63%** at N=6/12/24. These are prior-version, provisional figures: the
  combined book used no-rf Sharpe while the trial inputs appear rf-adjusted, so the entire
  calculation must be rerun under one convention and an auditable effective-trial count.
- **Tail evidence of diversification:** the combined book's kurtosis (5.5) sits well below
  the equity sleeve's (7.3), and its 6-year MaxDD (−7.4%) is half the equity sleeve's
  (−14.9%) — the diversification shows up in the moments and the tails, not just the Sharpe.

The point is not that these are high — it's that uncertainty and trial-count sensitivity are
stated. None reaches a conventional 95% confidence threshold, and DSR is not the probability
of future profit. Treat the current range as diagnostic until the convention-consistent rerun.

## 7. Honest limitations

- The intraday momentum edge is **decaying / crowded** (weak 2025; cost-fragile).
- The 2020–21 backtest years run on **thinner ML-training history** (~5 months of 2019
  warmup vs ~4 years for later years) — those years' gate behavior understates the live model.
- The trend sleeve's design (universe, 3/6/12m horizons) was chosen with knowledge of the
  full historical sample — a mild selection effect, mitigated but not eliminated by the
  deflated-Sharpe haircut; risk-parity weights are computed full-sample (no walk-forward
  weighting yet), and the live `2.95` portfolio scalar is also full-history calibrated.
- Standalone book Sharpe is **modest (~0.9 rf)**, not the mythical 2+. That is the realistic
  ceiling for a retail systematic trader on liquid equities, and saying so *is* the rigor.
- The news-sentiment filter is an **unvalidated overlay**, not a systematic edge (kept as an
  optional risk overlay, excluded from the alpha claims).
- The technology tilt is also **unvalidated**. Its presence in the code is research plumbing,
  not evidence that Intel or a semiconductor basket improves the portfolio.
- The time-aware earnings layer is **unvalidated**. The earlier broad event filter failed to
  improve the book, current legacy dates stop in 2025, and historical point-in-time revenue,
  guidance, and call data are not yet populated. Both earnings flags remain off.
- The RFS newsy-month candidate is **unvalidated**. The current SPY-only test has a short
  2007–2026 history, uses the paper's ex-post volatility convention, and shows weak
  full-period performance despite a stronger post-June-2021 slice. No live sizing or order
  path exists for it.
- The shared execution-cost model and swing cash-accounting correction were added after
  the cached headline return series above was produced. Those numbers remain the recorded
  prior-version result; regenerate all equity/book caches before presenting them as results
  of the new cost engine.
- The recorded DSR mixes Sharpe conventions and uses a judgmental 6/12/24 effective-trial
  range; it is not publication-grade until recomputed consistently from a trial registry.

---

## 7. Repository guide

| Area | Files |
|---|---|
| Live system | `ibkr2.py` (equity engine + risk controls + integrated sleeves), `trend_sleeve_live.py`, `tech_sleeve_live.py`, `check_trend_positions.py` |
| Backtest engine | `ibkr_backtest.py` (`StrategyBacktester`) |
| Trend research | `trend_follow.py`, `trend_follow_v2.py`, `pull_trend_ibkr.py` |
| Technology candidate | `tech_sleeve_research.py`, `tech_sleeve_live.py`; see `TECH_SLEEVE.md` |
| Earnings candidate | `earnings_events.py`, `earnings_parser.py`, `earnings_history_extract.py`, `earnings_replay.py`; see `EARNINGS_LAYER.md` |
| RFS newsy-month candidate | `rfs_newsy_month.py`, `tests/test_rfs_newsy_month.py`; see `RFS_NEWSY_MONTH.md` |
| Carry (killed) | `carry_sleeve.py`, `carry_diag.py`, `pull_carry_ibkr.py` |
| Other sleeves (killed) | `stat_arb.py`, `stat_arb_diag.py`, `sleeve_reversal.py`, `sleeve_lowvol.py`, `overnight_drift.py` |
| Combination / cost | `trading_costs.py`, `execution_journal.py`, `execution_cost_report.py`, `cost_stress.py`, `combine_book.py`, `combine_book_v2.py`, `feature_sweep.py`; see `COST_MODEL.md` |

---

## 8. Roadmap

1. ✅ **2020–22 extension done** (IBKR 1-min pulled back to 2019; `extend_history.py`) —
   crisis map closed at book level; v2 confirmed as the book's trend engine.
2. **Rigor layer:** ✅ Probabilistic + Deflated Sharpe done (`deflated_sharpe.py`).
   Remaining: walk-forward analysis.
3. **Paper-trade the integrated book forward** — true out-of-sample; validates the live risk wiring.
4. **RFS newsy-month candidate gate** — freeze the aggregate-market specification, replace
   ex-post scaling with trailing-only sizing, and test walk-forward plus book correlation.
5. **Technology candidate gate** — acquire disjoint IBKR data, run cost-aware standalone and
   combined-book tests, then either reject it or enable a small paper allocation.
6. *(Optional, deliberate)* a **long-vol tail hedge** to fill the catastrophic-gap quadrant,
   or a **crypto microstructure** research piece — both new, data-gated frontiers.
