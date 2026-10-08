# Failure autopsy (2026-10-03)

Every failed trial in `TRIAL_REGISTRY.md` (T-01..T-95) is sorted by the most likely reason it
failed, using the evidence already recorded. The aim is to find causes shared by several failures
and test those causes directly, instead of re-tuning individual strategies on the same data.

## Failures grouped by cause

| Cause | Trials | Evidence already in hand |
|---|---|---|
| **1. Public, dated news is priced within a day** (large caps) | T-84/85 earnings moves, T-86/87 + T-95 index changes, T-93 guidance revisions | Event studies: the whole move happens on the reaction day or in the first minutes after the release (additions: 93% done by 17:30, biggest minute 17:16). Nothing drifts afterwards, and it did not drift in 2005–19 either (guidance −0.56, additions −0.81% after the announcement). |
| **2. Worked before, gone since about 2020** (published anomalies in large caps) | T-30..32 Taking Sides, T-33/34 seasonality, T-35..37 crowding (MOM, REV, IVOL, ISS, SI, SEAS), T-73..75 ML ranking, T-15..28 intramonth | Seasonality: t +2.34 on 2010–19 with the same code, then t −1.35. Intramonth: Sharpe 0.91 to 2011, 0.22 from 2012, dead everywhere since 2024. Non-earnings reversal: t −2.2 in 2005–19, −0.3 since. All six anomaly legs ≤ +0.16 since 2020. |
| **3. Costs and turnover** | T-61 every-minute decisions, T-32 short-interest change | Gross edge per trade 3.4 → 0.9 bp at 4× the trades; short-interest change roughly flat gross at 3.3 turnover a month. ETF sleeve Sharpe 0.83 at 0.5 bp, 0.47 at 1 bp. |
| **4. Never had an edge** | T-69 stock mean reversion (lost before costs), T-70 paper momentum on stocks, T-23 T+1 settlement | Negative gross, or the opposite of the mechanism. |
| **5. Wrong yardstick or unreliable labels** | T-94 vol sizing, T-88/90 distillation | HAR was a weaker benchmark than the live 60-day rule; two Claude models agreed only 63% on the guidance label. |
| **6. Construction: adding things diluted the book** | T-43/45 trend breadth, T-29 RFS sizing, T-40/41 equity trend gate | Additions earned half per slot; hedges cost more than they saved in 2020–25. |
| **7. Not enough evidence** | T-81 ridge vol (t 2.16 vs 2.24), T-11 short interest, T-22 intramonth (no holdout) | Effects in the right direction, short samples. |

Causes 1 and 2 cover most of the trials, and they point the same way. **Large US stocks since 2020
are efficient for public information and published patterns.** Causes 3–7 are specific to one
strategy each and are already understood.

## Hypotheses to test (each pre-registered before any run)

**H1. Less-watched stocks still have the effects large caps lost.**
- **Prediction:** the patterns that failed in the S&P 500 since 2020 still work in small and mid caps
  2020+, before and after realistic costs. The patterns are earnings drift, short-term reversal,
  momentum and seasonality.
- **Why:** less analyst coverage, fewer quant funds, and capacity too small for large funds.
- **Data needed:**
  - daily prices for S&P 400/600 or Russell 2000 members, from IBKR;
  - point-in-time membership, to avoid survivorship. This is the hard part: delisted small caps
    are where survivorship bias bites;
  - SEC 8-K earnings dates for those companies (same EDGAR method).
- **Risk:** higher costs in small caps can eat the effect. The test must use spreads that scale with
  size.

**H2. The decay is about crowding, not about the calendar.**
- **Prediction:** within the S&P 500, the failed patterns still work in the *least*-covered, least
  institutionally held names. They fail in the crowded ones.
- **Data:** existing data only (13F holdings from the crowding study).
- **Note:** T-37 already showed the mirror trade at +0.44, read after the fact. A blind test on
  2013–19 13F data, then 2020+, is the clean version. It is cheaper than H1 but has fewer names.

**H3. Costs, not signal, separate the near-misses.**
- **Prediction:** for failures with a positive gross edge, a slower version (lower turnover) keeps
  most of the gross edge and turns net-positive.
- **Data:** existing data only.
- **Note:** few failures had a positive gross edge since 2020, so this is likely to close quickly.
  It is still worth a table.

## Recommended order

1. **H1 data feasibility check** (no returns): can IBKR supply point-in-time small/mid-cap prices,
   including delisted names?
2. **H2**, as the cheap version of the same idea.
3. **H1 proper**, if the data exists.
4. **H3**, as a one-page table.
