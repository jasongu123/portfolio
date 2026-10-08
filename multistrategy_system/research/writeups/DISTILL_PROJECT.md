# Distilling Claude into a small earnings-release model (started 2026-10-02)

**Goal.** Build a model of my own that reads company earnings press releases nearly as well as a
large language model, then test whether what it reads predicts stock returns. The plan was
pre-registered in `distill_label.py` before any label, model or return existed. Trials:
T-2026-10-02-88 (student quality) and -89 (student signal).

**Data (free).**
- Every 8-K with item 2.02 filed by S&P 500 companies (EDGAR, 2004–2026), with its press-release
  exhibit. Built with `earnings_text.py`.
- Narrative text only: HTML tables and number-heavy lines are dropped.

**Stage A: teacher** (`distill_label.py`).
- Claude Haiku 4.5 labels 2,500 releases, stratified by year 2005–2026.
- Labels: tone (5 levels), guidance (raised / maintained / lowered / initiated / withdrawn / mixed
  / none), comparison with expectations (beat / in line / miss / not stated), hedging (low /
  medium / high).
- Message Batches API at half price, temperature 0, structured output (enum-only schema; no
  numeric constraints, the lesson from the news-engine bug).
- The user runs it with their own key; the key never touches the project files.

**Stage B: student** (`distill_student.py`).
- FinBERT (BERT-base pre-trained on financial text) with four classification heads, fine-tuned on
  the teacher's labels on the Mac's GPU.
- Splits: 1,600 train, 400 validation, 500 held out.
- The quality gate is agreement with the teacher on the held-out 500, and each field must beat
  the most-common-label baseline.

**Stage C: finance.**
- The student labels every release.
- A long-short trade on its score after earnings, held 20 days and market-hedged, is compared
  with the classic Loughran–McDonald dictionary tone.
- Gate: Sharpe ≥ 0.5 after costs, date-clustered t ≥ 2, profitable over the last 24 months, and at
  least 0.10 Sharpe above the dictionary baseline.

**Stage A result.**
- 2,467 releases labelled by Claude Haiku 4.5 for **$3.94** (7.49M input / 79k output tokens, batch).
- Seven chunks of ≤400 requests, after a single 30 MB upload failed with an SSL error.
- None failed or invalid.

**Stage B result: FAIL as registered** (held-out 467 releases, student vs teacher):

| Field | Agreement | Gate | Most-common-label baseline | Verdict |
|---|---|---|---|---|
| Tone (exact) | 73.0% | ≥60% | 65.3% | pass |
| Tone (within one level) | 94.9% | ≥90% | — | pass |
| Compared with expectations | 85.0% | ≥70% | 73.4% | pass |
| Guidance | 58.0% | ≥75% | 24.6% | **fail** (far above baseline) |
| Hedging | 74.3% | ≥60% | 74.9% | **fail** (no better than always "medium") |

**Likely cause: the student reads much less than the teacher.**
- FinBERT takes 512 tokens, about the first 2,000 characters. The teacher read 12,000.
- In 44% of releases the first mention of outlook or guidance comes after character 2,000, and
  the actual guidance section is often later still.
- Hedging language is spread through the whole release.
- The two fields the student can see early (headline tone, "beat/miss" statements) pass.
- Guidance also has rare classes (withdrawn 0.6%, mixed 1.4%) with too few examples to learn.

**Stage C: on hold** (no signal test on a student that failed its gate).

## Version 2 (T-2026-10-02-90): FAIL as registered, on guidance only

**Changes from v1.**
- The student reads the teacher's full 12,000-character input, in up to 8 chunks of 512 tokens.
  The chunks are pooled (mean + max).
- It trains on 3,955 Haiku labels (the 2,467 from v1 plus 1,500 new).
- It is tested on a **fresh** 500-release set (seed 44) that nothing had touched before.
- Sonnet 5.5 labelled the same 500 releases to measure how far the two teachers agree.

**Results** (fresh test, 495 releases, student vs Haiku):

| Field | v1 | v2 | Gate | Most-common-label baseline | Verdict |
|---|---|---|---|---|---|
| Tone (exact) | 73.0% | 72.1% | ≥60% | 61.2% | pass |
| Tone (within one level) | 94.9% | 93.5% | ≥90% | — | pass |
| Compared with expectations | 85.0% | 90.1% | ≥70% | 74.1% | pass |
| Guidance | 58.0% | **66.5%** | ≥75% | 28.3% | **fail** |
| Hedging | 74.3% | 79.0% | ≥60% | 71.7% | pass |

**Teacher agreement** (the same 495 releases; not part of the decision):

| Field | Sonnet 5.5 vs Haiku | Student vs Haiku | Student vs Sonnet 5.5 |
|---|---|---|---|
| Tone | 77.0% | 72.1% | 63.6% |
| Guidance | 63.4% | 66.5% | 49.5% |
| Compared with expectations | 90.3% | 90.1% | 86.7% |
| Hedging | 47.7% | 79.0% | 43.0% |

**Reading it.**
- **Reading the whole release helped.** Guidance rose 8.5 points and hedging moved from no better
  than the baseline to 7 points above it.
- **The guidance gate is above what the label can support.** The student agrees with Haiku more
  often than Sonnet 5.5 agrees with Haiku (66.5% vs 63.4%). Two strong models can't agree on
  "guidance" 75% of the time, because "maintained vs raised vs initiated" usually needs the
  previous quarter's guidance, which a single release doesn't contain. The main student errors
  are exactly those: maintained→raised (34), initiated→raised (17), none→maintained (22).
- **Hedging passes, but it measures Haiku's style, not a shared concept.** The two teachers agree
  on hedging less than half the time, so the student has learned how Haiku labels hedging.
- **The tone and compared-with-expectations labels are solid.** The teachers agree 77% and 90%,
  and the student is close to that.
- The gate stays as registered: **FAIL**. Lowering it after seeing the result is not allowed.

**Stage C (T-89) stays on hold.** A v3 would need new, pre-registered labels that one document can
actually support. Examples: guidance given yes/no; explicit raised/lowered/reaffirmed wording
yes/no; beat/miss stated. Sonnet 5.5 would be the teacher, and the gate would be set relative to
teacher-vs-teacher agreement.

## Version 3 (T-2026-10-02-91 / -92): FAIL as registered (close on two fields)

**The idea: ask only what the release itself shows.**

v2's guidance question needed last quarter's guidance, so models had to guess. v3 splits it into
two questions answered by the text alone:

| v2 label | v3 label(s) | Why it should be reproducible |
|---|---|---|
| guidance: raised / maintained / lowered / initiated / withdrawn / mixed / none | **guidance_given**: does the text give numbers for a future period? yes / no | numbers are either there or not |
| | **guidance_revision**: does the text *explicitly say* raised / lowered (incl. withdrawn) / reaffirmed / mixed? else not_stated | only explicit wording counts; first-time guidance = not_stated |
| hedging: low / medium / high | **cautionary**: does management name a specific headwind or problem (outside the legal boilerplate)? yes / no | a named problem is either there or not |
| tone, vs_expectations | unchanged | teachers already agreed 77% / 90% |

**Teacher:** Claude Sonnet 5.5, the stronger model, instead of Haiku.

**The gate is set from the teachers, before the student is trained.**
1. Sonnet and Haiku both label the fresh 500-release test set.
2. If they agree on a field less than 70% of the time, that field is not reproducible. It is
   reported but doesn't count.
3. For every other field, the student must agree with Sonnet at least as often as Haiku does,
   minus 3 points, and must beat "always guess the most common label" by 3 points.
4. Tone also keeps the within-one-level check (90%).
5. PASS needs every counted field to pass and at least one guidance field to count.

In plain words: the free model must label as well as Claude Haiku does. The gate numbers are
written to `results/distill/gates_v3.json` (with a timestamp) when the labels arrive, before training.

**Stage C for v3 (T-92)** runs only after a PASS. It is T-89's trade and gates, with score = tone +
(raised − lowered) + (beat − miss), using only the fields that passed.

**Steps**
1. `python3 earnings_text.py --v3`: fresh test set (done by Claude).
2. In your terminal with your key: `python3 distill_label.py --v3 --submit`. It labels, waits,
   saves, and fixes the gate.
3. `nohup python3 distill_student.py --train-v3 > results/distill/train_v3.log 2>&1 &` (~2.5 h).

**v3 labels.**
- Sonnet labelled 4,450 training releases; Sonnet and Haiku both labelled the 494 test releases.
- Cost: **$25.33**, against an estimate of $17. Sonnet billed about 370 output tokens per release,
  not the 40 assumed.
- Teacher agreement (Haiku vs Sonnet):
  - tone 76.1%;
  - guidance_given 91.7%;
  - guidance_revision 87.8% (v2's guidance label: 63%);
  - vs_expectations 92.3%;
  - cautionary 83.8% (v2's hedging label: 48%).
- **Labels you can read straight off the text are reproducible.** That alone is a finding.

**v3 result: FAIL as registered** (fresh test of 494 releases, student vs Sonnet; the gate was
fixed 2026-10-02 22:31 before training):

| Field | Student | Needed | Haiku vs Sonnet | Verdict |
|---|---|---|---|---|
| Tone (exact) | 68.2% | 73.1% | 76.1% | **fail** (25 releases short) |
| Tone (within one) | 98.0% | 90% | — | pass |
| Guidance given | 88.66% | 88.70% | 91.7% | **fail by one release** (438 of 494; 439 needed) |
| Guidance revision | 89.3% | 84.9% | 87.8% | pass (better than Haiku) |
| vs expectations | 88.9% | 89.3% | 92.3% | **fail by three releases** |
| Cautionary | 84.2% | 80.8% | 83.8% | pass (better than Haiku) |

**Reading it.**
- The student labels guidance revisions and cautionary language **better than Claude Haiku
  does**, judged against Sonnet.
- It is within about 3.5 points of Haiku on guidance-given and beat/miss.
- Tone is the real miss: mostly neutral vs positive, and very-positive vs positive. Tone is the
  most subjective label (the teachers agree only 76%).
- **Stage C (T-92) does not run.** The pass rule required every counted field to pass, and three
  did not. Two misses are by 1 and 3 releases out of 494, but the rule is the rule.

**Deviation note (written 2026-10-03, before any v3 student label of new releases or any return).**
- **Sampling noise on 494 test releases** is about ±1.4 points (one standard error at ~89%
  accuracy).
- **The near-misses are within that noise.** guidance_given missed by 0.04 points (1 release) and
  vs_expectations by 0.4 points (3 releases). Tone's miss (4.9 points, about 3.5 standard errors)
  is real.
- **The registered verdict stands:** T-91 FAIL, and T-92 (which needed every field) does not run.
- **What moves forward, and why:** only **guidance_revision**, which passed its registered bar
  outright (89.3% vs 84.9%, above Haiku's 87.8%), goes into a new, separately pre-registered
  trading test, T-2026-10-03-93 (`guidance_drift.py`).
- **Why that is allowed:** that test does not rely on any field that failed.
- **Two disclosures:**
  - This is the fourth look at this data family, and the field was chosen after seeing it pass.
  - T-93 therefore uses a stricter t threshold (2.5) and must beat a plain keyword search.

## Guidance-revision drift (T-2026-10-03-93): FAIL — the labels are right, the market is faster

40,967 releases labelled by the v3 student (2005–2026; student training/test releases excluded).
The student calls 15.3% "raised" and 2.3% "lowered".

**What the stock does** (stock-specific return, signed in the labelled direction; 2020–2026):

| | Reaction day | Next day | Days 2–21 (the trade) |
|---|---|---|---|
| "Raised" (2,064) | **+1.01%** | +0.07% | −0.12% |
| "Lowered" (264) | **+3.82%** (the stock falls 3.8%) | −0.03% | +0.10% |

**The trade** (2020-01 to 2026-08, net of costs):

| Rule | Value | Needed | |
|---|---|---|---|
| G1 Sharpe | 0.26 | ≥ 0.50 | fail |
| G2 t | 0.70 | ≥ 2.5 | fail |
| G3 last 24 months | +29.7% | > 0 | ok |
| G4 concentration | 201% of profit in the best 24 months | ≤ 75% | fail |
| G5 vs keyword search | 0.26 vs 0.40 | ≥ keyword + 0.10 | fail |

- **Context (no decision):**
  - 2005–2019 Sharpe −0.56.
  - Placebo (labels shuffled within each year, 20 runs): median −0.19, and the real book beats 19
    of 20. That is a slight edge over luck, nowhere near the bar.
- **Mechanism:**
  - The student's labels clearly identify the news: on the reaction day "raised" stocks beat the
    market by 1% and "lowered" stocks lose 3.8%.
  - All of that happens on the first day. From the day after, there is no drift either way.
  - S&P 500 prices absorb explicit guidance changes within a day, so a trade entered after the
    reaction day has nothing left to earn.
  - A plain keyword search finds the same events about as well (the two agree on 62% of the
    releases either one trades; the keyword book 0.40 vs the student 0.26). The ML model adds
    nothing here.
- **What would still be open:**
  - Capturing the reaction-day move itself. That needs entry before the price moves: seconds
    after an after-close release, in after-hours trading, which is the premarket news engine's
    territory, not a daily backtest.
  - The literature's drift may survive in small caps; this universe is S&P 500 only.
