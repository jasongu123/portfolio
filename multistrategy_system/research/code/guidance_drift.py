"""
Guidance-revision drift — RESEARCH ONLY. Trial T-2026-10-03-93, pre-registered 2026-10-03 before any
new student label or any return was computed. Nothing here touches the live engine.

Question: when an earnings release explicitly says the company RAISED (or LOWERED) its guidance,
does the stock keep drifting that way over the following month, after the first-day reaction?
(The management-forecast and post-earnings-drift literature says it historically did.)

Why this field: the v3 distilled student (DISTILL_PROJECT.md, T-91) passed its registered bar on
guidance_revision outright (89.3% agreement with Sonnet 5.5 vs 84.9% needed; Haiku 87.8%). T-91 as a
whole failed (tone; two fields by 1 and 3 releases), so this test uses ONLY guidance_revision.
Disclosed: 4th look at this data family; field chosen after it passed; hence G2 t >= 2.5 and G5.

Labels
  student   models/finbert_earnings_v3 reads the first 12,000 narrative characters of every item-2.02
            release (earnings_text.py --all); label = argmax of guidance_revision.
            Releases used to train or test the student (v1 sample, train2, test2, test3) are excluded.
  keyword   baseline, fixed here (tightened before any run so it is not a straw man: the first draft
            read "Outlook: revenue increased 5%" as a raise): a raise verb (raise.., increase..,
            lift.., boost.., upgrade..) or a cut verb (lower.., reduce.., cut(s), withdr.., suspend..)
            followed within one sentence (60 characters) by guidance/outlook/forecast, or
            guidance/outlook/forecast directly followed by a past-tense verb ("guidance was raised").
            Raise only -> raised; cut only -> lowered; both or neither -> not traded.
Events: reaction day = the trading day of SEC acceptance if accepted before 16:00 New York time,
  else the next trading day (earnings_moves.reaction_days; acceptance times are UTC). The stock must
  be an S&P 500 member (point-in-time), price >= $5 and >= 260 days of history on the reaction day.
Trade: raised -> +1, lowered -> -1, entered at the close of reaction day + 1 (the first-day reaction
  is not traded), held 20 trading days; each position earns stock minus the equal-weight member
  return. Each leg (L raised, S lowered) holds its open positions in equal weight (0 when none).
  Costs 6 bp at entry and 6 bp at exit per position; shorts pay 0.5% a year borrow.
  Book = 0.5 x L + 0.5 x S (an empty leg contributes 0).
Gate, on 2020-01-01 .. the last complete holding period, student book:
  G1 net Sharpe >= 0.50
  G2 Newey-West (20 lags) t of the mean daily net return >= 2.5
  G3 net return over the last 24 months > 0
  G4 best 24-month block <= 75% of total net profit
  G5 net Sharpe >= the keyword book's net Sharpe + 0.10
  PASS (all five) -> candidate for a paper forward test; no live change from this trial.
Reported, no decision: each leg alone; 2005-2019 (survivorship: the universe is 2020-2026 members);
  event study (mean signed stock-specific return on t+1 and t+2..t+21 per label); student vs keyword
  agreement; events per year; a placebo (added before any run, after a random-label dry run gave
  gross book Sharpes from -0.74 to +0.61): the student's labels shuffled across releases within
  each year, 20 times, and the real book's rank among them.

  python3 guidance_drift.py --label     student labels for every release (Mac GPU, ~2 h; resumable)
  python3 guidance_drift.py             the test -> results/guidance_drift/
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from earnings_moves import reaction_days, sr, conc, COST_SIDE, BORROW  # noqa: E402
from taking_sides import membership  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'guidance_drift')
INDEX = os.path.join(BD, 'sec_8k_202_index.csv')
TEXT_DIR = os.path.join(BD, 'earnings_text')
STUDENT_LABELS = os.path.join(BD, 'earnings_labels3_student_all.csv')
USED_SETS = ['earnings_text_sample.csv', 'earnings_text_train2.csv', 'earnings_text_test2.csv',
             'earnings_text_test3.csv']
MAIN, CONTEXT = ('2020-01-01', '2026-12-31'), ('2005-01-01', '2019-12-31')
HOLD, NW_LAGS = 20, 20
GATE = {'sharpe': 0.50, 't': 2.5, 'conc': 0.75, 'vs_keyword': 0.10}

_TARGET = r'(?:guidance|outlook|forecast)'
_UP = r'(?:rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|upgrad(?:e|es|ed|ing))'
_DOWN = r'(?:lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|cut(?:s|ting)?|withdr(?:aw|aws|ew|awn|awing)|suspend(?:s|ed|ing)?)'


_UP_PAST = r'(?:raised|increased|lifted|boosted|upgraded)'
_DOWN_PAST = r'(?:lowered|reduced|cut|withdrawn|suspended)'


def _near(verb, past):
    return re.compile(rf'\b{verb}\b[^.]{{0,60}}\b{_TARGET}\b'
                      rf'|\b{_TARGET}\s+(?:(?:is|was|has been|have been|are|were|being)\s+)?{past}\b', re.I)


UP_RE, DOWN_RE = _near(_UP, _UP_PAST), _near(_DOWN, _DOWN_PAST)


def keyword_label(text):
    up, down = bool(UP_RE.search(text)), bool(DOWN_RE.search(text))
    return 'raised' if up and not down else 'lowered' if down and not up else 'none'


def _used():
    used = set()
    for f in USED_SETS:
        used |= set(pd.read_csv(os.path.join(BD, f)).accession)
    return used


def label_all():
    """Student v3 labels for every downloaded release not used to train or test it. Resumable."""
    import torch
    from transformers import AutoTokenizer
    import distill_student as ds
    from distill_label import FIELDS_V3, INPUT_CHARS
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(ds.MODEL_DIR_V3)
    model = ds.build_model_v2(FIELDS_V3)
    model.load_state_dict(torch.load(os.path.join(ds.MODEL_DIR_V3, 'student.pt'), map_location='cpu'))
    model.to(device).eval()
    idx = pd.read_csv(INDEX)
    used = _used()
    done = set(pd.read_csv(STUDENT_LABELS).accession) if os.path.exists(STUDENT_LABELS) else set()
    todo = [a for a in idx.accession if a not in used and a not in done
            and os.path.exists(os.path.join(TEXT_DIR, f'{a}.txt'))]
    print(f"{len(todo)} releases to label ({len(done)} already labelled, {len(used)} excluded as student data)", flush=True)
    t0, step = time.time(), 2000
    for i in range(0, len(todo), step):
        part = todo[i:i + step]
        texts = [open(os.path.join(TEXT_DIR, f'{a}.txt')).read()[:INPUT_CHARS] for a in part]
        keep = [k for k, t in enumerate(texts) if len(t) >= 200]
        part, texts = [part[k] for k in keep], [texts[k] for k in keep]
        probs = ds.predict_v2(model, tok, [ds.chunk_ids(tok, t) for t in texts], device, FIELDS_V3)
        rows = pd.DataFrame({'accession': part})
        for k, v in FIELDS_V3.items():
            rows[k] = [v[j] for j in probs[k].argmax(1)]
            for c, name in enumerate(v):
                rows[f'p_{k}_{name}'] = probs[k][:, c].round(4)
        rows['keyword'] = [keyword_label(t) for t in texts]
        rows.to_csv(STUDENT_LABELS, mode='a', header=not os.path.exists(STUDENT_LABELS), index=False)
        print(f"  {i + len(part)}/{len(todo)} ({(time.time() - t0) / 60:.0f} min)", flush=True)


def stock_specific_returns():
    """Stock minus equal-weight member return, and the eligibility mask (earnings_moves.build_events)."""
    px = pd.read_csv(os.path.join(BD, 'sp500_prices_long.csv'), index_col=0, parse_dates=True).sort_index()
    uni = pd.read_csv(os.path.join(BD, 'sp500_universe_2020_2026.csv')).iloc[:, 0].astype(str).tolist()
    chg = pd.read_csv(os.path.join(BD, 'sp500_changes.csv'), parse_dates=['date'])
    mem = membership(chg, uni)
    days = px.index
    r = px.pct_change(fill_method=None)
    member = pd.DataFrame(False, index=days, columns=px.columns)
    for s in px.columns:
        if s in mem:
            a, rm = mem[s]
            m = np.ones(len(days), bool)
            if pd.notna(a):
                m &= days >= a
            if pd.notna(rm) and (pd.isna(a) or rm > a):
                m &= days < rm
            member[s] = m
    ok = member & (px >= 5) & (px.notna().cumsum() >= 260)
    mkt = r.where(ok).mean(axis=1)
    return r.sub(mkt, axis=0), ok, days


def events(labels, col, e, ok, days):
    """One row per traded release: reaction day, symbol, direction (+1 raised, -1 lowered)."""
    idx = pd.read_csv(INDEX)
    ev = idx.merge(labels[['accession', col]], on='accession', how='inner')
    ev = ev[ev[col].isin(['raised', 'lowered'])].copy()
    ev['react'] = reaction_days(ev, days)
    ev = ev.dropna(subset=['react'])
    ev = ev[ev.symbol.isin(e.columns)]
    ev = ev[[bool(ok.at[d, s]) for d, s in zip(ev.react, ev.symbol)]]
    ev['dir'] = np.where(ev[col] == 'raised', 1, -1)
    return ev.drop_duplicates(['symbol', 'react', 'dir']).reset_index(drop=True)


def leg(ev, e, days, direction):
    """Calendar-time leg: equal weight across open positions, net of costs (earnings_moves.leg_returns)."""
    sub = ev[ev.dir == direction]
    pos = {d: i for i, d in enumerate(days)}
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    n = len(days)
    g_sum, n_sum, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for d, s in zip(sub.react, sub.symbol):
        i, j = pos[pd.Timestamp(d)], col[s]
        start, end = i + 2, i + 1 + HOLD
        if end >= n:
            continue
        for t in range(start, end + 1):
            g = direction * (0.0 if np.isnan(E[t, j]) else E[t, j])
            c = (COST_SIDE if t == start else 0.0) + (COST_SIDE if t == end else 0.0) + (BORROW / 252 if direction < 0 else 0.0)
            g_sum[t] += g
            n_sum[t] += g - c
            cnt[t] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        gross = np.where(cnt > 0, g_sum / cnt, 0.0)
        net = np.where(cnt > 0, n_sum / cnt, 0.0)
    return pd.Series(gross, index=days), pd.Series(net, index=days), pd.Series(cnt, index=days)


def nw_t(x, lags=NW_LAGS):
    x = np.asarray(x.dropna(), float)
    n = len(x)
    if n < 50:
        return float('nan')
    u = x - x.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return float(x.mean() / np.sqrt(var / n))


def event_study(ev, e, days):
    pos = {d: i for i, d in enumerate(days)}
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    out = []
    for d, s, sg in zip(ev.react, ev.symbol, ev.dir):
        i, j = pos[pd.Timestamp(d)], col[s]
        if i + 21 >= len(days):
            out.append((np.nan, np.nan, np.nan))
            continue
        out.append((E[i, j] * sg, np.nansum(E[i + 1, j]) * sg, np.nansum(E[i + 2:i + 22, j]) * sg))
    return pd.DataFrame(out, columns=['reaction day', 't+1', 't+2..t+21'], index=ev.index)


def book(ev, e, days):
    gl, nl, cl = leg(ev, e, days, 1)
    gs, ns, cs = leg(ev, e, days, -1)
    return {'book': 0.5 * nl + 0.5 * ns, 'raised leg': nl, 'lowered leg': ns,
            'book gross': 0.5 * gl + 0.5 * gs, 'n_raised': cl, 'n_lowered': cs}


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    labels = pd.read_csv(STUDENT_LABELS)
    e, ok, days = stock_specific_returns()
    last = days[-1 - HOLD - 1]
    emit(f"Labelled releases: {len(labels)} (student data excluded). Student guidance_revision: " +
         ", ".join(f"{k} {v:.1%}" for k, v in labels.guidance_revision.value_counts(normalize=True).items()))
    emit("Keyword baseline: " + ", ".join(f"{k} {v:.1%}" for k, v in labels.keyword.value_counts(normalize=True).items()))
    both = labels[labels.guidance_revision.isin(['raised', 'lowered']) | labels.keyword.isin(['raised', 'lowered'])]
    stud = both.guidance_revision.where(both.guidance_revision.isin(['raised', 'lowered']), 'none')
    emit(f"Student vs keyword on releases either one trades ({len(both)}): same call {(stud == both.keyword).mean():.1%}\n"
         + pd.crosstab(stud.rename('student'), both.keyword.rename('keyword')).to_string())

    res = {}
    for name, col in (('student', 'guidance_revision'), ('keyword', 'keyword')):
        ev = events(labels, col, e, ok, days)
        b = book(ev, e, days)
        res[name] = (ev, b)
        ev.assign(year=pd.to_datetime(ev.react).dt.year).groupby(['year', 'dir']).size().unstack(fill_value=0) \
            .rename(columns={1: 'raised', -1: 'lowered'}).to_csv(os.path.join(OUT, f'events_per_year_{name}.csv'))
        pd.DataFrame({k: v for k, v in b.items()}).to_csv(os.path.join(OUT, f'daily_{name}.csv'))

    emit("\nEvent study, mean signed stock-specific return % (positive = moves the labelled way):")
    for name in ('student', 'keyword'):
        ev = res[name][0]
        es = event_study(ev, e, days)
        for (lo, hi), per in ((MAIN, '2020-26'), (CONTEXT, '2005-19')):
            m = (pd.to_datetime(ev.react) >= lo) & (pd.to_datetime(ev.react) <= hi)
            for d, lab in ((1, 'raised'), (-1, 'lowered')):
                x = es[m & (ev.dir == d)].dropna()
                if len(x):
                    emit(f"  {name:8s} {per:8s} {lab:8s} n={len(x):5d}  " +
                         "  ".join(f"{c} {x[c].mean() * 100:+.2f}" for c in x.columns))

    def stats(x):
        return {'sharpe': sr(x), 't': nw_t(x), 'ann %': x.mean() * 252 * 100}

    emit("\nCalendar-time books, net of costs:")
    for (lo, hi), per in ((MAIN, '2020-01 .. last complete'), (CONTEXT, '2005-2019 (context)')):
        hi = min(pd.Timestamp(hi), last)
        for name in ('student', 'keyword'):
            b = res[name][1]
            for k in ('book', 'raised leg', 'lowered leg', 'book gross'):
                s = stats(b[k][lo:hi])
                emit(f"  {per:26s} {name:8s} {k:12s} Sharpe {s['sharpe']:+.2f}  t(NW) {s['t']:+.2f}  ann {s['ann %']:+.2f}%")

    lab_st = labels[['accession', 'guidance_revision']].merge(pd.read_csv(INDEX)[['accession', 'filing_date']], on='accession')
    lab_st['year'] = lab_st.filing_date.str[:4]
    plac = []
    for seed in range(20):
        sh = lab_st.copy()
        sh['guidance_revision'] = sh.groupby('year').guidance_revision.transform(
            lambda v: v.sample(frac=1, random_state=seed).values)
        plac.append(sr(book(events(sh, 'guidance_revision', e, ok, days), e, days)['book'][MAIN[0]:last]))
    real = sr(res['student'][1]['book'][MAIN[0]:last])
    emit(f"\nPlacebo (labels shuffled within year, 20 runs), 2020+ net book Sharpe: median {np.median(plac):+.2f}, "
         f"range {min(plac):+.2f} .. {max(plac):+.2f}; real {real:+.2f} beats {sum(p < real for p in plac)} of 20")

    x = res['student'][1]['book'][MAIN[0]:last]
    kw = res['keyword'][1]['book'][MAIN[0]:last]
    s_st, s_kw = sr(x), sr(kw)
    last24 = x[x.index > x.index[-1] - pd.DateOffset(months=24)].sum()
    c = conc(x)
    g = [('G1 net Sharpe >= 0.50', s_st >= GATE['sharpe'], f"{s_st:.2f}"),
         ('G2 NW t >= 2.5', nw_t(x) >= GATE['t'], f"{nw_t(x):.2f}"),
         ('G3 last 24m > 0', last24 > 0, f"{last24 * 100:+.2f}%"),
         ('G4 best 24m <= 75% of profit', c <= GATE['conc'], f"{c:.0%}" if np.isfinite(c) else 'no profit'),
         ('G5 Sharpe >= keyword + 0.10', s_st >= s_kw + GATE['vs_keyword'], f"{s_st:.2f} vs keyword {s_kw:.2f}")]
    emit("\nGate (T-2026-10-03-93), student book, 2020-01 .. " + f"{last:%Y-%m-%d}:")
    for name, passed, val in g:
        emit(f"  {name:32s} {val:24s} {'ok' if passed else 'FAIL'}")
    emit(f"\nT-93 VERDICT: {'PASS' if all(p for _, p, _ in g) else 'FAIL'}")
    open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', action='store_true')
    a = ap.parse_args()
    if a.label:
        label_all()
    else:
        main()
