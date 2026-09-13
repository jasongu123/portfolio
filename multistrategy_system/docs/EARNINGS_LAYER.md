# Earnings layer

The earnings layer is integrated with the existing equity algorithm, but it is
disabled by default. Its first release has two deliberately limited functions:

1. journal point-in-time earnings states and shadow candidates; and
2. optionally block **new** core/tech entries during a scheduled earnings
   window.

It cannot submit an earnings order. Existing exits always bypass the earnings
gate.

## Architecture

```text
IBKR/WSH schedule + release/call document + quote reaction
                              |
                              v
                    earnings_parser.py
                              |
                    structured, evidenced JSON
                              |
                              v
                   earnings_events.sqlite3
                              |
                              v
                existing core + tech algorithm
                    |                    |
             entry embargo        shadow signal only
                    |
                    v
               ordinary risk/cost path -> IBKR
```

Free-form model prose never becomes an order. The parser extracts facts into a
strict schema; deterministic code evaluates missing fields, contradictions,
confidence, scheduled-call status, and separate market confirmation.

## Files

- `earnings_events.py`: SQLite journal, point-in-time state machine, and
  entry-only embargo.
- `earnings_parser.py`: provider-neutral parser contract, strict validation,
  source hashes, prompt/model versioning, and evidence requirements.
- `earnings_history_extract.py`: resumable IBKR event-window downloader.
- `earnings_replay.py`: exposes only assessments available at each historical
  bar timestamp.
- `ibkr2.py`: creates the layer only when requested, journals state transitions
  after 4 PM, and checks the embargo inside the shared `execute_trade` entry
  path used by both core and tech.

## Controls

All controls default to false/off:

```powershell
$env:EARNINGS_OBSERVER_ENABLED = "false"
$env:EARNINGS_ENTRY_EMBARGO_ENABLED = "false"
$env:EARNINGS_CONFIDENCE_FLOOR = "0.75"
```

There is intentionally no `EARNINGS_TRADING_ENABLED` flag. No earnings order
route exists in this version.

Do not enable the embargo from the existing `earnings_dates.json` without
refreshing and checking it. The current file ends around May 2025 and contains
date-only records. Date-only records block entries on the stated day but do not
guess whether the announcement is before or after market. Confirmed WSH event
records are needed for the richer windows:

- after-market report: 15:45 report day through 10:30 next weekday;
- before-market report: 15:45 previous weekday through 10:30 report day;
- during-market report: 09:30 through 16:00 report day.

These windows are not yet a performance claim. They are conservative research
defaults and must be compared with the unfiltered algorithm.

## Historical extraction

Do not retrieve seven continuous years for this experiment. Refresh the date
list first (and verify dates against WSH/company IR):

```powershell
py -3 .\pull_earnings.py
```

With TWS paper logged in on port 7497 and the trading bot stopped, inspect the
request plan:

```powershell
py -3 .\earnings_history_extract.py --port 7497 --dry-run
```

Initial screen—trades and midpoint around each event:

```powershell
py -3 .\earnings_history_extract.py --port 7497 --years 7
```

Detailed execution follow-up for surviving variants:

```powershell
py -3 .\earnings_history_extract.py --port 7497 --years 3 --types TRADES BID ASK
```

Every request asks for five days of one-minute bars with `useRTH=0`. Results
are written under `earnings_history/raw/<SYMBOL>/<DATE>/`. The SQLite manifest
makes the job resumable: completed files are skipped, and failures can be
retried by running the same command again.

The extractor never imports the live trading program and never uses an order
API.

## Parser and replay

`EarningsParser` accepts a model adapter callable. The model sees only:

- the document available at that timestamp;
- consensus values explicitly frozen before the event; and
- the required JSON schema.

It does not see later calls, price outcomes, revised consensus, or future
documents. Market confirmation is added by a separate stage, so a document
model cannot confirm its own interpretation.

Replay reads the journal at each bar timestamp:

```powershell
py -3 .\earnings_replay.py `
  --db .\earnings_events.sqlite3 `
  --symbol AAPL `
  --bars .\earnings_history\raw\AAPL\2026-07-30\trades_1m.csv
```

The release assessment can be visible before the call; the call assessment
cannot. A scheduled call keeps the state at `WAITING_FOR_CALL` until a call
assessment becomes available.

## Promotion sequence

1. Leave both flags false and run the test suite.
2. Audit/refresh event dates and report-time classifications.
3. Run historical event-window extraction and point-in-time replay.
4. Enable observer only; record future releases, calls, quotes, and shadow
   decisions.
5. Backtest the ordinary algorithm with and without the entry embargo.
6. Consider paper earnings orders only through a separately reviewed future
   implementation.

The previous broad event filter did not improve the 2023–2025 book, so this
layer is not presumed beneficial merely because earnings risk sounds
important. It remains unvalidated until the narrower, time-aware version
improves after-cost walk-forward results.
