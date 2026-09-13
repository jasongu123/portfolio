"""Replay saved earnings states against historical bar timestamps.

This is the leakage barrier between the parser journal and future earnings
research.  At each bar, it asks ``EarningsEngine.state_at`` what was visible at
that exact time.  Later call assessments therefore cannot appear in an earlier
release decision.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from earnings_events import EarningsEngine, EarningsEventStore


@dataclass(frozen=True)
class ReplayDecision:
    timestamp: dt.datetime
    symbol: str
    state: str
    signal: str
    reason: str
    event_id: str | None


def replay_timestamps(
    store: EarningsEventStore,
    symbol: str,
    timestamps: Iterable[dt.datetime],
    *,
    confidence_floor: float = 0.75,
) -> list[ReplayDecision]:
    engine = EarningsEngine(
        store,
        enabled=True,
        embargo_enabled=False,
        confidence_floor=confidence_floor,
    )
    output = []
    for timestamp in sorted(timestamps):
        state, reason, event_id = engine.state_at(symbol, timestamp)
        output.append(
            ReplayDecision(
                timestamp=timestamp,
                symbol=symbol.strip().upper(),
                state=state,
                signal=engine.shadow_signal(symbol, timestamp),
                reason=reason,
                event_id=event_id,
            )
        )
    return output


def load_timestamps(path: str | Path, column: str = "datetime") -> list[dt.datetime]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [dt.datetime.fromisoformat(row[column]) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--bars", required=True)
    parser.add_argument("--timestamp-column", default="datetime")
    parser.add_argument("--confidence-floor", type=float, default=0.75)
    args = parser.parse_args(argv)

    store = EarningsEventStore(args.db)
    timestamps = load_timestamps(args.bars, args.timestamp_column)
    decisions = replay_timestamps(
        store,
        args.symbol,
        timestamps,
        confidence_floor=args.confidence_floor,
    )
    counts = Counter(decision.state for decision in decisions)
    print(f"Replayed {len(decisions)} bars for {args.symbol.upper()}")
    for state, count in sorted(counts.items()):
        print(f"  {state}: {count}")
    candidates = [decision for decision in decisions if decision.signal != "HOLD"]
    print(f"Shadow candidate bars: {len(candidates)}")
    for decision in candidates[:20]:
        print(
            f"  {decision.timestamp.isoformat()} {decision.signal} "
            f"({decision.reason})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

