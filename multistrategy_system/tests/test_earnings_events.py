from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from earnings_events import (
    ELIGIBLE_LONG,
    HOLD,
    WAITING_FOR_CALL,
    EarningsEngine,
    EarningsEvent,
    EarningsEventStore,
    assessment_from_mapping,
)
from earnings_history_extract import estimated_minutes, load_legacy_events, plan_jobs
from earnings_parser import EarningsParser, ParserInput, validate_model_output


ET = ZoneInfo("America/New_York")


class EarningsEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EarningsEventStore(Path(self.temp.name) / "events.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def _engine(self, *, enabled=True, embargo=True):
        return EarningsEngine(
            self.store,
            enabled=enabled,
            embargo_enabled=embargo,
        )

    def test_after_market_window_blocks_entries_through_next_morning(self):
        engine = self._engine()
        engine.ingest_event(
            EarningsEvent(
                event_id="intc-2026q2",
                symbol="INTC",
                report_date="2026-07-30",
                time_of_day="AFTER_MARKET",
                status="CONFIRMED",
            )
        )

        self.assertFalse(
            engine.should_block_entry(
                "INTC", dt.datetime(2026, 7, 30, 15, 44, tzinfo=ET)
            )[0]
        )
        self.assertTrue(
            engine.should_block_entry(
                "INTC", dt.datetime(2026, 7, 30, 15, 45, tzinfo=ET)
            )[0]
        )
        self.assertTrue(
            engine.should_block_entry(
                "INTC", dt.datetime(2026, 7, 31, 10, 30, tzinfo=ET)
            )[0]
        )
        self.assertFalse(
            engine.should_block_entry(
                "INTC", dt.datetime(2026, 7, 31, 10, 31, tzinfo=ET)
            )[0]
        )

    def test_before_market_window_starts_at_previous_weekday_close(self):
        engine = self._engine()
        engine.ingest_event(
            EarningsEvent(
                event_id="wmt-2026q2",
                symbol="WMT",
                report_date="2026-08-17",  # Monday
                time_of_day="BEFORE_MARKET",
            )
        )
        self.assertTrue(
            engine.should_block_entry(
                "WMT", dt.datetime(2026, 8, 14, 15, 45, tzinfo=ET)
            )[0]
        )

    def test_legacy_unspecified_date_does_not_infer_prior_day_close(self):
        engine = self._engine()
        engine.ingest_event(
            EarningsEvent(
                event_id="legacy-aapl",
                symbol="AAPL",
                report_date="2026-08-03",
            )
        )
        self.assertFalse(
            engine.should_block_entry(
                "AAPL", dt.datetime(2026, 7, 31, 15, 59, tzinfo=ET)
            )[0]
        )
        self.assertTrue(
            engine.should_block_entry(
                "AAPL", dt.datetime(2026, 8, 3, 10, 0, tzinfo=ET)
            )[0]
        )

    def test_confirmed_timed_event_wins_over_legacy_duplicate(self):
        engine = self._engine()
        engine.ingest_event(
            EarningsEvent(
                event_id="legacy-intc",
                symbol="INTC",
                report_date="2026-07-30",
            )
        )
        engine.ingest_event(
            EarningsEvent(
                event_id="wsh-intc",
                symbol="INTC",
                report_date="2026-07-30",
                time_of_day="AFTER_MARKET",
                status="CONFIRMED",
                source="WSH",
            )
        )
        chosen = engine.event_near(
            "INTC", dt.datetime(2026, 7, 30, 15, 50, tzinfo=ET)
        )
        self.assertEqual(chosen.event_id, "wsh-intc")

    def test_point_in_time_replay_cannot_see_later_call(self):
        engine = self._engine(embargo=False)
        event = EarningsEvent(
            event_id="nvda-2026q2",
            symbol="NVDA",
            report_date="2026-08-05",
            time_of_day="AFTER_MARKET",
            release_at=dt.datetime(2026, 8, 5, 16, 5, tzinfo=ET),
            call_at=dt.datetime(2026, 8, 5, 17, 0, tzinfo=ET),
        )
        engine.ingest_event(event)
        engine.ingest_assessment(
            {
                "assessment_id": "release",
                "event_id": event.event_id,
                "symbol": "NVDA",
                "document_type": "release",
                "available_at": dt.datetime(2026, 8, 5, 16, 8, tzinfo=ET),
                "assessed_at": dt.datetime(2026, 8, 5, 16, 9, tzinfo=ET),
                "direction": "bullish",
                "confidence": 0.9,
                "market_confirmation": "confirmed",
            }
        )
        engine.ingest_assessment(
            {
                "assessment_id": "call",
                "event_id": event.event_id,
                "symbol": "NVDA",
                "document_type": "call_complete",
                "available_at": dt.datetime(2026, 8, 5, 18, 0, tzinfo=ET),
                "assessed_at": dt.datetime(2026, 8, 5, 18, 1, tzinfo=ET),
                "direction": "bearish",
                "confidence": 0.9,
                "market_confirmation": "confirmed",
            }
        )

        before_call = self.store.latest_assessment(
            "NVDA", dt.datetime(2026, 8, 5, 16, 30, tzinfo=ET)
        )
        after_call = self.store.latest_assessment(
            "NVDA", dt.datetime(2026, 8, 5, 18, 2, tzinfo=ET)
        )
        self.assertEqual(before_call.assessment_id, "release")
        self.assertEqual(after_call.assessment_id, "call")
        self.assertEqual(
            engine.state_at(
                "NVDA", dt.datetime(2026, 8, 5, 17, 30, tzinfo=ET)
            )[0],
            WAITING_FOR_CALL,
        )

    def test_parser_assessment_needs_separate_market_confirmation(self):
        engine = self._engine(embargo=False)
        event = EarningsEvent(
            event_id="aapl-2026q3",
            symbol="AAPL",
            report_date="2026-08-04",
            time_of_day="AFTER_MARKET",
        )
        engine.ingest_event(event)

        def extractor(_payload):
            return {
                "direction": "bullish",
                "confidence": 0.95,
                "fields": {"guidance": "raised"},
                "evidence": [{"field": "guidance", "excerpt": "raised range"}],
                "missing_required_fields": [],
                "contradictions": [],
            }

        parser = EarningsParser(
            engine,
            extractor,
            model_id="test-model",
            clock=lambda: dt.datetime(2026, 8, 4, 16, 6, tzinfo=ET),
        )
        assessment = parser.parse(
            ParserInput(
                event_id=event.event_id,
                symbol="AAPL",
                document_type="release",
                available_at=dt.datetime(2026, 8, 4, 16, 5, tzinfo=ET),
                source_text="Management raised the full-year guidance range.",
                consensus={"eps": 1.0},
            )
        )
        self.assertEqual(assessment.market_confirmation, "unavailable")
        self.assertEqual(
            engine.state_at(
                "AAPL", assessment.assessed_at + dt.timedelta(seconds=1)
            )[0],
            HOLD,
        )
        confirmed = engine.record_market_confirmation(
            assessment.assessment_id,
            "confirmed",
            assessment.assessed_at + dt.timedelta(minutes=5),
            market_fields={"residual_return": 0.02, "spread_pct": 0.001},
        )
        self.assertEqual(confirmed.market_confirmation, "confirmed")
        self.assertEqual(
            engine.state_at(
                "AAPL", confirmed.assessed_at + dt.timedelta(seconds=1)
            )[0],
            ELIGIBLE_LONG,
        )

    def test_invalid_parser_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            validate_model_output(
                {
                    "direction": "bullish",
                    "confidence": 2,
                    "fields": {},
                    "evidence": [],
                    "missing_required_fields": [],
                    "contradictions": [],
                }
            )

    def test_assessment_mapping_rejects_unknown_document_type(self):
        with self.assertRaisesRegex(ValueError, "document_type"):
            assessment_from_mapping(
                {
                    "event_id": "e1",
                    "symbol": "AAPL",
                    "document_type": "rumor",
                    "available_at": dt.datetime.now(ET),
                    "direction": "neutral",
                    "confidence": 0.5,
                }
            )


class EarningsHistoryPlanningTests(unittest.TestCase):
    def test_plans_event_windows_without_continuous_history(self):
        with tempfile.TemporaryDirectory() as temp:
            date_file = Path(temp) / "dates.json"
            date_file.write_text(
                '{"AAPL": ["2026-01-29", "2026-04-30"], "MSFT": ["2026-01-28"]}',
                encoding="utf-8",
            )
            events = load_legacy_events(date_file)
            jobs = plan_jobs(
                events,
                Path(temp) / "history",
                data_types=("TRADES", "MIDPOINT"),
                symbols=("AAPL",),
                start=dt.date(2026, 2, 1),
            )
        self.assertEqual(len(jobs), 2)
        self.assertEqual({job.data_type for job in jobs}, {"TRADES", "MIDPOINT"})
        self.assertEqual(jobs[0].report_date, dt.date(2026, 4, 30))
        self.assertIn("useRTH", __import__("earnings_history_extract").__doc__)

    def test_pacing_estimate(self):
        self.assertAlmostEqual(estimated_minutes(120, 5), 10.0)


if __name__ == "__main__":
    unittest.main()
