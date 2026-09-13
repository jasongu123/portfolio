from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from data_ingestion.storage import DataStore
from news_analyzer import NewsAnalyzer
from premarket_news import (
    bounded_limit_price,
    closing_action,
    directional_return,
    evaluate_candidate,
)


VALID_ASSESSMENT = {
    "event_type": "earnings",
    "scope": "company",
    "direction": "bullish",
    "severity": 3,
    "horizon": "intraday",
    "novelty": "new",
    "relevance": 0.95,
    "confidence": 0.8,
    "headline": "Company raises guidance",
    "catalyst": "guidance raise",
    "surprise_summary": "Guidance increased above the prior range",
    "reason": "The update is material and company-specific.",
    "evidence_article_ids": ["article-1"],
}


class FakeMessages:
    def __init__(self, result: dict):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text=json.dumps(self.result))],
            usage=SimpleNamespace(input_tokens=120, output_tokens=70),
        )


class FakeAnthropic:
    def __init__(self, result: dict):
        self.messages = FakeMessages(result)


class FakeIBKR:
    def isConnected(self):
        return True

    def get_news_provider_codes(self):
        return ["BRFG"]

    def get_stock_news(self, symbol, max_results=10, lookback_days=4):
        return [
            {
                "provider_code": "BRFG",
                "article_id": "article-1",
                "time": "20260713 10:00:00",
                "headline": f"{symbol} raises full-year guidance",
            }
        ]


class NewsAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = DataStore(self.temp.name)
        self.fake_api = FakeAnthropic(VALID_ASSESSMENT)
        self.analyzer = NewsAnalyzer(
            ibkr_client=FakeIBKR(),
            anthropic_client=self.fake_api,
            store=self.store,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_previous_close_includes_friday_for_monday_premarket(self):
        monday = dt.datetime(2026, 7, 13, 7, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(
            self.analyzer.previous_regular_close(monday),
            dt.datetime(2026, 7, 10, 16, 0),
        )

    def test_structured_event_is_deduplicated_and_journaled(self):
        now = dt.datetime(2026, 7, 13, 7, 0, tzinfo=ZoneInfo("America/New_York"))
        context = {"quote_mid": 205.0, "previous_close": 200.0, "gap_pct": 0.025}
        first = self.analyzer.analyze_premarket_sentiment(
            "AAPL", market_context=context, now=now
        )
        second = self.analyzer.analyze_premarket_sentiment(
            "AAPL", market_context=context, now=now
        )
        self.assertEqual(first["_status"], "success")
        self.assertAlmostEqual(first["score"], 0.8)
        self.assertEqual(second["_status"], "no_new_event")
        self.assertEqual(len(self.fake_api.messages.calls), 1)
        call = self.fake_api.messages.calls[0]
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["output_config"]["format"]["type"], "json_schema")
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM news_assessments").fetchone()
        self.assertEqual(row["symbol"], "AAPL")
        self.assertEqual(row["direction"], "bullish")
        self.assertEqual(row["input_tokens"], 120)

    def test_invalid_result_is_rejected(self):
        invalid = dict(VALID_ASSESSMENT, severity=9)
        with self.assertRaises(ValueError):
            self.analyzer._validate_result(invalid)


class PremarketPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assessment = dict(VALID_ASSESSMENT, _status="success")
        self.context = {
            "quote_mid": 101.0,
            "is_synthetic_quote": False,
            "spread_pct": 0.002,
            "gap_pct": 0.01,
        }

    def test_material_company_event_is_eligible(self):
        action, _ = evaluate_candidate("AAPL", self.assessment, self.context)
        self.assertEqual(action, "BUY")

    def test_market_story_is_not_stamped_across_every_stock(self):
        assessment = dict(self.assessment, scope="market", event_type="macro")
        self.assertEqual(evaluate_candidate("AAPL", assessment, self.context)[0], "HOLD")
        self.assertEqual(evaluate_candidate("SPY", assessment, self.context)[0], "BUY")

    def test_contradictory_gap_blocks_candidate(self):
        context = dict(self.context, gap_pct=-0.01)
        self.assertEqual(evaluate_candidate("AAPL", self.assessment, context)[0], "HOLD")

    def test_short_lifecycle_is_direction_aware(self):
        self.assertAlmostEqual(directional_return(100.0, 90.0, -10), 0.10)
        self.assertEqual(closing_action(-10), "BUY")
        self.assertEqual(closing_action(10), "SELL")

    def test_limit_price_is_bounded_around_mid(self):
        quote = {"bid": 99.0, "ask": 101.0, "mid": 100.0}
        self.assertEqual(bounded_limit_price("BUY", quote, 0.003), 100.3)
        self.assertEqual(bounded_limit_price("SELL", quote, 0.003), 99.7)


class NewsJournalTests(unittest.TestCase):
    def test_decision_outcomes_can_be_updated(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DataStore(temp)
            store.record_news_assessment(
                {
                    "assessment_id": "a1",
                    "symbol": "AAPL",
                    "workflow": "premarket",
                    "assessed_at": "2026-07-13T11:00:00+00:00",
                    "model_id": "test",
                    "prompt_version": "v1",
                    "source_status": "available",
                    "status": "success",
                    "event_ids_json": "[]",
                    "event_count": 1,
                }
            )
            store.record_premarket_decision(
                {
                    "decision_id": "d1",
                    "assessment_id": "a1",
                    "symbol": "AAPL",
                    "decided_at": "2026-07-13T07:00:00-04:00",
                    "action": "BUY",
                    "status": "eligible",
                    "reason": "test",
                }
            )
            store.update_premarket_decision(
                "d1", price_0930=102.0, return_0930=0.02, status="submitted"
            )
            with store.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM premarket_news_decisions WHERE decision_id='d1'"
                ).fetchone()
            self.assertEqual(row["status"], "submitted")
            self.assertEqual(row["price_0930"], 102.0)


if __name__ == "__main__":
    unittest.main()
