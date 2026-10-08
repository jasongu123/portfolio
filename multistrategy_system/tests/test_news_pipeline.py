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




class SchemaConstraintTests(unittest.TestCase):
    def test_schema_has_no_keywords_structured_outputs_reject(self):
        from news_analyzer import NEWS_ASSESSMENT_SCHEMA
        banned = {'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
                  'minLength', 'maxLength', 'minItems', 'maxItems'}

        def walk(node):
            if isinstance(node, dict):
                self.assertFalse(banned & set(node), node)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(NEWS_ASSESSMENT_SCHEMA)



class ModelRequestParamsTests(unittest.TestCase):
    def test_haiku_4_5_unchanged(self):
        from news_analyzer import model_request_params
        p = model_request_params("claude-haiku-4-5-20251001")
        self.assertEqual((p["temperature"], p["max_tokens"]), (0, 500))
        self.assertNotIn("thinking", p)

    def test_sonnet_5_5_has_no_temperature_and_thinking_off(self):
        from news_analyzer import model_request_params
        p = model_request_params("claude-sonnet-5-5")
        self.assertNotIn("temperature", p)
        self.assertEqual(p["thinking"], {"type": "between_tools"})
        self.assertIn("format", p["output_config"])

    def test_opus_5_5_low_effort_and_unknown_models_safe(self):
        from news_analyzer import model_request_params
        p = model_request_params("claude-opus-5-5")
        self.assertEqual(p["output_config"]["effort"], "low")
        self.assertNotIn("temperature", p)
        q = model_request_params("claude-haiku-5-5")
        self.assertNotIn("temperature", q)
        self.assertNotIn("thinking", q)
        self.assertEqual(q["max_tokens"], 4000)

    def test_sonnet_5_5_end_to_end_with_a_thinking_block_first(self):
        """The analyzer sends Sonnet 5.5 settings and reads the text block after a thinking block."""
        class ThinkingFirst(FakeMessages):
            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="thinking", thinking=""),
                             SimpleNamespace(type="text", text=json.dumps(self.result))],
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1))

        with tempfile.TemporaryDirectory() as folder:
            store = DataStore(folder)
            fake = FakeAnthropic(VALID_ASSESSMENT)
            fake.messages = ThinkingFirst(VALID_ASSESSMENT)
            analyzer = NewsAnalyzer(ibkr_client=FakeIBKR(), anthropic_client=fake, store=store,
                                    model="claude-sonnet-5-5")
            now = dt.datetime(2026, 7, 13, 7, 0, tzinfo=ZoneInfo("America/New_York"))
            res = analyzer.analyze_premarket_sentiment("AAPL", market_context={}, now=now)
        self.assertEqual(res["_status"], "success")
        call = fake.messages.calls[0]
        self.assertNotIn("temperature", call)
        self.assertEqual(call["thinking"], {"type": "between_tools"})


class DataRootTests(unittest.TestCase):
    def test_default_root_is_the_project_folder_not_the_working_directory(self):
        import os
        from unittest import mock
        from data_ingestion import storage
        project = os.path.dirname(os.path.dirname(os.path.abspath(storage.__file__)))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RESEARCH_DATA_ROOT", None)
            here = os.getcwd()
            with tempfile.TemporaryDirectory() as elsewhere:
                os.chdir(elsewhere)                          # e.g. started from VS Code's folder
                try:
                    self.assertEqual(storage.default_data_root(), os.path.join(project, "data_lake"))
                finally:
                    os.chdir(here)                           # leave before the folder is deleted (Windows
                                                             # cannot delete the current directory)
        with mock.patch.dict(os.environ, {"RESEARCH_DATA_ROOT": "/tmp/custom_root"}):
            self.assertEqual(storage.default_data_root(), "/tmp/custom_root")


if __name__ == '__main__':
    unittest.main()
