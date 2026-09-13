from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_ingestion.alpha_vantage import AlphaVantageCollector
from data_ingestion.ibkr_news import parse_ibkr_news_time
from data_ingestion.macro import (
    parse_fred_csv,
    parse_nyfed_rates_json,
    parse_treasury_xml,
)
from data_ingestion.storage import DataStore


class DataStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = DataStore(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_raw_objects_are_content_addressed_and_deduplicated(self) -> None:
        first = self.store.save_raw(
            source="test", dataset="sample", content=b"same", extension="txt"
        )
        second = self.store.save_raw(
            source="test", dataset="sample", content=b"same", extension="txt"
        )
        self.assertEqual(first, second)
        with self.store.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM raw_objects").fetchone()
            relative = conn.execute("SELECT relative_path FROM raw_objects").fetchone()[0]
        self.assertEqual(row["count"], 1)
        self.assertTrue((Path(self.temp.name) / relative).is_file())

    def test_news_headline_symbols_merge_and_body_updates(self) -> None:
        common = {
            "provider_code": "BRFG",
            "article_id": "abc123",
            "published_at": "2026-07-12T12:00:00+00:00",
            "headline": "Example headline",
        }
        self.store.upsert_news_headlines([{**common, "symbols": ["AAPL"]}])
        self.store.upsert_news_headlines([{**common, "symbols": ["MSFT"]}])
        self.store.update_news_body("BRFG", "abc123", "Full article", 0)
        with self.store.connect() as conn:
            article = conn.execute(
                "SELECT body FROM news_articles WHERE provider_code='BRFG' AND article_id='abc123'"
            ).fetchone()
            symbols = conn.execute(
                "SELECT symbol FROM news_article_symbols ORDER BY symbol"
            ).fetchall()
        self.assertEqual(article["body"], "Full article")
        self.assertEqual([row["symbol"] for row in symbols], ["AAPL", "MSFT"])


class MacroParserTests(unittest.TestCase):
    def test_treasury_atom_xml(self) -> None:
        content = b"""<?xml version='1.0' encoding='utf-8'?>
        <feed xmlns='http://www.w3.org/2005/Atom'
              xmlns:m='http://schemas.microsoft.com/ado/2007/08/dataservices/metadata'
              xmlns:d='http://schemas.microsoft.com/ado/2007/08/dataservices'>
          <entry><content><m:properties>
            <d:NEW_DATE>2026-07-10T00:00:00</d:NEW_DATE>
            <d:DailyTreasuryYieldCurveRateDataId>12345</d:DailyTreasuryYieldCurveRateDataId>
            <d:BC_1MONTH>4.31</d:BC_1MONTH>
            <d:BC_10YEAR>4.18</d:BC_10YEAR>
            <d:BC_30YEARDISPLAY>4.72</d:BC_30YEARDISPLAY>
            <d:BC_30YEAR m:null='true'></d:BC_30YEAR>
          </m:properties></content></entry>
        </feed>"""
        rows = parse_treasury_xml(content, dataset_name="daily_treasury_yield_curve")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["observation_date"], "2026-07-10")
        self.assertEqual({row["value"] for row in rows}, {4.31, 4.18})

    def test_fred_csv_skips_missing_values(self) -> None:
        content = b"observation_date,BAMLH0A0HYM2\n2026-07-09,2.91\n2026-07-10,.\n"
        rows = parse_fred_csv(content, series_id="BAMLH0A0HYM2")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], 2.91)

    def test_nyfed_json(self) -> None:
        content = json.dumps(
            {
                "refRates": [
                    {
                        "effectiveDate": "2026-07-10",
                        "type": "SOFR",
                        "percentRate": 3.78,
                        "volumeInBillions": 2140,
                    }
                ]
            }
        ).encode()
        rows = parse_nyfed_rates_json(content)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["series_id"], "SOFR")
        self.assertEqual(rows[0]["value"], 3.78)


class AlphaVantageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = DataStore(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @mock.patch("data_ingestion.alpha_vantage.fetch_bytes")
    def test_earnings_response_is_normalized_and_key_is_redacted(self, fetch: mock.Mock) -> None:
        fetch.return_value = (
            json.dumps(
                {
                    "symbol": "AAPL",
                    "annualEarnings": [{"fiscalDateEnding": "2025-09-27", "reportedEPS": "7.46"}],
                    "quarterlyEarnings": [
                        {
                            "fiscalDateEnding": "2026-06-27",
                            "reportedDate": "2026-07-30",
                            "reportedEPS": "1.55",
                            "estimatedEPS": "1.48",
                            "surprise": "0.07",
                            "surprisePercentage": "4.7297",
                        }
                    ],
                }
            ).encode(),
            {"content-type": "application/json"},
        )
        collector = AlphaVantageCollector(self.store, api_key="top-secret", request_pause_seconds=0)
        self.assertEqual(collector.collect_earnings("aapl"), 2)
        with self.store.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM alpha_earnings").fetchone()[0]
            source_url = conn.execute("SELECT source_url FROM raw_objects").fetchone()[0]
        self.assertEqual(count, 2)
        self.assertNotIn("top-secret", source_url)
        self.assertIn("REDACTED", source_url)

    @mock.patch("data_ingestion.alpha_vantage.fetch_bytes")
    def test_listing_response_is_normalized(self, fetch: mock.Mock) -> None:
        fetch.return_value = (
            b"symbol,name,exchange,assetType,ipoDate,delistingDate,status\n"
            b"TEST,Test Corp,NYSE,Stock,2020-01-02,null,Active\n",
            {"content-type": "text/csv"},
        )
        collector = AlphaVantageCollector(self.store, api_key="test", request_pause_seconds=0)
        self.assertEqual(collector.collect_listings("2026-07-11", "active"), 1)
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM alpha_listings").fetchone()
        self.assertEqual(row["symbol"], "TEST")
        self.assertEqual(row["exchange"], "NYSE")


class IBKRHelpersTests(unittest.TestCase):
    def test_historical_news_time_formats(self) -> None:
        self.assertEqual(
            parse_ibkr_news_time("20260712 09:30:00"),
            "2026-07-12T09:30:00+00:00",
        )
        self.assertEqual(
            parse_ibkr_news_time("20260712-09:30:00"),
            "2026-07-12T09:30:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
