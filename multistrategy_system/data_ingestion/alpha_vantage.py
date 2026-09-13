"""Alpha Vantage collectors for free earnings and point-in-time listings."""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.parse
from typing import Any, Iterable

from .http import fetch_bytes
from .storage import DataStore, utc_now


API_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(RuntimeError):
    pass


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, "", "None", "null"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_payload(payload: dict[str, Any]) -> None:
    for key in ("Error Message", "Information", "Note"):
        if payload.get(key):
            raise AlphaVantageError(str(payload[key]))


class AlphaVantageCollector:
    def __init__(
        self,
        store: DataStore,
        api_key: str | None = None,
        request_pause_seconds: float = 1.0,
    ):
        self.store = store
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise AlphaVantageError(
                "Set ALPHAVANTAGE_API_KEY. A free key is available from alphavantage.co."
            )
        self.request_pause_seconds = max(0.0, request_pause_seconds)

    def _url(self, **params: str) -> str:
        safe_params = dict(params)
        safe_params["apikey"] = self.api_key
        return f"{API_URL}?{urllib.parse.urlencode(safe_params)}"

    @staticmethod
    def _redacted_url(url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        redacted = [(key, "REDACTED" if key.lower() == "apikey" else value) for key, value in query]
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(redacted), parsed.fragment)
        )

    def collect_earnings(self, symbol: str) -> int:
        symbol = symbol.strip().upper()
        url = self._url(function="EARNINGS", symbol=symbol)
        content, headers = fetch_bytes(url, headers={"Accept": "application/json"})
        payload = json.loads(content)
        _validate_payload(payload)
        raw_id = self.store.save_raw(
            source="alpha_vantage",
            dataset="earnings",
            content=content,
            extension="json",
            source_url=self._redacted_url(url),
            content_type=headers.get("content-type"),
            metadata={"symbol": symbol},
        )

        rows: list[dict[str, Any]] = []
        for period_type, key in (("quarterly", "quarterlyEarnings"), ("annual", "annualEarnings")):
            for item in payload.get(key, []):
                fiscal_date = str(item.get("fiscalDateEnding") or "")
                if not fiscal_date:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "period_type": period_type,
                        "fiscal_date_ending": fiscal_date,
                        "reported_date": str(item.get("reportedDate") or ""),
                        "report_time": str(item.get("reportTime") or ""),
                        "reported_eps": _float_or_none(item.get("reportedEPS")),
                        "estimated_eps": _float_or_none(item.get("estimatedEPS")),
                        "surprise": _float_or_none(item.get("surprise")),
                        "surprise_percentage": _float_or_none(item.get("surprisePercentage")),
                        "retrieved_at": utc_now(),
                        "raw_object_id": raw_id,
                        "raw_json": json.dumps(item, sort_keys=True),
                    }
                )

        with self.store.connect() as conn:
            conn.executemany(
                """
                INSERT INTO alpha_earnings
                    (symbol, period_type, fiscal_date_ending, reported_date, report_time,
                     reported_eps, estimated_eps, surprise, surprise_percentage,
                     retrieved_at, raw_object_id, raw_json)
                VALUES (:symbol, :period_type, :fiscal_date_ending, :reported_date,
                        :report_time, :reported_eps, :estimated_eps, :surprise,
                        :surprise_percentage, :retrieved_at, :raw_object_id, :raw_json)
                ON CONFLICT(symbol, period_type, fiscal_date_ending, reported_date)
                DO UPDATE SET
                    report_time=excluded.report_time,
                    reported_eps=excluded.reported_eps,
                    estimated_eps=excluded.estimated_eps,
                    surprise=excluded.surprise,
                    surprise_percentage=excluded.surprise_percentage,
                    retrieved_at=excluded.retrieved_at,
                    raw_object_id=excluded.raw_object_id,
                    raw_json=excluded.raw_json
                """,
                rows,
            )
        if rows:
            self.store.set_checkpoint("alpha_vantage", "earnings", symbol, utc_now())
        return len(rows)

    def collect_earnings_many(self, symbols: Iterable[str], max_calls: int = 25) -> dict[str, int]:
        results: dict[str, int] = {}
        for index, symbol in enumerate(symbols):
            if index >= max_calls:
                break
            clean = symbol.strip().upper()
            if not clean:
                continue
            results[clean] = self.collect_earnings(clean)
            if index + 1 < max_calls and self.request_pause_seconds:
                time.sleep(self.request_pause_seconds)
        return results

    def collect_listings(self, as_of_date: str, state: str = "active") -> int:
        if state not in {"active", "delisted"}:
            raise ValueError("state must be 'active' or 'delisted'")
        url = self._url(function="LISTING_STATUS", date=as_of_date, state=state)
        content, headers = fetch_bytes(url, headers={"Accept": "text/csv"})
        text = content.decode("utf-8-sig")
        if text.lstrip().startswith("{"):
            payload = json.loads(text)
            _validate_payload(payload)
            raise AlphaVantageError(f"Expected CSV listing data, received: {payload}")

        raw_id = self.store.save_raw(
            source="alpha_vantage",
            dataset="listing_status",
            content=content,
            extension="csv",
            source_url=self._redacted_url(url),
            content_type=headers.get("content-type"),
            metadata={"as_of_date": as_of_date, "state": state},
        )
        normalized = []
        for item in csv.DictReader(io.StringIO(text)):
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            normalized.append(
                {
                    "as_of_date": as_of_date,
                    "state": state,
                    "symbol": symbol,
                    "name": item.get("name"),
                    "exchange": item.get("exchange"),
                    "asset_type": item.get("assetType"),
                    "ipo_date": item.get("ipoDate"),
                    "delisting_date": item.get("delistingDate"),
                    "status": item.get("status"),
                    "retrieved_at": utc_now(),
                    "raw_object_id": raw_id,
                    "raw_json": json.dumps(item, sort_keys=True),
                }
            )

        with self.store.connect() as conn:
            conn.executemany(
                """
                INSERT INTO alpha_listings
                    (as_of_date, state, symbol, name, exchange, asset_type, ipo_date,
                     delisting_date, status, retrieved_at, raw_object_id, raw_json)
                VALUES (:as_of_date, :state, :symbol, :name, :exchange, :asset_type,
                        :ipo_date, :delisting_date, :status, :retrieved_at,
                        :raw_object_id, :raw_json)
                ON CONFLICT(as_of_date, state, symbol) DO UPDATE SET
                    name=excluded.name,
                    exchange=excluded.exchange,
                    asset_type=excluded.asset_type,
                    ipo_date=excluded.ipo_date,
                    delisting_date=excluded.delisting_date,
                    status=excluded.status,
                    retrieved_at=excluded.retrieved_at,
                    raw_object_id=excluded.raw_object_id,
                    raw_json=excluded.raw_json
                """,
                normalized,
            )
        self.store.set_checkpoint(
            "alpha_vantage", "listing_status", f"{as_of_date}:{state}", utc_now()
        )
        return len(normalized)
