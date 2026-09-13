"""Collectors for free Treasury, FRED, and New York Fed macro data."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Iterable

from .http import fetch_bytes
from .storage import DataStore, utc_now


TREASURY_DATASETS = {
    "nominal": "daily_treasury_yield_curve",
    "real": "daily_treasury_real_yield_curve",
    "bills": "daily_treasury_bill_rates",
}

DEFAULT_FRED_SERIES = (
    "BAMLH0A0HYM2",  # ICE BofA US High Yield option-adjusted spread
    "BAMLC0A0CM",    # ICE BofA US Corporate option-adjusted spread
    "BAA10Y",        # long-history Moody's Baa minus 10-year Treasury spread
    "NFCI",          # Chicago Fed National Financial Conditions Index
    "STLFSI4",       # St. Louis Fed Financial Stress Index
    "T10Y2Y",        # 10-year minus 2-year Treasury spread
    "T10Y3M",        # 10-year minus 3-month Treasury spread
    "T10YIE",        # 10-year breakeven inflation
    "DFII10",        # 10-year real Treasury yield
    "SOFR",          # secured overnight financing rate
    "DFF",           # effective federal funds rate
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", ".", "NA", "N/A", "null", "None"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _iso_date(value: str) -> str:
    text = value.strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date: {value}")


def parse_treasury_xml(
    content: bytes,
    *,
    dataset_name: str,
    raw_object_id: int | None = None,
) -> list[dict[str, Any]]:
    """Parse Treasury's Atom/OData XML without depending on namespace prefixes."""
    root = ET.fromstring(content)
    observations: list[dict[str, Any]] = []
    retrieved_at = utc_now()

    for entry in root.iter():
        if _local_name(entry.tag) != "entry":
            continue
        properties = next(
            (node for node in entry.iter() if _local_name(node.tag) == "properties"),
            None,
        )
        if properties is None:
            continue
        values = {_local_name(child.tag): (child.text or "").strip() for child in properties}
        date_text = values.get("NEW_DATE") or values.get("Record_Date") or values.get("record_date")
        if not date_text:
            continue
        observation_date = _iso_date(date_text)

        for field, raw_value in values.items():
            field_upper = field.upper()
            if (
                field in {"NEW_DATE", "Record_Date", "record_date", "Id", "INDEX_DATE"}
                or field_upper.endswith("ID")
                or field_upper.endswith("DISPLAY")
            ):
                continue
            numeric = _as_float(raw_value)
            if numeric is None:
                continue
            observations.append(
                {
                    "source": "treasury",
                    "series_id": f"{dataset_name}:{field}",
                    "observation_date": observation_date,
                    "value": numeric,
                    "realtime_start": "",
                    "realtime_end": "",
                    "retrieved_at": retrieved_at,
                    "raw_object_id": raw_object_id,
                    "metadata_json": json.dumps({"field": field}, sort_keys=True),
                }
            )
    return observations


def parse_fred_csv(
    content: bytes,
    *,
    series_id: str,
    raw_object_id: int | None = None,
) -> list[dict[str, Any]]:
    text = content.decode("utf-8-sig")
    rows = csv.DictReader(io.StringIO(text))
    retrieved_at = utc_now()
    observations: list[dict[str, Any]] = []
    for row in rows:
        date_text = row.get("DATE") or row.get("observation_date")
        value = row.get(series_id)
        if date_text is None and row:
            date_text = next(iter(row.values()))
        if value is None and len(row) >= 2:
            value = list(row.values())[1]
        numeric = _as_float(value)
        if not date_text or numeric is None:
            continue
        observations.append(
            {
                "source": "fred",
                "series_id": series_id,
                "observation_date": _iso_date(date_text),
                "value": numeric,
                "realtime_start": "",
                "realtime_end": "",
                "retrieved_at": retrieved_at,
                "raw_object_id": raw_object_id,
                "metadata_json": "{}",
            }
        )
    return observations


def parse_nyfed_rates_json(
    content: bytes,
    *,
    raw_object_id: int | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(content)
    candidates: Any = payload
    if isinstance(payload, dict):
        for key in ("refRates", "rates", "results", "data"):
            if isinstance(payload.get(key), list):
                candidates = payload[key]
                break
    if not isinstance(candidates, list):
        return []

    retrieved_at = utc_now()
    observations: list[dict[str, Any]] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        date_text = row.get("effectiveDate") or row.get("effective_date") or row.get("date")
        rate_type = row.get("type") or row.get("rateType") or row.get("rate_name") or "SOFR"
        numeric = _as_float(row.get("percentRate", row.get("rate", row.get("value"))))
        if not date_text or numeric is None:
            continue
        observations.append(
            {
                "source": "nyfed",
                "series_id": str(rate_type).upper(),
                "observation_date": _iso_date(str(date_text)),
                "value": numeric,
                "realtime_start": "",
                "realtime_end": "",
                "retrieved_at": retrieved_at,
                "raw_object_id": raw_object_id,
                "metadata_json": json.dumps(row, sort_keys=True, default=str),
            }
        )
    return observations


class MacroCollector:
    def __init__(self, store: DataStore):
        self.store = store

    def collect_treasury_year(self, year: int, dataset: str = "nominal") -> int:
        data_key = TREASURY_DATASETS[dataset]
        query = urllib.parse.urlencode({"data": data_key, "field_tdr_date_value": str(year)})
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?{query}"
        content, headers = fetch_bytes(url, headers={"Accept": "application/xml"})
        raw_id = self.store.save_raw(
            source="treasury",
            dataset=data_key,
            content=content,
            extension="xml",
            source_url=url,
            content_type=headers.get("content-type"),
            metadata={"year": year},
        )
        rows = parse_treasury_xml(content, dataset_name=data_key, raw_object_id=raw_id)
        count = self.store.upsert_macro(rows)
        self.store.set_checkpoint("treasury", data_key, "last_year", str(year))
        return count

    def collect_treasury_range(
        self,
        start_year: int,
        end_year: int | None = None,
        datasets: Iterable[str] = ("nominal", "real"),
    ) -> int:
        end_year = end_year or dt.date.today().year
        total = 0
        for dataset in datasets:
            for year in range(start_year, end_year + 1):
                total += self.collect_treasury_year(year, dataset)
        return total

    def collect_fred_series(self, series_id: str) -> int:
        query = urllib.parse.urlencode({"id": series_id})
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
        content, headers = fetch_bytes(url, headers={"Accept": "text/csv"})
        raw_id = self.store.save_raw(
            source="fred",
            dataset=series_id,
            content=content,
            extension="csv",
            source_url=url,
            content_type=headers.get("content-type"),
        )
        rows = parse_fred_csv(content, series_id=series_id, raw_object_id=raw_id)
        count = self.store.upsert_macro(rows)
        if rows:
            self.store.set_checkpoint("fred", series_id, "last_observation", rows[-1]["observation_date"])
        return count

    def collect_fred(self, series_ids: Iterable[str] = DEFAULT_FRED_SERIES) -> int:
        return sum(self.collect_fred_series(series_id) for series_id in series_ids)

    def collect_nyfed_sofr(self, start_date: str = "2018-04-03", end_date: str | None = None) -> int:
        end_date = end_date or dt.date.today().isoformat()
        query = urllib.parse.urlencode(
            {"startDate": start_date, "endDate": end_date, "type": "rate"}
        )
        url = f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?{query}"
        content, headers = fetch_bytes(url, headers={"Accept": "application/json"})
        raw_id = self.store.save_raw(
            source="nyfed",
            dataset="sofr",
            content=content,
            extension="json",
            source_url=url,
            content_type=headers.get("content-type"),
            metadata={"start_date": start_date, "end_date": end_date},
        )
        rows = parse_nyfed_rates_json(content, raw_object_id=raw_id)
        count = self.store.upsert_macro(rows)
        if rows:
            latest = max(row["observation_date"] for row in rows)
            self.store.set_checkpoint("nyfed", "sofr", "last_observation", latest)
        return count
