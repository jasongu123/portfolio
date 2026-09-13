"""Standalone Interactive Brokers historical-news archiver.

This client intentionally uses its own API connection and client id.  It does
not import or mutate the live strategy in ``ibkr2.py``.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from typing import Any, Iterable

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

from .storage import DataStore


BENIGN_ERROR_CODES = {2104, 2106, 2107, 2108, 2158}


def parse_ibkr_news_time(value: str) -> str | None:
    """Normalize the date formats observed in ``historicalNews`` callbacks."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit() and len(text) >= 10:
        try:
            return dt.datetime.fromtimestamp(int(text), tz=dt.timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    for fmt in (
        "%Y%m%d %H:%M:%S",
        "%Y%m%d-%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            parsed = dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
            return parsed.isoformat()
        except ValueError:
            continue
    return text


class IBKRNewsError(RuntimeError):
    pass


class IBKRNewsCollector(EWrapper, EClient):
    """Blocking facade around IBKR's asynchronous news API."""

    def __init__(
        self,
        store: DataStore,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 91,
    ):
        EClient.__init__(self, self)
        self.store = store
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._id_lock = threading.Lock()
        self._next_request_id = int(time.time()) % 1_000_000 + 100_000
        self._pending_lock = threading.RLock()
        self._contracts: dict[int, dict[str, Any]] = {}
        self._headlines: dict[int, dict[str, Any]] = {}
        self._articles: dict[int, dict[str, Any]] = {}
        self._providers_event = threading.Event()
        self._providers: list[dict[str, str]] = []
        self._contract_cache: dict[str, int] = {}
        self.errors: list[str] = []

    def nextValidId(self, orderId: int) -> None:  # noqa: N802 - IB API callback
        self._ready.set()

    def error(self, req_id: int, *args: Any) -> None:
        """Handle both older and newer ibapi callback signatures."""
        if len(args) >= 3 and isinstance(args[1], (int, float)):
            code, message = int(args[1]), str(args[2])
        elif len(args) >= 2:
            code, message = int(args[0]), str(args[1])
        else:
            code, message = -1, " ".join(str(arg) for arg in args)
        if code in BENIGN_ERROR_CODES:
            return
        formatted = f"IBKR error {code}: {message}"
        self.errors.append(formatted)
        with self._pending_lock:
            for collection in (self._contracts, self._headlines, self._articles):
                pending = collection.get(req_id)
                if pending is not None:
                    pending["error"] = formatted
                    pending["event"].set()

    def newsProviders(self, providers: Any) -> None:  # noqa: N802
        self._providers = [
            {"code": str(provider.code), "name": str(provider.name)}
            for provider in providers
            if getattr(provider, "code", None)
        ]
        self._providers_event.set()

    def contractDetails(self, reqId: int, contractDetails: Any) -> None:  # noqa: N802
        with self._pending_lock:
            pending = self._contracts.get(reqId)
            if pending is not None:
                pending["items"].append(contractDetails)

    def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
        with self._pending_lock:
            pending = self._contracts.get(reqId)
            if pending is not None:
                pending["event"].set()

    def historicalNews(  # noqa: N802
        self,
        requestId: int,
        time: str,
        providerCode: str,
        articleId: str,
        headline: str,
    ) -> None:
        with self._pending_lock:
            pending = self._headlines.get(requestId)
            if pending is not None:
                pending["items"].append(
                    {
                        "provider_code": providerCode,
                        "article_id": articleId,
                        "published_at": parse_ibkr_news_time(time),
                        "ibkr_time": time,
                        "headline": headline,
                        "symbols": [pending["symbol"]],
                    }
                )

    def historicalNewsEnd(self, requestId: int, hasMore: bool) -> None:  # noqa: N802
        with self._pending_lock:
            pending = self._headlines.get(requestId)
            if pending is not None:
                pending["has_more"] = bool(hasMore)
                pending["event"].set()

    def newsArticle(self, requestId: int, articleType: int, articleText: str) -> None:  # noqa: N802
        with self._pending_lock:
            pending = self._articles.get(requestId)
            if pending is not None:
                pending["article_type"] = int(articleType)
                pending["body"] = articleText
                pending["event"].set()

    def _request_id(self) -> int:
        with self._id_lock:
            self._next_request_id += 1
            return self._next_request_id

    def connect_and_start(self, timeout: float = 10.0) -> None:
        if self.isConnected() and self._ready.is_set():
            return
        endpoint = f"{self.host}:{self.port}"
        self._ready.clear()
        self.connect(self.host, self.port, self.client_id)
        if not self.isConnected():
            raise IBKRNewsError(f"Could not connect to TWS/Gateway at {endpoint}")
        self._thread = threading.Thread(target=self.run, name="ibkr-news-api", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.disconnect()
            raise IBKRNewsError(
                f"Timed out connecting to TWS/Gateway at {endpoint}; "
                "check that API connections are enabled"
            )

    def close(self) -> None:
        if self.isConnected():
            self.disconnect()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_providers(self, timeout: float = 8.0, refresh: bool = False) -> list[dict[str, str]]:
        if self._providers and not refresh:
            return list(self._providers)
        self._providers_event.clear()
        self.reqNewsProviders()
        if not self._providers_event.wait(timeout):
            raise IBKRNewsError(
                "Timed out waiting for IBKR news providers; verify the account's news subscriptions"
            )
        raw = json.dumps(self._providers, sort_keys=True).encode()
        self.store.save_raw(
            source="ibkr",
            dataset="news_providers",
            content=raw,
            extension="json",
            metadata={"host": self.host, "port": self.port},
        )
        return list(self._providers)

    def resolve_stock(self, symbol: str, timeout: float = 8.0) -> int:
        symbol = symbol.strip().upper()
        cached = self._contract_cache.get(symbol)
        if cached:
            return cached
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        req_id = self._request_id()
        pending = {"event": threading.Event(), "items": [], "error": None}
        with self._pending_lock:
            self._contracts[req_id] = pending
        try:
            self.reqContractDetails(req_id, contract)
            if not pending["event"].wait(timeout):
                raise IBKRNewsError(f"Timed out resolving contract for {symbol}")
        finally:
            with self._pending_lock:
                self._contracts.pop(req_id, None)
        if pending["error"]:
            raise IBKRNewsError(str(pending["error"]))
        candidates = [
            detail for detail in pending["items"] if getattr(detail.contract, "conId", 0)
        ]
        if not candidates:
            raise IBKRNewsError(f"No SMART/USD stock contract found for {symbol}")
        con_id = int(candidates[0].contract.conId)
        self._contract_cache[symbol] = con_id
        return con_id

    def fetch_headlines(
        self,
        symbol: str,
        *,
        lookback_days: int = 3,
        max_results: int = 100,
        timeout: float = 20.0,
    ) -> list[dict[str, Any]]:
        symbol = symbol.strip().upper()
        con_id = self.resolve_stock(symbol, timeout=min(timeout, 10.0))
        providers = self.get_providers()
        provider_codes = "+".join(provider["code"] for provider in providers)
        if not provider_codes:
            raise IBKRNewsError("IBKR returned no subscribed news providers")
        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=max(1, lookback_days))
        req_id = self._request_id()
        pending = {
            "event": threading.Event(),
            "items": [],
            "has_more": False,
            "error": None,
            "symbol": symbol,
        }
        with self._pending_lock:
            self._headlines[req_id] = pending
        try:
            self.reqHistoricalNews(
                req_id,
                con_id,
                provider_codes,
                start.strftime("%Y%m%d-%H:%M:%S"),
                end.strftime("%Y%m%d-%H:%M:%S"),
                min(max(1, max_results), 300),
                [],
            )
            if not pending["event"].wait(timeout):
                raise IBKRNewsError(f"Timed out waiting for news for {symbol}")
        finally:
            with self._pending_lock:
                self._headlines.pop(req_id, None)
        if pending["error"]:
            raise IBKRNewsError(str(pending["error"]))
        items = list(pending["items"])
        raw_payload = {
            "symbol": symbol,
            "con_id": con_id,
            "provider_codes": provider_codes,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "has_more": pending["has_more"],
            "items": items,
        }
        self.store.save_raw(
            source="ibkr",
            dataset="historical_news",
            content=json.dumps(raw_payload, sort_keys=True).encode(),
            extension="json",
            metadata={"symbol": symbol, "has_more": pending["has_more"]},
        )
        self.store.upsert_news_headlines(items)
        self.store.set_checkpoint("ibkr", "historical_news", symbol, end.isoformat())
        return items

    def fetch_article_body(
        self,
        provider_code: str,
        article_id: str,
        timeout: float = 10.0,
    ) -> tuple[int | None, str | None]:
        req_id = self._request_id()
        pending = {
            "event": threading.Event(),
            "article_type": None,
            "body": None,
            "error": None,
        }
        with self._pending_lock:
            self._articles[req_id] = pending
        try:
            self.reqNewsArticle(req_id, provider_code, article_id, [])
            if not pending["event"].wait(timeout):
                raise IBKRNewsError(f"Timed out retrieving article {provider_code}/{article_id}")
        finally:
            with self._pending_lock:
                self._articles.pop(req_id, None)
        if pending["error"]:
            raise IBKRNewsError(str(pending["error"]))
        article_type = pending["article_type"]
        body = pending["body"]
        if body is not None:
            raw = json.dumps(
                {
                    "provider_code": provider_code,
                    "article_id": article_id,
                    "article_type": article_type,
                    "body": body,
                },
                sort_keys=True,
            ).encode()
            self.store.save_raw(
                source="ibkr",
                dataset="news_articles",
                content=raw,
                extension="json",
                metadata={"provider_code": provider_code, "article_id": article_id},
            )
            self.store.update_news_body(provider_code, article_id, body, article_type)
        return article_type, body

    def collect_symbols(
        self,
        symbols: Iterable[str],
        *,
        lookback_days: int = 3,
        max_results: int = 100,
        include_bodies: bool = True,
    ) -> dict[str, dict[str, int]]:
        results: dict[str, dict[str, int]] = {}
        for raw_symbol in symbols:
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue
            headlines = self.fetch_headlines(
                symbol, lookback_days=lookback_days, max_results=max_results
            )
            bodies = 0
            body_errors = 0
            if include_bodies:
                seen: set[tuple[str, str]] = set()
                for item in headlines:
                    key = (str(item["provider_code"]), str(item["article_id"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        _, body = self.fetch_article_body(*key)
                        bodies += int(body is not None)
                    except IBKRNewsError:
                        body_errors += 1
            results[symbol] = {
                "headlines": len(headlines),
                "bodies": bodies,
                "body_errors": body_errors,
            }
        return results

    def __enter__(self) -> "IBKRNewsCollector":
        self.connect_and_start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
