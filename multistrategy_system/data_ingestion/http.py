"""Small dependency-free HTTP helper used by public-data collectors."""

from __future__ import annotations

import time
import urllib.error
import urllib.request


DEFAULT_USER_AGENT = "ibkr-systematic-research/1.0 (personal research)"


def fetch_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 45,
    retries: int = 3,
) -> tuple[bytes, dict[str, str]]:
    """Fetch a URL with bounded exponential retry.

    Returns the response body and a lower-cased response-header mapping.  HTTP
    4xx responses other than 429 are not retried.
    """
    request_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "*/*"}
    request_headers.update(headers or {})
    last_error: Exception | None = None

    for attempt in range(retries):
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                return body, response_headers
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 and 400 <= exc.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 4))

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")
