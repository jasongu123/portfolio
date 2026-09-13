"""Helpers for formatting Interactive Brokers historical-data durations."""

from __future__ import annotations


def duration_from_days(days: int) -> str:
    """Return an IBKR duration string, using years beyond the 365-day limit."""
    days = int(days)
    if days < 1:
        raise ValueError("days must be at least 1")
    if days > 365:
        years = (days + 364) // 365
        return f"{years} Y"
    return f"{days} D"
