"""Pure policy and position helpers for the premarket news paper sleeve."""

from __future__ import annotations

from typing import Any


def evaluate_candidate(
    symbol: str,
    assessment: dict[str, Any],
    context: dict[str, Any],
    *,
    min_relevance: float = 0.65,
    min_confidence: float = 0.65,
    max_spread_pct: float = 0.01,
    contradictory_gap_pct: float = 0.005,
) -> tuple[str, str]:
    """Map a model classification into a deterministic paper decision."""
    if assessment.get("_status") != "success":
        return "HOLD", f"assessment status={assessment.get('_status')}"
    direction = assessment.get("direction")
    severity = int(assessment.get("severity", 0) or 0)
    novelty = assessment.get("novelty")
    relevance = float(assessment.get("relevance", 0) or 0)
    confidence = float(assessment.get("confidence", 0) or 0)
    scope = assessment.get("scope")
    if direction not in {"bullish", "bearish"}:
        return "HOLD", f"direction={direction}"
    if severity < 3:
        return "HOLD", f"severity={severity}; require 3"
    if novelty not in {"new", "update"}:
        return "HOLD", f"novelty={novelty}"
    if relevance < min_relevance:
        return "HOLD", f"relevance={relevance:.2f} below {min_relevance:.2f}"
    if confidence < min_confidence:
        return "HOLD", f"confidence={confidence:.2f} below {min_confidence:.2f}"
    if scope == "market" and symbol not in {"SPY", "QQQ", "IWM"}:
        return "HOLD", "market-wide catalyst is expressed only through index ETFs"
    if context.get("quote_mid") is None or context.get("is_synthetic_quote"):
        return "HOLD", "no executable premarket bid/ask quote"
    spread_pct = context.get("spread_pct")
    if spread_pct is None or spread_pct > max_spread_pct:
        spread_text = "unknown" if spread_pct is None else f"{spread_pct:.2%}"
        return "HOLD", f"spread={spread_text}; max {max_spread_pct:.2%}"
    gap_pct = context.get("gap_pct")
    sign = 1 if direction == "bullish" else -1
    if gap_pct is not None and sign * gap_pct <= -contradictory_gap_pct:
        return "HOLD", f"premarket gap {gap_pct:+.2%} contradicts {direction} event"
    action = "BUY" if direction == "bullish" else "SELL"
    gap_text = "unknown" if gap_pct is None else f"{gap_pct:+.2%}"
    return action, f"material {assessment.get('event_type')} event; gap={gap_text}"


def directional_return(entry_price: float, current_price: float, position: int) -> float:
    if position == 0:
        raise ValueError("position must be non-zero")
    sign = 1 if position > 0 else -1
    return ((current_price / entry_price) - 1.0) * sign


def closing_action(position: int) -> str:
    if position > 0:
        return "SELL"
    if position < 0:
        return "BUY"
    raise ValueError("cannot close a flat position")


def bounded_limit_price(direction: str, quote: dict[str, Any], buffer_pct: float) -> float | None:
    mid = quote.get("mid")
    bid = quote.get("bid")
    ask = quote.get("ask")
    if direction == "BUY":
        reference = ask or mid or bid
        if reference is None:
            return None
        cap = mid * (1 + buffer_pct) if mid else reference
        return round(min(reference, cap), 2)
    if direction == "SELL":
        reference = bid or mid or ask
        if reference is None:
            return None
        floor = mid * (1 - buffer_pct) if mid else reference
        return round(max(reference, floor), 2)
    raise ValueError(f"unsupported direction: {direction}")
