"""Dependency-light transaction and holding cost estimates.

The execution model treats all costs as one-way dollar amounts.  Signed share
quantities are supported for compatibility with callers that use negative
shares for sells; costs are based on the absolute quantity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional


def _finite_number(name: str, value: float) -> float:
    """Return *value* as a finite float, raising ValueError otherwise."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _nonnegative_number(name: str, value: float) -> float:
    number = _finite_number(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


@dataclass(frozen=True)
class CostModelConfig:
    """Configuration for execution and holding cost estimates."""

    commission_per_share: float = 0.0035
    min_commission: float = 0.35
    fallback_execution_bps: float = 1.0
    extra_slippage_bps: float = 0.0
    impact_coefficient: float = 0.0
    annual_borrow_rate: float = 0.0
    annual_financing_rate: float = 0.0
    day_count: float = 360

    def __post_init__(self) -> None:
        for name in (
            "commission_per_share",
            "min_commission",
            "fallback_execution_bps",
            "extra_slippage_bps",
            "impact_coefficient",
            "annual_borrow_rate",
            "annual_financing_rate",
        ):
            value = _nonnegative_number(name, getattr(self, name))
            object.__setattr__(self, name, value)

        day_count = _finite_number("day_count", self.day_count)
        if day_count <= 0.0:
            raise ValueError("day_count must be positive")
        object.__setattr__(self, "day_count", day_count)


@dataclass(frozen=True)
class CostEstimate:
    """Dollar cost breakdown for one execution or holding period."""

    notional: float = 0.0
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0
    borrow: float = 0.0
    financing: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "notional",
            "commission",
            "spread",
            "slippage",
            "impact",
            "borrow",
            "financing",
        ):
            value = _nonnegative_number(name, getattr(self, name))
            object.__setattr__(self, name, value)

    @property
    def total(self) -> float:
        """Total dollar cost represented by this estimate."""

        return (
            self.commission
            + self.spread
            + self.slippage
            + self.impact
            + self.borrow
            + self.financing
        )

    @property
    def total_bps(self) -> float:
        """Total cost in basis points of the estimate's notional."""

        if self.notional == 0.0:
            return 0.0
        return self.total / self.notional * 10_000.0

    def as_dict(self) -> Dict[str, float]:
        """Return the full breakdown, including computed totals."""

        result = asdict(self)
        result["total"] = self.total
        result["total_bps"] = self.total_bps
        return result


def _config_or_default(config: Optional[CostModelConfig]) -> CostModelConfig:
    if config is None:
        return CostModelConfig()
    if not isinstance(config, CostModelConfig):
        raise TypeError("config must be a CostModelConfig or None")
    return config


def estimate_execution_cost(
    shares: float,
    price: float,
    config: Optional[CostModelConfig] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    daily_volatility: Optional[float] = None,
    dollar_adv: Optional[float] = None,
) -> CostEstimate:
    """Estimate one-way execution costs for a trade.

    A complete, uncrossed quote uses the quoted half-spread.  Missing or
    crossed quotes use ``fallback_execution_bps`` instead.  Market impact is
    included when both ``daily_volatility`` and ``dollar_adv`` are supplied:

    ``notional * impact_coefficient * daily_volatility * sqrt(notional / dollar_adv)``.
    """

    cfg = _config_or_default(config)
    share_count = _finite_number("shares", shares)
    execution_price = _nonnegative_number("price", price)

    quote_bid = None if bid is None else _nonnegative_number("bid", bid)
    quote_ask = None if ask is None else _nonnegative_number("ask", ask)
    volatility = (
        None
        if daily_volatility is None
        else _nonnegative_number("daily_volatility", daily_volatility)
    )
    adv = None if dollar_adv is None else _nonnegative_number("dollar_adv", dollar_adv)

    quantity = abs(share_count)
    if quantity == 0.0:
        return CostEstimate()

    notional = quantity * execution_price
    commission = max(quantity * cfg.commission_per_share, cfg.min_commission)

    quote_is_valid = (
        quote_bid is not None
        and quote_ask is not None
        and quote_bid > 0.0
        and quote_ask > 0.0
        and quote_ask >= quote_bid
    )
    if quote_is_valid:
        spread = quantity * (quote_ask - quote_bid) / 2.0
    else:
        spread = notional * cfg.fallback_execution_bps / 10_000.0

    slippage = notional * cfg.extra_slippage_bps / 10_000.0

    impact = 0.0
    if volatility is not None and adv is not None and cfg.impact_coefficient > 0.0:
        if adv <= 0.0:
            raise ValueError("dollar_adv must be positive when estimating impact")
        impact = (
            notional
            * cfg.impact_coefficient
            * volatility
            * math.sqrt(notional / adv)
        )

    return CostEstimate(
        notional=notional,
        commission=commission,
        spread=spread,
        slippage=slippage,
        impact=impact,
    )


def estimate_holding_cost(
    short_notional: float = 0.0,
    borrowed_cash: float = 0.0,
    holding_days: float = 1,
    config: Optional[CostModelConfig] = None,
) -> CostEstimate:
    """Estimate simple borrow and financing costs over a holding period."""

    cfg = _config_or_default(config)
    short_value = _nonnegative_number("short_notional", short_notional)
    cash_value = _nonnegative_number("borrowed_cash", borrowed_cash)
    days = _nonnegative_number("holding_days", holding_days)

    fraction_of_year = days / cfg.day_count
    borrow = short_value * cfg.annual_borrow_rate * fraction_of_year
    financing = cash_value * cfg.annual_financing_rate * fraction_of_year

    return CostEstimate(
        notional=short_value + cash_value,
        borrow=borrow,
        financing=financing,
    )


__all__ = [
    "CostEstimate",
    "CostModelConfig",
    "estimate_execution_cost",
    "estimate_holding_cost",
]
