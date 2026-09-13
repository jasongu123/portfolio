"""Paper-only technology tilt using the existing equity signal engine.

The sleeve is deliberately an allocation/risk boundary, not a claim of new
alpha.  It reuses ``EnhancedTradingStrategy`` for signals while keeping a
disjoint universe, a small gross budget, per-name caps, and separate execution
cost attribution.  It shares the equity IBKR connection so the legacy market-
data callbacks remain serialized; no symbol may be owned by both sleeves.

The sleeve is disabled unless ``TECH_SLEEVE_ENABLED=true`` and ``ibkr2.py`` is
in paper mode.  Research must be run with ``tech_sleeve_research.py`` before
enabling it on the VM.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable


TECH_SLEEVE_NAME = "tech"
TECH_SYMBOLS = ("INTC", "AMD", "AVGO", "QCOM", "MU")


class TechSleeve:
    """Route a disjoint tech universe through the existing stock strategy."""

    def __init__(
        self,
        strategy,
        risk_manager,
        *,
        enabled: bool = False,
        paper_mode: bool = True,
        symbols: Iterable[str] = TECH_SYMBOLS,
        core_symbols: Iterable[str] = (),
        sizing_multiplier: float = 0.10,
        max_position_pct: float = 0.02,
        gross_cap_pct: float = 0.10,
        require_rf_models: bool = True,
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
        self.sizing_multiplier = float(sizing_multiplier)
        self.max_position_pct = float(max_position_pct)
        self.gross_cap_pct = float(gross_cap_pct)
        self.require_rf_models = bool(require_rf_models)
        self._running = False
        self._last_missing_models: tuple[str, ...] | None = None
        self.last_check: dt.datetime | None = None
        self.last_signals: dict[str, str] = {}

        if not self.symbols or any(not symbol for symbol in self.symbols):
            raise ValueError("tech sleeve requires at least one valid symbol")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("tech sleeve symbols must be unique")
        overlap = set(self.symbols) & {
            str(symbol).strip().upper() for symbol in core_symbols
        }
        if overlap:
            raise ValueError(
                "tech sleeve must be disjoint from the equity sleeve; overlap="
                + ",".join(sorted(overlap))
            )
        for name, value in (
            ("sizing_multiplier", self.sizing_multiplier),
            ("max_position_pct", self.max_position_pct),
            ("gross_cap_pct", self.gross_cap_pct),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.sizing_multiplier > 1:
            raise ValueError("sizing_multiplier must not exceed 1")
        if self.max_position_pct > 1 or self.gross_cap_pct > 1:
            raise ValueError("tech sleeve percentage caps must not exceed 1")

        requested = bool(enabled)
        self.enabled = requested and bool(paper_mode)
        if requested and not paper_mode:
            print("[tech] requested outside paper mode — sleeve DISABLED")
        elif self.enabled:
            print(
                f"[tech] paper sleeve enabled: {', '.join(self.symbols)}; "
                f"gross cap {self.gross_cap_pct:.1%}, "
                f"per-name cap {self.max_position_pct:.1%}"
            )
        else:
            print("[tech] sleeve disabled (set TECH_SLEEVE_ENABLED=true after validation)")

    @property
    def position_symbols(self) -> set[str]:
        return set(self.symbols) if self.enabled else set()

    def _account_value(self) -> float:
        try:
            value = float(self.risk_manager.get_account_value())
            return value if value > 0 else 0.0
        except Exception as exc:
            print(f"[tech] account value unavailable ({exc}); entries disabled this cycle")
            return 0.0

    def _last_price(self, symbol: str) -> float:
        data = self.strategy.symbol_data[symbol]
        intraday = data.get("intraday_data")
        try:
            if intraday is not None and not intraday.empty:
                price = float(intraday.iloc[-1]["close"])
                if price > 0:
                    return price
        except Exception:
            pass
        for field in ("entry_price", "last_entry_price"):
            try:
                price = float(data.get(field) or 0.0)
                if price > 0:
                    return price
            except (TypeError, ValueError):
                pass
        return 0.0

    def _estimated_gross(self) -> float:
        gross = 0.0
        for symbol in self.symbols:
            data = self.strategy.symbol_data[symbol]
            quantity = abs(float(data.get("current_position", 0) or 0))
            gross += quantity * self._last_price(symbol)
        return gross

    def maybe_check(self, now: dt.datetime) -> int:
        """Check tech signals once from the main serialized trading loop.

        Returns the number of order requests handed to the shared strategy.
        All failures are isolated so this sleeve cannot stop the core engine.
        """

        if not self.enabled or self._running:
            return 0

        if self.require_rf_models:
            use_rf = getattr(self.strategy, "use_rf_signals", {})
            models = getattr(self.strategy, "rf_models", {})
            missing_models = tuple(
                symbol
                for symbol in self.symbols
                if not use_rf.get(symbol, False) or symbol not in models
            )
            if missing_models:
                if missing_models != self._last_missing_models:
                    print(
                        "[tech] waiting for per-symbol RF models: "
                        + ", ".join(missing_models)
                    )
                    self._last_missing_models = missing_models
                return 0
            self._last_missing_models = None

        self._running = True
        requested_orders = 0
        try:
            self.strategy.update_allocations(
                self.symbols,
                allocation_group=TECH_SLEEVE_NAME,
            )
            account_value = self._account_value()
            gross_limit = account_value * self.gross_cap_pct
            per_name_limit = account_value * min(
                self.max_position_pct,
                self.gross_cap_pct / len(self.symbols),
            )
            reserved_gross = self._estimated_gross()

            for symbol in self.symbols:
                try:
                    signal = self.strategy.generate_signal(symbol)
                    self.last_signals[symbol] = signal
                    if signal == "HOLD":
                        continue

                    current_position = int(
                        self.strategy.symbol_data[symbol].get(
                            "current_position", 0
                        ) or 0
                    )
                    max_entry_notional = None
                    if current_position == 0:
                        remaining = max(0.0, gross_limit - reserved_gross)
                        max_entry_notional = min(per_name_limit, remaining)
                        if max_entry_notional <= 0:
                            print(
                                f"[tech] RISK BLOCK [{symbol}]: "
                                "tech gross budget exhausted"
                            )
                            continue

                    order_id = self.strategy.execute_trade(
                        symbol,
                        signal,
                        position_size_multiplier=self.sizing_multiplier,
                        max_entry_notional=max_entry_notional,
                        journal_sleeve=TECH_SLEEVE_NAME,
                    )
                    if order_id is not None:
                        requested_orders += 1
                        if current_position == 0 and max_entry_notional is not None:
                            # Reserve the full cap for this cycle.  This is
                            # conservative when the actual order is smaller.
                            reserved_gross += max_entry_notional
                except Exception as exc:
                    print(f"[tech] {symbol} isolated error: {exc}")

            self.last_check = now
            return requested_orders
        except Exception as exc:
            print(f"[tech] cycle error (core equity unaffected): {exc}")
            return requested_orders
        finally:
            self._running = False
