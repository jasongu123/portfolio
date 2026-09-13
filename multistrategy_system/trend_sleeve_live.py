"""
Live (paper) cross-asset TREND sleeve — v2 (3/6/12m ensemble), integrated into ibkr2.py.

How it plugs in: TradingSystem.__init__ creates one TrendSleeve, and the main loop
calls trend_sleeve.maybe_rebalance(now_ET) once per minute. The sleeve self-guards to
ONE rebalance per month, runs the heavy work in a BACKGROUND THREAD (never stalls the
60s equity loop), and owns its OWN isolated IBKR connection (TREND_CLIENT_ID) for all
data / position / order traffic — so nothing it does can touch the equity client's
callbacks, order-id, or event loop. One process, fully integrated, firewalled.

DISJOINT UNIVERSE (critical): drops SPY/IWM/QQQ — the names your equity book trades —
so the two sleeves never fight over a shared position. The 16 remaining ETFs span
intl/EM equity, rates, commodities, FX, and real estate: pure diversifiers.

Strategy = the validated trend_follow_v2 logic at the latest bar:
  signal = avg sign of 3/6/12-month momentum; size = sign * min(TARGET_VOL/realized_vol,
  MAX_LEV) spread across active names, scaled to CAPITAL_BUDGET and the calibrated
  portfolio-level volatility scalar, long/short, monthly.

SAFETY: paper sizing only — per-name notional cap + gross cap; disables itself on any
connect/fetch error; every public method is wrapped so it can never raise into ibkr2.
For production you'd switch MKT->LMT and add fill reconciliation; this is a paper test.
"""

import os
import json
import time
import threading
import datetime

import numpy as np
import pandas as pd

from ibapi.client import EClient, Contract, Order
from ibapi.wrapper import EWrapper

from execution_journal import ExecutionCostJournal
from trading_costs import CostModelConfig, estimate_execution_cost

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, 'trend_sleeve_state.json')

TREND_CLIENT_ID = 90                      # isolated from the equity client
EQUITY_OVERLAP = {'SPY', 'QQQ', 'IWM'}    # traded by the equity book — exclude to stay disjoint
UNIVERSE = {
    'EFA': 'Equity-Intl', 'EEM': 'Equity-EM', 'EWJ': 'Equity-Japan',
    'SHY': 'Rates-Short', 'IEF': 'Rates-Mid', 'TLT': 'Rates-Long',
    'GLD': 'Commod-Gold', 'SLV': 'Commod-Silver', 'DBC': 'Commod-Broad',
    'USO': 'Commod-Oil', 'DBA': 'Commod-Ag',
    'UUP': 'FX-USD', 'FXE': 'FX-EUR', 'FXY': 'FX-JPY', 'FXB': 'FX-GBP',
    'VNQ': 'RealEstate',
}
SYMBOLS = [s for s in UNIVERSE if s not in EQUITY_OVERLAP]   # 16 disjoint names

HORIZONS = [63, 126, 252]                 # ~3 / 6 / 12 months
VOL_LOOKBACK = 60
TARGET_VOL_PER_ASSET = 0.10
MAX_LEVERAGE = 3.0
PORTFOLIO_SCALAR = 2.95                   # calibrated 16-ETF scalar for ~12% backtest vol
CAPITAL_BUDGET = 300_000                  # paper $ notional allocated to the trend sleeve (tune freely)
PER_NAME_NOTIONAL_CAP = 120_000           # hard per-ETF safety cap
GROSS_CAP = 600_000                       # hard total-gross safety cap
MIN_ORDER_SHARES = 1
REBALANCE_MIN_HOUR_ET = 10                # rebalance after 10:00 ET (skip the open auction)
FETCH_PACE_S = 3                          # seconds between historical requests (pacing)


def _env_nonnegative_float(name, default):
    try:
        value = float(os.environ.get(name, default))
        if not np.isfinite(value) or value < 0:
            raise ValueError
        return value
    except (TypeError, ValueError):
        print(f"[trend] invalid {name}; using {default}")
        return float(default)


TREND_COST_CONFIG = CostModelConfig(
    commission_per_share=_env_nonnegative_float(
        'COST_COMMISSION_PER_SHARE', 0.0035),
    min_commission=_env_nonnegative_float('COST_MIN_COMMISSION', 0.35),
    fallback_execution_bps=_env_nonnegative_float(
        'COST_FALLBACK_EXECUTION_BPS', 1.0),
    extra_slippage_bps=_env_nonnegative_float(
        'COST_EXTRA_SLIPPAGE_BPS', 0.0),
    impact_coefficient=_env_nonnegative_float(
        'COST_IMPACT_COEFFICIENT', 0.0),
)

BENIGN = {2104, 2106, 2107, 2108, 2158, 2100, 2150}
FATAL = {162, 165, 200, 300, 321, 354, 366, 420}


class _TrendClient(EWrapper, EClient):
    """Dedicated, isolated IBKR connection used only by the trend sleeve."""

    def __init__(self, execution_journal=None):
        EClient.__init__(self, self)
        self.execution_journal = execution_journal
        self.order_id = None
        self.connected_evt = threading.Event()
        self._bars = {}
        self._done = {}
        self._positions = {}
        self._pos_evt = threading.Event()
        self._nid = 1
        self._quote_lock = threading.RLock()
        self._quotes = {}
        self._quote_events = {}

    def nextValidId(self, oid):
        self.order_id = oid
        self.connected_evt.set()

    def error(self, reqId, code, msg, misc=None):
        if code in BENIGN:
            return
        print(f"[trend] err req {reqId} {code}: {msg}")
        if reqId in self._done and code in FATAL:
            self._done[reqId].set()
        if reqId in self._quote_events and code in FATAL:
            self._quote_events[reqId].set()

    # ---- decision-time quote snapshots (observe-only cost analytics) ----
    def tickPrice(self, reqId, tickType, price, attrib):
        if price is None or price <= 0:
            return
        with self._quote_lock:
            quote = self._quotes.get(reqId)
            if quote is None:
                return
            if tickType in {1, 66}:       # bid / delayed bid
                quote['bid'] = float(price)
            elif tickType in {2, 67}:     # ask / delayed ask
                quote['ask'] = float(price)
            elif tickType in {4, 68}:     # last / delayed last
                quote['last'] = float(price)
            if quote.get('bid') and quote.get('ask'):
                self._quote_events[reqId].set()

    def tickSnapshotEnd(self, reqId):
        with self._quote_lock:
            event = self._quote_events.get(reqId)
            if event is not None:
                event.set()

    def fetch_quote(self, symbol, timeout=5):
        c = Contract()
        c.symbol = symbol
        c.secType = 'STK'
        c.exchange = 'SMART'
        c.currency = 'USD'
        with self._quote_lock:
            rid = self._nid
            self._nid += 1
            event = threading.Event()
            self._quote_events[rid] = event
            self._quotes[rid] = {'bid': None, 'ask': None, 'last': None}
        try:
            self.reqMktData(rid, c, '', True, False, [])
            event.wait(timeout=timeout)
        finally:
            try:
                self.cancelMktData(rid)
            except Exception:
                pass
        with self._quote_lock:
            quote = dict(self._quotes.pop(rid, {}))
            self._quote_events.pop(rid, None)
        bid, ask = quote.get('bid'), quote.get('ask')
        if bid and ask and ask >= bid:
            quote['mid'] = (bid + ask) / 2.0
        else:
            quote['mid'] = None
        return quote

    # ---- daily bars (ADJUSTED_LAST = total return) ----
    def historicalData(self, reqId, bar):
        ds = str(bar.date).strip()[:8]
        try:
            d = datetime.datetime.strptime(ds, '%Y%m%d').date()
        except ValueError:
            return
        self._bars.setdefault(reqId, {})[d] = float(bar.close)

    def historicalDataEnd(self, reqId, s, e):
        if reqId in self._done:
            self._done[reqId].set()

    def fetch_daily(self, symbol, duration='2 Y', timeout=30):
        c = Contract()
        c.symbol = symbol
        c.secType = 'STK'
        c.exchange = 'SMART'
        c.currency = 'USD'
        rid = self._nid
        self._nid += 1
        ev = threading.Event()
        self._done[rid] = ev
        self._bars[rid] = {}
        self.reqHistoricalData(rid, c, '', duration, '1 day', 'ADJUSTED_LAST', 1, 1, False, [])
        ev.wait(timeout=timeout)
        return pd.Series(self._bars.get(rid, {})).sort_index()

    # ---- positions (only trend-universe names; disjoint from equity book) ----
    def position(self, account, contract, pos, avgCost):
        if contract.secType == 'STK':
            self._positions[contract.symbol] = pos

    def positionEnd(self):
        self._pos_evt.set()

    def read_positions(self, timeout=10):
        self._positions = {}
        self._pos_evt.clear()
        self.reqPositions()
        self._pos_evt.wait(timeout=timeout)
        self.cancelPositions()
        return dict(self._positions)

    # ---- orders ----
    def market_order(self, symbol, action, qty, reference_price=None):
        c = Contract()
        c.symbol = symbol
        c.secType = 'STK'
        c.exchange = 'SMART'
        c.currency = 'USD'
        o = Order()
        o.orderType = 'MKT'
        o.action = action
        o.totalQuantity = int(qty)
        o.eTradeOnly = False
        o.firmQuoteOnly = False
        o.tif = 'DAY'
        self.order_id += 1
        order_id = self.order_id

        try:
            quote = self.fetch_quote(symbol)
        except Exception as exc:
            print(f"[trend] quote unavailable for {symbol} ({exc}); order unchanged")
            quote = {'bid': None, 'ask': None, 'mid': None}
        reference = quote.get('mid') or reference_price
        if self.execution_journal is not None and reference is not None:
            try:
                predicted = estimate_execution_cost(
                    qty,
                    reference,
                    config=TREND_COST_CONFIG,
                    bid=quote.get('bid'),
                    ask=quote.get('ask'),
                )
                self.execution_journal.record_order_intent(
                    client_id=TREND_CLIENT_ID,
                    order_id=order_id,
                    strategy='time_series_momentum_v2',
                    sleeve='trend',
                    symbol=symbol,
                    side=action,
                    qty=qty,
                    decision_timestamp=datetime.datetime.now(
                        datetime.timezone.utc),
                    bid=quote.get('bid'),
                    ask=quote.get('ask'),
                    mid=quote.get('mid'),
                    order_type='MKT',
                    predicted_commission=predicted.commission,
                    predicted_spread_cost=predicted.spread,
                    predicted_slippage_cost=predicted.slippage,
                    predicted_impact_cost=predicted.impact,
                    predicted_total_cost=predicted.total,
                )
                print(
                    f"[trend] COST OBSERVE {symbol}: predicted one-way "
                    f"${predicted.total:.2f} ({predicted.total_bps:.2f}bp); "
                    "order unchanged"
                )
            except Exception as exc:
                print(
                    f"[trend] COST OBSERVE {symbol} unavailable ({exc}); "
                    "order unchanged"
                )
        self.placeOrder(order_id, c, o)
        return order_id

    def execDetails(self, reqId, contract, execution):
        if self.execution_journal is not None:
            self.execution_journal.record_execution(
                exec_id=str(execution.execId),
                client_id=int(getattr(execution, 'clientId', TREND_CLIENT_ID)),
                order_id=int(execution.orderId),
                perm_id=getattr(execution, 'permId', None),
                symbol=getattr(contract, 'symbol', ''),
                side=execution.side,
                shares=execution.shares,
                fill_price=execution.price,
                fill_time=(
                    getattr(execution, 'time', None)
                    or datetime.datetime.now(datetime.timezone.utc)
                ),
            )

    def commissionReport(self, commissionReport):
        if self.execution_journal is not None:
            self.execution_journal.record_commission(
                str(commissionReport.execId), commissionReport.commission
            )


class TrendSleeve:
    def __init__(self, host='127.0.0.1', port=7497, execution_journal=None):
        self.enabled = False
        self._running = False
        self.client = None
        self.execution_journal = execution_journal
        if self.execution_journal is None:
            self.execution_journal = ExecutionCostJournal(
                os.path.join(SCRIPT_DIR, 'trading_system.db')
            )
        self.state = self._load_state()
        try:
            self.client = _TrendClient(self.execution_journal)
            self.client.connect(host, port, TREND_CLIENT_ID)
            threading.Thread(target=self.client.run, daemon=True).start()
            if self.client.connected_evt.wait(timeout=8):
                self.enabled = True
                print(f"[trend] sleeve connected (clientId {TREND_CLIENT_ID}, port {port}) — "
                      f"{len(SYMBOLS)} disjoint ETFs, monthly rebalance")
            else:
                print("[trend] could NOT connect — sleeve DISABLED (equity system unaffected)")
        except Exception as e:
            print(f"[trend] init failed ({e}) — sleeve DISABLED")

    # ---- state ----
    def _load_state(self):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {'last_rebalance_month': None}

    def _save_state(self):
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[trend] state save failed: {e}")

    def _is_due(self, now):
        return self.state.get('last_rebalance_month') != [now.year, now.month]

    # ---- signal -> target shares ----
    def _target_shares(self, prices):
        rets = prices.pct_change(fill_method=None)
        raw = {}
        for c in prices.columns:
            p = prices[c].dropna()
            if len(p) < max(HORIZONS) + 5:
                continue
            sigs = [(1.0 if p.iloc[-1] / p.iloc[-h] - 1.0 > 0 else -1.0) for h in HORIZONS]
            sign = float(np.mean(sigs))
            vol = rets[c].dropna().tail(VOL_LOOKBACK).std() * np.sqrt(252)
            if not vol or vol <= 0:
                continue
            raw[c] = sign * min(TARGET_VOL_PER_ASSET / vol, MAX_LEVERAGE)
        n = len([v for v in raw.values() if v != 0]) or 1
        targets, gross = {}, 0.0
        for c, w in raw.items():
            unbounded_notional = CAPITAL_BUDGET * PORTFOLIO_SCALAR * w / n
            notional = max(
                -PER_NAME_NOTIONAL_CAP,
                min(PER_NAME_NOTIONAL_CAP, unbounded_notional),
            )
            px = prices[c].dropna().iloc[-1]
            sh = int(round(notional / px)) if px > 0 else 0
            targets[c] = sh
            gross += abs(sh * px)
        if gross > GROSS_CAP and gross > 0:                 # global gross safety scale-down
            k = GROSS_CAP / gross
            # Truncate toward zero so share rounding cannot leak above the hard cap.
            targets = {c: int(sh * k) for c, sh in targets.items()}
        return targets

    # ---- scheduling (called every loop from ibkr2; non-blocking) ----
    def maybe_rebalance(self, now):
        if not self.enabled or self._running:
            return
        try:
            if now.hour < REBALANCE_MIN_HOUR_ET or not self._is_due(now):
                return
            self._running = True
            threading.Thread(target=self._do_rebalance, args=(now,), daemon=True).start()
        except Exception as e:
            print(f"[trend] schedule error (isolated): {e}")
            self._running = False

    def _do_rebalance(self, now):
        try:
            print(f"[trend] monthly rebalance due {now.date()} — fetching {len(SYMBOLS)} daily series...")
            data = {}
            for s in SYMBOLS:
                ser = self.client.fetch_daily(s)
                if len(ser):
                    data[s] = ser
                time.sleep(FETCH_PACE_S)
            if len(data) < 8:
                print(f"[trend] only {len(data)} series fetched — abort cycle, retry next day")
                return
            prices = pd.DataFrame(data).sort_index()
            targets = self._target_shares(prices)
            current = self.client.read_positions()
            print(f"[trend] targets (signed shares): { {k: v for k, v in targets.items() if v} }")
            placed = 0
            for s, tgt in targets.items():
                cur = int(current.get(s, 0))
                diff = tgt - cur
                if abs(diff) < MIN_ORDER_SHARES:
                    continue
                action = 'BUY' if diff > 0 else 'SELL'
                reference_price = prices[s].dropna().iloc[-1]
                self.client.market_order(
                    s,
                    action,
                    abs(diff),
                    reference_price=reference_price,
                )
                placed += 1
                print(f"[trend]   {action} {abs(diff):>5} {s:4s}  (cur {cur} -> tgt {tgt})")
                time.sleep(0.5)
            self.state['last_rebalance_month'] = [now.year, now.month]
            self.state['last_rebalance_date'] = str(now.date())
            self.state['last_targets'] = targets
            self._save_state()
            print(f"[trend] rebalance complete — {placed} orders placed.")
        except Exception as e:
            print(f"[trend] rebalance error (sleeve isolated, equity unaffected): {e}")
        finally:
            self._running = False
