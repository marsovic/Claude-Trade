"""Glue between the engine and a broker.

* ``SimExecution``  - fills on the next daily bar (paper trading, backtests).
* ``LiveExecution`` - real broker. Orders decided after the close are kept as
  "planned" and sent when the US session opens; fills are reconciled by
  polling the broker; protective stops rest at the broker when it allows it,
  otherwise the bot watches prices and sells itself.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

from invest.brokers.base import BrokerError, Fill, LiveBroker, OrderRequest
from invest.brokers.sim import SimBroker
from invest.data import ET

log = logging.getLogger("invest")


class SimExecution:
    def __init__(self, s, state, fees):
        self.s = s
        self.state = state
        self.broker = SimBroker(s, state, fees)
        self.fractional = s.fractional

    def resolve(self, symbol: str) -> str | None:
        return symbol

    def queue(self, engine, order: OrderRequest) -> bool:
        self.broker.queue(order)
        return True

    def settle(self, engine, day, data) -> None:
        fills, expired = self.broker.settle_open(day, data)
        for f in fills:
            engine.apply_fill(f)
        for o in expired:
            engine.on_order_dead(o, "limit not reached at the open" if o.kind == "LIMIT"
                                 else "no bar for the symbol")
        for f in self.broker.settle_stops(day, data, engine.state.positions):
            engine.apply_fill(f)

    def on_stop_change(self, engine, pos) -> None:
        pass  # the simulator reads pos.stop_loss directly

    def on_position_closed(self, engine, pos, fill) -> None:
        self.broker.cancel_symbol(pos.symbol)

    def liquidate(self, engine, pos, day) -> None:
        slip = self.s.slippage_bps / 1e4
        price = pos.last_price * (1 - slip)
        fx = engine.fx_now
        fees = self.broker._fee_eur(pos.qty, price, pos.currency, fx)
        engine.apply_fill(Fill(symbol=pos.symbol, side="SELL", qty=pos.qty, price=price,
                               fees_eur=fees, usd_per_eur=fx, day=str(day.date()),
                               ts=int(pd.Timestamp(day).timestamp() * 1000), reason="death"))

    def broker_equity(self) -> float | None:
        return None

    def broker_cash(self) -> float | None:
        return None


class LiveExecution:
    UNKNOWN_LIMIT = 5  # syncs before an order the broker does not know is dropped

    def __init__(self, s, state, store, broker: LiveBroker, clock=time.time):
        self.s = s
        self.state = state
        self.store = store
        self.broker = broker
        self.clock = clock
        self.fractional = broker.fractional and s.fractional
        self._tickers: dict[str, str | None] = {}
        self._unknown: dict[str, int] = {}
        self._account = None

    # ------------------------------------------------------------ helpers
    def _log(self, kind, message, symbol=None):
        self.store.log_decision(kind, message, symbol, "LONG" if symbol else None)
        log.info("%s %s %s", kind, symbol or "", message)

    def resolve(self, symbol: str) -> str | None:
        if symbol not in self._tickers:
            try:
                self._tickers[symbol] = self.broker.resolve(symbol)
            except BrokerError as exc:
                self._log("error", f"cannot resolve instrument: {exc}", symbol)
                return None
        return self._tickers[symbol]

    def refresh_account(self):
        try:
            self._account = self.broker.account()
        except BrokerError as exc:
            self._log("error", f"account unavailable: {exc}")
        return self._account

    def broker_equity(self) -> float | None:
        return self._account.equity_eur if self._account else None

    def broker_cash(self) -> float | None:
        return self._account.cash_eur if self._account else None

    # ------------------------------------------------------------- orders
    def queue(self, engine, order: OrderRequest) -> bool:
        if not order.broker_ticker:
            order.broker_ticker = self.resolve(order.symbol) or ""
        if not order.broker_ticker:
            self._log("skip", "not tradable at this broker", order.symbol)
            return False
        order.status = "planned"
        self.state.pending.append(order.to_dict())
        return True

    def _track(self, order: OrderRequest) -> None:
        self.state.open_orders[order.id] = order.to_dict()

    def submit_planned(self, engine) -> int:
        """Send the orders planned after the last close (call at the session open)."""
        sent = 0
        orders = [OrderRequest.from_dict(o) for o in self.state.pending]
        orders.sort(key=lambda o: o.side != "SELL")
        self.state.pending = []
        for o in orders:
            if o.side == "SELL":
                pos = self.state.positions.get(o.symbol)
                if pos is None:
                    continue
                self._cancel_stop(pos)
                o.qty = pos.qty
            try:
                o.broker_id = self.broker.submit(o)
            except BrokerError as exc:
                self._log("error", f"{o.side} {o.qty:g} failed: {exc}", o.symbol)
                if o.side == "SELL" and o.symbol in self.state.positions:
                    self.on_stop_change(engine, self.state.positions[o.symbol])
                continue
            o.status = "submitted"
            self._track(o)
            sent += 1
            self._log("order", f"sent {o.side} {o.kind} {o.qty:g}"
                      + (f" limit {o.limit_price:.4g}" if o.limit_price else "")
                      + f" ({o.reason}) id {o.broker_id}", o.symbol)
        return sent

    def _cancel_stop(self, pos) -> None:
        if not pos.stop_order_id:
            return
        od = self.state.open_orders.pop(pos.stop_order_id, None)
        if od and od.get("broker_id"):
            try:
                self.broker.cancel(od["broker_id"])
            except BrokerError as exc:
                self._log("error", f"could not cancel the stop: {exc}", pos.symbol)
        pos.stop_order_id = ""

    def on_stop_change(self, engine, pos) -> None:
        if not self.broker.resting_stops:
            return  # watched by check_monitored_stops
        self._cancel_stop(pos)
        qty = pos.qty if self.fractional else float(int(pos.qty))
        if qty <= 0:
            return
        order = OrderRequest(symbol=pos.symbol, side="SELL", qty=qty, kind="STOP",
                             stop_price=pos.stop_loss, reason="stop", sleeve=pos.sleeve,
                             currency=pos.currency, broker_ticker=pos.broker_ticker)
        try:
            order.broker_id = self.broker.submit(order)
        except BrokerError as exc:
            self._log("error", f"stop not placed ({exc}): the bot will watch it itself",
                      pos.symbol)
            return
        order.status = "submitted"
        self._track(order)
        pos.stop_order_id = order.id

    def on_position_closed(self, engine, pos, fill) -> None:
        if pos.stop_order_id and pos.stop_order_id != (fill.order or {}).get("id"):
            self._cancel_stop(pos)

    def liquidate(self, engine, pos, day) -> None:
        self._cancel_stop(pos)
        order = OrderRequest(symbol=pos.symbol, side="SELL", qty=pos.qty, kind="MARKET",
                             reason="death", sleeve=pos.sleeve, currency=pos.currency,
                             broker_ticker=pos.broker_ticker)
        try:
            order.broker_id = self.broker.submit(order)
            self._log("order", f"liquidation SELL {pos.qty:g} sent", pos.symbol)
        except BrokerError as exc:
            self._log("error", f"liquidation failed, SELL IT BY HAND: {exc}", pos.symbol)

    # ---------------------------------------------------------- reconcile
    def sync(self, engine) -> int:
        if not self.state.open_orders:
            return 0
        try:
            self.broker.begin_sync()
        except BrokerError as exc:
            self._log("error", f"sync failed: {exc}")
            return 0
        applied = 0
        for oid, od in list(self.state.open_orders.items()):
            o = OrderRequest.from_dict(od)
            try:
                upd = self.broker.order_update(o)
            except BrokerError as exc:
                self._log("error", f"order status unavailable: {exc}", o.symbol)
                continue
            filled = upd.status == "filled" or (
                upd.status in ("cancelled", "rejected") and upd.filled_qty > 0)
            if filled:
                del self.state.open_orders[oid]
                engine.apply_fill(self._fill(engine, o, upd))
                applied += 1
            elif upd.status in ("cancelled", "rejected"):
                del self.state.open_orders[oid]
                if o.kind == "STOP":
                    pos = self.state.positions.get(o.symbol)
                    if pos and pos.stop_order_id == o.id:
                        pos.stop_order_id = ""
                        self._log("warning", f"stop {upd.status} by the broker: watched by "
                                  "the bot until it is placed again", o.symbol)
                else:
                    engine.on_order_dead(o, upd.status)
            elif upd.status == "unknown":
                self._unknown[oid] = self._unknown.get(oid, 0) + 1
                if self._unknown[oid] >= self.UNKNOWN_LIMIT:
                    del self.state.open_orders[oid]
                    self._log("warning", f"order {o.broker_id} unknown to the broker, "
                              "dropped", o.symbol)
        return applied

    def _fill(self, engine, o: OrderRequest, upd) -> Fill:
        price = upd.avg_price or o.limit_price or o.stop_price or 0.0
        qty = upd.filled_qty or o.qty
        fx = engine.fx_now
        fees = upd.fees_eur
        if not fees:
            foreign = o.currency != "EUR"
            cost = self.s.fees.cost(qty, price, foreign)
            fees = cost / fx if foreign else cost
        ts = upd.filled_at or int(self.clock() * 1000)
        day = datetime.fromtimestamp(ts / 1000, tz=ET).date()
        reason = o.reason
        if o.kind == "STOP":
            pos = self.state.positions.get(o.symbol)
            reason = "trailing stop" if pos is not None and pos.stop_loss >= pos.entry_price \
                else "stop"
        return Fill(symbol=o.symbol, side=o.side, qty=qty, price=price, fees_eur=fees,
                    usd_per_eur=fx, day=str(day), ts=ts, reason=reason, order=o.to_dict(),
                    fill_id=o.broker_id)

    def settle(self, engine, day, data) -> None:
        """End of day: reconcile fills, drop unfilled entries, detect manual closes."""
        self.sync(engine)
        for od in list(self.state.open_orders.values()):
            if od["side"] == "BUY" and od["kind"] in ("LIMIT", "MARKET"):
                try:
                    self.broker.cancel(od.get("broker_id", ""))
                    self._log("skip", "entry not filled today: cancelled", od["symbol"])
                except BrokerError as exc:
                    self._log("error", f"could not cancel stale entry: {exc}", od["symbol"])
        self.sync(engine)
        account = self.refresh_account()
        if account is None:
            return
        held = {p.ticker for p in account.positions.values() if p.qty > 0}
        selling = {od["symbol"] for od in self.state.open_orders.values()
                   if od["side"] == "SELL" and od["kind"] != "STOP"}
        for sym, pos in list(self.state.positions.items()):
            if pos.broker_ticker and pos.broker_ticker not in held and sym not in selling:
                self._log("warning", "position no longer at the broker (closed by hand?)",
                          sym)
                self._cancel_stop(pos)
                engine.apply_fill(Fill(symbol=sym, side="SELL", qty=pos.qty,
                                       price=pos.last_price, fees_eur=0.0,
                                       usd_per_eur=engine.fx_now, day=str(day.date()),
                                       ts=int(self.clock() * 1000),
                                       reason="closed outside the bot"))

    def check_monitored_stops(self, engine, prices: dict[str, float]) -> None:
        """Brokers without resting stops: sell when the last price breaks the stop."""
        selling = {od["symbol"] for od in self.state.open_orders.values() if od["side"] == "SELL"}
        for sym, pos in list(self.state.positions.items()):
            if self.broker.resting_stops and pos.stop_order_id:
                continue
            price = prices.get(sym)
            if price is None or price > pos.stop_loss or sym in selling:
                continue
            order = OrderRequest(symbol=sym, side="SELL", qty=pos.qty, kind="MARKET",
                                 reason="stop (watched)", sleeve=pos.sleeve,
                                 currency=pos.currency, broker_ticker=pos.broker_ticker)
            try:
                order.broker_id = self.broker.submit(order)
            except BrokerError as exc:
                self._log("error", f"stop sale failed: {exc}", sym)
                continue
            order.status = "submitted"
            self._track(order)
            self._log("order", f"price {price:.4g} broke the stop {pos.stop_loss:.4g}: "
                      "SELL at market", sym)

    def broker_prices(self) -> dict[str, float]:
        """Last prices of held positions, from the broker (map back to symbols)."""
        if self._account is None:
            return {}
        by_ticker = {p.ticker: p.price for p in self._account.positions.values()}
        return {sym: by_ticker[pos.broker_ticker]
                for sym, pos in self.state.positions.items()
                if pos.broker_ticker in by_ticker and by_ticker[pos.broker_ticker] > 0}
